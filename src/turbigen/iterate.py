"""Closing the loop between a design and the CFD it predicts.

A mean-line design is not self-consistent with its own solution: the flow
leaves a blade at a different angle from the metal, arrives at a different
incidence, and loses more than the design assumed. An :class:`Iterator` names
one such mismatch, measures it, and says which design variable to move.

The split is between physics and numerics, and it is the whole point of the
module. An iterator is nearly declarative --- which knobs it owns, how to
measure its error, and constants --- while every piece of arithmetic lives once
in :func:`step`, over a flat table assembled from all of them::

    name                  u        e       gain  clip  tol
    dchi_TE[0]          -1.42    +0.31     1.0   2.0   1.0
    dchi_LE[1]          +0.85    -0.12    -1.0   2.0   1.0
    mean_line.Ys[0]      0.05    -0.004    0.5    -    0.005

After assembly the iterators disappear, so a better step rule --- a secant, or
one warm-started from a fit over previous runs --- is a change to :func:`step`
alone, and touches no iterator. That table is also what such a fit would
consume, which is why every run records its errors whether or not anything is
iterating.

The package this replaces mutates a live config from inside `update()`, which a
frozen :class:`~turbigen.config.Config` cannot allow and which is the same
confusion between a design and its result that the rebuild exists to remove.
Here an iterator returns a new config and owns no state at all.
"""

import dataclasses
import logging
from typing import ClassVar

import numpy as np
from numpy.polynomial import legendre

import ember.average
import ember.cut
from turbigen.node import Node
from turbigen.result import Result

logger = logging.getLogger("turbigen.iterate")
"""Iteration-level messages: the table, the verdict, what the stepper noticed.

Named apart from what one run says so that `iterate` can quieten a hundred runs
on the console without losing the few lines that describe the iteration itself.
"""

N_SPAN_CUT = 101
"""Meridional points defining the span curve a blade surface is cut along."""

TINY = 1e-9
"""Below this a nominal value is treated as zero for a relative tolerance."""

DU_MIN = 0.25
"""Smallest move, in tolerance-equivalent steps, that may update the Jacobian.

The slope a secant infers has error of order `noise / du`, and the errors here
are measured from a march that is only partly converged --- the same deviation
slope read +1.27 from a 200-step solve and about +0.3 from a 50-step one. A
quarter of a step is where an update stops saying more than the noise does.
"""

COND_MAX = 1e6
"""Above this condition number the Jacobian is not trusted to be inverted."""

FLAT = 0.1
"""A diagonal below this fraction of its prior counts as a flat response."""


class Iterator(Node):
    """Base for design iterators.

    A member declares the knobs it owns, measures the error those knobs should
    null, and leaves every decision about how far to move to :func:`step`.
    """

    from_solution: ClassVar[bool] = True
    """Whether this iterator's error is measured from the CFD solution.

    False for one measured from the design alone. Such an iterator converges in
    pure numpy, so it is run to convergence by :func:`resolve` *inside* every
    pass rather than across them --- which is what keeps its knob consistent
    with a blade the solution iterators have since recambered.

    Declared rather than inferred, for the same reason :meth:`paths` is: what
    an iterator measures its error from is knowledge only its author has, and
    guessing it wrong fails silently in both directions.
    """

    gain: float = 1.0
    """How much of the error to subtract from the unknown.

    Carries the sign of the local sensitivity as well as its size: the step is
    always ``u -= gain * e``, so an iterator whose error *falls* as its knob
    rises declares a negative gain. Reciprocal of an assumed slope, so it is
    the crudest possible Newton step.
    """

    clip: float = 0.0
    """Largest change in one iteration, in the units of the unknown.

    Zero for no limit. A clip is what keeps a bad early step --- taken on a
    field that has not settled --- from throwing the design somewhere it cannot
    be meshed.
    """

    tolerance: float = 1.0
    """Error below which this iterator is converged, in the units of the error."""

    #
    # TO BE IMPLEMENTED BY AN ITERATOR
    #

    def unknowns(self, config):
        """Return the design variables this iterator owns, by name.

        Names are global, so two iterators claiming one name is an error rather
        than a race.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement unknowns(self, config)"
        )

    def with_unknowns(self, config, values):
        """Return `config` with this iterator's unknowns set to `values`.

        Touches only its own fields, so applying two iterators in either order
        gives the same config.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement with_unknowns(self, config, values)"
        )

    def paths(self, config):
        """Return the config leaves this iterator moves, as `node.flatten`
        spells them.

        The same knobs as :meth:`unknowns`, named the other way round. A knob
        is often a reduction over several leaves --- ``dchi_TE[0]`` is the mean
        recamber of a row, spread over ``blades[0].sections[*].dchi_TE`` ---
        so the two namings cannot be derived from one another, and what reads
        an archive of designs needs the leaf spelling to tell a design variable
        from an iterated one.

        Declared rather than inferred: a knob whose leaves went unnamed would
        be taken for a design variable, and a predictor would then use the
        recamber it is trying to predict as an input. `test_iterate.py` asserts
        that what this returns is what `with_unknowns` actually writes.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement paths(self, config)"
        )

    def error(self, config, result):
        """Return what each unknown should null, under the same names.

        Measured while the grid is alive, because some of these are properties
        of the three-dimensional field and exist nowhere else. Return an empty
        dict when the run gives nothing to measure: a march that diverged is
        not a reason to fail, only a reason not to step.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement error(self, config, result)"
        )

    #
    # PROVIDED
    #

    def tolerances(self, config):
        """Return the tolerance on each unknown, by name."""
        return {name: self.tolerance for name in self.unknowns(config)}

    def clips(self, config):
        """Return the largest step for each unknown, by name."""
        return {name: self.clip for name in self.unknowns(config)}


#
# THE STEPPER
#
# Everything below is generic. It knows names, numbers and tolerances, and
# nothing about angles, blades or mean lines.
#


def selected(config, from_solution):
    """Return `config` carrying only the iterators of one speed.

    The stepper takes a config and reads `config.iterate` off it, so selecting
    a subset is done by handing it a config that holds only those --- which
    keeps every function below generic over *which* iterators it is stepping,
    without a second argument threaded through all of them.
    """
    return dataclasses.replace(
        config,
        iterate=tuple(
            iterator
            for iterator in config.iterate
            if iterator.from_solution == from_solution
        ),
    )


def unknowns(config):
    """Return every configured iterator's unknowns, merged.

    Raises if two iterators claim the same name, which would otherwise make the
    result depend on the order they happen to appear in.
    """
    merged = {}
    for iterator in config.iterate:
        for name, value in iterator.unknowns(config).items():
            if name in merged:
                raise ValueError(
                    f"Two iterators both claim the design variable {name!r}; "
                    "each one must own the variables it moves."
                )
            merged[name] = value
    return merged


def errors(config, result):
    """Return every configured iterator's error, merged."""
    merged = {}
    for iterator in config.iterate:
        merged.update(iterator.error(config, result))
    return merged


def converged(config, result):
    """Return whether every measured error is within its tolerance.

    An unknown whose error was not measured counts as unconverged: the run had
    nothing to say about it, which is not the same as it being right.
    """
    measured = measured_errors(config, result)

    for iterator in config.iterate:
        for name, tolerance in iterator.tolerances(config).items():
            if name not in measured or not np.abs(measured[name]) <= tolerance:
                return False

    return True


def measured_errors(config, result):
    """Return the errors `result` reports, preferring the ones it recorded.

    A run stores what its iterators measured, so re-measuring would repeat work
    that is not free --- :meth:`Incidence.error` cuts the grid --- and would
    fail outright for a result whose grid has since been released.
    """
    if result.error:
        return dict(result.error)
    return errors(config, result)


def properties(config):
    """Return the gain, clip and tolerance of every unknown, by name."""
    gain, clip, tolerance = {}, {}, {}

    for iterator in config.iterate:
        clips = iterator.clips(config)
        tolerances = iterator.tolerances(config)
        for name in iterator.unknowns(config):
            gain[name] = iterator.gain
            clip[name] = clips[name]
            tolerance[name] = tolerances[name]

    return gain, clip, tolerance


def step(config, result, history=()):
    """Return the config to try next, from the errors `result` reports.

    A Newton step on an approximate Jacobian: `B dx = -e`, clipped per key. `B`
    starts as the diagonal the declared gains already assert --- ``u -= gain *
    e`` is a Newton step under exactly that assumption, sign included --- and
    is improved by a rank-one Broyden update for each move the run has already
    paid for. **With no history the step is arithmetically identical to
    ``u -= gain * e``**, so a first iteration is never worse than it was.

    What that buys is the off-diagonal terms. The exit angle of a row sets the
    inlet angle of the next, so correcting one row's deviation moves the next
    row's incidence by a comparable amount; a diagonal step cannot see that and
    propagates a correction one row per iteration.

    Parameters
    ----------
    config : Config
        The design that was run.
    result : Result
        What running it achieved.
    history : sequence
        Earlier ``(unknowns, errors)`` pairs from this run, oldest first.
        Numbers only: a `Result` holds a live grid, and keeping one per
        iteration would pin gigabytes to read a few dozen floats.

    """
    measured = measured_errors(config, result)
    values = unknowns(config)
    gain, clip, tolerance = properties(config)

    # A knob with nothing measured is held, as is one with no gain, which is
    # how an iterator says it does not want to move.
    names = [
        name
        for name in values
        if name in measured and gain[name] and tolerance[name] > TINY
    ]
    if not names:
        logger.debug("Nothing measured to correct towards, so nothing moves.")
        return config

    # Worked in units of each knob's own tolerance, which is the only scale
    # declared for it. Degrees of recamber and a loss coefficient otherwise
    # share one Euclidean norm in the Broyden update, and the update -- being
    # least-change in that norm -- would spend itself entirely on whichever
    # variable happened to carry the larger numbers.
    u_scale = np.array([abs(gain[name]) * tolerance[name] for name in names])
    e_scale = np.array([tolerance[name] for name in names])
    prior = np.array([np.sign(gain[name]) for name in names])

    jacobian = _jacobian(names, prior, history, (values, measured), u_scale, e_scale)
    _report_flat(names, jacobian)

    change = _newton(jacobian, np.array([measured[n] for n in names]) / e_scale, prior)

    # The clip is the trust bound, and the reason a flat response degrades to
    # the old behaviour rather than to a wild excursion.
    limit = np.array(
        [
            clip[name] / scale if clip[name] else np.inf
            for name, scale in zip(names, u_scale)
        ]
    )
    change = np.clip(change, -limit, limit)

    moved = {
        name: values[name] + change[i] * u_scale[i] for i, name in enumerate(names)
    }

    for iterator in config.iterate:
        mine = {
            name: moved[name] for name in iterator.unknowns(config) if name in moved
        }
        if mine:
            config = iterator.with_unknowns(config, mine)

    return config


def _jacobian(names, prior, history, current, u_scale, e_scale):
    """Return the scaled Jacobian, from the prior and every informative move.

    Rebuilt from the trajectory on every call rather than carried between
    calls, so that `step` keeps no state and can be reasoned about one call at
    a time. It costs a few matrix operations on a handful of numbers.
    """
    jacobian = np.diag(prior)

    trajectory = list(history) + [current]
    for (values, errs), (values_next, errs_next) in zip(trajectory, trajectory[1:]):
        if not all(
            name in mapping
            for mapping in (values, errs, values_next, errs_next)
            for name in names
        ):
            # A pass that measured different knobs cannot be differenced.
            continue

        du = np.array([values_next[n] - values[n] for n in names]) / u_scale
        de = np.array([errs_next[n] - errs[n] for n in names]) / e_scale

        length = float(du @ du)
        if np.sqrt(length) < DU_MIN:
            logger.debug("A move too small to learn from, so the Jacobian stands.")
            continue

        jacobian = jacobian + np.outer(de - jacobian @ du, du) / length

    return jacobian


def _newton(jacobian, error, prior):
    """Return the step solving `jacobian @ change = -error`, or the prior's."""
    try:
        if np.linalg.cond(jacobian) > COND_MAX:
            raise np.linalg.LinAlgError("the Jacobian is ill-conditioned")
        return np.linalg.solve(jacobian, -error)
    except np.linalg.LinAlgError as err:
        # Never propagate a NaN into a design: fall back to the step the gains
        # alone would have taken, which the clip then bounds as usual.
        logger.info(f"Stepping on the declared gains instead, because {err}.")
        return -error * prior


def _report_flat(names, jacobian):
    """Say when a knob has stopped moving its own error.

    Incidence against leading-edge recamber is known to go flat and then flip,
    and a design variable that no longer changes what it is meant to control is
    worth seeing in the log before it is worth acting on.
    """
    for i, name in enumerate(names):
        if abs(jacobian[i, i]) < FLAT:
            logger.info(
                f"The response of {name} has gone flat "
                f"(slope {jacobian[i, i]:.3g} of an expected 1)."
            )


def format_table(config, result):
    """Return a one-line-per-unknown summary of where the iteration stands."""
    measured = measured_errors(config, result)
    values = unknowns(config)

    tolerances = {}
    for iterator in config.iterate:
        tolerances.update(iterator.tolerances(config))

    width = max(len(name) for name in values) if values else 0
    lines = [f"{'name':<{width}}  {'value':>10}  {'error':>10}  {'tol':>9}  ok"]
    for name, value in values.items():
        error = measured.get(name, np.nan)
        tolerance = tolerances[name]
        ok = "y" if np.abs(error) <= tolerance else "n"
        lines.append(
            f"{name:<{width}}  {value:10.4g}  {error:10.4g}  {tolerance:9.3g}  {ok}"
        )

    return "\n".join(lines)


def resolve(config, max_iter=10):
    """Return `config` with its design-only knobs converged.

    The same stepper as :func:`converge`, over the iterators whose error comes
    from the design rather than the solution, against a "run" that designs and
    nothing else. Pure numpy and typically two passes, so it is cheap enough to
    repeat --- which is the point: it runs inside every solve rather than once
    before them, so a knob stays consistent with a design the solution
    iterators keep moving.

    Written as its own loop rather than as :func:`converge` with a designing
    `run`, which very nearly works. What does not carry over is the rest of
    what `converge` is: a divergence guard for a march there has not been, a
    per-iteration log of directories that do not exist, and a history kept as
    numbers because a `Result` pins a grid --- when here there is no grid at
    all. Twelve lines that share every piece of arithmetic beat a callable that
    has to pretend to solve.

    Parameters
    ----------
    config : Config
        Where to start. Not modified.
    max_iter : int
        Most passes before giving up.

    Returns
    -------
    config : Config
        The same config with its design-only knobs moved onto their targets,
        or `config` itself when there are none to move.

    """
    inner = selected(config, from_solution=False)
    if not inner.iterate:
        return config

    history = []
    for _ in range(max_iter):
        # No solver, no grid, no output directory: designing is the whole run.
        result = Result(machine=inner.design())

        if converged(inner, result):
            logger.debug(f"Resolved the design-only knobs: {unknowns(inner)}")
            # The full list goes back on, so what comes out of here is the
            # config that was handed in with some of its leaves moved, rather
            # than one that has quietly lost the iterators it will need next.
            return dataclasses.replace(inner, iterate=config.iterate)

        stepped = step(inner, result, history)
        history.append((unknowns(inner), measured_errors(inner, result)))
        inner = stepped

    raise ValueError(
        f"The design-only knobs {sorted(unknowns(inner))} did not converge in "
        f"{max_iter} passes. These are solved without CFD, so this is a "
        f"property of the design rather than of a march: check that the target "
        f"is reachable."
    )


def converge(config, run, max_iter=10):
    """Iterate `config` until every error is within tolerance.

    Parameters
    ----------
    config : Config
        Where to start. Not modified: each iteration returns a new one.
    run : callable
        Takes ``(config, i_iter)`` and returns the :class:`~turbigen.result.
        Result` of solving it. Injected rather than imported, so this module
        knows nothing of output directories, restarts or the CLI --- and a test
        can iterate a cheap analytic stand-in with no CFD at all.
    max_iter : int
        Most iterations to run before giving up.

    Returns
    -------
    config : Config
        The configuration that produced `result`.
    result : Result
        What the last iteration achieved.
    converged : bool

    """
    result = None

    # Numbers only. A Result holds the live grid it was measured from -- tens
    # of megabytes for the smallest case here and gigabytes for a real machine
    # -- and keeping one per iteration would stop any of them being freed to
    # read the handful of floats the step actually uses.
    history = []

    for i_iter in range(max_iter):
        result = run(config, i_iter)

        # A march that blew up measures nothing: its mixed-out mean line is
        # whatever the NaNs averaged to, and stepping on that would move the
        # design somewhere arbitrary and call it a correction. Stopping leaves
        # the evidence in place -- the iteration that diverged is still on
        # disk, with its report -- which is what someone needs to see.
        if result.history is not None and not result.converged:
            logger.warning(
                f"Iteration {i_iter} diverged, so there is nothing to correct "
                "towards. Stopping with the design that produced it."
            )
            return config, result, False

        # Only the solution iterators are stepped here. A design-only knob is
        # already on its target -- `resolve` put it there inside the run that
        # just finished -- so its error is ~0 while its *value* has moved,
        # which is a zero slope. Fed to the Broyden update that is a flat
        # response, and the least-change update spends itself explaining it
        # rather than the knobs that genuinely need moving. `run` still gets
        # the whole config, so the nested resolve still sees its own.
        stepping = selected(config, from_solution=True)

        # Tabled from the same view, for the same reason and one more: the
        # value this config holds for a design-only knob is the one it started
        # with, since the resolve that moved it happened on a copy inside the
        # run. Printed here it would sit beside an error measured after the
        # move, which is two different designs on one row.
        logger.info(f"Iteration {i_iter}:\n{format_table(stepping, result)}")

        if converged(stepping, result):
            logger.info(f"Converged after {i_iter + 1} iteration(s).")
            return config, result, True

        # The stepped config is what the next pass runs, so the one returned
        # alongside a result is always the one that produced it.
        stepped = dataclasses.replace(
            step(stepping, result, history), iterate=config.iterate
        )
        history.append((unknowns(stepping), measured_errors(stepping, result)))
        config = stepped

    logger.warning(f"Not converged after {max_iter} iteration(s).")
    return config, result, False


#
# THE ITERATORS
#


def _recamber_unknowns(config, field):
    """Return the mean recamber of each row, under `field`.

    One number per row rather than one per section: the spanwise distribution
    of recamber is a design decision, and an iterator correcting a mean-line
    mismatch has nothing to say about it.
    """
    return {
        f"{field}[{i_row}]": float(
            np.mean([getattr(section, field) for section in blade.sections])
        )
        for i_row, blade in enumerate(config.blades)
    }


def _with_recamber(config, field, values):
    """Return `config` with each row's mean recamber under `field` set.

    Applied as a uniform shift, so whatever spanwise distribution the design
    asked for survives being iterated.
    """
    blades = list(config.blades)

    for i_row, blade in enumerate(blades):
        name = f"{field}[{i_row}]"
        if name not in values:
            continue

        current = np.mean([getattr(section, field) for section in blade.sections])
        shift = values[name] - current
        blades[i_row] = dataclasses.replace(
            blade,
            sections=tuple(
                dataclasses.replace(section, **{field: getattr(section, field) + shift})
                for section in blade.sections
            ),
        )

    return dataclasses.replace(config, blades=tuple(blades))


def _recamber_paths(config, field):
    """Return every section's `field`, as `node.flatten` spells it.

    Every section, although :func:`_recamber_unknowns` reads one number per
    row: the knob is a row mean, so moving it writes to all of them.
    """
    return {
        f"blades[{i_row}].sections[{i_section}].{field}"
        for i_row, blade in enumerate(config.blades)
        for i_section in range(len(blade.sections))
    }


class Deviation(Iterator):
    """Match the exit flow angle to the design by moving the trailing edge.

    Flow leaves a blade less turned than the metal. Recambering the trailing
    edge by the shortfall is the classical fix, and one iteration of CFD
    measures the shortfall exactly rather than correlating it.
    """

    type: ClassVar[str] = "deviation"

    gain: float = 1.0
    clip: float = 2.0
    tolerance: float = 1.0
    """Permissible error on exit flow angle [deg]."""

    def unknowns(self, config):
        return _recamber_unknowns(config, "dchi_TE")

    def with_unknowns(self, config, values):
        return _with_recamber(config, "dchi_TE", values)

    def paths(self, config):
        return _recamber_paths(config, "dchi_TE")

    def error(self, config, result):
        if result.actual is None or result.machine is None:
            logger.debug("No mixed-out mean line, so no deviation to measure.")
            return {}

        nominal = result.machine.mean_line
        return {
            f"dchi_TE[{i_row}]": float(
                result.actual.row(i_row).Alpha_rel[1] - nominal.row(i_row).Alpha_rel[1]
            )
            for i_row in range(len(config.blades))
        }


class Incidence(Iterator):
    """Set the leading edge to meet the flow at a chosen incidence.

    Measured on the three-dimensional field at one span fraction rather than
    from the mixed-out mean line, because incidence is a local property of the
    leading edge and the whole point of moving it is that the mean line does
    not see what the tip and the hub are doing.
    """

    type: ClassVar[str] = "incidence"

    target: float = 0.0
    """Incidence to aim for [deg]."""

    spf: float = 0.5
    """Span fraction to measure the incidence at."""

    upstream: float = 0.05
    """Where to read the flow angle, as a fraction of the gap ahead of the row."""

    # Negative because the metal angle rises with the recamber while the
    # incidence is measured against it, so the error falls as the knob rises.
    #
    # Damped an order of magnitude below the measured slope of -0.99, because
    # incidence against leading-edge recamber is flat over a range and then
    # flips, and a Newton-sized first step is taken before anything has been
    # learned about which of those a design is sitting in. The secant recovers
    # the rate from the second iteration on, so the cost of being timid here is
    # one gentle step rather than a slow iteration.
    gain: float = -0.1
    clip: float = 2.0
    tolerance: float = 1.0
    """Permissible error on local incidence [deg]."""

    def unknowns(self, config):
        return _recamber_unknowns(config, "dchi_LE")

    def with_unknowns(self, config, values):
        return _with_recamber(config, "dchi_LE", values)

    def paths(self, config):
        return _recamber_paths(config, "dchi_LE")

    def error(self, config, result):
        if result.grid is None or result.machine is None:
            logger.debug("No solved grid, so no incidence to measure.")
            return {}

        measured = {}
        for i_row in range(len(config.blades)):
            incidence = _incidence(result, i_row, self.spf, self.upstream)
            if np.isfinite(incidence):
                measured[f"dchi_LE[{i_row}]"] = incidence - self.target
            else:
                logger.info(f"Could not measure the incidence of row {i_row}.")

        return measured


def _incidence(result, i_row, spf, upstream):
    """Return the incidence onto row `i_row` at span fraction `spf` [deg].

    The flow angle a little ahead of the leading edge, mass-averaged over the
    pitch, minus the metal angle there. Both are relative-frame yaw angles in
    the same convention, so the difference is the incidence as a designer means
    it.
    """
    machine = result.machine
    annulus = machine.annulus

    # A span cut of the whole machine, as the contour plot takes.
    m = np.linspace(0.0, annulus.mmax, annulus.n_segment * 50 + 1)
    xr_curve = annulus.evaluate_xr(m, spf).T
    cut = ember.cut.structured_meridional(result.grid, xr_curve)
    if not len(cut):
        return np.nan

    # Rows occupy the odd segments, so the leading edge is at an integer
    # station and the flow is read a fraction of the gap ahead of it.
    xr_read = annulus.evaluate_xr(2 * i_row + 1 - upstream, spf)

    block, i_read = _nearest_station(cut, xr_read)
    if block is None:
        return np.nan

    # Mass-averaged over the pitch: an area average would let the wake, which
    # carries little of the flow, pull the angle around.
    weight = np.asarray(block.rhoVm[i_read])
    alpha = np.asarray(block.Alpha_rel[i_read])
    flow = float(np.sum(weight * alpha) / np.sum(weight))

    metal = float(machine.rows[i_row].blade.chi(spf)[0])

    return flow - metal


def _nearest_station(cut, xr):
    """Return the block of `cut` holding `xr`, and the streamwise index of it."""
    best, best_block, best_index = np.inf, None, None

    for block in cut:
        # The pitchwise index is immaterial: a span cut holds one meridional
        # position per streamwise station, to within the mesh's own skew.
        distance = np.hypot(
            np.asarray(block.x)[:, 0] - xr[0], np.asarray(block.r)[:, 0] - xr[1]
        )
        index = int(np.argmin(distance))
        if distance[index] < best:
            best, best_block, best_index = distance[index], block, index

    return best_block, best_index


class MeanLine(Iterator):
    """Relax nominal design variables towards what the CFD achieved.

    Loss, blockage and the like are guesses when a mean line is drawn, and the
    solution measures them. Moving the design onto its own answer is what makes
    the mean line describe the machine that was built rather than the one that
    was assumed.
    """

    type: ClassVar[str] = "mean_line"

    variables: tuple[str, ...] = ()
    """Names of the design variables to relax, as the mean-line design spells
    them."""

    gain: float = 0.5
    tolerance: float = 0.01
    """Permissible error, as a fraction of the nominal value."""

    def unknowns(self, config):
        merged = {}
        for name in self.variables:
            values = self._values(config, name)
            merged.update(zip(self._names(name, len(values)), map(float, values)))
        return merged

    def with_unknowns(self, config, values):
        design = config.mean_line

        replacements = {}
        for name in self.variables:
            current = self._values(config, name)
            names = self._names(name, len(current))
            if not any(key in values for key in names):
                continue

            moved = [values.get(key, now) for key, now in zip(names, current)]
            # Restored to the shape it was declared in, so a scalar design
            # variable does not silently become a one-element list.
            replacements[name] = moved[0] if len(moved) == 1 else tuple(moved)

        if not replacements:
            return config

        return dataclasses.replace(
            config, mean_line=dataclasses.replace(design, **replacements)
        )

    def paths(self, config):
        # Alone among the iterators, this one's knobs *are* leaves of the
        # config, one apiece, so `_names` already spells them the way
        # `node.flatten` does and there is nothing to translate.
        return set(self.unknowns(config))

    def error(self, config, result):
        if result.actual is None:
            logger.debug("No mixed-out mean line, so no design variables to match.")
            return {}

        achieved = config.mean_line.backward(result.actual)

        merged = {}
        for name in self.variables:
            nominal = self._values(config, name)
            actual = np.atleast_1d(np.asarray(achieved[name], dtype=float))
            merged.update(
                zip(self._names(name, len(nominal)), map(float, nominal - actual))
            )
        return merged

    def tolerances(self, config):
        """Return absolute tolerances, scaled from the relative one declared.

        Design variables have no common scale --- a loss coefficient of 0.05
        sits beside a stage loading of 1.6 --- so one absolute number cannot
        serve them all. A nominal value of zero has nothing to be relative to,
        and falls back to taking the tolerance as absolute.
        """
        merged = {}
        for name in self.variables:
            nominal = self._values(config, name)
            for key, value in zip(self._names(name, len(nominal)), nominal):
                scale = np.abs(value) if np.abs(value) > TINY else 1.0
                merged[key] = self.tolerance * float(scale)
        return merged

    def _values(self, config, name):
        """Return the nominal value of design variable `name`, as an array."""
        if not hasattr(config.mean_line, name):
            raise ValueError(
                f"The mean-line design has no variable {name!r} to iterate; "
                f"it takes {sorted(f.name for f in dataclasses.fields(config.mean_line))}."
            )
        return np.atleast_1d(np.asarray(getattr(config.mean_line, name), dtype=float))

    @staticmethod
    def _names(name, count):
        """Return the table key of each element of design variable `name`."""
        if count == 1:
            return [f"mean_line.{name}"]
        return [f"mean_line.{name}[{i}]" for i in range(count)]


NAME_LOG_MU = "fluid.log_mu"
"""Table key of the viscosity knob. See :class:`SurfaceReynolds`."""


class SurfaceReynolds(Iterator):
    """Set the viscosity to reach a surface Reynolds number.

    A Reynolds number is what a cascade is actually specified at --- it is the
    number a designer carries between machines, where a viscosity in
    kg/m/s is not. But it cannot simply be inverted for `mu`: it is measured
    against a blade surface length and a mean-line reference state, so it needs
    a whole design, which needs a viscosity to exist first.

    That circularity is what makes this an iterator rather than a formula, and
    the reason it is *this* kind of iterator is that closing it costs no CFD.
    Everything it reads --- :meth:`turbigen.machine.Machine.Re_surf` --- comes
    off the design, so :func:`resolve` converges it in pure numpy inside every
    pass, and the solution iterators never see it.

    The package this replaces meant to do this arithmetically and never
    finished: `turbigen.config.set_mu_from_Re_surf` raises
    `NotImplementedError` on its first line and is called whenever a config
    names `Re_surf`, so every configuration that asks for one has been dead.
    There is accordingly nothing to stay bug-compatible with.
    """

    type: ClassVar[str] = "Re_surf"
    from_solution: ClassVar[bool] = False

    target: float
    """Surface Reynolds number to design for [--]."""

    i_row: int = 0
    """Index of the blade row whose Reynolds number meets the target.

    There is one viscosity and one Reynolds number per row, so only one row can
    be placed exactly and the rest follow from the design. The first row by
    default, which is what the abandoned implementation indexed.
    """

    gain: float = -1.0
    """Exactly the Newton step, rather than an approximation to one.

    At fixed geometry `Re_surf` is exactly proportional to `1/mu`, so in the
    logarithmic knob below the residual is linear with unit slope, and the
    stepper's `u -= gain * e` at `gain = -1` lands on the answer in one move.
    Negative because the Reynolds number *falls* as the viscosity rises, which
    is the sign convention :attr:`Iterator.gain` documents.
    """

    tolerance: float = 0.01
    """Converged inside this fractional error on the Reynolds number.

    In log units, so it reads directly as a relative error to within a
    percent of itself.
    """

    def unknowns(self, config):
        # The knob is log(mu), not mu. Viscosity is multiplicative -- the
        # residual is a ratio and spans orders of magnitude between fluids --
        # so a step of constant size in mu means nothing, and a *scalar* gain
        # cannot be the Newton step for a knob whose sensitivity scales with
        # its own value. In the log it can, exactly. The table shows the log
        # because that is what is being solved for; the config still holds mu,
        # which is what `paths` reports.
        return {NAME_LOG_MU: float(np.log(self._mu(config)))}

    def with_unknowns(self, config, values):
        if NAME_LOG_MU not in values:
            return config

        mu = float(np.exp(values[NAME_LOG_MU]))
        try:
            fluid = dataclasses.replace(config.fluid, mu=mu)
        except TypeError as err:
            raise ValueError(
                f"Cannot reach a surface Reynolds number by changing the "
                f"viscosity of a {type(config.fluid).__name__}, which has no "
                f"mu to change."
            ) from err

        return dataclasses.replace(config, fluid=fluid)

    def paths(self, config):
        # The leaf, not the knob. `node.flatten` spells the config's own field,
        # and it is mu that is written there even though log(mu) is what moves
        # -- which is exactly the mismatch this method exists to bridge.
        return {"fluid.mu"}

    def error(self, config, result):
        if result.machine is None:
            logger.debug("No machine, so no surface Reynolds number to measure.")
            return {}

        Re_surf = result.machine.Re_surf()
        if not len(Re_surf):
            raise ValueError(
                "A surface Reynolds number is measured against a blade surface, "
                "so iterating on one needs a blades: section in the config."
            )
        if not 0 <= self.i_row < len(Re_surf):
            raise ValueError(
                f"i_row={self.i_row} is out of range for a machine with "
                f"{len(Re_surf)} blade row(s)."
            )

        return {NAME_LOG_MU: float(np.log(Re_surf[self.i_row] / self.target))}

    @staticmethod
    def _mu(config):
        """Return the viscosity this iterator moves [kg/m/s]."""
        mu = getattr(config.fluid, "mu", None)
        if mu is None:
            raise ValueError(
                f"Cannot reach a surface Reynolds number by changing the "
                f"viscosity of a {type(config.fluid).__name__}, which has no "
                f"mu to change."
            )
        return mu


#
# THE REPEATING STAGE
#


def span_fractions(cut):
    """Return the span fraction of each *face* of a structured span cut.

    By arc length along the cut, not by index: `ember.cut
    .interpolate_to_structured` clusters its nodes cosine-wise, which on a
    seventeen-point cut differs from uniform by a tenth of the span --- and
    differs most at the endwalls, which is exactly where a profile is doing
    something.

    Faces rather than nodes because :func:`ember.average.mass_average` reduces
    over faces, so a nodal span fraction would be one longer than what it
    returns.
    """
    x = np.asarray(cut.x)[:, 0]
    r = np.asarray(cut.r)[:, 0]
    arc = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x), np.diff(r)))])
    spf = arc / arc[-1]
    return 0.5 * (spf[:-1] + spf[1:])


def exit_profile(result, order, offset=None):
    """Return the Legendre coefficients of the profile leaving `result`.

    Cut the last station, mass-average each quantity over the pitch, subtract
    the mixed-out mean, normalise by that station's own dynamic head and
    dynamic temperature, and fit.

    Normalising by the *exit* station's own scales rather than the inlet's is
    what makes "repeating" mean the shape repeats: a stage raises or drops the
    level, and it is the redistribution about that level which comes round
    again.

    Parameters
    ----------
    result : Result
        A solved run.
    order : int
        Highest Legendre mode to fit. Modes start at 1: the constant is
        dropped, a profile being a redistribution rather than a level.
    offset : float or None
        Cut plane offset in blade chords; `mixout.CUT_OFFSET` by default.

    Returns
    -------
    dict
        One tuple of `order` coefficients per column of
        :data:`turbigen.bconds.InletProfile.COLUMNS`.

    """
    from turbigen import bconds, mixout  # noqa: PLC0415 - avoids a cycle

    grid, machine = result.grid, result.machine
    xr = mixout.cut_planes(
        machine.annulus, mixout.CUT_OFFSET if offset is None else offset
    )[-1]

    cut = ember.cut.unstructured(grid, xr)
    if cut is None:
        raise ValueError(f"The exit cut plane at {xr.tolist()} misses the grid.")

    # Structured so that the pitch is an axis to average over. Sized from the
    # block it came out of, so the profile is measured at the resolution the
    # mesh has rather than one chosen here.
    block = grid[-1]
    structured = ember.cut.interpolate_to_structured(
        cut, (block.shape[1], block.shape[2])
    )

    mean = ember.average.mix_out(cut)
    spf = span_fractions(structured)

    def pitchwise(name):
        """Mass-average one quantity over the pitch, leaving the span.

        Mass-weighted rather than area-weighted, so a low-momentum wake cannot
        pull the profile around --- the same choice `_incidence` makes.
        """
        return ember.average.mass_average(
            getattr(structured, name), structured, axes=(1,)
        )

    # The scales are the exit station's own, and both vanish with the flow.
    q = float(mean.Po) - float(mean.P)
    dT = float(mean.To) - float(mean.T)

    deficit = {
        "DPo": (pitchwise("Po") - float(mean.Po)) / q,
        "DTo": (pitchwise("To") - float(mean.To)) / dT,
        "DAlpha": pitchwise("Alpha") - float(mean.Alpha),
        "DBeta": pitchwise("Beta") - float(mean.Beta),
    }

    coefficients = {}
    for name in bconds.InletProfile.COLUMNS:
        if name not in Repeat.COLUMNS:
            continue
        # legfit returns modes 0 upwards; the constant is dropped rather than
        # carried, so a measured level cannot leak into a profile that is
        # defined as a redistribution.
        fit = legendre.legfit(2.0 * spf - 1.0, deficit[name], order)
        coefficients[name] = tuple(float(value) for value in fit[1:])

    return coefficients


class Repeat(Iterator):
    """Pass the exit profile back to the inlet, until the stage feeds itself.

    A repeating stage --- the middle of a multistage machine --- is fed by its
    own exit. So the inlet profile is not something to state but something to
    find, and finding it is a fixed point.

    **The copy is the existing step rule.** With the error taken as
    ``inlet - outlet``, :func:`step`'s own ``u -= gain * e`` at ``gain = 1``
    gives exactly ``u_new = outlet``, so this needs no loop and no stepper of
    its own; ``gain`` below one is the relaxation the package this replaces
    called ``relaxation_factor``.

    What is passed upstream is Legendre coefficients rather than a sampled
    profile. A sampled one is three columns over as many span stations as the
    mesh has, which would make a dense Broyden Jacobian of that size squared
    and archive a mesh artefact into every `output.yaml`; the coefficients of a
    low-order fit are few, independent, smooth over mesh noise, and a
    resolution somebody chose.

    **Low order is a claim about the physics.** A Legendre fit to an endwall
    boundary layer is pointwise poor and integrally good: order 4 recovers only
    a third of the wall deficit but gets the blockage to within 4 per cent, and
    the blockage stops improving past order 8 while the pointwise error keeps
    falling. That is the right trade only if what propagates round a repeating
    loop is the integrated deficit rather than the wall value --- which it
    should be, the near-wall flow being re-established by the no-slip wall just
    downstream of the inlet plane. If that turns out to be wrong the answer is
    a wall-clustered fitting coordinate, not a higher order.

    ``DBeta`` is not carried: pitch angle at a repeating station is essentially
    zero, and a fourth column would be noise.
    """

    type: ClassVar[str] = "repeat"

    COLUMNS: ClassVar[tuple[str, ...]] = ("DPo", "DTo", "DAlpha")
    """The profile columns this iterator owns."""

    ANGLES: ClassVar[tuple[str, ...]] = ("DAlpha", "DBeta")
    """Those measured in degrees rather than in fractions of a scale."""

    order: int = 3
    """Highest Legendre mode passed upstream, the modes starting at 1."""

    offset: float = 0.5
    """Where to read the exit profile, in blade chords past the trailing edge.

    Far enough that the blade wakes have begun to mix but the plane is still in
    the machine. The package this replaces reads at the same distance.
    """

    gain: float = 1.0
    """One copies the exit profile outright; less under-relaxes it."""

    atol_head: float = 0.01
    """Converged when ``DPo`` and ``DTo`` are within this [--].

    In fractions of dynamic head and of dynamic temperature, which is what
    those columns are measured in.
    """

    atol_angle: float = 0.1
    """Converged when ``DAlpha`` is within this [deg]."""

    clip_head: float = 0.2
    """Most ``DPo`` and ``DTo`` may move in one iteration [--]."""

    clip_angle: float = 5.0
    """Most ``DAlpha`` may move in one iteration [deg]."""

    def __post_init__(self):
        if self.order < 1:
            raise ValueError(
                f"repeat.order must be at least 1, got {self.order}. Mode 0 is "
                f"the constant, which a profile does not carry."
            )

    #
    # THE PROTOCOL
    #

    def names(self):
        """Return the table key of every coefficient, in a fixed order."""
        return [
            f"inlet_profile.{name}[{mode}]"
            for name in self.COLUMNS
            for mode in range(self.order)
        ]

    def unknowns(self, config):
        profile = config.inlet_profile
        stored = {}
        for name in self.COLUMNS:
            values = getattr(profile, name, ()) if profile is not None else ()
            # Zeros where there is no profile yet, or where it is shorter than
            # this iterator wants: a uniform inlet is the first iteration.
            stored[name] = tuple(values) + (0.0,) * (self.order - len(values))

        return {
            f"inlet_profile.{name}[{mode}]": float(stored[name][mode])
            for name in self.COLUMNS
            for mode in range(self.order)
        }

    def with_unknowns(self, config, values):
        from turbigen import bconds  # noqa: PLC0415 - avoids a cycle

        current = self.unknowns(config)
        moved = {**current, **{k: v for k, v in values.items() if k in current}}

        columns = {
            name: tuple(
                moved[f"inlet_profile.{name}[{mode}]"] for mode in range(self.order)
            )
            for name in self.COLUMNS
        }

        return dataclasses.replace(config, inlet_profile=bconds.Legendre(**columns))

    def paths(self, config):
        # The one iterator whose knobs are its leaves, one for one, so the two
        # namings coincide rather than needing translation.
        return set(self.unknowns(config))

    def error(self, config, result):
        if result.grid is None or result.machine is None:
            logger.debug("No grid, so no exit profile to pass upstream.")
            return {}

        measured = exit_profile(result, self.order, self.offset)
        current = self.unknowns(config)

        # inlet minus outlet, so that `u -= gain * e` at gain one lands on the
        # outlet exactly.
        return {
            f"inlet_profile.{name}[{mode}]": (
                current[f"inlet_profile.{name}[{mode}]"] - measured[name][mode]
            )
            for name in self.COLUMNS
            for mode in range(self.order)
        }

    #
    # TWO SCALES, NOT ONE
    #

    def _by_column(self, head, angle):
        """Return `head` or `angle` for each knob, by which column it is in."""
        return {
            f"inlet_profile.{name}[{mode}]": (angle if name in self.ANGLES else head)
            for name in self.COLUMNS
            for mode in range(self.order)
        }

    def tolerances(self, config):
        """Return a tolerance per knob, in that knob's own units.

        The inherited `tolerance` is unused, and so is `clip`. `Iterator`
        carries one of each because most members want one of each; a member
        whose columns are measured in different units has no way to say so
        through them. Ignored outright rather than blended, so setting one
        cannot quietly do half of something.
        """
        del config
        return self._by_column(self.atol_head, self.atol_angle)

    def clips(self, config):
        del config
        return self._by_column(self.clip_head, self.clip_angle)
