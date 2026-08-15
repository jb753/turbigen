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
frozen :class:`~turbigen2.config.Config` cannot allow and which is the same
confusion between a design and its result that the rebuild exists to remove.
Here an iterator returns a new config and owns no state at all.
"""

import dataclasses
import logging
from typing import ClassVar

import numpy as np

import ember.cut
from turbigen2.node import Node

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


def converge(config, run, max_iter=10):
    """Iterate `config` until every error is within tolerance.

    Parameters
    ----------
    config : Config
        Where to start. Not modified: each iteration returns a new one.
    run : callable
        Takes ``(config, i_iter)`` and returns the :class:`~turbigen2.result.
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

        logger.info(f"Iteration {i_iter}:\n{format_table(config, result)}")

        if converged(config, result):
            logger.info(f"Converged after {i_iter + 1} iteration(s).")
            return config, result, True

        # The stepped config is what the next pass runs, so the one returned
        # alongside a result is always the one that produced it.
        stepped = step(config, result, history)
        history.append((unknowns(config), measured_errors(config, result)))
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
