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

After assembly the iterators disappear, and the step is ``u -= gain * e`` on
every knob, clipped. A better step rule --- one warm-started from a fit over
previous runs, say --- is a change to :func:`step` alone, and touches no
iterator. That table is also what such a fit would
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

import ember.average
import ember.cut
import numpy as np

import turbigen.clark
import turbigen.loading
import turbigen.shapespace
import turbigen.util
from turbigen.design import DesignError
from turbigen.node import Node
from turbigen.result import Result

logger = logging.getLogger("turbigen.iterate")
"""Iteration-level messages: the table, the verdict, what the stepper noticed.

Named apart from what one run says so that `iterate` can quieten a hundred runs
on the console without losing the few lines that describe the iteration itself.
"""


class MeasurementError(Exception):
    """A knob's error could not be measured from the solution.

    Raised rather than returned as an absent knob, because a knob missing from
    the table is one the loop stops correcting without ever saying so: the
    iteration carries on, reports the knobs it still has, and calls itself
    converged on a design nothing checked. What cannot be measured is a fact
    about the run, and the run should stop and say it.

    The callers that legitimately have nothing to measure say so themselves:
    see :func:`errors`, whose `strict` is how a report draws a march that blew
    up.
    """


TINY = 1e-9
"""Below this a nominal value is treated as zero when something is relative to it."""

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

    targets: ClassVar[tuple[str, ...]] = ()
    """Fields of this iterator that say what the design is aiming for.

    A sweep over a target is a sweep over designs, so
    :mod:`turbigen.database` measures distance along these as well as along
    the machine itself. Everything else an iterator holds --- gains, clips,
    tolerances --- is how the answer is reached, not what it is, and two runs
    differing only in those are one design.

    Declared rather than deduced, for the reason :attr:`from_solution` is.
    """

    gain: float | tuple[float, ...] = 1.0
    """How much of the error to subtract from the unknown.

    Carries the sign of the local sensitivity as well as its size: the step is
    always ``u -= gain * e``, so an iterator whose error *falls* as its knob
    rises declares a negative gain. Reciprocal of an assumed slope, so it is
    the crudest possible Newton step.

    Kept for the whole run: no step is fitted to the trajectory. Secants
    learned from a few clipped moves on a partly converged march were
    dominated by noise, and more than once came out with the wrong sign.
    Every step is therefore the one the declared gain asserts, and the sign
    has to follow from what the knob *is*.

    **One number or one per knob.** A scalar is a single declared prior, spread
    over every knob the iterator owns; a sequence carries them separately, in
    the order :meth:`unknowns` returns. A design is declared with the scalar,
    because a prior is a statement about a *kind* of knob and the count is
    rarely known while writing a file. A sequence is matched to the knobs it
    was written for --- see :meth:`_gain_each`, which refuses the mismatch it
    can see and cannot see the rest.
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

    def check(self, machine):
        """Raise :class:`~turbigen.design.DesignError` if `machine` cannot
        meet this iterator's target.

        Called on every design, before anything is solved, so a batch screens
        out a point whose target is infeasible rather than spending a job
        finding out. Reads the design alone: whatever needs a flow field is
        :meth:`error`'s business. Does nothing by default.
        """

    def tolerances(self, config):
        """Return the tolerance on each unknown, by name."""
        return {name: self.tolerance for name in self.unknowns(config)}

    def clips(self, config):
        """Return the largest step for each unknown, by name."""
        return {name: self.clip for name in self.unknowns(config)}

    def gains(self, config):
        """Return the gain of each unknown, by name.

        A scalar :attr:`gain` is spread over every knob; a sequence is matched
        to them in the order :meth:`unknowns` returns, which is the order a
        config file has to declare it in.
        """
        names = list(self.unknowns(config))
        return dict(zip(names, self._gain_each(names)))

    def _gain_each(self, names):
        """Return the gain of each name in `names`, spreading a scalar over them.

        A declared sequence has to match the knobs it claims to describe. It
        will not when a config carries one written for a different design ---
        a blade that has gained a section, a :class:`ClarkProfile` whose
        thickness order was changed --- and silently reinterpreting those numbers
        against knobs they were never meant for is worse than refusing them,
        because it would be a wrong sensitivity rather than an absent one.
        """
        names = list(names)
        if isinstance(self.gain, (int, float)):
            return [float(self.gain)] * len(names)

        if len(self.gain) != len(names):
            raise ValueError(
                f"{type(self).__name__} was given {len(self.gain)} gain(s) for "
                f"{len(names)} knob(s): {names}. A sequence of gains is one per "
                f"unknown, in that order; write a single number to declare one "
                f"prior for all of them."
            )
        return [float(value) for value in self.gain]


class Iteration(Node):
    """Closing the loop between a design and the CFD that tests it.

    A mapping rather than the bare list of iterators it used to be, because
    the loop has a setting of its own --- how many passes to allow --- and a
    list has nowhere to put one. It lived at the top of the file instead,
    where it read as a property of the case rather than of the iteration, and
    where it was the one top-level key belonging to no stage of the design.

    The other two commands with settings of their own, `chic:` and `batch:`,
    are mappings for the same reason; this makes the three alike.
    """

    correct: tuple[Iterator, ...] = ()
    """The mismatches to correct, one iterator each."""

    max_iter: int = 10
    """Most design iterations before giving up.

    A budget, not a target: a design that converges stops without reaching it.
    Here rather than beside the design it bounds, because it says how hard to
    try rather than what to build --- and an archived case still records what
    it was run under, which is why it is a key at all rather than a flag.
    """

    def check(self, machine):
        """Raise :class:`~turbigen.design.DesignError` if any iterator's target
        is infeasible for `machine`."""
        for iterator in self.correct:
            iterator.check(machine)


#
# THE STEPPER
#
# Everything below is generic. It knows names, numbers and tolerances, and
# nothing about angles, blades or mean lines.
#


def selected(config, from_solution):
    """Return `config` carrying only the iterators of one speed.

    The stepper takes a config and reads `config.iterate.correct` off it, so
    selecting a subset is done by handing it a config holding only those ---
    keeps every function below generic over *which* iterators it is stepping,
    without a second argument threaded through all of them.
    """
    return dataclasses.replace(
        config,
        iterate=dataclasses.replace(
            config.iterate,
            correct=tuple(
                iterator
                for iterator in config.iterate.correct
                if iterator.from_solution == from_solution
            ),
        ),
    )


def unknowns(config):
    """Return every configured iterator's unknowns, merged.

    Raises if two iterators claim the same name, which would otherwise make the
    result depend on the order they happen to appear in.
    """
    merged = {}
    for iterator in config.iterate.correct:
        for name, value in iterator.unknowns(config).items():
            if name in merged:
                raise ValueError(
                    f"Two iterators both claim the design variable {name!r}; "
                    "each one must own the variables it moves."
                )
            merged[name] = value
    return merged


def errors(config, result, strict=True):
    """Return every configured iterator's error, merged.

    **Strict by default**, which is what the iteration wants: an iterator that
    cannot measure raises :class:`MeasurementError`, and a loop correcting a
    design on knobs it can no longer see is worse than a loop that stops.

    `strict=False` is for a caller that is describing a run rather than
    steering one --- a report of a march that diverged has nothing to measure
    and is exactly the report someone needs. Each iterator that cannot measure
    is logged and its knobs omitted, and the rest are returned as usual.

    Tolerant of any exception, not only :class:`MeasurementError`: a field of
    NaNs fails wherever it is first touched, often deep in a property call that
    knows nothing of iterators, and the report it takes away is the same.
    """
    merged = {}
    for iterator in config.iterate.correct:
        try:
            merged.update(iterator.error(config, result))
        except Exception as err:
            if strict:
                raise
            logger.warning(
                f"Leaving the knobs of {type(iterator).__name__} out of the "
                f"table, because {err}"
            )
    return merged


def converged(config, result):
    """Return whether every measured error is within its tolerance.

    An unknown whose error was not measured counts as unconverged: the run had
    nothing to say about it, which is not the same as it being right.
    """
    measured = measured_errors(config, result)

    for iterator in config.iterate.correct:
        for name, tolerance in iterator.tolerances(config).items():
            if name not in measured or not np.abs(measured[name]) <= tolerance:
                return False

    return True


def measured_errors(config, result, strict=True):
    """Return the errors `result` reports, preferring the ones it recorded.

    A run stores what its iterators measured, so re-measuring would repeat work
    that is not free --- :meth:`Incidence.error` cuts the grid --- and would
    fail outright for a result whose grid has since been released.
    """
    if result.error:
        return dict(result.error)
    return errors(config, result, strict=strict)


def properties(config):
    """Return the gain, clip and tolerance of every unknown, by name."""
    gain, clip, tolerance = {}, {}, {}

    for iterator in config.iterate.correct:
        gains = iterator.gains(config)
        clips = iterator.clips(config)
        tolerances = iterator.tolerances(config)
        for name in iterator.unknowns(config):
            gain[name] = gains[name]
            clip[name] = clips[name]
            tolerance[name] = tolerances[name]

    return gain, clip, tolerance


def step(config, result):
    """Return the config to try next, from the errors `result` reports.

    ``u -= gain * e`` on every knob, each clipped to its own :attr:`~Iterator.
    clip`. The clip is the trust bound: what keeps a bad early step, taken on
    a field that has not settled, from throwing the design somewhere it cannot
    be meshed.

    **Clipped per knob, not scaled as a whole.** Scaling the step until its
    worst knob is inside its limit hands every knob to whichever one is
    furthest over: one thickness coefficient marching against its clip every
    pass once held a circulation coefficient to a fifth of its own step for a
    whole run.

    Parameters
    ----------
    config : Config
        The design that was run.
    result : Result
        What running it achieved.

    """
    measured = measured_errors(config, result)
    values = unknowns(config)
    gain, clip, tolerance = properties(config)

    # A knob with no gain is held, which is how an iterator says it does not
    # want to move; so is one whose tolerance is nothing to measure against.
    # Nothing else may be missing: `errors` is strict for the callers that
    # step a design, so a name absent here is a knob the loop would correct
    # without ever having looked at it.
    missing = sorted(set(values) - set(measured))
    if missing:
        raise MeasurementError(
            f"the table is missing {missing}, so those knobs would be stepped "
            f"on nothing. An iterator that cannot measure raises rather than "
            f"returning fewer knobs, so this is a caller that assembled a "
            f"table from a tolerant `errors(..., strict=False)`."
        )

    moved = {}
    for name in values:
        if not gain[name] or not tolerance[name] > TINY:
            continue
        change = -gain[name] * measured[name]
        if clip[name]:
            change = float(np.clip(change, -clip[name], clip[name]))
        moved[name] = values[name] + change

    if not moved:
        logger.debug("Nothing measured to correct towards, so nothing moves.")
        return config

    for iterator in config.iterate.correct:
        mine = {
            name: moved[name] for name in iterator.unknowns(config) if name in moved
        }
        if mine:
            config = iterator.with_unknowns(config, mine)

    return config


def format_table(config, result):
    """Return a one-line-per-unknown summary of where the iteration stands.

    Tolerant of a knob it cannot measure, which it renders as `nan`: this
    describes a run rather than steering one, and a table that raised would
    take the log line away from exactly the run someone needs to read.
    """
    measured = measured_errors(config, result, strict=False)
    values = unknowns(config)

    tolerances = {}
    for iterator in config.iterate.correct:
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
    per-iteration log of directories that do not exist. Ten lines that share
    every piece of arithmetic beat a callable that has to pretend to solve.

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
    if not inner.iterate.correct:
        return config

    for _ in range(max_iter):
        # No solver, no grid, no output directory: designing is the whole run.
        result = Result(machine=inner.design())

        if converged(inner, result):
            logger.debug(f"Resolved the design-only knobs: {unknowns(inner)}")
            # The full list goes back on, so what comes out of here is the
            # config that was handed in with some of its leaves moved, rather
            # than one that has quietly lost the iterators it will need next.
            return dataclasses.replace(inner, iterate=config.iterate)

        inner = step(inner, result)

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
        # just finished, on a copy -- so this config holds its old value
        # beside an error measured after the move, and stepping the pair
        # would move it somewhere neither describes. `run` still gets the
        # whole config, so the nested resolve still sees its own.
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

        # `iterate` is put back whole because `stepping` holds only the
        # solution iterators.
        config = dataclasses.replace(step(stepping, result), iterate=config.iterate)

    logger.warning(f"Not converged after {max_iter} iteration(s).")
    return config, result, False


#
# THE ITERATORS
#


GAP_HINT = (
    "A section above a clearance gap has no blade to cut: either the march is "
    "not a flow field, or the section belongs below the gap."
)
"""Why a blade surface may be missing, appended to the error that says so."""


def _check_i_row(i_row, n_row):
    """Raise unless `i_row` indexes one of `n_row` blade rows."""
    if not 0 <= i_row < n_row:
        raise ValueError(
            f"i_row={i_row} is out of range for a machine with {n_row} blade row(s)."
        )


def _require_grid(result, what):
    """Raise :class:`MeasurementError` unless `result` has a grid to cut."""
    if result.grid is None or result.machine is None:
        raise MeasurementError(
            f"there is no solved grid to cut, so {what} is unmeasured."
        )


def _merged(current, values):
    """Return `current` with those of its own keys that `values` holds moved."""
    return current | {k: v for k, v in values.items() if k in current}


def _with_blade(config, i_row, **changes):
    """Return `config` with the fields `changes` names set on row `i_row`."""
    blades = list(config.blades)
    blades[i_row] = dataclasses.replace(blades[i_row], **changes)
    return dataclasses.replace(config, blades=tuple(blades))


def _with_sections(config, i_row, fn):
    """Return `config` with row `i_row`'s sections replaced by `fn(i, section)`.

    An iterator that measures at one span fraction moves every section by the
    same amount, so whatever spanwise variation the design asked for survives
    being iterated: only one span was ever measured, and this has nothing to
    say about the others.
    """
    sections = tuple(fn(i, s) for i, s in enumerate(config.blades[i_row].sections))
    return _with_blade(config, i_row, sections=sections)


def _recamber_unknowns(config, field):
    """Return the mean recamber of each row, under `field`.

    One number per row rather than one per section: the spanwise distribution
    of recamber is a design decision, and :class:`Deviation`, which matches a
    single mixed-out angle, has nothing to say about it. :class:`Incidence`
    does, so it owns its sections individually rather than through this.
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
    for i_row, blade in enumerate(config.blades):
        name = f"{field}[{i_row}]"
        if name not in values:
            continue

        current = np.mean([getattr(section, field) for section in blade.sections])
        shift = values[name] - current
        config = _with_sections(
            config,
            i_row,
            lambda _, s, shift=shift: dataclasses.replace(
                s, **{field: getattr(s, field) + shift}
            ),
        )

    return config


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

    gain: float = 0.5
    clip: float = 1.0
    """Largest recamber in one iteration [deg].

    One degree, not two: a single large recamber moves the blade count and
    throws the solution off, and a tighter clip bounds that excursion to
    something the next iteration can walk back.
    """
    tolerance: float = 0.5
    """Permissible error on exit flow angle [deg]."""

    def unknowns(self, config):
        return _recamber_unknowns(config, "dchi_TE")

    def with_unknowns(self, config, values):
        return _with_recamber(config, "dchi_TE", values)

    def paths(self, config):
        return _recamber_paths(config, "dchi_TE")

    def error(self, config, result):
        if result.actual is None or result.machine is None:
            raise MeasurementError(
                "there is no mixed-out mean line to read an exit angle off, so "
                "the deviation of every row is unmeasured."
            )

        nominal = result.machine.mean_line
        return {
            f"dchi_TE[{i_row}]": float(
                result.actual[:, i_row].Alpha_rel[1] - nominal[:, i_row].Alpha_rel[1]
            )
            for i_row in range(len(config.blades))
        }


def _sections(config):
    """Every ``(row index, section index, section)``, in streamwise order."""
    return [
        (i_row, i_section, section)
        for i_row, blade in enumerate(config.blades)
        for i_section, section in enumerate(blade.sections)
    ]


class Incidence(Iterator):
    """Set the leading edge to meet the flow at a chosen incidence.

    Measured from the stagnation point on the blade itself, at each section's
    own span fraction, rather than from the mixed-out mean line: incidence is a
    local property of the leading edge, and the whole point of moving one is
    that the mean line does not see what the tip and the hub are doing.

    On the blade rather than on a plane ahead of it, because where the flow
    attaches is the thing being controlled. A flow angle read upstream --- what
    this class used to do --- answers a slightly different question at every
    distance it is read from, and the distance was a setting nobody had grounds
    to choose.

    One knob per section, not the row mean :class:`Deviation` uses: a blade
    with N sections has N independent leading-edge angles, and each should meet
    the flow it actually sees. Collapsing them to a mean nulls the incidence at
    one span and leaves the rest with whatever the starting distribution gave.

    **Positive onto the pressure surface, on every row.** What the blade
    measures is the flow angle less the metal angle, and that has the sign of
    the angles rather than of the aerofoil: the same number is incidence onto
    the pressure surface of a row whose metal angle falls and onto the suction
    surface of one whose angle rises. So :attr:`target` is turned into that
    frame row by row, by the blade's own reading of which side its pressure
    surface is on (:attr:`~turbigen.blade.Blade.suction_is_upper`), and the
    error stays in the frame the measurement and :attr:`gain` are in --- which
    is what lets one negative gain serve rows turning either way.
    """

    type: ClassVar[str] = "incidence"

    targets: ClassVar[tuple[str, ...]] = ("target",)

    target: float = 0.0
    """Incidence to aim for, positive onto the pressure surface [deg].

    The same physical incidence on every row, whichever way it turns: a
    positive target asks for flow arriving from the pressure side of the
    metal angle, the side that demands more turning.
    """

    # Negative because incidence falls as the recamber rises. Small because
    # the stagnation point moves several degrees for one degree of metal.
    gain: float = -0.05
    clip: float = 1.0
    tolerance: float = 8.0
    """Permissible error on local incidence [deg].

    Loose next to a deviation, because it is measured on the nose: how far the
    surface turns across the cell the flow stagnated in bounds how accurate a
    discrete solution can be about it, and on a mesh that resolves a blade well
    that is still a few degrees. `_report_resolution` says when a case is
    asking for more than its own mesh can answer.
    """

    def unknowns(self, config):
        return {
            f"dchi_LE[{i_row}][{i_section}]": float(section.dchi_LE)
            for i_row, i_section, section in _sections(config)
        }

    def with_unknowns(self, config, values):
        for i_row, blade in enumerate(config.blades):
            names = [f"dchi_LE[{i_row}][{j}]" for j in range(len(blade.sections))]
            if any(name in values for name in names):
                config = _with_sections(
                    config,
                    i_row,
                    lambda j, s, names=names: (
                        dataclasses.replace(s, dchi_LE=values[names[j]])
                        if names[j] in values
                        else s
                    ),
                )

        return config

    def paths(self, config):
        return {
            f"blades[{i_row}].sections[{i_section}].dchi_LE"
            for i_row, i_section, _ in _sections(config)
        }

    def error(self, config, result):
        _require_grid(result, "the incidence onto every row")

        # One cut of each blade, not one per section: the cut is the expensive
        # part of the measurement and does not depend on span.
        surfaces = turbigen.util.cut_blade_surfs(result.grid)

        measured = {}
        for i_row, i_section, section in _sections(config):
            incidence = _incidence(
                result, surfaces[i_row], i_row, section.spf, self.tolerance
            )
            if not np.isfinite(incidence):
                raise MeasurementError(
                    f"the incidence onto row {i_row} section {i_section} could "
                    f"not be measured at spf={section.spf:.2f}. A section above "
                    f"a clearance gap has no blade surface to stagnate on: "
                    f"either the march is not a flow field, or the section sits "
                    f"in the gap and belongs below it."
                )
            measured[f"dchi_LE[{i_row}][{i_section}]"] = (
                incidence - self._target_measured(result, i_row)
            )

        return measured

    def _target_measured(self, result, i_row):
        """Return :attr:`target` as the flow angle less the metal angle [deg].

        Onto the pressure surface means flow arriving from the lower-angle side
        of the metal where the angle rises through the row, and from the
        higher-angle side where it falls, so the sign flips with the turning.
        Read off the blade's own assignment of its surfaces rather than off
        the turning at this section, so that "pressure surface" is the one the
        pressure-side thickness was hung on even on a blade twisted enough for
        a section to turn the other way.
        """
        blade = result.machine.rows[i_row].blade
        return self.target if blade.suction_is_upper else -self.target


def _incidence(result, surface, i_row, spf, tolerance=0.0):
    """Return the incidence onto row `i_row` at span fraction `spf` [deg].

    How far the flow has swept the stagnation point around the nose, as an
    angle. The dividing streamline meets the wall along its normal, so the yaw
    of the surface normal where the flow stagnates is the angle the flow
    arrives at; the metal angle it is measured against is the camber direction
    at the leading edge, which the blade knows exactly rather than by
    measurement --- a camber line is *built* from `chi_LE`.

    That makes this the angle the reference implementation subtends at the
    centre of the leading edge circle, without needing the circle: the radius
    entered there only as the constant relating an arc to an angle, and cancels
    out of a construction made from tangents. It is `R_LE` that would have been
    needed to keep it, and `R_LE` belongs to one thickness distribution rather
    than to the interface every distribution implements.

    NaN when there is nothing to measure --- a section above a clearance gap,
    or a leading edge the stagnation point could not be found on --- which the
    caller drops rather than steps on.
    """
    annulus = result.machine.annulus
    blade = result.machine.rows[i_row].blade

    if surface is None:
        return np.nan

    cut, xr = turbigen.util.cut_section(surface[0], annulus, i_row, spf)
    if cut is None:
        # Above a clearance gap the blade has no surface to cut, the span there
        # being trimmed off as flow rather than wall.
        return np.nan

    # The thickness vanishes at m = 0, so the first point of either surface is
    # the nose. It anchors the stagnation search window, which is what makes
    # the search robust on a strongly asymmetric leading edge.
    xrt_nose = blade.evaluate_section(spf, nchord=turbigen.loading.N_CHORD_NOSE)[0][
        :, 0
    ]

    i_stag, found = turbigen.util.get_i_stag(cut, xrt_LE=xrt_nose)
    if not found[0]:
        return np.nan

    # Refined to between nodes, because the integer index is a step function of
    # the flow: a leading edge that moves by less than a cell would show no
    # change at all, and then a whole cell's worth at once.
    zeta_stag = turbigen.util.get_zeta_stag(cut, i_stag)

    # Downstream meridional direction at the leading edge, off the same
    # annulus curve the cut was taken along. Arc length around a nose is
    # positive on both sides of it and cannot say which way is downstream.
    e_m = xr[:, 1] - xr[:, 0]
    e_m = e_m / np.linalg.norm(e_m)

    chi = float(blade.evaluate_chi(spf)[0])
    flow = float(turbigen.util.surface_normal_yaw(cut, zeta_stag, e_m, chi)[0])

    _report_resolution(cut, i_stag, e_m, chi, i_row, spf, tolerance)

    # Wrapped, because a normal yaw runs to a half turn either way while a
    # metal angle does not, and their difference is small by construction.
    return (flow - chi + 180.0) % 360.0 - 180.0


def _report_resolution(cut, i_stag, e_m, chi, i_row, spf, tolerance):
    """Say when the nose is meshed too coarsely for the tolerance asked of it.

    How far the surface turns across the cell the flow stagnated in, read off
    the same normal the measurement uses, and so needing no leading edge radius
    and no assumption about the shape of the nose.

    An estimate of accuracy, not of step size. The answer this reports on is
    smooth --- the stagnation point is located between nodes, and sweeping the
    span fraction across a cell moves the measured incidence by hundredths of a
    degree, not in steps. What a cell of nose bounds is how far the pressure
    peak of a *discrete* solution can sit from the real one, which is a
    truncation error: it does not shrink as the solver converges, and no
    refinement of the peak's location within the mesh can see it.

    Compared against the tolerance rather than against a constant, because
    coarse has no meaning here on its own: a nose good to five degrees is ample
    for a design that wants ten and hopeless for one that wants one.
    """
    zeta_line = turbigen.util.get_zeta(cut)
    i = int(np.clip(i_stag[0], 1, cut.shape[0] - 2))

    # One call an arc length: `surface_normal_yaw` reads one point per j-line,
    # and the two wanted here are two points on the same one.
    neighbours = [
        turbigen.util.surface_normal_yaw(cut, zeta_line[k, :], e_m, chi)[0]
        for k in (i - 1, i + 1)
    ]
    turn = 0.5 * abs(neighbours[1] - neighbours[0])

    if turn > 2 * tolerance:
        logger.info(
            f"The leading edge of row {i_row} at spf={spf:.2f} turns "
            f"{turn:.1f} deg across the cell the flow stagnated in, so its "
            f"incidence is unlikely to be accurate to the {tolerance:.1f} deg "
            "being iterated to. Refine the nose, or ask for less."
        )


N_LOOP = 501
"""Points a target curve is integrated on to give the circulation it asks for.

Dense enough that the trapezoid is not what a `Co` error is limited by: the
Clark curve is piecewise and smooth, so the error falls as the square of the
spacing and five hundred intervals put it far below any tolerance a design
would set on a circulation coefficient.
"""


class RowIterator(Iterator):
    """Base for an iterator that shapes one blade row, measured at one span.

    One row per iterator: a stator and a rotor want different loading, so two
    rows means two entries. No `type`, so it is never read from a file itself.
    """

    i_row: int = 0
    """Index of the blade row to shape."""

    spf: float = 0.5
    """Span fraction to measure the distribution at [--]."""

    def _blade(self, config):
        """Return the blade this iterator shapes, checking that it exists."""
        _check_i_row(self.i_row, len(config.blades))
        return config.blades[self.i_row]


def _circulation_count(config, i_row):
    """Return row `i_row`'s blade count design, which has to have a `Co`."""
    from turbigen.blade import Circulation

    _check_i_row(i_row, len(config.blades))

    count = config.blades[i_row].count
    if not isinstance(count, Circulation):
        raise ValueError(
            f"Setting the level of a loading distribution means moving the "
            f"blade count, and row {i_row} counts its blades with a "
            f"{type(count).__name__}, which has no circulation coefficient "
            f"to move. Set count: {{type: Co, Co: 0.7}}."
        )
    return count


def _with_circulation(config, i_row, Co):
    """Return `config` with row `i_row`'s circulation coefficient set to `Co`."""
    count = _circulation_count(config, i_row)
    return _with_blade(config, i_row, count=dataclasses.replace(count, Co=Co))


class ClarkProfile(RowIterator):
    """Shape a two-sided thickness to a Clark loading distribution.

    Drives a :class:`~turbigen.thickness.ClarkThickness` towards
    :mod:`turbigen.clark` on both surfaces at once, with the leading-edge
    recamber, and the camber line between, moved with it.

    **The knobs are shape-space coefficients**
    (:attr:`~turbigen.thickness.ClarkThickness.tau_coeff`): leading edge
    radius, interior perturbations and wedge angle as one Bernstein curve.
    Each thickens its surface locally and speeds up the flow over it, so one
    gain, sign included, covers them all. Each is measured where it acts, at
    :attr:`~turbigen.thickness.ClarkThickness.m_ctl` mapped to each surface's
    own fraction of length.

    **Shared ends.** One nose radius and one wedge angle serve both surfaces,
    so each end's error is the mean of the two surfaces' residuals, the
    least-squares move for one knob. At the trailing edge the cost is that
    equal and opposite errors read as converged; per-surface control lives in
    the interior.

    **The nose's other half belongs to the recamber.** Equal and opposite
    errors at the nose are a leading-edge loading error, which no thickness
    can remove and incidence can: `dchi_LE` is driven by the suction residual
    less the pressure one at the first station. Every section of the row is
    recambered together, as the thickness is, so this owns the row's
    `dchi_LE` outright and refuses to sit beside an :class:`Incidence`
    iterator, which would move the same leaves towards a different target.

    **The level belongs to the blade count.** Thickness redistributes
    circulation but cannot create it, so `Co` is driven by the circulation
    error, integrated over the whole cut by
    :class:`~turbigen.loading.ClarkMeasurement` in the same units as `Co`. The
    shape knobs see the residuals with that level removed.

    **Measured from the geometric leading edge**, not the stagnation point, so
    the target does not move as the thickness does.
    """

    type: ClassVar[str] = "clark_profile"

    targets: ClassVar[tuple[str, ...]] = ("Ma_peak", "z_peak", "Ma_LE", "Ma_PS")

    Ma_peak: float = 1.2
    """Target peak Mach number over the trailing edge value [--].

    One more than the diffusion factor
    :class:`turbigen.metric.DiffusionFactor` records.
    """

    Ma_peak_max: float = 1.1
    """Largest suction peak Mach number the target may ask for [--].

    The peak itself, `Ma_peak` times the row exit relative Mach number, not
    the ratio. A Clark curve has no shock in it, so the default allows only a
    mildly supersonic peak; set it lower to keep a margin, or to `.inf` to
    switch the check off.
    """

    z_peak: float = 0.55
    """Target surface fraction of the suction peak [--]."""

    Ma_LE: float = 1.6
    """Height of the target's leading edge acceleration [--].

    The value Clark's ramp line takes at its reference station, which is what
    sets how hard the suction surface accelerates out of the nose. Not a Mach
    number the curve is required to reach at that station: see `Z_LE` in
    :mod:`turbigen.clark` for why the two are not the same thing.

    Carries the `Ma_2 / Ma_1` factor, Clark (2019) parameter 3, so the same
    number means the same style of leading edge across rows of differing
    duty. Divided back out once,
    on the way into :mod:`turbigen.clark`, which works in plain `Ma / Ma_TE`
    throughout --- a curve whose pieces are built from differences between its
    own parameters cannot carry two normalisations at once.
    """

    Ma_PS: float = 1.0
    """Target pressure-surface Mach number on the pre-acceleration plateau [--].

    Carries the `Ma_2 / Max_1` factor --- the exit Mach number over the inlet
    *axial* one, both relative --- and is divided by it on the way into
    :mod:`turbigen.clark`. Not the `Ma_2 / Ma_1` :attr:`Ma_LE` carries, though
    the two agree wherever the flow enters a row nearly axially.

    **Why the plateau is referred to a different velocity from the ramp.** What
    this states is how far the pressure surface may be allowed to fall before
    the passage accelerates it again, which is a claim about diffusion through
    the passage; what is there to diffuse is what passes through it, and that
    is the axial component. :attr:`Ma_LE` is a statement about how hard the
    flow is turned around the nose, which the blade meets at whatever angle it
    arrives, magnitude and all. Referring both to the magnitude makes the
    plateau demand rise with inlet swirl for no aerodynamic reason: at fifty
    degrees of relative inlet angle the two differ by half as much again, and a
    row asked for a plateau it cannot reach shows up as a loading residual that
    no thickness coefficient can move.

    :attr:`Ma_peak` keeps its plain `Ma / Ma_TE`, being a statement about the
    trailing edge value rather than about the inlet.
    """

    gain: float | tuple[float, ...] = 0.5
    """How much of the error to subtract from each shape knob.

    **One number covers every shape knob**, which is the whole reason the knobs
    are shape-space coefficients: a thicker surface is a faster one wherever
    the thickening happens, so nose, interior and wedge all share a sign.

    **Positive**, and the sign is a
    statement about what a coefficient *is* rather than a guess. A gain is the
    reciprocal of an assumed slope, the step being `u -= gain * e`; here the
    slope is positive, since raising a coefficient thickens the surface,
    accelerates the flow over it and lifts the `fac` the error is measured in.
    A surface running faster than its target is therefore one to thin. The
    *size* is still a guess, and kept for the whole run.
    A step `-gain * e` converges while `gain` times the true slope is between
    zero and two, and the slopes measured are 0.2 to 1.5, so one is about as
    large as it can be in theory. **In practice it is too large, so a half.**
    The margin from 1.5 to two is small beside the noise a restarted field and
    a recamber add to each residual. At a gain of one, a turbine stator went
    sixteen passes with its worst shape residual swinging between 2.5 and 11
    tolerances and never settling. Restarted from its last design at a half,
    that residual fell from 3.3 to 1.0 tolerances in two passes.

    :attr:`gain_Co` carries the level beside it. The two agree on sign; what
    keeps them apart is that a circulation coefficient and a shape-space
    coefficient are not the same quantity. See :meth:`gains`.
    """

    clip: float = 0.1
    """Largest change in one shape-space coefficient per iteration [--]."""

    tolerance: float = 0.02
    """Converged when every shape residual is within this [--]."""

    gain_Co: float = 0.5
    """How much of the level error to subtract from `Co`, as a prior [--].

    **A half, though the error is in the units of the knob.** The level is a
    circulation coefficient too high or too low, and `Co` is the circulation
    coefficient, so a slope of one is what a coefficient measured against
    itself ought to have. Measured, it is nearer two: moving `Co` changes the
    blade count, and the pitch moves the incidence, the deviation and the
    loading the level is read from along with it. At a gain of one a run sat
    in a period-two orbit of about 0.03 either side for a dozen passes; at a
    half the same case converged in ten. The half also damps the step a
    noisy reading of the level asks for, and the blade count is the knob every
    other one on the row answers to.

    Positive, because more circulation per blade is a bigger loop. **Read only while
    :attr:`gain` is a scalar**: declared as a sequence, `gain` carries every
    knob including this one.
    """

    clip_Co: float = 0.05
    """Largest change in the circulation coefficient per iteration [--]."""

    tolerance_Co: float = 0.02
    """Converged when the circulation is within this of the target's [--].

    A `Co` of order 0.7, so this is three per cent or so of it. Not tighter,
    because the level read off one solution scatters by about that much from
    pass to pass with `Co` itself held still.
    """

    tolerance_tau_LE: float = 0.02
    """Converged when the leading-edge shape residual is within this [--].

    The same as :attr:`tolerance` by default, but declared apart from it,
    because the nose is the knob most often left short of its target: a
    loading that wants a sharper leading edge than :attr:`R_LE_lim` allows
    holds ``tau_LE`` against that bound every pass, with a residual no step
    can clear. Widening this one keeps a clamped nose from stalling a design
    the other knobs have converged, without slackening them.
    """

    gain_dchi_LE: float = 20.0
    """How much of the leading-edge loading error to recamber by [deg].

    **Positive**, because :meth:`error` turns the loading error into the frame
    of the recamber, row by row: a nose loaded too heavily means flow arriving
    too far onto the pressure surface, and which way `dchi_LE` moves to take
    it off depends on which side the pressure surface is on
    (:attr:`~turbigen.blade.Blade.suction_is_upper`). One positive gain then
    serves rows turning either way, as one negative gain serves
    :class:`Incidence`.

    **The size is a guess**, from a stagnation point that moves several
    degrees for one of metal and a first station a tenth of the way along the
    surface: of order 0.05 in `fac` for a degree, so twenty is about the
    full Newton step. :attr:`clip_dchi_LE` bounds what a wrong guess costs.

    **Read only while :attr:`gain` is a scalar**, as :attr:`gain_Co` is.
    """

    clip_dchi_LE: float = 1.0
    """Largest leading-edge recamber in one iteration [deg].

    The clip :class:`Incidence` defaults to, for the same knob.
    """

    tolerance_dchi_LE: float = 0.02
    """Converged when the leading-edge loading error is within this [--].

    The difference of two residuals, so the same number is a tighter
    criterion than :attr:`tolerance` on either one.
    """

    R_LE_lim: tuple[float, float] = (0.01, 0.2)
    """Bounds on the leading edge radius, normalised by chord [--].

    **A bound on where the knob arrives, which no clip provides.** `clip`
    limits one step and not a run of them, so a nose can be thickened by a
    hundredth every pass and be six times its original size ten passes later
    without any single step being refused. Measured on a stage whose loading
    target the nose could not meet: `R_LE` climbed from 0.016 to 0.097 over
    nine passes, monotonically, and the two iterations that followed diverged
    in the CFD --- a nose radius of a tenth of a chord is a cylinder rather
    than a leading edge.

    **Clamped, not refused.** The closed-section guard above returns the
    config untouched, which is right for a section that cannot be meshed at
    all; here it would freeze ten other knobs because one reached a limit.
    Clamping lets the rest move and leaves this one against its bound with its
    error still in the table, which is where a knob that cannot get what it
    wants should be visible.

    The loop is not made to converge by this. A radius that walks one way
    without turning is a target the nose cannot reach: a shared end reports
    only the mean of the two surfaces, and a nose asked to be sharp on one
    side and blunt on the other is asking for a recamber rather than a
    radius. What the bound buys is a design that stays
    meshable and says so, rather than one that runs away and announces it as a
    divergence two iterations later. The default upper bound is wider than
    that run suggests, a guard against a runaway rather than a statement of a
    sensible nose.
    """

    def __post_init__(self):
        if not 0.0 < self.z_peak < 1.0:
            raise ValueError(
                f"A loading peak sits on the surface, so z_peak must satisfy "
                f"0 < z_peak < 1, got {self.z_peak}."
            )
        for name in ("Ma_peak", "Ma_peak_max", "Ma_LE", "Ma_PS"):
            if not getattr(self, name) > 0.0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}.")
        for name in ("tolerance_tau_LE", "tolerance_dchi_LE"):
            if not getattr(self, name) > 0.0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}.")

    #
    # THE PROTOCOL
    #

    def check(self, machine):
        """Raise unless the target's suction peak is within :attr:`Ma_peak_max`.

        :attr:`Ma_peak` is over the trailing edge value, so the peak it asks
        for is that times the row exit relative Mach number off the nominal
        mean line.
        """
        _check_i_row(self.i_row, machine.mean_line.n_row)
        Ma_TE = float(machine.mean_line[:, self.i_row].Ma_rel[1])
        if self.Ma_peak * Ma_TE > self.Ma_peak_max:
            raise DesignError(
                f"Row {self.i_row}: target suction peak "
                f"Ma_peak * Ma_TE = {self.Ma_peak} * {Ma_TE:.3f} = "
                f"{self.Ma_peak * Ma_TE:.3f} exceeds "
                f"Ma_peak_max={self.Ma_peak_max}."
            )

    def unknowns(self, config):
        """Return the level, the recamber, then every shape-space coefficient.

        **`Co` and `dchi_LE` lead, and the order is load-bearing.** A sequence
        :attr:`gain` is matched to this order, and the first two positions are
        the ones that cannot move when a blade changes order: after the shape
        knobs, a gain written for a circulation coefficient would silently be
        read as one for a thickness.

        The recamber and each coefficient are the mean over the row's
        sections, as :meth:`with_unknowns` shifts them all together.
        """
        level = {f"Co[{self.i_row}]": float(_circulation_count(config, self.i_row).Co)}
        recamber = {self._dchi_name: self._dchi_LE(config)}
        coefficients = self._flat_coeff(self._coefficients(config))
        return level | recamber | {
            name: float(value)
            for name, value in zip(self._names(self._order(config)), coefficients)
        }

    def with_unknowns(self, config, values):
        current = self.unknowns(config)
        moved = _merged(current, values)

        co_name = f"Co[{self.i_row}]"
        if moved[co_name] != current[co_name]:
            config = _with_circulation(config, self.i_row, moved[co_name])

        recamber = moved[self._dchi_name] - current[self._dchi_name]
        if recamber:
            config = _with_sections(
                config,
                self.i_row,
                lambda _, s: dataclasses.replace(s, dchi_LE=s.dchi_LE + recamber),
            )

        order = self._order(config)
        names = self._names(order)
        shift = self._unflat(
            np.array([moved[name] - current[name] for name in names]), order
        )
        if not np.any(shift):
            return config

        # A uniform shift, for the reason `_with_sections` gives, but not
        # through it: the moved sections are checked before being written.
        #
        # Through `with_tau_coeff` rather than onto the config leaves, because
        # moving an end coefficient moves the straight line beneath the whole
        # curve and every interior perturbation has to be recomputed against
        # it --- see there.
        sections = tuple(
            dataclasses.replace(
                section,
                thickness=section.thickness.with_tau_coeff(
                    section.thickness.tau_coeff + shift
                ),
            )
            for section in config.blades[self.i_row].sections
        )

        sections = self._within_R_LE(sections)

        closed = self._too_thin(sections)
        if closed is not None:
            logger.info(
                f"Moving row {self.i_row}'s thickness this far would leave "
                f"{closed}, which is not a section that can be meshed; "
                f"holding the shape where it is."
            )
            return config

        return _with_blade(config, self.i_row, sections=sections)

    def _within_R_LE(self, sections):
        """Return `sections` with any nose radius pulled back inside its bounds.

        Written on the radius rather than on the shape-space coefficient it
        comes from, because the bound is a statement about the aerofoil: a
        reader asking how blunt a nose is allowed to get should not have to
        square a coefficient to find out.
        """
        lo, hi = self.R_LE_lim
        held = []
        for section in sections:
            R_LE = float(section.thickness.R_LE)
            bounded = min(max(R_LE, lo), hi)
            if bounded == R_LE:
                held.append(section)
                continue

            logger.info(
                f"Row {self.i_row}'s nose would go to R_LE={R_LE:.4g}, outside "
                f"the {lo:.4g} to {hi:.4g} it is allowed; holding it at "
                f"{bounded:.4g}."
            )
            coeff = section.thickness.tau_coeff.copy()
            coeff[:, 0] = turbigen.shapespace.tau_LE(bounded)
            held.append(
                dataclasses.replace(
                    section, thickness=section.thickness.with_tau_coeff(coeff)
                )
            )
        return tuple(held)

    def paths(self, config):
        order = self._order(config)
        paths = {f"blades[{self.i_row}].count.Co"}
        for i_section in range(len(config.blades[self.i_row].sections)):
            paths |= {f"blades[{self.i_row}].sections[{i_section}].dchi_LE"}
            stem = f"blades[{self.i_row}].sections[{i_section}].thickness"
            paths |= {f"{stem}.R_LE"}
            paths |= {f"{stem}.tanwedge"}
            paths |= {
                f"{stem}.coeff[{i_surf}][{j}]"
                for i_surf in (0, 1)
                for j in range(order - 1)
            }
        return paths

    def error(self, config, result):
        _require_grid(result, f"the loading profile of row {self.i_row}")
        self._check(config)
        _circulation_count(config, self.i_row)

        measured = turbigen.loading.measure_clark_profile(
            result, self.i_row, self.spf, self._thickness(config).m_ctl
        )
        if measured is None:
            raise MeasurementError(
                f"both surfaces of row {self.i_row} could not be read at "
                f"spf={self.spf:.2f}, so its loading profile is unmeasured. "
                f"{GAP_HINT}"
            )
        residual = measured.fac - self.target(measured.z, result.machine)

        # What the blade count got wrong, as a circulation: the loop the blade
        # drew against the loop the target asks for, both as `Co`. The target's
        # is integrated from `target` itself on a dense grid rather than from a
        # closed form written here, for the reason `target` is public at all
        # --- a second opinion on the Clark curve would be free to contradict
        # the one the design is iterated against.
        dense = np.linspace(0.0, 1.0, N_LOOP)
        wanted = self.target(np.stack((dense, dense)), result.machine)
        Co_target = float(
            np.trapezoid(wanted[0], dense)
            - measured.length_ratio * np.trapezoid(wanted[1], dense)
        )
        level = float(measured.Co - Co_target)

        # Taken back off the residuals, so what is left is the shape's alone: a
        # thickness redistributes circulation and cannot create it, and a
        # coefficient chasing the loop would push every knob one way for
        # nothing. A uniform shift of `d` on one surface and `-d` on the other
        # moves the loop by `d (1 + L_ps/L_ss)`, so that is what a level of
        # this size stands on -- and at equal surface lengths it is the half
        # each that the two surfaces being treated alike has always meant.
        delta = level / (1.0 + measured.length_ratio)
        shape = residual - np.array([[delta], [-delta]])

        # What the recamber can answer for and a shared nose radius cannot: the
        # suction surface running ahead of the pressure one at the first
        # station, which is the nose loaded too heavily and the flow arriving
        # too far onto the pressure surface. From `shape`, so the level --- a
        # `+delta` on one surface and `-delta` on the other --- is not read as
        # incidence. Turned into the recamber's frame as `Incidence` turns its
        # target: a rising metal angle takes flow off the pressure surface
        # where that surface is the lower one, and puts it on where it is the
        # upper, so the sign flips with the turning. See `gain_dchi_LE`.
        loading_LE = float(shape[0][0] - shape[1][0])
        blade = result.machine.rows[self.i_row].blade
        sgn = -1.0 if blade.suction_is_upper else 1.0

        errors = {f"Co[{self.i_row}]": level, self._dchi_name: sgn * loading_LE}
        return errors | dict(
            zip(self._names(self._order(config)), map(float, self._flat_error(shape)))
        )

    def target(self, z, machine):
        """Return the Clark distribution on each surface at `z`, shape (2, n).

        Public because the surface-distribution plot draws the same curve this
        iterates against --- a report drawing a target of its own would be free
        to contradict the design it describes.
        """
        # The one place the factors the two front parameters carry are taken
        # back out; see the attributes. Everything past this line is plain
        # `Ma / Ma_TE`. The two are not divided by the same thing: the ramp is
        # referred to the velocity the blade meets and the plateau to the axial
        # component of it, which agree at low swirl and part company as it
        # rises.
        Ma_LE = self.Ma_LE / turbigen.loading.mach_ratio(machine, self.i_row)
        Ma_PS = self.Ma_PS / turbigen.loading.mach_ratio_axial(machine, self.i_row)

        z = np.asarray(z, dtype=float)
        return np.stack(
            (
                turbigen.clark.suction(z[0], self.Ma_peak, self.z_peak, Ma_LE, Ma_PS),
                turbigen.clark.pressure(z[1], self.Ma_peak, self.z_peak, Ma_LE, Ma_PS),
            )
        )

    #
    # A LEVEL AND A SHAPE, WHICH ARE NOT THE SAME KIND OF KNOB
    #

    def _by_knob(self, config, shape_value, level_value, recamber_value):
        """Return a value for `Co`, one for `dchi_LE`, and one for every shape knob."""
        values = {f"Co[{self.i_row}]": level_value, self._dchi_name: recamber_value}
        return values | {name: shape_value for name in self._names(self._order(config))}

    def gains(self, config):
        """Return the gain of each knob, with the level's declared separately.

        A scalar :attr:`gain` describes the *shape* knobs only, and
        :attr:`gain_Co` and :attr:`gain_dchi_LE` supply the level and the
        recamber.

        All three are positive; they are split because of units. A gain is
        knob units per error unit, and `Co` is a circulation coefficient and
        `dchi_LE` an angle where the others are shape-space coefficients, so
        one number spread over them would be different assumed slopes wearing
        one value. :meth:`clips` splits for the same reason and has no
        sequence form to fall back on. Only :meth:`tolerances` splits without
        needing to --- a level residual and a shape residual are both in `fac`
        --- and it splits so that the two can be converged to different
        criteria.

        A sequence :attr:`gain` declares every knob instead --- `Co` at index
        zero and `dchi_LE` at one, as :meth:`unknowns` orders them --- and
        then carries both itself, leaving `gain_Co` and `gain_dchi_LE` unread.
        """
        names = list(self.unknowns(config))
        if isinstance(self.gain, (int, float)):
            return self._by_knob(
                config,
                float(self.gain),
                float(self.gain_Co),
                float(self.gain_dchi_LE),
            )
        return dict(zip(names, self._gain_each(names)))

    def clips(self, config):
        return self._by_knob(config, self.clip, self.clip_Co, self.clip_dchi_LE)

    def tolerances(self, config):
        by_knob = self._by_knob(
            config, self.tolerance, self.tolerance_Co, self.tolerance_dchi_LE
        )
        by_knob[f"tau_LE[{self.i_row}]"] = self.tolerance_tau_LE
        return by_knob

    #
    # KNOBS, AS A FLAT TABLE AND AS A PAIR OF CURVES
    #

    def _names(self, order):
        """Return the table key of each shape knob, in a fixed order.

        `Co` is not among them: it is not a leaf of a thickness distribution,
        and every method walking this to touch a coefficient would otherwise
        have to skip it by hand. :meth:`unknowns` puts it in front.

        The order is load-bearing, because a sequence :attr:`gain` is matched
        to it. The three ends come before the interior so that their index does
        not move when a design changes order, which would otherwise read a nose
        sensitivity back as a mid-chord one.
        """
        return [
            f"tau_LE[{self.i_row}]",
            f"tau_TE[{self.i_row}]",
            *(f"tau[{self.i_row}][{i}][{k}]" for i in (0, 1) for k in range(1, order)),
        ]

    def _flat_coeff(self, c):
        """Return coefficients `(2, order+1)` in :meth:`_names` order.

        Both ends are read off the suction row alone, the two rows being equal
        there by construction --- `with_tau_coeff` refuses either end that is
        not.
        """
        return [c[0][0], c[0][-1], *c[0][1:-1], *c[1][1:-1]]

    def _flat_error(self, e):
        """Return residuals `(2, order+1)` in :meth:`_names` order.

        Where :meth:`_flat_coeff` reads one shared value, this takes the *mean*
        of the two surfaces: one knob serves both, so what it can answer for is
        how wrong they are together, and the mean is the least-squares move for
        it. Both ends null where the surfaces are equally wrong in opposite
        directions, that being the part one knob cannot reach.
        """
        return [
            0.5 * (e[0][0] + e[1][0]),
            0.5 * (e[0][-1] + e[1][-1]),
            *e[0][1:-1],
            *e[1][1:-1],
        ]

    def _unflat(self, values, order):
        """Return a `(2, order+1)` coefficient shift from :meth:`_names` order.

        The inverse of :meth:`_flat_coeff`: both end knobs are written back to
        *both* rows, which is what keeps the surfaces sharing them however far
        the loop moves them.
        """
        shift = np.zeros((2, order + 1))
        shift[:, 0] = values[0]
        shift[:, -1] = values[1]
        shift[0, 1:-1] = values[2 : order + 1]
        shift[1, 1:-1] = values[order + 1 :]
        return shift

    #
    # WHAT THE CONFIG HAS TO PROVIDE
    #

    @property
    def _dchi_name(self):
        """Return the table key of this row's leading-edge recamber."""
        return f"dchi_LE[{self.i_row}]"

    def _dchi_LE(self, config):
        """Return the row's mean leading-edge recamber [deg]."""
        return float(np.mean([s.dchi_LE for s in self._blade(config).sections]))

    def _order(self, config):
        """Return the Bernstein degree of this row's thickness curves."""
        return self._thickness(config).order

    def _thickness(self, config):
        """Return the first section's thickness, which sets the order for all."""
        self._check(config)
        return config.blades[self.i_row].sections[0].thickness

    def _coefficients(self, config):
        """Return the row's mean shape-space coefficients, `(2, order+1)`.

        Averaged over the sections because only one span fraction is measured
        and :meth:`with_unknowns` moves them all together.
        """
        self._check(config)
        return np.mean(
            [
                section.thickness.tau_coeff
                for section in config.blades[self.i_row].sections
            ],
            axis=0,
        )

    def _too_thin(self, sections):
        """Return what is wrong with `sections`, or None if nothing is.

        A shape knob has no bound of its own that keeps an aerofoil closed:
        `clip` limits one step, not where a run of them arrives, and a
        thickness driven negative is a section whose surfaces have crossed. It
        fails in the mesher rather than here, a long way from the step that
        caused it, so it is caught while the step that caused it is still in
        hand.
        """
        m = np.linspace(0.0, 1.0, 201)[1:-1]
        for i_section, section in enumerate(sections):
            for i_surf, t in enumerate(section.thickness.thick_both(m)):
                if np.any(t <= 0.0):
                    surface = ("suction", "pressure")[i_surf]
                    return (
                        f"section {i_section}'s {surface} surface with no "
                        f"thickness at m={m[int(np.argmin(t))]:.2f}"
                    )
        return None

    def _check(self, config):
        """Raise unless this row's thickness can carry the knobs, and its
        recamber is this iterator's alone."""
        from turbigen.thickness import ClarkThickness

        if any(isinstance(it, Incidence) for it in config.iterate.correct):
            raise ValueError(
                f"A clark_profile iterator on row {self.i_row} sets that row's "
                f"leading-edge recamber from its loading, and an incidence "
                f"iterator sets every row's from the stagnation point; the two "
                f"would move the same dchi_LE towards different targets. Drop "
                f"the incidence iterator."
            )

        orders = set()
        for i_section, section in enumerate(self._blade(config).sections):
            thickness = section.thickness
            where = f"row {self.i_row} section {i_section}"

            if not isinstance(thickness, ClarkThickness):
                raise ValueError(
                    f"Shaping a loading profile with thickness needs a "
                    f"distribution that can differ side to side, and {where} "
                    f"has a {type(thickness).__name__}, which is the same "
                    f"both sides of its camber line. Set thickness: "
                    f"{{type: clark, ...}}."
                )

            if not thickness.coeff[0]:
                raise ValueError(
                    f"{where} has no interior thickness coefficients, so there "
                    f"is nothing between its nose and its trailing edge to "
                    f"shape. Write coeff out in full, as two rows of at least "
                    f"one zero apiece."
                )

            orders.add(thickness.order)

        if len(orders) > 1:
            raise ValueError(
                f"Every section of row {self.i_row} must carry the same number "
                f"of thickness coefficients, since they are interpolated over "
                f"the span field by field, got orders {sorted(orders)}."
            )


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
    tolerance: float = 0.02
    """Permissible error, as a fraction of the nominal value."""

    clip: float = 0.2
    """Largest change in one iteration, as a fraction of the nominal value.

    Relative for the reason :attr:`tolerance` is, and to the same nominal:
    design variables have no common scale, so an absolute number would mean
    something different for every one of them. Written absolutely --- which is
    what :attr:`Iterator.clip` means everywhere else, the other iterators
    moving angles in degrees and coefficients that are already dimensionless
    --- a single ``clip: 0.01`` beside a two-row ``Ys`` of ``[0.030, 0.076]``
    permits a third of the first and an eighth of the second, while the
    tolerance a line above it means two per cent of each. Two conventions, two
    adjacent numbers, and a loss coefficient free to walk a third of its value
    per pass.

    Zero for no limit, as it is on the base class.
    """

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
            raise MeasurementError(
                "there is no mixed-out mean line, so the design variables "
                f"{sorted(self.variables)} are unmeasured."
            )

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
        """Return absolute tolerances, scaled from the relative one declared."""
        return self._scaled(config, self.tolerance)

    def clips(self, config):
        """Return absolute clips, scaled from the relative one declared.

        A clip of zero stays zero, which is what the stepper reads as no limit
        --- scaling it by a nominal value would be scaling nothing.
        """
        return self._scaled(config, self.clip)

    def _scaled(self, config, declared):
        """Return `declared` against each variable's own nominal value.

        Design variables have no common scale --- a loss coefficient of 0.05
        sits beside a stage loading of 1.6 --- so one absolute number cannot
        serve them all. A nominal value of zero has nothing to be relative to,
        and falls back to taking the number as absolute.
        """
        merged = {}
        for name in self.variables:
            nominal = self._values(config, name)
            for key, value in zip(self._names(name, len(nominal)), nominal):
                scale = np.abs(value) if np.abs(value) > TINY else 1.0
                merged[key] = declared * float(scale)
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
    kg/m/s is not. But it cannot simply be inverted for ``mu:``: it is measured
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
    targets: ClassVar[tuple[str, ...]] = ("target",)

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
            raise MeasurementError(
                "there is no machine to take a surface Reynolds number off. "
                "This one is measured from the design rather than from a "
                "march, so there is always a machine to read unless something "
                "handed this a result it never designed."
            )

        Re_surf = result.machine.Re_surf()
        if not len(Re_surf):
            raise ValueError(
                "A surface Reynolds number is measured against a blade surface, "
                "so iterating on one needs a blades: section in the config."
            )
        _check_i_row(self.i_row, len(Re_surf))

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


WALL_TRIM = 1
"""Stations dropped at each endwall before an exit profile is fitted.

The outermost face of the cut sits a fraction of a per cent of span from the
wall, where the pitch average reads the wall value itself. One face is enough
to keep that out of the fit; dropping half a per cent of span instead was
measured to change the fit error of the sweep10 profiles by nothing worth
having.
"""


def exit_profile(result, modes, order, offset=None):
    """Return the modal coefficients of the profile leaving `result`.

    Cut the last station, mass-average each quantity over the pitch, subtract
    the mixed-out mean, normalise by that station's own dynamic head and
    dynamic temperature, drop the outermost station at each wall, and fit.

    Normalising by the *exit* station's own scales rather than the inlet's is
    what makes "repeating" mean the shape repeats: a stage raises or drops the
    level, and it is the redistribution about that level which comes round
    again.

    Parameters
    ----------
    result : Result
        A solved run.
    modes : turbigen.bconds.Modes
        The shapes to fit, whose :meth:`~turbigen.bconds.Modes.fit` is weighted
        by span rather than by how the cut clusters its stations.
    order : int
        Number of modes to fit. The constant is fitted and dropped, a profile
        being a redistribution rather than a level.
    offset : float or None
        Cut plane offset in blade chords; `annulus.CUT_OFFSET` by default.

    Returns
    -------
    dict
        One tuple of `order` coefficients per column of :attr:`Repeat.COLUMNS`.

    """
    spf, deficit, _ = exit_deficit(result, offset)

    inner = slice(WALL_TRIM, spf.size - WALL_TRIM)
    return {
        name: modes.fit(name, spf[inner], deficit[name][inner], order)
        for name in Repeat.COLUMNS
    }


def exit_deficit(result, offset=None):
    """Return the pitch-averaged spanwise profile leaving `result`, unfitted.

    The measurement :func:`exit_profile` fits, kept apart so that the sampled
    profile itself can be studied: what order it needs, and where a fit to it
    goes wrong.

    Parameters
    ----------
    result : Result
        A solved run.
    offset : float or None
        Cut plane offset in blade chords; `annulus.CUT_OFFSET` by default.

    Returns
    -------
    spf : (nspan,) array
        Span fraction of each face.
    deficit : dict
        ``DPo``, ``DTo``, ``DAlpha`` and ``DBeta`` at each `spf`, about the
        mixed-out mean, the first two in fractions of the scales below.
    scales : dict
        The mixed-out ``Po``, ``To``, ``Alpha``, ``Beta``, and the dynamic head
        ``q`` and dynamic temperature ``dT`` the deficits are normalised by.

    """
    from turbigen import annulus

    grid, machine = result.grid, result.machine
    xr = machine.annulus.cut_planes(annulus.CUT_OFFSET if offset is None else offset)[
        -1
    ]

    cut = ember.cut.unstructured(grid, xr)
    if cut is None:
        raise ValueError(f"The exit cut plane at {xr.tolist()} misses the grid.")

    # Structured so that the pitch is an axis to average over.
    nj, nk = 137, 113  # Brute force to avoid any loss of resolution
    structured = ember.cut.interpolate_to_structured(cut, (nj, nk))

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

    scales = {
        "Po": float(mean.Po),
        "To": float(mean.To),
        "Alpha": float(mean.Alpha),
        "Beta": float(mean.Beta),
        "q": q,
        "dT": dT,
    }

    return spf, deficit, scales


class Repeat(Iterator):
    """Pass the exit profile back to the inlet, until the stage feeds itself.

    A repeating stage, in the middle of a multistage machine, is fed by its own
    exit, so the inlet profile is a fixed point to find rather than an input.

    **The copy is the existing step rule.** With the error taken as
    ``inlet - outlet``, :func:`step`'s ``u -= gain * e`` at ``gain = 1`` gives
    ``u_new = outlet``; a smaller gain relaxes it.

    **Modal coefficients, not samples**, are passed upstream: few, independent,
    smooth over mesh noise, and independent of the mesh. The modes are chosen
    by :attr:`basis`: Legendre polynomials by default, or POD modes from
    earlier runs, which follow sharp hub and tip features better at low order.

    **The fit is weighted by span**, so wall-clustered cut stations do not get
    extra emphasis.

    ``DBeta`` is not carried: pitch angle at a repeating station is essentially
    zero.
    """

    type: ClassVar[str] = "repeat"

    COLUMNS: ClassVar[tuple[str, ...]] = ("DPo", "DTo", "DAlpha")
    """The profile columns this iterator owns."""

    ANGLES: ClassVar[tuple[str, ...]] = ("DAlpha", "DBeta")
    """Those measured in degrees rather than in fractions of a scale."""

    order: int = 4
    """Number of modes passed upstream, per column."""

    basis: str | None = None
    """Absolute path to a POD basis file, or None for Legendre modes.

    Fixed for a sweep: the coefficients a run writes are in these modes, and a
    database only blends samples whose profiles share them. See
    :func:`turbigen.pod.build_basis` for making one.
    """

    offset: float = 0.5
    """Where to read the exit profile, in blade chords past the trailing edge.

    Far enough that the blade wakes have begun to mix but the plane is still in
    the machine. The package this replaces reads at the same distance.
    """

    gain: float = 0.8
    """One copies the exit profile outright; less under-relaxes it.

    Relaxation only: the loop stops where the error is null, and `gain` scales
    the path to that point rather than moving it. What fraction of the exit
    profile the converged inlet actually carries is
    :attr:`transfer_To` and nothing else.
    """

    transfer_To: float = 0.5
    """Fraction of the exit stagnation temperature profile fed back upstream.

    One is a strictly repeating stage: whatever temperature redistribution
    leaves, arrives. That is the right statement for pressure and for angle,
    which the blade row re-establishes, and it overstates the temperature
    profile, which mixes out between stages instead --- a hot streak that
    survives one stage intact survives every stage, and the loop compounds it.

    Below one the converged inlet carries that fraction of the exit profile,
    because the error is measured against the damped exit rather than the raw
    one. It is a claim about how much interstage mixing there is, so it belongs
    here as a number somebody chooses rather than as a relaxation that would
    slow the loop down and land in the same place anyway.

    Only the temperature is damped. Pressure and angle keep their full
    feedback. One is the loop as it was; the default of a half says the
    stage mixes out half of what reaches it.
    """

    atol_head: float = 0.01
    """Converged when ``DPo`` and ``DTo`` are within this [--].

    In fractions of dynamic head and of dynamic temperature, which is what
    those columns are measured in.
    """

    atol_angle: float = 1.0
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
        if not 0.0 <= self.transfer_To <= 1.0:
            raise ValueError(
                f"repeat.transfer_To must be between 0 and 1, got "
                f"{self.transfer_To}. It is the fraction of the exit "
                f"temperature profile that comes round again; above one the "
                f"loop amplifies its own profile, and below zero it inverts it."
            )
        if self.basis is not None:
            # Read now, so a missing file or a basis with too few modes is a
            # config error rather than a failure after the first march.
            modes = self.modes()
            for name in self.COLUMNS:
                if modes.n_modes(name) < self.order:
                    raise ValueError(
                        f"repeat.order is {self.order}, but the POD basis at "
                        f"{self.basis} has {modes.n_modes(name)} {name} mode(s)."
                    )

    def modes(self):
        """Return the :class:`~turbigen.bconds.Modes` profiles are fitted in."""
        from turbigen import bconds

        if self.basis is None:
            return bconds.LegendreModes()
        return bconds.PodModes(self.basis)

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
        moved = _merged(self.unknowns(config), values)

        columns = {
            name: tuple(
                moved[f"inlet_profile.{name}[{mode}]"] for mode in range(self.order)
            )
            for name in self.COLUMNS
        }

        return dataclasses.replace(
            config, inlet_profile=self.modes().profile(**columns)
        )

    def paths(self, config):
        # The one iterator whose knobs are its leaves, one for one, so the two
        # namings coincide rather than needing translation.
        return set(self.unknowns(config))

    def error(self, config, result):
        _require_grid(result, "the exit profile that feeds the inlet")

        measured = exit_profile(result, self.modes(), self.order, self.offset)
        current = self.unknowns(config)

        # Inlet minus the exit profile this stage is fed by, so that
        # `u -= gain * e` at gain one lands on it exactly. That is the whole
        # exit profile except in temperature, where `transfer_To` says how much
        # of it survives to the next inlet -- and damping it here rather than
        # in the step is what moves the fixed point rather than the path to it.
        transferred = self.transfers()
        return {
            f"inlet_profile.{name}[{mode}]": (
                current[f"inlet_profile.{name}[{mode}]"]
                - transferred[name] * measured[name][mode]
            )
            for name in self.COLUMNS
            for mode in range(self.order)
        }

    def transfers(self):
        """Return the fraction of the exit profile each column feeds back."""
        return {
            name: (self.transfer_To if name == "DTo" else 1.0) for name in self.COLUMNS
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
