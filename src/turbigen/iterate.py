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
import itertools
import logging
from typing import ClassVar

import ember.average
import ember.cut
import numpy as np
from numpy.polynomial import legendre

import turbigen.clark
import turbigen.loading
import turbigen.shapespace
import turbigen.util
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

DU_MIN = 0.25
"""Smallest move, in tolerance-equivalent steps, that may update the Jacobian.

The slope a secant infers has error of order `noise / du`, and the errors here
are measured from a march that is only partly converged --- the same deviation
slope read +1.27 from a 200-step solve and about +0.3 from a 50-step one. A
quarter of a step is where an update stops saying more than the noise does.
"""

DU_PIN = 0.1
"""Below this a knob counts as not having moved, in tolerance-equivalent steps.

Per knob, where :data:`DU_MIN` is per block. A knob its geometry would not let
move --- a nose against its ``R_LE_lim`` bound, a thickness the mesher refuses
--- sits here while the rest of its block strides on. Its secant row is then a
frozen residual measured over someone else's move, which the update learns as
real coupling, and its column is a near-singular direction that amplifies the
block's Newton step. Held out of both until it moves again, it falls back to
the decoupled prior-gain step the same way a singular block does.
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

    gain: float | tuple[float, ...] = 1.0
    """How much of the error to subtract from the unknown.

    Carries the sign of the local sensitivity as well as its size: the step is
    always ``u -= gain * e``, so an iterator whose error *falls* as its knob
    rises declares a negative gain. Reciprocal of an assumed slope, so it is
    the crudest possible Newton step.

    An input only. A run measures the real slopes and they live in the
    Jacobian, which :func:`_jacobian` rebuilds from the history on every call;
    nothing writes one back into a design. A gain written into a config would
    be read against whatever knobs the *next* design has, and a sequence
    measured on one blade means nothing on another --- see :meth:`_gain_each`,
    which refuses the mismatch it can see and cannot see the rest.

    **One number or one per knob.** A scalar is a single declared prior, spread
    over every knob the iterator owns; a sequence carries them separately, in
    the order :meth:`unknowns` returns. A design is declared with the scalar,
    because a prior is a statement about a *kind* of knob and the count is
    rarely known while writing a file; a run measures a slope for each knob and
    writes back the sequence, because those are genuinely different numbers.
    Reducing them to one would blend, say, the sensitivity of a recamber at
    hub and casing, or of a camber coefficient at the front of a blade with one
    at the back --- measurements of different constants, not repeats of one.
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

    learns: ClassVar[bool] = True
    """Whether a run improves this iterator's sensitivities, or keeps them.

    True by default: the declared :attr:`gain` is a guess, and the Broyden
    update spends the run replacing it with what the trajectory measured.

    False where the sensitivity is closer to a definition than to a guess,
    and the loop is better off taking the step the gain asserts than a step
    fitted to a handful of near-parallel moves. A block that does not learn
    keeps `diag(sign(gain))`, and because the scaling folds `|gain|` into
    `u_scale`, the Newton step on it is exactly `u -= gain * e` --- the step
    the first iteration takes, kept for all of them.

    **What it gives up.** Broyden recovers within a run from a gain declared
    with the wrong sign; nothing here does. Only worth setting where the sign
    follows from what the knob *is*.
    """

    def blocks(self, config):
        """Return the Jacobian block each unknown belongs to, by name.

        A block is learned as a unit, so this says what is worth learning
        together. One block per iterator by default, which is the claim that
        an iterator's knobs act on its own errors and not usefully on anyone
        else's.

        Overridden where that is wrong. A blade's recamber and its thickness
        shape the same leading edge, so a run that moves one and measures the
        other is measuring a coupling that exists --- and an iterator boundary
        drawn between them is a claim that it does not.
        """
        return {name: id(self) for name in self.unknowns(config)}

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
        a blade that has gained a section, a :class:`LoadingProfile` whose
        `order` was changed --- and silently reinterpreting those numbers
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
    """
    merged = {}
    for iterator in config.iterate.correct:
        try:
            merged.update(iterator.error(config, result))
        except MeasurementError as err:
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


@dataclasses.dataclass(frozen=True)
class _Table:
    """The flat table the module docstring draws, ready for arithmetic.

    Assembled once by :func:`_assembled` so that everything reading a run ---
    the step it takes and the table it prints --- reads the same names, the
    same scales and the same Jacobian, rather than two nearly identical
    assemblies free to drift apart.
    """

    names: list
    values: dict
    measured: dict
    gain: dict
    clip: dict
    u_scale: np.ndarray
    e_scale: np.ndarray
    prior: np.ndarray
    jacobian: np.ndarray
    blocks: list
    pinned: np.ndarray
    """Per knob: did not move on the last iteration, so it is held out of the
    coupled Newton solve and given the decoupled prior step instead. See
    :data:`DU_PIN`."""


def _assembled(config, result, history):
    """Return the table for this run, or None when there is nothing to work on."""
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

    names = [name for name in values if gain[name] and tolerance[name] > TINY]
    if not names:
        return None

    # Worked in units of each knob's own tolerance, which is the only scale
    # declared for it. Degrees of recamber and a loss coefficient otherwise
    # share one Euclidean norm in the Broyden update, and the update -- being
    # least-change in that norm -- would spend itself entirely on whichever
    # variable happened to carry the larger numbers.
    u_scale = np.array([abs(gain[name]) * tolerance[name] for name in names])
    e_scale = np.array([tolerance[name] for name in names])
    prior = np.array([np.sign(gain[name]) for name in names])

    blocks = _blocks(config, names)
    jacobian = _jacobian(
        names, prior, history, (values, measured), u_scale, e_scale, blocks
    )
    _report_flat(names, jacobian)
    pinned = _pinned(names, values, history, u_scale)

    return _Table(
        names=names,
        values=values,
        measured=measured,
        gain=gain,
        clip=clip,
        u_scale=u_scale,
        e_scale=e_scale,
        prior=prior,
        jacobian=jacobian,
        blocks=blocks,
        pinned=pinned,
    )


def _pinned(names, values, history, u_scale):
    """Return a mask of the knobs that did not move on the last iteration.

    Read from the trajectory rather than reported by the iterators: the signal
    is "was told to move and did not", whatever refused it --- an ``R_LE_lim``
    bound, the meshability guard, a closed section. A knob that has simply
    converged sits here too, and holding one that is already on its target
    costs nothing; it rejoins its block the moment its error moves it again.
    """
    if not history:
        return np.zeros(len(names), dtype=bool)

    was = history[-1][0]
    du = np.array([values[n] - was[n] if n in was else 0.0 for n in names]) / u_scale
    return np.abs(du) < DU_PIN


def _blocks(config, names):
    """Return the indices into `names` each iterator owns, one array apiece.

    The Jacobian is block-diagonal over these, which is the claim that what
    the knobs in a block do to its errors is worth learning and what they do
    to another block's is not. That is not because the cross terms are zero
    --- a row's exit angle plainly sets the next row's inlet angle --- but
    because they were never identifiable from the trajectory a run takes: see
    :func:`step`.

    A block is not an iterator, though it is one by default. Each knob says
    which block it belongs to, so a coupling worth learning can cross an
    iterator boundary without the two iterators having to become one --- see
    :meth:`Iterator.blocks`.
    """
    seen, grouped, learns = {}, {}, {}
    for iterator in config.iterate.correct:
        for name, key in iterator.blocks(config).items():
            if name in seen and name in names:
                raise ValueError(
                    f"{name} is claimed by both {seen[name]} and "
                    f"{type(iterator).__name__}, so it belongs to no one block."
                )
            seen[name] = type(iterator).__name__
            grouped.setdefault(key, set()).add(name)

            # Learning is a property of the block, the update being applied to
            # it whole, so iterators sharing one have to agree. Refused rather
            # than resolved: either rule -- the doubters win, or the learners
            # do -- makes `learns` mean something different depending on what
            # it sits beside, which is worse than a setting that will not load.
            was = learns.setdefault(key, (iterator.learns, type(iterator).__name__))
            if was[0] != iterator.learns:
                raise ValueError(
                    f"{was[1]} and {type(iterator).__name__} share a Jacobian "
                    f"block but disagree on whether it learns "
                    f"({was[0]} against {iterator.learns}). A block is updated "
                    f"whole, so they must agree."
                )

    blocks = []
    for key, own in grouped.items():
        idx = np.array([i for i, name in enumerate(names) if name in own], dtype=int)
        if idx.size:
            blocks.append((idx, learns[key][0]))
    return blocks


def step(config, result, history=()):
    """Return the config to try next, from the errors `result` reports.

    A Newton step on an approximate Jacobian: `B dx = -e`, clipped per key. `B`
    starts as the diagonal the declared gains already assert --- ``u -= gain *
    e`` is a Newton step under exactly that assumption, sign included --- and
    is improved by a rank-one Broyden update for each move the run has already
    paid for. **With no history the step is arithmetically identical to
    ``u -= gain * e``**, so a first iteration is never worse than it was.

    What that buys is the off-diagonal terms *within an iterator*: the Jacobian
    is block-diagonal over :func:`_blocks`, one block per iterator, and each
    block is solved and bounded on its own.

    **A knob the geometry pinned last iteration is held out of its block.** A
    nose against its ``R_LE_lim`` bound moves nowhere however hard it is
    pushed, so in the block Newton solve it is a near-singular direction that
    amplifies the coupled knobs' step --- which is how one clamped nose came
    to swing a blade count by eight per cent. Held out, it takes the decoupled
    prior-gain step and rejoins its block the first iteration it moves again.
    See :data:`DU_PIN`.

    **The cross-iterator terms are deliberately not learned**, though they are
    plainly not zero --- the exit angle of a row sets the inlet angle of the
    next, so a deviation correction moves the following row's incidence. They
    are left at the prior because they were never identifiable from the
    trajectory a run takes. A twenty-iteration run offers at most nineteen
    rank-one updates against a full matrix of some hundreds of entries, each
    constraining it along one direction only, and those directions are strongly
    correlated because every one of them is the Newton step for a residual that
    moves slowly. Worse, every knob moves on every iteration, so a single
    `(du, de)` pair cannot say which knob caused which part of the error
    change, and the update spreads the credit along `du` regardless. Fitting
    that data to a block of ten is a question the run can answer; fitting it to
    a matrix of six hundred is one it cannot, and an unidentifiable term
    learned anyway is worse than a term left honestly at its prior.

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
    table = _assembled(config, result, history)
    if table is None:
        logger.debug("Nothing measured to correct towards, so nothing moves.")
        return config

    names, values, _gain, clip = table.names, table.values, table.gain, table.clip

    error = np.array([table.measured[n] for n in names]) / table.e_scale

    # The clip is the trust bound, and the reason a flat response degrades to
    # the old behaviour rather than to a wild excursion.
    limit = np.array(
        [
            clip[name] / scale if clip[name] else np.inf
            for name, scale in zip(names, table.u_scale)
        ]
    )

    # Solved and bounded a block at a time, both for the same reason: a block
    # is meant to be answerable for itself. Solving the assembled
    # block-diagonal in one go is the same arithmetic but hands every knob to
    # the worst-conditioned block, since `_newton` reads one condition number
    # and falls back for everything; bounding it in one go lets a block sitting
    # on its clip shrink the step of a block that is nowhere near its own.
    change = np.zeros(len(names))
    for idx, _ in table.blocks:
        # A knob its geometry would not let move last time is held out of the
        # coupled solve --- in the block it is a near-singular direction that
        # amplifies everyone else's step --- and falls back to the decoupled
        # prior-gain step, which the clip bounds. It rejoins the block the
        # first iteration it moves. All of a block pinned is a singular block,
        # which is the case `_newton` already degrades to the prior for.
        held = table.pinned[idx]
        active, stuck = idx[~held], idx[held]

        if active.size:
            step_now = _newton(
                table.jacobian[np.ix_(active, active)],
                error[active],
                table.prior[active],
            )
            change[active] = _bounded(step_now, limit[active])

        if stuck.size:
            step_prior = -table.prior[stuck] * error[stuck]
            change[stuck] = np.clip(step_prior, -limit[stuck], limit[stuck])

    moved = {
        name: values[name] + change[i] * table.u_scale[i]
        for i, name in enumerate(names)
    }

    for iterator in config.iterate.correct:
        mine = {
            name: moved[name] for name in iterator.unknowns(config) if name in moved
        }
        if mine:
            config = iterator.with_unknowns(config, mine)

    return config


def _jacobian(names, prior, history, current, u_scale, e_scale, blocks):
    """Return the scaled Jacobian, from the prior and every informative move.

    Block-diagonal over `blocks`, and each block carries its own secant
    condition ``B_k du_k = de_k`` --- the rank-one update is applied to the
    restricted vectors rather than applied whole and then masked, which would
    leave a matrix satisfying nothing in particular.

    **Both guards are per block, and that is the point of them being here.**
    A move is worth learning from when the knobs doing the learning moved, not
    when something elsewhere in the table did: a shared `DU_MIN` lets a block
    that barely moved update itself off another block's stride, which is how
    a Jacobian learns noise.

    Rebuilt from the trajectory on every call rather than carried between
    calls, so that `step` keeps no state and can be reasoned about one call at
    a time. It costs a few matrix operations on a handful of numbers.
    """
    jacobian = np.diag(prior)

    trajectory = list(history) + [current]
    for (values, errs), (values_next, errs_next) in itertools.pairwise(trajectory):
        du_all = np.array([values_next[n] - values[n] for n in names]) / u_scale
        de_all = np.array([errs_next[n] - errs[n] for n in names]) / e_scale

        for idx, learns in blocks:
            if not learns:
                continue

            du, de = du_all[idx], de_all[idx]

            if np.sqrt(du @ du) < DU_MIN:
                logger.debug("A move too small to learn from, so the block stands.")
                continue

            # Only the knobs that actually moved carry secant information. One
            # the geometry pinned has du ~ 0 while its block moved; its row is
            # a frozen residual over someone else's stride, and learning it is
            # learning noise as coupling. Restricted to the movers, the update
            # is the same arithmetic on the sub-block they span.
            moved = np.abs(du) >= DU_PIN
            sub = idx[moved]
            du_m, de_m = du[moved], de[moved]

            length = float(du_m @ du_m)
            if length == 0.0:
                continue

            block = jacobian[np.ix_(sub, sub)]
            jacobian[np.ix_(sub, sub)] = (
                block + np.outer(de_m - block @ du_m, du_m) / length
            )

    return jacobian


def _bounded(change, limit):
    """Return `change` scaled down until every knob is inside its limit.

    **Scaled, not clipped per knob.** Clipping each component on its own is the
    obvious reading of "no knob moves more than its clip", and it silently
    throws away the thing the Jacobian was solved for: a step over the limit in
    two knobs at once gets projected onto a *corner* of the box, so what
    survives is the sign pattern of the direction and none of its shape.

    That is not a slow step, it is a different one, and it cycles. Two runs of
    a two-knob loading iterator sat in a period-2 orbit doing it --- steps of
    exactly ``+(0.1, -0.1)`` and ``-(0.1, +0.1)`` alternating, each overshoot
    provoking the opposite corner and landing back where it had been two
    iterations before. Broyden could not break the orbit either, every move
    being collinear with the last, so it only ever learned along that one
    diagonal.

    Scaling keeps the direction and shortens the step, which is what a trust
    region is. The knob that binds still moves exactly its clip, so what the
    setting promises is unchanged; the others move less than they asked for,
    which is the price of going the right way.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        excess = np.max(np.where(np.isfinite(limit), np.abs(change) / limit, 0.0))

    if not np.isfinite(excess) or excess <= 1.0:
        return change

    return change / excess


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
    if not inner.iterate.correct:
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
        #
        # **The gains it carries are the ones it was declared with.** A run
        # measures slopes and they live in the Jacobian, which `_jacobian`
        # rebuilds from the history on every call --- so every direction the
        # run has already moved in is governed by the Broyden estimate rather
        # than by the declared gain, and writing a measured slope back into the
        # design would be a diagonal rescaling a Newton step is invariant to.
        # What it would not be invariant to is everything downstream: a gain
        # written into a config is read back against whatever knobs the *next*
        # design has, and a sequence measured on one blade means nothing on
        # another. So a gain is a declared prior and stays one.
        # `iterate` put back whole: `stepping` is a view holding only the
        # solution iterators, so the config `step` hands back has lost the
        # design-only ones. They are unchanged either way, gains included.
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


def _AEROFOIL_BLOCK(i_row):
    """Return the Jacobian block that shapes row `i_row`'s aerofoil.

    What :class:`Incidence`, :class:`Deviation` and :class:`ClarkProfile` all
    answer with for the same row, so their knobs are learned as one.
    """
    return ("aerofoil", int(i_row))


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

    gain: float = 0.5
    clip: float = 1.0
    """Largest recamber in one iteration [deg].

    One degree, not two: this knob shares a Jacobian block with the leading
    edge and the thickness, and a learned block that has picked up a bad
    sensitivity --- most often from a nose held against its ``R_LE_lim`` bound
    --- discharges it as a single large recamber that moves the blade count
    and throws the solution off. A tighter clip bounds that excursion to
    something the next iteration can walk back.
    """
    tolerance: float = 1.0
    """Permissible error on exit flow angle [deg]."""

    def unknowns(self, config):
        return _recamber_unknowns(config, "dchi_TE")

    def blocks(self, config):
        """Return the aerofoil block each recamber belongs to.

        A row's recamber and its thickness distribution shape the same
        aerofoil: moving the leading edge round changes the incidence, which
        is most of what the front thickness coefficients are for, and moving
        the trailing edge changes the exit angle the rear ones are shaped
        against. Learned together, that coupling is a term in the Jacobian;
        learned apart, it is two iterators pushing on one nose and reading
        each other's work as noise.

        Keyed by row, because that is as far as the coupling reaches --- one
        row's aerofoil is not the next one's.
        """
        return {
            name: _AEROFOIL_BLOCK(i_row)
            for i_row, name in enumerate(_recamber_unknowns(config, "dchi_TE"))
        }

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
    """

    type: ClassVar[str] = "incidence"

    target: float = 0.0
    """Incidence to aim for [deg]."""

    # Negative because the metal angle rises with the recamber while the
    # incidence is measured against it, so the error falls as the knob rises.
    #
    # Deliberately timid. A swept angle responds more steeply to recamber than
    # the flow angle this used to measure --- the stagnation point moves
    # several degrees around the nose for one degree of metal --- and a row
    # takes much of its incidence from the row upstream, so a first step is
    # taken before anything has been learned about either. The secant recovers
    # the rate from the second iteration on, so being cautious here costs a
    # gentle opening step rather than a slow iteration.
    gain: float = -0.05
    clip: float = 1.0
    tolerance: float = 5.0
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

    def blocks(self, config):
        """Return the aerofoil block each recamber belongs to.

        A row's recamber and its thickness distribution shape the same
        aerofoil: moving the leading edge round changes the incidence, which
        is most of what the front thickness coefficients are for, and moving
        the trailing edge changes the exit angle the rear ones are shaped
        against. Learned together, that coupling is a term in the Jacobian;
        learned apart, it is two iterators pushing on one nose and reading
        each other's work as noise.

        Keyed by row, because that is as far as the coupling reaches --- one
        row's aerofoil is not the next one's.
        """
        return {
            f"dchi_LE[{i_row}][{i_section}]": _AEROFOIL_BLOCK(i_row)
            for i_row, i_section, _ in _sections(config)
        }

    def with_unknowns(self, config, values):
        blades = list(config.blades)

        for i_row, blade in enumerate(blades):
            sections = list(blade.sections)
            moved = False
            for i_section, section in enumerate(sections):
                name = f"dchi_LE[{i_row}][{i_section}]"
                if name not in values:
                    continue
                sections[i_section] = dataclasses.replace(section, dchi_LE=values[name])
                moved = True
            if moved:
                blades[i_row] = dataclasses.replace(blade, sections=tuple(sections))

        return dataclasses.replace(config, blades=tuple(blades))

    def paths(self, config):
        return {
            f"blades[{i_row}].sections[{i_section}].dchi_LE"
            for i_row, i_section, _ in _sections(config)
        }

    def error(self, config, result):
        if result.grid is None or result.machine is None:
            raise MeasurementError(
                "there is no solved grid to cut, so the incidence onto every "
                "row is unmeasured."
            )

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
            measured[f"dchi_LE[{i_row}][{i_section}]"] = incidence - self.target

        return measured


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

N_COEFF = 1
"""Interior Bernstein coefficients a loading distribution moves.

One, because :class:`LoadingDistribution` chases one number: the Mach number
at `zeta_front`. A camber line with its ends pinned needs one interior degree
of freedom to move one point on the curve, and no more --- a second knob would
have nothing of its own left to null, since `PeakMach` already owns the level.
"""

BERNSTEIN_ORDER = N_COEFF + 1
"""Bernstein order carrying exactly :data:`N_COEFF` interior coefficients."""


class LoadingDistribution(Iterator):
    """Shape the leading-edge acceleration by moving the camber line.

    :class:`Deviation` and :class:`Incidence` correct the *ends* of a blade
    against the flow. This corrects one point in between: how hard the
    leading edge accelerates by `zeta_front`, which is what a designer chooses
    when they pick an aerodynamic style, and which Clark (2019) shows is a
    useful number to hold rather than let the mean-line design produce
    whatever it produces.

    The knob is the single interior coefficient of a
    :class:`~turbigen.camber.Bernstein` camber line --- see :data:`N_COEFF`.
    Its endpoint counterparts are pinned at zero, so the metal angles do not
    move and this iterator cannot fight the two that own them.

    Read straight off the suction surface at `zeta_front` rather than off a
    fit, so this needs no peak to exist and asks nothing of where one sits or
    how high it stands. `PeakMach` sits beside it and owns the level; between
    the two, neither where the peak is nor how it is reached is a target ---
    only the front acceleration and the overall diffusion are.

    One row per iterator, like :class:`SurfaceReynolds`: a stator and a rotor
    want different loading, so two rows means two entries.
    """

    type: ClassVar[str] = "loading"

    i_row: int = 0
    """Index of the blade row to shape."""

    fac_front: float = 1.8
    """Target leading-edge Mach number, normalised by duty [--].

    ``Ma(zeta_front) / Ma_TE * Ma_2 / Ma_1``, which is Clark (2019) parameter
    3. Referred to the trailing edge rather than to the peak because `Ma_TE` is
    a mean-line quantity, fixed by the duty, where the peak is a fitted one
    that moves with the loading --- and the ``Ma_2 / Ma_1`` factor is what
    makes the same number mean the same style of leading edge across rows of
    different duty, which is the whole reason the parameter is written this
    way.

    Typically greater than one on a turbine, the surface being faster at a
    tenth of its length than the mean line is at exit.
    """

    zeta_front: float = 0.2
    """Front anchor, and where the Mach number is read off [--].

    It is the boundary between what the leading edge decides and what the
    camber line does, so it says *where to measure* rather than what to want
    --- and below it the distribution belongs to :class:`Incidence` and the
    thickness, neither of which this moves.

    A fifth rather than a tenth, because a point that close to the nose still
    sits inside the sharp acceleration round it, which is not the camber
    line's to answer for.

    :class:`PeakMach` carries the same setting for its own fitted window, and
    the two should agree so that a design's front and peak describe the same
    curve.
    """

    spf: float = 0.5
    """Span fraction to measure the distribution at [--]."""

    gain: float = -0.5
    """How much of the error to subtract.

    **A starting direction, not a calibration.** Measured on two different
    cascades, the sign of the response came out opposite --- a first blade
    gave ``d(fac_front)/dc = -0.80`` while the blade in
    `examples/turbine_cascade_loading.yaml` gave ``+0.15`` for the
    corresponding coefficient. Which way the one Bernstein bump moves the
    front value is evidently a property of the blade rather than of the
    parametrisation, so there is no scalar here that is right in general, and
    one confident enough to matter would be wrong half the time.

    Small, accordingly. Two things follow from that and both are wanted: a
    first step taken on a wrong sign costs one iteration rather than an
    excursion, and the step it asks for sits *inside* :attr:`clip`.

    The Broyden update is what actually steers this iterator. It only has to
    avoid getting in its way.
    """

    clip: float = 0.1
    """Largest change in the coefficient per iteration [--].

    Measured against how far it actually travels. On the cascade in
    `examples/turbine_cascade_loading.yaml` it converged at about -0.23 from a
    start of zero, and at a clip of 0.05 the step saturated for five
    iterations running --- which is not merely slow: a step saturated at the
    clip keeps only the *sign* of what the Jacobian asked for.
    """

    tolerance: float = 0.05
    """Converged when the front Mach number is within this [--].

    Around 1.9 on the cascade measured here, so this is a few per cent of what
    it measures.
    """

    def __post_init__(self):
        if not 0.0 < self.zeta_front <= 1.0:
            raise ValueError(f"zeta_front must be in (0, 1], got {self.zeta_front}.")
        if not self.fac_front > 0.0:
            raise ValueError(
                f"fac_front must be positive, got {self.fac_front}. It is a "
                f"Mach number over a Mach number, and on a turbine it is "
                f"usually greater than one."
            )

    #
    # THE PROTOCOL
    #

    def names(self):
        """Return the table key of the knob, as a one-element list.

        Carrying the row, so that two entries shaping two rows cannot collide
        in `unknowns`. A list rather than a bare name because `unknowns` and
        `paths` are written generically over :data:`N_COEFF`.
        """
        return [f"camber_coeff[{self.i_row}][{j}]" for j in range(N_COEFF)]

    def unknowns(self, config):
        coefficients = self._coefficients(config)
        return {
            name: float(np.mean(coefficients[:, j]))
            for j, name in enumerate(self.names())
        }

    def with_unknowns(self, config, values):
        current = self.unknowns(config)
        moved = {**current, **{k: v for k, v in values.items() if k in current}}

        blades = list(config.blades)
        blade = blades[self.i_row]

        shift = np.array([moved[name] - current[name] for name in self.names()])

        # A uniform shift, as `_with_recamber` applies one: whatever spanwise
        # variation of the loading shape the design asked for survives being
        # iterated, because only one span fraction was ever measured and this
        # has nothing to say about the others.
        sections = tuple(
            dataclasses.replace(
                section,
                camber=dataclasses.replace(
                    section.camber,
                    coeff=tuple(np.asarray(section.camber.coeff) + shift),
                ),
            )
            for section in blade.sections
        )

        blades[self.i_row] = dataclasses.replace(blade, sections=sections)
        return dataclasses.replace(config, blades=tuple(blades))

    def paths(self, config):
        return {
            f"blades[{self.i_row}].sections[{i_section}].camber.coeff[{j}]"
            for i_section in range(len(config.blades[self.i_row].sections))
            for j in range(N_COEFF)
        }

    def error(self, config, result):
        if result.grid is None or result.machine is None:
            raise MeasurementError(
                f"there is no solved grid to cut, so the loading distribution "
                f"of row {self.i_row} is unmeasured."
            )

        self._check(config)

        measured = turbigen.loading.measure(
            result, self.i_row, self.spf, self.zeta_front
        )
        if measured is None or not np.isfinite(measured.fac_front):
            raise MeasurementError(
                f"no suction surface could be read on row {self.i_row} at "
                f"spf={self.spf:.2f}, so its front acceleration is unmeasured. "
                f"A section above a clearance gap has no blade to cut: either "
                f"the march is not a flow field, or the section belongs below "
                f"the gap."
            )

        return dict(zip(self.names(), (measured.fac_front - self.fac_front,)))

    #
    # WHAT THE CONFIG HAS TO PROVIDE
    #

    def _coefficients(self, config):
        """Return every section's interior coefficients, ``(n_section, N_COEFF)``."""
        self._check(config)
        return np.array(
            [section.camber.coeff for section in config.blades[self.i_row].sections],
            dtype=float,
        )

    def _check(self, config):
        """Raise unless this row's camber lines can carry the knob."""
        from turbigen.camber import Bernstein

        if not 0 <= self.i_row < len(config.blades):
            raise ValueError(
                f"i_row={self.i_row} is out of range for a machine with "
                f"{len(config.blades)} blade row(s)."
            )

        for i_section, section in enumerate(config.blades[self.i_row].sections):
            camber = section.camber
            where = f"row {self.i_row} section {i_section}"

            if not isinstance(camber, Bernstein):
                raise ValueError(
                    f"Shaping a loading distribution moves the interior "
                    f"coefficient of a Bernstein camber line, and {where} has "
                    f"a {type(camber).__name__} camber, which has none. Set "
                    f"camber: {{type: bernstein, order: {BERNSTEIN_ORDER}, "
                    f"coeff: [0.0]}}."
                )

            if camber.order != BERNSTEIN_ORDER or len(camber.coeff) != N_COEFF:
                raise ValueError(
                    f"Shaping a loading distribution needs exactly {N_COEFF} "
                    f"interior camber coefficient, so order must be "
                    f"{BERNSTEIN_ORDER} and coeff must be given in full; "
                    f"{where} has order={camber.order} with "
                    f"{len(camber.coeff)} coefficient(s). One is what a camber "
                    f"line has to give to move a single front value, and the "
                    f"coefficient is written out rather than zero-padded so "
                    f"that the one this moves is a leaf of the config."
                )


def _circulation_count(config, i_row):
    """Return row `i_row`'s blade count design, which has to have a `Co`.

    Shared by every iterator that moves the blade count -- :class:`PeakMach`
    and :class:`LoadingProfile` both do, and neither owns the other's
    validation.
    """
    from turbigen.blade import Circulation

    if not 0 <= i_row < len(config.blades):
        raise ValueError(
            f"i_row={i_row} is out of range for a machine with "
            f"{len(config.blades)} blade row(s)."
        )

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
    _circulation_count(config, i_row)
    blades = list(config.blades)
    blades[i_row] = dataclasses.replace(
        blades[i_row], count=dataclasses.replace(blades[i_row].count, Co=Co)
    )
    return dataclasses.replace(config, blades=tuple(blades))


class PeakMach(Iterator):
    """Set the level of the loading by moving the blade count.

    The companion to :class:`LoadingDistribution`, which shapes a distribution
    but cannot say how high it stands. At a fixed duty the area enclosed by the
    isentropic Mach loop is the blade circulation, which the pitch sets and a
    camber line only redistributes --- so the level belongs to the blade count,
    and this is the iterator that owns it.

    **Its own member rather than a third knob on the other one, because a gain
    is per-iterator.** :attr:`Iterator.gain` carries the sign of a knob's
    sensitivity as well as its size, and one scalar cannot carry two signs: the
    peak rises with the circulation coefficient while the shape targets fall
    with the camber coefficients. Folded together, a single negative gain drove
    this knob the wrong way at every iteration --- `Co` walked from 0.70 to
    0.57 while the peak it was meant to raise fell with it. Split, each member
    declares the sign it has.

    Moving blade count is what kept `DiffusionFactor` from being ported, on the
    grounds that it changes the mesh. It does: the mesher sizes the grid from
    the pitch, so `Co` of 0.70 and 0.75 meshed at 225 and 209 streamwise nodes
    on the example cascade. That puts a floor under :attr:`tolerance`, since
    remeshing moves the measurement a little for reasons that are
    discretisation rather than flow, but it does not prevent the loop --- every
    iteration remeshes anyway, and restarts interpolate in index space. The
    integer blade count is the smaller worry it looks: one blade is 0.36 per
    cent of `Co` on that cascade, far finer than any step taken, though it
    scales as ``1 / n_blade`` and would bite on a row with forty.
    """

    type: ClassVar[str] = "peak_Ma"

    i_row: int = 0
    """Index of the blade row whose loading level is set."""

    fac_peak: float = 1.2
    """Target peak Mach number over the trailing edge value [--].

    One more than the diffusion factor
    :class:`turbigen.metric.DiffusionFactor` records, so a target here is a
    statement about diffusion in the units a designer already reads.
    """

    spf: float = 0.5
    """Span fraction to measure the distribution at [--]."""

    zeta_front: float = 0.2
    """Front anchor of the window fitted [--].

    The peak is read from a fit rather than from a maximum of the data, so it
    depends on the window fitted. Must match the
    :class:`LoadingDistribution` alongside, or the two describe different
    curves.
    """

    zeta_TE: float = 0.98
    """Far end of the window fitted [--].

    `LoadingDistribution` reads its own target straight off the surface
    rather than from a fit, so it carries no matching setting of its own; only
    `zeta_front` needs to agree between the two.
    """

    gain: float = 1.5
    """How much of the error to subtract [--].

    **Positive, and measured rather than guessed.** The peak rises with the
    circulation coefficient --- more circulation per blade is a bigger loop ---
    at a slope of +0.50 across a sweep of `Co` from 0.6 to 0.8, +0.58 within a
    single run, and +0.58 again from replaying that run's Jacobian. A Newton
    step on the diagonal would be about +1.7, and this sits a little under it
    so that it undershoots rather than overshoots.

    Unlike :attr:`LoadingDistribution.gain`, which is a weak prior because the
    camber sensitivities changed sign between two cascades, this is a
    calibration: the sign follows from what a circulation coefficient *is*, and
    all three measurements agree on the size.
    """

    clip: float = 0.05
    """Largest change in the circulation coefficient per iteration [--]."""

    tolerance: float = 0.02
    """Converged when the peak Mach ratio is within this [--]."""

    def __post_init__(self):
        if not self.fac_peak > 0.0:
            raise ValueError(
                f"fac_peak must be positive, got {self.fac_peak}. It is the "
                f"peak Mach number over the trailing edge one, so on a turbine "
                f"it is greater than one."
            )
        if not 0.0 < self.zeta_front < self.zeta_TE <= 1.0:
            raise ValueError(
                f"A loading level needs 0 < zeta_front < zeta_TE <= 1, got "
                f"{self.zeta_front} and {self.zeta_TE}. The peak is fitted "
                f"inside that window."
            )

    def unknowns(self, config):
        return {f"Co[{self.i_row}]": float(_circulation_count(config, self.i_row).Co)}

    def with_unknowns(self, config, values):
        name = f"Co[{self.i_row}]"
        if name not in values:
            return config
        return _with_circulation(config, self.i_row, values[name])

    def paths(self, config):
        del config
        return {f"blades[{self.i_row}].count.Co"}

    def error(self, config, result):
        if result.grid is None or result.machine is None:
            raise MeasurementError(
                f"there is no solved grid to cut, so the loading level of row "
                f"{self.i_row} is unmeasured."
            )

        _circulation_count(config, self.i_row)

        measured = turbigen.loading.measure(
            result, self.i_row, self.spf, self.zeta_front, self.zeta_TE
        )
        if measured is None or not np.isfinite(measured.fac_peak):
            raise MeasurementError(
                f"no suction peak could be fitted on row {self.i_row} at "
                f"spf={self.spf:.2f}, so its loading level is unmeasured. A "
                f"section above a clearance gap has no blade to cut: either "
                f"the march is not a flow field, or the section belongs below "
                f"the gap."
            )

        return {f"Co[{self.i_row}]": measured.fac_peak - self.fac_peak}


def _target_fac(zeta, zeta_front, fac_front, zeta_peak, fac_peak, mach_ratio):
    """Return a two-line target, in the units :func:`turbigen.loading.measure_profile` reports.

    Front anchor to peak, and peak to the trailing edge -- the same shape
    `turbigen.util.loading_target` draws for a report, but built directly in
    `fac` units rather than absolute Mach numbers, since that is what
    :class:`LoadingProfile` compares its samples against. The trailing edge
    anchor is `mach_ratio` itself, not one: `Ma(1) / Ma_TE * mach_ratio` is
    `mach_ratio` by definition, whatever the duty.

    **The two anchors are not written in the same units.** `fac_front` carries
    the `Ma_2 / Ma_1` factor Clark's third parameter is defined with, and so is
    already in the units measured; `fac_peak` is plain `Ma_peak / Ma_TE`, the
    way `PeakMach` and `turbigen.metric.DiffusionFactor` state a peak, and is
    multiplied by `mach_ratio` here to reach them. Each end of the curve is
    written the way a designer already reads it, and the conversion happens
    once, here.
    """
    zeta = np.asarray(zeta, dtype=float)
    peak = fac_peak * mach_ratio
    front = fac_front + (peak - fac_front) * (zeta - zeta_front) / (
        zeta_peak - zeta_front
    )
    aft = peak + (mach_ratio - peak) * (zeta - zeta_peak) / (1.0 - zeta_peak)
    return np.where(zeta < zeta_peak, front, aft)


class LoadingProfile(Iterator):
    """Shape a whole suction-surface Mach distribution against a two-line template.

    :class:`LoadingDistribution` moves one point on the curve with one
    coefficient. This moves several at once: a higher-order
    :class:`~turbigen.camber.Bernstein` camber line gives `order - 1` interior
    coefficients, each with a characteristic position ``m = (j + 1) / order``
    on the camber line, and each is driven toward a target built from
    `zeta_front`/`fac_front` and `zeta_peak`/`fac_peak` -- two anchors and a
    straight line each side of the peak, read off wherever that coefficient's
    `m` actually lands on the *measured* surface, not at some fixed fraction
    of it, because the two are not the same fraction of the way along the
    chord. See :meth:`~turbigen.blade.Blade.evaluate_arc_length`.

    **This owns the level itself, rather than wanting a `PeakMach` beside
    it.** A camber line still cannot create circulation, only redistribute
    it -- see :class:`PeakMach` -- so the level is not free to be ignored
    here either. But with the whole curve sampled rather than one point,
    the level does not need a second iterator and a second fitted number to
    find it: it is the *mean* of every sampled point's error against the
    target, and what is left after subtracting that mean out of each one is
    the shape residual, blind to the level by construction. The mean drives
    `Co`, exactly as :class:`PeakMach` would; the residuals drive the camber
    coefficients. One iterator, one internally consistent target curve, and
    nothing that two separately-configured iterators could disagree about.

    That needs two *priors*, not one: the level rises with `Co` while the shape
    residuals fall with the camber coefficients, the same disagreement that
    made `PeakMach` a member of its own rather than a third knob on
    `LoadingDistribution`. A single declared number cannot carry both signs, so
    :attr:`gain_Co` is written beside :attr:`gain` --- see :meth:`gains`.

    **`fac_peak` here means what `PeakMach.fac_peak` means:** plain
    `Ma_peak / Ma_TE`, one more than the diffusion factor
    `turbigen.metric.DiffusionFactor` records, carrying no `Ma_2 / Ma_1`
    factor. `fac_front` does carry one, because Clark's parameter 3 is
    specifically a statement about the *front*, so the two anchors are written
    in different units and reconciled in one place --- see
    :func:`_target_fac`. The alternative, one consistent unit across both
    anchors, made the same physical peak read as two different numbers
    depending on which iterator asked for it, which is exactly what lets a
    report contradict the design it describes.

    Only points measured beyond `zeta_front` are driven, exactly as
    `LoadingDistribution` only drives one: below it the distribution belongs
    to `Incidence` and the thickness. A coefficient whose `m` maps inside
    that window is reported with an error of exactly zero rather than
    omitted, so it never moves but also never blocks convergence forever ---
    `converged` treats a knob no iterator ever measured as proof of nothing,
    which a coefficient excluded on purpose is not. Still worth noticing
    before choosing an `order` high enough to pack one in there: a knob held
    this way carries whatever value it started with, unexamined, for the rest
    of the run.
    """

    type: ClassVar[str] = "loading_profile"

    i_row: int = 0
    """Index of the blade row to shape."""

    spf: float = 0.5
    """Span fraction to measure the distribution at [--]."""

    order: int = 3
    """Bernstein order of the camber line; `order - 1` interior coefficients,
    one knob apiece."""

    zeta_front: float = 0.2
    """Front anchor, and the start of the driven window [--].

    Below it the distribution belongs to the leading edge, not the camber
    line -- see :attr:`LoadingDistribution.zeta_front`, which this means the
    same way.
    """

    fac_front: float = 1.8
    """Target leading-edge Mach number, normalised by duty [--].

    Written the same way :attr:`LoadingDistribution.fac_front` is --- see
    there for what the `Ma_2 / Ma_1` factor is for.
    """

    zeta_peak: float = 0.5
    """Target surface fraction of the peak [--]."""

    fac_peak: float = 1.2
    """Target peak Mach number over the trailing edge value [--].

    `Ma_peak / Ma_TE`, exactly as :attr:`PeakMach.fac_peak` states it, and one
    more than the diffusion factor. **Carries no `Ma_2 / Ma_1` factor, unlike
    :attr:`fac_front`** --- see the class docstring.
    """

    gain: float | tuple[float, ...] = -0.5
    """How much of the error to subtract from each knob.

    A starting direction, not a calibration --- see
    :attr:`LoadingDistribution.gain`, which the same caveat applies to.

    **As a scalar this describes the camber coefficients only**, with
    :attr:`gain_Co` carrying the level beside it; the two disagree on sign, so
    one number cannot be both. As a sequence it carries every knob, `Co` first
    and then one per coefficient, for a design that wants to state each
    separately. See :meth:`unknowns` for why `Co` leads.
    """

    clip: float = 0.1
    """Largest change in one camber coefficient per iteration [--]."""

    tolerance: float = 0.05
    """Converged when every driven point's shape residual is within this [--]."""

    gain_Co: float = 1.5
    """How much of the level error to subtract from `Co`, as a prior [--].

    Positive, for the reason :attr:`PeakMach.gain` is: the level rises with
    the circulation coefficient, and that sign is a calibration rather than a
    guess. That is the whole reason this is written apart from :attr:`gain`
    rather than being its first element: a scalar prior cannot carry two signs,
    and the level's is known.

    **Read only while :attr:`gain` is a scalar.** Declared as a sequence,
    `gain` carries every knob including this one, and what is written here no
    longer reaches the loop.
    """

    clip_Co: float = 0.05
    """Largest change in the circulation coefficient per iteration [--]."""

    tolerance_Co: float = 0.02
    """Converged when the mean level error is within this [--]."""

    def __post_init__(self):
        if self.order < 2:
            raise ValueError(
                f"order must be at least 2, got {self.order}. A Bernstein "
                f"camber line needs at least one interior coefficient to move."
            )
        if not 0.0 < self.zeta_front < self.zeta_peak < 1.0:
            raise ValueError(
                f"A loading profile needs 0 < zeta_front < zeta_peak < 1, "
                f"got zeta_front={self.zeta_front}, zeta_peak={self.zeta_peak}."
            )
        if not self.fac_front > 0.0:
            raise ValueError(f"fac_front must be positive, got {self.fac_front}.")
        if not self.fac_peak > 0.0:
            raise ValueError(f"fac_peak must be positive, got {self.fac_peak}.")
        if not isinstance(self.gain, (int, float)) and len(self.gain) != self.order:
            raise ValueError(
                f"A loading profile of order {self.order} has {self.order} "
                f"knobs --- the level and {self.order - 1} camber "
                f"coefficient(s) --- but was given {len(self.gain)} gain(s). "
                f"A sequence is one per knob with Co first; write a single "
                f"number to declare one prior for the camber and let gain_Co "
                f"carry the level."
            )

    #
    # THE PROTOCOL
    #

    def names(self):
        """Return the table key of each camber knob, in a fixed order.

        `Co[i_row]` is not among them: it is not a leaf of a Bernstein camber
        line, and every method below that walks `names()` to touch camber
        coefficients would otherwise have to skip it by hand.
        """
        return [f"camber_coeff[{self.i_row}][{j}]" for j in range(self.order - 1)]

    def unknowns(self, config):
        """Return the level first, then one camber coefficient per knob.

        **`Co` leads, and the order is load-bearing.** A sequence :attr:`gain`
        is matched to this order, so where `Co` sits decides which element of a
        calibration belongs to it. Putting it first fixes that at index zero
        whatever `order` is; last, it would move every time the camber line
        gained or lost a coefficient, and a gain measured for a circulation
        coefficient would silently be read as one for a camber knob.
        """
        coefficients = self._coefficients(config)
        level = {f"Co[{self.i_row}]": float(_circulation_count(config, self.i_row).Co)}
        return level | {
            name: float(np.mean(coefficients[:, j]))
            for j, name in enumerate(self.names())
        }

    def with_unknowns(self, config, values):
        current = self.unknowns(config)
        moved = {**current, **{k: v for k, v in values.items() if k in current}}

        co_name = f"Co[{self.i_row}]"
        if moved[co_name] != current[co_name]:
            config = _with_circulation(config, self.i_row, moved[co_name])

        blades = list(config.blades)
        blade = blades[self.i_row]

        shift = np.array([moved[name] - current[name] for name in self.names()])

        # A uniform shift, as `LoadingDistribution.with_unknowns` applies one:
        # whatever spanwise variation of the loading shape the design asked
        # for survives being iterated, because only one span fraction was
        # ever measured.
        sections = tuple(
            dataclasses.replace(
                section,
                camber=dataclasses.replace(
                    section.camber,
                    coeff=tuple(np.asarray(section.camber.coeff) + shift),
                ),
            )
            for section in blade.sections
        )

        blades[self.i_row] = dataclasses.replace(blade, sections=sections)
        return dataclasses.replace(config, blades=tuple(blades))

    def paths(self, config):
        paths = {
            f"blades[{self.i_row}].sections[{i_section}].camber.coeff[{j}]"
            for i_section in range(len(config.blades[self.i_row].sections))
            for j in range(self.order - 1)
        }
        paths.add(f"blades[{self.i_row}].count.Co")
        return paths

    def error(self, config, result):
        if result.grid is None or result.machine is None:
            raise MeasurementError(
                f"there is no solved grid to cut, so the loading profile of "
                f"row {self.i_row} is unmeasured."
            )

        self._check(config)
        _circulation_count(config, self.i_row)

        measured = turbigen.loading.measure_profile(
            result, self.i_row, self.spf, self.knob_m()
        )
        if measured is None:
            raise MeasurementError(
                f"no suction surface could be read on row {self.i_row} at "
                f"spf={self.spf:.2f}, so its loading profile is unmeasured. A "
                f"section above a clearance gap has no blade to cut: either "
                f"the march is not a flow field, or the section belongs below "
                f"the gap."
            )
        zeta, fac = measured

        mach_ratio = turbigen.loading.mach_ratio(result.machine, self.i_row)
        target = _target_fac(
            zeta,
            self.zeta_front,
            self.fac_front,
            self.zeta_peak,
            self.fac_peak,
            mach_ratio,
        )

        names = np.array(self.names())
        driven = zeta > self.zeta_front
        held = names[~driven]
        for name, z in zip(held, zeta[~driven]):
            logger.debug(
                f"{name} maps to zeta={z:.3f}, at or below "
                f"zeta_front={self.zeta_front:.2f}; holding it at zero error."
            )

        # A held knob never moves, so it has nothing new to answer for ---
        # reported as exactly zero rather than omitted, so it cannot block
        # convergence forever the way a genuine measurement failure should.
        errors = dict.fromkeys(held.tolist(), 0.0)

        if not np.any(driven):
            logger.info(
                f"Every knob of row {self.i_row}'s loading profile maps "
                f"inside zeta_front={self.zeta_front:.2f}, so none of it is "
                f"driven and the level is unmeasured."
            )
            return errors

        # The mean of the errors is what the level got wrong; what is left
        # over, per point, is blind to the level by construction and is the
        # shape's to answer for.
        residual = fac[driven] - target[driven]
        level = float(np.mean(residual))
        shape = residual - level

        errors.update(zip(names[driven].tolist(), shape.tolist()))
        errors[f"Co[{self.i_row}]"] = level
        return errors

    #
    # A LEVEL AND A SHAPE, WHICH ARE NOT THE SAME KIND OF KNOB
    #

    def _by_knob(self, shape_value, level_value):
        """Return `level_value` for `Co` and `shape_value` for every camber knob.

        In :meth:`unknowns` order, `Co` first --- see there for why that is
        fixed rather than incidental.
        """
        values = {f"Co[{self.i_row}]": level_value}
        return values | {name: shape_value for name in self.names()}

    def gains(self, config):
        """Return the gain of each knob, with the level's declared separately.

        A scalar :attr:`gain` describes the *camber* knobs only, and
        :attr:`gain_Co` supplies the level, because the two disagree on sign
        and no single number covers both. A sequence declares every knob
        instead --- `Co` at index zero, as :meth:`unknowns` orders them --- and
        then carries the level itself, leaving `gain_Co` unread.
        """
        names = list(self.unknowns(config))
        if isinstance(self.gain, (int, float)):
            return self._by_knob(float(self.gain), float(self.gain_Co))
        return dict(zip(names, self._gain_each(names)))

    def clips(self, config):
        del config
        return self._by_knob(self.clip, self.clip_Co)

    def tolerances(self, config):
        del config
        return self._by_knob(self.tolerance, self.tolerance_Co)

    #
    # WHAT THE CONFIG HAS TO PROVIDE
    #

    def knob_m(self):
        """Return the characteristic `m` of each interior coefficient.

        Public because the surface-distribution plot samples the achieved
        curve at exactly these points --- see
        `turbigen.post._draw_loading_profile`. A plot that guessed at its own
        sample positions would be drawing circles the iterator never read.
        """
        return np.arange(1, self.order) / self.order

    def _coefficients(self, config):
        """Return every section's interior coefficients, ``(n_section, order-1)``."""
        self._check(config)
        return np.array(
            [section.camber.coeff for section in config.blades[self.i_row].sections],
            dtype=float,
        )

    def _check(self, config):
        """Raise unless this row's camber lines can carry the knobs, and its
        blade count can carry the level."""
        from turbigen.camber import Bernstein

        if not 0 <= self.i_row < len(config.blades):
            raise ValueError(
                f"i_row={self.i_row} is out of range for a machine with "
                f"{len(config.blades)} blade row(s)."
            )

        for i_section, section in enumerate(config.blades[self.i_row].sections):
            camber = section.camber
            where = f"row {self.i_row} section {i_section}"

            if not isinstance(camber, Bernstein):
                raise ValueError(
                    f"Shaping a loading profile moves the interior "
                    f"coefficients of a Bernstein camber line, and {where} "
                    f"has a {type(camber).__name__} camber, which has none. "
                    f"Set camber: {{type: bernstein, order: {self.order}, "
                    f"coeff: {[0.0] * (self.order - 1)}}}."
                )

            if camber.order != self.order or len(camber.coeff) != self.order - 1:
                raise ValueError(
                    f"Shaping a loading profile of order {self.order} needs "
                    f"exactly {self.order - 1} interior camber coefficient(s) "
                    f"written out in full; {where} has order={camber.order} "
                    f"with {len(camber.coeff)} coefficient(s)."
                )


class ClarkProfile(Iterator):
    """Shape a two-sided thickness to a Clark loading distribution.

    :class:`LoadingProfile` drives a *camber* line against a two-line target
    read off the suction surface. This drives a
    :class:`~turbigen.thickness.ClarkThickness` against
    :mod:`turbigen.clark`, on both surfaces at once, with the camber line held
    where the design put it. Where that one asks what turning gives a loading,
    this asks what thickness does, on the turning already chosen.

    **The knobs are shape-space coefficients, not the perturbations a config
    holds.** A `ClarkThickness` stores a leading edge radius, a wedge angle and
    an interior perturbation on the straight line between them; written as one
    Bernstein curve, those are its first coefficient, its last, and the ones
    between --- see
    :attr:`~turbigen.thickness.ClarkThickness.tau_coeff`. Working in that space
    is what makes this tractable: every knob does the same kind of thing, which
    is to thicken its own surface locally and accelerate the flow over it, so
    **one declared gain covers all of them, sign included**, where a radius and
    a perturbation would each have needed a prior of their own. A positive
    leading edge radius comes free with it, being a square.

    **Each knob is read where it acts**, at
    :attr:`~turbigen.thickness.ClarkThickness.m_ctl`, mapped through each
    surface's own arc length to a surface fraction --- the two surfaces are not
    the same length, so one `m` is not one `z`.

    **A shared end is one knob against two residuals, and the mean is what it
    can answer for.** One nose radius serves both surfaces and one wedge angle
    serves both, so each end's error is the *mean* of the two surfaces'
    residuals there --- the least-squares move for a single knob, being the `x`
    that minimises `(r_s - x)^2 + (r_p - x)^2`. Reading one surface alone
    instead would move both to suit it, making the other worse by as much, and
    would report converged with an unbounded error on the surface nobody read.

    In that mean the level cancels exactly, so a shared end answers only for
    the common mode there and can neither be driven by the blade count nor
    fight it. What it costs is a null: an end where the two surfaces are
    equally wrong in opposite directions reads as converged, because that is
    the part one knob cannot reach. Both ends now carry that blind spot, and
    the interior is where per-surface authority lives --- see
    :attr:`~turbigen.thickness.ClarkThickness.tanwedge` for why the trailing
    edge gave it up.

    **Knobs and signals are born together**, which is what leaves the loop
    determined however the order changes: the knobs are read at
    :attr:`~turbigen.thickness.ClarkThickness.m_ctl`, which is a property of
    the order alone, so a coefficient and the point it is measured at arrive
    and depart as a pair. Sharing an end removes a knob and collapses two
    residuals into one at the same stroke.

    **The level belongs to the blade count**, as it does for every loading
    iterator: a thickness redistributes circulation and cannot create it, so
    the loop is `Co`'s to answer for and what is left of the residuals is the
    shape's. See :class:`PeakMach` for why it cannot simply be ignored.

    **And the level is a circulation, measured as one.** Not a reduction of
    the samples above: those are unevenly spaced, leave about a third of each
    surface unvisited at the two ends, and move when the curve `order`
    changes, so a mean of them would make the measured circulation of an
    unchanged flow depend on how the thickness happens to be parameterised.
    :class:`~turbigen.loading.ClarkMeasurement` integrates the whole cut
    instead, weights each surface by its own length, and normalises the way
    :class:`~turbigen.blade.Circulation` does --- so the error is
    `Co` achieved less `Co` asked for, in the units of the knob it drives, and
    the two are directly comparable rather than merely proportional.

    **Measured from the geometric leading edge, not the stagnation point**, so
    that a target stays still while the thickness under it moves. That leaves
    an `incidence` member shaping the same nose as this does, which is why the
    two are learned in one Jacobian block rather than kept apart --- see
    :meth:`blocks`. Held apart they read each other's work as noise: measured
    on a five-corner sweep, `dchi_LE` was the only knob whose response the
    stepper reported flat, on every run it appeared in, while sitting on its
    clip every pass.
    """

    type: ClassVar[str] = "clark_profile"

    i_row: int = 0
    """Index of the blade row to shape."""

    spf: float = 0.5
    """Span fraction to measure the distribution at [--]."""

    Ma_peak: float = 1.2
    """Target peak Mach number over the trailing edge value [--].

    What :attr:`PeakMach.fac_peak` states, under the name :mod:`turbigen.clark`
    gives it, and one more than the diffusion factor
    :class:`turbigen.metric.DiffusionFactor` records.
    """

    z_peak: float = 0.55
    """Target surface fraction of the suction peak [--]."""

    Ma_LE: float = 1.8
    """Height of the target's leading edge acceleration [--].

    The value Clark's ramp line takes at its reference station, which is what
    sets how hard the suction surface accelerates out of the nose. Not a Mach
    number the curve is required to reach at that station: see `Z_LE` in
    :mod:`turbigen.clark` for why the two are not the same thing.

    Carries the `Ma_2 / Ma_1` factor, written the way
    :attr:`turbigen.loading.Loading.fac_front` and
    :attr:`LoadingProfile.fac_front` are, so the same number means the same
    style of leading edge across rows of differing duty. Divided back out once,
    on the way into :mod:`turbigen.clark`, which works in plain `Ma / Ma_TE`
    throughout --- a curve whose pieces are built from differences between its
    own parameters cannot carry two normalisations at once.
    """

    Ma_PS: float = 0.6
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

    **Positive**, unlike :attr:`LoadingProfile.gain`, and the sign is a
    statement about what a coefficient *is* rather than a guess. A gain is the
    reciprocal of an assumed slope, the step being `u -= gain * e`; here the
    slope is positive, since raising a coefficient thickens the surface,
    accelerates the flow over it and lifts the `fac` the error is measured in.
    A surface running faster than its target is therefore one to thin. The
    *size* is still a guess, improved by the Broyden update inside a run
    rather than written back into the design.

    :attr:`gain_Co` carries the level beside it, as
    :attr:`LoadingProfile.gain_Co` does --- though for a different reason: the
    two agree on sign here, and what keeps them apart is that a circulation
    coefficient and a shape-space coefficient are not the same quantity. See
    :meth:`gains`.
    """

    clip: float = 0.05
    """Largest change in one shape-space coefficient per iteration [--]."""

    tolerance: float = 0.01
    """Converged when every shape residual is within this [--]."""

    gain_Co: float = 1.0
    """How much of the level error to subtract from `Co`, as a prior [--].

    **One, because the error is in the units of the knob.** The level is a
    circulation coefficient too high or too low, and `Co` is the circulation
    coefficient, so moving it by the whole error is the Newton step under a
    slope of one --- which is what a coefficient measured against itself
    should have. That the blade does not land exactly there is loss,
    deviation and the uniform acoustic speed
    :attr:`~turbigen.loading.ClarkMeasurement.Co` assumes; a few per cent on a
    step, not a different order of magnitude.

    Positive, for the reason :attr:`PeakMach.gain` is. **Read only while
    :attr:`gain` is a scalar**, exactly as :attr:`LoadingProfile.gain_Co` is.
    """

    clip_Co: float = 0.05
    """Largest change in the circulation coefficient per iteration [--]."""

    tolerance_Co: float = 0.01
    """Converged when the circulation is within this of the target's [--].

    A `Co` of order 0.7, so this is a per cent or so of it.
    """

    tolerance_tau_LE: float = 0.05
    """Converged when the leading-edge shape residual is within this [--].

    Declared apart from :attr:`tolerance`, and looser by default, because the
    nose is the knob most often left short of its target: a loading that wants
    a sharper leading edge than :attr:`R_LE_lim` allows holds ``tau_LE``
    against that bound every pass, with a residual no step can clear. One shape
    tolerance wider than the rest keeps a clamped nose from stalling a design
    the other knobs have converged, without slackening them.
    """

    R_LE_lim: tuple[float, float] = (0.02, 0.12)
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

    That costs the step its Newton direction for the pass, as any per-knob
    truncation does --- see :func:`_bounded`. Accepted because a bound reached
    is already an abnormal pass: the alternative, scaling the whole block to
    respect it, would let one saturated nose shorten every other knob's step.

    The loop is not made to converge by this. A radius that walks one way
    without turning is a target the nose cannot reach, and the nose is where
    :meth:`_flat_error` is blind by construction --- a shared end reports only
    the mean of the two surfaces. What the bound buys is a design that stays
    meshable and says so, rather than one that runs away and announces it as a
    divergence two iterations later.
    """

    def __post_init__(self):
        if not 0.0 < self.z_peak < 1.0:
            raise ValueError(
                f"A loading peak sits on the surface, so z_peak must satisfy "
                f"0 < z_peak < 1, got {self.z_peak}."
            )
        for name in ("Ma_peak", "Ma_LE", "Ma_PS"):
            if not getattr(self, name) > 0.0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}.")
        if not self.tolerance_tau_LE > 0.0:
            raise ValueError(
                f"tolerance_tau_LE must be positive, got {self.tolerance_tau_LE}."
            )

    #
    # THE PROTOCOL
    #

    def unknowns(self, config):
        """Return the level first, then every shape-space coefficient.

        **`Co` leads**, for the reason :meth:`LoadingProfile.unknowns` puts it
        first: a sequence :attr:`gain` is matched to this order, and index zero
        is the one position that cannot move when a blade changes order.

        Each coefficient is the mean over the row's sections, as
        :meth:`with_unknowns` shifts them all together.
        """
        level = {f"Co[{self.i_row}]": float(_circulation_count(config, self.i_row).Co)}
        coefficients = self._flat_coeff(self._coefficients(config))
        return level | {
            name: float(value)
            for name, value in zip(self._names(self._order(config)), coefficients)
        }

    def with_unknowns(self, config, values):
        current = self.unknowns(config)
        moved = {**current, **{k: v for k, v in values.items() if k in current}}

        co_name = f"Co[{self.i_row}]"
        if moved[co_name] != current[co_name]:
            config = _with_circulation(config, self.i_row, moved[co_name])

        order = self._order(config)
        names = self._names(order)
        shift = self._unflat(
            np.array([moved[name] - current[name] for name in names]), order
        )
        if not np.any(shift):
            return config

        blades = list(config.blades)
        blade = blades[self.i_row]

        # A uniform shift, as `LoadingProfile.with_unknowns` applies one: only
        # one span fraction was ever measured, so whatever spanwise variation
        # of the thickness the design asked for survives being iterated.
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
            for section in blade.sections
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

        blades[self.i_row] = dataclasses.replace(blade, sections=sections)
        return dataclasses.replace(config, blades=tuple(blades))

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
        if result.grid is None or result.machine is None:
            raise MeasurementError(
                f"there is no solved grid to cut, so the loading profile of "
                f"row {self.i_row} is unmeasured."
            )

        self._check(config)
        _circulation_count(config, self.i_row)

        measured = turbigen.loading.measure_clark_profile(
            result, self.i_row, self.spf, self._thickness(config).m_ctl
        )
        if measured is None:
            raise MeasurementError(
                f"both surfaces of row {self.i_row} could not be read at "
                f"spf={self.spf:.2f}, so its loading profile is unmeasured. A "
                f"section above a clearance gap has no blade to cut: either "
                f"the march is not a flow field, or the section belongs below "
                f"the gap."
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

        errors = {f"Co[{self.i_row}]": level}
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

    def blocks(self, config):
        """Return this row's aerofoil block for every knob.

        The same block :class:`Incidence` and :class:`Deviation` put this
        row's recamber in --- see :meth:`Deviation.blocks`. A thickness
        distribution and the angles the flow meets it at are one aerofoil, and
        the loop learns them as one.
        """
        return {name: _AEROFOIL_BLOCK(self.i_row) for name in self.unknowns(config)}

    def _by_knob(self, config, shape_value, level_value):
        """Return `level_value` for `Co` and `shape_value` for every shape knob."""
        values = {f"Co[{self.i_row}]": level_value}
        return values | {name: shape_value for name in self._names(self._order(config))}

    def gains(self, config):
        """Return the gain of each knob, with the level's declared separately.

        A scalar :attr:`gain` describes the *shape* knobs only, and
        :attr:`gain_Co` supplies the level.

        **Not for the reason :meth:`LoadingProfile.gains` splits them**, which
        is that its two disagree on sign; here they agree, both being positive.
        The reason is units. A gain is knob units per error unit, and `Co` is a
        circulation coefficient where the others are shape-space coefficients,
        so one number spread over both would be two different assumed slopes
        wearing one value. :meth:`clips` splits for the same reason and has no
        sequence form to fall back on. Only :meth:`tolerances` splits without
        needing to --- a level residual and a shape residual are both in `fac`
        --- and it splits so that the two can be converged to different
        criteria.

        A sequence :attr:`gain` declares every knob instead --- `Co` at index
        zero, as :meth:`unknowns` orders them --- and then carries the level
        itself, leaving `gain_Co` unread.
        """
        names = list(self.unknowns(config))
        if isinstance(self.gain, (int, float)):
            return self._by_knob(config, float(self.gain), float(self.gain_Co))
        return dict(zip(names, self._gain_each(names)))

    def clips(self, config):
        return self._by_knob(config, self.clip, self.clip_Co)

    def tolerances(self, config):
        by_knob = self._by_knob(config, self.tolerance, self.tolerance_Co)
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
        """Raise unless this row's thickness can carry the knobs."""
        from turbigen.thickness import ClarkThickness

        if not 0 <= self.i_row < len(config.blades):
            raise ValueError(
                f"i_row={self.i_row} is out of range for a machine with "
                f"{len(config.blades)} blade row(s)."
            )

        orders = set()
        for i_section, section in enumerate(config.blades[self.i_row].sections):
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

    learns: ClassVar[bool] = False
    """A loss coefficient relaxed onto its own answer, which needs no fitting.

    This does not correct a design against a target it might reach in several
    ways: it moves a nominal value onto the one the CFD measured, so the error
    *is* the distance left to travel and the sensitivity is one by
    construction. A Broyden update here fits that constant to a few
    near-parallel moves and gets something slightly wrong instead.
    """

    variables: tuple[str, ...] = ()
    """Names of the design variables to relax, as the mean-line design spells
    them."""

    gain: float = 0.5
    tolerance: float = 0.01
    """Permissible error, as a fraction of the nominal value."""

    clip: float = 0.0
    """Largest change in one iteration, as a fraction of the nominal value.

    Relative for the reason :attr:`tolerance` is, and to the same nominal:
    design variables have no common scale, so an absolute number would mean
    something different for every one of them. Written absolutely --- which is
    what :attr:`Iterator.clip` means everywhere else, the other iterators
    moving angles in degrees and coefficients that are already dimensionless
    --- a single ``clip: 0.01`` beside a two-row ``Ys`` of ``[0.030, 0.076]``
    permits a third of the first and an eighth of the second, while the
    tolerance a line above it means one per cent of each. Two conventions, two
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
        Cut plane offset in blade chords; `annulus.CUT_OFFSET` by default.

    Returns
    -------
    dict
        One tuple of `order` coefficients per column of
        :data:`turbigen.bconds.InletProfile.COLUMNS`.

    """
    from turbigen import annulus, bconds

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
    """One copies the exit profile outright; less under-relaxes it.

    Relaxation only: the loop stops where the error is null, and `gain` scales
    the path to that point rather than moving it. What fraction of the exit
    profile the converged inlet actually carries is
    :attr:`transfer_To` and nothing else.
    """

    transfer_To: float = 1.0
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
    feedback, so the default is exactly the loop as it was.
    """

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
        if not 0.0 <= self.transfer_To <= 1.0:
            raise ValueError(
                f"repeat.transfer_To must be between 0 and 1, got "
                f"{self.transfer_To}. It is the fraction of the exit "
                f"temperature profile that comes round again; above one the "
                f"loop amplifies its own profile, and below zero it inverts it."
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
        from turbigen import bconds

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
            raise MeasurementError(
                "there is no solved grid to cut, so the exit profile that "
                "feeds the inlet is unmeasured."
            )

        measured = exit_profile(result, self.order, self.offset)
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
