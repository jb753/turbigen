"""Choosing which designs to run next.

:mod:`~turbigen2.database` warm-starts a design from finished runs, and deduces
everything it needs from them. This is the other half: given a datum config and
bounds on some of its design variables, emit configs that cover the space. Run
them, and their answers are the database the next design starts from.

The two point opposite ways, which is why they are separate modules and
separate verbs. Reading *deduces* --- the design variables are whatever the
runs differ in, the ranges whatever they cover. Writing must be *told*, because
an empty archive differs in nothing. So `sample:` declares bounds, and that is
not a relapse into the old `IndependentConfig`: a design of experiments is a
statement of intent, not an observation.

Nothing here runs anything. A verb that both chose designs and executed them
would be a scheduler, and the package this replaces already has two of those.
"""

import dataclasses
import logging
import warnings

import numpy as np
from scipy.stats import qmc

from turbigen2 import node
from turbigen2.design import DesignError
from turbigen2.node import Node

logger = logging.getLogger("turbigen.sample")
"""What was asked for, what was drawn, and what would not design."""

ATTEMPTS_PER_POINT = 10
"""Points drawn per point wanted before a box is called hopeless."""

DIGITS = 4
"""Width the sequence index is zero-padded to when it names a file."""


class Sample(Node):
    """A box of design variables to cover with runs."""

    bounds: dict = dataclasses.field(default_factory=dict)
    """Range of each design variable, keyed by path: ``{path: [lo, hi]}``.

    Paths are spelled as :func:`~turbigen2.node.flatten` writes them, the same
    as `database.variables`, so a design variable is named identically wherever
    it appears.

    A mapping rather than a list of triples because this is the one section a
    user writes by hand and ``mean_line.psi: [1.2, 2.0]`` is the shortest
    honest spelling of it. The cost is that a `Config` holding one is no longer
    hashable, `dict` not being; nothing hashes a config, and equality and
    round-tripping --- which plenty relies on --- are unaffected.
    """

    seed: int = 0
    """Seed for the scrambled sequence. The space, so it lives in the file."""

    def paths(self):
        """Return the design variables in the fixed order they are drawn in.

        Sorted, so that the same file gives the same design whichever order the
        keys were typed in. A mapping preserves insertion order and YAML would
        hand that straight through, which would silently make two identical
        spaces produce two different batches.
        """
        return tuple(sorted(self.bounds))

    def limits(self):
        """Return the low and high bound of each design variable, as arrays."""
        pairs = [self.bounds[path] for path in self.paths()]
        return np.array([p[0] for p in pairs], float), np.array(
            [p[1] for p in pairs], float
        )

    def check(self, config):
        """Raise unless every bound names a real, continuous leaf of `config`.

        Checked against the datum rather than trusted, because a misspelled
        path would otherwise create the key it names --- `set_by_path` builds
        what a path implies --- and the strict unknown-key check would then
        reject every design in the batch with a message about the wrong thing.
        """
        leaves = node.flatten(config)

        for path in self.paths():
            if path not in leaves:
                raise ValueError(
                    f"The sample bound {path!r} is not a leaf of this config."
                )

            value = leaves[path]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(
                    f"The sample bound {path!r} is {value!r}, which is not a "
                    "number to move continuously."
                )
            if isinstance(value, int):
                # Rounding a continuous draw collapses neighbours into repeat
                # designs, which the predictor then averages as though they
                # were repeat runs. And the obvious integer, blade count,
                # changes the mesh.
                raise ValueError(
                    f"The sample bound {path!r} is a whole number, and whole "
                    "numbers are not sampled: rounding a continuous draw makes "
                    "duplicate designs."
                )

            lo, hi = self.bounds[path]
            if not lo < hi:
                raise ValueError(
                    f"The sample bound {path!r} is [{lo}, {hi}], which is empty."
                )


def generate(config, n, start=0):
    """Return `n` designs from the sequence, as ``(index, config)`` pairs.

    The index is the point's position in the Sobol sequence, not its position
    in the batch. Those differ whenever a point fails to design, and the
    difference is what makes an extension pick up in the right place.

    Parameters
    ----------
    config : Config
        The datum, whose `sample:` section says what to vary. Everything it
        does not name is carried through unchanged.
    n : int
        How many designs are wanted.
    start : int
        Where to enter the sequence. ``0`` begins a batch; anything else
        extends one.

    """
    spec = config.sample
    if spec is None:
        raise ValueError("Sampling needs a sample: section saying what to vary.")
    if not spec.bounds:
        raise ValueError("The sample: section names no design variables to vary.")

    spec.check(config)

    paths = spec.paths()
    lo, hi = spec.limits()

    if n & (n - 1):
        # scipy says the same thing, but only when a whole batch is drawn at
        # once, and screening means drawing one point at a time.
        logger.warning(
            f"Sobol' balance properties hold at powers of two, so {n} points "
            "cover the space slightly less evenly than 32 or 64 would."
        )

    engine = qmc.Sobol(d=len(paths), scramble=True, seed=spec.seed)
    if start:
        # Exactly the tail of one longer batch: fast_forward(k) then drawing n
        # gives the same points as drawing k + n and discarding the first k.
        # That equivalence is the whole reason this is Sobol and not a Latin
        # hypercube, whose stratification is defined by the batch size.
        engine.fast_forward(start)

    logger.info(
        f"Drawing {n} design(s) from index {start}, over "
        f"{len(paths)} design variable(s):\n" + _format_bounds(spec)
    )

    datum = config.to_dict()
    drawn = []
    index = start

    for _ in range(n * ATTEMPTS_PER_POINT):
        if len(drawn) == n:
            break

        values = lo + (hi - lo) * engine.random(1)[0]
        candidate = _build(config, datum, paths, values)

        if candidate is None:
            # Deterministic given the seed, so this index is not retried later:
            # it would draw the same point and fail the same way.
            logger.info(f"Point {index} does not design, so it is skipped.")
        else:
            drawn.append((index, candidate))

        index += 1

    if len(drawn) < n:
        raise ValueError(
            f"Only {len(drawn)} of {n} points designed in "
            f"{n * ATTEMPTS_PER_POINT} attempts. The bounds are probably "
            f"mostly outside what this design can do:\n{_format_bounds(spec)}"
        )

    return drawn


def _build(config, datum, paths, values):
    """Return the design at `values`, or None if it cannot be designed.

    Screened here rather than on the cluster. Designing costs no CFD --- it is
    a mean line, an annulus and some blades --- so a corner of the box that
    `solve_for` cannot reach or that `check_round_trip` refuses is found now,
    for nothing, instead of one wasted job at a time.
    """
    data = _strip(datum)
    for path, value in zip(paths, values):
        node.set_by_path(data, path, float(value))

    # Screening is deliberate probing of bad points, so numeric complaints are
    # expected rather than newsworthy: a corner where an equation of state
    # takes the log of a negative number is exactly a corner to reject. They
    # are caught rather than silenced, and logged below if the point survives.
    #
    # Logged rather than re-raised. A design that iterates through a bad guess
    # and recovers still warns on the way, so re-issuing would turn a
    # survivable hiccup into a hard failure for anyone who escalates warnings
    # -- as the test suite does -- and would detach the warning from the point
    # that caused it.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            # `type(config)` rather than importing Config, which would close
            # the cycle config -> sample -> config.
            candidate = type(config).from_dict(data)
            candidate.design()
        except DesignError as err:
            logger.debug(f"A point did not design: {err}")
            return None

    for warning in caught:
        logger.debug(f"A point designed, but warned: {warning.message}")

    return candidate


def _strip(datum):
    """Return a copy of the datum dict with the `sample:` section removed.

    A member of a batch is one point, not a space, so carrying the bounds into
    it would claim it is itself a design of experiments --- and sampling it
    again would expand one design into another N. What generated a batch is the
    user's own config file, which they keep; nothing is written to say so.
    """
    data = {key: value for key, value in datum.items() if key != "sample"}
    # Deep enough: `set_by_path` writes into nested containers, and those are
    # still shared with `datum` until this copies the branch it touches.
    return _deepcopy(data)


def _deepcopy(value):
    """Return a copy of the plain containers in `value`, sharing its leaves."""
    if isinstance(value, dict):
        return {key: _deepcopy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_deepcopy(item) for item in value]
    return value


INPUT_NAME = "input.yaml"
"""What a member's config is called inside its own directory.

Not what a run writes, which is ``output.yaml``, so a member and the answer it
reaches sit side by side and neither can overwrite the other.
"""


def member_name(index):
    """Return the path a design at sequence `index` is written to.

    A directory of its own, because one directory is one run: it is what gives
    every member somewhere to be run into, and what keeps a batch of thirty-two
    from sharing one ``output.yaml``, one ``restart.npz`` and one report.
    """
    return f"{index:0{DIGITS}d}/{INPUT_NAME}"


def next_index(directories):
    """Return the sequence index to carry on from, given earlier batches.

    Reads the member directory names rather than counting them, so a batch that
    skipped an infeasible point still says where the sequence had reached.
    Returns 0 when there is nothing to carry on from, which makes ``--continue``
    on an empty tree the same as starting.
    """
    highest = -1

    for directory in directories:
        for member in directory.iterdir():
            try:
                highest = max(highest, int(member.name))
            except ValueError:
                # Something else living beside the batch -- a log file, a
                # submission script -- which is not ours to interpret.
                continue

    return highest + 1


def _format_bounds(spec):
    """Return a line per design variable, saying what it is drawn between."""
    paths = spec.paths()
    width = max(len(path) for path in paths)
    lines = [f"{'variable':<{width}}  {'low':>10}  {'high':>10}"]
    for path in paths:
        lo, hi = spec.bounds[path]
        lines.append(f"{path:<{width}}  {lo:10.4g}  {hi:10.4g}")
    return "\n".join(lines)
