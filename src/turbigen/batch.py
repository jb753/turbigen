"""Choosing which designs to run next.

:mod:`~turbigen.database` warm-starts a design from finished runs, and deduces
everything it needs from them. This is the other half: given a datum config and
some design variables to vary, emit configs that cover the space. Run them, and
their answers are the database the next design starts from.

The two point opposite ways, which is why they are separate modules and
separate verbs. Reading *deduces* --- the design variables are whatever the
runs differ in, the ranges whatever they cover. Writing must be *told*, because
an empty archive differs in nothing. So `batch:` declares what varies, and that
is not a relapse into the old `IndependentConfig`: a design of experiments is a
statement of intent, not an observation.

There are two ways to say what varies, and they are the two questions anyone
asks of a design. `bounds:` gives a box to fill quasi-randomly, which is what
builds an archive worth interpolating in. `values:` names the points outright,
which is the parameter study you run to see a trend --- and which used to be a
shell loop over ``-s``, writing every design into one directory and clobbering
the last, because output goes beside the config it was given and a loop varies
the value without varying the path.

Nothing here runs anything. A verb that both chose designs and executed them
would be a scheduler, and the package this replaces already has two of those.
"""

import dataclasses
import itertools
import logging
import math
import warnings

import numpy as np
from scipy.stats import qmc

from turbigen import node
from turbigen.design import DesignError
from turbigen.node import Node

logger = logging.getLogger("turbigen.batch")
"""What was asked for, what was drawn, and what would not design."""

ATTEMPTS_PER_POINT = 10
"""Points drawn per point wanted before a box is called hopeless."""

DIGITS = 4
"""Width the sequence index is zero-padded to when it names a file."""

DEFAULT_NUMBER = 32
"""Points drawn from a box when the invocation does not say.

A power of two, because that is where Sobol' balance properties hold. Nothing
like it exists for a grid, whose count is the product of what it names.
"""


class Batch(Node):
    """A set of related runs, and how to choose them."""

    bounds: dict = dataclasses.field(default_factory=dict)
    """Range of each design variable, keyed by path: ``{path: [lo, hi]}``.

    Points are drawn from the box quasi-randomly, in an order whose every
    prefix fills it, so a batch can be extended without being regenerated.

    Paths are spelled as :func:`~turbigen.node.flatten` writes them, the same
    as `database.variables`, so a design variable is named identically wherever
    it appears.

    A mapping rather than a list of triples because this is the one section a
    user writes by hand and ``mean_line.psi: [1.2, 2.0]`` is the shortest
    honest spelling of it. The cost is that a `Config` holding one is no longer
    hashable, `dict` not being; nothing hashes a config, and equality and
    round-tripping --- which plenty relies on --- are unaffected.
    """

    values: dict = dataclasses.field(default_factory=dict)
    """Values of each design variable to run at: ``{path: [v0, v1, ...]}``.

    The members are the full factorial over these lists: every combination,
    once. A mapping says that each variable takes each of its values, and the
    product is the only reading of that which does not silently require the
    lists to be the same length.

    Named rather than drawn, so the batch is the same every time it is written
    and no seed enters into it. This is the parameter study --- three values of
    one variable, and a trend to plot --- as against `bounds`, which fills a
    space to be interpolated in later.
    """

    seed: int = 0
    """Seed for the scrambled sequence. The space, so it lives in the file.

    Read only when `bounds` is what varies: a grid of named values is already
    the same every time.
    """

    def is_grid(self):
        """Return whether this batch names its points rather than drawing them."""
        return bool(self.values)

    def paths(self):
        """Return the design variables in the fixed order they are varied in.

        Sorted, so that the same file gives the same batch whichever order the
        keys were typed in. A mapping preserves insertion order and YAML would
        hand that straight through, which would silently make two identical
        spaces produce two different batches --- and, for a grid, two different
        orderings of the same designs.
        """
        return tuple(sorted(self.values if self.is_grid() else self.bounds))

    def limits(self):
        """Return the low and high bound of each design variable, as arrays."""
        pairs = [self.bounds[path] for path in self.paths()]
        return np.array([p[0] for p in pairs], float), np.array(
            [p[1] for p in pairs], float
        )

    def levels(self):
        """Return the values each design variable takes, in `paths` order."""
        return tuple(tuple(self.values[path]) for path in self.paths())

    def size(self):
        """Return how many designs a grid holds, before any are screened out."""
        return math.prod(len(level) for level in self.levels())

    def check(self, config):
        """Raise unless this batch names real, varyable leaves of `config`.

        Checked against the datum rather than trusted, because a misspelled
        path would otherwise create the key it names --- `set_by_path` builds
        what a path implies --- and the strict unknown-key check would then
        reject every design in the batch with a message about the wrong thing.
        """
        if self.bounds and self.values:
            raise ValueError(
                "A batch: section says either bounds:, to fill a box, or "
                "values:, to run named points. It cannot say both."
            )
        if not (self.bounds or self.values):
            raise ValueError(
                "The batch: section names no design variables to vary. Give "
                "bounds: to fill a box, or values: to run named points."
            )

        leaves = node.flatten(config)

        for path in self.paths():
            if path not in leaves:
                raise ValueError(
                    f"The batch variable {path!r} is not a leaf of this config."
                )

            if self.is_grid():
                self._check_values(path)
            else:
                self._check_bounds(path, leaves[path])

    def _check_bounds(self, path, value):
        """Raise unless `path` is something a continuous draw can move."""
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                f"The batch bound {path!r} is {value!r}, which is not a "
                "number to move continuously."
            )
        if isinstance(value, int):
            # Rounding a continuous draw collapses neighbours into repeat
            # designs, which the predictor then averages as though they were
            # repeat runs. And the obvious integer, blade count, changes the
            # mesh. Named values cannot collide this way, so `values:` allows
            # them.
            raise ValueError(
                f"The batch bound {path!r} is a whole number, and whole "
                "numbers are not drawn: rounding a continuous draw makes "
                "duplicate designs. Name the values with values: instead."
            )

        lo, hi = self.bounds[path]
        if not lo < hi:
            raise ValueError(
                f"The batch bound {path!r} is [{lo}, {hi}], which is empty."
            )

    def _check_values(self, path):
        """Raise unless `path` names a list of distinct numbers to run at."""
        levels = self.values[path]

        if not isinstance(levels, (list, tuple)) or not levels:
            raise ValueError(
                f"The batch variable {path!r} is {levels!r}, and it needs a "
                "non-empty list of values to run at."
            )

        for level in levels:
            if isinstance(level, bool) or not isinstance(level, (int, float)):
                raise ValueError(
                    f"The batch variable {path!r} has the value {level!r}, "
                    "which is not a number."
                )

        # A repeated value is a repeated design, which costs a whole solve to
        # learn nothing. Refused rather than quietly collapsed, because the two
        # readings of a duplicate -- a typo, or a deliberate repeat run -- want
        # opposite things, and only one of them is likely.
        if len(set(levels)) != len(levels):
            raise ValueError(
                f"The batch variable {path!r} repeats a value, which would "
                "run the same design twice."
            )


def generate(config, n=None, start=0):
    """Return the designs this config's `batch:` section asks for.

    Pairs of ``(index, config)``. The index is the point's position in the
    Sobol' sequence, or in the grid, rather than its position in the batch.
    Those differ whenever a point fails to design, and keeping the position is
    what lets an extension pick up in the right place and a grid member say
    which combination it is.

    Parameters
    ----------
    config : Config
        The datum, whose `batch:` section says what to vary. Everything it does
        not name is carried through unchanged.
    n : int or None
        How many designs are wanted, when a box is being filled. ``None`` takes
        `DEFAULT_NUMBER`. Ignored by a grid, whose count is the product of what
        it names.
    start : int
        Where to enter the sequence. ``0`` begins a batch; anything else
        extends one. A grid has no tail to extend.

    """
    spec = config.batch
    if spec is None:
        raise ValueError("Writing a batch needs a batch: section saying what to vary.")

    spec.check(config)

    if spec.is_grid():
        return _grid(config, spec)

    return _sequence(config, spec, DEFAULT_NUMBER if n is None else n, start)


def _sequence(config, spec, n, start):
    """Return `n` designs drawn from the box, entering the sequence at `start`."""
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
        candidate, why = _build(config, datum, paths, values)

        if candidate is None:
            # Deterministic given the seed, so this index is not retried later:
            # it would draw the same point and fail the same way.
            logger.info(
                f"Point {index} ({_format_point(paths, values)}) does not "
                f"design, so it is skipped: {why}"
            )
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


def _grid(config, spec):
    """Return every combination of the named values, in a fixed order.

    Row-major over `paths`, so the last variable moves fastest and the
    ordering is a property of the file rather than of the machine that read
    it. The index is the position in the product and is kept when a point is
    skipped, so a member says which combination it is even in a grid with a
    hole in it.
    """
    paths = spec.paths()
    size = spec.size()

    logger.info(
        f"Running {size} design(s) over {len(paths)} design variable(s):\n"
        + _format_values(spec)
    )

    datum = config.to_dict()
    built = []

    for index, values in enumerate(itertools.product(*spec.levels())):
        candidate, why = _build(config, datum, paths, values)

        if candidate is None:
            # Warned rather than noted, unlike a drawn point: you named this
            # one, so its absence from the batch is news.
            logger.warning(
                f"Point {index} ({_format_point(paths, values)}) does not "
                f"design, so it is skipped: {why}"
            )
        else:
            built.append((index, candidate))

    if not built:
        raise ValueError(
            f"None of the {size} points in the grid could be designed:\n"
            + _format_values(spec)
        )

    return built


def _build(config, datum, paths, values):
    """Return ``(design, None)``, or ``(None, why)`` if it cannot be designed.

    Screened here rather than on the cluster. Designing costs no CFD --- it is
    a mean line, an annulus and some blades --- so a corner of the box that
    `solve_for` cannot reach or that `check_round_trip` refuses is found now,
    for nothing, instead of one wasted job at a time.
    """
    data = _strip(datum)
    for path, value in zip(paths, values):
        # float() rather than the value itself, so a drawn numpy scalar
        # serialises. A named whole number is left as it was typed: `values:`
        # allows integers, and a blade count must stay one.
        node.set_by_path(data, path, value if isinstance(value, int) else float(value))

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
            # the cycle config -> batch -> config.
            candidate = type(config).from_dict(data)
            candidate.design()
        except DesignError as err:
            # Returned rather than only logged at debug, because the caller is
            # what knows which point this was, and because a screening that
            # says only "does not design" is unreadable the one time it
            # surprises you --- a point that designs on your machine and not
            # on a CI runner is exactly when the reason is the whole message.
            # Any warning on the way out comes too: a numeric complaint
            # immediately before the failure is usually what explains it.
            why = str(err)
            for warning in caught:
                why += f" [warned: {warning.message}]"
            return None, why

    for warning in caught:
        logger.debug(f"A point designed, but warned: {warning.message}")

    return candidate, None


def _strip(datum):
    """Return a copy of the datum dict with the `batch:` section removed.

    A member of a batch is one point, not a set of them, so carrying the
    section into it would claim it is itself a design of experiments --- and
    batching it again would expand one design into another N. What generated a
    batch is the user's own config file, which they keep; nothing is written to
    say so.
    """
    data = {key: value for key, value in datum.items() if key != "batch"}
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
"""What a run's config is called inside a directory turbigen made for it.

Not what a run writes, which is ``output.yaml``, so the config and the answer
it reaches sit side by side and neither can overwrite the other.

Named here because a batch member was the first thing to need it, but the
convention is general: `iterate` and `chic` write it into the directories they
invent too, so that every directory a run happened in holds the input that
produced it. That is what lets ``output.yaml`` be refused as a target --- there
is always another file naming the same directory.
"""


def member_name(index):
    """Return the path a design at `index` is written to.

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


def _format_point(paths, values):
    """Return one point as it would be typed, for a message about it."""
    return ", ".join(f"{path}={value:g}" for path, value in zip(paths, values))


def _format_bounds(spec):
    """Return a line per design variable, saying what it is drawn between."""
    paths = spec.paths()
    width = max(len(path) for path in paths)
    lines = [f"{'variable':<{width}}  {'low':>10}  {'high':>10}"]
    for path in paths:
        lo, hi = spec.bounds[path]
        lines.append(f"{path:<{width}}  {lo:10.4g}  {hi:10.4g}")
    return "\n".join(lines)


def _format_values(spec):
    """Return a line per design variable, saying what it is run at."""
    paths = spec.paths()
    width = max(len(path) for path in paths)
    lines = [f"{'variable':<{width}}  values"]
    for path in paths:
        levels = ", ".join(f"{level:g}" for level in spec.values[path])
        lines.append(f"{path:<{width}}  {levels}")
    return "\n".join(lines)
