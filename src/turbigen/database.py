"""Starting an iteration from designs already run.

:func:`~turbigen.iterate.converge` begins wherever the file left its knobs,
then spends CFD on walking them to the answer. When similar machines have
already been solved, that first guess is needlessly bad: a design close to one
already run should start near its neighbour's converged recambers, not at
whatever was typed.

So a run is a *sample*: its config records the knobs it ended on, and its
``result:`` records whether it got there. Reading a directory of them gives a
scattered map of converged designs, and a new design is warm-started by
blending the nearest.

Nothing is declared. The package this replaces asks for the independent
variables in the file, as `IndependentConfig`, with limits for each; here the
design variables are whatever the finished runs *differ in*, and the range to
normalise them by is whatever range they cover. What is not a design variable
is told rather than guessed: an :class:`~turbigen.iterate.Iterator` declares
which config leaves it moves, and those are outputs of the iteration rather
than inputs to it.

The prediction is inverse distance weighting --- see :func:`_predict` for why
that, and not the polynomial surrogate it replaces.
"""

import logging
from pathlib import Path

import numpy as np

from turbigen import iterate, node
from turbigen.node import Node

logger = logging.getLogger("turbigen.iterate")
"""Shares the iteration's logger: a warm start is iteration -1.

Named for `iterate` rather than for a run so that the same console filter
covers both, and so the few lines describing where a design started sit beside
the table describing where it went.
"""

SUBTREE = ("fluid", "mean_line", "annulus", "blades")
"""Config keys a design variable may be found under.

The design, and nothing about how it was executed. Without this a machine run
at a finer mesh or for more steps would count as a different design, and
`solver.n_step` would become an axis of the space.
"""

EPS = 1e-9
"""Below this normalised distance a query is treated as sitting on a sample."""


class Database(Node):
    """Finished runs to start an iteration from."""

    path: str = ""
    """Glob matching the case files to read, relative to the config file.

    Recursive patterns are the point: ``../runs/**/output.yaml``. Matching the
    per-iteration subdirectories of an earlier iterate as well as its final
    answer is harmless, because what makes a match a sample is read from its
    ``result:`` rather than from where it sits.

    ``output.yaml`` is written by any command with something to record, `report`
    included, so matching one is not by itself evidence that a run happened
    there. What makes a match a sample is its ``result:``, and a report that
    reached no answer writes the config alone --- which :func:`_sample` skips,
    on the same line that skips a march which blew up.
    """

    power: float = 2.0
    """Exponent on inverse distance. Higher weights the nearest sample more."""

    variables: tuple[str, ...] = ()
    """Design variables to use, as `node.flatten` spells them.

    Empty, and normally left so: the point of this module is that they are
    deduced. An escape hatch for a database whose runs happen to differ in
    something that is not a design variable.
    """

    def load(self, config, anchor, exclude=()):
        """Return the finished designs to start `config` from.

        Parameters
        ----------
        config : Config
            The design being started, whose knobs a sample must match.
        anchor : Path
            Directory `path` is resolved against.
        exclude : sequence of Path
            Directories whose contents are not samples.

        """
        wanted = set(iterate.unknowns(config))
        excluded = tuple(Path(directory).resolve() for directory in exclude)

        samples = []
        for match in sorted(Path(anchor).glob(self.path)):
            resolved = match.resolve()

            # A run whose own output is under the glob would otherwise be
            # started from itself, predicting its own answer with no error.
            if any(resolved.is_relative_to(directory) for directory in excluded):
                continue

            sample = _sample(resolved, wanted)
            if sample is not None:
                samples.append(sample)

        return samples

    def candidates(self, config, samples):
        """Return the design variables to measure distance in, sorted.

        Every leaf of the design that the samples differ in, less the ones an
        iterator owns. A leaf they all agree on says nothing about which sample
        is nearest, and would divide by a zero range on the way to saying it.
        """
        if self.variables:
            return tuple(self.variables)

        owned = set()
        for iterator in config.iterate.correct:
            owned |= set(iterator.paths(config))

        flat = [node.flatten(sample) for sample in samples]

        # A query cannot be placed on an axis its own config does not have, and
        # a sample cannot be placed on one it is missing either.
        shared = set(node.flatten(config))
        for leaves in flat:
            shared &= set(leaves)

        varying = set()
        for path in shared - owned:
            if _root(path) not in SUBTREE:
                continue
            values = [leaves[path] for leaves in flat]
            if not all(_is_number(value) for value in values):
                continue
            if max(values) > min(values):
                varying.add(path)

        return tuple(sorted(varying))


def warm_start(config, anchor, exclude=()):
    """Return `config` with its iterators started from designs already run.

    Returns `config` untouched, saying why, whenever there is nothing to go on:
    warm starting is an optimisation, and a design that cannot be warm-started
    is a design that starts where its file says.

    Parameters
    ----------
    config : Config
        The design about to be iterated.
    anchor : Path
        Directory the database glob is resolved against, which is the one
        holding the config file rather than the process working directory.
    exclude : sequence of Path
        Directories whose contents are not samples, normally this run's own
        output.

    """
    if config.database is None:
        return config

    names = tuple(sorted(iterate.unknowns(config)))
    if not names:
        logger.warning("Nothing is being iterated, so there is nothing to start.")
        return config

    samples = config.database.load(config, anchor, exclude)
    if not samples:
        logger.warning(
            "No finished runs matched the database, so the design starts "
            "where its file left it."
        )
        return config

    # No bail on an empty set of variables. Samples that differ in nothing are
    # repeats of one design, every one of them sits on top of the query, and
    # the mean of what they converged to is the right answer -- which is the
    # same path a lone sample takes to being copied.
    variables = config.database.candidates(config, samples)
    flat = [node.flatten(sample) for sample in samples]

    X = np.array([[leaves[path] for path in variables] for leaves in flat], float)
    U = np.array(
        [[iterate.unknowns(s)[name] for name in names] for s in samples], float
    )
    xq = np.array([node.flatten(config)[path] for path in variables], float)

    lo, span = _scale(X)
    predicted = dict(
        zip(
            names, _predict((xq - lo) / span, (X - lo) / span, U, config.database.power)
        )
    )

    logger.info(
        f"Starting from {len(samples)} finished run(s), over "
        f"{len(variables)} design variable(s):\n"
        + _format_table(iterate.unknowns(config), predicted)
    )

    for iterator in config.iterate.correct:
        mine = {
            name: predicted[name]
            for name in iterator.unknowns(config)
            if name in predicted
        }
        if mine:
            config = iterator.with_unknowns(config, mine)

    return config


def _sample(path, wanted):
    """Return the config at `path` if it is a sample, and None if it is not."""
    # Imported here because `case` reads a `Config`, and a `Config` holds a
    # `Database`: at module scope this closes a cycle that only stays unbroken
    # while `turbigen/__init__.py` happens to import `case` before `config`.
    from turbigen import case

    try:
        # No machine designed: the knobs and the design variables are both read
        # off the config, and the result is wanted only to say whether the run
        # finished.
        config, result = case.read(path, design=False)
    except Exception as err:
        logger.warning(f"Skipping {path}, which did not read as a case: {err}")
        return None

    if result is None or not result.converged:
        # Either never run, or a march that blew up. Its knobs are where the
        # iteration happened to be standing, not an answer.
        return None

    if not iterate.converged(config, result):
        # Run to convergence, but the *design* had not settled: an intermediate
        # iteration, whose recambers are on their way somewhere.
        return None

    if set(iterate.unknowns(config)) != wanted:
        # A different row count, or a different set of iterators. There is no
        # correspondence between its knobs and this design's.
        return None

    return config


def _scale(X):
    """Return the offset and divisor normalising `X` onto the unit cube.

    A column every sample agrees on has no range to divide by --- which one
    sample makes true of every column --- so it is left alone rather than
    special-cased. Every query then sits at distance zero along it, which is
    the right answer: it separates nothing.
    """
    lo = X.min(axis=0)
    span = X.max(axis=0) - lo
    span[span == 0.0] = 1.0
    return lo, span


def _predict(xq, X, U, power):
    """Return the inverse-distance-weighted blend of `U` at `xq`.

    Chosen over the polynomial surrogate this replaces because the sample count
    is the binding constraint. A total-order cubic in eight design variables is
    165 terms against perhaps fifteen finished runs, which is why the old fit
    needed an adaptive order, a degrees-of-freedom fraction and a train/test
    split to stop it overfitting, and why it could still return an unmeshable
    blade outside the sample hull.

    This has nothing to choose and nothing to condition. It also *cannot* leave
    the hull: the result is a convex combination of values that converged, so
    no clip is needed to keep a warm start meshable. The price is that it
    carries no trend --- its gradient is zero at every sample, and far from all
    of them it decays to their mean --- so it interpolates between designs
    already run rather than extrapolating beyond them. For a starting point
    that Broyden then refines, bounded beats accurate.
    """
    distance = np.linalg.norm(X - xq, axis=1)

    # Sitting on a sample, return it: exactly reproducing a design already run
    # is worth more than blending it with its neighbours, and it is what makes
    # a single sample degrade to a copy rather than to a division by zero.
    #
    # Averaged rather than first-past-the-post, because two runs of the same
    # design are both on top of the query and neither is more authoritative
    # than the other. Their knobs differ by convergence noise, and the mean of
    # them is a better answer than whichever the glob happened to sort first.
    on_top = distance < EPS
    if on_top.any():
        return U[on_top].mean(axis=0)

    weight = distance**-power
    return (weight @ U) / weight.sum()


def _format_table(before, after):
    """Return a line per knob, saying where it started and where it now does."""
    width = max(len(name) for name in before)
    lines = [f"{'name':<{width}}  {'was':>10}  {'now':>10}"]
    for name in sorted(before):
        lines.append(f"{name:<{width}}  {before[name]:10.4g}  {after[name]:10.4g}")
    return "\n".join(lines)


def _root(path):
    """Return the config key a flattened path starts under."""
    return path.split(".")[0].split("[")[0]


def _is_number(value):
    """Return whether `value` is a number a distance can be measured along.

    Booleans are not, although Python says they are integers: a switch that
    happens to differ between two runs is a different machine, not a machine
    a little further along an axis.
    """
    return isinstance(value, (int, float)) and not isinstance(value, bool)
