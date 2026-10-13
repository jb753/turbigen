"""Runs of many cases, each in a numbered directory of its own.

Two of them. The design loop solves a config into ``iter_NNNN`` directories,
moving the design onto its own solution each pass, and promotes the settled
answer to the directory above when it converges. A characteristic sweep holds
a converged design's geometry and steps its operating point into
``chic_NNNN`` directories until a point will not converge.

Both are built on :func:`turbigen.pipeline.solve`, one call per directory, and
neither knows anything of command-line arguments: :mod:`turbigen.cli` decides
where `out_dir` is and hands it over.
"""

import dataclasses
import logging
from pathlib import Path

from turbigen import batch, case, chic, database, iterate, pipeline, warm_field
from turbigen.pipeline import HISTORY_NAME, OUTPUT_NAME, RESTART_NAME

logger = logging.getLogger("turbigen")


def write_input(config, out_dir):
    """Record `config` as the input of the run about to happen in `out_dir`.

    For the verbs that solve into directories they invent. `run` needs nothing
    of the sort -- the file it was handed is already sitting there, whatever it
    was called -- but an iteration's config exists only in memory, having been
    moved there by the iterator, so without this the only record of what
    `iter_0003` solved is the config half of its own `output.yaml`.

    That matters because `output.yaml` is not a file anyone may hand back to
    us. Writing the input is what keeps every directory a run happened in
    addressable: `iter_0003/input.yaml` re-solves that iteration alone, and
    reports it against the field already beside it.
    """
    path = out_dir / batch.INPUT_NAME
    case.write(path, config)
    logger.debug(f"Wrote the config being solved to {path}")
    return path


def design_is_settled(config_path, config):
    """Return whether the design at `config_path` has already converged.

    The same two-part test `database` uses to decide whether a finished run
    counts as a sample (`database.py:236`): the march reached an answer, *and*
    the iterators it was judged by are inside their tolerances. One definition
    of "this design is finished" for the whole package rather than a second one
    here.

    Logged with its reason, as `batch_verb` logs its choice, so "why did this
    iterate" and "why did this not" are both answerable from the log file. That
    matters more than usual because the test cannot see an override: `-s
    mean_line.psi=1.8` invalidates the stored result and this will not notice.
    """
    try:
        _, result = case.read(config_path, design=False)
    except Exception as err:
        logger.debug(f"No usable result beside {config_path}: {err}")
        result = None

    if result is None or not result.converged:
        logger.info("Converging the design first: this case has no converged run.")
        return False

    if not iterate.converged(config, result):
        logger.info(
            "Converging the design first: the stored run finished, but its "
            "design errors are outside their tolerances."
        )
        return False

    logger.info(
        "Sweeping straight away: the stored run converged with its design "
        "errors inside their tolerances."
    )
    return True


@dataclasses.dataclass
class _Chain:
    """Solve one iteration per call, chaining the field and keeping the record.

    What `iterate.converge` calls, satisfying the ``(config, i_iter)`` contract
    it asks for while carrying the two things that have to survive between
    calls. A closure did this and carried `previous` by `nonlocal`; a second
    piece of hidden state in a nested function is where that stops being
    reasonable.

    **It owns the trajectory because it is the only thing that sees it.**
    `converge` holds a history of its own for the Jacobian and cannot reach
    the report, which is drawn inside `solve`; this sits between the two, so
    the record can go down to the report without `converge` learning anything
    about directories. That is why nothing was added to `Result` and nothing
    was read back off disk.

    The grid and the march history are dropped from what is kept. A stripped
    `Result` is a few hundred numbers where a grid is tens of megabytes, which
    is the difference between keeping every pass and keeping none.
    """

    out_dir: Path
    """Where the numbered iteration directories go."""

    previous: Path | None = None
    """The field the next call starts from, advanced as each one finishes."""

    warm: object | None = None
    """A :class:`turbigen.warm_field.Seed` for the first solve, when there is a
    database neighbour to start from. Used only while `previous` is None."""

    trajectory: list = dataclasses.field(default_factory=list)
    """Every ``(config, result)`` pair so far, oldest first."""

    def __call__(self, config_now, i_iter):
        iter_dir = self.out_dir / f"iter_{i_iter:04d}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        iterate.logger.info(f"Iteration {i_iter} in {iter_dir}")

        # Where this iteration's knobs stood, which the next one moves: the
        # sequence is reproducible from the datum, but no single member of it
        # is.
        write_input(config_now, iter_dir)

        # Chained: each iteration starts from the field the last one reached,
        # which is most of the saving. Index-space interpolation covers the
        # mesh moving with the design. The first pass has no chained field, so
        # it takes the warm seed instead when there is one.
        warm = self.warm if self.previous is None else None

        # And the first pass is the one a soft start is for, whatever field it
        # begins from: nothing of this design has been marched yet. `previous`
        # cannot answer that -- a loop handed a restart has one from the
        # outset -- but the trajectory can, being empty until a call returns.
        first = not self.trajectory

        result = pipeline.solve(
            config_now,
            iter_dir,
            self.previous,
            trajectory=self.trajectory,
            warm=warm,
            soft=first,
            retry=True,
        )
        self.previous = iter_dir / RESTART_NAME

        self.trajectory.append(
            (config_now, dataclasses.replace(result, grid=None, history=None))
        )

        return result


def converge_design(config, out_dir, previous=None):
    """Iterate `config` to convergence, keeping every iteration under `out_dir`.

    The whole of an iteration, so that `chic` composes it rather than writing a
    second copy --- the same reason `solve` is one function that `iterate` calls
    repeatedly. Written out twice, the two would drift, which is what happened
    to `turbigen.main`.

    A config with nothing to correct still gets its design point solved once,
    because the sweep that follows needs a field to start from and an answer to
    be a departure from. That branch is `chic`'s alone: `iterate` refuses an
    empty `correct:` list before it ever gets here, so the one pass is not a
    degenerate iteration but the whole of what a fixed geometry needs before it
    can be swept. It runs through the same `_Chain` as any other iteration,
    which is what keeps one definition of where an iteration writes.

    Returns
    -------
    config : Config
        The design that produced `result`.
    result : Result
        What the last iteration achieved.
    converged : bool
    field : Path
        The flow field it reached, for whatever runs next.

    """
    # The database is globbed and parsed once, here: the geometry blend in
    # `warm_start` and the field seed in `nearest_field` both want the same
    # finished runs, and reading a directory of result files is the cost.
    samples = (
        config.database.load_samples(config, out_dir, exclude=(out_dir,))
        if config.database is not None
        else None
    )

    def seed(cfg):
        """A neighbour's field, perturbed onto this design, or None."""
        if previous is not None or not samples:
            return None
        field = database.nearest_field(cfg, samples)
        return warm_field.Seed(field) if field is not None else None

    # Iteration -1: warm start the knobs from the database, excluding this
    # run's own directory, where its iterations will land. Skipped when there
    # are no knobs, as `warm_start` would only warn about nothing to do.
    if iterate.unknowns(config):
        config = database.warm_start(
            config, out_dir, exclude=(out_dir,), samples=samples
        )

    # Built after the warm start, never before: the seed is the nearest
    # neighbour to this design, and the warm start is what moves the design.
    runner = _Chain(out_dir=out_dir, previous=previous, warm=seed(config))

    if config.iterate.correct:
        config, result, converged = iterate.converge(
            config, runner, config.iterate.max_iter
        )
    else:
        # Nothing to correct, so the loop is one pass of it.
        result = runner(config, 0)
        converged = result.converged

    field = promote_final(out_dir, runner.previous.parent, converged)

    return _with_achieved(config, result), result, converged, field


def _with_achieved(config, result):
    """Return `config` running where `result` says the machine ended up.

    The operating point is the one thing a solve can change about the config it
    was given, a throttled exit being handed a mass flow and finding the
    pressure that passes it. Applied on the way out of `converge_design` so
    that whatever runs next -- a sweep, above all -- inherits the pressure the
    design converged at rather than the guess it started from.

    Iterators do not move it, so taking it off the last result is taking it off
    the run that produced the design being returned.
    """
    if result is None or result.operating_point is None:
        return config

    return dataclasses.replace(config, operating_point=result.operating_point)


PROMOTED = (OUTPUT_NAME, RESTART_NAME, HISTORY_NAME, "post.pdf")
"""What a settled design leaves at the root of its directory.

Everything a `run` leaves beside its config, so that a finished `iterate`
directory reads as one: a database glob, a script reading a result and a
`--restart` need not know whether a design took one solve or six.
"""


KEPT_PER_ITERATION = (batch.INPUT_NAME, OUTPUT_NAME, HISTORY_NAME)
"""What an intermediate iteration keeps once the design has settled.

The config it solved, the answer it reached, and the march that got there ---
kilobytes apiece, and the only record of how the design moved. What goes is the
flow field and the report drawn from it, which are the megabytes and which
nothing reads again: `database` filters an unsettled iteration out by
definition, `chic` reads only the config it was given, and no code globs
`iter_*` at all. Both are recoverable by re-running that iteration's
`input.yaml`, which is what makes deleting them a tidy-up rather than a loss.
"""


def promote_final(out_dir, iter_dir, converged):
    """Move the last iteration's artefacts to `out_dir`, and prune the rest.

    Only when the design settled. **A root `output.yaml` therefore means this
    design converged**, which is a far more useful thing for the directory to
    say than "here is wherever the iteration happened to stop". An unsettled
    run keeps every iteration whole, because that is exactly when the history
    is what you came to look at.

    Moved rather than copied, and rather than linked as this once was. A copy
    of `restart.npz` is megabytes duplicated and two files free to disagree; a
    symlink avoids both but needs a filesystem that has them, and left the same
    answer reachable by two paths --- which `database` counted twice, the
    settled iteration being precisely the one that survives its filters.
    Moving has neither problem: one answer, in one place, as a real file.

    Returns the field to carry on from, which `chic` sweeps from and which has
    moved out from under the caller.
    """
    if not converged:
        iterate.logger.info(
            "The design did not settle, so every iteration is kept whole and "
            f"nothing is promoted to {out_dir}."
        )
        return iter_dir / RESTART_NAME

    for name in PROMOTED:
        source = iter_dir / name
        if not source.is_file():
            continue

        destination = out_dir / name
        if destination.exists():
            destination.unlink()
        source.rename(destination)

    iterate.logger.info(f"Moved the settled design's artefacts to {out_dir}")

    _prune_iterations(out_dir)

    return out_dir / RESTART_NAME


def _prune_iterations(out_dir):
    """Cut every iteration directory back to what is worth keeping."""
    removed = 0
    for iter_dir in sorted(out_dir.glob("iter_*")):
        if not iter_dir.is_dir():
            continue

        for entry in iter_dir.iterdir():
            if entry.is_file() and entry.name not in KEPT_PER_ITERATION:
                entry.unlink()
                removed += 1

    if removed:
        iterate.logger.info(
            f"Removed {removed} intermediate file(s); re-run an iteration's "
            f"{batch.INPUT_NAME} to rebuild one."
        )


def characteristic(config, out_dir, previous=None, settled=False):
    """Converge the design unless `settled`, then sweep it into ``chic_NNNN``.

    Each point is an ordinary run in a directory of its own, started from the
    field of the last point that converged. Whether the design has `settled`
    is the caller's to say --- see :func:`design_is_settled`, which reads the
    case file beside a config --- so this is a function of the config and the
    directory alone.

    Returns
    -------
    points : list of turbigen.chic.Point
        Every point run, in the order it was run.
    bracket : tuple of float
        As :func:`turbigen.chic.sweep` returns it.

    Raises
    ------
    ValueError
        If the design had to be converged first and did not converge.
    """
    # The design first, unless it is already done. Every verb implies the
    # ones before it, and a characteristic of a machine still being
    # redesigned is a characteristic of no machine in particular.
    if not settled:
        config, _, converged, previous = converge_design(config, out_dir, previous)
        if not converged:
            raise ValueError(
                "The design did not converge, so there is no machine to "
                "sweep a characteristic of. Fix that with 'iterate' first."
            )

    def run(config_now, i_point):
        """Solve one point of the characteristic, in a directory of its own."""
        nonlocal previous

        point_dir = out_dir / f"chic_{i_point:04d}"
        point_dir.mkdir(parents=True, exist_ok=True)
        chic.logger.info(f"Point {i_point} in {point_dir}")

        # This point's own operating point exists nowhere else: the sweep
        # moved it, and the datum describes the whole characteristic rather
        # than any one station on it.
        write_input(config_now, point_dir)

        # Chained, and near the limit this is what keeps a point converging
        # at all: the smallest perturbation from a field that worked.
        result = pipeline.solve(config_now, point_dir, previous)
        if result.converged:
            # A diverged point is not somewhere to start the next one from,
            # and the next one is a bisection back towards what did work.
            previous = point_dir / RESTART_NAME

        return result

    points, bracket = chic.sweep(config, run)

    chic.logger.info(chic.format_table(points, bracket))

    return points, bracket
