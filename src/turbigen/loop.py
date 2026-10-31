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
import itertools
from pathlib import Path

import ember.yaml_util

from turbigen import (
    batch,
    case,
    chic,
    database,
    include,
    iterate,
    nozzle,
    pipeline,
    restart,
    warm_field,
)
from turbigen.pipeline import HISTORY_NAME, OUTPUT_NAME, RESTART_NAME

logger = logging.getLogger("turbigen")


def write_input(config, out_dir, start=None):
    """Record `config` as the input of the run about to happen in `out_dir`,
    starting from the field at `start`, or from the guess if None.

    For the verbs that solve into directories they invent. `run` needs nothing
    of the sort -- the file it was handed is already sitting there, whatever it
    was called -- but an iteration's config exists only in memory, having been
    moved there by the iterator, so without this the only record of what
    `iter_0003` solved is the config half of its own `output.yaml`.

    That matters because `output.yaml` is not a file anyone may hand back to
    us. Writing the input is what keeps every directory a run happened in
    addressable: `iter_0003/input.yaml` re-solves that iteration alone, and
    reports it against the field already beside it.

    Its `restart:` is the field this run starts from, not the one the loop was
    seeded with, so that re-solving it alone starts where it did.
    """
    field = str(Path(start).resolve()) if start is not None else None
    config = dataclasses.replace(config, restart=field)
    path = out_dir / batch.INPUT_NAME
    case.write(path, config)
    logger.debug(f"Wrote the config being solved to {path}")
    return path


def settled_design(config_path, config, quiet=False):
    """Return `config` ready to sweep and the result its design point reached,
    if the design has already converged, or None if it has to be converged
    first.

    The same two-part test `database` uses to decide whether a finished run
    counts as a sample (`database.py:236`): the march reached an answer, *and*
    the iterators it was judged by are inside their tolerances. One definition
    of "this design is finished" for the whole package rather than a second one
    here.

    Two places hold such an answer. The config's own `result:`, when it is a
    copied `output.yaml`. Or the `output.yaml` beside it, which is where a run
    leaves one --- including the settled design `-o` promotes over its copy ---
    but which answers for this config only if the field beside it is stamped as
    this design's solution, since an `iterate` run in place leaves the starting
    design's config beside the settled design's answer. The stamp also sees an
    override, which the config's own `result:` cannot: `-s mean_line.psi=1.8`
    invalidates the answer beside it.

    From `output.yaml`, the config comes back at the operating point the run
    reached, as :func:`converge_design` hands one on, so the sweep departs from
    where the field was marched to rather than from the guess.

    Logged with its reason, as `batch_verb` logs its choice, so "why did this
    iterate" and "why did this not" are both answerable from the log file;
    at debug level when `quiet`, for a caller only looking ahead.
    """
    config_path = Path(config_path)
    say = logger.debug if quiet else logger.info

    found = _stored(config_path)
    if found is None:
        found = _stored_beside(config_path, config, say)
        if found is not None:
            config = dataclasses.replace(
                config, operating_point=found[0].operating_point
            )

    result = found[1] if found is not None else None
    if result is None or not result.converged:
        say("Converging the design first: this case has no converged run.")
        return None

    if not iterate.converged(config, result):
        say(
            "Converging the design first: the stored run finished, but its "
            "design errors are outside their tolerances."
        )
        return None

    say(
        "Sweeping straight away: the stored run converged with its design "
        "errors inside their tolerances."
    )
    return config, result


def _stored(path):
    """Return the config and result of the case file at `path`, or None if it
    holds no result or cannot be read."""
    try:
        stored, result = case.read(path, design=False)
    except Exception as err:
        logger.debug(f"No usable result in {path}: {err}")
        return None

    return None if result is None else (stored, result)


def _stored_beside(config_path, config, say):
    """Return the config and result of the `output.yaml` beside `config_path`,
    or None unless the field beside it is stamped as `config`'s solution."""
    output = config_path.parent / OUTPUT_NAME
    field = config_path.parent / RESTART_NAME
    if output.resolve() == config_path.resolve() or not field.is_file():
        return None

    if restart.read_stamp(field) != restart.design_stamp(iterate.resolve(config)):
        say(
            f"The answer in {output} is not this design's, its field being "
            "stamped for another, so it does not count as settled."
        )
        return None

    return _stored(output)


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
        write_input(config_now, iter_dir, self.previous)

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


def converge_design(config, out_dir, previous=None, promote_input=False):
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

    `promote_input` says the config at the root of `out_dir` is turbigen's own
    copy, made by ``-o``, and so may be replaced by the settled design's: see
    :func:`promote_final`.

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

    field = promote_final(out_dir, runner.previous.parent, converged, promote_input)

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
`restart:` need not know whether a design took one solve or six.
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


def promote_final(out_dir, iter_dir, converged, promote_input=False):
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

    With `promote_input`, the settled iteration's config goes up too, over the
    root one, so that the config beside the promoted field is the design that
    field solves: without it, `report` on the root config resolves the
    starting design, which the field's stamp does not match, and the settled
    design's own config is left with no field beside it. Its `restart:` names
    that field, so re-running it continues from its own solution; the
    `output.yaml` beside it still records where the settled run started. The
    starting design is not lost, `iter_0000` keeping its `input.yaml`. Only for
    a root config turbigen copied there itself, because one handed to it in
    place is someone's own file, and `run` leaves that alone; so it is replaced
    only under ``-o``, and otherwise kept with a note why.

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

        # The record of what it started from was written relative to where it
        # was, so it is rewritten against where it is now.
        if name == OUTPUT_NAME:
            _rebase_restart(destination, iter_dir)

    if promote_input:
        _promote_input(iter_dir / batch.INPUT_NAME, out_dir / batch.INPUT_NAME)
    else:
        iterate.logger.info(
            f"Left the config in {out_dir} as it is, not being a copy made by "
            f"-o; the settled design's is {iter_dir / batch.INPUT_NAME}."
        )

    iterate.logger.info(f"Moved the settled design's artefacts to {out_dir}")

    _prune_iterations(out_dir)

    return out_dir / RESTART_NAME


def _promote_input(source, destination):
    """Write the settled config at `source` over the root one at `destination`,
    starting from the field promoted beside it.

    Copied rather than moved: the iteration keeps its own config, as every
    pruned iteration does, so its directory still reads as one run.
    """
    data = ember.yaml_util.read_yaml(source)
    data["restart"] = RESTART_NAME
    ember.yaml_util.write_yaml(data, destination)
    iterate.logger.info(f"Wrote the settled design's config over {destination}")


def _rebase_restart(path, was_in):
    """Rewrite the `restart:` of the case file at `path`, which was written in
    the directory `was_in`, to name the same field from where it is now."""
    data = ember.yaml_util.read_yaml(path)
    if not isinstance(data, dict) or data.get("restart") is None:
        return

    data = include.relative(include.anchor(data, was_in), path.parent)
    ember.yaml_util.write_yaml(data, path)


def _forget_pruned_start(path):
    """Drop the `restart:` of the input at `path` if its field has been pruned.

    So that re-solving a pruned iteration alone starts from the guess, saying
    so, rather than refusing a field that no longer exists. Its `output.yaml`
    keeps the record of where it did start.
    """
    data = ember.yaml_util.read_yaml(path)
    if not isinstance(data, dict):
        return

    start = data.get("restart")
    if start is None or (path.parent / start).is_file():
        return

    data["restart"] = None
    ember.yaml_util.write_yaml(data, path)
    iterate.logger.debug(
        f"Cleared the restart: of {path}, whose field {start} was pruned; "
        "re-solving it alone starts from the guess."
    )


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

    # Only once every field has gone, since an input names its neighbour's.
    for path in sorted(out_dir.glob(f"iter_*/{batch.INPUT_NAME}")):
        _forget_pruned_start(path)

    if removed:
        iterate.logger.info(
            f"Removed {removed} intermediate file(s); re-run an iteration's "
            f"{batch.INPUT_NAME} to rebuild one."
        )


def characteristic(config, out_dir, previous=None, design=None, promote_input=False):
    """Converge the design unless it has settled, then sweep it into
    ``chic_NNNN``.

    Each point is an ordinary run in a directory of its own, started from the
    field of the last point that stood. With more than one of `chic.speeds`,
    each speedline's points go under ``speed_S.SSS/``, the nozzle being sized
    at the design speed under ``speed_1.000/``, and each speedline departs
    from that throat run at its own speed: the first point there, soft
    started from the nearest speed already run. A choke leg with a nozzle runs
    on the open duct, from the pressure that passes the datum's flow.

    `design` is the result the design point reached when it has already
    settled --- see :func:`settled_design`, which reads it from the case file
    beside a config --- and None to converge the design first, so this is a
    function of the config and the directory alone.

    Returns
    -------
    points : list of turbigen.chic.Point
        Every point run, in the order it was run.
    bracket : tuple of float
        As :func:`turbigen.chic.sweep` returns it, for the design speed's
        stall leg, or its choke leg when there is no stall leg.

    Raises
    ------
    ValueError
        If the design had to be converged first and did not converge.
    """
    # The design first, unless it is already done. Every verb implies the
    # ones before it, and a characteristic of a machine still being
    # redesigned is a characteristic of no machine in particular.
    if design is None:
        config, design, converged, previous = converge_design(
            config, out_dir, previous, promote_input
        )
        if not converged:
            raise ValueError(
                "The design did not converge, so there is no machine to "
                "sweep a characteristic of. Fix that with 'iterate' first."
            )

    speeds = config.chic.speeds
    layout = tuple(speeds) != (1.0,)

    # Each speedline numbers its own points, in a directory of its own when
    # there is more than one; the field chained from point to point is shared,
    # so that whichever speedline runs next starts from the last success
    state = {"previous": previous}

    def runner(speed):
        base = out_dir / f"speed_{speed:.3f}" if layout else out_dir
        counter = itertools.count()

        def run(config_now, _, soft=False):
            """Solve one point of the characteristic, in a directory of its own."""
            i_point = next(counter)
            point_dir = base / f"chic_{i_point:04d}"
            point_dir.mkdir(parents=True, exist_ok=True)
            chic.logger.info(f"Point {i_point} in {point_dir}")

            # This point's own operating point exists nowhere else: the sweep
            # moved it, and the datum describes the whole characteristic
            # rather than any one station on it.
            write_input(config_now, point_dir, state["previous"])

            # Chained, and near the limit this is what keeps a point
            # converging at all: the smallest perturbation from a field that
            # worked.
            result = pipeline.solve(config_now, point_dir, state["previous"], soft=soft)
            if result.converged:
                # A diverged point is not somewhere to start the next one
                # from, and the next one is a fine step on from what did work.
                state["previous"] = point_dir / RESTART_NAME

            return result

        return run

    def leg(run, field):
        """Return `run` for one leg of a sweep, departing from `field`, each
        point restarted from the one `chic.sweep` names by its index."""
        fields = []

        def run_leg(config_now, start):
            state["previous"] = field if start is None else fields[start]
            result = run(config_now, None)
            fields.append(state["previous"] if result.converged else None)
            return result

        return run_leg

    # Sized at the design speed, whatever speeds are swept: its throttle line
    # is what puts a datum on every other speedline. The design point itself,
    # on the open duct, is where the design speed's choke leg departs from.
    design_run = runner(1.0)
    opens = {1.0: (config, state["previous"])}
    if config.chic.nozzle:
        config = size_nozzle(config, design, design_run)

    # Where each speedline departs from, by speed: its datum's field, and the
    # mass flow that passed there
    datums = {1.0: (state["previous"], None)}
    name = chic.setting_name(config)
    points, bracket = [], None

    # Nearest the design speed first, each new speed restarted from the
    # nearest one already run
    for speed in sorted(speeds, key=lambda speed: (abs(speed - 1.0), speed)):
        config_now = chic.at_speed(config, speed)
        run = design_run if speed == 1.0 else runner(speed)

        if speed not in datums:
            datum = _speed_datum(config_now, speed, datums, run, state)
            if datum is None:
                continue
            datums[speed] = datum

        field, mdot = datums[speed]
        if config.chic.stall:
            stall_points, stall_bracket = chic.sweep(config_now, leg(run, field))
            chic.logger.info(f"Speed {speed:.4g} towards stall:")
            chic.logger.info(chic.format_table(stall_points, stall_bracket, name))
            points += stall_points
            if speed == 1.0 or bracket is None:
                bracket = stall_bracket

        if config.chic.choke:
            # Back to the datum, the stall leg's fields being further from
            # choke; with a nozzle, onto the open duct at the datum's flow
            if not config.chic.nozzle:
                config_choke, start = config_now, field
            elif speed in opens:
                config_choke, start = opens[speed]
            else:
                state["previous"] = field
                config_choke = _open_datum(config_now, speed, mdot, run)
                if config_choke is None:
                    continue
                start = state["previous"]
            choke_points, choke_bracket = chic.sweep(
                config_choke, leg(run, start), direction=-1
            )
            choke_name = chic.setting_name(config_choke, direction=-1)
            chic.logger.info(f"Speed {speed:.4g} towards choke:")
            chic.logger.info(chic.format_table(choke_points, choke_bracket, choke_name))
            points += choke_points
            if not config.chic.stall and (speed == 1.0 or bracket is None):
                bracket = choke_bracket

    return points, bracket


def _speed_datum(config, speed, datums, run, state):
    """Run the sized throat at a new `speed`, from the nearest speed run so
    far, and return its field and the mass flow it passed, or None if it
    would not converge."""
    nearest = min(datums, key=lambda done: abs(done - speed))
    state["previous"] = datums[nearest][0]

    if not getattr(config.solver, "n_step_soft", 0):
        chic.logger.warning(
            f"Moving to speed {speed:.4g} from {nearest:.4g} without a soft "
            f"start, solver.n_step_soft being zero."
        )
    chic.logger.info(f"Speed {speed:.4g}: datum from speed {nearest:.4g}.")

    result = run(config, None, soft=True)
    if not result.converged:
        chic.logger.warning(
            f"The sized throat did not converge at speed {speed:.4g}, so that "
            f"speedline is skipped."
        )
        return None

    mdot = float(result.actual.outlet.mdot)
    chic.logger.info(f"Speed {speed:.4g}: datum passed mdot={mdot:.5g} kg/s.")
    return state["previous"], mdot


def _open_datum(config, speed, mdot, run):
    """Throttle the open duct to the `mdot` the sized throat passed at
    `speed`, and return `config` at the pressure it settled at, for a choke
    leg to depart from, or None if it did not settle.

    A throat cannot be opened far enough to choke the machine, the duct
    narrowing before it gets there, so the choke leg is a back pressure. A
    choked throat holds a speedline's datum where a pressure alone may not,
    so the pressure is found by throttling to the flow rather than guessed.
    """
    mdot_design = float(config.design().mean_line.outlet.mdot)
    point = dataclasses.replace(
        config.operating_point, nozzle_area=None, mdot_adjust=mdot / mdot_design - 1.0
    )
    chic.logger.info(
        f"Speed {speed:.4g}: the open duct throttled to the datum's "
        f"mdot={mdot:.5g} kg/s, for the choke leg to depart from."
    )

    result = run(dataclasses.replace(config, operating_point=point), None)
    if not result.converged or result.operating_point is None:
        chic.logger.warning(
            f"The open duct did not settle at the datum's flow at speed "
            f"{speed:.4g}, so that speedline has no choke leg."
        )
        return None
    return dataclasses.replace(config, operating_point=result.operating_point)


def size_nozzle(config, design, run):
    """Return `config` with a choked exit nozzle passing the design's flow.

    Sized one-dimensionally from the stagnation state and mass flow the design
    point reached at the machine outlet, and solved once. A throat with
    boundary layers and a spanwise profile passes a little other than the
    one-dimensional flow, but it is not rescaled to make that up: whatever it
    passes, a choked throat is a point on the design speedline, which is all a
    datum has to be. That throat is the datum the sweep closes from, and its
    field the one it starts from.

    Parameters
    ----------
    config : Config
        The settled design, with a pressure at exit.
    design : Result
        What the design point achieved.
    run : callable
        As :func:`turbigen.chic.sweep` takes it.

    Returns
    -------
    config : Config

    Raises
    ------
    ValueError
        If the design point has no mixed-out mean line to size from, or the
        throat does not converge, there being no datum to sweep from.
    """
    if design.actual is None:
        raise ValueError(
            "Sizing a nozzle needs the mixed-out mean line the design point "
            "reached, and its result holds none."
        )
    target = float(design.actual.outlet.mdot)
    annulus = config.design().annulus
    area = nozzle.throat_fraction(design.actual.outlet, annulus)
    chic.logger.info(
        f"Sizing a choked nozzle to pass mdot={target:.5g} kg/s: a throat of "
        f"{area:.5g} of the outlet area, one-dimensionally."
    )

    result = run(chic.at_nozzle(config, area), None)
    if not result.converged:
        raise ValueError(
            f"The nozzle with a throat of {area:.5g} of the outlet area "
            f"did not converge, so there is no datum to sweep from."
        )
    mdot = float(result.actual.outlet.mdot)
    chic.logger.info(
        f"Throat {area:.5g} passed mdot={mdot:.5g} kg/s, "
        f"{mdot / target - 1.0:+.2%} from the design point."
    )

    return chic.at_nozzle(config, area)
