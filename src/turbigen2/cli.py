"""Command line interface.

See CLI.md in this package for the full plan. Every verb it specifies is
implemented: `design`, `report`, `run`, `iterate` and `batch`.

Two conventions are worth stating.

Everything a run says goes through the logging system, on one stream, to
stderr. Nothing here is meant to be piped -- the artefacts of a run are its
files, and the tables are for a person reading along -- so there is no second
channel to keep in order, and the console and `log_turbigen2.txt` are the same
transcript. Results are ordinary `INFO` records rather than a level of their
own: the existing turbigen CLI emits its tables as *warnings*, so that raising
the level to quieten a run would not also hide them, which leaves a genuine
warning with nothing to distinguish it. There is no `--quiet` here either: the
one place a run is genuinely too loud is `iterate`, which quietens the console
by logger name on its own, and a shell already knows how to redirect.

And a run writes `output.yaml` beside the config it was given. The output
location is therefore never derived and never typed, and an input file is never
overwritten by the run that reads it -- which matters because the file written
is the *resolved* config, every default expanded, and writing that over a
hand-kept file would lose its comments to the safe loader. One directory is one
run. `design` writes nothing at all, ever, so it can be used to experiment
with a design, or driven from a notebook, without leaving anything behind;
everything worth keeping comes from a verb whose output is its point.
"""

import argparse
import contextlib
import dataclasses
import datetime
import logging
import sys
from pathlib import Path
from timeit import default_timer as timer

import yaml

import turbigen
import ember.convergence_history
import ember.yaml_util
from turbigen2 import (
    batch,
    bconds,
    case,
    database,
    guess,
    iterate,
    job,
    mixout,
    node,
    plugins,
    post,
    restart,
)
from turbigen2.config import Config
from turbigen2.result import Result

# The modules in this package log under the turbigen logger, so configure that
# one rather than introducing a second hierarchy for the same distribution.
logger = logging.getLogger("turbigen")

# ember logs its own march -- the convergence line every n_step_log steps, the
# FMG levels, a divergence -- under its own logger, which reaches no handler of
# ours by itself. Configured here, alongside ours, because logging policy
# belongs to the program and not to the library: importing turbigen2 from a
# notebook or another tool leaves ember's logger exactly as that caller set it.
LOGGER_NAMES = ("turbigen", "ember")

run_log = logging.getLogger("turbigen.run")
"""What one run produced: its tables, its verdict, the files it wrote.

Separated from the session's own messages by name rather than by level, so that
`iterate` can quieten the console for a hundred runs while the log file still
records every one of them in full. The package this replaces gets the same
effect by raising the level and emitting its results as *warnings*, which is
why a genuine warning there is indistinguishable from a startup banner.
"""

OUTPUT_NAME = "output.yaml"
"""What a run calls the resolved config and the answer it reached.

Deliberately not the name of anything anyone hands us. An input is therefore
never a candidate for being overwritten, which needs no check and no special
case: the rule holds by construction rather than by inspection.
"""

LOG_NAME = "log_turbigen2.txt"
"""What a run calls its transcript, beside everything else it wrote."""

RESTART_NAME = "restart.npz"
"""What a run calls the flow field it leaves behind, and `--restart` looks for."""

HISTORY_NAME = "conv.cnv"
"""What a run calls its convergence history, written beside the flow field.

ember's own CNV format, which is a pickle: its `to_json` writes three files of
plotting points, drops the residuals and the divergence flag, and has no
reader, so it cannot bring a history back. The consequence of the format is
handled where it is read rather than avoided here --- see `read_history`.
"""


#
# CONFIG OVERRIDES
#
# Reimplemented rather than imported from turbigen.main, which installs a
# sys.excepthook at module scope: importing it would change exception handling
# for the whole process as a side effect.
#


def apply_overrides(data, overrides):
    """Apply ``KEY=VALUE`` overrides in place on the config dict `data`.

    Values are parsed as YAML so that types, lists and mappings all work.
    Applied before the config is built, so a mistyped key is caught by the
    strict unknown-key check rather than being silently merged in.
    """
    for item in overrides:
        key, separator, raw = item.partition("=")
        if not separator:
            raise ValueError(f"override {item!r} is not in KEY=VALUE form")
        node.set_by_path(data, key, yaml.safe_load(raw))


#
# PLUMBING
#


BATCH_PREFIX = "batch_"
"""What a batch of designs is called, before its number."""


def existing_batches(parent):
    """Return the batch directories already under `parent`."""
    return [
        entry
        for entry in Path(parent).glob(f"{BATCH_PREFIX}*")
        if entry.is_dir() and _batch_number(entry) is not None
    ]


def next_batch_dir(parent):
    """Return the batch directory to write next, under `parent`.

    Numbering carries on from the highest that already exists rather than
    counting how many there are, so a deleted batch in the middle does not
    cause the next one to overwrite a later one.

    Where a batch goes is not a choice, for the same reason it is not one for a
    run: it goes beside the datum that generated it. That also makes the layout
    record which datum a batch came from, which nothing else does --- the
    resolved bounds are logged, but the provenance was otherwise yours to
    remember.
    """
    numbers = [_batch_number(entry) for entry in existing_batches(parent)]

    # Zero-padded so the directories sort in creation order, in a shell glob and
    # in a file browser alike. Wider numbers still parse below, so a project
    # that runs past 9999 batches carries on rather than colliding.
    return Path(parent) / f"{BATCH_PREFIX}{max(numbers, default=-1) + 1:04d}"


def _batch_number(entry):
    """Return the number a batch directory carries, or None if it carries none."""
    try:
        return int(entry.name[len(BATCH_PREFIX) :])
    except ValueError:
        # Something else living beside the batches, which is not ours to
        # interpret.
        return None


_HANDLER_TAG = "_turbigen2_handler"

# What the banner said, kept so that a log file opened later in the run starts
# with it too: the version and start time are exactly what someone reading the
# file months afterwards needs, and by then stderr is long gone.
_banner = []


def _add_handler(handler):
    """Attach a handler to every logger we drive, tagged so it can come off."""
    handler.setFormatter(logging.Formatter("%(message)s"))
    setattr(handler, _HANDLER_TAG, True)
    for line in _banner:
        handler.handle(
            logging.LogRecord("turbigen", logging.INFO, __file__, 0, line, None, None)
        )
    for name in LOGGER_NAMES:
        logging.getLogger(name).addHandler(handler)


def setup_logging(verbose):
    """Send everything to stderr, at a level `verbose` sets.

    Handlers this module added are replaced rather than accumulated, so that
    calling main() more than once in a process reconfigures properly.
    logging.basicConfig would not: it is a no-op after the first call, so a
    second invocation would keep writing to the first one's stderr.
    """
    removed = set()
    for name in LOGGER_NAMES:
        target = logging.getLogger(name)
        for handler in list(target.handlers):
            if getattr(handler, _HANDLER_TAG, False):
                target.removeHandler(handler)
                removed.add(handler)
        target.setLevel(logging.DEBUG if verbose else logging.INFO)

    # Closed once every logger has let go, not while one still holds it: the
    # same handler object is shared between them.
    for handler in removed:
        if isinstance(handler, logging.FileHandler):
            handler.close()

    _add_handler(logging.StreamHandler(sys.stderr))


def load_config(config_path, args):
    """Read the config file, apply overrides, and build a Config.

    Discovery is done here rather than through `Config.from_file` because the
    overrides have to be applied to the raw dict, before it is validated.
    """
    config_path = Path(config_path)

    # Designs must be registered before the config is built, so that the type
    # keys it names can be resolved.
    plugins.discover(config_path.parent)

    data = ember.yaml_util.read_yaml(config_path)

    # Dropped before the overrides are applied, so that `-s result.x=1` cannot
    # reach into a previous run's answer. Re-running a case rewrites it anyway.
    data.pop(case.RESULT_KEY, None)

    apply_overrides(data, args.overrides)

    return Config.from_dict(data)


#
# TARGETS
#
# A verb acts on one config file or on many. Many is what a job array and a
# local queue are made of, and it is also the serial loop that would otherwise
# be written in bash around every batch.
#


def targets(args):
    """Return the config files this invocation acts on."""
    return [Path(name) for name in args.CONFIG_YAML]


def check_clobber(args, paths):
    """Raise if this invocation would destroy answers it cannot show you first.

    One target overwrites: you named that directory by naming the file in it,
    and re-running a case after a tweak is the loop this verb exists for.
    Several refuse, because a batch is cluster hours whose loss is discovered a
    day later, and the check costs a `stat` each.
    """
    if len(paths) < 2 or getattr(args, "force", False):
        return

    existing = [
        path for path in paths if (Path(path).resolve().parent / OUTPUT_NAME).is_file()
    ]
    if not existing:
        return

    raise ValueError(
        f"{len(existing)} of {len(paths)} targets have already been run, "
        f"starting with {existing[0].parent / OUTPUT_NAME}. Re-running would "
        "replace their answers; pass --force if that is what you want."
    )


def each(args, one):
    """Run `one(args, config_path)` for every target, or submit them all.

    Returns the worst exit code any of them reached, so a batch reports failure
    if any member failed, and a script driving one need not parse the log.
    """
    paths = targets(args)

    check_clobber(args, paths)

    if getattr(args, "queue", False):
        return submit_targets(args, paths)

    status = 0
    for path in paths:
        if len(paths) > 1:
            logger.info(f"--- {path}")
        status = max(status, one(args, path))

    return status


#
# SUBMISSION
#


def submit(config, paths, verb, options=()):
    """Send every path to the queue `config` names, and run none of them.

    The `job:` section says *how* to submit and `--queue` says *whether*, so
    submission is never implied by a config file. The package this replaces
    submits whenever the key is present, which makes a run re-exec itself and
    obliges every entry point to carry a `--no-job` escape hatch.
    """
    if config.job is None:
        raise ValueError(
            "--queue needs a job: section saying where to submit, as in "
            "'job: {type: slurm, hours: 4}'."
        )

    tasks = [
        job.Task(config=path, name=Path(path).resolve().parent.name) for path in paths
    ]

    config.job.submit(tasks, verb, options)

    return 0


def submit_targets(args, paths):
    """Submit this invocation's own targets, as the verb that was typed."""
    # The queue is read from the first target only. Which one to use is a
    # property of where you are, so a batch whose members disagreed about it
    # would be describing something that cannot happen.
    return submit(load_config(paths[0], args), paths, args.command, task_options(args))


def batch_verb(config):
    """Return the verb a submitted batch should be run as.

    `iterate` when the datum says how to iterate, `run` otherwise. Inferred
    from the section rather than asked for, the same way the depth of a design
    is set by what the config contains --- and the inference matters, because
    a batch submitted as `run` builds an archive `database` reads back as
    empty: a sample must have converged *and* have its errors inside their
    tolerances, which is what iterating is for.

    Logged, so that "why did this iterate" and "why did this not" are both
    answerable from the batch's own log file.
    """
    verb = "iterate" if config.iterate else "run"

    reason = "an iterate: section" if config.iterate else "no iterate: section"
    batch.logger.info(f"Submitting as '{verb}': the datum has {reason}.")

    return verb


def task_options(args):
    """Return the flags a submitted invocation must carry, as command line.

    Everything that changes what a run *does*, and nothing that decides where
    it happens: `--queue` is consumed here, and `--force` has already had its
    effect, since each submitted job is a single target and single targets
    overwrite anyway.

    Rebuilt from the parsed arguments rather than by editing `sys.argv`, so an
    option written any of the ways argparse accepts it comes out in one form.
    """
    options = []

    for override in args.overrides:
        options += ["-s", override]

    if args.verbose:
        options.append("-v")

    restart = getattr(args, "restart", None)
    if restart is True:
        options.append("--restart")
    elif restart:
        options += ["--restart", str(Path(restart).resolve())]

    return options


#
# VERBS
#


def cmd_design(args):
    """Design the machine and report it."""
    return each(args, _design_one)


def _design_one(args, config_path):
    """Design one config file, writing nothing.

    The one verb that is pure, always. Everything up to and including blade
    geometry is computation on numpy arrays, so this is what you run while
    changing a number and watching the tables move, and it must be safe to run
    anywhere without leaving anything behind. Anything worth keeping comes from
    `report`, which is the verb whose output is its point.
    """
    config = load_config(config_path, args)

    # Resolved here too, so that the tables `design` prints describe the same
    # machine `run` would solve. Nothing is written, which is the verb's whole
    # promise: the resolved config is used and discarded.
    config = iterate.resolve(config)

    run_log.info(config.design().to_string())

    return 0


def prepare(config, restart_path=None):
    """Return the resolved config, the machine, and a grid ready to solve.

    Shared by every verb that needs a grid, so there is one definition of
    "ready to solve" rather than one per verb. `report` stops here and `run`
    carries on, which is what makes the grid a report draws the one `run`
    actually solves. Written out twice instead, the two would drift -- which is
    what happened to `turbigen.main`, where the pipeline appears in both
    branches of one `if` and again in ninety-three unreachable lines that no
    longer match either.

    The grid is None when the config has no `mesh:` section. Whether that is an
    error belongs to the verb: `run` cannot proceed without one, while a report
    of a mean-line design is a perfectly good thing to want.

    The config comes back because it may not be the one that went in: knobs
    whose target is a property of the design alone are converged here, so the
    caller has the viscosity that was actually used rather than the guess it
    started from. `solve` archives what this returns, which is what makes an
    `output.yaml` record the design that ran.
    """
    config = iterate.resolve(config)

    # Each stage reports as it finishes rather than the verb reporting them all
    # at the end, so what is read on the way past is in the order it happened:
    # the mean line, then the grid it was meshed onto, then where the flow
    # field in that grid came from.
    machine = config.design()
    run_log.info(machine.to_string())

    if config.mesh is None:
        return config, machine, None

    grid = config.mesh.mesh(machine)
    run_log.info(grid_string(grid))

    bconds.apply(grid, machine)
    guess.apply(grid, machine)

    # A stored field supersedes the meridional guess. Applied after it rather
    # than instead of it, so that a block the restart cannot fill is still
    # left with something sane in it.
    if restart_path is not None:
        restart.apply(grid, restart_path)

    return config, machine, grid


def stored_field(config_path):
    """Return the flow field a previous run left beside `config_path`.

    Found rather than named, because a report of a run that has one always
    wants it, and there is no second thing it could sensibly mean. That is what
    lets the re-plot be the same command as the plot, with no flag between
    them. Nothing raises when there is none: a case that has not been solved
    still has geometry worth drawing.
    """
    field = Path(config_path).resolve().parent / RESTART_NAME

    if not field.is_file():
        return None

    logger.info(f"Using the flow field at {field}")
    return field


def resolve_restart(args, config_path):
    """Return the flow field to start from, if one was asked for.

    Bare `--restart` means the field a run left beside the config it was given,
    which is what makes re-plotting one in place a flag rather than a path to
    type. A named file still wins, so a field can come from anywhere.
    """
    if not args.restart:
        return None

    if args.restart is not True:
        return Path(args.restart)

    restart_path = Path(config_path).resolve().parent / RESTART_NAME
    if not restart_path.is_file():
        raise ValueError(
            f"No {RESTART_NAME} beside {config_path} to restart from. Point at "
            "a config a run has written beside, or name a field file to read."
        )
    return restart_path


def save_history(path, history):
    """Write `history` beside the flow field it belongs to."""
    history.write_cnv(path)
    logger.debug(f"Wrote the convergence history to {path}")


def read_history(path):
    """Return the convergence history at `path`, or None if there is not one.

    A history is a bonus rather than a requirement, so nothing here raises: a
    re-plot without one is the report minus its convergence page, which beats a
    re-plot that refuses to run because a file an older version wrote will not
    unpickle. That is the price of ember's CNV format, paid here rather than by
    the caller.
    """
    if not path.is_file():
        return None

    try:
        return ember.convergence_history.ConvergenceHistory.read_cnv(path)
    except Exception as err:
        logger.warning(f"Could not read the convergence history at {path}: {err}")
        return None


def cmd_report(args):
    """Describe a case as fully as it can be described, and draw it."""
    return each(args, _report_one)


def _report_one(args, config_path):
    """Draw one config file, using whatever a run has already left beside it.

    Everything the case supports and nothing it does not: a mean line alone
    gives the geometry pages, a `mesh:` section adds the grid, and a flow field
    left by a previous run turns those into a picture of the solution. Each
    standard processor draws nothing when what it needs is absent, so there is
    no mode to select and no flag to remember.

    Re-plotting is therefore the same command as plotting, and re-meshing to
    get there costs seconds against the minutes of the march it stands in for
    -- which is why the grid is not worth serialising.
    """
    with logging_into(args, config_path) as out_dir:
        config = load_config(config_path, args)

        config, machine, grid = prepare(config, stored_field(config_path))

        # Looked for whether or not there is a field to go with it: a history
        # beside the config means a run happened here, and its convergence page
        # is worth drawing either way.
        history = read_history(config_path.parent / HISTORY_NAME)

        result = Result(machine=machine, grid=grid, history=history)

        write_report(config, result, out_dir)

    return 0


def solve(config, out_dir, restart_path=None):
    """Design, mesh and solve `config`, writing everything into `out_dir`.

    The whole of a run, so that `iterate` composes runs rather than writing a
    second copy of one -- which is how `turbigen.main` came to hold the same
    pipeline three times over, two of them unreachable and already drifted.
    """
    config, machine, grid = prepare(config, restart_path)

    if grid is None:
        raise ValueError("The 'run' command needs a mesh: section in the config file.")

    history = config.solver.solve(grid)
    converged = config.solver.converged(history)

    # Reduce the solution to a mean line. A diverged grid has nothing to mix
    # out, and even a converged one can refuse, so this must not cost the run
    # the output it has already earned.
    actual = None
    try:
        actual = mixout.mean_line(grid, machine)
    except Exception as err:
        logger.warning(f"Could not mix out the solution: {err}")

    result = Result(
        machine=machine,
        grid=grid,
        actual=actual,
        converged=converged,
        history=history,
    )

    # Measured whether or not anything is iterating: the exit angle a row
    # achieved and the incidence its leading edge saw are observations of the
    # flow, and they can only be taken while the grid is in memory.
    result = dataclasses.replace(result, error=iterate.errors(config, result))

    run_log.info(convergence_string(history, converged))
    if actual is not None:
        run_log.info(actual.to_string())

    # Written whatever happened, and written first. A march that did not
    # converge is the one most likely to be picked up and continued, so
    # withholding its field would be exactly backwards -- and a post-processor
    # that raises must not be able to discard a solution the CFD has already
    # been paid for.
    restart_path = out_dir / RESTART_NAME
    restart.save(restart_path, grid)
    run_log.info(f"Wrote the flow field to {restart_path}")

    # Beside the field, and for the same reason: it is what a re-plot needs to
    # draw the convergence page, and it costs a few kilobytes.
    save_history(out_dir / HISTORY_NAME, history)

    _write_output(config, result, out_dir)

    return result


def cmd_run(args):
    """Design, mesh and solve, then report."""
    return each(args, _run_one)


def _run_one(args, config_path):
    """Solve one config file, writing everything beside it."""
    with logging_into(args, config_path) as out_dir:
        config = load_config(config_path, args)

        if config.solver is None:
            raise ValueError(
                "The 'run' command needs a solver: section in the config file."
            )

        result = solve(config, out_dir, resolve_restart(args, config_path))

    # Non-zero on a failed solve, so a script driving a sweep can tell without
    # parsing the log. Everything written above is still written: a diverged
    # run is exactly the one whose output someone needs to look at.
    return 0 if result.converged else 2


def cmd_iterate(args):
    """Run repeatedly, moving the design onto its own solution."""
    # Once for the invocation rather than once per target: the filter is added
    # to the console handler, and adding it again for every config in a batch
    # would stack a dozen copies of the same test.
    if not args.verbose and not args.queue:
        _quieten_the_runs()

    return each(args, _iterate_one)


def _iterate_one(args, config_path):
    """Iterate one config file, keeping every iteration beside it."""
    with logging_into(args, config_path) as out_dir:
        config = load_config(config_path, args)

        if config.solver is None:
            raise ValueError(
                "The 'iterate' command needs a solver: section in the config file."
            )
        if not config.iterate:
            raise ValueError(
                "The 'iterate' command needs an iterate: section saying what to "
                "correct; without one, use 'run'."
            )

        previous = resolve_restart(args, config_path)

        # Iteration -1: where the knobs start. Anchored on the config file's
        # own directory, because a config is often run from somewhere else, and
        # excluding that same directory because it is where this run's own
        # iterations will land -- one directory being one run, nothing else of
        # anyone's is in there to lose.
        config = database.warm_start(config, out_dir, exclude=(out_dir,))

        def run(config_now, i_iter):
            """Solve one iteration into a directory of its own."""
            nonlocal previous

            iter_dir = out_dir / f"iter_{i_iter:04d}"
            iter_dir.mkdir(parents=True, exist_ok=True)
            iterate.logger.info(f"Iteration {i_iter} in {iter_dir}")

            # Chained: each iteration starts from the field the last one
            # reached, which is most of the saving. Index-space interpolation
            # covers the mesh moving with the design.
            result = solve(config_now, iter_dir, previous)
            previous = iter_dir / RESTART_NAME

            return result

        config, result, converged = iterate.converge(config, run, config.max_iter)

        # The answer, on a console that has been shown only the iteration
        # table. Not for a march that blew up, whose mixed-out mean line is
        # whatever its NaNs averaged to and would read as a result.
        if result is not None and result.converged and result.actual is not None:
            iterate.logger.info(result.actual.to_string())

        _link_final(out_dir, previous.parent)

    return 0 if converged else 2


def cmd_batch(args):
    """Write configs covering the design space, ready to be run."""
    paths = targets(args)
    if len(paths) > 1:
        raise ValueError(
            "The 'batch' command covers one design space, so it takes one "
            "config file as its datum."
        )

    config = load_config(paths[0], args)

    if config.batch is None:
        raise ValueError(
            "The 'batch' command needs a batch: section saying which design "
            "variables to vary, and between what bounds or at what values."
        )
    # Checked before anything is created, so a misspelled variable does not
    # leave an empty batch behind and burn a number on its way out.
    config.batch.check(config)
    _check_grid_options(args, config.batch)

    # Scanned before the new batch directory exists, so it cannot count itself.
    # A batch is never written into, only beside, so nothing can be lost.
    datum_dir = paths[0].resolve().parent

    start = 0
    if args.carry_on:
        start = batch.next_index(existing_batches(datum_dir))
        batch.logger.info(f"Carrying on from index {start}.")

    out_dir = _open_batch(args, datum_dir)

    members = []
    for index, member in batch.generate(config, args.number, start):
        member_path = out_dir / batch.member_name(index)
        # A member is a directory, because one directory is one run: it is what
        # gives every member an `output.yaml` of its own to be run into.
        member_path.parent.mkdir(parents=True, exist_ok=True)
        member.to_file(member_path)
        members.append(member_path)

    batch.logger.info(f"Wrote {len(members)} design(s) to {out_dir}")

    if args.queue:
        submit(config, members, batch_verb(config))

    # The one thing this verb puts on stdout, everything else being on stderr.
    # A numbered batch cannot be named in advance, so without it a script has
    # no way to find what it just made: BATCH=$(turbigen2 batch case.yaml).
    print(out_dir, flush=True)

    return 0


def _check_grid_options(args, spec):
    """Refuse the options a grid of named values cannot honour.

    Both are properties of a *sequence*, and a grid is not one: its count is
    the product of what it names, and a finite product has no tail to carry on
    from. Refused rather than ignored, and refused here rather than inside
    `generate`, so that a batch number is not burned before the complaint.
    """
    if not spec.is_grid():
        return

    if args.number is not None:
        raise ValueError(
            "A batch: section with values: runs every combination of them, so "
            "there is no -n to choose. Use bounds: to draw a chosen number of "
            "designs from a box."
        )

    if args.carry_on:
        raise ValueError(
            "A batch: section with values: is already the whole grid, so there "
            "is nothing to --continue. Widen values: and write another batch."
        )


def _quieten_the_runs():
    """Keep the per-run tables and the march off the console while iterating.

    An iterate is tens of runs, and printing each one in full buries the few
    lines that describe the iteration itself. Filtered by logger name on the
    console handler alone, so `log_turbigen2.txt` still holds every run
    complete --- where the package this replaces raises the level instead, and
    loses the detail from its log file as well as from the screen.

    Warnings and errors are never filtered, whoever emits them.
    """

    def quiet(record):
        if record.levelno >= logging.WARNING:
            return True
        return not record.name.startswith(("turbigen.run", "ember"))

    for handler in _console_handlers():
        handler.addFilter(quiet)


def _console_handlers():
    """Return the stderr handlers this module attached, without the log file."""
    seen = {}
    for name in LOGGER_NAMES:
        for handler in logging.getLogger(name).handlers:
            if getattr(handler, _HANDLER_TAG, False) and not isinstance(
                handler, logging.FileHandler
            ):
                seen[id(handler)] = handler
    return list(seen.values())


def _link_final(out_dir, iter_dir):
    """Point `final` at the last iteration, and `output.yaml` at its answer.

    Symlinks rather than copies of the answer: the iterations are the record of
    how the design got where it did, and duplicating megabytes to name one of
    them would only invite the two to disagree.

    The second link is what makes `output.yaml` mean "what this run achieved"
    whichever verb produced it, so a database glob and a script reading a
    result need not know whether a design took one solve or six.
    """
    _link(out_dir / "final", iter_dir.name, directory=True)
    _link(out_dir / OUTPUT_NAME, f"final/{OUTPUT_NAME}", directory=False)


def _link(link, target, directory):
    """Point `link` at `target`, replacing whatever was there."""
    try:
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(target, target_is_directory=directory)
    except OSError as err:
        # A filesystem without symlinks is a reason to say so, not to lose a
        # finished set of iterations.
        logger.warning(f"Could not link {link} to {target}: {err}")
        return

    logger.info(f"Linked {link} to {target}")


def convergence_string(history, converged):
    """Report how a march ended, using ember's own summary of the last record.

    The verdict is ours; the numbers underneath it are ember's, because a
    history knows how to describe itself and a second formatter here would be
    one more thing to keep in step. Note that no step count is quoted: records
    are written every `n_step_log` steps, so the last record is not in general
    the last step marched, and reporting it as one would be wrong.
    """
    verdict = "converged" if converged else "NOT converged"
    return f"Solver: {verdict}\n{history.format_message()}"


@contextlib.contextmanager
def logging_into(args, config_path):
    """Yield where this verb writes for `config_path`, teeing the log into it.

    Beside the config, always: the location is never derived and never typed,
    and a verb that writes has nowhere else it could sensibly put things.

    The handler comes off again at the end, so a batch of targets leaves one
    complete transcript in each of their directories rather than every run
    after the first appending to the first one's file.
    """
    out_dir = Path(config_path).resolve().parent

    # Recorded on the arguments so that main() can name it again at the end,
    # where it is most use: a long run scrolls its first line out of sight.
    args.out_dir = out_dir

    handler = logging.FileHandler(out_dir / LOG_NAME)
    _add_handler(handler)
    logger.info(f"Output directory: {out_dir}")

    try:
        yield out_dir
    finally:
        for name in LOGGER_NAMES:
            logging.getLogger(name).removeHandler(handler)
        handler.close()


def _open_batch(args, datum_dir):
    """Create and return the directory a batch of designs is written into.

    Numbered rather than named, because a batch is many designs and hours of
    solving to come: writing into an existing one would destroy work, where a
    single run written over a single run is a recoverable mistake.
    """
    out_dir = next_batch_dir(datum_dir)
    out_dir.mkdir(parents=True)
    args.out_dir = out_dir
    _add_handler(logging.FileHandler(out_dir / LOG_NAME))
    logger.info(f"Output directory: {out_dir}")
    return out_dir


def _write_output(config, result, out_dir):
    """Write what a run achieved, and draw it.

    Only the verbs that solve call this, and that is what makes it safe:
    `output.yaml` is written by whoever has a real answer to put in it, so no
    verb can replace a converged run's `result:` with an empty one of its own.
    An earlier arrangement had `mesh --restart --write` writing back over the
    config it had just read, guarded by comparing the two -- a guard that only
    existed because the wrong verb was writing.
    """
    config_path = out_dir / OUTPUT_NAME
    case.write(config_path, config, result)
    run_log.info(f"Wrote resolved configuration to {config_path}")

    write_report(config, result, out_dir)


def grid_string(grid):
    """One-line summary of the size of a grid."""
    return f"Mesh: n_block={len(grid)}, n_cell/1e6={grid.size / 1e6:.2f}"


def processors(config):
    """Return the post-processors to run, standard set first.

    A configured entry *replaces* the standard plot of its own type rather than
    adding to it, so naming one in a config tunes it instead of producing two
    of them. That was the rule in the package this replaces as well, but it got
    there by inserting into the user's own list from `__post_init__`, so the
    config that ran was not the config that was written. Nothing is mutated
    here: the standard set is a property of the report, and `post_process`
    means what it says.
    """
    configured = {p.type for p in config.post_process}
    standard = [p for p in post.STANDARD if p.type not in configured]
    return standard + list(config.post_process)


def write_report(config, result, out_dir):
    """Run the post-processors and collect their figures into one PDF.

    Nothing is produced without an output directory, so the figures are only
    made when there is somewhere to put them. With one, a report is always
    written: the standard plots cost a fraction of a solve, and a run whose
    output nobody looks at is worse than a page nobody needed.
    """
    # Imported here so that the CLI does not pay for matplotlib until there is
    # something to plot.
    import matplotlib  # noqa: PLC0415

    # Only claim the backend if nothing has chosen one yet. pyplot in sys.modules
    # means a caller -- a notebook driving main(), say -- is already plotting,
    # and switching it out from under them would be rude.
    if "matplotlib.pyplot" not in sys.modules:
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt  # noqa: PLC0415
    from matplotlib.backends.backend_pdf import PdfPages  # noqa: PLC0415

    path = out_dir / "post.pdf"
    n_page = 0
    with PdfPages(path) as pdf:
        for processor in processors(config):
            logger.debug(f"Running post-processor {processor}")
            # Figures are closed as they are written rather than collected and
            # closed at the end, so a long report holds one at a time.
            for figure in processor.report(config, result):
                pdf.savefig(figure)
                plt.close(figure)
                n_page += 1

    # Nothing to draw is normal -- a mean-line design gives every standard plot
    # nothing to work with -- and matplotlib writes no file for an empty
    # document, so there is no report to announce and none to find.
    if not n_page:
        run_log.info("No figures were produced, so no report was written.")
        return None

    run_log.info(f"Wrote report to {path}")
    return path


#
# ENTRY POINT
#


def _make_parser():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "CONFIG_YAML",
        nargs="+",
        help=(
            "one or more configuration files in yaml format; several are run "
            "one after another, or submitted together with --queue"
        ),
    )
    common.add_argument(
        "-s",
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "override a config value; the value is parsed as YAML and the key "
            "is dotted, with integer segments indexing into lists, e.g. "
            "-s mean_line.psi=1.8 (repeatable)"
        ),
    )
    common.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="output more diagnostic information on stderr",
    )
    parser = argparse.ArgumentParser(
        prog="turbigen2",
        description=(
            "turbigen2 is an experimental rebuild of the turbigen design "
            "system. Each command carries the design one stage further through "
            "the pipeline."
        ),
    )
    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version=f"%(prog)s {turbigen.__version__}",
    )

    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")
    design = commands.add_parser(
        "design",
        parents=[common],
        help="design the mean line and print it, writing nothing",
        description=(
            "Design the mean line and geometry from a configuration file and "
            "print them. Nothing is ever written, so this is what to run while "
            "changing a number and watching the tables move. Use 'report' for "
            "figures."
        ),
    )
    design.set_defaults(func=cmd_design)

    report = commands.add_parser(
        "report",
        parents=[common],
        help="draw a case, using whatever a run has left beside it",
        description=(
            "Design the machine, mesh it if the config says how, pick up any "
            f"{RESTART_NAME} a previous run left beside the config, and write "
            "post.pdf. Each standard plot draws nothing when what it needs is "
            "absent, so a mean-line design gives the geometry pages and a "
            "solved case gives the flow. Re-plotting a finished run is "
            "therefore the same command, with no flag between them. The grid "
            "itself is never written, because how a mesh is serialised is a "
            "property of the solver that will read it."
        ),
    )
    report.set_defaults(func=cmd_report)

    run = commands.add_parser(
        "run",
        parents=[common],
        help="design, mesh and solve, then report",
        description=(
            "Design the machine from a configuration file, mesh it, apply "
            "boundary conditions and an initial guess, and solve. Everything "
            f"is written beside the config, in {OUTPUT_NAME} and its "
            "companions. Exits 2 if the solver did not converge, having "
            "written its output anyway."
        ),
    )
    _add_force_argument(run)
    _add_queue_argument(run)
    _add_restart_argument(run)
    run.set_defaults(func=cmd_run)

    iterate_ = commands.add_parser(
        "iterate",
        parents=[common],
        help="run repeatedly, moving the design onto its own solution",
        description=(
            "Solve the machine, measure how far its design is from what the "
            "flow actually did, correct the design, and solve again. Each "
            "iteration is an ordinary run in a directory of its own beside the "
            "config, with 'final' linked to the last and every iteration kept. "
            "Needs an iterate: section. Exits 2 if the design had not "
            "converged by max_iter."
        ),
    )
    _add_force_argument(iterate_)
    _add_queue_argument(iterate_)
    _add_restart_argument(iterate_)
    iterate_.set_defaults(func=cmd_iterate)

    batch_ = commands.add_parser(
        "batch",
        parents=[common],
        help="write configs covering a design space, ready to be run",
        description=(
            "Write one config per design over the design variables the batch: "
            "section names. With bounds:, designs are drawn from a Sobol' "
            "sequence over the box; with values:, the batch is every "
            "combination of the values named, which is the parameter study a "
            "shell loop over --set cannot write. Points that cannot be "
            "designed are skipped, so no cluster time is spent finding that "
            "out. Nothing is run unless --queue asks for it. The batch is "
            "written beside the datum config, in the next free batch_NNNN, "
            "whose path is printed on stdout."
        ),
    )
    batch_.add_argument(
        "-n",
        "--number",
        type=int,
        default=None,
        metavar="N",
        help=(
            f"how many designs to draw from bounds: (default "
            f"{batch.DEFAULT_NUMBER}; Sobol' balance holds at powers of two). "
            "Not for values:, whose count is the product of what it names"
        ),
    )
    _add_queue_argument(batch_)
    batch_.add_argument(
        "--continue",
        dest="carry_on",
        action="store_true",
        help=(
            "extend the batches already beside the datum config, starting "
            "after the highest member index they hold; bounds: only"
        ),
    )
    batch_.set_defaults(func=cmd_batch)

    return parser


def _add_force_argument(parser):
    """Add --force, which allows several targets to overwrite their answers."""
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            f"overwrite an existing {OUTPUT_NAME} when several config files "
            "are given; one config file on its own always overwrites"
        ),
    )


def _add_queue_argument(parser):
    """Add --queue, for the verbs that cost enough time to be worth queueing.

    A flag rather than a config key deciding it, so submission is never implied
    by a file: the key says how, this says whether. That is what keeps the
    recursion one level deep and needs no `--no-job` to break it.
    """
    parser.add_argument(
        "-Q",
        "--queue",
        action="store_true",
        help=(
            "submit to the queue named by the job: section instead of running "
            "here; every config becomes one submission"
        ),
    )


def _add_restart_argument(parser):
    """Add --restart, which every verb that builds a grid can use.

    Not on the common parent, because `design` never makes a grid to put a
    field on and should not advertise a flag it would ignore.
    """
    parser.add_argument(
        "--restart",
        metavar="NPZ",
        nargs="?",
        const=True,
        help=(
            "load the flow field in NPZ, as written by a previous run, "
            "instead of the meridional guess; interpolated in index space if "
            "the mesh resolution has changed. With no NPZ given, reads "
            f"{RESTART_NAME} from beside the config file, which re-plots a run "
            "in place"
        ),
    )


def _format_elapsed(seconds):
    """Human-readable elapsed time, in whichever unit reads best."""
    if seconds < 60.0:
        return f"{seconds:.1f} s"
    return f"{seconds / 60.0:.2f} min"


def main(argv=None):
    """Parse arguments and run the requested command."""
    args = _make_parser().parse_args(argv)
    setup_logging(args.verbose)

    # The banner says which code ran and when, so that a log file kept beside a
    # set of results still identifies them long after the run.
    started = datetime.datetime.now().replace(microsecond=0).isoformat()
    _banner[:] = [
        f"*** TURBIGEN2 v{turbigen.__version__} ***",
        f"Starting at {started}",
    ]
    for line in _banner:
        logger.info(line)
    start_tic = timer()

    try:
        return args.func(args)
    except Exception as err:
        # A user error in a config file should read as a message, not a stack
        # trace. The traceback is one -v away when it is actually wanted.
        if args.verbose:
            logger.exception("Error encountered, quitting...")
        else:
            logger.error(f"{type(err).__name__}: {err}")
        return 1
    finally:
        # In a finally block so that a run which fell over still reports how
        # long it took to get there, and where it left what it had written.
        logger.info(f"Total time: {_format_elapsed(timer() - start_tic)}")
        if out_dir := getattr(args, "out_dir", None):
            logger.info(f"Output directory was: {out_dir}")


if __name__ == "__main__":
    sys.exit(main())
