"""Command line interface.

See CLI.md in this package for the full plan. The `design` and `mesh` verbs are
implemented; `run` and `iterate` are specified there.

Two conventions are worth stating.

Everything a run says goes through the logging system, on one stream, to
stderr. Nothing here is meant to be piped -- the artefacts of a run are its
files, and the tables are for a person reading along -- so there is no second
channel to keep in order, and the console and `log_turbigen2.txt` are the same
transcript. Results are ordinary `INFO` records rather than a level of their
own: the existing turbigen CLI emits its tables as *warnings*, so that raising
the level to quieten a run would not also hide them, which leaves a genuine
warning with nothing to distinguish it. Here `--quiet` raises the level of the
console handler alone, so the log file records a quiet run in full.

And an output directory is a property of the verb rather than of the config
file: `design` writes nothing at all unless asked to, so it can be used to
experiment with a design, or driven from a notebook, without leaving anything
behind.
"""

import argparse
import datetime
import logging
import sys
from pathlib import Path
from timeit import default_timer as timer

import yaml

import turbigen
import turbigen.util
import ember.yaml_util
from turbigen2 import bconds, case, guess, mixout, plugins, post, restart
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


#
# CONFIG OVERRIDES
#
# Reimplemented rather than imported from turbigen.main, which installs a
# sys.excepthook at module scope: importing it would change exception handling
# for the whole process as a side effect.
#


def _segment(text):
    """Parse a dotted-key segment; integer-like segments index into lists."""
    try:
        return int(text)
    except ValueError:
        return text


def _get_child(node, segment):
    """Return node[segment] if present, else None, tolerating type mismatches."""
    if isinstance(segment, int):
        if isinstance(node, list) and segment < len(node):
            return node[segment]
        return None
    return node.get(segment) if isinstance(node, dict) else None


def _set_child(node, segment, value):
    """Assign node[segment] = value, growing lists with None as needed."""
    if isinstance(segment, int):
        if not isinstance(node, list):
            raise ValueError(f"cannot index non-list with integer key {segment}")
        node.extend([None] * (segment + 1 - len(node)))
        node[segment] = value
    else:
        if not isinstance(node, dict):
            raise ValueError(f"cannot set string key {segment!r} on non-mapping")
        node[segment] = value


def apply_overrides(data, overrides):
    """Apply ``KEY=VALUE`` overrides in place on the config dict `data`.

    Keys are dotted paths, integer segments index into lists, and values are
    parsed as YAML so that types, lists and mappings all work. Applied before
    the config is built, so a mistyped key is caught by the strict unknown-key
    check rather than being silently merged in.
    """
    for item in overrides:
        key, separator, raw = item.partition("=")
        if not separator:
            raise ValueError(f"override {item!r} is not in KEY=VALUE form")
        value = yaml.safe_load(raw)
        segments = [_segment(s) for s in key.split(".")]
        node = data
        for segment, following in zip(segments[:-1], segments[1:]):
            child = _get_child(node, segment)
            if not isinstance(child, (dict, list)):
                child = [] if isinstance(following, int) else {}
                _set_child(node, segment, child)
            node = child
        _set_child(node, segments[-1], value)


#
# PLUMBING
#


def resolve_out_dir(spec):
    """Create and return the output directory named by `spec`.

    A ``*`` in the name is replaced by the next free number, so ``run_*`` gives
    ``run_0``, ``run_1`` and so on.
    """
    if "*" in str(spec):
        spec = turbigen.util.next_numbered_dir(str(spec))
    out_dir = Path(spec).absolute()
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


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


def setup_logging(verbose, quiet=False):
    """Send everything to stderr, at a level `verbose` and `quiet` set.

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

    console = logging.StreamHandler(sys.stderr)
    if quiet:
        # Quiet is about the console, not about the record. The level goes on
        # the handler rather than the logger, so a file handler opened later
        # still writes the tables, and `-q -o` leaves a complete log behind.
        console.setLevel(logging.WARNING)
    _add_handler(console)


def load_config(args):
    """Read the config file, apply overrides, and build a Config.

    Discovery is done here rather than through `Config.from_file` because the
    overrides have to be applied to the raw dict, before it is validated.
    """
    config_path = Path(args.CONFIG_YAML)

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
# VERBS
#


def cmd_design(args):
    """Design the machine and report it."""
    config = load_config(args)
    out_dir = _open_output(args)

    machine = config.design()
    result = Result(machine=machine)

    logger.info(machine.to_string())

    _write_output(config, result, out_dir)
    return 0


def prepare(config, restart_path=None):
    """Return the machine and a grid ready to solve.

    Shared by every verb that needs a grid, so there is one definition of
    "ready to solve" rather than one per verb. `mesh` stops here and `run`
    carries on, which is what makes plotting the output of `mesh` show the grid
    `run` would actually solve. Written out twice instead, the two would drift
    -- which is what happened to `turbigen.main`, where the pipeline appears in
    both branches of one `if` and again in ninety-three unreachable lines that
    no longer match either.
    """
    if config.mesh is None:
        raise ValueError("This command needs a mesh: section in the config file.")

    # Each stage reports as it finishes rather than the verb reporting them all
    # at the end, so what is read on the way past is in the order it happened:
    # the mean line, then the grid it was meshed onto, then where the flow
    # field in that grid came from.
    machine = config.design()
    logger.info(machine.to_string())

    grid = config.mesh.mesh(machine)
    logger.info(grid_string(grid))

    bconds.apply(grid, machine)
    guess.apply(grid, machine)

    # A stored field supersedes the meridional guess. Applied after it rather
    # than instead of it, so that a block the restart cannot fill is still
    # left with something sane in it.
    if restart_path is not None:
        restart.apply(grid, restart_path)

    return machine, grid


def cmd_mesh(args):
    """Design the machine, mesh it, and report both."""
    config = load_config(args)
    out_dir = _open_output(args)

    machine, grid = prepare(config)

    result = Result(machine=machine, grid=grid)

    _write_output(config, result, out_dir)
    return 0


def cmd_run(args):
    """Design, mesh and solve, then report."""
    config = load_config(args)

    if config.solver is None:
        raise ValueError(
            "The 'run' command needs a solver: section in the config file."
        )
    if not args.out:
        raise ValueError("The 'run' command writes results, so it needs --out.")

    out_dir = _open_output(args)

    machine, grid = prepare(config, args.restart)

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

    logger.info(convergence_string(history, converged))
    if actual is not None:
        logger.info(actual.to_string())

    # Written whatever happened, and written first. A march that did not
    # converge is the one most likely to be picked up and continued, so
    # withholding its field would be exactly backwards -- and a post-processor
    # that raises must not be able to discard a solution the CFD has already
    # been paid for.
    restart_path = out_dir / "restart.npz"
    restart.save(restart_path, grid)
    logger.info(f"Wrote the flow field to {restart_path}")

    _write_output(config, result, out_dir)

    # Non-zero on a failed solve, so a script driving a sweep can tell without
    # parsing the log. Everything written above is still written: a diverged
    # run is exactly the one whose output someone needs to look at.
    return 0 if converged else 2


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


def _open_output(args):
    """Return the output directory, if one was asked for, logging into it."""
    if not args.out:
        return None
    out_dir = resolve_out_dir(args.out)
    # Recorded on the arguments so that main() can name it again at the end,
    # where it is most use: a long run scrolls its first line out of sight.
    args.out_dir = out_dir
    _add_handler(logging.FileHandler(out_dir / "log_turbigen2.txt"))
    logger.info(f"Output directory: {out_dir}")
    return out_dir


def _write_output(config, result, out_dir):
    """Write the resolved config and the report, if there is anywhere to."""
    if out_dir is None:
        # Only mentioned when the config asked for post-processing of its own:
        # that is a request left unmet. Nobody asked for the standard plots, so
        # their absence from a run that writes nothing is not news.
        if config.post_process:
            logger.info("No output directory given, so no report was written.")
        return

    config_path = out_dir / "config.yaml"
    case.write(config_path, config, result)
    logger.info(f"Wrote resolved configuration to {config_path}")
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
        logger.info("No figures were produced, so no report was written.")
        return None

    logger.info(f"Wrote report to {path}")
    return path


#
# ENTRY POINT
#


def _make_parser():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "CONFIG_YAML", help="filename of configuration data in yaml format"
    )
    common.add_argument(
        "-o",
        "--out",
        metavar="DIR",
        help=(
            "write results to DIR, creating it if needed; a '*' is replaced by "
            "the next free number, as in run_* -> run_0. Without this, nothing "
            "is written"
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
    common.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help=(
            "show only warnings and errors on the console; a log file written "
            "under --out still records the run in full"
        ),
    )

    parser = argparse.ArgumentParser(
        prog="turbigen2",
        description=(
            "turbigen2 is an experimental rebuild of the turbigen design "
            "system. Each command carries the design one stage further through "
            "the pipeline; 'design' and 'mesh' are implemented so far."
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
        help="design the mean line and report it",
        description=(
            "Design the mean line from a configuration file and print it. "
            "Nothing is written and no directory is created unless --out is "
            "given."
        ),
    )
    design.set_defaults(func=cmd_design)

    mesh = commands.add_parser(
        "mesh",
        parents=[common],
        help="design the machine, generate a grid, and report both",
        description=(
            "Design the machine from a configuration file, mesh it, and print "
            "the result. Nothing is written and no directory is created unless "
            "--out is given; the grid itself is not written, because how a mesh "
            "is serialised is a property of the solver that will read it."
        ),
    )
    mesh.set_defaults(func=cmd_mesh)

    run = commands.add_parser(
        "run",
        parents=[common],
        help="design, mesh and solve, then report",
        description=(
            "Design the machine from a configuration file, mesh it, apply "
            "boundary conditions and an initial guess, and solve. Requires "
            "--out, because a run produces artefacts worth keeping. Exits 2 if "
            "the solver did not converge, having written its output anyway."
        ),
    )
    run.add_argument(
        "--restart",
        metavar="NPZ",
        help=(
            "start from the flow field in NPZ, as written by a previous run, "
            "instead of the meridional guess; interpolated in index space if "
            "the mesh resolution has changed"
        ),
    )
    run.set_defaults(func=cmd_run)

    return parser


def _format_elapsed(seconds):
    """Human-readable elapsed time, in whichever unit reads best."""
    if seconds < 60.0:
        return f"{seconds:.1f} s"
    return f"{seconds / 60.0:.2f} min"


def main(argv=None):
    """Parse arguments and run the requested command."""
    args = _make_parser().parse_args(argv)
    setup_logging(args.verbose, args.quiet)

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
