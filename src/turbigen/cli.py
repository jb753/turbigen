"""Command line interface.

See CLI.md in this package for the full plan. Every verb it specifies is
implemented: `design`, `report`, `run`, `iterate` and `batch`.

Two conventions are worth stating.

Everything a run says goes through the logging system, on one stream, to
stderr. Nothing here is meant to be piped -- the artefacts of a run are its
files, and the tables are for a person reading along -- so there is no second
channel to keep in order, and the console and `log_turbigen.txt` are the same
transcript. Results are ordinary `INFO` records rather than a level of their
own: the existing turbigen CLI emits its tables as *warnings*, so that raising
the level to quieten a run would not also hide them, which leaves a genuine
warning with nothing to distinguish it. There is no `--quiet` here either: the
one place a run is genuinely too loud is `iterate`, which quietens the console
by logger name on its own, and a shell already knows how to redirect.

And a run writes `output.yaml` beside the config it was given. The output
location is therefore never derived, and an input file is never overwritten by
the run that reads it -- which matters because the file written is the
*resolved* config, every default expanded, and writing that over a hand-kept
file would lose its comments to the safe loader. One directory is one run.

`-o` moves the whole directory rather than splitting it: the config is copied
into the workdir and the run happens there, so config and output stay together
and nothing downstream has to know. It is also what keeps replacing an answer a
rare enough thing to be worth refusing -- a variant goes somewhere new, and
`-f` is for the case where you really did mean to write over what is there.

`design` writes nothing at all, ever, so it can be used to experiment with a
design, or driven from a notebook, without leaving anything behind; everything
worth keeping comes from a verb whose output is its point.
"""

import argparse
import contextlib
import dataclasses
import datetime
import logging
import sys
from pathlib import Path
from timeit import default_timer as timer

import numpy as np
import yaml

import turbigen
import ember.convergence_history
import ember.yaml_util
from turbigen import (
    batch,
    bconds,
    case,
    chic,
    database,
    guess,
    include,
    iterate,
    job,
    mixout,
    node,
    plugins,
    post,
    restart,
)
from turbigen.config import Config
from turbigen.result import Result

# The modules in this package log under the turbigen logger, so configure that
# one rather than introducing a second hierarchy for the same distribution.
logger = logging.getLogger("turbigen")

# ember logs its own march -- the convergence line every n_step_log steps, the
# FMG levels, a divergence -- under its own logger, which reaches no handler of
# ours by itself. Configured here, alongside ours, because logging policy
# belongs to the program and not to the library: importing turbigen from a
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

Deliberately not the name of anything anyone hands us, and now enforced rather
than merely conventional: `check_not_output` refuses it as a target in every
verb. An input is therefore never a candidate for being overwritten, because a
verb cannot overwrite a file it will not read.

The check earns its keep now that `report` writes here too. While only the
solving verbs did, the rule held by construction --- nothing read an
`output.yaml` and wrote one back. `report` reads a config and writes one
beside it, so without the refusal it would write over its own input.
"""

LOG_NAME = "log_turbigen.txt"
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

PLACEHOLDER = "%"
"""Where a number goes in a directory name `-o` was given.

One spelling, not two. The package this replaces accepted `%` and `*` for
overlapping jobs, and which of them numbered a run was a thing to remember
rather than to work out.
"""

DIGITS = 4
"""How wide a directory number is written.

Zero-padded so directories sort in creation order, in a shell glob and in a
file browser alike. Wider numbers still parse, so a project that runs past
9999 carries on rather than colliding.
"""


def numbered_dirs(parent, head, tail=""):
    """Return the directories under `parent` named `head` + digits + `tail`."""
    return [
        entry
        for entry in Path(parent).glob(f"{head}*{tail}")
        if entry.is_dir() and dir_number(entry, head, tail) is not None
    ]


def next_numbered_dir(parent, head, tail=""):
    """Return the next free `head`NNNN`tail` under `parent`.

    Numbering carries on from the highest that already exists rather than
    counting how many there are, so a deleted directory in the middle does not
    cause the next one to overwrite a later one. That property is why this is
    one function and not two: `batch` needs it so a lost batch cannot take a
    later batch's number, and `-o` needs it for exactly the same reason.
    """
    numbers = [
        dir_number(entry, head, tail) for entry in numbered_dirs(parent, head, tail)
    ]

    return Path(parent) / f"{head}{max(numbers, default=-1) + 1:0{DIGITS}d}{tail}"


def dir_number(entry, head, tail=""):
    """Return the number in `entry`'s name, or None if it carries none."""
    name = entry.name
    if not (name.startswith(head) and name.endswith(tail)):
        return None

    middle = name[len(head) : len(name) - len(tail)] if tail else name[len(head) :]
    try:
        return int(middle)
    except ValueError:
        # Something else living beside the numbered directories, which is not
        # ours to interpret.
        return None


def existing_batches(parent):
    """Return the batch directories already under `parent`."""
    return numbered_dirs(parent, BATCH_PREFIX)


def next_batch_dir(parent):
    """Return the batch directory to write next, under `parent`.

    Where a batch goes is not a choice, for the same reason it is not one for a
    run: it goes beside the datum that generated it. That also makes the layout
    record which datum a batch came from, which nothing else does --- the
    resolved bounds are logged, but the provenance was otherwise yours to
    remember.
    """
    return next_numbered_dir(parent, BATCH_PREFIX)


def resolve_workdir(workdir):
    """Return the directory `-o` names, numbering it if it holds a `%`.

    `-o runs/v%` is the next free `runs/vNNNN`. Without a placeholder the path
    is taken as typed, so numbering is asked for rather than imposed.

    Worth noting what this does to the rest of the CLI: a numbered workdir is
    free by construction, so it can never hold an answer, and `check_clobber`
    has nothing to refuse. Numbering and `-f` are therefore the two ways of not
    losing a run, and asking for one means never needing the other.

    **A run that fails still leaves its workdir**, holding the config it tried
    and the log saying how far it got, and a numbered one still consumes its
    number. Both are intended. The transcript of a failure is the most useful
    thing in the directory at that moment, and deleting it to keep the
    numbering tidy would throw away the evidence for the sake of the filing.
    A number is cheap; the log of the run that did not work is not.
    """
    text = str(workdir)
    if PLACEHOLDER not in text:
        return Path(workdir)

    if text.count(PLACEHOLDER) > 1:
        raise ValueError(
            f"-o takes at most one '{PLACEHOLDER}', which is where the number "
            f"goes; {text!r} has {text.count(PLACEHOLDER)}."
        )

    path = Path(workdir)
    if PLACEHOLDER not in path.name:
        raise ValueError(
            f"-o can only number the last part of a path, and {text!r} puts "
            f"'{PLACEHOLDER}' higher up. Number the directory being made, not "
            "one of its parents."
        )

    head, _, tail = path.name.partition(PLACEHOLDER)

    return next_numbered_dir(path.parent, head, tail)


_HANDLER_TAG = "_turbigen_handler"

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


def load_document(config_path, args):
    """Read the config file and return the document it asks for, as a dict.

    Everything that happens before validation: plugins registered, includes
    assembled, a previous answer dropped, overrides applied. What comes back is
    what the user asked for and nothing more --- no defaults filled in --- which
    is what makes it the right thing to copy into a workdir. `output.yaml` is
    where the fully expanded version belongs.

    Separate from `load_config` because `-o` needs the document without the
    `Config`, and because the discovery it does has to happen against the
    *original* directory: the registry is global, so a design found here is
    still registered when the copy is read back from somewhere else entirely.
    """
    config_path = Path(config_path)

    # Designs must be registered before the config is built, so that the type
    # keys it names can be resolved.
    plugins.discover(config_path.parent)

    # Includes are resolved first, so an override is the last word and applies
    # to the assembled document rather than to whichever fragment defined the
    # key it names.
    data = include.read(config_path)

    # Dropped before the overrides are applied, so that `-s result.x=1` cannot
    # reach into a previous run's answer. Re-running a case rewrites it anyway.
    data.pop(case.RESULT_KEY, None)

    apply_overrides(data, args.overrides)

    return data


def load_config(config_path, args):
    """Read the config file, apply overrides, and build a Config.

    Discovery is done here rather than through `Config.from_file` because the
    overrides have to be applied to the raw dict, before it is validated.
    """
    return Config.from_dict(load_document(config_path, args))


#
# TARGETS
#
# A verb acts on one config file or on many. Many is what a job array and a
# local queue are made of, and it is also the serial loop that would otherwise
# be written in bash around every batch.
#


def targets(args):
    """Return the config files this invocation acts on.

    Every verb comes through here --- `each` for five of them and `cmd_batch`
    for itself --- so the one rule about what may be handed to turbigen is
    stated once, and applies flatly.
    """
    paths = [Path(name) for name in args.CONFIG_YAML]

    for path in paths:
        check_not_output(path)

    return paths


def check_not_output(path):
    """Raise if `path` is a file turbigen wrote rather than one it was given.

    An `output.yaml` only exists because some other config was run, and that
    config is still there: `run` leaves the file it was handed, whatever it was
    named, and `batch`, `iterate` and `chic` write `input.yaml` into every
    directory they invent. So there is always another file naming the same
    directory, and wanting this one is not a case to support.

    Flat across every verb, `design` included. `design` writes nothing and so
    could read this safely, but a rule with an exception in it is one more
    thing to remember than a rule without, and the exception would buy only the
    ability to type a name that always has an equivalent.

    This is what makes `OUTPUT_NAME` an invariant rather than a convention: a
    verb cannot overwrite the file it read, because it will not read it.
    """
    if path.name != OUTPUT_NAME:
        return

    # Named rather than described, because the whole point is that there is
    # something else to type, and guessing at it is the user's least favourite
    # part of an error message.
    siblings = sorted(
        entry.name
        for entry in path.resolve().parent.glob("*.yaml")
        if entry.name != OUTPUT_NAME
    )

    if len(siblings) == 1:
        instead = f"Run on {path.parent / siblings[0]} instead."
    elif siblings:
        listed = ", ".join(siblings)
        instead = f"The configs beside it are: {listed}."
    else:
        # Nothing left to point at: the original was deleted, or this file was
        # copied out of its run on its own. Adopting it as an input is fine,
        # but it should be a thing you did rather than a thing that happened.
        instead = (
            f"Nothing else is beside it, so copy it to {batch.INPUT_NAME} if you "
            "mean to adopt it as an input."
        )

    raise ValueError(f"{path} is a file turbigen wrote, not one to run on. {instead}")


def out_dirs(args, paths):
    """Return the directory each target will be worked in.

    A workdir if `-o` named one, and each config's own directory otherwise.
    Computed before anything is created, so that a refusal happens before a
    directory is made rather than after.
    """
    if workdir := getattr(args, "workdir", None):
        return [Path(workdir)]

    return [path.resolve().parent for path in paths]


def check_clobber(args, directories):
    """Raise if this invocation would replace an answer already recorded.

    One rule, whatever the number of targets. It used to be that a single
    target overwrote silently and several refused, on the grounds that a batch
    is cluster hours and one re-run is a recoverable mistake --- but the count
    of paths on the command line is a poor proxy for how much is at stake, and
    "did I mean all of these" is not what it measures. Now anything that would
    replace an answer says so, and `-o` is how you run a variant without
    replacing one.

    Two things keep this from firing where it has no business. It is **scoped
    by capability**: a verb that offers no `--force` cannot be destroying
    anything, which excludes `design` (writes nothing) and `report` (never
    removes an answer it did not reach) without naming either. And it keys on a
    recorded `result:` rather than on the file existing, because `output.yaml`
    stopped meaning "a run finished here" once `report` began writing one --- so
    plotting a batch does not then block running it.
    """
    # Not `getattr(..., False)`: the absence of the attribute is the signal, and
    # is different from the flag being present and unset.
    if not hasattr(args, "force") or args.force:
        return

    answered = [
        directory
        for directory in directories
        if _records_an_answer(Path(directory) / OUTPUT_NAME)
    ]
    if not answered:
        return

    where = answered[0] / OUTPUT_NAME
    count = (
        "" if len(answered) == 1 else f"{len(answered)} of {len(directories)} targets, "
    )
    raise ValueError(
        f"{count}{where} already records an answer. Running here would replace "
        "it; pass -f to do that, or -o to work somewhere new."
    )


def redirect(args, paths):
    """Return the targets to act on, honouring `-o` by moving the config.

    Without `-o`, the paths as given. With one, the config is copied into the
    workdir as `input.yaml` and *that* becomes the target, so everything
    downstream --- `logging_into`, `stored_field`, `prepare` --- works on a
    directory holding both the config and its output, exactly as if the file
    had always lived there. No verb learns about the flag.

    Done this way rather than as a second output path because the colocation is
    load-bearing: `report` finds `restart.npz` beside the config, `plugins`
    walks up from it, and `iterate` and `chic` write an `input.yaml` into every
    directory they invent. A flag that split the two would break all three.
    """
    workdir = getattr(args, "workdir", None)
    if workdir is None:
        return paths

    if len(paths) > 1:
        raise ValueError(
            f"-o names one directory, but {len(paths)} config files were given. "
            "One directory is one run, so run them one at a time, or leave -o "
            "off and let each write beside itself."
        )

    return [_copy_into_workdir(args, paths[0], Path(workdir))]


def _copy_into_workdir(args, config_path, workdir):
    """Write `config_path` into `workdir` as `input.yaml`, and return the copy.

    What lands there is the *document*: includes assembled, overrides applied,
    defaults left out. Expanding the includes is not optional, because their
    paths resolve against the directory the config came from and would dangle
    the moment it moves. Baking the overrides in is what makes the workdir a
    record of what was asked for rather than of what was typed.

    Comments do not survive, the safe loader having dropped them. That is the
    cost of a generated directory, and the reason the original is left alone.
    """
    data = load_document(config_path, args)

    # Validated before a directory is made, so a config with a typo in it fails
    # where it was typed rather than leaving an empty workdir behind.
    Config.from_dict(data)

    copied = workdir / batch.INPUT_NAME
    _check_not_someone_elses(args, copied, data)

    workdir.mkdir(parents=True, exist_ok=True)

    ember.yaml_util.write_yaml(data, copied)
    logger.info(f"Copied the config to {copied}")

    return copied


def _check_not_someone_elses(args, copied, data):
    """Raise if `copied` is a config that did not come from this invocation.

    The other guards cover `output.yaml`, which is never a target, and a
    recorded answer, which `check_clobber` refuses. Neither covers a config
    sitting in the workdir with no answer beside it yet --- an unrun batch
    member, or something being drafted --- and `-o` pointed at one used to
    replace it without a word.

    Compared as parsed documents rather than as text, so re-running `-o` into
    the same directory after a failure is silent when the config has not
    changed, which is the case where insisting on `-f` would be noise. Anything
    that will not parse counts as different: unreadable is not the same as
    absent, and the file is somebody's either way.
    """
    if getattr(args, "force", False) or not copied.is_file():
        return

    try:
        existing = ember.yaml_util.read_yaml(copied)
    except Exception:
        existing = None

    if existing == data:
        return

    raise ValueError(
        f"{copied} is a different config from the one being copied there. "
        "Writing it would lose whatever it says; pass -f to do that anyway, "
        "or name a workdir that is empty."
    )


def each(args, one):
    """Run `one(args, config_path)` for every target, or submit them all.

    Returns the worst exit code any target reached, so a run over several
    reports failure if any of them failed and a script need not parse the log.
    A solve that did not converge is a 2 and the next target still runs: a
    diverged march is an answer about that design, not a reason to doubt the
    rest.

    **An exception stops the whole invocation, deliberately.** A config that
    will not load, a design that will not close, a mesh that cannot be built:
    these say the set of configs is wrong rather than that one member of it is
    unlucky, and the ones behind it are likely wrong the same way. Better to
    stop while the message is still on the screen than to bury it under the
    thirty that followed and have it found tomorrow.

    Two consequences to know about. The targets after the failure do not run,
    so a serial sweep is resumed by fixing the config and running the rest ---
    which `output.yaml` makes safe, the finished ones refusing to be redone.
    And `--queue` does not behave this way: it submits every path without
    loading any but the first, so the same command queued gets answers for the
    good members. That is the difference between validating locally and handing
    work to a scheduler, not an inconsistency to be ironed out.
    """
    paths = targets(args)

    # Settled once, here, rather than wherever the workdir is next wanted: a
    # `%` is resolved by looking at what exists, so asking twice invites two
    # answers. Written back onto the arguments so that everything downstream
    # sees the directory that was chosen, the same way `logging_into` records
    # where a verb wrote.
    if getattr(args, "workdir", None) is not None:
        args.workdir = str(resolve_workdir(args.workdir))

        # A bare `--restart` means the field beside the config you named, and
        # `-o` must not change what it points at. Resolved before the redirect,
        # or it would look in the workdir being created -- which never holds a
        # field, so the two flags together could not be used at all. Continuing
        # from what you have while writing somewhere new is the whole of a warm
        # start, and the obvious reason to combine them.
        if getattr(args, "restart", None) is True and len(paths) == 1:
            args.restart = str(resolve_restart(args, paths[0]))

    # Checked against where the work will land, and before `redirect` creates
    # anything there.
    check_clobber(args, out_dirs(args, paths))

    paths = redirect(args, paths)

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
    it happens. `--queue` is consumed here, and `-o` has already had its whole
    effect: the paths being submitted are the copies in the workdir, so passing
    it on would redirect a config that is already there.

    `--force` **is** carried, unlike the two above. Replacing an answer needs
    saying wherever the run actually happens, and a submitted job re-checks on
    the cluster: without this a queued re-run would refuse itself, hours later
    and out of sight. It used to be dropped on the grounds that a single target
    overwrote anyway, which stopped being true when that special case went.

    Rebuilt from the parsed arguments rather than by editing `sys.argv`, so an
    option written any of the ways argparse accepts it comes out in one form.
    """
    options = []

    if getattr(args, "force", False):
        options.append("--force")

    for override in args.overrides:
        options += ["-s", override]

    if args.verbose:
        options.append("-v")

    if getattr(args, "svg", False):
        options.append("--svg")

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

    bconds.apply(grid, machine, config.operating_point, config.inlet_profile)
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

    The resolved config is written too, which is the one thing a report leaves
    that is not a picture. It is also the only way to get one without paying
    for a solve, `design` having promised to write nothing at all.
    """
    with logging_into(args, config_path) as out_dir:
        config = load_config(config_path, args)

        field = stored_field(config_path)
        config, machine, grid = prepare(config, field)

        # Looked for whether or not there is a field to go with it: a history
        # beside the config means a run happened here, and its convergence page
        # is worth drawing either way.
        history = read_history(config_path.parent / HISTORY_NAME)

        answer = reconstruct(config, machine, grid, history, field)

        result = answer or Result(machine=machine, grid=grid, history=history)

        write_report(config, result, out_dir, svg=args.svg)
        _write_report_output(config, answer, out_dir)

    return 0


def reconstruct(config, machine, grid, history, field):
    """Return the answer `field` records for `config`, or None if it cannot.

    A report has the same ingredients a run has once its march is over -- the
    solved grid, the history beside it -- so it can measure the same three
    things `solve` does and archive an identical `result:`. What it does not
    have is the run itself, so it must first establish that the field it picked
    up is the solution to the config in front of it and not merely a useful
    starting guess for it. That is the stamp's whole purpose.

    None is returned wherever any part of that fails. It means "this report has
    no answer to record", never "the answer is that it did not converge": the
    two are very different to `turbigen.database`, which drops any sample whose
    `converged` is false, so guessing here would silently delete designs from a
    later fit.
    """
    if field is None or grid is None:
        return None

    stamp = restart.read_stamp(field)
    if stamp is None:
        logger.info(
            f"The field at {field} carries no design stamp, so this report "
            "cannot tell which design it solves and will not record an answer."
        )
        return None

    if stamp != restart.design_stamp(config):
        logger.warning(
            f"The field at {field} was written for a different design than "
            "this config describes, so it is drawn but not recorded as its "
            "answer."
        )
        return None

    # Beyond here the field is this config's solution. The remaining guards are
    # about whether the answer can be *described*, not whether it is the right
    # one.
    if config.solver is None or history is None:
        logger.info(
            "Without a solver: section and a convergence history there is no "
            "saying whether this field converged, so no answer is recorded."
        )
        return None

    try:
        actual = mixout.mean_line(grid, machine)
    except Exception as err:
        logger.warning(f"Could not mix out the stored field: {err}")
        return None

    result = Result(
        machine=machine,
        actual=actual,
        grid=grid,
        converged=config.solver.converged(history),
        history=history,
    )

    return dataclasses.replace(result, error=iterate.errors(config, result))


def _write_report_output(config, answer, out_dir):
    """Write the resolved config, and the answer if this report reached one.

    **A report never removes a `result:` that is already there.** With an
    answer it writes one indistinguishable from the run's own; without one it
    writes the config alone -- unless that would drop an answer already on
    disk, in which case it writes nothing and says so.

    That last case is the whole reason this is not simply `_write_output`. A
    config edited since it was run still draws perfectly well, and re-plotting
    it must not be the thing that discards the answer being re-plotted.
    """
    path = out_dir / OUTPUT_NAME

    if answer is not None:
        case.write(path, config, answer)
        run_log.info(f"Wrote resolved configuration and its answer to {path}")
        return path

    if _records_an_answer(path):
        logger.warning(
            f"Leaving {path} as it is: this report reached no answer of its "
            "own, and the one recorded there is not this report's to discard."
        )
        return None

    case.write(path, config)
    run_log.info(f"Wrote resolved configuration to {path}")
    return path


def _records_an_answer(path):
    """Return whether `path` is a case file that already holds a result."""
    if not path.is_file():
        return False

    try:
        _, result = case.read(path, design=False)
    except Exception as err:
        # Unreadable is not the same as empty, and overwriting a file we cannot
        # parse is exactly the mistake this guard exists to prevent.
        logger.warning(f"Could not read the answer already in {path}: {err}")
        return True

    return result is not None


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


def solve(config, out_dir, restart_path=None, svg=False):
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

        # Last, because it is the answer to the question the config asked, and
        # what someone reads first when scrolling back. Guarded for the same
        # reason the mix-out above is: a table is a report of a solution the
        # CFD has already been paid for, and it must not be able to cost the
        # run the output written below.
        try:
            run_log.info(design_variable_string(config, result))
        except Exception as err:
            logger.warning(f"Could not compare the design against its solution: {err}")

    # Written whatever happened, and written first. A march that did not
    # converge is the one most likely to be picked up and continued, so
    # withholding its field would be exactly backwards -- and a post-processor
    # that raises must not be able to discard a solution the CFD has already
    # been paid for.
    restart_path = out_dir / RESTART_NAME
    restart.save(restart_path, grid, config)
    run_log.info(f"Wrote the flow field to {restart_path}")

    # Beside the field, and for the same reason: it is what a re-plot needs to
    # draw the convergence page, and it costs a few kilobytes.
    save_history(out_dir / HISTORY_NAME, history)

    _write_output(config, result, out_dir, svg=svg)

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

        result = solve(
            config, out_dir, resolve_restart(args, config_path), svg=args.svg
        )

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

        _, result, converged, _ = converge_design(
            config, out_dir, resolve_restart(args, config_path)
        )

        # The answer, on a console that has been shown only the iteration
        # table. Not for a march that blew up, whose mixed-out mean line is
        # whatever its NaNs averaged to and would read as a result.
        if result is not None and result.converged and result.actual is not None:
            iterate.logger.info(result.actual.to_string())

    return 0 if converged else 2


def cmd_chic(args):
    """Converge the design, then sweep it to its stability limit."""
    if not args.verbose and not args.queue:
        _quieten_the_runs()

    return each(args, _chic_one)


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


def _chic_one(args, config_path):
    """Sweep one config file, keeping every point beside it."""
    with logging_into(args, config_path) as out_dir:
        config = load_config(config_path, args)

        if config.solver is None:
            raise ValueError(
                "The 'chic' command needs a solver: section in the config file."
            )
        if config.chic is None:
            raise ValueError(
                "The 'chic' command needs a chic: section saying how far to "
                "step and how finely to pin the limit."
            )

        previous = resolve_restart(args, config_path)

        # The design first, unless it is already done. Every verb implies the
        # ones before it, and a characteristic of a machine still being
        # redesigned is a characteristic of no machine in particular.
        if not design_is_settled(config_path, config):
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
            result = solve(config_now, point_dir, previous)
            if result.converged:
                # A diverged point is not somewhere to start the next one from,
                # and the next one is a bisection back towards what did work.
                previous = point_dir / RESTART_NAME

            return result

        points, bracket = chic.sweep(config, run)

        chic.logger.info(chic.format_table(points, bracket))

    # The sweep did its job whenever it bracketed something, which is what the
    # verb is for -- a point that refused to converge is the answer here rather
    # than a failure, unlike every other verb.
    return 0 if any(point.converged for point in points) else 2


def converge_design(config, out_dir, previous=None):
    """Iterate `config` to convergence, keeping every iteration under `out_dir`.

    The whole of an iteration, so that `chic` composes it rather than writing a
    second copy --- the same reason `solve` is one function that `iterate` calls
    repeatedly. Written out twice, the two would drift, which is what happened
    to `turbigen.main`.

    A config with no `iterate:` section still gets its design point solved
    once, because the sweep that follows needs a field to start from and an
    answer to be a departure from.

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
    if not config.iterate:
        design_dir = out_dir / "iter_0000"
        design_dir.mkdir(parents=True, exist_ok=True)
        iterate.logger.info(f"Design point in {design_dir}")

        write_input(config, design_dir)

        result = solve(config, design_dir, previous)
        field = promote_final(out_dir, design_dir, result.converged)

        return config, result, result.converged, field

    # Iteration -1: where the knobs start. Anchored on the config file's own
    # directory, because a config is often run from somewhere else, and
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

        # Where this iteration's knobs stood, which the next one moves: the
        # sequence is reproducible from the datum, but no single member of it
        # is.
        write_input(config_now, iter_dir)

        # Chained: each iteration starts from the field the last one reached,
        # which is most of the saving. Index-space interpolation covers the
        # mesh moving with the design.
        result = solve(config_now, iter_dir, previous)
        previous = iter_dir / RESTART_NAME

        return result

    config, result, converged = iterate.converge(config, run, config.max_iter)

    field = promote_final(out_dir, previous.parent, converged)

    return config, result, converged, field


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
    # no way to find what it just made: BATCH=$(turbigen batch case.yaml).
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
    console handler alone, so `log_turbigen.txt` still holds every run
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


def _write_output(config, result, out_dir, svg=False):
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

    write_report(config, result, out_dir, svg=svg)


def grid_string(grid):
    """One-line summary of the size of a grid."""
    return f"Mesh: n_block={len(grid)}, n_cell/1e6={grid.size / 1e6:.2f}"


def _design_variable_rows(config, result):
    """Yield ``(name, nominal, actual, is_variable)`` for every inverted key.

    **Both columns come from the same `backward()`**, one applied to the
    nominal mean line and one to the mixed-out actual, rather than reading the
    nominal off the config's own fields. Three things follow.

    The comparison is like for like: whatever definition of loss or loading the
    design uses, both sides are measured through it, so a difference is the
    flow differing and never the two sides being computed differently.

    Diagnostics get a nominal column for free. `backward` returns reaction,
    pressure ratio and efficiency alongside the design variables, and those are
    where a mismatch usually shows first. The package this replaces reached
    them through a second loop over "additional vars not in nominal" and left
    the nominal column blank, so the one comparison worth making was the one it
    could not print.

    And it is sound, because there are two states and not three: `solve_for`
    raises if it cannot hit its targets and `check_round_trip` raises if the
    inverted variables disagree with the fields that asked for them, so a
    nominal mean line that exists *is* the requested design.

    Design variables are still marked, because a variable you set and a number
    you read are different kinds of thing even when they are printed the same
    way. That is field membership rather than the order `backward` happens to
    return its keys in, which is only the author's convention.
    """
    variables = {field.name for field in dataclasses.fields(config.mean_line)}

    nominal = config.mean_line.backward(result.nominal)
    actual = config.mean_line.backward(result.actual)

    for name, value in nominal.items():
        # A design may declare a variable as not invertible, and it may return
        # one the other call did not; neither is an error, and neither can be
        # compared.
        if value is None or actual.get(name) is None:
            continue

        was, now = np.atleast_1d(value), np.atleast_1d(actual[name])
        if was.shape != now.shape:
            continue

        for i, (one, other) in enumerate(zip(was, now)):
            label = name if was.size == 1 else f"{name}[{i}]"
            yield label, float(one), float(other), name in variables


def design_variable_string(config, result):
    """Return a table of what the design asked for against what it achieved.

    The most valuable few lines a run prints: a mean line states an intent, and
    this is the only place that intent and the CFD are put side by side in the
    same units.

    Errors are `nominal - actual`, which is the sign
    :meth:`turbigen.iterate.MeanLine.error` already uses, so a row here and a
    row of the iteration table describe one number the same way round.
    """
    rows = list(_design_variable_rows(config, result))
    if not rows:
        return "Design variables: nothing that backward() returns can be compared."

    width = max(len(name) for name, _, _, _ in rows)
    header = (
        f"{'name':<{width}}  {'nominal':>10}  {'actual':>10}  "
        f"{'err':>10}  {'err/%':>8}"
    )
    lines = ["Design variables:", header, "-" * len(header)]

    # Set variables first, then what was read off the answer, with a rule
    # between. Within each, the order the design returned them in, which is the
    # order its author thought about them.
    for wanted in (True, False):
        block = [row for row in rows if row[3] is wanted]
        if not block:
            continue
        if not wanted:
            lines.append("-" * len(header))

        for name, was, now, _ in block:
            error = was - now
            # A nominal of zero has nothing to be relative to. Recamber and
            # swirl angles are routinely zero by design, so this is the common
            # case rather than a guard against the impossible.
            relative = f"{error / was * 100.0:8.2f}" if was else f"{'--':>8}"
            lines.append(
                f"{name:<{width}}  {was:10.4g}  {now:10.4g}  {error:10.3g}  {relative}"
            )

    return "\n".join(lines)


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


def write_report(config, result, out_dir, svg=False):
    """Run the post-processors and collect their figures into one PDF.

    Nothing is produced without an output directory, so the figures are only
    made when there is somewhere to put them. With one, a report is always
    written: the standard plots cost a fraction of a solve, and a run whose
    output nobody looks at is worse than a page nobody needed.

    `svg` additionally writes each figure as its own file, for a document that
    places them one at a time. Off by default, by the same rule: a directory of
    pictures nobody opens is worse than the one PDF that holds them. The names
    carry the post-processor that drew each figure rather than a page number,
    so adding a plot cannot silently rename the images after it.
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
        for i_processor, processor in enumerate(processors(config)):
            logger.debug(f"Running post-processor {processor}")
            # Figures are closed as they are written rather than collected and
            # closed at the end, so a long report holds one at a time.
            for i_figure, figure in enumerate(processor.report(config, result)):
                pdf.savefig(figure)
                if svg:
                    figure.savefig(
                        out_dir / f"post_{i_processor:02d}_{processor.type}"
                        f"_{i_figure}.svg"
                    )
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
        prog="turbigen",
        description=(
            "turbigen is an experimental rebuild of the turbigen design "
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
            f"post.pdf and {OUTPUT_NAME}. Each standard plot draws nothing when "
            "what it needs is absent, so a mean-line design gives the geometry "
            "pages and a solved case gives the flow. Re-plotting a finished run "
            "is therefore the same command, with no flag between them. The "
            f"{OUTPUT_NAME} carries an answer only when the stored field is "
            "stamped as this design's solution, and a report never removes one "
            "it cannot reproduce. The grid itself is never written, because how "
            "a mesh is serialised is a property of the solver that will read it."
        ),
    )
    _add_svg_argument(report)
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
    _add_out_dir_argument(run)
    _add_queue_argument(run)
    _add_restart_argument(run)
    _add_svg_argument(run)
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
    _add_out_dir_argument(iterate_)
    _add_queue_argument(iterate_)
    _add_restart_argument(iterate_)
    iterate_.set_defaults(func=cmd_iterate)

    chic_ = commands.add_parser(
        "chic",
        parents=[common],
        help="sweep a characteristic until the solution will not stand up",
        description=(
            "Converge the design, then hold its geometry fixed and step the "
            "back pressure until a point will not converge, halving the step "
            "and coming back at it from the last good field until the limit "
            "is pinned to chic.step_min. Each point is an ordinary run in a "
            "directory of its own. A case whose stored result says the design "
            "has already settled skips straight to the sweep. Needs a chic: "
            "section. What it finds is where a steady solver stops "
            "converging, which is not the surge line."
        ),
    )
    _add_force_argument(chic_)
    _add_out_dir_argument(chic_)
    _add_queue_argument(chic_)
    _add_restart_argument(chic_)
    chic_.set_defaults(func=cmd_chic)

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
    """Add --force, which allows a run to replace an answer already recorded.

    Only on the verbs that solve. `check_clobber` treats the absence of this
    flag as the verb having nothing to destroy, so adding it to a verb that
    writes no answer would switch a guard on rather than off.
    """
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help=(
            f"replace an answer already recorded in {OUTPUT_NAME} here; "
            "without it, a directory that has been run refuses to be run again"
        ),
    )


def _add_svg_argument(parser):
    """Add --svg, for the two verbs that draw one case.

    `design` has no figures to write, and `iterate` and `chic` draw a directory
    of runs rather than a case, which is not what a document places. A flag
    rather than a config key, because how a report is consumed is a property of
    who is reading it, not of the machine being designed: one config feeds both
    a person opening post.pdf and a page placing the figures individually.
    """
    parser.add_argument(
        "--svg",
        action="store_true",
        help=(
            "also write each figure as its own SVG beside post.pdf, named "
            "after the post-processor that drew it, for embedding one at a time"
        ),
    )


def _add_out_dir_argument(parser):
    """Add -o, which runs a config in a directory of its own.

    A redirection of the target rather than a second output path: the config is
    copied into the workdir and the run happens there, so config and output
    stay in one directory. That is what the rest of the system assumes.
    """
    # argparse runs %-formatting over help text, so the placeholder has to be
    # doubled to survive being printed.
    shown = PLACEHOLDER * 2

    parser.add_argument(
        "-o",
        "--out-dir",
        dest="workdir",
        metavar="DIR",
        help=(
            "work in DIR instead of beside the config: the config is copied "
            f"there as {batch.INPUT_NAME}, with its includes expanded and any "
            "--set applied, and everything the run writes lands there. This is "
            "how to try a variant without replacing the answer you already "
            f"have. A '{shown}' in the last part of DIR is replaced by the next "
            f"free number, so -o runs/v{shown} writes runs/v0000, then "
            "runs/v0001"
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
    """Add --restart, for the verbs that march a grid rather than just draw one.

    Not on the common parent, and the two verbs left out are left out for
    different reasons. `design` never makes a grid to put a field on, so the
    flag would do nothing. `report` makes one, but takes its field from beside
    the config and nowhere else, deliberately: that is what makes a report a
    consistent picture of one directory. A field named from elsewhere is
    guaranteed not to match the stamp, so it could only ever draw a hybrid and
    refuse to record it --- flexibility that has nothing behind it.
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
        f"*** TURBIGEN v{turbigen.__version__} ***",
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
