"""Command line interface.

See CLI.md in this package for the full plan. Every verb it specifies is
implemented: `design`, `report`, `run`, `iterate` and `batch`.

This module is the command line and nothing else: arguments, where each run's
directory is, logging and submission. What happens to one case --- preparing a
grid, solving it, measuring it, recovering a stored answer --- is
:mod:`turbigen.pipeline`, and drawing it is :func:`turbigen.post.write_report`.

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
location is therefore never derived, and an input file is never overwritten
*in place* by the run that reads it -- which matters because the file written
is the *resolved* config, every default expanded, and writing that over a
hand-kept file would lose its comments to the safe loader. `-o` may read an
`output.yaml` and write its result elsewhere; only a write that lands back on
the file it read is refused. One directory is one run.

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
import functools
import logging
import sys
from collections.abc import Callable
from pathlib import Path
from timeit import default_timer as timer

import ember.convergence_history
import ember.yaml_util
import yaml

import turbigen
from turbigen import (
    batch,
    case,
    chic,
    database,
    include,
    iterate,
    job,
    node,
    pipeline,
    plugins,
    post,
    restart,
    warm_field,
)
from turbigen.config import Config
from turbigen.pipeline import HISTORY_NAME, OUTPUT_NAME, run_log
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


LOG_NAME = "log_turbigen.txt"
"""What a run calls its transcript, beside everything else it wrote."""

RESTART_NAME = restart.RESTART_NAME
"""Re-exported from :mod:`turbigen.restart`, where the name now lives so that
:func:`turbigen.database.nearest_field` can find a neighbour's field without
importing the CLI."""


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

NUMBERING_ATTEMPTS = 64
"""Times to re-ask for a free number before giving up.

Each loss means another process claimed the number first, so this bounds how
many runs can be launched into one directory at the same instant rather than
how hard anything retries. Far above any real fan-out, and finite so that a
directory nothing can be created in fails rather than spinning.
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

    **The directory is created here, and that is what makes the number safe to
    hold.** Scanning says what is free and creating says it is taken; between
    the two, another process asking the same question gets the same answer.
    Two runs launched together would land on one directory and interleave
    their iterations into it, silently where their configs happened to match.
    So the number is claimed by making the directory exclusively --- an atomic
    operation --- and a loser simply asks again and takes the next one. Which
    means a caller wanting the config checked before anything appears on disk
    has to check it before calling this; see :func:`each`.
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

    for _ in range(NUMBERING_ATTEMPTS):
        candidate = next_numbered_dir(path.parent, head, tail)
        try:
            candidate.mkdir(parents=True)
        except FileExistsError:
            # Somebody claimed it between the scan and here. Ask again: the
            # scan reads the highest that exists, so the next answer is past
            # whatever they took.
            continue
        return candidate

    raise ValueError(
        f"Could not claim a numbered directory under {path.parent} after "
        f"{NUMBERING_ATTEMPTS} attempts. Something else is making them as "
        f"fast as they can be asked for."
    )


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

    # Ember's log messages use Greek symbols (eps, psi, zeta) for its
    # coefficients; a console whose encoding cannot represent them --- cp1252,
    # the Windows default for a redirected stream --- would otherwise crash
    # `emit` with a UnicodeEncodeError. backslashreplace degrades to `\uXXXX`
    # instead, which is always encodable, rather than losing the log line.
    stream = sys.stderr
    if hasattr(stream, "reconfigure"):
        try:
            stream.reconfigure(errors="backslashreplace")
        except Exception:
            pass
    _add_handler(logging.StreamHandler(stream))


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
        check_not_output(args, path)

    return paths


def _would_overwrite_input(args, path):
    """True if this invocation's `output.yaml` is `path` itself.

    The only reason `output.yaml` is refused as a target: a run writes it, and
    writing the *resolved* config over a hand-kept one loses its comments. That
    only happens when the write lands where the file was read.

    Without `-o` it always does --- output goes beside the input. With a `%`
    placeholder it never does: the number names a directory that does not exist
    yet, and that is checked here without resolving it, so the `%` is still
    only resolved once, later. With a literal `-o dir` the run writes
    `dir/output.yaml`, which is `path` only if `path` already lives in `dir`.
    """
    workdir = getattr(args, "workdir", None)

    if workdir is None:
        return True

    if PLACEHOLDER in str(workdir):
        return False

    return Path(workdir).resolve() == path.resolve().parent


def check_not_output(args, path):
    """Raise if running on `path` would overwrite it with the run's own output.

    `output.yaml` is the file a run writes beside the config it was given.
    Handing that file back as the config is refused when the write would land
    on it --- the default, where output sits beside input --- because the file
    written is the *resolved* config, every default expanded, and writing that
    over a hand-kept file would lose its comments to the safe loader.

    It is allowed when `-o` sends the run somewhere else: the file read and the
    file written are then different files, and nothing is lost. `report`, which
    also writes here, is covered the same way.
    """
    if path.name != OUTPUT_NAME or not _would_overwrite_input(args, path):
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

    # Validated before anything is written into the directory. `each` has
    # already checked this document when a `%` was resolved -- which it must,
    # the claim creating the directory -- but a `-o` naming a path outright
    # arrives here unchecked, and either way a config that will not build has
    # no business being copied anywhere.
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
        # Before the workdir is resolved, because resolving a `%` now *makes*
        # the directory in order to claim its number -- so a config with a typo
        # in it has to fail here, or it would leave an empty numbered directory
        # behind and consume a number to say nothing. `_copy_into_workdir`
        # validates again on the document it actually writes; this is the same
        # check moved early, not a second opinion.
        for path in paths:
            Config.from_dict(load_document(path, args))

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
    from the key rather than asked for, the same way the depth of a design
    is set by what the config contains --- and the inference matters, because
    a batch submitted as `run` builds an archive `database` reads back as
    empty: a sample must have converged *and* have its errors inside their
    tolerances, which is what iterating is for.

    Logged, so that "why did this iterate" and "why did this not" are both
    answerable from the batch's own log file.
    """
    verb = "iterate" if config.iterate.correct else "run"

    reason = (
        "an iterate: key naming what to correct"
        if config.iterate.correct
        else "nothing to correct"
    )
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


@dataclasses.dataclass(frozen=True)
class Verb:
    """A command that runs each config file it is given, one at a time."""

    name: str
    """What is typed after `turbigen`."""

    one: Callable
    """``one(args, config_path)``, returning an exit code for that config."""

    help: str
    """One line, for the command list in `turbigen --help`."""

    description: str
    """The paragraph at the top of `turbigen <name> --help`."""

    flags: tuple[Callable, ...] = ()
    """The `_add_*_argument` functions giving this command its options."""

    needs: tuple[str, ...] = ()
    """Keys of :data:`REQUIREMENTS` a config must satisfy, checked in order."""

    quiet: bool = False
    """Keep the per-run tables off the console, unless `-v` or `--queue`."""


REQUIREMENTS = {
    "solver": (
        lambda config: config.solver is not None,
        "needs a solver: section in the config file.",
    ),
    "iterate": (
        lambda config: bool(config.iterate.correct),
        (
            "needs an iterate: key with a correct: list saying what to correct; "
            "without one, use 'run'."
        ),
    ),
    "chic": (
        lambda config: config.chic is not None,
        "needs a chic: section saying how far to step and how finely to pin the limit.",
    ),
}
"""What a command can require of a config, and the message when it is missing."""


def require(config, command):
    """Raise unless `config` has every section the command `command` needs."""
    (verb,) = (verb for verb in VERBS if verb.name == command)
    for need in verb.needs:
        satisfied, message = REQUIREMENTS[need]
        if not satisfied(config):
            raise ValueError(f"The '{command}' command {message}")


def _dispatch(verb, args):
    """Run `verb` over every config file named, or submit them all."""
    # Once for the invocation rather than once per target: the filter is added
    # to the console handler, and adding it again for every config in a batch
    # would stack a dozen copies of the same test.
    if verb.quiet and not args.verbose and not args.queue:
        _quieten_the_runs()

    return each(args, verb.one)


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

    machine = config.design()
    run_log.info(machine.to_string())

    # The same table `run` prints last, off the same `backward()` -- but with
    # no CFD to mix out, there is no actual to set beside the nominal, so it
    # comes back one column narrower. See `design_variable_string`.
    try:
        run_log.info(pipeline.design_variable_string(config, Result(machine=machine)))
    except Exception as err:
        logger.warning(f"Could not print design variables: {err}")

    return 0


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

        field = pipeline.stored_field(config_path)
        config, machine, grid = pipeline.prepare(config, field)

        # Looked for whether or not there is a field to go with it: a history
        # beside the config means a run happened here, and its convergence page
        # is worth drawing either way.
        history = pipeline.read_history(config_path.parent / HISTORY_NAME)

        answer = pipeline.reconstruct(config, machine, grid, history, field)

        result = answer or Result(machine=machine, grid=grid, history=history)

        post.write_report([(config, result)], out_dir, svg=args.svg)
        _write_report_output(config, answer, out_dir)

    return 0


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


def _run_one(args, config_path):
    """Solve one config file, writing everything beside it."""
    with logging_into(args, config_path) as out_dir:
        config = load_config(config_path, args)

        require(config, args.command)

        result = pipeline.solve(
            config,
            out_dir,
            resolve_restart(args, config_path),
            svg=args.svg,
            soft=True,
            retry=True,
        )

    # Non-zero on a failed solve, so a script driving a sweep can tell without
    # parsing the log. Everything written above is still written: a diverged
    # run is exactly the one whose output someone needs to look at.
    return 0 if result.converged else 2


def _iterate_one(args, config_path):
    """Iterate one config file, keeping every iteration beside it."""
    with logging_into(args, config_path) as out_dir:
        config = load_config(config_path, args)

        require(config, args.command)

        _, result, converged, _ = converge_design(
            config, out_dir, resolve_restart(args, config_path)
        )

        # The answer, on a console that has been shown only the iteration
        # table. Not for a march that blew up, whose mixed-out mean line is
        # whatever its NaNs averaged to and would read as a result.
        if result is not None and result.converged and result.actual is not None:
            iterate.logger.info(result.actual.to_string())

    return 0 if converged else 2


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

        require(config, args.command)

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
            result = pipeline.solve(config_now, point_dir, previous)
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

    # Iteration -1: where the knobs start. Anchored on the config file's own
    # directory, because a config is often run from somewhere else, and
    # excluding that same directory because it is where this run's own
    # iterations will land -- one directory being one run, nothing else of
    # anyone's is in there to lose.
    #
    # Guarded on there being knobs at all rather than on `correct:`, which is
    # the test `warm_start` itself makes of what it was handed: a design point
    # with nothing to correct has nothing to start warm, and reaching in to be
    # told so would warn about a config that is perfectly in order.
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
    _check_edges_options(args, config.batch)

    # Scanned before the new batch directory exists, so it cannot count itself.
    # A batch is never written into, only beside, so nothing can be lost.
    datum_dir = paths[0].resolve().parent

    start = 0
    if args.carry_on:
        start = batch.next_index(existing_batches(datum_dir))
        batch.logger.info(f"Carrying on from index {start}.")

    out_dir = _open_batch(args, datum_dir)

    members = []
    for index, member in batch.generate(config, args.number, start, args.edges):
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


def _check_edges_options(args, spec):
    """Refuse ``--edges`` where it does not apply, before a number is burned.

    Like :func:`_check_grid_options`, and for the same reason: the surface of
    the box is a closed set drawn from `bounds:` alone, so it has no `-n` to
    size it and no tail to `--continue`, and there is nothing for it to walk
    when the section names its points with `values:`.
    """
    if args.edges is None:
        return

    if spec.is_grid():
        raise ValueError(
            "--edges walks the surface of a bounds: box; this batch: section "
            "names its points with values:. Give bounds: instead."
        )

    if args.number is not None:
        raise ValueError(
            "--edges sizes the batch from the box surface, so there is no -n "
            "to choose as well."
        )

    if args.carry_on:
        raise ValueError(
            "--edges is the whole surface of the box, a closed set, so there "
            "is nothing to --continue."
        )

    if args.edges < 2:
        raise ValueError(
            f"--edges is {args.edges}; it takes at least 2, the two ends of each bound."
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

    # Explicit encoding rather than the platform default (cp1252 on Windows):
    # this file is turbigen's own artifact, read back by a text editor, not by
    # whatever console produced it, so it should always be UTF-8.
    handler = logging.FileHandler(out_dir / LOG_NAME, encoding="utf-8")
    _add_handler(handler)
    logger.info(f"Output directory: {out_dir}")

    try:
        yield out_dir
    except Exception:
        # Logged here rather than left to `main`, which does log it but only
        # after this handler has been taken away in the `finally` below --- so
        # the traceback would reach the console and never the file sitting in
        # the workdir beside the run that failed. A directory whose log stops
        # mid-iteration saying nothing is the one case where the transcript was
        # most wanted, and re-raising leaves the exit status to `main` as
        # before.
        logger.exception("The run failed, and stopped here")
        raise
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
    _add_handler(logging.FileHandler(out_dir / LOG_NAME, encoding="utf-8"))
    logger.info(f"Output directory: {out_dir}")
    return out_dir


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
    for verb in VERBS:
        sub = commands.add_parser(
            verb.name,
            parents=[common],
            help=verb.help,
            description=verb.description,
        )
        for add_flag in verb.flags:
            add_flag(sub)
        sub.set_defaults(func=functools.partial(_dispatch, verb))

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
    batch_.add_argument(
        "--edges",
        type=int,
        default=None,
        metavar="N",
        help=(
            "instead of drawing from bounds:, run the surface of the box: the "
            "N-level grid over it, keeping only points with a coordinate at a "
            "bound. bounds: only, and not with -n or --continue"
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


VERBS = (
    Verb(
        "design",
        _design_one,
        help="design the mean line and print it, writing nothing",
        description=(
            "Design the mean line and geometry from a configuration file and "
            "print them. Nothing is ever written, so this is what to run while "
            "changing a number and watching the tables move. Use 'report' for "
            "figures."
        ),
    ),
    Verb(
        "report",
        _report_one,
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
        flags=(_add_svg_argument,),
    ),
    Verb(
        "run",
        _run_one,
        help="design, mesh and solve, then report",
        description=(
            "Design the machine from a configuration file, mesh it, apply "
            "boundary conditions and an initial guess, and solve. Everything "
            f"is written beside the config, in {OUTPUT_NAME} and its "
            "companions. Exits 2 if the solver did not converge, having "
            "written its output anyway."
        ),
        flags=(
            _add_force_argument,
            _add_out_dir_argument,
            _add_queue_argument,
            _add_restart_argument,
            _add_svg_argument,
        ),
        needs=("solver",),
    ),
    Verb(
        "iterate",
        _iterate_one,
        help="run repeatedly, moving the design onto its own solution",
        description=(
            "Solve the machine, measure how far its design is from what the "
            "flow actually did, correct the design, and solve again. Each "
            "iteration is an ordinary run in a directory of its own beside the "
            "config, with 'final' linked to the last and every iteration kept. "
            "Needs an iterate: key naming what to correct. Exits 2 if the "
            "design had not converged by iterate.max_iter."
        ),
        flags=(
            _add_force_argument,
            _add_out_dir_argument,
            _add_queue_argument,
            _add_restart_argument,
        ),
        needs=("solver", "iterate"),
        quiet=True,
    ),
    Verb(
        "chic",
        _chic_one,
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
        flags=(
            _add_force_argument,
            _add_out_dir_argument,
            _add_queue_argument,
            _add_restart_argument,
        ),
        needs=("solver", "chic"),
        quiet=True,
    ),
)
"""Every command that works through its config files one at a time, in the
order `--help` lists them. `batch` is not here: it takes one datum and has
arguments of its own, so :func:`_make_parser` adds it by hand."""


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
    except Exception:
        # A failure gets its traceback, whatever it is. Most of them are raised
        # from a config file or a plugin, which are the user's own code in the
        # sense that matters: the file and line are what say which of their
        # lines to look at. Summarising to `Type: message` reads tidily for the
        # errors raised deliberately against user input, but those cannot be
        # told apart by type from the ones that mean something is broken --
        # both arrive as ValueError -- so suppressing the trace for one
        # suppresses it for the other, which is the expensive half of the
        # trade.
        logger.exception("Error encountered, quitting...")
        return 1
    finally:
        # In a finally block so that a run which fell over still reports how
        # long it took to get there, and where it left what it had written.
        logger.info(f"Total time: {_format_elapsed(timer() - start_tic)}")
        if out_dir := getattr(args, "out_dir", None):
            logger.info(f"Output directory was: {out_dir}")


if __name__ == "__main__":
    sys.exit(main())
