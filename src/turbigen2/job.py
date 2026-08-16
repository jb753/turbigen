"""Where a run executes.

A :class:`Job` is a config node like any other, chosen by ``type:``, so adding
a queue system is one class registering itself and nothing in the CLI changes.
That is the whole reason this is a family rather than a flag per backend: we do
not have a flag per solver either.

The stage interface is the usual one::

    job.submit(tasks, verb, options) -> ids   # framework: validate, log
    job.forward(tasks, verb, options) -> ids  # the author writes this

A :class:`Task` is one config file to run. The verb and its options are shared
by every task in a submission, so only the config path varies --- which is what
lets a SLURM array put the varying part in a file and the fixed part in the
script once.

Two things this deliberately does not do.

**It is never implied.** The package this replaces submits whenever a ``job:``
key is present, so ``turbigen config.yaml`` silently re-execs itself inside a
job and every entry point needs a ``--no-job`` escape hatch to break the
recursion. Here the key says *how* to submit and ``--queue`` says *whether*, so
there is one level and no negative flag.

**It never reaches the design.** A partition is a property of where you are,
not of the machine, so nothing here is read by any design stage, and
`database.SUBTREE` already excludes it from being mistaken for a design
variable.
"""

import dataclasses
import logging
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import ClassVar

from turbigen2.node import Node

logger = logging.getLogger("turbigen.job")
"""What was submitted, where, and what the queue called it."""

TASKS_NAME = "tasks.txt"
"""Where a SLURM array keeps the config path of each of its tasks."""

SCRIPT_NAME = "submit.sh"
"""What the generated sbatch script is called."""

DEFAULT_COMMAND = "turbigen2"
"""The console script, for when this is not running as one."""


@dataclasses.dataclass(frozen=True)
class Task:
    """One config file to run somewhere else."""

    config: Path
    """The config file to run."""

    name: str
    """What the queue should call it. The config's own directory, since one
    directory is one run."""


def run_process(argv, cwd):
    """Run `argv` in `cwd` and return its stdout, raising if it failed.

    The one place this module reaches outside the process, so a test replaces
    this rather than each backend growing an injected runner --- which would be
    a field on a frozen config node that no config file ever sets.
    """
    logger.debug(f"Running {shlex.join(argv)} in {cwd}")

    proc = subprocess.run(argv, cwd=cwd, text=True, capture_output=True)

    if proc.returncode != 0:
        message = proc.stderr.strip() or proc.stdout.strip()
        raise RuntimeError(f"{argv[0]} failed: {message}")

    return proc.stdout.strip()


class Job(Node):
    """Base for queue systems."""

    command: str = ""
    """The turbigen2 executable a submitted job should run. Empty means the one
    running now, which is right whenever the queue shares this filesystem and
    this environment; name it when a compute node needs a different path."""

    #
    # TO BE IMPLEMENTED BY A BACKEND
    #

    def forward(self, tasks, verb, options):
        """Send `tasks` to the queue and return what it called them.

        Parameters
        ----------
        tasks : tuple of Task
            The config files to run, one job each.
        verb : str
            The command to run them with, as in ``run`` or ``iterate``.
        options : tuple of str
            Flags shared by every task, already in command-line form.

        Returns
        -------
        list of str
            Whatever identifies the submitted work in this queue.

        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement forward(self, tasks, verb, options)"
        )

    #
    # FRAMEWORK
    #

    def submit(self, tasks, verb, options=()):
        """Submit `tasks`, reporting what the queue made of them."""
        tasks = tuple(tasks)
        options = tuple(options)

        if not tasks:
            raise ValueError("There is nothing to submit.")

        logger.info(f"Submitting {len(tasks)} job(s) to {self.type}.")

        ids = self.forward(tasks, verb, options)

        for job_id in ids:
            logger.info(f"Submitted {job_id}")

        return ids

    def executable(self):
        """Return the turbigen2 command a submitted job should run."""
        if self.command:
            return self.command

        # Resolved, because a job starts in a directory of the queue's
        # choosing, and often a shell of one too: a relative argv[0] or one
        # found on an interactive PATH need not resolve there.
        found = shutil.which(sys.argv[0]) or shutil.which(DEFAULT_COMMAND)

        # Nothing found means this is not running as the console script at all
        # -- `python -m`, or a notebook -- so name the script and let the
        # queue's own PATH find it, rather than submitting `-c` as a command.
        return str(Path(found).resolve()) if found else DEFAULT_COMMAND

    def argv(self, config, verb, options):
        """Return the command line one task runs."""
        return [self.executable(), verb, str(Path(config).resolve()), *options]


class Slurm(Job):
    """Submit to SLURM as a single job array.

    **Zero or empty means unstated**, and an unstated setting is left out of the
    script entirely, so sbatch's own ``SBATCH_ACCOUNT``, ``SBATCH_PARTITION``
    and friends still apply. A cluster that already sets those in your profile
    therefore needs nothing here but ``type: slurm``.

    One array rather than one submission per config, and the array indexes
    *lines of a file* rather than directory names. The package this replaces
    indexes numbered directories and so refuses anything but a consecutive
    range --- which the batches `turbigen2 batch` writes are not, since a point
    that will not design is skipped and never retried.
    """

    type: ClassVar[str] = "slurm"

    hours: float = 0.0
    """Wall-clock time limit [hr]."""

    account: str = ""
    """Account to charge the compute time to."""

    partition: str = ""
    """Partition to run in."""

    qos: str = ""
    """Quality of service level."""

    gres: str = ""
    """Generic consumable resources, as in ``gpu:1``."""

    nodes: int = 0
    """Nodes per job."""

    tasks: int = 0
    """Tasks per job."""

    cpus: int = 0
    """CPUs per task."""

    mail_type: str = ""
    """When to send mail, as in ``FAIL``."""

    max_concurrent: int = 0
    """Most array members to run at once."""

    def forward(self, tasks, verb, options):
        # Everything goes in the tasks' common parent, and sbatch runs there,
        # so the array's own `slurm-%A_%a.out` files land beside the script
        # that made them rather than wherever you happened to be standing.
        base = _common_dir(tasks)

        tasks_path = base / TASKS_NAME
        tasks_path.write_text(
            "".join(f"{Path(task.config).resolve()}\n" for task in tasks)
        )
        logger.debug(f"Wrote {len(tasks)} task(s) to {tasks_path}")

        script_path = base / SCRIPT_NAME
        script_path.write_text(self.script(base.name, verb, options))
        logger.debug(f"Wrote the submission script to {script_path}")

        array = f"1-{len(tasks)}"
        if self.max_concurrent:
            array += f"%{self.max_concurrent}"

        out = run_process(["sbatch", f"--array={array}", SCRIPT_NAME], base)

        # sbatch says "Submitted batch job 12345".
        return [f"SLURM job {out.split()[-1]} ({len(tasks)} array task(s)) in {base}"]

    def script(self, name, verb, options):
        """Return the sbatch script for an array over `TASKS_NAME`."""
        # $CONFIG is left unquoted by shlex.join and quoted by hand, so the
        # shell expands it: quoting it as a literal is exactly the bug that
        # would run every array task on a file called "$CONFIG".
        command = " ".join(
            (
                shlex.join([self.executable(), verb]),
                '"$CONFIG"',
                shlex.join(options),
            )
        ).strip()

        return "\n".join(
            [
                "#!/bin/bash",
                *self.directives(name),
                "",
                f'CONFIG=$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" {TASKS_NAME})',
                command,
                "",
            ]
        )

    def directives(self, name):
        """Return the ``#SBATCH`` lines, omitting everything left unstated."""
        lines = [f"#SBATCH --job-name=turbigen2_{name}"]

        if self.hours:
            lines.append(f"#SBATCH --time={_time_string(self.hours)}")

        for flag, value in (
            ("--account", self.account),
            ("--partition", self.partition),
            ("--qos", self.qos),
            ("--gres", self.gres),
            ("--mail-type", self.mail_type),
            ("--nodes", self.nodes),
            ("--ntasks", self.tasks),
            ("--cpus-per-task", self.cpus),
        ):
            if value:
                lines.append(f"#SBATCH {flag}={value}")

        return lines


class Tsp(Job):
    """Queue locally through task-spooler.

    A real queue --- slots, job ids, listing, cancellation --- for the price of
    a small Debian package, in place of the flock'd text file, PID file, SIGHUP
    cancel-all and systemd unit that the package this replaces hand-rolls to
    the same end. Watch it with ``tsp -l``, read a job with ``tsp -c ID``, stop
    the rest with ``tsp -C``.
    """

    type: ClassVar[str] = "tsp"

    slots: int = 4
    """Jobs to run at once. Set on the queue itself, so it outlives this
    submission and applies to whatever is already waiting."""

    cpus: int = 0
    """Slots one job occupies, for work that wants more than one core. Zero
    leaves it at task-spooler's own default of one."""

    def forward(self, tasks, verb, options):
        binary = _find_tsp()

        if self.slots:
            run_process([binary, "-S", str(self.slots)], Path.cwd())

        ids = []
        for task in tasks:
            argv = [binary, "-L", task.name]
            if self.cpus:
                argv += ["-N", str(self.cpus)]
            argv += self.argv(task.config, verb, options)

            # Queued from the config's own directory, so a job inherits the cwd
            # a hand-typed run would have had.
            job_id = run_process(argv, Path(task.config).resolve().parent)
            ids.append(f"tsp job {job_id} ({task.name})")

        return ids


def _find_tsp():
    """Return the task-spooler binary, raising a useful error if there is none.

    Debian renames it to ``tsp`` because moreutils already ships a ``ts``;
    elsewhere it keeps its own name. Tried in that order, so a Debian box
    cannot pick up the timestamper by mistake.
    """
    for name in ("tsp", "ts"):
        if found := shutil.which(name):
            return found

    raise RuntimeError(
        "task-spooler is not installed: no 'tsp' or 'ts' on PATH. "
        "On Debian and Ubuntu, `apt install task-spooler`."
    )


def _common_dir(tasks):
    """Return the deepest directory holding every task."""
    return Path(
        os.path.commonpath([str(Path(task.config).resolve().parent) for task in tasks])
    )


def _time_string(hours):
    """Return `hours` as SLURM's ``H:MM:SS``, via seconds so nothing rounds up
    into an invalid minute or hour."""
    total = int(round(hours * 3600.0))
    whole_hours, rest = divmod(total, 3600)
    minutes, seconds = divmod(rest, 60)
    return f"{whole_hours:02d}:{minutes:02d}:{seconds:02d}"
