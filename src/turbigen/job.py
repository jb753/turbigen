"""Classes for submitting jobs to a queue."""

import numpy as np
import sys
import subprocess
from abc import ABC, abstractmethod
import dataclasses
from turbigen import util
from pathlib import Path

SBATCH_FILE = "submit.sh"
SBATCH_ARRAY = "submit_array.sh"

ERROR_HANDLER_STR = r"""

trap 'handle_error' ERR
handle_error() {
    echo "# Command failed, starting a shell on ${HOSTNAME}. Attach using:" > failed.txt
    echo "ssh -t $HOSTNAME tmux att" >> failed.txt
    # Run the shell in a detached tmux session
    # Starting a tmux sesison without a tty seems flaky
    # Fix this by redirecting to a file handle
    export TMUX=""
    tmux new -d 'exec bash' &> /dev/null
    # Keep the job running until it times out
    sleep 36h
}

"""

logger = util.make_logger()


@dataclasses.dataclass
class BaseJob(ABC):
    """Define the interface for a queue job."""

    @abstractmethod
    def submit(self, config):
        """Send a job to the queue."""
        raise NotImplementedError()

    def submit_array(self, fnames):
        """Submit many jobs the queue."""
        # Trivial implementation as a default
        for fname in fnames:
            self.submit(fname)

    def to_dict(self):
        return dataclasses.asdict(self)


@dataclasses.dataclass
class Slurm(BaseJob):
    """Submit a job to SLURM."""

    hours: float
    """Time limit in wall-clock hours for the job."""

    account: str
    """Name of the account to charge compute time."""

    partition: str
    """Which cluster partition to use."""

    gres: str = None
    """Generic consumable resources specification."""

    qos: str = None
    """Quality of service level for the job."""

    tasks: int = 1
    """Number of tasks to run in parallel."""

    nodes: int = 1
    """Number of nodes to run the job on."""

    mail_type: str = "FAIL"
    """Type of email notification to send."""

    hold_on_fail: bool = False
    """Whether to hold the node on failure."""

    max_concurrent: int = 0
    """Maximum number of simultaneous jobs to run from an array, 0 for no limit."""

    def _get_sbatch_header(self, jobname):
        # QOS and gres if needed
        qos_str = f"#SBATCH --qos={self.qos}" if self.qos else ""
        gres_str = f"#SBATCH --gres={self.gres}" if self.gres else ""

        # Convert fractional hours to time string
        hours, frac_hours = divmod(self.hours, 1)
        mins = frac_hours * 60
        timestr = f"{int(hours):02d}:{int(mins):02d}:00"

        # Prepare a submission script
        sbatch_str = f"""#!/bin/bash
#SBATCH -J {jobname}
#SBATCH -p {self.partition}
#SBATCH -A {self.account}
#SBATCH --mail-type={self.mail_type}
#SBATCH --nodes={self.nodes}
#SBATCH --ntasks={self.tasks}
#SBATCH --time={timestr}
{gres_str}
{qos_str}"""

        return sbatch_str

    def submit(self, fname):
        """Submit a config file as a SLURM job.

        Parameters
        ----------
        fname : Path
            Path to the config file to submit.

        """

        workdir = fname.parent
        jobname = f"turbigen_{workdir.name}"

        # Get header and add the command
        sbatch_str = self._get_sbatch_header(jobname)

        # Error handler if needed
        if self.hold_on_fail:
            sbatch_str += ERROR_HANDLER_STR

        sbatch_str += f"""
turbigen --no-job {fname}

"""
        self.sbatch(sbatch_str, workdir / SBATCH_FILE)

    def sbatch(self, sbatch_str, sbatch_path):
        """Write out the sbatch script and run through sbatch."""

        # Write out the submission script
        with sbatch_path.open("w") as f:
            f.write(sbatch_str)

        # Run sbatch in the workdir specified in the config
        # This ensures that slurm.out is kept with the job
        sbatch_out = subprocess.run(
            ["sbatch", sbatch_path.name],
            text=True,
            cwd=sbatch_path.parent,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Check for errors
        if sbatch_out.returncode != 0:
            logger.iter(sbatch_out.stderr)
            logger.iter("Error submitting job, exiting.")
            sys.exit(1)

        # Extract the job id from the output and print it
        jid = sbatch_out.stdout.strip().split(" ")[-1]
        logger.iter(f"Submitted SLURM jobid={jid} in {sbatch_path.parent}")

    def submit_array(self, fnames):
        """Submit many config files as a SLURM job array.

        Parameters
        ----------
        fnames : list of Path
            List of paths to the config files to submit.

        """

        # Check that the fnames are all in the same directory
        base_dir = fnames[0].parent.parent
        for fname in fnames:
            if fname.parent.parent != base_dir:
                raise ValueError(
                    "All config files must be in the same directory for job arrays."
                )

        # Check that the directories are consecutive numbers
        try:
            nums = [int(fname.parent.name) for fname in fnames]
            assert np.all(np.diff(nums) == 1)
        except (ValueError, AssertionError):
            raise ValueError(
                "Job array must be a consecutive range of numbered directories."
            )

        width = len(str(fnames[0].parent.name))

        maxstr = "%{self.max_concurrent}" if self.max_concurrent else ""

        sbatch_str = self._get_sbatch_header(f"turbigen_{base_dir.name}_array")
        sbatch_str += f"#SBATCH --array={nums[0]}-{nums[-1]}{maxstr}"
        if self.hold_on_fail:
            sbatch_str += ERROR_HANDLER_STR
        sbatch_str += rf"""

WORKDIR="{base_dir}/$(printf "%0{width}d\n" $SLURM_ARRAY_TASK_ID)"

turbigen --no-job $WORKDIR/config.yaml

"""
        sbatch_path = base_dir / SBATCH_ARRAY
        self.sbatch(sbatch_str, sbatch_path)


@dataclasses.dataclass
class Local(BaseJob):
    """Submit a job to a local queue."""

    queue_file: str = "~/.turbigen/queue.txt"
    """List of pending jobs to monitor."""

    def __post_init__(self):
        if isinstance(self.queue_file, str):
            self.queue_file = Path(self.queue_file).expanduser()

        self.queue_file.parent.mkdir(parents=True, exist_ok=True)

    def to_dict(self):
        # Built-in dataclasses method gets us most of the way there
        data = dataclasses.asdict(self)
        data["queue_file"] = str(data["queue_file"])
        return data

    def submit(self, fname):
        """Submit a config file to the local queue.

        Parameters
        ----------
        fname : Path
            Path to the config file to submit.

        """

        # Check that the config file exists
        if not fname.exists():
            raise FileNotFoundError(f"Config file {fname} does not exist.")

        # Append fname to the queue file
        with open(self.queue_file, "a") as f:
            f.write(f"{fname}\n")

        logger.iter(f"Submitted local job {fname}")
