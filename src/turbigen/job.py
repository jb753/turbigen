"""Classes for submitting jobs to a queue."""

import os
import sys
import subprocess
from abc import ABC, abstractmethod
import dataclasses
from turbigen import util

SBATCH_FILE = "submit.sh"

ERROR_HANDLER_STR = r"""trap 'handle_error' ERR
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
}"""

logger = util.make_logger()

@dataclasses.dataclass
class BaseJob(ABC):
    """Define the interface for a queue job."""

    @abstractmethod
    def submit(self, config):
        """Send a job to the queue."""
        raise NotImplementedError()


@dataclasses.dataclass
class Slurm(BaseJob):
    """Submit a job to SLURM."""

    hours: float
    """Time limit in wall-clock hours for the job."""

    account: str
    """Name of the account to charge the compute time."""

    partition: str
    """Which cluster partition to use."""

    gres: str = ""
    """Generic consumable resources specification."""

    qos: str = ""
    """Quality of service level for this job in the queuing system."""

    tasks: int = 1
    """Number of tasks to run in parallel."""

    nodes: int = 1
    """Number of nodes to run the job on."""

    mail_type: str = 'FAIL'
    """Type of email notification to send."""

    hold_on_fail: bool = False
    """Whether to hold the node on failure."""

    def submit(self, fname):
        """Submit a config file as a SLURM job.

        Parameters
        ----------
        fname : Path
            Path to the config file to submit.

        """


        workdir = fname.parent

        # Error handler if needed
        if self.hold_on_fail:
            error_handler_str = ERROR_HANDLER_STR
        else:
            error_handler_str = ""

        # QOS if needed
        if self.qos:
            qos_str = f"#SBATCH --qos={self.qos}"
        else:
            qos_str = ""

        # Convert fractional hours to time string
        hours, frac_hours = divmod(self.hours, 1)
        mins = frac_hours * 60
        timestr = f"{hours:02d}:{mins:02d}:00"

        # Prepare a submission script
        sbatch_str = f"""#!/bin/bash
#SBATCH -J turbigen_{workdir.name}
#SBATCH -p {self.partition}
#SBATCH -A {self.account}
#SBATCH --mail-type={self.mail_type}
#SBATCH --nodes={self.nodes}
#SBATCH --ntasks={self.tasks}
#SBATCH --gres={self.gres}
#SBATCH --time={timestr}
{qos_str}

{error_handler_str}

# Invoke turbigen with the -J flag to ignore the job information
# in the config file and run directly on the compute node
turbigen -J {fname}

"""

        # Write out the submission script
        sbatch_path = workdir / SBATCH_FILE
        with open(sbatch_path, "w") as f:
            f.write(sbatch_str)

        # Run sbatch in the workdir specified in the config
        # This ensures that slurm.out is kept with the job
        sbatch_out = subprocess.run(
            ["sbatch", SBATCH_FILE],
            text=True,
            cwd=workdir,
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
        logger.iter(f"Submitted SLURM jobid={jid} in {workdir}")

        return jid



def _next_id(base_dir):
    # Find the ids of existing directories
    max_id = -1
    subdirs = next(os.walk(base_dir))[1]
    for d in subdirs:
        try:
            id_now = int(d)
            max_id = max(id_now, max_id)
        except ValueError:
            pass

    # Use the next available id
    next_id = max_id + 1

    return next_id



def _make_rundirs(base_dir, N):
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)
    next_id = _next_id(base_dir)
    ids = [next_id + n for n in range(N)]
    workdirs = [os.path.join(base_dir, f"{idn:04d}") for idn in ids]
    for d in workdirs:
        os.mkdir(d)
    return ids, workdirs



def submit_array(confs, basedir, Nmax):
    # Assign ids and make workdir for each config
    N = len(confs)
    logger.iter("Making workdirs...")
    ids, workdirs = _make_rundirs(basedir, N)

    job_name = os.path.basename(basedir) + "_array"

    maxstr = f"%{Nmax}" if Nmax else ""

    # Write a turbigen config to each dir
    logger.iter("Writing configs into workdirs...")
    for n in range(N):
        # Delete job info
        conf_out = confs[n].copy()
        conf_out.job = {}
        conf_out.workdir = workdirs[n]
        conf_out.write(os.path.join(workdirs[n], "config.yaml"))

    # Prepare submission script
    cj = confs[0].job
    nnode = cj.get("nodes", 1)
    ntask = cj.get("tasks", 1)
    gres = min((ntask, 4))
    sbatch_str = rf"""#!/bin/bash
#SBATCH -J turbigen_{job_name}
#SBATCH -p ampere
#SBATCH -A {cj['account']}
#SBATCH --mail-type=NONE
#SBATCH --nodes={nnode}
#SBATCH --ntasks={ntask}
#SBATCH --gres=gpu:{gres}
#SBATCH --time={'%02d' % cj['hours']}:00:00
#SBATCH --qos={cj.get('qos','gpu1')}
#SBATCH --array={ids[0]}-{ids[-1]}{maxstr}

cd {TURBIGEN_ROOT}
turbigen {basedir}/$(printf "%04d\n" $SLURM_ARRAY_TASK_ID)/config.yaml &>\
    {basedir}/$(printf "%04d\n" $SLURM_ARRAY_TASK_ID)/log_turbigen.txt

"""

    SBATCH_FILE = "submit_array.sh"

    # Write out
    with open(os.path.join(basedir, SBATCH_FILE), "w") as f:
        f.write(sbatch_str)

    orig_workdir = os.getcwd()
    os.chdir(basedir)

    # Run sbatch
    try:
        subprocess.check_output(
            f"sbatch {SBATCH_FILE}", shell=True, stderr=subprocess.PIPE
        )
        logger.iter("Submitted array job.")
    except subprocess.CalledProcessError as e:
        logger.info(e.stderr.decode("utf-8"))
        raise e
    os.chdir(orig_workdir)
