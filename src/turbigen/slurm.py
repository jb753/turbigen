"""Functions for running turbigen using the SLURM queue."""

import os
import subprocess
from turbigen.util import make_logger

SBATCH_FILE = "submit.sh"
YAML_FILE = "config.yaml"
LOG_FILE = "log_turbigen.txt"
TURBIGEN_ROOT = "/".join(__file__.split("/")[:-2])


logger = make_logger()


def _parse_jid(s):
    return int(s.decode("utf-8").strip().split(" ")[-1])


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


def _make_rundir(base_dir):
    """Inside base_dir, make new work dir in sequential integer format."""
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)

    # Use the next available id
    next_id = _next_id

    # Make a working directory with unique filename
    workdir = os.path.join(base_dir, f"{next_id:04d}")
    os.mkdir(workdir)

    # Return the working directory so that we can save input files there
    return workdir


def _make_rundirs(base_dir, N):
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)
    next_id = _next_id(base_dir)
    ids = [next_id + n for n in range(N)]
    workdirs = [os.path.join(base_dir, f"{idn:04d}") for idn in ids]
    for d in workdirs:
        os.mkdir(d)
    return ids, workdirs


def submit(conf, basedir=None, verbose=True):
    """Given a configuration, prepare and submit a SLURM job.

    Parameters
    ----------
    conf: Config object
        The"""

    conf.check()

    if basedir:
        workdir = _make_rundir(basedir)
        conf.workdir = workdir
        job_name = os.path.basename(basedir) + "_" + os.path.basename(workdir)
    else:
        workdir = conf.workdir
        job_name = os.path.basename(workdir)
    workdir = os.path.abspath(workdir)

    # Get paths
    sbatch_path = os.path.join(workdir, SBATCH_FILE)
    yaml_path = os.path.join(workdir, YAML_FILE)
    log_path = os.path.join(workdir, LOG_FILE)

    os.makedirs(workdir, exist_ok=True)

    # Put placeholders in template
    cj = conf.job

    depstr = (
        f"#SBATCH --dependency=afterany:{dep_jid}"
        if (dep_jid := cj.get("dependency"))
        else ""
    )

    hold = conf.job.get("hold_on_fail", False)
    if hold:
        error_handler_str = r"""trap 'handle_error' ERR
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
    else:
        error_handler_str = ""

    hours, frac_hours = divmod(cj["hours"], 1)
    mins = frac_hours * 60
    timestr = f"{hours:02d}:{mins:02d}:00"

    nnode = cj.get("nodes", 1)
    ntask = cj.get("tasks", 1)
    gres = min((ntask, 4))

    sbatch_str = f"""#!/bin/bash
#SBATCH -J turbigen_{job_name}
#SBATCH -p ampere
#SBATCH -A {cj['account']}
#SBATCH --mail-type=NONE
#SBATCH --nodes={nnode}
#SBATCH --ntasks={ntask}
#SBATCH --gres=gpu:{gres}
#SBATCH --time={timestr}
#SBATCH --qos={cj.get('qos','gpu1')}

{error_handler_str}

{depstr}

turbigen {yaml_path} &> {log_path}

"""

    if dep_jid := cj.get("dependency"):
        sbatch_str = sbatch_str.replace(
            "#DEPENDENCY", f"#SBATCH --dependency=afterany:{dep_jid}"
        )

    # Delete job info
    conf_out = conf.copy()
    conf_out.job = {}
    conf_out.write(yaml_path)

    # Write out
    with open(sbatch_path, "w") as f:
        f.write(sbatch_str)

    orig_workdir = os.getcwd()
    os.chdir(workdir)

    # Run sbatch
    try:
        sbatch_out = subprocess.check_output(
            f"sbatch {SBATCH_FILE}", shell=True, stderr=subprocess.PIPE
        )
    except subprocess.CalledProcessError as e:
        logger.info(e.stderr.decode("utf-8"))
        raise e

    jid = _parse_jid(sbatch_out)
    os.chdir(orig_workdir)
    logger.iter(f"Submitted SLURM jobid={jid} in {workdir}")

    return jid


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


def submit_batches(confs, basedir, Nrun, verbose=True):
    """Submit a list of configs in batches with maximum of Nrun concurrent jobs."""

    Nconf = len(confs)

    if Nrun > Nconf:
        conf_split = [
            confs,
        ]
    else:
        conf_split = [confs[irun::Nrun] for irun in range(Nrun)]

    iconf = 1
    for conf_set in conf_split:
        dependency = None
        for conf in conf_set:
            if 0 < iconf:
                logger.iter("\r", end="")
            conf.job["dependency"] = dependency
            submit(conf, basedir, verbose=False)
            logger.iter(f"Submitted {iconf}/{Nconf}", end="")
            iconf += 1
    logger.iter("")
