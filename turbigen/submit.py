"""Generate and submit a job using a set of input parameters."""
import json, glob, os, shutil
import numpy as np
import subprocess
from copy import deepcopy
import geometry

TS_SLURM_TEMPLATE = "submit.sh"

def read_params(params_file):
    """Load a parameters file and format data where needed.

    This is a thin wrapper around JSON loading."""

    # Load the data
    with open(params_file, "r") as f:
        params = json.load(f)

    # Sort out multi-dimensional thickness coeffs
    params["mesh"]["A"] = np.reshape(
        params["mesh"]["Aflat"], params["mesh"]["shape_A"]
    )
    for key in ("Aflat", "shape_A"):
        params["mesh"].pop(key)

    return params


def write_params(params, params_file):

    # Copy the nested dict
    pnow = {}
    for k, v in params.items():
        pnow[k] = v.copy()

    # Flatten the thickness coeffs and store their shape
    pnow["mesh"]["Aflat"] = pnow["mesh"]["A"].reshape(-1).tolist()
    pnow["mesh"]["shape_A"] = pnow["mesh"]["A"].shape
    pnow["mesh"].pop("A")
    # Write the file
    with open(params_file, "w") as f:
        json.dump(pnow, f)


def _create_new_job(base_dir, slurm_template):
    """Prepare a job working directory for TS run."""

    # Get all direcories in base dir and convert to integers
    dnames = glob.glob(base_dir + "/*")
    runids = [
        -1,
    ]
    for i in range(len(dnames)):
        try:
            runids.append(int(os.path.basename(dnames[i])))
        except ValueError:
            pass

    # Add one to the maximum identifier
    new_id = max(runids) + 1

    # Make a working directory with unique filename
    workdir = os.path.join(base_dir, "%04d" % new_id)
    print("Creating %s" % workdir)
    os.mkdir(workdir)

    # copy in a slurm submit script
    new_slurm = os.path.join(workdir, "submit.sh")
    shutil.copy(slurm_template, new_slurm)

    # Append run id to the slurm job name
    os.system(
        "sed -i 's/jobname/turbigen_%s_%04d/' %s"
        % (base_dir, new_id, new_slurm)
    )

    # Return the working directory so that we can save input files there
    return workdir

def wait_for_job(jobid):
    """Wait for a jobid to complete."""
    interval = 5
    while True:
        res = os.popen('sacct -j %d' % jobid).read()
        print(res)
        if "COMPLETED" in res:
            return True
        elif "FAILED" in res:
            return False
        else:
            os.system('sleep %d' % interval)

def wait_for_file(fname):
    """Wait for a file to exist."""
    interval = 5
    while True:
        if os.path.isfile(fname):
            return True
        else:
            os.system('sleep %d' % interval)

def run_parallel( write_func, params_all, base_dir ):
    """Run up to four sets of parameters in parallel."""
    N = len(params_all)
    # Set up N working directories
    workdirs = [prepare_run(write_func, params, base_dir) for params in params_all]

    # base_dir_N = (base_dir,)*N
    # args = zip(write_func_N, params_all, base_dir_N, range(N))
    # pool.map(run_star, args, chunksize=1)
    # from multiprocessing import Pool
    # # start all programs
    cmds = ['CUDA_VISIBLE_DEVICES=%d sh submit.sh' % n for n in range(N)]
    processes = []
    for cmd, wd in zip(cmds, workdirs):
        if wd:
            processes.append(subprocess.Popen(cmd,cwd=wd,shell=True)) 
        else:
            processes.append(None)
    for process in processes:
        if process:
            process.wait()

    # Load the processed data
    meta = []
    for workdir in workdirs:
        if workdir:
            with open(os.path.join(workdir, 'meta.json'), "r") as f:
                meta.append(json.load(f))
        else:
            meta.append(None)

    return meta

def make_objective( write_func, params_default, base_dir  ):
    def _objective(x):
        # Make parameters dicts corresponding to each row of x
        params = []
        for i in range(x.shape[0]):
            param_now = deepcopy(params_default)
            param_now["mesh"]["A"][:,:,1:-1] = x[i].reshape(2,2,2)
            params.append(param_now)
        # Run these parameter sets in parallel
        results = run_parallel( write_func, params, base_dir)

        # Apply stage loading constraint
        eta = np.reshape([np.nan if not m else m["eta"] for m in results],(-1,1))
        psi = np.reshape([0. if not m else m["psi"] for m in results],(-1,1))
        eta[psi<0.99*params_default["mean-line"]["psi"]] = np.nan

        return eta

    x0 = params_default["mesh"]["A"][:,:,1:-1].reshape(1,-1)
    return _objective, x0

def prepare_run(write_func, params, base_dir):
    """Get a working directory ready for a set of parameters.

    Parameters
    ----------
    write_func: callable
        Function with signature `write_func(fname, params)` that will write an
        input file at the requested location given the parameter set.
    params : dict
        All parameters required to design, mesh, and run the case.
    base_dir : string
        Folder for the current set of cases.

    Returns
    -------
    workdir:
        The created directory.

    jobid: int
        Job ID for the submitted job.

    """

    # Make the base dir if it does not exist already
    if not os.path.isdir(base_dir):
        os.mkdir(base_dir)

    # Get a working directory and file names
    workdir = _create_new_job(base_dir, TS_SLURM_TEMPLATE)
    input_file = os.path.join(workdir, "input.hdf5")
    params_file = os.path.join(workdir, "params.json")

    try:
        # Write out a turbostream grid in the new workdir
        write_func(input_file, params)
    except geometry.ConstraintError:
        # If this design does not work, then return sentinel value
        return None

    # Save the parameters
    write_params(params, params_file)

    # # Change to the working director and run
    # os.chdir(workdir)
    #     jid = int(os.popen("sbatch submit.sh").read().split(" ")[-1])
    #     retval = wait_for_job(jid)
    # os.chdir("../..")

    return os.path.abspath(workdir)
