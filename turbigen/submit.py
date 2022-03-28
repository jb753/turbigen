"""Generate and submit a job using a set of input parameters."""
import json, glob, os, shutil
import numpy as np

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


def run(write_func, params, base_dir):
    """Submit a cluster job corresponding to the given parameter set.

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

    # Write out a turbostream grid in the new workdir
    write_func(input_file, params)

    # Save the parameters
    write_params(params, params_file)

    # Change to the working director and run
    os.chdir(workdir)
    jid = int(os.popen("sbatch submit.sh").read().split(" ")[-1])
    os.chdir("../..")

    return jid
