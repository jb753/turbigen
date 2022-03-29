"""Generate and submit a job using a set of input parameters."""
import json, glob, os, shutil
import numpy as np
import subprocess
import geometry

TS_SLURM_TEMPLATE = "submit.sh"


class ParameterSet:
    """Encapsulate the set of parameters sufficient to run a case."""

    # Variable lists
    _var_names = {
        "mean-line": [
            "phi",
            "psi",
            "Lam",
            "Al1",
            "Ma2",
            "eta",
            "ga",
        ],
        "bcond": [
            "To1",
            "Po1",
            "rgas",
            "Omega",
        ],
        "3d": [
            "htr",
            "Re",
            "Co",
            "cx_rat",
        ],
        "mesh": [
            "dx_c",
            "min_Rins",
            "A",
        ],
        "run": [
            "guess_file",
        ],
    }

    def __init__(self, var_dict):
        """Create a parameter set using a dictionary."""

        # Loop over all parameters and assign to class
        for outer_name, inner_names in self._var_names.items():
            for var in inner_names:
                setattr(self, var, var_dict[outer_name][var])

    @classmethod
    def from_json(cls, fname):
        """Create a parameter set from a file on disk."""

        # Load the data
        with open(fname, "r") as f:
            dat = json.load(f)

        # Sort out multi-dimensional thickness coeffs
        dat["mesh"]["A"] = np.reshape(
            dat["mesh"]["Aflat"], dat["mesh"]["shape_A"]
        )
        for key in ("Aflat", "shape_A"):
            dat["mesh"].pop(key)

        # Pass dict to the normal init method
        return cls(dat)

    def to_dict(self):
        """Nested dictionary for this object."""
        dat = {}
        for outer_name, inner_names in self._var_names.items():
            dat[outer_name] = {}
            for var in inner_names:
                dat[outer_name][var] = getattr(self, var)
        return dat

    def write_json(self, fname):
        """Write this parameter set to a JSON file."""

        dat = self.to_dict()

        # Deal with multi-dimensional thickness coeffs
        dat["mesh"]["Aflat"] = self.A.reshape(-1).tolist()
        dat["mesh"]["shape_A"] = self.A.shape
        dat["mesh"].pop("A")

        # Write the file
        with open(fname, "w") as f:
            json.dump(dat, f)

    @property
    def nondimensional(self):
        """Return parameters needed for non-dimensional mean-line design."""
        return {k: getattr(self, k) for k in self._var_names["mean-line"]}

    @property
    def dimensional(self):
        """Return parameters needed to scale mean-line design to dimensions."""
        return {
            k: getattr(self, k)
            for k in self._var_names["bcond"] + self._var_names["3d"]
        }

    @property
    def mesh(self):
        """Return parameters needed to mesh."""
        return {k: getattr(self, k) for k in self._var_names["mesh"]}

    @property
    def cfd_input_file(self):
        """Return parameters needed to pre-process the CFD."""
        return {
            k: getattr(self, k)
            for k in self._var_names["bcond"]
            + [
                "guess_file",
            ]
        }

    def copy(self):
        """Return a copy of this parameter set."""
        return ParameterSet(self.to_dict())

    def sweep(self, var, vals):
        """Return a list of ParametersSets with var varied over vals."""
        out = []
        for val in vals:
            Pnow = self.copy()
            setattr(Pnow, var, val)
            out.append(Pnow)
        return out


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
        res = os.popen("sacct -j %d" % jobid).read()
        print(res)
        if "COMPLETED" in res:
            return True
        elif "FAILED" in res:
            return False
        else:
            os.system("sleep %d" % interval)


def wait_for_file(fname):
    """Wait for a file to exist."""
    interval = 5
    while True:
        if os.path.isfile(fname):
            return True
        else:
            os.system("sleep %d" % interval)


def run_parallel(write_func, params_all, base_dir):
    """Run up to four sets of parameters in parallel."""
    N = len(params_all)
    # Set up N working directories
    workdirs = [
        prepare_run(write_func, params, base_dir) for params in params_all
    ]

    # Start the processes
    cmds = ["CUDA_VISIBLE_DEVICES=%d sh submit.sh" % n for n in range(N)]
    processes = [
        subprocess.Popen(cmd, cwd=wd, shell=True)
        for cmd, wd in zip(cmds, workdirs)
    ]

    # Wait for all processes
    for process in processes:
        if process:
            process.wait()

    # Load the processed data
    meta = []
    for workdir in workdirs:
        with open(os.path.join(workdir, "meta.json"), "r") as f:
            meta.append(json.load(f))

    # Return processed metadata
    return meta


def make_objective_and_constraint(write_func, params_default, base_dir):
    Aref = params_default.A

    def Amod(x):
        Am = Aref + 0.0
        Am[:, :, 1:-1] = x.reshape(2, 2, 2)
        return Am

    def sweep_A(x):
        A = [Amod(xi) for xi in x]
        return params_default.sweep("A", A)

    def _constraint(x):
        return [check_constraint(write_func, Pi) for Pi in sweep_A(x) ]

    def _objective(x):
        # Make parameters corresponding to each row of x
        params = sweep_A(x)
        # Run these parameter sets in parallel
        results = run_parallel(write_func, params, base_dir)
        # Apply stage loading constraint
        eta = np.reshape(
            [np.nan if not m else m["eta"] for m in results], (-1, 1)
        )
        psi = np.reshape([0.0 if not m else m["psi"] for m in results], (-1, 1))
        eta[psi < 0.99 * params_default.psi] = np.nan

        return eta

    x0 = Aref[:, :, 1:-1].reshape(1, -1)
    return _objective, _constraint, x0

def check_constraint(write_func, params):
    """Before writing a file, check that geometry constraints are OK."""
    try:
        write_func(params)
        return True
    except geometry.GeometryConstraintError:
        return False

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

    # Write out a turbostream grid in the new workdir
    write_func(params, input_file)

    # Save the parameters
    params.write_json(params_file)

    return os.path.abspath(workdir)
