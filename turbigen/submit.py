"""Generate and submit a job using a set of input parameters."""
import json, glob, os, shutil
import numpy as np
import subprocess
from . import geometry, tabu


TURBIGEN_ROOT = os.path.join("/".join(__file__.split("/")[:-1]), "..")
TS_SLURM_TEMPLATE = os.path.join(TURBIGEN_ROOT, "submit.sh")
TABU_SLURM_TEMPLATE = os.path.join(TURBIGEN_ROOT, "submit_search.sh")

OBJECTIVE_KEYS = ["eta_lost", "psi", "phi", "Lam", "Ma2", "runid"]


def _concatenate_dict(list_of_dicts):
    return {
        k: np.reshape([d[k] for d in list_of_dicts], (-1, 1))
        for k in list_of_dicts[0]
    }


def _make_workdir(base_dir, slurm_template):
    """Make working directory for a single cluster job."""

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
    os.mkdir(workdir)

    # copy in a slurm submit script
    new_slurm = os.path.join(workdir, "submit.sh")
    shutil.copy(slurm_template, new_slurm)

    # Append run id to the slurm job name
    sedcmd = "sed -i 's?jobname?turbigen_%s_%04d?' %s" % (
        base_dir,
        new_id,
        new_slurm,
    )
    os.system(sedcmd)

    # Return the working directory so that we can save input files there
    return workdir


def _write_input(write_func, params, base_dir):
    """Write out an input file given a set of turbine parameters.

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
    workdir = _make_workdir(base_dir, TS_SLURM_TEMPLATE)
    input_file = os.path.join(workdir, "input.hdf5")
    params_file = os.path.join(workdir, "params.json")

    # Write out a turbostream grid in the new workdir
    write_func(params, input_file)

    # Save the parameters
    params.write_json(params_file)

    return os.path.abspath(workdir)


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
            "recamber",
            "tte",
        ],
        "run": [
            "guess_file",
            "rtol",
            "dampin",
        ],
    }

    def __init__(self, var_dict):
        """Create a parameter set using a dictionary."""

        # Loop over all parameters and assign to class
        for outer_name, inner_names in self._var_names.items():
            for var in inner_names:
                setattr(self, var, var_dict[outer_name][var])

    def __repr__(self):
        return (
            "phi=%.2f, psi=%.2f, Lam=%.2f, Ma2=%.2f"
            % (self.phi, self.psi, self.Lam, self.Ma2)
            )


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

    @classmethod
    def from_default(cls):
        """Create a default parameter set."""
        module_path = os.path.abspath(os.path.dirname(__file__))
        return cls.from_json(os.path.join(module_path, "default_params.json"))

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
            for k in self._var_names["bcond"] + ["guess_file", "dampin"]
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


def _run_parameters(write_func, params_all, base_dir):
    """Run one or more parameters sets in parallel."""

    try:
        N = len(params_all)
    except AttributeError:
        N = 1
        params_all = [
            params_all,
        ]

    # Set up N working directories
    workdirs = [
        _write_input(write_func, params, base_dir) for params in params_all
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


def _param_from_x(x, param_datum):
    """Perturb a datum turbine using a design vector x."""

    param = param_datum.copy()

    # First four elements are recambering angles
    param.recamber = x[:4].tolist()

    # Last eight elements are section shape parameters
    xr = np.reshape(x[4:], (2, 4))
    param.A = np.stack(
        [
            geometry.A_from_Rle_thick_beta(
                xi[0], xi[1:3], xi[3], param_datum.tte
            )
            for xi in xr
        ]
    )

    return param


def _assemble_bounds(
    Rle=(0.06, 0.5),
    dchi_in=(-30.0, 30),
    dchi_out=(-10.0, 10.0),
    beta=(8.0, 45.0),
    thick=(0.05, 0.5),
):
    """With pairs of bounds for each variable, assemble design vec limits."""
    return np.column_stack(
        ((dchi_in,) + (dchi_out,)) * 2 + ((Rle,) + (thick,) * 2 + (beta,)) * 2
    )


def _assemble_x0(Rle=0.08, dchi_in=-5.0, dchi_out=0.0, beta=10.0, thick=0.25):
    return np.atleast_2d(
        (dchi_in, dchi_out) * 2 + (Rle, thick, thick, beta) * 2
    )


def _assemble_dx(
    dRle=0.02, ddchi_in=2.0, ddchi_out=1.0, dbeta=2.0, dthick=0.04
):
    return np.atleast_2d(
        (ddchi_in, ddchi_out) * 2 + (dRle, dthick, dthick, dbeta) * 2
    )


def _constrain_x_param(x, write_func, param_datum):
    lower, upper = _assemble_bounds()
    input_ok = (x >= lower).all() and (x <= upper).all()
    param = _param_from_x(x, param_datum)
    if input_ok:
        return check_constraint(write_func, param)
    else:
        return False


def _metadata_to_y(meta):
    """Convert a metadata dictionary into objective vector y."""
    return np.array([float(meta[k]) for k in OBJECTIVE_KEYS])


def _param_to_y(param):
    """Convert parameter set into objective vector y."""
    y = np.array([getattr(param, k, np.nan) for k in OBJECTIVE_KEYS])
    return y


def _wrap_for_optimiser(write_func, param_datum, base_dir):
    """A closure that wraps turbine creation and running for the optimiser."""

    def _constraint(x):
        return [_constrain_x_param(xi, write_func, param_datum) for xi in x]

    def _objective(x):
        # Get parameter sets for all rows of x
        params = [_param_from_x(xi, param_datum) for xi in x]
        # Run these parameter sets in parallel
        metadata = _run_parameters(write_func, params, base_dir)
        # Extract the results
        y = np.stack([_metadata_to_y(m) for m in metadata])
        # Check error
        y_target = np.atleast_2d(_param_to_y(param_datum))
        err = y / y_target - 1.0
        # NaN out results that deviate too much from target
        ind_good = np.all(np.abs(err[:, 1:-1]) < param_datum.rtol, axis=1)
        y[~ind_good, 0] = np.nan
        return y

    return _objective, _constraint


def run_search(param, base_name):
    base_dir = os.path.join(TURBIGEN_ROOT, "run", base_name)
    if not os.path.isdir(base_dir):
        os.mkdir(base_dir)

    datum_file = os.path.join(base_dir, "datum_param.json")
    slurm_file = os.path.join(base_dir, "submit.sh")

    param.write_json(datum_file)
    shutil.copy(TABU_SLURM_TEMPLATE, slurm_file)

    # Append run id to the slurm job name
    sedcmd = "sed -i 's?jobname?turbigen_%s?' %s" % (base_name, slurm_file)
    os.system(sedcmd)

    subprocess.Popen("sbatch submit.sh", cwd=base_dir, shell=True).wait()


def _run_search(write_func):
    """Perform a tabu search of blade geometries for a parameter set."""

    base_dir = os.getcwd()

    mem_file = os.path.join(base_dir, "mem_tabu.json")
    datum_file = os.path.join(base_dir, "datum_param.json")

    if not os.path.isfile(datum_file):
        raise Exception("No datum parameters found.")

    param = ParameterSet.from_json(datum_file)

    # Wrapped objective and constraint
    obj, constr = _wrap_for_optimiser(write_func, param, base_dir)

    # Initial guess, step, tolerance
    x0 = _assemble_x0()
    dx = _assemble_dx()
    tol = dx / 4.0

    # Get a solution with low damping and use as initial guess
    param_damp = _param_from_x(x0.reshape(-1), param)
    param_damp.dampin = 3.0
    meta = _run_parameters(write_func, param_damp, base_dir)
    param.guess_file = os.path.join(
        base_dir, meta[0]["runid"], "output_avg.hdf5"
    )
    param.write_json(datum_file)

    # Setup the seach
    ts = tabu.TabuSearch(obj, constr, x0.shape[1], 6, tol, j_obj=(0,))

    if os.path.isfile(mem_file):
        ts.resume(mem_file)
    else:
        ts.mem_file = mem_file
        ts.search(x0, dx)


def check_constraint(write_func, params):
    """Before writing a file, check that geometry constraints are OK."""
    try:
        write_func(params)
        return True
    except geometry.GeometryConstraintError:
        return False
