"""Generate and submit a job using a set of input parameters."""
import json, glob, os, shutil
import numpy as np
import subprocess
from . import geometry, tabu, design


TURBIGEN_ROOT = os.path.join("/".join(__file__.split("/")[:-1]), "..")
TS_SLURM_TEMPLATE = os.path.join(TURBIGEN_ROOT, "submit.sh")
TABU_SLURM_TEMPLATE = os.path.join(TURBIGEN_ROOT, "submit_search.sh")

OBJECTIVE_KEYS = ["eta_lost", "psi", "phi", "Lam", "runid"]

X_KEYS = [
    "dchi_in",
    "dchi_out",
    "stag",
    "Rle",
    "thick_ps",
    "thick_ss1",
    "thick_ss2",
    "beta",
]

X_BOUNDS = {
    "dchi_in": (-16.0, 16.0),
    "dchi_out": (-5.0, 5.0),
    "stag": (-90.0, 90.0),
    "Rle": (0.05, 0.5),
    "thick_ps": (0.1, 0.5),
    "thick_ss1": (0.1, 0.5),
    "thick_ss2": (0.1, 0.5),
    "beta": (8.0, 24.0),
}

X_GUESS = {
    "dchi_in": 0.0,
    "dchi_out": 0.0,
    "stag": 0.0,
    "Rle": 0.06,
    "thick_ps": 0.2,
    "thick_ss1": 0.34,
    "thick_ss2": 0.24,
    "beta": 18.0,
}

X_STEP = {
    "dchi_in": 1.0,
    "dchi_out": 0.5,
    "stag": 1.0,
    "Rle": 0.01,
    "thick_ps": 0.02,
    "thick_ss1": 0.04,
    "thick_ss2": 0.04,
    "beta": 2.0,
}


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
            "stag",
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
        return "phi=%.2f, psi=%.2f, Lam=%.2f, Ma2=%.2f" % (
            self.phi,
            self.psi,
            self.Lam,
            self.Ma2,
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

    def set_stag(self):

        # Evaulate the meanline design to get target flow angles
        stg = design.nondim_stage_from_Lam(**self.nondimensional)

        # Set stagger guess
        tanAl_rot = np.tan(np.radians(stg.Alrel[1:]))
        tanAl_sta = np.tan(np.radians(stg.Al[:2]))
        self.stag[0] = np.degrees(np.arctan(np.mean(tanAl_sta)))
        self.stag[1] = np.degrees(np.arctan(np.mean(tanAl_rot)))


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


def _param_from_x(x, param_datum, row_index):
    """Perturb a datum turbine using a design vector x."""

    param = param_datum.copy()

    recam = x[:2]
    stag = x[2]

    offset = row_index * 2
    param.recamber = list(param.recamber)
    param.recamber[0 + offset] += recam[0]
    param.recamber[1 + offset] += recam[1]

    Rle = x[3]
    Ale = np.sqrt(2.0 * Rle)
    thick_ps = [0.5 * (x[4] + Ale), x[4]]
    thick_ss = x[5:7]
    thick = np.stack((thick_ps, thick_ss))
    if row_index:
        thick = np.flip(thick, axis=0)
    beta = x[7]

    param.stag = list(param.stag)
    param.stag[row_index] = stag

    Anew = geometry.A_from_Rle_thick_beta(Rle, thick, beta, param_datum.tte)

    param.A = param.A + 0.0
    param.A[row_index] = Anew

    return param


def _assemble_bounds(nrow):
    """Return a (2, nx*nrow) matrix of bounds for some number of rows."""
    return np.tile(np.column_stack([X_BOUNDS[k] for k in X_KEYS]), nrow)


def _assemble_guess(nrow):
    return np.tile(np.atleast_2d([X_GUESS[k] for k in X_KEYS]), nrow)


def _assemble_step(nrow):
    return np.tile(np.atleast_2d([X_STEP[k] for k in X_KEYS]), nrow)


def _constrain_x_param(x, write_func, param_datum, irow):
    lower, upper = _assemble_bounds(1)
    input_ok = (x >= lower).all() and (x <= upper).all()
    print(x >= lower)
    print(x <= upper)
    param = _param_from_x(x, param_datum, row_index=irow)
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


def _wrap_for_optimiser(write_func, param_datum, base_dir, irow):
    """A closure that wraps turbine creation and running for the optimiser."""

    def _constraint(x):
        return [
            _constrain_x_param(xi, write_func, param_datum, irow) for xi in x
        ]

    def _objective(x):
        # Get parameter sets for all rows of x
        params = [_param_from_x(xi, param_datum, irow) for xi in x]
        # Run these parameter sets in parallel
        metadata = _run_parameters(write_func, params, base_dir)
        # Extract the results
        y = np.stack([_metadata_to_y(m) for m in metadata])
        # Check error
        y_target = np.atleast_2d(_param_to_y(param_datum))
        err = y / y_target - 1.0
        ind_low_error = np.all(np.abs(err[:, 1:-1]) < param_datum.rtol, axis=1)
        # Check convergence
        ind_has_conv = [mi["resid"] < 1.0 for mi in metadata]
        # NaN out results that deviate too much from target
        ind_good = np.logical_and(ind_low_error, ind_has_conv)
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
    """Tabu search both blade rows in turn."""

    # Set up some file paths
    base_dir = os.getcwd()
    datum_file = os.path.join(base_dir, "datum_param.json")
    if not os.path.isfile(datum_file):
        raise Exception("No datum parameters found.")

    # Create the datum parameter set
    param = ParameterSet.from_json(datum_file)

    # Evaulate the meanline design to get target flow angles
    stg = design.nondim_stage_from_Lam(**param.nondimensional)
    Al_target = np.array((stg.Al[1], stg.Alrel[2]))

    # Record target circulation
    Co_target = np.array(param.Co)

    # Set stagger guess
    tanAl_rot = np.tan(np.radians(stg.Alrel[1:]))
    tanAl_sta = np.tan(np.radians(stg.Al[:2]))
    param.stag[0] = np.degrees(np.arctan(np.mean(tanAl_sta)))
    param.stag[1] = np.degrees(np.arctan(np.mean(tanAl_rot)))

    # Give up if the initial design violates constraint
    if not check_constraint(write_func, param):
        raise Exception("Violating constraint at datum design.")

    # Run a robust damped solution and use as initial guess
    print("HIGH-DAMPING INITIAL GUESS")
    param_damp = param.copy()
    param_damp.dampin = 3.0
    meta_damp = _run_parameters(write_func, param_damp, base_dir)[0]
    param.guess_file = os.path.join(
        base_dir, meta_damp["runid"], "output_avg.hdf5"
    )

    # Tune deviation, circulation, effy using fixed-point iteration
    print("CORRECTING ANGLES AND EFFY")
    print("  Target Al = %s" % str(Al_target))
    print("  Target Co = %s" % str(Co_target))
    for i in range(20):

        # Relaxation factor for increased stability
        rf = 0.3 if i < 5 else 1.0

        # Run initial guess to see how much deviation we have
        meta_dev = _run_parameters(write_func, param, base_dir)[0]
        Al_now = np.array((meta_dev["Alrel"][1], meta_dev["Alrel"][3]))
        Co_now = np.array(meta_dev["Co"])

        # Calculate corrections for the flow angles
        dev_vane, dev_blade = Al_target - Al_now
        param.recamber[1] += dev_vane * rf
        param.recamber[3] -= dev_blade * rf
        print("  Al = %s" % str(Al_now))

        # Update polytropic effy
        param.eta = meta_dev["eta"]
        print("  Effy = %.3f" % param.eta)

        # Update circulation coeff
        Co_old = np.array(param.Co)
        param.Co = (
            Co_old * (1.0 - rf) + rf * Co_old * Co_target / Co_now
        ).tolist()
        print("  Co = %s" % str(Co_now))

        # Use most recent solution as initial guess
        param.guess_file = os.path.join(
            base_dir, meta_dev["runid"], "output_avg.hdf5"
        )

        # Check for convergence
        err_Co = np.max(np.abs(Co_now / Co_target - 1.0))
        err_Al = np.max(np.abs(Al_target - Al_now))
        if (err_Co < param.rtol) and (err_Al < 0.5):
            print("  Tolerance reached, breaking.")
            break

    # Write out datum parameters for later inspection
    param.write_json(datum_file)

    # Optimise rotor first, then stator
    for irow in range(2):
        print("*******************")
        print("**OPTIMISE ROW %d**" % irow)
        print("*******************")
        param = _search_row(base_dir, param, write_func, irow)

    print("**************************")
    print("**FINISHED               *")
    print("**************************")


def _search_row(base_dir, param, write_func, irow):
    """Perform a tabu search of blade geometries for one row."""

    # Initial guess, step, tolerance
    x0 = _assemble_guess(1)
    dx = _assemble_step(1)
    tol = dx / 2.0

    x0[0, 2] = param.stag[irow] + 0.0

    # Give up if the initial guess violates constraint
    write_func(_param_from_x(x0[0], param, irow))

    # if not _constrain_x_param(x0[0], write_func, param, irow):
    #     raise Exception("Violating constraint at initial guess.")

    # Wrapped objective and constraint
    obj, constr = _wrap_for_optimiser(write_func, param, base_dir, irow)

    # Setup the seach
    ts = tabu.TabuSearch(obj, constr, x0.shape[1], 5, tol, j_obj=(0,))

    # Run the search
    ts.mem_file = os.path.join(base_dir, "mem_tabu_row_%d.json" % irow)
    ts.search(x0, dx)

    # Make a copy of the optimal solution
    xopt, yopt = ts.mem_med.get(0)

    id_opt = yopt[0, -1]
    id_opt_dir = os.path.join(base_dir, "%04d" % round(id_opt))
    opt_dir = os.path.join(base_dir, "opt_row_%d" % irow)
    shutil.copytree(id_opt_dir, opt_dir)

    # Now return the parameters corresponding to optimum
    return _param_from_x(xopt[0], param, irow)


def check_constraint(write_func, params):
    """Before writing a file, check that geometry constraints are OK."""
    try:
        write_func(params)
        return True
    except geometry.GeometryConstraintError:
        return False
