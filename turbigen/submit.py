"""Generate and submit a job using a set of input parameters."""
import json, glob, os, shutil, subprocess, sys
import numpy as np
from . import geometry, tabu, design


TURBIGEN_ROOT = os.path.join("/".join(__file__.split("/")[:-1]), "..")
TS_SLURM_TEMPLATE = os.path.join(TURBIGEN_ROOT, "submit.sh")
TABU_SLURM_TEMPLATE = os.path.join(TURBIGEN_ROOT, "submit_search.sh")

OBJECTIVE_KEYS = [
    "eta_lost_percent",
    "psi",
    "phi",
    "Lam",
    "Ma2",
    "runid",
    "resid",
]
CONSTR_KEYS = ["psi", "phi", "Lam", "Ma2"]

X_KEYS = [
    "dchi_in",
    "dchi_out",
    "stag",
    "Rle",
    "thick_ps",
    "thick_ss",
    "beta",
]

X_BOUNDS = {
    "dchi_in": (-0.0, 30.0),
    "dchi_out": (-5.0, 5.0),
    "stag": (-90.0, 90.0),
    "Rle": (0.05, 0.5),
    "thick_ps": (0.1, 0.5),
    "thick_ss": (0.1, 0.5),
    "beta": (8.0, 24.0),
}

X_BOUNDS_REL = {
    "dchi_in": (-2.0, 6.0),
    "dchi_out": (-10.0, 10.0),
    "stag": (-10.0, 10.0),
    "Rle": (-5.0, 20.0),
    "thick_ps": (-5.0, 20.0),
    "thick_ss": (-5, 20.0),
    "beta": (-4.0, 4.0),
}

X_GUESS = {
    "dchi_in": 0.0,
    "dchi_out": 0.0,
    "stag": 0.0,
    "Rle": 0.1,
    "thick_ps": 0.25,
    "thick_ss": 0.34,
    "beta": 16.0,
}

X_STEP = {
    "dchi_in": 5.0,
    "dchi_out": 1.0,
    "stag": 2.0,
    "Rle": 0.01,
    "thick_ps": 0.02,
    "thick_ss": 0.02,
    "beta": 4.0,
}

def load_results(metadata_file):
    with open(metadata_file) as f:
        return json.load(f)


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

def _make_rundir(base_dir):
    """Inside base_dir, make new work dir in four-digit integer format."""
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)
    # Make a working directory with unique filename
    case_str = "%012d" % np.random.randint(0,1e12)
    workdir = os.path.join(base_dir, case_str)
    os.mkdir(workdir)
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
            "Al3",
            "Ma2",
            "eta",
            "ga",
            "loss_rat",
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
            "resolution",
            "dx_c",
            "min_Rins",
            "A",
            "recamber",
            "tte",
            "stag",
        ],
        "run": ["guess_file", "rtol", "ilos", "resid","nchange"],
    }

    def __init__(self, var_dict):
        """Create a parameter set using a dictionary."""

        # Loop over all parameters and assign to class
        for outer_name, inner_names in self._var_names.items():
            for var in inner_names:
                try:
                    setattr(self, var, var_dict[outer_name][var])
                except KeyError:
                    setattr(self, var, None)



        if not np.any(self.stag):
            self.set_stag()

    def __repr__(self):
        return "phi=%.2f, psi=%.2f, Lam=%.2f, Ma2=%.2f, Co=%.2f,%.2f" % (
            self.phi,
            self.psi,
            self.Lam,
            self.Ma2,
            self.Co[0],
            self.Co[1],
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
                dat[outer_name][var] = np.copy(getattr(self, var))
        return dat

    def write_json(self, fname):
        """Write this parameter set to a JSON file."""

        # Check stagger is sensible
        if not np.any(self.stag):
            raise Exception('Bad stagger angles.')

        dat = self.to_dict()

        # Deal with multi-dimensional thickness coeffs
        dat["mesh"]["Aflat"] = self.A.reshape(-1)
        dat["mesh"]["shape_A"] = self.A.shape
        dat["mesh"].pop("A")

        # Convert any numpy arrays to plain lists for serialisation
        for k in dat:
            for j in dat[k]:
                try:
                    dat[k][j] = dat[k][j].tolist()
                except:
                    pass

        # Write the file
        with open(fname, "w") as f:
            json.dump(dat, f, indent=4)

    @property
    def stg(self):

        # Mean-line design using the non-dimensionals
        args = self.nondimensional
        if self.Lam == -1 and not self.Al3 == -1:
            args['Al13'] = (args['Al1'],args['Al3'])
            args.pop('Lam')
            args.pop('Al3')
            args.pop('Al1')
            return design.nondim_stage_from_Al(**args)
        else:
            args.pop('Al3')
            return design.nondim_stage_from_Lam(**args)


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
            for k in self._var_names["bcond"] + ["guess_file", "ilos","nchange"]
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

        # Set stagger guess
        if self.psi < 0.:
            tanAl_rot = np.tan(np.radians(self.stg.Alrel[:2]))
            tanAl_sta = np.tan(np.radians(self.stg.Al[1:]))
            self.stag[0] = np.degrees(np.arctan(np.mean(tanAl_rot)))
            self.stag[1] = np.degrees(np.arctan(np.mean(tanAl_sta)))
        else:
            tanAl_rot = np.tan(np.radians(self.stg.Alrel[1:]))
            tanAl_sta = np.tan(np.radians(self.stg.Al[:2]))
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

    max_proc = int(
        subprocess.check_output("nvidia-smi --list-gpus | wc -l", shell=True)
    )

    if N > max_proc:
        params_split = np.array_split(params_all, np.ceil(N / max_proc))
    else:
        params_split = [
            params_all,
        ]

    meta = []
    for params_now in params_split:

        # Set up N working directories
        workdirs = [
            _write_input(write_func, params, base_dir) for params in params_now
        ]

        Nb = len(params_now)
        # Start the processes
        cmds = ["CUDA_VISIBLE_DEVICES=%d sh submit.sh" % n for n in range(Nb)]
        processes = [
            subprocess.Popen(cmd, cwd=wd, shell=True)
            for cmd, wd in zip(cmds, workdirs)
        ]

        # Wait for all processes
        for process in processes:
            if process:
                process.wait()

        # Load the processed data
        for workdir in workdirs:
            try:
                with open(os.path.join(workdir, "meta.json"), "r") as f:
                    meta.append(json.load(f))
            except IOError:
                # Sentiel value if the results file is not there (calc NaN'd)
                meta.append(None)

    # Return processed metadata
    return meta


def _param_from_x(xn, param_datum, row_index):
    """Perturb a datum turbine using normalised design vector x."""

    # Get reference design and steps in dimensional terms
    xref = _assemble_guess(1)[0]
    dx = _assemble_step(1)[0]

    # Apply datum stagger
    xref[2] = param_datum.stag[row_index] + 0.0
    # Convert from normalised xn to dimensional x
    x = xn * dx + xref

    param = param_datum.copy()

    recam = x[:2]

    offset = row_index * 2
    # Flip direction of recamber in rotor
    if row_index:
        recam = [-r for r in recam]
    param.recamber = list(param.recamber)
    param.recamber[0 + offset] += recam[0]
    param.recamber[1 + offset] += recam[1]

    Rle = x[3]
    thick = np.stack(x[4:6])
    if row_index:
        thick = np.flip(thick, axis=0)
    beta = x[6]

    param.stag = [stag + 0.0 for stag in param.stag]
    param.stag[row_index] = x[2] + 0.0

    Anew = geometry.A_from_Rle_thick_beta(Rle, thick, beta, param_datum.tte)

    param.A = param.A + 0.0
    param.A[row_index] = Anew

    if np.any(param.stag == 0.0):
        raise Exception("Stagger should not be zero")

    return param


def _assemble_bounds(nrow):
    """Return a (2, nx*nrow) matrix of bounds for some number of rows."""
    return np.tile(np.column_stack([X_BOUNDS[k] for k in X_KEYS]), nrow)


def _assemble_bounds_rel(nrow):
    """Return a (2, nx*nrow) matrix of bounds for some number of rows."""
    return tuple([X_BOUNDS_REL[k] for k in X_KEYS])


def _assemble_guess(nrow):
    return np.tile(np.atleast_2d([X_GUESS[k] for k in X_KEYS]), nrow)


def _assemble_step(nrow):
    return np.tile(np.atleast_2d([X_STEP[k] for k in X_KEYS]), nrow)


def _constrain_x_param(xn, write_func, param_datum, irow, verbose=False):

    # lower, upper = _assemble_bounds(1)

    # xref = _assemble_guess(1)[0]
    # dx = _assemble_step(1)[0]
    # xref[2] = param_datum.stag[irow] + 0.
    # x = xn*dx + xref

    # lower, upper = _assemble_bounds(1)
    # lower_ok = x >= lower
    # upper_ok = x <= upper
    # if verbose and not lower_ok.all():
    #     print('x lower bound violated:')
    #     print(x)
    #     print(lower)
    # if verbose and not upper_ok.all():
    #     print('x upper bound violated:')
    #     print(x)
    #     print(upper)
    # input_ok = lower_ok.all() and upper_ok.all()

    input_ok = True

    param = _param_from_x(xn, param_datum, row_index=irow)

    if input_ok:
        geom_ok = check_constraint(write_func, param)
        if verbose and not geom_ok:
            print("geometry generation failed")
        return geom_ok
    else:
        return False


def _metadata_to_y(meta):
    """Convert a metadata dictionary into objective vector y."""
    return np.array([float(meta[k]) for k in OBJECTIVE_KEYS])


def _param_to_y(param):
    """Convert parameter set into objective vector y."""
    y = np.array([getattr(param, k, np.nan) for k in OBJECTIVE_KEYS])
    return y


def _wrap_for_optimiser(
    write_func, param_datum, base_dir, irow, nan_constr=True
):
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
        if nan_constr:
            # NaN out results that deviate too much from target
            ind_good = np.logical_and(ind_low_error, ind_has_conv)
            y[~ind_good, 0] = np.nan
        return y

    return _objective, _constraint


def _wrap_for_grad(write_func, param_datum, base_dir, irow, eps, verbose=False):
    """Closure to wraps turbine creation and running with gradient."""

    # Store objective values so that we do not need to re-evaluate when
    # constrainign
    _cache = {}
    _cache_grad = {}

    def printv(s):
        """Print a string if verbosity is enabled."""
        if verbose:
            print(s)

    def _eval_func(x):
        try:
            if np.shape(x[0]) == ():
                # Only one x
                xt = tuple(x)
                if xt in _cache:
                    printv("Cache: %s" % str(xt))
                    return _cache[xt]
                printv("Eval: %s" % str(xt))
                param = _param_from_x(x, param_datum, irow)
                metadata = _run_parameters(write_func, param, base_dir)[0]
                y = _metadata_to_y(metadata)
                _cache[xt] = y
                return y
            else:
                # Run in parallel
                printv("Eval parallel: %s" % str(x))
                params = [_param_from_x(xi, param_datum, irow) for xi in x]
                metadata = _run_parameters(write_func, params, base_dir)
                y = np.stack([_metadata_to_y(mi) for mi in metadata])
                return y
        except:
            return np.nan * np.ones((len(OBJECTIVE_KEYS),))

    def _eval_grad(x0):

        xt0 = tuple(x0)
        if xt0 in _cache_grad:
            printv("Cache grad: %s" % str(xt0))
            return _cache_grad[xt0]

        # Perturb x about initial point
        dx = eps * np.ones((len(x0),))
        x = np.insert(x0 + np.diag(dx), 0, x0, axis=0)

        # Evaluate and get gradient
        # y = np.stack([_eval_func(xi) for xi in x])
        y = _eval_func(x)

        grad_y = (y[1:, :] - y[(0,), :]) / np.tile(dx, (y.shape[1], 1)).T

        # Save grad of all vars in cache for later
        _cache_grad[xt0] = grad_y

        return grad_y

    def _f(x):
        return _eval_func(x)[0]

    def _df_dx(x0):
        return _eval_grad(x0)[:, 0]

    def _constraint(x, ind, upper):
        # Determine errors with respect to target values
        y = _eval_func(x)
        y_target = _param_to_y(param_datum)
        err = (y / y_target - 1.0)[ind]
        err_max = param_datum.rtol
        if upper:
            return -err + err_max
        else:
            return err + err_max

    def _constraint_grad(x, ind, upper):
        # Determine errors with respect to target values
        dy_dx = _eval_grad(x)
        y_target = _param_to_y(param_datum)
        grad_norm = dy_dx[:, ind] / y_target[ind]
        if upper:
            return -grad_norm
        else:
            return grad_norm

    # get indexes into rows of y for the constrained variables
    ind_constr = [OBJECTIVE_KEYS.index(k) for k in CONSTR_KEYS]

    # assemble dict of constraints for SLSQP
    upper = [
        {
            "type": "ineq",
            "fun": _constraint,
            "jac": _constraint_grad,
            "args": (i, True),
        }
        for i in ind_constr
    ]
    lower = [
        {
            "type": "ineq",
            "fun": _constraint,
            "jac": _constraint_grad,
            "args": (i, False),
        }
        for i in ind_constr
    ]
    all_constr = upper + lower

    return _f, _df_dx, all_constr, _cache


def run_search(param, base_name, group_name):
    base_dir = os.path.join(TURBIGEN_ROOT, group_name, base_name)
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


def _solve_dev(Al_target, write_func, param, base_dir):
    """Recamber a parameter set to get zero deviation."""

    Co_ref = np.array(param.Co)

    def iter_row(x, ind):
        Co_now = np.tile(Co_ref, (4, 1))
        Co_now[:, ind] = x
        params_sweep = param.sweep("Co", Co_now.tolist())
        meta_sweep = _run_parameters(write_func, params_sweep, base_dir)
        err = np.ones((4,)) * np.nan
        for i, mi in enumerate(meta_sweep):
            if mi:
                if not ind:
                    err[i] = mi["Al"][1] - Al_target[0]
                else:
                    err[i] = mi["Alrel"][3] - Al_target[1]
        return err, meta_sweep

    Co_guess = np.array([0.0, 0.1, 0.2, 0.3]) + Co_ref[0]
    Co = Co_guess
    err, _ = iter_row(Co_guess, 0)

    tol = 0.1
    err_min = np.nanmin(np.abs(err))
    # err_min_old = np.inf

    has_dup = (np.unique(err, return_counts=True)[1] > 1).any()

    while not has_dup and (err_min > tol):
        ic = np.nanargmax(err > 0.0)
        Co_guess = np.linspace(Co[ic - 1], Co[ic], 6)[1:-1]
        err = np.append(err, iter_row(Co_guess, 0)[0])
        Co = np.append(Co, Co_guess)
        # err_min_old = err_min
        err_min = np.nanmin(np.abs(err))
        isrt = np.argsort(Co)
        Co, err = Co[isrt], err[isrt]
        has_dup = (np.unique(err, return_counts=True)[1] > 1).any()
        print(Co)
        print(err)

    ii_min = np.nanargmin(np.abs(err))
    Co_stator = Co[ii_min]
    print("Co_stator %.2f" % Co_stator)

    Co_guess = np.array([0.0, 0.1, 0.2, 3.0]) + Co_ref[1]
    Co = Co_guess
    err, meta = iter_row(Co_guess, 1)
    has_dup = (np.unique(err, return_counts=True)[1] > 1).any()
    err_min = np.nanmin(np.abs(err))

    while not has_dup and (err_min > tol):

        print(Co)
        print(err)
        ic = np.nanargmax(err > 0.0)
        print(ic)
        Co_guess = np.linspace(Co[ic - 1], Co[ic], 6)[1:-1]
        print(Co_guess)

        err_new, meta_new = iter_row(Co_guess, 1)

        err = np.append(err, err_new)
        meta = meta + meta_new
        Co = np.append(Co, Co_guess)

        err_min = np.nanmin(np.abs(err))

        isrt = np.argsort(Co)
        Co, err = Co[isrt], err[isrt]
        meta = [meta[i] for i in isrt]
        has_dup = np.sum(np.abs(err) == err_min) > 1
        print(Co)
        print(err)

    ii_min = np.nanargmin(np.abs(err))
    Co_rotor = Co[ii_min]
    print("Co_rotor %.2f" % Co_rotor)
    meta_out = meta[ii_min]

    param_out = param.copy()
    param_out.Co = [Co_stator, Co_rotor]

    return param_out, meta_out


def _initial_search(write_func):

    # Set up some file paths
    base_dir = os.getcwd()
    datum_file = os.path.join(base_dir, "datum_param.json")
    if not os.path.isfile(datum_file):
        raise Exception("No datum parameters found.")

    # Create the datum parameter set
    param = ParameterSet.from_json(datum_file)
    param.set_stag()

    # Give up if the initial design violates constraint
    if not check_constraint(write_func, param):
        raise Exception("Violating constraint at datum design.")

    # Run a robust damped solution and use as initial guess
    print("HIGH-DAMPING INITIAL GUESS")
    param_damp = param.copy()
    param_damp.dampin = 3.0
    param_damp.ilos = 1
    meta_damp = _run_parameters(write_func, param_damp, base_dir)[0]
    param.guess_file = os.path.join(
        base_dir, meta_damp["runid"], "output_avg.hdf5"
    )

    # Die if the initial guess diverges
    if meta_damp is None:
        print("** guess diverged, quitting.")
        sys.exit()

    param.ilos = 1

    # Record target circulation and flow angles
    Co_target = np.array(param.Co)
    stg = design.nondim_stage_from_Lam(**param.nondimensional)
    Al_target = np.array((stg.Al[1], stg.Alrel[2]))

    # Tune deviation, circulation, effy using a shotgun method

    print("CORRECTING ANGLES AND EFFY")
    print("  Target Al = %s" % str(Al_target))
    print("  Target Co = %s" % str(Co_target))
    # A single initial guess to see how much deviation we have
    meta_dev = _run_parameters(write_func, param, base_dir)[0]
    for i in range(25):

        Al_now = np.array((meta_dev["Alrel"][1], meta_dev["Alrel"][3]))
        Co_now = np.array(meta_dev["Co"])
        print("  Al = %s" % str(Al_now))
        print("  Co = %s" % str(Co_now))

        Co_on_target = (np.abs(Co_now / Co_target - 1.0) < param.rtol).all()
        Al_on_target = (np.abs(Al_now - Al_target) < 0.5).all()
        if Co_on_target and Al_on_target:
            print("  Tolerance reached, breaking.")
            break

        # Die if the initial guess diverges
        if meta_dev is None:
            print("** diverged, quitting.")
            sys.exit()

        # Update polytropic effy
        param.eta = meta_dev["eta"]

        # Deviations for the flow angles
        dev_vane, dev_blade = Al_target - Al_now

        # Relative changes for circulation coeff
        dCo = Co_target / Co_now - 1.0

        # Use most recent solution as initial guess
        param.guess_file = os.path.join(
            base_dir, meta_dev["runid"], "output_avg.hdf5"
        )

        # Make an array of parametrs with different amounts of change
        params_fac = []
        for fac in [0.25, 0.5, 0.75, 1.0]:
            param_new = param.copy()
            # Update angles
            param_new.recamber = list(param_new.recamber)
            param_new.recamber[1] += dev_vane * fac
            param_new.recamber[3] -= dev_blade * fac
            # Update circulation coeff
            Co_old = np.array(param_new.Co)
            param_new.Co = list(Co_old * (dCo * fac + 1.0))
            params_fac.append(param_new)

        # Run all
        meta_fac = _run_parameters(write_func, params_fac, base_dir)

        if None in meta_fac:
            print("** diverged, quitting.")
            sys.exit()

        # Check circulation errors
        err_Co = np.array(
            [
                np.max(np.abs(np.array(m["Co"]) / Co_target - 1.0))
                for m in meta_fac
            ]
        )
        # err_Al = np.array(
        # [
        # np.max(np.abs(Al_target - np.array((m["Alrel"][1], m["Alrel"][3]))))
        # for m in meta_fac
        # ]
        # )

        # Continue loop with closest circulation coeff
        iCo = err_Co.argmin()
        param = params_fac[iCo]
        meta_dev = meta_fac[iCo]

    # Write out datum parameters for later inspection
    param.write_json(datum_file)

    return param


def _run_search(write_func):
    """Tabu search both blade rows in turn."""

    # Set up datum parameters with corrections for deviation, circulation
    _initial_search(write_func)

    # Set up some file paths
    base_dir = os.getcwd()
    datum_file = os.path.join(base_dir, "datum_param.json")
    if not os.path.isfile(datum_file):
        raise Exception("No datum parameters found.")

    param = ParameterSet.from_json(datum_file)

    # Optimise rotor first, then stator
    for irow in range(2):
        print("*******************")
        print("**OPTIMISE ROW %d**" % irow)
        print("*******************")
        param = _search_row(base_dir, param, write_func, irow)

    # Write out optimum params
    opt_file = os.path.join(base_dir, "opt_param.json")
    param.write_json(opt_file)

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
