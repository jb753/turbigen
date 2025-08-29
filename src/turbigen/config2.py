"""Initial thoughts on an improved config class."""

import dataclasses
import traceback
from copy import deepcopy
import gzip
import numpy as np
import pickle
import sys
import importlib
from pathlib import Path
import turbigen.fluid
import turbigen.flowfield
import turbigen.meanline_design
import turbigen.solvers.base
import turbigen.base
import turbigen.iterators
import turbigen.average
import turbigen.op_point
import turbigen.grid
import turbigen.post
import turbigen.geometry
import turbigen.yaml
import turbigen.annulus
import turbigen.inlet
import turbigen.mesh
import turbigen.blade
import turbigen.dspace
import turbigen.nblade
import turbigen.job
from turbigen import util
from typing import List
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

logger = util.make_logger()


@dataclasses.dataclass
class TurbigenConfig:
    """Top level configuration class for turbigen.

    A run is uniquely defined by an instance of this class.

    """

    workdir: Path
    """Directory in which to store run data."""

    inlet: turbigen.inlet.InletConfig
    """Settings for the inlet boundary condition."""

    mean_line: turbigen.meanline_design.MeanLineDesigner
    """Settings for the mean-line designer."""

    annulus: turbigen.annulus.AnnulusDesigner = None
    """Settings for the annulus designer."""

    blades: List[List[turbigen.blade.BladeDesigner]] = dataclasses.field(
        default_factory=list
    )
    """Settings for the blade designers."""

    nblade: List[turbigen.nblade.BladeNumberConfig] = dataclasses.field(
        default_factory=list
    )
    """Settings for blade number selection."""

    mesh: turbigen.mesh.Mesher = None
    """Settings for mesh generation."""

    solver: turbigen.solvers.base.BaseSolver = None
    """Settings for flow solution."""

    plugdir: Path = None
    """Directory to search for custom plugins."""

    operating_point: turbigen.op_point.OperatingPoint = None
    """Settings for off-design operation and throttling."""

    iterate: List[turbigen.iterators.IteratorConfig] = dataclasses.field(
        default_factory=list
    )

    max_iter: int = 20
    """Maximum number of iterations to perform."""

    fac_nstep_initial: float = 1.0
    """Multiplier on nstep for the first run of iterating case."""

    """Settings for blade number selection."""
    grid: turbigen.grid.Grid = None
    guess: turbigen.grid.Grid = None

    cut_offset: float = 0.02
    """Spacing of CFD solution cuts away from blade edges, as fraction of chord."""

    mean_line_actual: dict = dataclasses.field(default_factory=dict)

    post_process: list = dataclasses.field(default_factory=list)

    job: turbigen.job.BaseJob = None
    """Settings for queue job submission."""

    converged: bool = False
    """Flag to indicate iterative convergence."""

    design_space: turbigen.dspace.DesignSpace = None
    """Settings for design space mapping."""

    basename: str = "config.yaml"

    _fast_init: bool = False
    """Flag to not read large object from file on init."""

    mixed_out_flowfield: dict = None

    post_3d: dict = dataclasses.field(default_factory=dict)
    """Results post-processed from the full 3D flow field."""

    def copy(self):
        """Return a copy of the configuration."""
        return deepcopy(self)

    @property
    def fname(self):
        return self.workdir / self.basename

    Re_surf: float = None
    """Set viscosity using a Reynolds number."""

    save_iteration_grids: bool = False
    """Save grid and guess at each iteration to workdir."""

    @property
    def nrow(self):
        return len(self.blades)

    def save(self, fname=None, overwrite_pkl=True, use_gzip=True, write_grids=True):
        """Save the configuration to a YAML file inside workdir.

        The working directory will be created if it does not exist.
        """

        if fname is None:
            fname = self.fname

        if not self.workdir.exists():
            self.workdir.mkdir(parents=True)

        # Check that the blades are not recambered
        for row in self.blades:
            for blade in row:
                if blade.is_recambered:
                    raise Exception(
                        "Cannot write configuration with recambered blades.\n"
                        "Use `undo_recamber()` to revert the camber parameters to degreeof of recamber."
                    )

        data = self.to_dict()

        # Convert grid objects to filenames
        for k in ["grid", "guess"]:
            val = getattr(self, k)
            # If not there remove the key
            if val is None or not write_grids:
                del data[k]
            else:
                # Otherwise, save the grid to a separate pickle
                # and replace the grid with the filename in yaml
                fname_pkl = self.workdir / f"{k}.pkl.gz"
                data[k] = str(fname_pkl)

                if fname_pkl.exists() and not overwrite_pkl:
                    logger.debug(f"Not overwriting existing {fname_pkl}")
                    continue
                else:
                    logger.debug(f"Saving {k} to {fname_pkl}")
                    util.safe_pickle_dump(val, fname_pkl, zip=use_gzip)

        if hasattr(self.mean_line, "actual"):
            data["mixed_out_flowfield"] = self.mean_line.actual.to_dump()
        if not data["mixed_out_flowfield"]:
            del data["mixed_out_flowfield"]

        # Convert convergence history to a filename
        if self.solver and (conv := self.solver.convergence):
            fname_conv = self.workdir / "convergence.npz"
            conv.save(fname_conv)
            data["solver"]["convergence"] = str(fname_conv)

        conf_fname = self.workdir / fname
        logger.debug(f"Saving configuration to {conf_fname}")
        try:
            turbigen.yaml.write_yaml(data, conf_fname)
        except Exception as e:
            logger.error(f"Failed to save configuration to {conf_fname}")
            logger.error(data)
            logger.error(e)
            sys.exit(1)

        return conf_fname

    def to_dict(self):
        """Convert the config to a dictionary."""

        # Built-in dataclasses method gets us most of the way there
        data = dataclasses.asdict(self)

        # Put work and plug dir into a string
        data["workdir"] = str(data["workdir"])
        if data["plugdir"]:
            data["plugdir"] = str(data["plugdir"])

        # Convert the meanline designer to a dictionary
        data["mean_line"] = self.mean_line.to_dict()

        # Convert the annulus designer to a dictionary
        if self.annulus:
            data["annulus"] = self.annulus.to_dict()

        # Convert the blade designer to a dictionary
        data["blades"] = []
        for row in self.blades:
            if len(row) == 1:
                data["blades"].append(row[0].to_dict())
            else:
                data["blades"].append([])
                for blade in row:
                    data["blades"][-1].append(blade.to_dict())

        # Restore the mesh type
        if self.mesh:
            data["mesh"]["type"] = util.camel_to_snake(self.mesh.__class__.__name__)

        # Restore the solver type
        if self.solver:
            data["solver"] = self.solver.to_dict()
            data["solver"]["type"] = util.camel_to_snake(self.solver.__class__.__name__)

        # If no acutal meanline, remove it
        if not self.mean_line_actual:
            del data["mean_line_actual"]

        # If no job, remove it
        if not self.job:
            del data["job"]
        else:
            data["job"] = self.job.to_dict()
            # Add the job type to the dictionary
            data["job"]["type"] = util.camel_to_snake(self.job.__class__.__name__)

        # If no iterators, remove it
        if not self.iterate:
            del data["iterate"]
        # Otherwise, convert the iterators to a dictionary
        else:
            iters = {}
            for iiter, iter in enumerate(self.iterate):
                k = util.camel_to_snake(iter.__class__.__name__)
                iters[k] = data["iterate"][iiter]
            data["iterate"] = iters

        # Add type info to post processors
        if self.post_process:
            for i, post in enumerate(self.post_process):
                data["post_process"][i]["type"] = util.camel_to_snake(
                    post.__class__.__name__
                )

        if self.design_space:
            # Convert the design space to a dictionary
            data["design_space"] = self.design_space.to_dict()

        # Remove keys starting with '_'
        # These are not part of the configuration
        for k in list(data.keys()):
            if k.startswith("_"):
                data.pop(k)

        return data

    def find_plugins(self):
        """Find and load plugins from the plugdir."""

        logger.iter(f"Importing plugins from {self.plugdir}")
        # Find all python files recursively in the plugdir
        py_files = list(self.plugdir.rglob("*.py"))
        for py_file in py_files:
            # Exclude hidden files and directories
            if any(part.startswith(".") for part in py_file.parts):
                continue
            try:
                # Get the module name
                module_name = py_file.stem
                # Import the module from file
                spec = importlib.util.spec_from_file_location(
                    f"turbigen.plugin.{module_name}", py_file
                )
                module = importlib.util.module_from_spec(spec)
                sys.modules[f"turbigen.plugin.{module_name}"] = module
                spec.loader.exec_module(module)
                logger.iter(f"Loaded plugin: {py_file}")
            except Exception as e:
                logger.iter(f"Failed to load {py_file}, error:")
                logger.iter(e)
                sys.exit(1)

    def __post_init__(self):
        """Convert input basic types to our desired types."""

        # Convert workdir str to Path object
        self.workdir = Path(self.workdir).absolute()

        # Convert plugdir str to Path object
        # And look for plugins
        if self.plugdir:
            self.plugdir = Path(self.plugdir).absolute()
            self.find_plugins()

        # If grid or guess is a filename, load and unpickle it
        for k in ["grid", "guess"]:
            val = getattr(self, k)
            if isinstance(val, str) and not self._fast_init:
                try:
                    with gzip.open(Path(val), "rb") as f:
                        setattr(self, k, pickle.load(f))
                except gzip.BadGzipFile:
                    # If gzip fails, try loading without it
                    with open(Path(val), "rb") as f:
                        setattr(self, k, pickle.load(f))

        # Convert inlet dict to InletConfig object
        self.inlet = util.init_subclass_by_signature(
            turbigen.inlet.InletConfig, self.inlet
        )

        # Set up the meanline designer
        MeanLineDesigner = util.get_subclass_by_name(
            turbigen.meanline_design.MeanLineDesigner, self.mean_line.pop("type")
        )
        self.mean_line = MeanLineDesigner(self.mean_line)

        if isinstance(self.mixed_out_flowfield, dict):
            self.mean_line.actual = turbigen.meanline_data.meanline_from_dump(
                self.mixed_out_flowfield, self.inlet.get_inlet()
            )

        # Set up the annulus designer
        if self.annulus:
            AnnulusDesigner = util.get_subclass_by_name(
                turbigen.annulus.AnnulusDesigner, self.annulus.pop("type", "smooth")
            )
            self.annulus = AnnulusDesigner(self.annulus)

        if self.operating_point:
            self.operating_point = turbigen.op_point.OperatingPoint(
                **self.operating_point
            )

        # Set up the blade designers
        blades = []
        for row in self.blades:
            # Check for no splitters
            if not isinstance(row, list):
                row = [
                    row,
                ]
            blades.append([])
            for blade in row:
                blades[-1].append(turbigen.blade.BladeDesigner(**blade))
        self.blades = blades

        # Convert nblade dict to NbladeConfig objects
        self.nblade = [
            util.init_subclass_by_signature(turbigen.nblade.BladeNumberConfig, d)
            for d in self.nblade
        ]

        # Set up the mesher
        if self.mesh:
            Mesher = util.get_subclass_by_name(
                turbigen.mesh.Mesher, self.mesh.pop("type", "h")
            )
            self.mesh = Mesher(**self.mesh)

        # Lazy import the solver
        if self.solver:
            solver_name = self.solver.pop("type")
            importlib.import_module(f".{solver_name}", package="turbigen.solvers")
            Solver = util.get_subclass_by_name(
                turbigen.solvers.base.BaseSolver, solver_name
            )
            self.solver = Solver(**self.solver)
            # If solver has convergence history, load it
            if isinstance(self.solver.convergence, str) and not self._fast_init:
                self.solver.convergence = turbigen.solvers.base.ConvergenceHistory.load(
                    self.solver.convergence, self.inlet.get_inlet()
                )

        # Convert iterator dicts to Config objects
        if self.iterate:
            iters = []
            iter_cls = []
            for k, v in self.iterate.items():
                # Find a subclass for this iterator
                cls = util.get_subclass_by_name(turbigen.iterators.IteratorConfig, k)
                iter_cls.append(cls)
                if v:
                    # Pass the dictionary to the subclass
                    iters.append(cls(**v))
                # Null content implies all defaults
                else:
                    iters.append(cls())
            self.iterate = iters

            # Ensure that incidence is always the first iterator=
            # This is because any iterators that change geometry
            # Will make the CFD grid not match the blade geometry
            if turbigen.iterators.Incidence in iter_cls:
                index = iter_cls.index(turbigen.iterators.Incidence)
                self.iterate.insert(0, self.iterate.pop(index))
                assert self.iterate[0].__class__ == turbigen.iterators.Incidence

        # Check the iterators
        for iterator in self.iterate:
            iterator.check(self)

        # Setup the post processors
        if self.post_process:
            posts = []
            # post_process is a list of dicts
            # So that we can have, e.g. multiple 'contour' processors
            for ip, p in enumerate(self.post_process):
                # Get the type of this processor
                if not (type := p.pop("type")):
                    raise Exception(
                        f"Missing type key in post_process list at index {ip}"
                    )
                # Find a subclass for this processor
                cls = util.get_subclass_by_name(turbigen.post.BasePost, type)
                if p:
                    # Pass the dictionary to the subclass
                    posts.append(cls(**p))
                # Null content implies all defaults
                else:
                    posts.append(cls())
            self.post_process = posts
        else:
            self.post_process = []

        # Configure job submission if present
        if j := self.job:
            if not (type := j.pop("type")):
                raise Exception("Missing type key in job settings")
            cls = util.get_subclass_by_name(turbigen.job.BaseJob, type)
            self.job = cls(**j)

        # Add some default post processors
        defaults = [
            turbigen.post.SurfaceDistribution(),
            turbigen.post.Convergence(),
            turbigen.post.Annulus(),
            turbigen.post.Metadata(),
        ]
        for d in defaults:
            found = False
            for p in self.post_process:
                if isinstance(p, d.__class__):
                    found = True
            # If not already in the list, insert it
            # at the start
            if not found:
                self.post_process.insert(0, d)

        # Init the design space
        if self.design_space:
            if isinstance(self.design_space, dict):
                self.design_space = turbigen.dspace.DesignSpace(**self.design_space)
                if not self.design_space.basedir:
                    self.design_space.basedir = Path(self.workdir)
                else:
                    self.design_space.basedir = Path(self.design_space.basedir)
            self.design_space.setup()

    def get_mean_line_nominal(self):
        """Calculate the nominal mean-line flow field."""

        So1 = self.inlet.get_inlet()
        logger.info(f"Inlet: {So1}")

        # Mean-line design
        self.mean_line.setup_mean_line(So1)
        logger.info(self.mean_line.nominal)

        # Check mean-line design for problems
        logger.debug("Checking mean-line conservation...")
        if not self.mean_line.nominal.check():
            self.mean_line.nominal.show_debug()
            raise Exception(
                "Mean-line conservation checks failed, have printed debugging information"
            ) from None
        logger.debug("Checking mean-line inversion...")
        self.mean_line.check_backward(self.mean_line.nominal)
        self.mean_line.nominal.warn()

    def get_geometry(self):
        """Get the annulus and blade geometry."""

        # Annulus design
        logger.info("Designing annulus...")

        if not self.annulus:
            logger.error("No annulus defined, quitting.")
            sys.exit(0)

        self.annulus.setup_annulus(self.mean_line.nominal)
        logger.info(f"{self.annulus}")

        # Blade design
        logger.info("Designing blades...")

        if not self.blades:
            logger.error("No blades defined, quitting.")
            sys.exit(0)

        for irow, row in enumerate(self.blades):
            # Set meridional locations
            for blade in row:
                blade.set_streamsurface(self.annulus.xr_row(irow))

    def get_nblade(self):
        Nb = np.full((len(self.blades),), 0, dtype=int)
        for irow, row in enumerate(self.blades):
            # Set number of blades using main blade
            Nb[irow] = np.round(
                self.nblade[irow].get_blade_number(
                    self.mean_line.nominal.get_row(irow), row[0]
                )
            )
        return Nb

    def check_pitch_chord(self, s_cm_lim=(0.2, 4.0)):
        # Warn if blade spacings are too narrow or wide
        rref = 0.5 * (
            self.mean_line.nominal.rrms[::2] + self.mean_line.nominal.rrms[1::2]
        )
        s = 2.0 * np.pi * rref / self.get_nblade()
        s_cm = s / self.annulus.chords(0.5)[1:-1:2]
        if np.any(s_cm < s_cm_lim[0]):
            logger.warning(
                "WARNING: narrow blade spacings may cause problems with meshing"
            )
        if np.any(s_cm > s_cm_lim[1]):
            logger.warning(
                "WARNING: large blade spacings may cause problems with meshing"
            )

    def get_gaps(self):
        """Return non-dimensional tip gaps as fraction of span."""

        # Relative gaps from blade definition
        gap_span = np.full((self.nrow,), 0.0)
        chord = self.annulus.chords(0.5)[1::2]
        span = self.mean_line.nominal.span
        span = 0.5 * (span[::2] + span[1::2])  # Average span for each row
        for irow, row in enumerate(self.blades):
            # Choose reference length
            if row[0].tip_ref == "span":
                gap_span[irow] = row[0].tip
            elif row[0].tip_ref == "chord":
                gap_span[irow] = row[0].tip * chord[irow] / span[irow]
            elif row[0].tip_ref == "absolute":
                gap_span[irow] = row[0].tip / span[irow]
            else:
                logger.error(
                    f"Unknown tip reference length {row[0].tip_ref}, quitting."
                )
                sys.exit(1)

        return gap_span

    def apply_recamber(self):
        # Apply recamber to the blades
        for irow, row in enumerate(self.blades):
            for blade in row:
                blade.apply_recamber(self.mean_line.nominal.get_row(irow))

    def undo_recamber(self):
        # Undo recamber to the blades
        for irow, row in enumerate(self.blades):
            for blade in row:
                blade.undo_recamber(self.mean_line.nominal.get_row(irow))

    def get_ell(self):
        """Find suction surface lengths for each row."""
        return np.array(
            [self.blades[irow][0].surface_length(0.5) for irow in range(self.nrow)]
        )

    def setup_mesh(self):
        if not self.mesh:
            logger.error("No mesh configured, quitting.")
            sys.exit(0)

        # Find wall distances for each row
        dsurf = np.array(
            [
                self.mesh.get_dwall(
                    self.mean_line.nominal.get_row(irow),
                    self.blades[irow][0].surface_length(0.5),
                )
                for irow in range(len(self.blades))
            ]
        )

        # Hub and casing wall distances are row means
        dhub = dcas = np.mean(dsurf)

        mesh_dir = self.workdir / self.mesh.meshdir
        Omega = self.mean_line.nominal.Omega[::2]
        self.grid = self.mesh.make_grid(
            mesh_dir, self.get_machine(), dhub, dcas, dsurf, Omega
        )

        logger.info(f"ncell/1e6={self.grid.ncell / 1e6:.1f}")

        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots()
        # b = self.grid[0]
        # C = b[:, b.nj // 2, :]
        # ax.plot(C.x, C.rt, "k-")
        # ax.plot(C.x.T, C.rt.T, "k-")
        # ax.axis("equal")
        # plt.show()
        #
        self.grid.check_coordinates()
        self.grid.calculate_wall_distance()

        # Reset camber
        for irow, row in enumerate(self.blades):
            # Apply recamber, set meridional locations for
            # main and splitters
            for blade in row:
                blade.set_streamsurface(self.annulus.xr_row(irow))

        # Choose whether the blocks are real or perfect
        So1 = self.inlet.get_inlet()
        if isinstance(So1, turbigen.fluid.PerfectState):
            self.grid = turbigen.grid.Grid([b.to_perfect() for b in self.grid])
        elif isinstance(So1, turbigen.fluid.RealState):
            self.grid = turbigen.grid.Grid([b.to_real() for b in self.grid])
        else:
            raise Exception("Unrecognised inlet state type")

    def get_machine(self):
        return turbigen.geometry.Machine(
            self.annulus, self.blades, self.get_nblade(), self.get_gaps(), None
        )

    def apply_bconds(self):
        # Get nominal exit pressure, mdot, shaft speed
        Omega = self.mean_line.nominal.Omega[::2].copy()
        Pout = self.mean_line.nominal.P[-1]
        mdot = self.mean_line.nominal.mdot[-1]

        # Alter the operating point if needed
        if self.operating_point:
            logger.info("Setting operating point...")
            if Omega_adjust := self.operating_point.Omega_adjust:
                Omega *= 1.0 + Omega_adjust
                logger.info(f"Omega/Omega_design={1.0 + Omega_adjust:.3g}")
            if PR_ts_adjust := self.operating_point.PR_ts_adjust:
                Pout /= 1.0 + PR_ts_adjust
                logger.info(f"PR/PR_design={1.0 + PR_ts_adjust:.3g}")
            if mdot_adjust := self.operating_point.mdot_adjust:
                mdot *= 1.0 + mdot_adjust
                logger.info(f"mdot/mdot_design={1.0 + mdot_adjust:.3g}")
            if self.operating_point.throttle:
                pid = self.operating_point.pid
                # Constants are scaled by meanline Delta P / mdot
                logger.info(f"Exit PID constants={pid}")
                scale = (
                    np.ptp(self.mean_line.nominal.P) / self.mean_line.nominal.mdot[-1]
                )
                Kpid = np.array(pid) * scale
                self.grid.apply_throttle(mdot, Kpid)

        # Set the rotation types
        gaps = self.get_gaps()
        rot_types = []
        for irow in range(self.nrow):
            if Omega[irow]:
                if gaps[irow]:
                    rot_types.append("tip_gap")
                else:
                    rot_types.append("shroud")
            else:
                rot_types.append("stationary")
        self.grid.apply_rotation(rot_types, Omega)

        # Inlet boundary condition
        # Set inlet pitch angle using orientation of
        # the inlet patch grid (assuming on a constant i face)
        # This allow the annulus lines to differ from mean-line pitch angle
        Ain = self.grid.inlet_patches[0].get_cut().dAi.sum(axis=(-1, -2, -3))
        Beta1 = np.degrees(np.arctan2(Ain[1], Ain[0]))
        Alpha1 = self.mean_line.nominal.Alpha[0]
        self.grid.apply_inlet(self.inlet.get_inlet(), Alpha1, Beta1)

        # Apply profile if available
        if self.inlet.profiles is not None:
            logger.info("Applying inlet profile...")
            self.grid.inlet_patches[0].set_profile(
                self.inlet.spf,
                self.inlet.profiles,
            )

        # Outlet boundary condition
        self.grid.apply_outlet(Pout)

    def apply_guess(self):
        # Apply 3D guess if available
        if self.guess:
            logger.info("Applying 3D guess...")
            self.grid.apply_guess_3d(self.guess)
        else:
            # Apply crude guess from mean_line
            logger.info("Applying 2D guess...")
            self.grid.apply_guess_meridional(
                self.mean_line.nominal.interpolate_guess(self.annulus)
            )

        # Update the outlet static pressure based on the guess
        # This helps running multiple iterations of a throttled case
        self.grid.update_outlet()

    def run_solver(self):
        if not self.solver:
            logger.error("No solver configured, quitting.")
            sys.exit(0)

        run_args = self.grid, self.get_machine, self.workdir / "solve"

        if self.solver.soft_start:
            logger.info("Soft start...")
            self.solver.robust().run(*run_args)
        self.solver.run(*run_args)

    def get_mean_line_actual(self):
        """Extract the actual mean-line flow field by mixing out CFD result."""

        # Find meridional coordinates of the cut planes
        xr_cut = self.annulus.get_offset_planes(self.cut_offset)

        # Take the cuts, form a list of [(Cmix, Amix, Dsmix)]
        cuts = [
            turbigen.average.mix_out_unstructured(
                self.grid.unstructured_cut_marching(xri)
            )
            for xri in xr_cut
        ]

        # Unpack the list
        Cmix, Amix, Dsmix = zip(*cuts)

        # Stack the cuts to form a mean-line flow field
        Call = turbigen.base.stack(Cmix)

        # Copy Omega and Nb from nominal
        Call.Omega = np.concatenate(
            [g[0].Omega.flat[0] * np.ones((2,)) for g in self.grid.row_blocks]
        )
        Nb = np.concatenate(
            [g[0].Nb * np.ones((2,)) for g in self.grid.row_blocks]
        ).astype(int)

        # Assemble the meanline flowfield
        self.mean_line.actual = turbigen.meanline_data.make_mean_line_from_flowfield(
            Amix, Call, Dsmix
        )
        self.mean_line.actual.Nb = self.mean_line.nominal.Nb = Nb

        # Back-calculate the design variables
        self.mean_line_actual = self.mean_line.backward(self.mean_line.actual)

    def calculate_design_var_errors(self):
        """Calculate differences between nominal and actual design variables."""

        # Absolute error (dict comprehension)
        err = {
            k: v - self.mean_line_actual[k]
            for k, v in self.mean_line.design_vars.items()
        }

        # Relative error (dict comprehension, checking for zero nominal values)
        with np.errstate(divide="ignore", invalid="ignore"):
            rel_err = {
                k: (v - self.mean_line_actual[k]) / v * 100.0
                for k, v in self.mean_line.design_vars.items()
            }

        # Make very small values zero
        eps = 1e-6
        for k, v in err.items():
            if np.isscalar(v):
                if np.abs(v) < eps:
                    err[k] = 0.0
            else:
                err[k] = np.where(np.abs(v) < eps, 0.0, v)

        for k, v in rel_err.items():
            if np.isscalar(v):
                if np.abs(v) < eps:
                    rel_err[k] = 0.0
            else:
                rel_err[k] = np.where(np.abs(v) < eps, 0.0, v)

        return err, rel_err

    def format_design_vars_table(self):
        """Format nominal and actual design variables for printing."""

        # Initialise with header row
        table = [["Variable", "Nominal", "Actual", "Err_abs", "Err_rel/%"]]

        # Add rows for each design variable
        err, rel_err = self.calculate_design_var_errors()

        for k, v in self.mean_line.design_vars.items():
            # Make very small values zero
            if np.isscalar(v):
                table.append(
                    [
                        k,
                        f"{v:.3g}",
                        f"{self.mean_line_actual[k]:.3g}",
                        f"{err[k]:.2g}",
                        f"{rel_err[k]:.1f}",
                    ]
                )
            else:
                # Each element of v is a row in the table
                for i, vi in enumerate(v):
                    table.append(
                        [
                            f"{k}[{i}]",
                            f"{vi:.3g}",
                            f"{self.mean_line_actual[k][i]:.3g}",
                            f"{err[k][i]:.2g}",
                            f"{rel_err[k][i]:.2g}",
                        ]
                    )

        # Additional vars not in nominal
        for k, v in self.mean_line_actual.items():
            if k not in self.mean_line.design_vars:
                if np.isscalar(v):
                    table.append(
                        [
                            k,
                            "",
                            f"{self.mean_line_actual[k]:.3g}",
                            "",
                            "",
                        ]
                    )
                else:
                    # Each element of v is a row in the table
                    for i, vi in enumerate(v):
                        table.append(
                            [
                                f"{k}[{i}]",
                                "",
                                f"{self.mean_line_actual[k][i]:.3g}",
                                "",
                                "",
                            ]
                        )

        # Find column widths
        ncol = len(table[0])
        widths = np.array([max(len(str(row[i])) for row in table) for i in range(ncol)])

        # Add padding
        table_pad = [
            "  ".join(f"{row[i]:>{widths[i]}}" for i in range(ncol)) for row in table
        ]

        # Add continuous separator after header
        table_pad.insert(1, "-" * (sum(widths + 2) - 2))

        # Add efficiency row
        table_pad.append(
            f"Efficiency/%: eta_tt={self.mean_line.actual.eta_tt * 100.0:.1f}, "
            f"eta_ts={self.mean_line.actual.eta_ts * 100:.1f}"
        )

        # Join the lines
        table_pad = "\n".join(table_pad)

        return table_pad

    def set_mu_from_Re_surf(self):
        ell = self.get_ell()
        ml = self.mean_line.nominal
        mu = (ml.rho_ref * ml.V_ref * ell)[0] / self.Re_surf
        try:
            self.inlet.mu = mu
            self.mean_line.nominal.mu = mu
        except TypeError:
            raise Exception(
                "Cannot set Reynolds number by changing viscosity of a real gas."
            )

    def design_and_run(self, skip, skip_post=False):
        """Run a configuration file through the CFD solver.

        This will do the following:
            1. Get inlet state;
            2. Design the nominal meanline;
            3. Design the annulus;
            4. Design the blades;
            5. Generate the mesh;
            6. Run the CFD solver;
            7. Extract the actual meanline from CFD;
            8. Calculate the actual design variables.

        """

        self.get_mean_line_nominal()

        # Print real gas table limits
        if isinstance(self.mean_line.nominal, turbigen.fluid.RealState):
            self.show_table_limits()

        self.get_geometry()
        self.apply_recamber()

        self.check_pitch_chord()
        logger.info(f"Nblade: {self.get_nblade()}")
        logger.info(f"Tip gaps/span: {self.get_gaps()}")

        # Handle restarts
        if self.grid:
            # If we already have a grid, use it as the guess
            self.guess = self.grid
            # Change CFD settings to resume the simulation
            self.solver = self.solver.restart()

        # Set viscosity from Reynolds number if given
        if self.Re_surf:
            self.set_mu_from_Re_surf()
        Re_surf = self.get_ell() / self.mean_line.nominal.L_visc
        logger.info(f"Re_surf={util.format_array(Re_surf)}")

        # We are now ready to generate mesh and run CFD
        # There are three cases to consider
        # (1) Skipping from guess: just use existing mesh and solution
        # (2) Skipping from cold: mesh but do not run the CFD solver
        # (3) Normal operation: mesh and run the CFD solver

        # Generate mesh in cases (2) and (3)
        if not (skip and self.grid):
            logger.info("Generating mesh...")
            self.setup_mesh()  # Overwrite self.grid with a new mesh
            self.apply_guess()
            self.apply_bconds()
        else:
            logger.info("Skipping and already have a guess, not generating mesh...")

        # In case (3), run the CFD solver
        if not skip:
            self.run_solver()

        # The flow field is ready in grid, post-process it
        self.get_mean_line_actual()
        self.undo_recamber()
        logger.info("Post-processing...")
        if not skip_post:
            self.post_process_all()
        logger.info("Done post-processing.")

    def interpolate_all_iterators(self):
        """Use fitted design space to set values for all iterated variables."""

        for iterator in self.iterate:
            iterator.interpolate(self)

    def step_iterate(self):
        """Apply all iterators to the configuration."""

        log_data = {}
        converged = {}

        for iterator in self.iterate:
            conv_now, log_data_now = iterator.update(self)

            # Update the overall convergence flag and log data
            name = util.camel_to_snake(iterator.__class__.__name__)
            converged[name] = conv_now
            log_data.update(log_data_now)

        return converged, log_data

    def show_table_limits(self):
        # """Return limiting property values and deltas for gas table generation."""

        ml = self.mean_line.nominal

        # Min/max entropy
        smin = ml.s.min()
        smax = ml.s.max()

        # Minimum pressure
        Pmin = ml.P.min()

        # Maximum temperature
        Tmax = ml.To.max()

        logger.info(
            f"Real gas table limits: "
            f"smin={smin:.3g} J/kgK, smax={smax:.3g} J/kgK , "
            f"Pmin={Pmin:.3g} Pa, Tmax={Tmax:.3g} K"
        )

    def post_process_all(self):
        # Initialise the pdf
        with PdfPages(self.workdir / "post.pdf") as pdf:
            for poster in self.post_process:
                logger.debug(f"Running post function {poster}")
                try:
                    poster.post(self, pdf)
                except Exception:
                    logger.error(f"Failed to run post function {poster}")
                    traceback.print_exc()
        # Ensure all figures are closed
        plt.close("all")
