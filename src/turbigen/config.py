"""Initial thoughts on an improved config class."""

import logging
import resource


import dataclasses
import traceback
from copy import deepcopy
import numpy as np
import sys
import importlib
from pathlib import Path
import turbigen.fluid
import turbigen.meanline_new
import turbigen.solvers.base
import turbigen.base
import turbigen.iterators
import turbigen.average
import turbigen.op_point

import turbigen.post
import turbigen.geometry
import turbigen.annulus
import turbigen.inlet
import turbigen.mesh
import turbigen.hmesh

import turbigen.ohmesh
import turbigen.blade
import turbigen.dspace
import turbigen.nblade
import turbigen.job
from turbigen import util
from typing import List
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

import ember.grid
import ember.cut
import ember.average
import ember.util

logger = logging.getLogger("turbigen")


def _log_ram(label):
    rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    logger.debug(f"RAM [{label}]: {rss_gb:.2f} GB")


@dataclasses.dataclass
class TurbigenConfig:
    """Top level configuration class for turbigen.

    A run is uniquely defined by an instance of this class.

    """

    work_dir: Path
    """Directory in which to store run data."""

    fluid: turbigen.fluid.FluidConfig
    """Equation of state."""

    mean_line: turbigen.meanline_new.MeanLineConfig
    """Settings for the mean-line designer."""

    inlet: turbigen.inlet.InletConfig = None
    """Settings for the inlet boundary condition."""

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

    plug_dir: Path = None
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
    grid: ember.grid.Grid = None
    guess: ember.grid.Grid = None

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

    metadata_file: Path = None

    def copy(self):
        """Return a copy of the configuration."""
        return deepcopy(self)

    @property
    def fname(self):
        return self.work_dir / self.basename

    @property
    def task_id(self):
        """Integer indentifier for this run, extracted from work_dir."""
        try:
            return int(self.work_dir.name.split("_")[-1])
        except Exception:
            raise ValueError(
                "Could not extract task_id from work_dir name, ensure it ends with '_123'."
            )

    Re_surf: float = None
    """Set viscosity using a Reynolds number."""

    save_iteration_grids: bool = False
    """Save grid and guess at each iteration to work_dir."""

    ignore_guess: bool = False
    """Always use quasi-3D mean-line guess, even if a 3D guess is available."""

    @property
    def nrow(self):
        return len(self.blades)

    def save(self, fname=None, overwrite_pkl=True, use_gzip=True, write_grids=True):
        """Save the configuration to a YAML file inside work_dir.

        The working directory will be created if it does not exist.
        """

        if fname is None:
            fname = self.fname

        if not self.work_dir.exists():
            self.work_dir.mkdir(parents=True)

        # Check that the blades are not recambered
        for row in self.blades:
            for blade in row:
                if blade.is_recambered:
                    raise Exception(
                        "Cannot write configuration with recambered blades.\n"
                        "Use `undo_recamber()` to revert the camber parameters to degreeof of recamber."
                    )

        data = self.to_dict()

        # # Convert grid objects to filenames
        # for k in ["grid", "guess"]:
        #     val = getattr(self, k)
        #     # If not there remove the key
        #     if val is None or not write_grids:
        #         del data[k]
        #     else:
        #         # Otherwise, save the grid to a separate pickle
        #         # and replace the grid with the filename in yaml
        #         fname_pkl = self.work_dir / f"{k}.pkl.gz"
        #         data[k] = str(fname_pkl)
        #         if fname_pkl.exists() and not overwrite_pkl:
        #             logger.debug(f"Not overwriting existing {fname_pkl}")
        #             continue
        #         else:
        #             logger.debug(f"Saving {k} to {fname_pkl}")
        #             util.safe_pickle_dump(val, fname_pkl, zip=use_gzip)

        # if hasattr(self.mean_line, "actual"):
        #     data["mixed_out_flowfield"] = self.mean_line.actual.to_dump()
        # if not data["mixed_out_flowfield"]:
        #     del data["mixed_out_flowfield"]

        # # Convert convergence history to a filename
        # if self.solver and (conv := self.solver.convergence):
        #     fname_conv = self.work_dir / "convergence.npz"
        #     conv.save(fname_conv)
        #     data["solver"]["convergence"] = str(fname_conv)

        conf_fname = self.work_dir / fname
        logger.debug(f"Saving configuration to {conf_fname}")
        try:
            turbigen.yaml_utils.write_yaml(data, conf_fname)
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
        data["work_dir"] = str(data["work_dir"])
        if data["plug_dir"]:
            data["plug_dir"] = str(data["plug_dir"])

        for k in ["grid", "guess"]:
            data.pop(k, None)

        # Now convert any nested objects with to_dict methods
        for k in ["mean_line", "fluid", "annulus"]:
            obj = getattr(self, k)
            if not obj:
                continue
            data[k] = obj.to_dict()

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

        logger.warning(f"Importing plugins from {self.plug_dir}")
        # Find all python files recursively in the plugdir
        py_files = list(self.plug_dir.rglob("*.py"))
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
                logger.warning(f"Loaded plugin: {py_file}")
            except Exception as e:
                logger.warning(f"Failed to load {py_file}, error:")
                logger.warning(e)
                sys.exit(1)

    def __post_init__(self):
        """Convert input basic types to our desired types."""

        # Convert work_dir str to Path object
        self.work_dir = Path(self.work_dir).absolute()

        self.fluid = turbigen.fluid.FluidConfig.from_dict(self.fluid)

        # Convert plugdir str to Path object
        # And look for plugins
        if self.plug_dir:
            self.plug_dir = Path(self.plug_dir).absolute()
            self.find_plugins()

        # # If grid or guess is a filename, load and unpickle it
        # for k in ["grid", "guess"]:
        #     val = getattr(self, k)
        #     if isinstance(val, str) and not self._fast_init:
        #         try:
        #             with gzip.open(Path(val), "rb") as f:
        #                 setattr(self, k, pickle.load(f))
        #         except gzip.BadGzipFile:
        #             # If gzip fails, try loading without it
        #             with open(Path(val), "rb") as f:
        #                 setattr(self, k, pickle.load(f))

        # Convert inlet dict to InletConfig object
        if self.inlet:
            self.inlet = turbigen.inlet.InletConfig(**self.inlet)

        # Set up the meanline designer
        self.mean_line = turbigen.meanline_new.MeanLineConfig.from_dict(self.mean_line)

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
            mesh_type = self.mesh.pop("type", "h")
            if mesh_type == "h":
                Mesher = turbigen.hmesh.H
            elif mesh_type == "oh":
                Mesher = turbigen.ohmesh.OH
            self.mesh = Mesher(**self.mesh)

        # Lazy import the solver
        if self.solver:
            solver_name = self.solver.pop("type", "ember")
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
                    self.design_space.basedir = Path(self.work_dir)
                else:
                    self.design_space.basedir = Path(self.design_space.basedir)
            self.design_space.setup()

    def get_mean_line_nominal(self):
        """Calculate the nominal mean-line flow field."""

        # Mean-line design
        logger.info("Designing mean line...")
        self.mean_line.set_nominal(self.fluid.fluid)

        # Check mean-line design for problems
        logger.debug("Checking mean line...")
        self.mean_line.check_nominal()

        self.mean_line.warn()

    def adjust_ref(self):
        """Set thermodynamic datum and reference scales from nominal mean line."""

        fluid, L_ref = self.mean_line.nominal.adjust_ref()

        P_dtm, T_dtm = fluid.get_datum()
        logger.info("Setting reference scales:")
        logger.info(f"P_dtm={P_dtm:.2e} Pa, T_dtm={T_dtm:.1f} K")
        logger.info(
            f"rho_ref={fluid.rho_ref:.2f} kg/m^3, V_ref={fluid.V_ref:.1f} m/s, "
            f"L_ref={L_ref:.2f} m, Rgas_ref={fluid.Rgas_ref:.1f} J/kg/K"
        )

    def get_geometry(self):
        """Get the annulus and blade geometry."""

        # Annulus design
        logger.info("Designing annulus...")

        if not self.annulus:
            logger.error("No annulus defined, quitting.")
            sys.exit(0)

        self.annulus.setup_annulus(self.mean_line.nominal)
        logger.info(self.annulus.to_string())

        # Copy annulus x-coords into the mean-line
        self.mean_line.nominal.set_x(self.annulus.x_rms)

        # Blade design
        logger.info("Designing blades...")

        if not self.blades:
            logger.error("No blades defined, quitting.")
            sys.exit(0)

        for irow, row in enumerate(self.blades):
            # Set meridional locations
            for blade in row:
                blade.set_streamsurface(self.annulus.xr_row(irow))

    def blade_table(self):
        """Tabular string of per-row blade properties."""
        Nb = self.get_nblade()
        gaps = self.get_gaps()
        Re_surf = self.calculate_Re_surf()
        s_cm = self.get_pitch_chord()
        properties = [
            ("N_blade", Nb, "d"),
            ("Gap/m", gaps, ".4f"),
            ("s/cm", s_cm, ".3f"),
            ("Re_surf/1e5", Re_surf / 1e5, ".3f"),
        ]
        return util.format_table("Blades:", self.nrow, properties, paired=False)

    def get_nblade(self):
        Nb = np.full((len(self.blades),), 0, dtype=int)
        for irow, row in enumerate(self.blades):
            # Set number of blades using main blade
            if irow >= len(self.nblade):
                logger.error(
                    f"Missing 'nblade' entry for row {irow} in the input file."
                )
                sys.exit(1)
            Nb[irow] = np.round(
                self.nblade[irow]
                .get_blade_number(self.mean_line.nominal.get_row(irow), row[0])
                .item()
            )
        return Nb

    def get_pitch_chord(self):
        """Pitch-to-chord ratio at mid-span for each row."""
        rref = 0.5 * (
            self.mean_line.nominal.r_rms[::2] + self.mean_line.nominal.r_rms[1::2]
        )
        s = 2.0 * np.pi * rref / self.get_nblade()
        return s / self.annulus.chords(0.5)[1:-1:2]

    def check_pitch_chord(self, s_cm_lim=(0.2, 4.0)):
        # Warn if blade spacings are too narrow or wide
        s_cm = self.get_pitch_chord()
        if np.any(s_cm < s_cm_lim[0]):
            logger.warning(
                "WARNING: narrow blade spacings may cause problems with meshing"
            )
        if np.any(s_cm > s_cm_lim[1]):
            logger.warning(
                "WARNING: large blade spacings may cause problems with meshing"
            )

    def get_gaps(self):
        """Return dimensional tip gaps."""

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

        gap = gap_span * span
        return gap

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
        dsurf = self.calculate_d_wall()

        # Hub and casing wall distances are row means
        dhub = dcas = np.mean(dsurf)

        mesh_dir = self.work_dir / self.mesh.meshdir
        Omega = self.mean_line.nominal.Omega[::2]
        self.grid = self.mesh.make_grid(
            mesh_dir, self.get_machine(), dhub, dcas, dsurf, Omega
        )
        _log_ram("after make_grid")

        self.grid.set_L_ref(self.mean_line.nominal.L_ref)

        logger.info(f"n_cell/1e6={self.grid.size / 1e6:.1f}")

        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots()
        # b = self.grid[0]
        # C = b[:, b.nj // 2, :]
        # ax.plot(C.x, C.rt, "k-")
        # ax.plot(C.x.T, C.rt.T, "k-")
        # ax.axis("equal")
        # plt.show()
        #
        _log_ram("after set L_ref")
        self.grid.check_coordinates()
        _log_ram("after check_coords")
        self.grid.calculate_wdist()
        _log_ram("after wdst calc")

    def plot_mesh(self, spf=0.5):
        """Plot the structured mesh at a span cut and show interactively."""
        import turbigen.util_post

        blocks = turbigen.util_post.cut_span(self.grid, self.annulus, spf)
        blocks = ember.util.pitchwise_repeat(blocks, n=2)
        fig, ax = plt.subplots()
        ax.set_aspect("equal")
        ax.axis("off")
        for b in blocks:
            ax.plot(b.x, b.rt, "k-", lw=0.5)
            ax.plot(b.x.T, b.rt.T, "k-", lw=0.5)
        plt.tight_layout()
        plt.show()

        # Reset camber
        for irow, row in enumerate(self.blades):
            # Apply recamber, set meridional locations for
            # main and splitters
            for blade in row:
                blade.set_streamsurface(self.annulus.xr_row(irow))

    def get_machine(self):
        return turbigen.geometry.Machine(
            self.annulus, self.blades, self.get_nblade(), self.get_gaps(), None
        )

    def apply_bconds(self):
        # Get nominal exit pressure, mdot, shaft speed
        Omega = self.mean_line.nominal.Omega[::2].copy()
        Pout = self.mean_line.nominal.P[-1]
        mdot = self.mean_line.nominal.mdot[-1]
        Po1 = self.mean_line.nominal.Po[0]
        To1 = self.mean_line.nominal.To[0]
        Alpha1 = self.mean_line.nominal.Alpha[0]
        Beta1 = self.mean_line.nominal.Beta[0]

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
                scale = (
                    np.ptp(self.mean_line.nominal.P) / self.mean_line.nominal.mdot[-1]
                )
                Kpid = np.array(pid) * scale
                logger.info(f"Exit PID constants={pid}")
                # self.grid.apply_throttle(mdot, Kpid)
                outlet = self.grid.patches.outlet[0]
                outlet.set_throttle(mdot / outlet.Nb, tuple(Kpid))

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

        # # Inlet boundary condition
        # # Set inlet pitch angle using orientation of
        # # the inlet patch grid (assuming on a constant i face)
        # # This allow the annulus lines to differ from mean-line pitch angle
        # Ain = self.grid.inlet_patches[0].get_cut().dAi.sum(axis=(-1, -2, -3))
        # Beta1 = np.degrees(np.arctan2(Ain[1], Ain[0]))
        # Alpha1 = self.mean_line.nominal.Alpha[0]

        self.grid.patches.inlet[0].set_Po_To_Alpha_Beta(Po1, To1, Alpha1, Beta1)

        # Apply profile if available
        if self.inlet is not None and self.inlet.profiles is not None:
            logger.info("Applying inlet profile...")
            self.grid.inlet_patches[0].set_profile(
                self.inlet.spf,
                self.inlet.profiles,
            )

        # Outlet boundary condition
        self.grid.patches.outlet[0].set_P(Pout)
        self.grid.patches.outlet[0].set_backflow(
            self.mean_line.nominal.ho[-1],
            self.mean_line.nominal.s[-1],
            self.mean_line.nominal.Vr[-1],
            self.mean_line.nominal.Vt[-1],
        )

    def apply_guess(self):
        # Apply 3D guess if available, unless ignore_guess is set
        if self.guess and not self.ignore_guess:
            logger.info("Applying 3D guess...")
            self.grid.apply_guess_restart(self.guess)
        else:
            # Apply crude guess from mean_line
            if self.ignore_guess and self.guess:
                logger.warning(
                    "Ignoring 3D guess, applying quasi-3D mean-line guess..."
                )
            else:
                logger.info("Applying 2D guess...")

            block_guess = self.mean_line.nominal.to_quasi3d(
                self.annulus, self.get_nblade()
            )
            self.grid.apply_guess_quasi3d(block_guess)

        # Update the outlet static pressure based on the guess
        # This helps running multiple iterations of a throttled case
        self.grid.update_throttle()

    def run_solver(self):
        if not self.solver:
            logger.error("No solver configured, quitting.")
            sys.exit(0)

        run_args = self.grid, self.get_machine, self.work_dir / "solve"

        if self.solver.soft_start:
            logger.info("Soft start...")
            self.solver.robust().run(*run_args)
        logger.info("Running solver")
        self.solver.run(*run_args)

    def get_mean_line_actual(self):
        """Extract the actual mean-line flow field by mixing out CFD result."""

        # Find meridional coordinates of the cut planes
        xr_cut = self.annulus.get_offset_planes(self.cut_offset)

        # Take the cuts
        cuts = [ember.cut.unstructured(self.grid, xri.T) for xri in xr_cut]

        # Mix out and assemble into actual mean-line flow field
        self.mean_line.actual = self.mean_line.nominal.copy()
        for i, C in enumerate(cuts):
            try:
                Cm = ember.average.mix_out(C)
            except Exception:
                print("Failed to mix out row", i)
                print(C.conserved.mean(axis=(0, 1)))
                print(C.xrt.mean(axis=(0, 1)))
                print(C.shape)
                print(f"{C.dA.shape=}")
                print(f"{C.dA_tri.shape=}")
                print(f"{C.dA_quad.shape=}")
                print(C.dA[..., 0].min(), C.dA[..., 0].max())
                print("t", C.t.min(), C.t.max())
                print(ember.average.total_area(C))
                print(ember.average.flow_conserved(C))
                quit()
            self.mean_line.actual[i].set_r_rms(Cm.r)
            self.mean_line.actual[i].set_conserved(Cm.conserved)

        logger.info(self.mean_line.actual.to_string())

        # Back-calculate the design variables
        self.mean_line_actual = self.mean_line._backward(self.mean_line.actual)

        # Save the cuts as tm3 files for post-processing
        for i, C in enumerate(cuts):
            logger.info(f"Saving tm3 cuts for row {i}")
            C.to_tm3(self.work_dir / f"cut_{i}_Mam.tm3", Ma=C.Mam)

        for irow in range(self.nrow):
            # Calculate loss coefficient
            i_inlet = irow * 2
            i_exit = irow * 2 + 1
            Po1 = self.mean_line.actual.Po_rel[i_inlet]
            P2 = self.mean_line.actual.P[i_exit]
            C = cuts[i_exit]
            Yp = (Po1 - C.Po_rel) / (Po1 - P2)
            C.to_tm3(self.work_dir / f"cut_{i_exit}_Yp.tm3", Yp=Yp)

        del cuts

    def calculate_design_var_errors(self):
        """Calculate differences between nominal and actual design variables."""

        # Absolute error (dict comprehension), skip None actual values
        err = {
            k: v - self.mean_line_actual[k]
            for k, v in self.mean_line.design_vars.items()
            if self.mean_line_actual.get(k) is not None
        }

        # Relative error (dict comprehension, checking for zero nominal values)
        with np.errstate(divide="ignore", invalid="ignore"):
            rel_err = {
                k: (v - self.mean_line_actual[k]) / v * 100.0
                for k, v in self.mean_line.design_vars.items()
                if self.mean_line_actual.get(k) is not None
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
            actual = self.mean_line_actual.get(k)
            if actual is None:
                continue
            # Make very small values zero
            if np.isscalar(v):
                table.append(
                    [
                        k,
                        f"{v:.3g}",
                        f"{actual:.3g}",
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
                            f"{actual[i]:.3g}",
                            f"{err[k][i]:.2g}",
                            f"{rel_err[k][i]:.2g}",
                        ]
                    )

        # Additional vars not in nominal
        for k, v in self.mean_line_actual.items():
            if k not in self.mean_line.design_vars:
                if np.isscalar(v) or np.ndim(v) == 0:
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

    def calculate_Re_surf(self):
        """Calculate surface Reynolds number for all rows.

        Returns
        -------
        Re_surf : (nrow,) ndarray
            Surface Reynolds number for each blade row.
        """
        row_ref = [self.mean_line.nominal.get_ref(i) for i in range(self.nrow)]
        L_visc = np.array([row.mu / row.rho / row.V_rel for row in row_ref])
        ell = self.get_ell()
        return ell / L_visc

    def calculate_d_wall(self):
        """Calculate wall cell spacing for all rows using flat plate correlations.

        Uses the viscous length scale from local flow properties and yplus
        setting to estimate the wall-normal cell spacing required for
        resolving the boundary layer.

        Returns
        -------
        d_wall : (nrow,) ndarray
            Wall cell spacing for each blade row [m].
        """
        Re_surf = self.calculate_Re_surf()
        logger.debug("Calculating wall cell spacing using flat plate correlations...")
        logger.debug(f"Surface Reynolds numbers: {Re_surf}")
        row_ref = [self.mean_line.nominal.get_ref(i) for i in range(self.nrow)]

        # Flat plate skin friction correlation
        Cf = (2.0 * np.log10(Re_surf) - 0.65) ** -2.3
        logger.debug(f"Skin friction coefficients: {Cf}")

        # Shear stress at wall
        tauw = (
            Cf
            * 0.5
            * np.array([row.rho for row in row_ref])
            * np.array([row.V_rel**2 for row in row_ref])
        )
        logger.debug(f"Wall shear stresses: {tauw}")

        # Friction velocity
        Vtau = np.sqrt(tauw / np.array([row.rho for row in row_ref]))
        logger.debug(f"Friction velocities: {Vtau}")

        # Viscous length scale
        Lvisc = (
            np.array([row.mu for row in row_ref])
            / np.array([row.rho for row in row_ref])
            / Vtau
        )
        logger.debug(f"Viscous length scales: {Lvisc}")
        logger.debug(f"yplus setting: {self.mesh.yplus}")

        dwall = self.mesh.yplus * Lvisc
        logger.debug(f"Calculated wall cell spacings: {dwall}")

        return dwall

    def set_mu_from_Re_surf(self):
        raise NotImplementedError("set_mu_from_Re_surf is not implemented yet.")
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

    def design_and_run(self, skip, skip_post=False, plot_mesh=None):
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

        _log_ram("start")

        # Calculate the nominal mean-line flow
        self.get_mean_line_nominal()
        _log_ram("after mean-line nominal")

        # Use the mean-line to adjust reference scales
        self.adjust_ref()

        logger.info(self.mean_line.nominal.to_string())

        self.get_geometry()
        self.apply_recamber()
        _log_ram("after geometry")

        self.check_pitch_chord()

        logger.info(self.blade_table())

        # # Handle restarts
        if self.grid:
            # If we already have a grid, use it as the guess
            self.guess = [b.get_restart() for b in self.grid]
            del self.grid
            # Change CFD settings to resume the simulation
            self.solver = self.solver.restart()
            logger.info("Restarting from existing grid and solution...")

        # Set viscosity from Reynolds number if given
        if self.Re_surf:
            self.set_mu_from_Re_surf()

        # We are now ready to generate mesh and run CFD
        # There are three cases to consider
        # (1) Skipping from guess: just use existing mesh and solution
        # (2) Skipping from cold: mesh but do not run the CFD solver
        # (3) Normal operation: mesh and run the CFD solver

        # Generate mesh in cases (2) and (3)
        if not (skip and self.grid):
            logger.info(f"Generating {self.mesh.__class__.__name__} mesh...")
            self.setup_mesh()  # Overwrite self.grid with a new mesh
            self.grid.set_fluid(self.mean_line.nominal.fluid)
            _log_ram("after mesh generation")
            if plot_mesh is not None:
                self.plot_mesh(plot_mesh)
                sys.exit(0)
            self.apply_bconds()
            _log_ram("after apply bconds")
            self.apply_guess()
            _log_ram("after apply guess")
        else:
            logger.info("Skipping and already have a guess, not generating mesh...")

        # In case (3), run the CFD solver
        if not skip:
            logger.info(f"Running solver {self.solver.__class__.__name__}...")
            self.run_solver()
            _log_ram("after solver")
            self.solver.convergence.to_json(self.work_dir)
        else:
            logger.info("Skipping solver run.")

        # The flow field is ready in grid, post-process it
        logger.info("Post-processing...")
        self.get_mean_line_actual()
        _log_ram("after mean-line actual")
        self.undo_recamber()

    def interpolate_all_iterators(self):
        """Use fitted design space to set values for all iterated variables."""

        for iterator in self.iterate:
            iterator.interpolate(self)

    def step_iterate(self):
        """Apply all iterators to the configuration."""

        log_data = {}
        converged = {}
        tol = {}

        for iterator in self.iterate:
            conv_now, log_data_now = iterator.update(self)
            _log_ram(f"after iterator {iterator}")

            # Update the overall convergence flag and log data
            name = util.camel_to_snake(iterator.__class__.__name__)
            converged[name] = conv_now
            log_data.update(log_data_now)
            tol.update(iterator.get_tolerances(self))

        return converged, log_data, tol

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
        with PdfPages(self.work_dir / "post.pdf") as pdf:
            for poster in self.post_process:
                logger.debug(f"Running post function {poster}")
                try:
                    poster.post(self, pdf)
                except Exception:
                    logger.error(f"Failed to run post function {poster}")
                    traceback.print_exc()
                _log_ram(f"after post {poster}")
        # Ensure all figures are closed
        plt.close("all")
