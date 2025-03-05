"""Initial thoughts on an improved config class."""

import dataclasses
import numpy as np
import pickle
from pathlib import Path
import turbigen.fluid
import turbigen.meanline2
import turbigen.solver
import turbigen.grid
import turbigen.geometry
import turbigen.yaml
import importlib
import turbigen.annulus
import turbigen.inlet
import turbigen.mesh
import turbigen.blade
import turbigen.nblade
from turbigen import util
from typing import List

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

    mean_line: turbigen.meanline2.MeanLineDesigner
    """Settings for the mean-line designer."""

    annulus: turbigen.annulus.AnnulusDesigner
    """Settings for the annulus designer."""

    blades: List[List[turbigen.blade.BladeDesigner]]
    """Settings for the blade designers."""

    nblade: List[turbigen.nblade.BladeNumberConfig]
    """Settings for blade number selection."""

    mesh: turbigen.mesh.Mesher
    """Settings for mesh generation."""

    solver: turbigen.solver.BaseSolver
    """Settings for flow solution."""

    grid: turbigen.grid.Grid = None
    guess: turbigen.grid.Grid = None

    @property
    def nrow(self):
        return len(self.blades)

    def save(self, fname: str = "config.yaml"):
        """Save the configuration to a YAML file inside workdir.

        The working directory will be created if it does not exist.
        """

        if not self.workdir.exists():
            self.workdir.mkdir(parents=True)

        data = self.to_dict()

        # Convert grid objects to filenames
        for k in ["grid", "guess"]:
            val = getattr(self, k)
            # If not there remove the key
            if val is None:
                del data[k]
            else:
                # Otherwise, save the grid to a separate pickle
                # and replace the grid with the filename
                fname = self.workdir / f"{k}.pkl"
                pickle.dump(val, fname.open("wb"))
                data[k] = str(fname)

        conf_fname = self.workdir / fname
        turbigen.yaml.write_yaml(data, conf_fname)

        return conf_fname

    def to_dict(self):
        """Convert the config to a dictionary."""

        # Built-in dataclasses method gets us most of the way there
        data = dataclasses.asdict(self)

        # Put workdir into a string
        data["workdir"] = str(data["workdir"])

        # Convert the meanline designer to a dictionary
        data["mean_line"] = self.mean_line.to_dict()

        # Convert the annulus designer to a dictionary
        data["annulus"] = self.annulus.to_dict()

        # Convert the annulus designer to a dictionary
        data["blades"] = []
        for row in self.blades:
            if len(row) == 1:
                data["blades"].append(row[0].to_dict())
            else:
                data["blades"].append([])
                for blade in row:
                    data["blades"][-1].append(blade.to_dict())

        # Restore the mesh type
        data["mesh"]["type"] = util.camel_to_snake(self.mesh.__class__.__name__)

        # Restore the solver type
        data["solver"]["type"] = util.camel_to_snake(self.solver.__class__.__name__)

        return data

    def __post_init__(self):
        """Convert input basic types to our desired types."""

        # Convert workdir str to Path object
        self.workdir = Path(self.workdir).absolute()

        # Convert inlet dict to InletConfig object
        self.inlet = util.init_subclass_by_signature(
            turbigen.inlet.InletConfig, self.inlet
        )

        # Set up the meanline designer
        MeanLineDesigner = util.get_subclass_by_name(
            turbigen.meanline2.MeanLineDesigner, self.mean_line.pop("type")
        )
        self.mean_line = MeanLineDesigner(self.mean_line)

        # Set up the annulus designer
        AnnulusDesigner = util.get_subclass_by_name(
            turbigen.annulus.AnnulusDesigner, self.annulus.pop("type", "smooth")
        )
        self.annulus = AnnulusDesigner(self.annulus)

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
        Mesher = util.get_subclass_by_name(
            turbigen.mesh.Mesher, self.mesh.pop("type", "h")
        )
        self.mesh = Mesher(**self.mesh)

        # Lazy import the solver
        solver_name = self.solver.pop("type")
        importlib.import_module(f".{solver_name}", package="turbigen.solvers")
        Solver = util.get_subclass_by_name(turbigen.solver.BaseSolver, solver_name)
        self.solver = Solver(**self.solver)

        # If a filename is present in the grid key, load and unpickle it
        if self.grid:
            self.grid = pickle.load((self.workdir / self._grid_fname).open("rb"))

    def get_mean_line_nominal(self):
        """Calculate the nominal mean-line flow field."""

        So1 = self.inlet.get_inlet()
        logger.info(f"Inlet: {So1}")

        # Mean-line design
        self.mean_line.setup_mean_line(So1)
        logger.info(self.mean_line.nominal)

        # Check mean-line design for problems
        logger.info("Checking mean-line conservation...")
        if not self.mean_line.nominal.check():
            self.mean_line.nominal.show_debug()
            raise Exception(
                "Mean-line conservation checks failed, have printed debugging information"
            ) from None
        logger.info("Checking mean-line inversion...")
        self.mean_line.check_backward(self.mean_line.nominal)
        self.mean_line.nominal.warn()

    def get_geometry(self):
        """Get the annulus and blade geometry."""

        # Annulus design
        logger.info("Designing annulus...")
        self.annulus.setup_annulus(self.mean_line.nominal)
        logger.info(f"{self.annulus}")

        # Blade design
        logger.info("Designing blades...")
        for irow, row in enumerate(self.blades):
            # Apply recamber, set meridional locations for
            # main and splitters
            for blade in row:
                blade.apply_recamber(self.mean_line.nominal)
                blade.set_streamsurface(self.annulus.xr_row(irow))

        self.check_pitch_chord()
        logger.info(f"Nblade: {self.get_nblade()}")
        logger.info(f"Tip gaps: {self.get_gaps()}")

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
        # Get dimensional tip gaps
        Href = 0.5 * (
            self.mean_line.nominal.span[::2] + self.mean_line.nominal.span[1::2]
        )
        return Href * np.array([b[0].tip for b in self.blades])

    def get_Re_surf(self):
        # Find wall distances for each row
        ell = np.array(
            [self.blades[irow][0].surface_length(0.5) for irow in range(self.nrow)]
        )
        Re_surf = ell / self.mean_line.nominal.L_visc
        logger.info(f"Re_surf={util.format_array(Re_surf)}")

    def setup_mesh(self):
        logger.info("Making mesh...")

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
        self.grid = self.mesh.make_grid(mesh_dir, self.get_machine(), dhub, dcas, dsurf)

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

    def get_machine(self):
        return turbigen.geometry.Machine(
            self.annulus, self.blades, self.get_nblade(), self.get_gaps(), None
        )

    def apply_bconds(self):
        # Set the rotation types
        Omega = self.mean_line.nominal.Omega[::2]
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

        # Outlet boundary condition
        self.grid.apply_outlet(self.mean_line.nominal.P[-1])

    def apply_guess(self):
        # Choose whether the blocks are real or perfect
        So1 = self.inlet.get_inlet()
        g = self.grid
        if isinstance(So1, turbigen.fluid.PerfectState):
            self.grid = turbigen.grid.Grid([b.to_perfect() for b in g])
        elif isinstance(So1, turbigen.fluid.RealState):
            self.grid = turbigen.grid.Grid([b.to_real() for b in g])
        else:
            raise Exception("Unrecognised inlet state type")

        # Apply crude guess from mean_line
        self.grid.apply_guess_meridional(
            self.mean_line.nominal.interpolate_guess(self.annulus)
        )

        if self.guess:
            g.apply_guess_3d(self.guess)

    def run_solver(self):
        self.solver.run(self.grid, self.get_machine)

    def design_and_run(self):
        """Run a configuration file through the CFD solver.

        This will do the following:
            1. Get inlet state;
            2. Design the meanline;
            3. Design the annulus;
            4. Design the blades;
            5. Generate the mesh;
            6. Run the flow solver;

        """

        self.get_mean_line_nominal()
        self.get_geometry()
        self.get_Re_surf()
        self.setup_mesh()
        self.apply_bconds()
        self.apply_guess()
        self.run_solver()
