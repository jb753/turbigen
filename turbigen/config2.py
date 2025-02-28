"""Initial thoughts on an improved config class."""

import dataclasses
from pathlib import Path
import turbigen.fluid
import turbigen.meanline2
import turbigen.solver
import turbigen.grid
import turbigen.yaml
import importlib
import turbigen.annulus
import turbigen.inlet
import turbigen.mesh
import turbigen.blade
from turbigen import util
from typing import List


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

    mesh: turbigen.mesh.Mesher
    """Settings for mesh generation."""

    solver: turbigen.solver.BaseSolver
    """Settings for flow solution."""

    guess: turbigen.grid.Grid = None

    def save(self, fname: str = "config.yaml"):
        """Save the configuration to a YAML file.

        The working directory will be created if it does not exist.
        """
        if not self.workdir.exists():
            self.workdir.mkdir(parents=True)
            turbigen.yaml.write_yaml(self.to_dict(), self.workdir / fname)

    def to_dict(self):
        """Convert the config to a dictionary."""

        # Built-in dataclasses method gets us most of the way there
        data = dataclasses.asdict(self)

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
        self.workdir = Path(self.workdir)

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


#
#
# if __name__ == "__main__":
#     inlet_dat = {"Po": 101325, "To": 288.15, "cp": 1000, "gamma": 1.4}
#     inlet = util.init_subclass_by_signature(InletConfig, inlet_dat)
#     print(type(inlet))
#     print(inlet)
