"""An experimental config architecture built on a single serialisation protocol.

Everything in a config file is a :class:`~turbigen.node.Node`: a frozen
dataclass that builds itself from a dict and turns back into one. Importing
this package registers the built-in fluids and designs.
"""

import importlib.metadata

from turbigen import designs  # noqa: F401 - registers the built-in designs
from turbigen import bconds  # noqa: F401 - boundary conditions for a new grid
from turbigen import case  # noqa: F401 - reading and writing a config with its result
from turbigen import mixout  # noqa: F401 - reducing a solution to a mean line
from turbigen import guess  # noqa: F401 - initial flow field for a new grid
from turbigen import iterate  # noqa: F401 - closing the loop on a design
from turbigen.annulus import (
    Annulus,
    AnnulusDesign,
    AspectRatio,
    FixedAxialChord,
    PchipAnnulus,
    StreamSurface,
)
from turbigen.blade import (
    Blade,
    BladeCount,
    BladeDesign,
    Circulation,
    FixedCount,
    Row,
    Section,
)
from turbigen.camber import CamberDesign, CamberLine, Quadratic
from turbigen.config import Config
from turbigen.database import Database
from turbigen.hmesh import H
from turbigen.mesh import Mesher, WallSpacing
from turbigen.thickness import Taylor, ThicknessDesign
from turbigen.design import DesignError, MeanLineDesign, check_round_trip
from turbigen.fluid import Fluid, PerfectFluid
from turbigen.iterate import Iterator
from turbigen.job import Job, Slurm, Task, Tsp
from turbigen.machine import Machine
from turbigen.meanline import MeanLine
from turbigen.post import (
    AnnulusPlot,
    ContourPlot,
    ConvergencePlot,
    Post,
    SectionsPlot,
    SurfacePlot,
)
from turbigen.batch import Batch
from turbigen.chic import Chic
from turbigen.bconds import InletProfile, Legendre, OperatingPoint, Sampled
from turbigen.result import Result
from turbigen.solver import Ember, Solver
from turbigen.node import Node

__version__ = importlib.metadata.version("turbigen")
"""Version of the distribution this package ships in.

The distribution is still called `turbigen`, being what is installed, and this
is one of two packages inside it. Read from the metadata rather than imported
from the package being replaced, which is on its way out.
"""

__all__ = [
    "Annulus",
    "AnnulusDesign",
    "AspectRatio",
    "PchipAnnulus",
    "Blade",
    "BladeCount",
    "BladeDesign",
    "CamberDesign",
    "CamberLine",
    "Circulation",
    "Config",
    "Database",
    "FixedAxialChord",
    "FixedCount",
    "bconds",
    "case",
    "mixout",
    "Ember",
    "guess",
    "Solver",
    "H",
    "Mesher",
    "Quadratic",
    "Row",
    "Section",
    "StreamSurface",
    "Taylor",
    "ThicknessDesign",
    "WallSpacing",
    "Machine",
    "DesignError",
    "Fluid",
    "Iterator",
    "MeanLine",
    "MeanLineDesign",
    "Chic",
    "Node",
    "InletProfile",
    "Legendre",
    "Sampled",
    "OperatingPoint",
    "AnnulusPlot",
    "ContourPlot",
    "ConvergencePlot",
    "Post",
    "SectionsPlot",
    "SurfacePlot",
    "Result",
    "Batch",
    "Job",
    "Slurm",
    "Task",
    "Tsp",
    "PerfectFluid",
    "check_round_trip",
]
