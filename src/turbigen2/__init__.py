"""An experimental config architecture built on a single serialisation protocol.

Everything in a config file is a :class:`~turbigen2.node.Node`: a frozen
dataclass that builds itself from a dict and turns back into one. Importing
this package registers the built-in fluids and designs.
"""

from turbigen2 import designs  # noqa: F401 - registers the built-in designs
from turbigen2 import bconds  # noqa: F401 - boundary conditions for a new grid
from turbigen2 import case  # noqa: F401 - reading and writing a config with its result
from turbigen2 import mixout  # noqa: F401 - reducing a solution to a mean line
from turbigen2 import guess  # noqa: F401 - initial flow field for a new grid
from turbigen2 import iterate  # noqa: F401 - closing the loop on a design
from turbigen2.annulus import (
    Annulus,
    AnnulusDesign,
    AspectRatio,
    FixedAxialChord,
    PchipAnnulus,
    StreamSurface,
)
from turbigen2.blade import (
    Blade,
    BladeCount,
    BladeDesign,
    Circulation,
    FixedCount,
    Row,
    Section,
)
from turbigen2.camber import CamberDesign, CamberLine, Quadratic
from turbigen2.config import Config
from turbigen2.database import Database
from turbigen2.hmesh import H
from turbigen2.mesh import Mesher, WallSpacing
from turbigen2.thickness import Taylor, ThicknessDesign
from turbigen2.design import DesignError, MeanLineDesign, check_round_trip
from turbigen2.fluid import Fluid, PerfectFluid
from turbigen2.iterate import Iterator
from turbigen2.job import Job, Slurm, Task, Tsp
from turbigen2.machine import Machine
from turbigen2.meanline import MeanLine
from turbigen2.post import (
    AnnulusPlot,
    ContourPlot,
    ConvergencePlot,
    Post,
    SectionsPlot,
    SurfacePlot,
)
from turbigen2.batch import Batch
from turbigen2.chic import Chic
from turbigen2.bconds import InletProfile, Legendre, OperatingPoint, Sampled
from turbigen2.result import Result
from turbigen2.solver import Ember, Solver
from turbigen2.node import Node

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
