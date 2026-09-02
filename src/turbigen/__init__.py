"""An experimental config architecture built on a single serialisation protocol.

Everything in a config file is a :class:`~turbigen.node.Node`: a frozen
dataclass that builds itself from a dict and turns back into one. Importing
this package registers the built-in fluids and designs.
"""

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
    RowAnnulus,
)
from turbigen.blade import (
    Blade,
    BladeCount,
    BladeDesign,
    Circulation,
    DiffusionFactor,
    FixedCount,
    Row,
    SectionDesign,
)
from turbigen.camber import Bernstein, CamberDesign, CamberLine, Quadratic
from turbigen.config import Config
from turbigen.database import Database
from turbigen.hmesh import H
from turbigen.mesh import Mesher, WallSpacing
from turbigen.thickness import Taylor, ThicknessDesign
from turbigen.design import DesignError, MeanLineDesign
from turbigen.fluid import Fluid, PerfectFluid, RealFluid
from turbigen.iterate import Iteration, Iterator
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
    VelocityTrianglePlot,
)
from turbigen.batch import Batch
from turbigen.chic import Chic
from turbigen.bconds import InletProfile, Legendre, OperatingPoint, Sampled
from turbigen.result import Result
from turbigen.solver import Ember, Solver
from turbigen.node import Node

# _version.py is generated at build time by setuptools-scm (see pyproject.toml);
# it is gitignored, so fall back to a runtime metadata lookup when running from
# an uninstalled source tree that has never been built.
try:
    from turbigen._version import __version__
except ImportError:  # pragma: no cover - only hit in a bare, unbuilt source tree
    from importlib.metadata import version, PackageNotFoundError

    try:
        __version__ = version("turbigen")
    except PackageNotFoundError:
        __version__ = "0.0.0+unknown"

__all__ = [
    "Annulus",
    "AnnulusDesign",
    "AspectRatio",
    "PchipAnnulus",
    "Bernstein",
    "Blade",
    "BladeCount",
    "BladeDesign",
    "CamberDesign",
    "CamberLine",
    "Circulation",
    "Config",
    "Database",
    "DiffusionFactor",
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
    "SectionDesign",
    "RowAnnulus",
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
    "VelocityTrianglePlot",
    "Result",
    "Batch",
    "Job",
    "Slurm",
    "Task",
    "Tsp",
    "PerfectFluid",
    "RealFluid",
]
