"""An experimental config architecture built on a single serialisation protocol.

Everything in a config file is a :class:`~turbigen.node.Node`: a frozen
dataclass that builds itself from a dict and turns back into one. Importing
this package registers the built-in fluids and designs.
"""

from turbigen import (
    bconds,
    case,
    designs,
    guess,
    iterate,
    mixout,
)
from turbigen.annulus import (
    Annulus,
    AnnulusDesign,
    AspectRatio,
    FixedAxialChord,
    PchipAnnulus,
    RowAnnulus,
)
from turbigen.batch import Batch
from turbigen.bconds import InletProfile, Legendre, OperatingPoint, Sampled
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
from turbigen.chic import Chic
from turbigen.config import Config
from turbigen.database import Database
from turbigen.design import DesignError, MeanLineDesign
from turbigen.fluid import Fluid, PerfectFluid, RealFluid
from turbigen.hmesh import H
from turbigen.iterate import Iteration, Iterator
from turbigen.job import Job, Slurm, Task, Tsp
from turbigen.machine import Machine
from turbigen.meanline import MeanLine
from turbigen.mesh import Mesher, WallSpacing
from turbigen.metric import Metric, SurfaceDissipation
from turbigen.node import Node
from turbigen.post import (
    AnnulusPlot,
    ContourPlot,
    ConvergencePlot,
    Post,
    SectionsPlot,
    SpanwisePlot,
    SurfacePlot,
    VelocityTrianglePlot,
)
from turbigen.result import Result
from turbigen.solver import Ember, Solver
from turbigen.thickness import Taylor, ThicknessDesign

# _version.py is generated at build time by setuptools-scm (see pyproject.toml);
# it is gitignored, so fall back to a runtime metadata lookup when running from
# an uninstalled source tree that has never been built.
try:
    from turbigen._version import __version__
except ImportError:  # pragma: no cover - only hit in a bare, unbuilt source tree
    from importlib.metadata import PackageNotFoundError, version

    try:
        __version__ = version("turbigen")
    except PackageNotFoundError:
        __version__ = "0.0.0+unknown"

__all__ = [
    "Annulus",
    "AnnulusDesign",
    "AnnulusPlot",
    "AspectRatio",
    "Batch",
    "Bernstein",
    "Blade",
    "BladeCount",
    "BladeDesign",
    "CamberDesign",
    "CamberLine",
    "Chic",
    "Circulation",
    "Config",
    "ContourPlot",
    "ConvergencePlot",
    "Database",
    "DesignError",
    "DiffusionFactor",
    "Ember",
    "FixedAxialChord",
    "FixedCount",
    "Fluid",
    "H",
    "InletProfile",
    "Iterator",
    "Job",
    "Legendre",
    "Machine",
    "MeanLine",
    "MeanLineDesign",
    "Mesher",
    "Metric",
    "Node",
    "OperatingPoint",
    "PchipAnnulus",
    "PerfectFluid",
    "Post",
    "Quadratic",
    "RealFluid",
    "Result",
    "Row",
    "RowAnnulus",
    "Sampled",
    "SectionDesign",
    "SectionsPlot",
    "Slurm",
    "Solver",
    "SpanwisePlot",
    "SurfaceDissipation",
    "SurfacePlot",
    "Task",
    "Taylor",
    "ThicknessDesign",
    "Tsp",
    "VelocityTrianglePlot",
    "WallSpacing",
    "bconds",
    "case",
    "guess",
    "mixout",
]
