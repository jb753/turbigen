"""An experimental config architecture built on a single serialisation protocol.

Everything in a config file is a :class:`~turbigen2.node.Node`: a frozen
dataclass that builds itself from a dict and turns back into one. Importing
this package registers the built-in fluids and designs.
"""

from turbigen2 import designs  # noqa: F401 - registers the built-in designs
from turbigen2.annulus import Annulus, AnnulusDesign, FixedAxialChord
from turbigen2.config import Config
from turbigen2.design import DesignError, MeanLineDesign, check_round_trip
from turbigen2.fluid import Fluid, PerfectFluid
from turbigen2.machine import Machine
from turbigen2.meanline import MeanLine
from turbigen2.post import AnnulusPlot, Post
from turbigen2.result import Result
from turbigen2.node import Node

__all__ = [
    "Annulus",
    "AnnulusDesign",
    "Config",
    "FixedAxialChord",
    "Machine",
    "DesignError",
    "Fluid",
    "MeanLine",
    "MeanLineDesign",
    "Node",
    "AnnulusPlot",
    "Post",
    "Result",
    "PerfectFluid",
    "check_round_trip",
]
