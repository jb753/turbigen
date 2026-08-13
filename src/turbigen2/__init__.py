"""An experimental config architecture built on a single serialisation protocol.

Everything in a config file is a :class:`~turbigen2.node.Node`: a frozen
dataclass that builds itself from a dict and turns back into one. Importing
this package registers the built-in fluids and designs.
"""

from turbigen2 import designs  # noqa: F401 - registers the built-in designs
from turbigen2.config import Config
from turbigen2.design import DesignError, MeanLineDesign, check_round_trip
from turbigen2.fluid import Fluid, PerfectFluid
from turbigen2.meanline import MeanLine
from turbigen2.node import Node

__all__ = [
    "Config",
    "DesignError",
    "Fluid",
    "MeanLine",
    "MeanLineDesign",
    "Node",
    "PerfectFluid",
    "check_round_trip",
]
