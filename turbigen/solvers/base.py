"""Define the basic interface that all solvers must conform to."""
from dataclasses import dataclass
from pathlib import Path
import os


@dataclass
class BaseSolver:
    """Settings and methods common to all solvers."""

    workdir: Path
    """Working directory to run the simulation in."""

    environment_script: Path
    """Setup environment shell script to be sourced before running."""

    skip: bool = False
    """False to run the CFD as normal, True to write out initial guess and read
    back in, or use a previous solution if available."""

    soft_start: bool = False
    """Run a robust initial guess solution first, then restart."""

    _ntask: int = 1  # Number of tasks for parallel executeion
    _nnode: int = 1  # Number of nodes for parallel executeion
    _name: str = "base"

    def _robust(self):
        """Create a copy of the config with more robust settings."""
        raise NotImplementedError()

    def __post_init__(self):
        """Validate the input data"""
        if not os.path.isdir(self.workdir):
            raise Exception(f"Working directory {self.workdir} does not exist")
        if self._ntask < 1:
            raise Exception(f"ntask={self._ntask} should be > 0")
        if self._nnode < 1:
            raise Exception(f"nnode={self._nnode} should be > 0")
