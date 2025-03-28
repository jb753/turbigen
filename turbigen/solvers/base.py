"""Define the basic interface that all solvers must conform to."""

import dataclasses
import numpy as np


@dataclasses.dataclass
class BaseSolver:
    """Settings and methods common to all solvers."""

    skip: bool = False
    """False to run the CFD as normal, True to write out initial guess and read
    back in, or use a previous solution if available."""

    soft_start: bool = False
    """Run a robust initial guess solution first, then restart."""

    ntask: int = 1  # Number of tasks for parallel executeion
    nnode: int = 1  # Number of nodes for parallel executeion
    _name: str = "base"

    def _robust(self):
        """Create a copy of the config with more robust settings."""
        raise NotImplementedError()

    def __post_init__(self):
        """Validate the input data"""
        if self.ntask < 1:
            raise Exception(f"ntask={self.ntask} should be > 0")
        if self.nnode < 1:
            raise Exception(f"nnode={self.nnode} should be > 0")

    def replace(self, **kwargs):
        return dataclasses.replace(self, **kwargs)
