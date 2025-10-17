import turbigen.util
import os
from pathlib import Path
from dataclasses import dataclass
from turbigen.solvers.base import BaseSolver

logger = turbigen.util.make_logger()


@dataclass
class Ember(BaseSolver):
    """Settings with default values for Plot3D export."""

    _name = "ember"

    workdir: Path = None

    def robust(self):
        """Change settings for a more stable simulation."""
        return self

    def restart(self):
        """Restart the simulation from a previous solution."""
        return self

    def run(self, grid, machine, workdir):

        logger.info("Ember solver placeholder - no action taken.")

        # self.convergence = run(grid, self, machine, workdir)
