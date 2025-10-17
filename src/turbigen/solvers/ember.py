import turbigen.util
import os
from pathlib import Path
from dataclasses import dataclass
from turbigen.solvers.base import BaseSolver

logger = turbigen.util.make_logger()

import numpy as np

import ember.loop
import ember.config


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

        config = ember.config.SolverConfig(
            order=3,
            n_levels=3,
            n_step_avg=250,
            n_step=100,
            cfl_min=0.5,
            cfl_max=1.5,
            sf4=0.01,
            sf2_adapt=2.0,
        )
        self.conv = ember.loop.multigrid(grid, config)
        # self.convergence = run(grid, self, machine, workdir)

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        b = grid[0]
        C = b[:, b.nj // 2, :]
        lev = np.linspace(0, 1, 11)
        cm = ax.contourf(C.x, C.r * C.t, C.Ma, lev, cmap="cubehelix")
        plt.colorbar(cm)
        ax.axis("equal")

        fig, ax = plt.subplots()
        b = grid[0]
        C = b[:, b.nj // 2, :]
        cm = ax.contourf(C.x, C.r * C.t, C.Alpha, cmap="cubehelix")
        plt.colorbar(cm)
        ax.axis("equal")

        print(b.rho.min(), b.rho.max())
        plt.show()
