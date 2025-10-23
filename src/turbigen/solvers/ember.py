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
        r_ref = grid.get_r_ref()
        Nb = np.array([row[0].Nb for row in grid.rows])
        pitch_ref = 2.0 * np.pi * r_ref / Nb
        print(f"r_ref: {r_ref}")
        print(f"Nb: {Nb}")
        print(f"pitch_ref: {pitch_ref}")
        xllim = 0.03 * pitch_ref
        print(f"xllim: {xllim}")

        config = ember.config.SolverConfig(
            order=3,
            n_levels=3,
            n_step_avg=250,
            n_step=1000,
            n_step_ramp=500,
            cfl_min=0.5,
            cfl_max=8.0,
            rf_inlet=0.2,
            sf2_adapt=1.0,
            sf4=0.01,
            Ki=0.6,
            xllim=xllim,
        )
        self.convergence = ember.loop.multigrid(grid, config)
