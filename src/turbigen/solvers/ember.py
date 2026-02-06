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
            n_step_avg=1000,
            n_step_log=50,
            n_step=4000,
            n_step_ramp=0,
            cfl_min=0.1,
            cfl_max=4.0,
            xllim=xllim,
            sf4=0.001,
            sf2_adapt=0.5,
            end_stagger=0.0,
            rtol_conserved=(1e-2, 1e-2, 1e-2, 1e-2, 1e-2),
        )

        try:
            self.convergence = ember.loop.multiloop(grid, config)
        except SystemExit:
            pass

        import matplotlib.pyplot as plt

        # fig, ax = plt.subplots()
        # b = grid[0]
        # C = b[:, b.nj // 2, :]
        # ax.contourf(C.x, C.r * C.t, C.Ma_rel)
        # ax.axis("equal")
        # plt.show()
