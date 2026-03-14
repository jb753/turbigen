import turbigen.util
import os
from pathlib import Path
from dataclasses import dataclass
from turbigen.solvers.base import BaseSolver

logger = turbigen.util.make_logger()

import numpy as np

import ember.run
import ember.patch
import ember.config
import ember.fortran


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

        for patch in grid.patches.outlet:
            # patch.set_radial_equilibrium(rf=0.1)
            patch.set_P(patch.P * 0.95)

        import matplotlib.pyplot as plt

        config = ember.config.SolverConfig(
            n_step=10000,
            n_step_avg=1,
            n_step_log=50,
            n_levels=3,
            cfl_min=0.1,
            debug=True,
            cfl_max=4.0,
            rtol=1e-6,
            inviscid=True,
            # shear_work=False,
            full_mgrid=True,
            fac_mgrid=0.5,
            # i_level_stop=2,
            xllim=xllim,
            sf4=0.02,
            sf2_adapt=1.0,
            const_smoothing=False,
        )

        try:
            self.convergence = ember.run.loop(grid, config)
        except SystemExit:
            pass

        b = grid[0]
        C = b[:, b.nj // 2, b.nk // 2]

        fig, ax = plt.subplots()
        ax.plot(C.x, C.cfl_cell)
        ax.set_ylabel("CFL number")

        fig, ax = plt.subplots()
        ax.plot(C.x, C.Alpha_rel)
        ax.set_ylabel("Relative flow angle")

        fig, ax = plt.subplots()
        ax.plot(C.x, C.Alpha)
        ax.set_ylabel("Absolute flow angle")

        print(C.U.mean())

        fig, ax = plt.subplots()
        b = grid[0]
        C = b[:, 10, :]
        lev = np.arange(0.0, 1.001, 0.1)
        cm = ax.contourf(C.x, C.rt, C.Ma_rel, levels=lev)
        ax.axis("equal")
        plt.colorbar(cm, ax=ax)

        plt.show()

        fig, ax = plt.subplots()
        b = grid[0]
        C = b[:, b.nj // 2, :]
        cm = ax.contourf(C.x, C.rt, C.Alpha_rel)
        ax.axis("equal")
        ax.set_title("Relative flow angle")
        plt.colorbar(cm, ax=ax)

        fig, ax = plt.subplots()
        b = grid[0]
        C = b[:, b.nj // 2, :]
        cm = ax.contourf(C.x, C.rt, C.Alpha)
        ax.axis("equal")
        ax.set_title("Absolute flow angle")
        plt.colorbar(cm, ax=ax)

        fig, ax = plt.subplots()
        b = grid[0]
        C = b[:, b.nj // 2, :]
        cm = ax.contourf(C.x, C.rt, C.Vx)
        ax.axis("equal")
        ax.set_title("Axial velocity")
        plt.colorbar(cm, ax=ax)

        fig, ax = plt.subplots()
        b = grid[0]
        C = b[:, b.nj // 2, :]
        cm = ax.contourf(C.x, C.rt, C.To)
        ax.axis("equal")
        ax.set_title("Stagnation temp")
        plt.colorbar(cm, ax=ax)

        plt.show()
