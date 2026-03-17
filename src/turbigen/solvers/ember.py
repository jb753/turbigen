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
            patch.set_radial_equilibrium(rf=0.5)
            # patch.set_P(patch.P * 0.9)

        import matplotlib.pyplot as plt

        # xllim = 0.0

        # b = grid[0]
        # C = b[:, b.nj // 2, :]
        # fig, ax = plt.subplots()
        # ax.contourf(C.x, C.rt, C.wdist)
        # ax.axis("equal")
        # print(C.wdist.min(), C.wdist.max())
        # plt.show()
        # quit()

        config = ember.config.SolverConfig(
            n_step=249,
            n_step_avg=1,
            n_step_log=50,
            n_levels=3,
            cfl_min=0.05,
            debug=True,
            cfl_max=4.0,
            rtol=1e-6,
            inviscid=False,
            shear_work=True,
            full_mgrid=True,
            fac_mgrid=0.5,
            rf_inlet=4.0,
            rf_outlet=4.0,
            i_level_stop=1,
            xllim=xllim,
            sf4=0.01,
            sf2_adapt=1.0,
            const_smoothing=False,
        )

        try:
            self.convergence = ember.run.loop(grid, config)
        except SystemExit:
            pass

        b = grid[0]
        print(b.shape)

        C = b[-1, :, :]
        fig, ax = plt.subplots()
        ax.contourf(C.rt, C.r, C.Vx, levels=10)
        fig, ax = plt.subplots()
        ax.contourf(C.rt, C.r, C.Vt, levels=10)
        fig, ax = plt.subplots()
        ax.plot(C.P[:, 0], C.r[:, 0])
        ax.set_title("Pressure distribution ")
        plt.show()

        fig, ax = plt.subplots()
        C = b[:, b.nj // 2, :]
        iplot = 116
        kplot = 72 // 2
        ax.plot(C.x, C.rt, "k-", lw=0.5)
        ax.plot(C.x.T, C.rt.T, "k-", lw=0.5)
        ax.plot(C.x[iplot, kplot], C.rt[iplot, kplot], "r*")
        ax.axis("equal")

        fig, ax = plt.subplots()
        ax.contourf(C.x, C.rt, C.Ma_rel, levels=10)
        ax.axis("equal")

        fig, ax = plt.subplots()
        C.cfl_cell[-1, -1, :] = np.nan
        cm = ax.contourf(C.x, C.rt, C.cfl_cell[..., -1], levels=10)
        plt.colorbar(cm, ax=ax)
        ax.axis("equal")
        plt.show()

        plt.show()
