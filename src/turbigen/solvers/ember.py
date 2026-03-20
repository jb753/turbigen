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
        xllim = 0.03 * pitch_ref[0]
        print(f"xllim: {xllim}")

        for patch in grid.patches.outlet:
            patch.set_adjustment("dynamic_head", K=2.0, rf=1.0)

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
            n_step=4000,
            n_step_avg=1000,
            n_step_log=100,
            n_levels=3,
            cfl_min=0.2,
            cfl_max=4.0,
            rtol=1e-6,
            xllim=xllim,
            sf4=0.01,
            sf2_adapt=2.0,
            rf_inlet=4.0,
            rf_outlet=2.0,
            inviscid=False,
            shear_work=True,
            full_mgrid=True,
            fac_mgrid=0.5,
            rf_visc=0.2,
            # cfl_smooth_floor=0.2,
            # i_level_stop=2,
            # debug=True,
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

        fig, ax = plt.subplots()
        ax.axis("equal")
        for b in grid:
            C = b[:, b.nj // 2, :]
            ax.plot(C.x, C.rt, "k-", lw=0.5)
            ax.plot(C.x.T, C.rt.T, "k-", lw=0.5)

        fig, ax = plt.subplots()
        ax.axis("equal")
        C = grid.patches.outlet[0].get_cut().squeeze()
        ax.contourf(C.z, C.y, C.Vx)
        ax.contour(C.z, C.y, C.Vx, levels=(0,))

        # fig, ax = plt.subplots()
        # lev = np.arange(0.0, 4.0, 0.05)
        # ax.axis("equal")
        # for b in grid:
        #     C = b[:, 72, :]
        #     ax.contourf(
        #         C.x[:-1, :-1], C.rt[:-1, :-1], C.cfl_cell[:-1, :-1, 0], levels=lev
        #     )

        # fig, ax = plt.subplots()
        # ax.axis("equal")
        # ax.set_title("fx")
        # for b in grid:
        #     C = b[:, 72, :].squeeze()
        #     f_body = b.f_body[1, :, 72, :].squeeze()
        #     print(f_body.min(), f_body.max())
        #     ax.contourf(C.x[:-1, :-1].squeeze(), C.rt[:-1, :-1].squeeze(), f_body)

        # fig, ax = plt.subplots()
        # ax.axis("equal")
        # ax.set_title("fr")
        # for b in grid:
        #     C = b[:, 72, :].squeeze()
        #     f_body = b.f_body[2, :, 72, :].squeeze()
        #     print(f_body.min(), f_body.max())
        #     ax.contourf(C.x[:-1, :-1].squeeze(), C.rt[:-1, :-1].squeeze(), f_body)

        # fig, ax = plt.subplots()
        # ax.axis("equal")
        # ax.set_title("ft")
        # for b in grid:
        #     C = b[:, 72, :].squeeze()
        #     f_body = b.f_body[3, :, 72, :].squeeze()
        #     print(f_body.min(), f_body.max())
        #     ax.contourf(C.x[:-1, :-1].squeeze(), C.rt[:-1, :-1].squeeze(), f_body)

        # fig, ax = plt.subplots()
        # ax.axis("equal")
        # ax.set_title("fe")
        # for b in grid:
        #     C = b[:, 72, :].squeeze()
        #     f_body = b.f_body[4, :, 72, :].squeeze()
        #     print(f_body.min(), f_body.max())
        #     ax.contourf(C.x[:-1, :-1].squeeze(), C.rt[:-1, :-1].squeeze(), f_body)

        fig, ax = plt.subplots()
        lev = np.arange(0.0, 1.1, 0.1)
        ax.axis("equal")
        for b in grid:
            C = b[:, b.nj // 2, :]
            ax.contourf(C.x, C.rt, C.Ma_rel, levels=lev)

        # fig, ax = plt.subplots()
        # C.cfl_cell[-1, -1, :] = np.nan
        # cm = ax.contourf(C.x, C.rt, C.cfl_cell[..., -1], levels=10)
        # plt.colorbar(cm, ax=ax)
        # ax.axis("equal")

        plt.show()
