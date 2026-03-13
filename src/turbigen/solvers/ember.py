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
            patch.set_P(patch.P * 0.9)

        print("beans")
        rotp = [p for p in grid[0].patches.rotating]
        for p in rotp:
            grid[0].patches.remove(p)
            print(f"removed rotating patch: {p}")

        # Omega = rotp[0].Omega
        # ilte = []
        # for p in grid[0].patches.periodic:
        #     if p.ien > 0 and p.ien < grid[0].ni - 1:
        #         ilte.append(p.ien)
        #     if p.ist > 0 and p.ist < grid[0].ni - 1:
        #         ilte.append(p.ist)
        # ile, ite = np.unique(ilte)
        # ile = int(ile)
        # ite = int(ite)
        # print(ile)
        # print(ite)
        # b = grid[0]
        # b.patches.append(ember.patch.RotatingPatch(i=(ile, ite), k=0))
        # b.patches.append(ember.patch.RotatingPatch(i=(ile, ite), k=-1))
        # # b.patches.append(ember.patch.RotatingPatch(j=0))
        # # b.patches.append(ember.patch.RotatingPatch(j=-1))

        import matplotlib.pyplot as plt

        # fig, ax = plt.subplots()
        # b = grid[0]

        # C = b[b.ni // 2, :, :]
        # cm = ax.contourf(C.z, C.y, C.wdist)
        # ax.axis("equal")
        # ax.set_title("Dwall")
        # plt.colorbar(cm, ax=ax)
        # plt.show()

        config = ember.config.SolverConfig(
            n_step=5000,
            n_step_avg=500,
            n_step_log=50,
            n_levels=3,
            cfl_min=0.1,
            cfl_max=4.0,
            rtol=1e-6,
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

        # fig, ax = plt.subplots()
        # b = grid[0]
        # C = b[:, b.nj // 2, :]
        # cm = ax.contourf(C.x, C.rt, C.wdist)
        # ax.axis("equal")
        # ax.set_title("Dwall")
        # plt.colorbar(cm, ax=ax)
        # plt.show()
        # quit()

        # fig, ax = plt.subplots()
        # b = grid[0]
        # C = b[:, b.nj // 2, b.nk // 2]
        # ax.plot(C.x, C.cfl_cell)
        # plt.show()
        # quit()

        fig, ax = plt.subplots()
        b = grid[0]
        C = b[:, b.nj // 2, :]
        cm = ax.contourf(C.x, C.rt, C.wdist)
        ax.axis("equal")
        ax.set_title("Dwall")
        plt.colorbar(cm, ax=ax)
        plt.show()
        quit()

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
