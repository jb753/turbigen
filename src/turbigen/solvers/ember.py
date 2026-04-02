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

        # b = gridwe[0]
        # C = b[:, b.nj // 2, :]
        # fig, ax = plt.subplots()
        # ax.contourf(C.x, C.rt, C.wdist)
        # ax.axis("equal")
        # print(C.wdist.min(), C.wdist.max())
        # plt.show()
        # quit()

        b = grid[0]

        config = ember.config.SolverConfig(
            n_step=5000,
            n_step_avg=1000,
            n_step_log=100,
            n_levels=3,
            cfl_min=0.2,
            cfl_max=2.0,
            rtol=1e-6,
            xllim=xllim,
            sf4=0.02,
            sf2_adapt=2.0,
            rf_inlet=2.0,
            rf_outlet=1.0,
            rf_mixing=0.1,
            inviscid=False,
            shear_work=True,
            full_mgrid=True,
            # restrict_mode="average",
            # v_cycle=True,
            fac_mgrid=0.5,
            # i_level_stop=1,
            # debug=True,
        )

        try:
            self.convergence = ember.run.loop(grid, config)
        except SystemExit:
            pass

        fname_out = workdir.parent / "soln.pkl"
        logger.info(f"Saving solution to {fname_out}")
        grid.write_emb(fname_out, compress=False)
