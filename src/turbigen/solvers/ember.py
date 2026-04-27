import logging
import numpy as np
import ember.run
import ember.patch
import ember.config
import ember.fortran
from pathlib import Path
from dataclasses import dataclass
from turbigen.solvers.base import BaseSolver

logger = logging.getLogger("turbigen")
logging.getLogger("ember").parent = logger


@dataclass
class Ember(BaseSolver):
    """Settings with default values for Plot3D export."""

    _name = "ember"

    workdir: Path = None

    n_step: int = 2500
    n_step_avg: int = 500
    full_mgrid: bool = True

    def robust(self):
        """Change settings for a more stable simulation."""
        return self

    def restart(self):
        """Restart the simulation from a previous solution."""
        return self.replace(full_mgrid=False)

    def run(self, grid, machine, workdir):
        r_ref = grid.get_r_ref()
        Nb = np.array([row[0].Nb for row in grid.rows])
        pitch_ref = 2.0 * np.pi * r_ref / Nb
        logger.info(f"r_ref: {r_ref}")
        logger.info(f"Nb: {Nb}")
        logger.info(f"pitch_ref: {pitch_ref}")
        xllim = 0.03 * pitch_ref[0]
        logger.info(f"xllim: {xllim}")

        for patch in grid.patches.outlet:
            patch.set_adjustment(K_dyn=1.0, radial_equilibrium=True, rf=0.1)
        # patch.set_adjustment("dynamic_head", K=2.0, rf=1.0)

        # import matplotlib.pyplot as plt

        # xllim = 0.0

        # b = gridwe[0]
        # C = b[:, b.nj // 2, :]
        # fig, ax = plt.subplots()
        # ax.contourf(C.x, C.rt, C.wdist)
        # ax.axis("equal")
        # print(C.wdist.min(), C.wdist.max())
        # plt.show()
        # quit()

        config = ember.config.SolverConfig(
            n_step=self.n_step,
            n_step_avg=self.n_step_avg,
            n_step_log=100,
            n_levels=4,
            # cfl_max=1.0,
            cfl_min=0.1,
            # rf_mix=0.1,
            xllim=xllim,
            full_mgrid=self.full_mgrid,
            fac_restart=0.5,
            # fac_mgrid=0.5,
            # conservative_smoothing=True,
            conservative_smoothing=True,
            # inviscid=True,
            # fac_mgrid=0.0,
            gain_filt=20.0,
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
