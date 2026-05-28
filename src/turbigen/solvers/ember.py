import logging
import numpy as np
import ember.run
import ember.patch
import ember.config
import ember.fortran
import ember.body_force
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
    i_level_stop: int = 0
    rf_mix: float = 0.05
    delta_filt: float = 1.0
    sf_cusp: float = 0.0
    gain_filt: float = 20.0
    fac_mgrid: float = 0.8
    inviscid: bool = False
    v_cycle: bool = False
    restrict_avg: bool = False
    radial_equilibrium: bool = True
    cfl_bnd_max: float = 0.5
    cfl_bnd_min: float = 0.0
    cfl_min: float = 0.1
    debug: bool = False
    sf2: float = 1.0
    sf4: float = 0.01
    sf_mix: float = 0.1
    sf_outlet: float = 0.1
    deswirl: float = 0.0
    cfl_max: float = 4.0

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

        if self.radial_equilibrium:
            for patch in grid.patches.outlet:
                patch.set_adjustment(K_dyn=2.0, radial_equilibrium=True, rf=0.01)

        body_forces = ()
        if self.deswirl > 0.0:
            mmax = machine.ann.mmax
            m_ramp = np.linspace(mmax - 0.5, mmax, 73)
            spf = np.linspace(0.0, 1.0, 65)
            m_grid, spf_grid = np.meshgrid(m_ramp, spf, indexing="ij")
            xr_arr = machine.ann.evaluate_xr(m_grid, spf_grid)
            xr = np.stack([xr_arr[0].ravel(), xr_arr[1].ravel()], axis=-1)
            gain = (self.deswirl * (m_grid - (mmax - 0.5)) / 0.5).ravel()
            bf = ember.body_force.DeswirlBodyForce(grid, xr=xr, gain=gain, k_mult=0.5)
            body_forces = (bf,)
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
            n_step_log=50,
            n_levels=4,
            cfl_max=self.cfl_max,
            cfl_bnd_min=self.cfl_bnd_min,
            cfl_bnd_max=self.cfl_bnd_max,
            cfl_min=self.cfl_min,
            rf_mix=self.rf_mix,
            xllim=xllim,
            full_mgrid=self.full_mgrid,
            fac_restart=0.25,
            fac_mgrid=self.fac_mgrid,
            fac_mg_smooth=0.0,
            inviscid=self.inviscid,
            sf2P=self.sf2,
            sf4=self.sf4,
            vort_absolute=False,
            delta_filt=self.delta_filt,
            gain_filt=self.gain_filt,
            i_level_stop=self.i_level_stop,
            v_cycle=self.v_cycle,
            restrict_avg=self.restrict_avg,
            debug=self.debug,
            body_forces=body_forces,
            sf_cusp=self.sf_cusp,
        )

        try:
            self.convergence = ember.run.loop(grid, config)
        except SystemExit:
            pass

        fname_out = workdir.parent / "soln.pkl"
        logger.info(f"Saving solution to {fname_out}")
        grid.write_emb(fname_out, compress=False)
