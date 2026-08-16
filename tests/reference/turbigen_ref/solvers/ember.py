import logging
import ember.run
import ember.patch
import ember.config
import ember.fortran
from pathlib import Path
from dataclasses import dataclass
from turbigen_ref.solvers.base import BaseSolver

logger = logging.getLogger("turbigen")
logging.getLogger("ember").parent = logger


@dataclass
class Ember(BaseSolver):
    """Settings with default values for Plot3D export."""

    _name = "ember"

    workdir: Path = None

    n_step: int = 2500
    n_step_avg: int = 500
    n_step_freeze: int = 0
    full_mgrid: bool = True
    i_level_stop: int = 0
    rf_mix: float = 0.05
    delta_filt: float = 1.0
    sf_cusp: float = 0.0
    gain_filt: float = 20.0
    fac_mgrid: float = 0.8
    mg_mode: str = "jameson"
    inviscid: bool = False
    restrict_avg: bool = False
    radial_equilibrium: bool = True
    cfl_bnd_max: float = 0.5
    cfl_bnd_min: float = 0.0
    cfl_min: float = 0.1
    sf2P: float = 1.0
    sf2T: float = 1.0
    n_levels: float = 3
    sf4: float = 0.01
    sf_mix: float = 0.1
    sf_outlet: float = 0.1
    deswirl: float = 0.0
    cfl_max: float = 4.0
    rtol: float = 1e-6

    def robust(self):
        """Change settings for a more stable simulation."""
        return self

    def restart(self):
        """Restart the simulation from a previous solution."""
        return self.replace(full_mgrid=False)

    def run(self, grid, machine, workdir):
        # xllim = grid.get_xllim()[0]
        # logger.info(f"xllim: {xllim}")

        if self.radial_equilibrium:
            for patch in grid.patches.outlet:
                patch.set_adjustment(K_dyn=0.0, radial_equilibrium=True, rf=0.01)

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

        # Write out a guess pickle
        guess_file = workdir.parent / "guess.pkl"
        logger.info(f"Writing guess solution to {guess_file}")
        grid.write_emb(guess_file, compress=False)

        config = ember.config.SolverConfig(
            n_step=self.n_step,
            n_step_avg=self.n_step_avg,
            n_step_freeze=self.n_step_freeze,
            n_step_log=50,
            n_levels=self.n_levels,
            cfl_max=self.cfl_max,
            cfl_bnd_min=self.cfl_bnd_min,
            cfl_bnd_max=self.cfl_bnd_max,
            cfl_min=self.cfl_min,
            rf_mix=self.rf_mix,
            # xllim=1e6,
            full_mgrid=self.full_mgrid,
            fac_restart=0.25,
            fac_mgrid=self.fac_mgrid,
            mg_mode=self.mg_mode,
            fac_mg_smooth=0.0,
            inviscid=self.inviscid,
            sf2P=self.sf2P,
            sf2T=self.sf2T,
            sf4=self.sf4,
            delta_filt=self.delta_filt,
            gain_filt=self.gain_filt,
            # rf_inlet_P=0.0,
            i_level_stop=self.i_level_stop,
            rf_inlet_P=0.05,
            restrict_avg=self.restrict_avg,
            rtol=self.rtol,
        )

        try:
            self.convergence = ember.run.loop(grid, config)

            # Write out convergence history
            hist_file = workdir.parent / "conv.cnv"
            logger.info(f"Writing convergence history to {hist_file}")
            self.convergence.write_cnv(hist_file)

            # Remove guess file if the simulation completed successfully
            if guess_file.exists():
                guess_file.unlink()

        except SystemExit:
            pass

        fname_out = workdir.parent / "soln.pkl"
        logger.info(f"Saving solution to {fname_out}")
        grid.write_emb(fname_out, compress=False)
