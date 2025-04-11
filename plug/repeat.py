import turbigen.iterators
import dataclasses
import numpy as np


@dataclasses.dataclass
class Repeat(turbigen.iterators.IteratorConfig):
    """Settings for repeating stage."""

    relaxation_factor: float = 0.5
    """Factor controlling size of changes."""

    To_frac: float = 0.5
    """Fraction of varation in To to pass upstream."""

    rtol: float = 0  # 0.001
    """Relative tolerance for convergence of Po and To."""

    atol: float = 0.01
    """Absolute tolerance for convergence of angles."""

    dAlpha_max: float = 0.0
    """Clip the variations in yaw."""

    dBeta_max: float = 0.0
    """Clip the variations in pitch."""

    dTo_max: float = 0.1
    """Clip the variations in To."""

    dPo_max: float = 0.1
    """Clip the variations in Po."""

    def check(self, config):
        pass

    def update(self, config) -> bool:
        """Pass the outlet profiles upstream."""

        log_data = {}

        # Cut out the outlet profiles
        C_out = config.grid.outlet_patches[0].get_cut()

        # Mix out to uniformity to get reference state
        Cm_out = C_out.mix_out()[0]

        # Calculate factors
        Cs = C_out.squeeze()
        fac_Po = Cs.Po.mean(axis=-1) / Cm_out.Po
        fac_To = Cs.To.mean(axis=-1) / Cm_out.To
        spf = C_out.spf.mean(axis=-1).squeeze()
        dAlpha = Cs.Alpha.mean(axis=-1) - Cm_out.Alpha
        dBeta = Cs.Beta.mean(axis=-1) - Cm_out.Beta

        # Clip angle swings
        dAlpha = np.clip(dAlpha, -self.dAlpha_max, self.dAlpha_max)
        dBeta = np.clip(dBeta, -self.dBeta_max, self.dBeta_max)

        # Scale the To factor
        dTo = fac_To - 1.0
        dTo = np.clip(dTo * self.To_frac, -self.dTo_max, self.dTo_max)
        fac_To = dTo + 1.0

        # Clip the Po
        dPo = fac_Po - 1.0
        dPo = np.clip(dPo, -self.dPo_max, self.dPo_max)
        fac_Po = dPo + 1.0

        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots(layout="constrained")
        # ax.plot(fac_Po, spf, label="Po")
        # ax.plot(fac_To, spf, label="To")
        # ax.legend()
        # fig, ax = plt.subplots(layout="constrained")
        # ax.plot(dAlpha, spf, label="dAlpha")
        # ax.plot(dBeta, spf, label="dBeta")
        # ax.legend()
        # plt.show()
        # # quit()

        inlet = config.inlet

        # No previous inlet, initialise
        if inlet.spf is None:
            inlet.spf = spf
            inlet.fac_Po = fac_Po
            inlet.fac_To = fac_To
            inlet.dAlpha = dAlpha
            inlet.dBeta = dBeta
            err = np.max(np.abs(fac_Po - 1.0))
        # Compare with the previous inlet
        else:
            fac_Po_old = np.interp(
                spf,
                inlet.spf,
                inlet.fac_Po,
            )
            fac_To_old = np.interp(
                spf,
                inlet.spf,
                inlet.fac_To,
            )
            dAlpha_old = np.interp(
                spf,
                inlet.spf,
                inlet.dAlpha,
            )
            dBeta_old = np.interp(
                spf,
                inlet.spf,
                inlet.dBeta,
            )
            err = np.max(np.abs(fac_Po - fac_Po_old))
            inlet.spf = spf
            rf = self.relaxation_factor
            rf1 = 1.0 - rf
            inlet.fac_Po = rf * fac_Po + rf1 * fac_Po_old
            inlet.fac_To = rf * fac_To + rf1 * fac_To_old
            inlet.dAlpha = rf * dAlpha + rf1 * dAlpha_old
            inlet.dBeta = rf * dBeta + rf1 * dBeta_old

        return err < self.rtol, {"Rep_dPo": err}
