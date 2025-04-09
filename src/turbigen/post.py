"""Generic post-processor class."""

from abc import ABC, abstractmethod
import dataclasses
import numpy as np
from turbigen import util
import matplotlib.pyplot as plt


@dataclasses.dataclass
class BasePost(ABC):
    """Base class for post-processing."""

    @abstractmethod
    def post(self, config, pdf):
        """Perform the post processing on a config object."""
        raise NotImplementedError()


@dataclasses.dataclass
class Convergence(BasePost):
    dn_smooth: int = 0
    """Smoothing window for the time series."""

    rtol_loss: float = 0.01

    """Smoothing window for the time series."""

    def post(self, config, pdf):
        """Make a plot of convergence history of the CFD run."""

        meanline = config.mean_line.nominal
        conv = config.solver.convergence

        if conv is None:
            logger.info("No simulation log returned, skipping convergence plot.")
            return

        # Choose type of machine
        if meanline.P[-1] > meanline.P[0]:
            # Is compressor, reference to inlet velocity
            Vref = meanline.V_rel[0]
        else:
            # Is turbine, reference to exit velocity
            Vref = meanline.V_rel[-1]
        dhref = 0.5 * Vref**2

        # Get non-dimensionals
        Texit = meanline.T[-1]
        state = conv.state
        Ys = (state.s[1] - state.s[0]) * Texit / dhref
        CWx = (state.h[1] - state.h[0]) / dhref

        # Normalise work and loss as percent
        # changes with respect to final value
        dYs = (Ys / Ys[-1] - 1.0) * 100.0
        if meanline.U.any():
            dCWx = (CWx / CWx[-1] - 1.0) * 100.0
        else:
            # Fall back to absolute in a cascade
            dCWx = CWx * 100.0
        ylim = np.array([-10.0, 10.0])
        ytick = [-8, -4, -2, -1, 0, 1, 2, 4, 8]

        if self.dn_smooth:
            conv.resid = util.moving_average_1d(conv.resid, self.dn_smooth)
            dCWx = util.moving_average_1d(dCWx, self.dn_smooth)
            dYs = util.moving_average_1d(dYs, self.dn_smooth)

        dYs_reversed = np.flip(dYs)
        istep_conv = np.flip(conv.istep)[
            np.argmax(np.abs(dYs_reversed) > rtol_loss * 100.0)
        ]

        # Do the plotting
        _, ax = plt.subplots(1, 3, layout="constrained")
        ax[0].plot(conv.istep, np.log10(conv.resid), marker="")
        ax[0].set_title("log(Residual)")
        ax[1].plot(conv.istep, dCWx, marker="")
        ax[1].set_title("dWork/percent")
        ax[1].set_ylim(ylim)
        ax[1].set_yticks(ytick)
        ax[2].plot(conv.istep, dYs, marker="")
        ax[2].set_ylim(2 * ylim)
        ax[2].set_yticks(ytick)
        ax[2].set_title("dLoss/percent")

        ax[0].annotate(
            f"istep_conv={istep_conv}",
            xy=(1.0, 1.0),
            xytext=(-5.0, -5.0),
            xycoords="axes fraction",
            textcoords="offset points",
            ha="right",
            va="top",
            backgroundcolor="w",
            color="C1",
        )
        ax[0].annotate(
            f"istep_avg={conv.istep_avg}",
            xy=(1.0, 1.0),
            xytext=(-5.0, -25.0),
            xycoords="axes fraction",
            textcoords="offset points",
            ha="right",
            va="top",
            backgroundcolor="w",
            color="C2",
        )

        for axi in ax:
            axi.set_xlabel("nstep")
            axi.set_xticks(())
            distep = conv.istep[1] - conv.istep[0]
            axi.set_xlim(conv.istep[0], conv.istep[-1] + distep)
            axi.axvline(conv.istep_avg, color="C2", linestyle="--")
            axi.axvline(istep_conv, color="C1", linestyle=":")

        pdf.savefig()
        plt.close()
