"""Generic post-processor class."""

from abc import ABC, abstractmethod
import dataclasses
import numpy as np
from turbigen import util
import matplotlib.pyplot as plt

logger = util.make_logger()

LABELS = {
    "Mas": r"Isentropic Mach Number, $\mathit{Ma}_s$",
    "Ys": "Entropy Loss Coefficient, $Y_s$",
    "Ma_rel": r"Relative Mach Number, $\mathit{Ma}^\mathrm{rel}$",
}


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
            np.argmax(np.abs(dYs_reversed) > self.rtol_loss * 100.0)
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


@dataclasses.dataclass
class Metadata(BasePost):
    def post(self, config, pdf):
        """Make a slide with some text metadata."""

        _, ax = plt.subplots(layout="constrained")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        left = 0.05
        ax.set_title("Metadata:")
        ax.text(left, 0.95, f"workdir={str(config.workdir)}")
        pdf.savefig()
        plt.close()


def calculate_nondim(C, ml, vname):
    """Calculate a non-dimensional varaiable over a cut.

    Parameters
    ----------
    C : FlowField object
        The cut to evaluate.
    ml : MeanLine object
        A single-row meanline object used to provide reference values.
    vname: str
        String indicating which variable to calculate.

    Returns
    -------
    y : ndarray
        The non-dimensional variable.
    ylabel : str
        Label for the y-axis.

    """

    # Isentropic from inlet entropy to local static
    Cs = C.copy().set_P_s(C.P, ml.s[0])
    hs = Cs.h
    ho = C.ho_rel

    # Ensure ho > hs
    dh = ho - hs
    hs += np.min(dh)

    # Evaluate velocity and Mach
    Vs = np.sqrt(2.0 * np.maximum(ho - hs, 0.0))
    Mas = Vs / C.a

    is_compressor = ml.P[1] > ml.P[0]

    if is_compressor:
        Ys = ml.T[1] * (C.s - ml.s[0]) / ml.halfVsq_rel[0]
    else:
        Ys = ml.T[1] * (C.s - ml.s[0]) / ml.halfVsq_rel[1]

    if vname == "Mas":
        return Mas
    elif vname == "Ys":
        return Ys
    elif vname == "Ma_rel":
        return C.Ma_rel
    else:
        raise ValueError(f"Unknown variable {vname} requested.")


@dataclasses.dataclass
class SurfaceDistribution(BasePost):
    variable: str = "Mas"
    """Which variable to plot."""

    spf: dict = dataclasses.field(default_factory=lambda: ({}))
    """Mapping of row index to span fraction(s) to plot."""

    offset: int = 0
    """How many points away from the wall."""

    def post(self, config, pdf):
        """Plot distribution of a quantity around blade surface."""

        # Default to plotting on the designed sections
        if not (spf := self.spf):
            spf = {irow: config.blades[irow][0].spf for irow in range(config.nrow)}

        # Loop over rows
        for irow, spfrow in spf.items():
            if not spfrow:
                continue

            # Setup figure
            _, ax = plt.subplots(layout="constrained")
            ax.set_title(f"Row {irow}")
            ax.set_xlabel(r"Surface Distance, $\zeta/\zeta_\mathrm{TE}$")
            ax.set_xlim((0.0, 1.0))

            label = LABELS.get(self.variable, self.variable)
            ax.set_ylabel(label)

            # Cut the entire blade
            C = config.grid.cut_blade_surfs(self.offset)[irow][0]

            # Loop over span fractions
            for spfi in spfrow:
                # Slice at required span fractions
                xrc = config.annulus.get_span_curve(spfi)
                Ci = C.meridional_slice(xrc)

                # Get the variable
                y = calculate_nondim(
                    Ci, config.mean_line.actual.get_row(irow), self.variable
                )

                # Extract surface distance and normalise
                zeta_stag = Ci.zeta_stag
                # Shift zeta=0 to minimum Mas
                # if self.variable == "Mas":
                # zeta_stag -= zeta_stag[np.argmin(y)]
                # Calculate maximum zeta only on main blade
                zeta_max = zeta_stag.max(axis=0)
                zeta_min = np.abs(zeta_stag.min(axis=0))
                zeta_norm = zeta_stag.copy()
                zeta_norm[zeta_norm < 0.0] /= zeta_min
                zeta_norm[zeta_norm > 0.0] /= zeta_max

                ax.plot(
                    np.abs(zeta_norm),
                    y,
                    label=f"spf={spfi}",
                    linestyle="-",
                    marker="",
                )

            # Finish this row
            pdf.savefig()
            plt.close()
        #
        # _, ax = plt.subplots(layout="constrained")
        # ax.set_xlim(0, 1)
        # ax.set_ylim(0, 1)
        # ax.axis("off")
        # left = 0.05
        # ax.set_title("Metadata:")
        # ax.text(left, 0.95, f"workdir={str(config.workdir)}")
        # pdf.savefig()
        # plt.close()
