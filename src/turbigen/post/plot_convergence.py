"""Save plots of convergence history."""

import os
import turbigen.util
import matplotlib.pyplot as plt
import numpy as np

logger = turbigen.util.make_logger()


def post(
    _,
    machine,
    meanline,
    conv,
    postdir,
    write_raw=False,
    rtol_loss=0.01,
    dn_smooth=0,
):
    """plot_convergence(write_raw=False)
    Save plots of convergence history of the calculation.

    Parameters
    ----------
    write_raw: bool
        Save data to a csv file.
    """  # noqa:E501

    if conv:
        logger.info("Plotting convergence...")
    else:
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

    if dn_smooth:
        conv.resid = turbigen.util.moving_average_1d(conv.resid, dn_smooth)
        dCWx = turbigen.util.moving_average_1d(dCWx, dn_smooth)
        dYs = turbigen.util.moving_average_1d(dYs, dn_smooth)

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

    # Write out
    pltname = os.path.join(postdir, "convergence.pdf")
    plt.savefig(pltname)

    if write_raw:
        rawname = os.path.join(postdir, "convergence_raw")
        np.savetxt(rawname, conv.raw_data())
