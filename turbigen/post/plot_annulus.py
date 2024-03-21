"""Save plots of annulus lines."""
import os
import turbigen.util
import matplotlib.pyplot as plt
import numpy as np

logger = turbigen.util.make_logger()


def post(
    grid,
    machine,
    meanline,
    postdir,
    show_axis=False,
    show_control_points=True,
    show_blades=True,
):

    logger.info("Plotting annulus lines")

    m = np.linspace(0.0, 1.0, 100)

    fig, ax = plt.subplots()
    ax.set_xlabel("Axial Coordinate, $x$")
    ax.set_ylabel("Radial Coordinate, $r$")
    plt.tight_layout(pad=0.1)
    ann = machine.ann

    if show_blades:
        grey = np.ones((3,)) * 0.4
        Npts = 100
        for irow in range(machine.Nrow):
            isten = (irow * 2 + 1, irow * 2 + 2)
            mhub = np.linspace(
                *ann.hub.mctrl[
                    isten,
                ],
                Npts
            )
            mcas = np.linspace(
                *ann.cas.mctrl[
                    isten,
                ],
                Npts
            )
            xrhub = ann.hub.xr(mhub)
            xrcas = ann.cas.xr(mcas)
            spf = np.linspace(0.0, 1.0, Npts).reshape(1, -1)
            spf1 = 1.0 - spf

            xr_LE = spf * xrcas[:, (0,)] + spf1 * xrhub[:, (0,)]
            xr_TE = spf * xrcas[:, (-1,)] + spf1 * xrhub[:, (-1,)]
            ax.plot(*xr_LE, "-", color=grey)
            ax.plot(*xr_TE, "-", color=grey)

            xr_d1 = spf * xrcas + spf1 * xrhub
            xr_d2 = spf1 * xrcas + spf * xrhub
            ax.plot(*xr_d1, "-", color=grey, solid_capstyle="butt")
            ax.plot(*xr_d2, "-", color=grey, solid_capstyle="butt")

    ax.plot(*ann.hub.xr(m), "k-")
    ax.plot(*ann.cas.xr(m), "k-")
    if show_control_points:
        ax.plot(*ann.hub.xr(ann.hub.mctrl), "ro", fillstyle="none", ms=10)
        ax.plot(*ann.cas.xr(ann.cas.mctrl), "ro", fillstyle="none", ms=10)
    x = ann.hub.xr((0.0, 1.0))[0]
    if show_axis:
        ax.plot(x, np.zeros_like(x), "k-.")

    ax.axis("equal")
    ax.axis("off")
    ax.grid("off")
    pltname = os.path.join(postdir, "annulus.pdf")
    plt.savefig(pltname)
