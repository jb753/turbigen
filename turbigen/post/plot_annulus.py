"""Save plots of annulus lines."""

import os
import turbigen.util
import matplotlib.pyplot as plt
import numpy as np

logger = turbigen.util.make_logger()


def post(
    _,
    machine,
    meanline,
    postdir,
    mnorm_traverse=[],
    show_axis=False,
    show_control_points=True,
    show_blades=True,
    write_raw=False,
    compare=False,
):
    """plot_annulus(mnorm_cut=[], show_axis=False, show_control_points=True, show_blades=True, write_raw=False, compare=None)
    Plot a meridional x-r view of the annulus lines of the turbomachine.

    Parameters
    ----------
    mnorm_cut: list
        Show cut planes at the given normalised meridional coordinates.
    show_axis: bool
        Add a dot-dash centreline at zero radius to show the axis of the machine.
    show_control_points: bool
        Show the control points at the inlet and exit of each row defining the shape.
    show_blades: bool
        Hatch bladed areas in the plot.
    write_raw: bool
        Save the hub and casing coordinates to an npz file.
    compare: str
        Path to an xrt data file with meridional coordinates to compare to current annulus.

    """  # noqa:E501
    logger.info("Plotting annulus lines")

    m = np.linspace(0.0, 1.0, 100)

    fig, ax = plt.subplots()
    ax.axis("off")
    # ax.set_xlabel("Axial Coordinate, $x$")
    # ax.set_ylabel("Radial Coordinate, $r$")
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
                Npts,
            )
            mcas = np.linspace(
                *ann.cas.mctrl[
                    isten,
                ],
                Npts,
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

    for tcut in mnorm_traverse:
        xrc = ann.get_cut_plane(tcut)[0]
        ax.plot(*xrc, "-", color="C0")

    xr_hub = ann.hub.xr(m)
    xr_cas = ann.cas.xr(m)
    ax.plot(*xr_hub, "k-")
    ax.plot(*xr_cas, "k-")
    if show_control_points:
        ax.plot(*ann.hub.xr(ann.hub.mctrl), "ro", fillstyle="none", ms=10)
        ax.plot(*ann.cas.xr(ann.cas.mctrl), "ro", fillstyle="none", ms=10)
    x = ann.hub.xr((0.0, 1.0))[0]
    if show_axis:
        ax.plot(x, np.zeros_like(x), "k-.")

    if compare:
        if compare_dat := compare[irow]:
            xrrt_all = turbigen.util.read_sections(compare_dat)
            for ispf, xrrt in enumerate(xrrt_all):
                x1c = xrrt[0]
                x2c = xrrt[1]
                ax.plot(x1c, x2c, ".", ms=1, color=f"C{ispf}")

    ax.axis("equal")
    ax.axis("off")
    ax.grid("off")
    plt.tight_layout(pad=0.1)
    pltname = os.path.join(postdir, "annulus.pdf")
    plt.savefig(pltname)

    if write_raw:
        rawname = os.path.join(postdir, "annulus_raw")
        np.savez_compressed(rawname, xr_hub=xr_hub, xr_cas=xr_cas)

    plt.close()
