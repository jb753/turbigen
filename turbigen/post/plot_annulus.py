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
    __,
    postdir,
    mnorm_traverse=[],
    show_axis=False,
    show_blades=True,
    write_raw=False,
    compare=False,
):
    """plot_annulus(mnorm_cut=[], show_axis=False, show_blades=True, write_raw=False, compare=None)
    Plot a meridional x-r view of the annulus lines of the turbomachine.

    Parameters
    ----------
    mnorm_cut: list
        Show cut planes at the given normalised meridional coordinates.
    show_axis: bool
        Add a dot-dash centreline at zero radius to show the axis of the machine.
    show_blades: bool
        Hatch bladed areas in the plot.
    write_raw: bool
        Save the hub and casing coordinates to an npz file.
    compare: str
        Path to an xrt data file with meridional coordinates to compare to current annulus.

    """  # noqa:E501
    logger.info("Plotting annulus lines")

    fig, ax = plt.subplots()
    ax.axis("off")
    # ax.set_xlabel("Axial Coordinate, $x$")
    # ax.set_ylabel("Radial Coordinate, $r$")
    ann = machine.ann
    bld = machine.bld

    # if show_blades:
    #     grey = np.ones((3,)) * 0.4
    #     Npts = 100
    #     spf = np.linspace(0.0, 1.0, Npts)
    #     mE = np.array([0.0, 1.0])
    #     for irow in range(machine.Nrow):
    #         # Plot LE and TE lines
    #         xrt_LE = np.full((3, Npts), np.nan)
    #         xrt_TE = np.full((3, Npts), np.nan)
    #         if bld[irow]:
    #             for jspf in range(Npts):
    #                 xrt_LE[:, jspf], xrt_TE[:, jspf] = (
    #                     bld[irow].evaluate_section(spf[jspf], m=mE)[0].T
    #                 )
    #             ax.plot(*xrt_LE[:2], "-", color=grey)
    #             ax.plot(*xrt_TE[:2], "-", color=grey)
    #
    for tcut in mnorm_traverse:
        xrc = ann.get_cut_plane(tcut)[0]
        ax.plot(*xrc, "-", color="C0")

    xr_hub, xr_cas = ann.get_coords().transpose(0, 2, 1)
    ax.plot(*xr_hub, "k-")
    ax.plot(*xr_cas, "k-")
    if show_axis:
        ax.plot(xr_hub[0, (0, -1)], np.zeros((2,)), "k-.")

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
