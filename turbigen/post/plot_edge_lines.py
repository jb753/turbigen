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
    irows,
    compare,
):
    """plot_edge_lines(irows, compare=None)
    Plot views of the leading and trailing edges.
    """

    # Loop over rows
    for irow in irows:

        logger.info(f"Plotting LE/TE row={irow}")

        # Gather the data
        mE = np.array([0.0, 1.0])
        Npts = 100
        spf = np.linspace(0.0, 1.0, Npts)
        xrt_edge = np.full((2, 3, Npts), np.nan)
        for jspf in range(Npts):
            xrt_edge[:, :, jspf] = (
                machine.bld[irow].evaluate_section(spf[jspf], m=mE)[0].T
            )

        rrt_edge = np.stack(
            (
                xrt_edge[:, 0, :],
                xrt_edge[:, 1, :] * xrt_edge[:, 2, :] + 0.02,
            ),
            axis=1,
        )

        fig, ax = plt.subplots()
        ax.plot(*rrt_edge[0], label="LE", color="C0")
        ax.plot(*rrt_edge[1], label="TE", color="C1")
        ax.axis("equal")

        if compare:
            if compare_dat := compare[irow]:

                # Loop over sections in the file
                for xrrt in turbigen.util.read_sections(compare_dat):

                    # Get min and max rad
                    imin = np.argmin(xrrt[1])
                    imax = np.argmax(xrrt[1])
                    ax.plot(*xrrt[(0, 2), imin], "x", color="C0")
                    ax.plot(*xrrt[(0, 2), imax], "x", color="C1")

                    # ax.plot(x1c, x2c + xoff, "x", color="k", ms=2)
                    # ax.plot(x1c, x2c + xoff, "x", color=f"C{ispf}", ms=2)

            # dt = surf.pitch * 0.2
            # ax.set_ylim(tstag - dt, tstag + dt)
            # ax.set_xlim(mstag - dt, mstag + dt)

            # ax.plot(mpLE, xrtLE[2], "bx")

            # ax.plot(mpcam, xrtcam[2], "m-")

        ax.legend()
        ax.set_aspect("equal", adjustable="box")
        plt.show()
        # ax.axis("off")

        plotname = os.path.join(postdir, f"section_row_{irow}.pdf")
        plt.tight_layout(pad=0)
        plt.savefig(plotname)
        plt.close()
