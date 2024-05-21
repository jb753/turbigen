"""Save pressure field around the nose."""
import os
import turbigen.util
import matplotlib.pyplot as plt

logger = turbigen.util.make_logger()


def post(
    grid,
    machine,
    meanline,
    postdir,
    row_spf,
    coord_sys="mpt",
    compare=None,
    K_offset=0.0,
):

    # Loop over rows
    for irow, spfrow in enumerate(row_spf):

        if not spfrow:
            continue

        logger.info(f"Plotting section row={irow} at spf={spfrow}")

        fig, ax = plt.subplots()

        # Loop over span fractions
        for ispf, spf in enumerate(spfrow):

            jspf = grid.spf_index(spf)

            surf = grid.cut_blade_surfs()[irow][0][:, jspf, :].squeeze()

            mlim = (-0.1, 1.1)
            mp_from_xr, spf_actual = turbigen.util.get_mp_from_xr(
                grid, machine, irow, spf, mlim
            )

            mps = mp_from_xr(surf.xr)

            if coord_sys == "mpt":
                x1 = mps
                x2 = surf.t
            elif coord_sys == "xrt":
                x1 = surf.x
                x2 = surf.rt
            elif coord_sys == "yz":
                x1 = -surf.y
                x2 = surf.z

            xoff = K_offset * (spf - 0.5) * x2.ptp()

            ax.plot(x1, x2 + xoff, "-", label=f"spf={spf}")

            if compare:
                if compare_dat := compare[irow]:
                    xrrt = turbigen.util.read_sections(compare_dat)[ispf]
                    ax.plot(xrrt[0], xrrt[2] + xoff, ".", color=f"C{ispf}")

            # dt = surf.pitch * 0.2
            # ax.set_ylim(tstag - dt, tstag + dt)
            # ax.set_xlim(mstag - dt, mstag + dt)

            # ax.plot(mpLE, xrtLE[2], "bx")

            # ax.plot(mpcam, xrtcam[2], "m-")

        ax.legend()
        ax.set_aspect("equal", adjustable="box")
        # ax.axis("off")

        plotname = os.path.join(postdir, f"section_row_{irow}.pdf")
        plt.tight_layout(pad=0)
        plt.savefig(plotname)
        plt.close()
