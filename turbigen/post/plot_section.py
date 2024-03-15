"""Save pressure field around the nose."""
import os
import turbigen.util
import matplotlib.pyplot as plt

logger = turbigen.util.make_logger()


def post(grid, machine, meanline, postdir, row_spf):

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

            ax.plot(mps, surf.t, "-", label=f"spf={spf}")

            # dt = surf.pitch * 0.2
            # ax.set_ylim(tstag - dt, tstag + dt)
            # ax.set_xlim(mstag - dt, mstag + dt)

            # ax.plot(mpLE, xrtLE[2], "bx")

            # ax.plot(mpcam, xrtcam[2], "m-")

        ax.legend()
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")

        plotname = os.path.join(postdir, f"section_row_{irow}.pdf")
        plt.tight_layout(pad=0)
        plt.savefig(plotname)
        plt.close()
