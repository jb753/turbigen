"""Save pressure field around the nose."""
import numpy as np
import os
import turbigen.util
import matplotlib.pyplot as plt

logger = turbigen.util.make_logger()


def post(grid, machine, meanline, postdir, row_spf):

    # Loop over rows
    for irow, spfrow in enumerate(row_spf):

        if not spfrow:
            continue

        logger.info(f"Plotting row={irow} at spf={spfrow}")

        # Extract reference pressure from mean-line
        iin = irow * 2
        iout = iin + 1
        Po1, Po2 = meanline.Po_rel[
            (iin, iout),
        ]
        P1, P2 = meanline.P[
            (iin, iout),
        ]

        # Loop over span fractions
        for ispf, spf in enumerate(spfrow):

            jspf = grid.spf_index(spf)

            surf = grid.cut_blade_surfs()[irow][0].squeeze()

            mp_from_xr, spf_actual = turbigen.util.get_mp_from_xr(
                grid, machine, irow, spf, (-0.2, 0.2)
            )

            cut = grid.cut_span(spf)

            fig, ax = plt.subplots()

            Pall = np.concatenate([b.P.reshape(-1) for b in cut])
            mpall = np.concatenate([mp_from_xr(b.xr).reshape(-1) for b in cut])
            Pred = Pall[np.abs(mpall) < 0.18]

            if Po2 > Po1:
                # Compressor
                Cpall = (Pred - Po1) / (Po1 - P1)
            else:
                # Turbine
                Cpall = (Pred - Po1) / (Po1 - P2)

            dCp = 0.1
            lev_Cp = turbigen.util.clipped_levels(Cpall, dCp)

            for b in cut:

                P = b.P

                if Po2 > Po1:
                    # Compressor
                    Cp = (P - Po1) / (Po1 - P1)
                else:
                    # Turbine
                    Cp = (P - Po1) / (Po1 - P2)

                mpb = mp_from_xr(b.xr)

                if mpb.ptp() < b.pitch * 0.01:
                    continue

                ax.contourf(mpb, b.t, Cp, lev_Cp)
                if grid.is_hmesh:
                    ax.contourf(mpb, b.t + b.pitch, Cp, lev_Cp)

            mps = mp_from_xr(surf.xr)
            mstag = mps[surf.i_stag[jspf], jspf]
            tstag = surf.t[surf.i_stag[jspf], jspf]

            xrtLE = machine.bld[irow].get_LE_cent(spf_actual)
            mpLE = mp_from_xr(xrtLE[:2])

            xrtcam = machine.bld[irow].get_camber_line(spf_actual)
            mpcam = mp_from_xr(xrtcam[:2])

            if grid.is_hmesh:
                tstag += surf.pitch
                xrtLE[2] += surf.pitch
                xrtcam[2] += surf.pitch

            ax.plot(mstag, tstag, "r+")

            dt = surf.pitch * 0.2
            ax.set_ylim(tstag - dt, tstag + dt)
            ax.set_xlim(mstag - dt, mstag + dt)

            ax.plot(mpLE, xrtLE[2], "bx")

            ax.plot(mpcam, xrtcam[2], "m-")

            ax.set_aspect("equal", adjustable="box")
            ax.axis("off")

            plotname = os.path.join(postdir, f"nose_row_{irow}_spf_{spf}.pdf")
            plt.tight_layout(pad=0)
            plt.savefig(plotname)
            plt.close()
