"""Save pressure field around the nose."""

import numpy as np
import os
import turbigen.util
import matplotlib.pyplot as plt

logger = turbigen.util.make_logger()


def post(grid, machine, meanline, _, postdir, row_spf, fac_Rle=1.0):
    # Loop over rows
    for irow, spfrow in enumerate(row_spf):
        if not spfrow:
            continue

        logger.info(f"Plotting nose row={irow} at spf={spfrow}")

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
            # Get a reasonably fine meridional curve to define the visualisation surf
            xr_vis = machine.ann.get_span_curve(spf, n=101)

            # Take cuts on the vis surface
            surf = grid.cut_blade_surfs()[irow][0]
            surf = surf.meridional_slice(xr_vis)
            cut = grid.cut_span_unstructured(xr_vis)

            # fig, ax = plt.subplots()
            # ax.plot(*xr, "k-x")
            # # fig, ax = plt.subplots(2,1)
            # # ii = np.arange(0,xr.shape[1]-1)
            # # dx = np.diff(xr[0])
            # # dr = np.diff(xr[1])
            # # i0x = np.where(dx==0.)
            # # i0r = np.where(dr==0.)
            # # ax[0].plot(ii,dx, "k-x")
            # # ax[0].plot(ii[i0x],dx[i0x], "ro")
            # # ax[1].plot(ii,dr, "k-x")
            # # ax[1].plot(ii[i0r],dr[i0r], "ro")
            # # plt.show()
            # # quit()
            # fig, ax = plt.subplots()
            # ax.plot(*surf.squeeze().xr, "rx")
            # ax.plot(*xr_ref, "k-")
            # ax.plot(*xr_ref2, "m-x")
            # ax.plot(*SS.xr, "bo")
            # ax.axis("equal")
            # plt.show()

            # quit()

            # jspf = grid.spf_index(spf)

            # Now generate a mapping from xr to meridional distance
            # around the leading edge only
            mlim_le = irow * 2 + 1 + np.array([-0.2, 0.3])
            xr_ref = machine.ann.get_span_curve(spf, n=1999, mlim=mlim_le)
            mp_from_xr = machine.ann.get_mp_from_xr(xr_ref)

            # Find coordinates of the stagnation point in this mapping
            mps = mp_from_xr(surf.xr)
            mstag = mps[surf.i_stag].squeeze()
            tstag = surf.t[surf.i_stag].squeeze()
            xrstag = surf.xr[:, surf.i_stag].squeeze()

            Pall = np.concatenate([b.P.reshape(-1) for b in cut])
            mpall = np.concatenate([mp_from_xr(b.xr).reshape(-1) for b in cut])

            # Clip pressures outside the plot range
            mlim_plot = mp_from_xr(xr_ref[:, (1, -2)])
            Pred = Pall[mpall > mlim_plot[0]]
            Pred = Pall[mpall < mlim_plot[-1]]

            if Po2 > Po1:
                # Compressor
                Cpall = (Pred - Po1) / (Po1 - P1)
            else:
                # Turbine
                Cpall = (Pred - Po1) / (Po1 - P2)

            dCp = 0.1
            lev_Cp = turbigen.util.clipped_levels(Cpall, dCp)

            fig, ax2 = plt.subplots(1, 2)
            ax = ax2[0]
            # ax.plot(*surf.squeeze().xr, "rx")
            for b in cut:
                P = b.P

                if Po2 > Po1:
                    # Compressor
                    Cp = (P - Po1) / (Po1 - P1)
                else:
                    # Turbine
                    Cp = (P - Po1) / (Po1 - P2)

                mpb = mp_from_xr(b.xr)

                if np.ptp(mpb) < b.pitch * 0.01:
                    continue

                ax.contourf(mpb, b.t, Cp, lev_Cp)
                if grid.is_hmesh:
                    ax.contourf(mpb, b.t + b.pitch, Cp, lev_Cp)

            xrtLE = machine.bld[irow].get_LE_cent(spf, fac_Rle=fac_Rle)
            mpLE = mp_from_xr(xrtLE[:2])

            xrtcam = machine.bld[irow].get_camber_line(spf)
            mpcam = mp_from_xr(xrtcam[:2])

            ax2[1].plot(*xrstag, "r+")
            ax2[1].plot(*xrtLE[:2], "bx")
            ax2[1].plot(*xrtcam[:2], "m-")
            # ax2[1].plot(*xr_vis, "k-")
            ax2[1].plot(*surf.xr, "k-")

            # xrtLLE = machine.bld[irow].get_LE_cent(spf, fac_Rle=100.0)
            # mpLLE = mp_from_xr(xrtLLE[:2])
            # xrLEline = machine.ann.evaluate_xr(1.0, np.linspace(0.0, 1.0))

            # fig, ax = plt.subplots()
            # ax.plot(*xr_vis, "k-")
            # ax.plot(*xrtLLE[:2],'r^')
            # ax.plot(*xrLEline,'b-')
            # ax.plot(*surf.xr,'m-')
            # ax.plot(*surf_hub.xr,'m-')
            # ax.plot(*surf_cas.xr,'m-')
            # ax.axis('equal')
            # plt.show()
            # quit()

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
