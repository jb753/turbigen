"""Save pressure field around the nose."""
import numpy as np
import os
import turbigen.util
import matplotlib.pyplot as plt
import scipy.interpolate

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

        # lev_Cp = np.concatenate(
        #     (
        #         np.linspace(-1.5,-0.2,5)[:-1],
        #         np.linspace(-0.2,0.1,15)
        #     )
        # )

        lev_Cp = np.linspace(-0.2, 0.1, 5)

        # Loop over span fractions
        for ispf, spf in enumerate(spfrow):

            # We want to plot along a general meridional surface
            # So brute force a mapping from x/r to meridional distance

            # Evaluate xr as a function of meridonal distance using machine geometry
            xr_row = machine.ann.xr_row(irow)
            m_ref = np.linspace(-0.2, 0.2, 5000)
            xr_ref = xr_row(spf, m_ref)

            # Calculate normalised meridional distance (angles are angles)
            dxr = np.diff(xr_ref, n=1, axis=1)
            dm = np.sqrt(np.sum(dxr**2.0, axis=0))
            rc = 0.5 * (xr_ref[1, 1:] + xr_ref[1, :-1])
            mp_ref = turbigen.util.cumsum0(dm / rc)
            assert (np.diff(mp_ref) > 0.0).all()

            def mp_from_xr(xr):
                func = scipy.interpolate.NearestNDInterpolator(xr_ref.T, mp_ref)
                xru = xr.reshape(2, -1)
                mpu = func(xru.T)
                return mpu.reshape(xr.shape[1:])

            cut = grid.cut_span(spf)

            fig, ax = plt.subplots()

            surf = grid.cut_blade_surfs()[irow][0].squeeze()
            jspf = grid.spf_index(spf)

            for b in cut:

                P = b.P

                if Po2 > Po1:
                    # Compressor
                    Cp = (P - Po1) / (Po1 - P1)
                else:
                    # Turbine
                    Cp = (P - Po1) / (Po1 - P2)

                mpb = mp_from_xr(b.xr)

                ax.contourf(mpb, b.t, Cp, lev_Cp, linewidth=0.1)
                # ax.contourf(mpb, b.t+b.pitch, Cp, lev_Cp,lw=0.1)

            mps = mp_from_xr(surf.xr)
            mstag = mps[surf.i_stag[jspf], jspf]
            tstag = surf.t[surf.i_stag[jspf], jspf] + surf.pitch
            ax.plot(mstag, tstag, "r+")
            dt = surf.pitch * 0.2
            ax.axis("equal")
            ax.set_ylim(tstag - dt, tstag + dt)
            ax.set_xlim(mstag - dt, mstag + dt)

            # ax.plot( mstag, tstag + surf.pitch, 'r+')

            # plt.show()

            plotname = os.path.join(postdir, f"nose_row_{irow}_spf_{spf}.pdf")
            plt.tight_layout()
            plt.savefig(plotname)
            plt.close()
