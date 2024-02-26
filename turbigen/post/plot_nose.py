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


        # Loop over span fractions
        for ispf, spf in enumerate(spfrow):

            # We intend to cut along the nearest j-line to target spf,
            # which may not coincide exactly.

            # Start by choosing a j-index to plot along
            jspf = grid.spf_index(spf)

            xr_row = machine.ann.xr_row(irow)

            # Get spf along LE
            xr_LE = xr_row(np.linspace(0.,1.,200),np.atleast_1d(0.))

            surf = grid.cut_blade_surfs()[irow][0].squeeze()
            spf_blade = surf.spf[:,jspf]
            spf_actual = spf_blade[surf.i_stag[jspf]]


            # We want to plot along a general meridional surface
            # So brute force a mapping from x/r to meridional distance

            # Evaluate xr as a function of meridonal distance using machine geometry
            m_ref = np.linspace(-0.2, 0.2, 5000)
            xr_ref = xr_row(spf_actual, m_ref)

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


            # fig3, ax3 = plt.subplots()
            # ax3.axis('equal')
            Pall = np.concatenate([b.P.reshape(-1) for b in cut])
            tall = np.concatenate([b.t.reshape(-1) for b in cut])
            mpall = np.concatenate([mp_from_xr(b.xr).reshape(-1) for b in cut])
            Pred = Pall[np.abs(mpall)<0.18]
            tred = tall[np.abs(mpall)<0.18]

            if Po2 > Po1:
                # Compressor
                Cpall = (Pred - Po1) / (Po1 - P1)
            else:
                # Turbine
                Cpall = (Pred - Po1) / (Po1 - P2)


            Cpmin = turbigen.util.qinv(Cpall, 0.001)
            Cpmax = turbigen.util.qinv(Cpall, 0.999)
            lev_Cp = np.linspace(Cpmin, Cpmax, 20)

            for b in cut:

                P = b.P

                if Po2 > Po1:
                    # Compressor
                    Cp = (P - Po1) / (Po1 - P1)
                else:
                    # Turbine
                    Cp = (P - Po1) / (Po1 - P2)

                mpb = mp_from_xr(b.xr)

                # ax3.plot(*b.xr,'.')

                #    raise Exception('Failed to unwrap the streamsurface')
                if mpb.ptp()< surf.pitch*0.01:
                    continue

                ax.contourf(mpb, b.t, Cp, lev_Cp)
                # ax.contourf(mpb, b.t+b.pitch, Cp, lev_Cp,lw=0.1)

            # for b in grid.cut_span(0.):
            #     ax3.plot(*b.xr,'b^')

            # for b in grid.cut_span(1.):
            #     ax3.plot(*b.xr,'b^')

            # ax3.plot(*xr_ref,'k-')
            # ax3.plot(*xr_hub,'b-')
            # ax3.plot(*xr_cas,'b-')
            # ax3.plot(*xr_LE,'b-')
            # plt.show()

            mps = mp_from_xr(surf.xr)
            mstag = mps[surf.i_stag[jspf], jspf]
            tstag = surf.t[surf.i_stag[jspf], jspf]
            if grid.is_hmesh:
                tstag += surf.pitch
            ax.plot(mstag, tstag, "r+")
            dt = surf.pitch * 0.2
            xlim = (mp_ref.min(), mp_ref.max())

            ax.set_ylim(tstag - dt, tstag + dt)
            ax.set_xlim(mstag - dt, mstag + dt)

            xrtLE = machine.bld[irow].get_LE_cent(spf_actual)
            mpLE = mp_from_xr(xrtLE[:2])
            ax.plot(mpLE, xrtLE[2], 'bx')
            # ax.set_xlim(xlim)
            # ax.set_ylim((tred.min(), tred.max()))

            ax.set_aspect('equal', adjustable='box')
            ax.axis('off')
            # ax.set_aspect("equal",adjustable='box')

            # ax.plot( mstag, tstag + surf.pitch, 'r+')

            # plt.show()

            plotname = os.path.join(postdir, f"nose_row_{irow}_spf_{spf}.pdf")
            plt.tight_layout(pad=0)
            plt.savefig(plotname)
            plt.close()
