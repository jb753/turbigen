"""Save plots of pressure distributions."""

import numpy as np
import os
import turbigen.util
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt

logger = turbigen.util.make_logger()


def post(
    grid,
    machine,
    meanline,
    _,
    postdir,
    row_spf,
    write_raw=False,
    use_rot=False,
    lim=None,
    fix_stag=False,
    note=None,
    show_DF=None,
    compare=None,
    offset=0,
    target=None,
):
    lnst = ["-", "--"]

    raw_data = {}

    nrow = machine.Nrow
    if not len(row_spf) == nrow:
        raise Exception(f"row_spf should be of length nrow={nrow}, got {len(row_spf)}")

    # Loop over rows
    nsectplt = 0
    for irow, spfrow in enumerate(row_spf):
        if not spfrow:
            continue
        nsectplt += 1

        logger.info(f"Plotting Ma_isen, row {irow} at spf={spfrow}")

        # Extract reference pressure from mean-line
        iin = irow * 2
        iout = iin + 1
        s1 = meanline.s[iin]
        Po1, Po2 = meanline.Po_rel[
            (iin, iout),
        ]
        if use_rot:
            P1, P2 = meanline.P_rot[
                (iin, iout),
            ]
        else:
            P1, P2 = meanline.P[
                (iin, iout),
            ]

        # Get all blades in this row
        surfs = grid.cut_blade_surfs(offset)[irow]

        # Meridional curves for target span fractions
        ist = irow * 2 + 1
        ien = ist + 1
        m = np.linspace(ist, ien, 101)
        spfrow = np.array(spfrow)
        xr_spf = machine.ann.evaluate_xr(
            m.reshape(-1, 1), spfrow.reshape(1, -1)
        ).reshape(2, -1, len(spfrow))

        fig, ax = plt.subplots(layout="constrained")
        ax.set_xlabel(r"Normalised Surface Distance, $\zeta/\zeta_\mathrm{TE}$")
        ax.set_xlim((0.0, 1.0))

        # Loop over span fractions
        for ispf, spf in enumerate(spfrow):
            # Find the j-index corresponding to current span fraction on main blade
            # jspf = np.argmin(np.abs(surfs[0].spf[1, :, 0] - spf))

            # Loop over main/splitter
            for isurf, surf in enumerate(surfs):
                # Take the cut
                C = surf.meridional_slice(xr_spf[:, :, ispf])

                # Isentropic from inlet entropy to local static
                Cs = C.copy().set_P_s(C.P_rot, s1)

                # Calculate isentropic velocity
                hs = Cs.h
                ho = C.ho
                Vs = np.sqrt(2.0 * np.maximum(ho - hs, 0.0))
                Vs_TE = 0.5 * (Vs[0] + Vs[-1]).item()
                Vs_peak = Vs.max()

                Mas = Vs / C.a

                Mas_TE = 0.5 * (Mas[0] + Mas[-1]).item()
                # Mas /= Mas_TE

                DF = Vs_peak / Vs_TE - 1.0
                print(DF)

                # Extract pressure and non-dimensionalise
                ax.set_ylabel(r"Isentropic Mach Number, $\mathrm{Ma}$")

                # Extract surface distance and normalise
                zeta_stag = C.zeta_stag
                # Calculate maximum zeta only on main blade
                zeta_max = zeta_stag.max(axis=0)
                zeta_min = np.abs(zeta_stag.min(axis=0))
                zeta_norm = zeta_stag.copy()
                zeta_norm[zeta_norm < 0.0] /= zeta_min
                zeta_norm[zeta_norm > 0.0] /= zeta_max

                if isurf == 0:
                    ax.plot(
                        np.abs(zeta_norm),
                        Mas,
                        label=f"spf={spf}",
                        color=f"C{ispf}",
                        linestyle=lnst[isurf],
                        marker="",
                    )
                else:
                    ax.plot(
                        np.abs(zeta_norm), Mas, color=f"C{ispf}", linestyle=lnst[isurf]
                    )

                if not lim:
                    lim = ax.get_ylim()
                ax.set_ylim(lim)
                Ntick = 8
                dtick = np.round(np.ptp(lim) / (Ntick - 1), decimals=1)
                ax.yaxis.set_major_locator(ticker.MultipleLocator(dtick))

                # Store the raw data
                key = f"row_{irow}_spf_{spf}_blade_{isurf}"
                raw_data[key] = np.stack((zeta_stag, Mas))

        plotname = os.path.join(postdir, f"Ma_isen_{irow}.pdf")
        if nsectplt > 1:
            ax.legend()
        ax.grid(True)
        plt.savefig(plotname)
        plt.close()
