"""Save plots of pressure distributions."""

import numpy as np
import os
import turbigen.util
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt

logger = turbigen.util.make_logger()


def get_var(C, ml, vname):
    # Isentropic from inlet entropy to local static
    Cs = C.copy().set_P_s(C.P, ml.s[0])
    hs = Cs.h
    ho = C.ho
    # Ensure ho > hs
    dh = ho - hs
    hs += np.min(dh)
    Vs = np.sqrt(2.0 * np.maximum(ho - hs, 0.0))
    Mas = Vs / C.a

    Vs_TE = 0.5 * (Vs[0] + Vs[-1]).item()
    Mas_TE = 0.5 * (Mas[0] + Mas[-1]).item()

    if vname == "Mas":
        y = Mas
        ylabel = "Isentropic Mach Number, $\mathit{Ma}_s$"
        ylim = {}

    elif vname == "Masf":
        y = Mas / Mas_TE
        ylabel = r"Isentropic Mach, $\mathit{Ma}_s/\mathit{Ma}_{s,\mathrm{TE}}$"
        ylim = {"bottom": 0.0}

    elif vname == "Vs":
        y = Vs / Vs_TE
        ylabel = "Isentropic Velocity, $V_s/V_{s,\mathrm{TE}}$"
        ylim = {}

    return y, ylabel, ylim


def post(
    grid,
    machine,
    meanline,
    _,
    postdir,
    row_spf,
    var="Cp",
    lim=None,
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

        logger.info(f"Plotting blade surface {var}, row {irow} at spf={spfrow}")

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
        ax.set_xlabel(r"Surface Distance, $\zeta/\zeta_\mathrm{TE}$")
        ax.set_xlim((0.0, 1.0))
        ax.set_ylim((0.0, 1.6))
        ax.set_xticks((0.0, 0.5, 1.0))
        ax.set_yticks(np.arange(0.0, 2.0, 0.4))

        # Loop over span fractions
        for ispf, spf in enumerate(spfrow):
            # Find the j-index corresponding to current span fraction on main blade
            # jspf = np.argmin(np.abs(surfs[0].spf[1, :, 0] - spf))

            # Loop over main/splitter
            for isurf, surf in enumerate(surfs):
                # Take the cut
                C = surf.meridional_slice(xr_spf[:, :, ispf])

                y, ylabel, ylim = get_var(C, meanline.get_row(irow), var)

                # Extract pressure and non-dimensionalise
                ax.set_ylabel(ylabel)
                # if ylim:
                # ax.set_ylim(**ylim)

                # Extract surface distance and normalise
                zeta_stag = C.zeta_stag
                # Calculate maximum zeta only on main blade
                zeta_max = zeta_stag.max(axis=0)
                zeta_min = np.abs(zeta_stag.min(axis=0))
                zeta_norm = zeta_stag.copy()
                zeta_norm[zeta_norm < 0.0] /= zeta_min
                zeta_norm[zeta_norm > 0.0] /= zeta_max

                if len(spfrow) > 1:
                    col = f"C{ispf}"
                else:
                    col = "k"

                if isurf == 0:
                    ax.plot(
                        np.abs(zeta_norm),
                        y,
                        label=f"spf={spf}",
                        color=col,
                        linestyle=lnst[isurf],
                        marker="",
                    )
                else:
                    ax.plot(
                        np.abs(zeta_norm), y, color=f"C{ispf}", linestyle=lnst[isurf]
                    )

                if not lim:
                    lim = ax.get_ylim()
                ax.set_ylim(lim)

                # Ntick = 8
                # dtick = np.round(np.ptp(lim) / (Ntick - 1), decimals=1)
                # ax.yaxis.set_major_locator(ticker.MultipleLocator(dtick))

        if target:
            ax.plot(*target, "ko", label="Target", markersize=6)

        plotname = os.path.join(postdir, f"blade_surf_{var}_{irow}.pdf")
        if nsectplt > 1:
            ax.legend()
        ax.grid(False)
        plt.savefig(plotname)
        plt.close()
