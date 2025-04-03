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
    """plot_pressure_distributions(row_spf, write_raw=False, use_rot=False, lim=None)
    Plot static pressure on blade surface as a function of chordwise distance.

    The horizontal axis is surface distance away from the stagnation point.
    The vertical axis is a static pressure coefficient defined using the mean-line inlet dynamic head in compressors,
    and the isentropic exit dynamic head in turbines.

    Parameters
    ----------
    row_spf: list
        For each row of the machine,  a nested list of span fractions to plot at. For example, in a three-row machine,
        to plot the first row at mid-span, the second row at three locations, and omit the third row, use
        `[[0.5,], [0.1, 0.5, 0.9,], []]`.
    write_raw: bool
        Save the raw pressure coefficient and surface distance data to an npz file.
    use_rot: bool
        Plot using the rotary static pressure to take out centrifugal effects.
    lim: (2,) iterable
        Vertical limits, set automatically if omitted.

    """

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

        logger.info(f"Plotting pressure distributions, row {irow} at spf={spfrow}")

        # Extract reference pressure from mean-line
        iin = irow * 2
        iout = iin + 1
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
                # snow = surf[:, jspf, :]

                snow = surf.meridional_slice(xr_spf[:, :, ispf])

                # Extract pressure and non-dimensionalise
                if use_rot:
                    P = snow.P_rot
                    ax.set_ylabel(r"Reduced Static Pressure, $C_{p^*}$")
                else:
                    P = snow.P
                    ax.set_ylabel(r"Static Pressure, $C_p$")
                if Po2 > Po1:
                    # Compressor
                    Cp = (P - Po1) / (Po1 - P1)
                else:
                    # Turbine
                    Cp = (P - Po1) / (Po1 - P2)

                # Extract surface distance and normalise
                zeta_stag = snow.zeta_stag
                # Calculate maximum zeta only on main blade
                zeta_max = zeta_stag.max(axis=0)
                zeta_min = np.abs(zeta_stag.min(axis=0))
                zeta_norm = zeta_stag.copy()
                zeta_norm[zeta_norm < 0.0] /= zeta_min
                zeta_norm[zeta_norm > 0.0] /= zeta_max

                Cp_TE = 0.5 * (Cp[-1] + Cp[0]).item()
                # Cp /= Cp_TE
                # # p *= -1.0
                # Cp = np.sqrt(Cp)

                if fix_stag:
                    Cp -= Cp.max()

                if isurf == 0:
                    ax.plot(
                        np.abs(zeta_norm),
                        Cp,
                        label=f"spf={spf}",
                        color=f"C{ispf}",
                        linestyle=lnst[isurf],
                        marker="",
                    )
                else:
                    ax.plot(
                        np.abs(zeta_norm), Cp, color=f"C{ispf}", linestyle=lnst[isurf]
                    )

                if show_DF:
                    if show_DF[ispf]:
                        Cpmin = Cp.min()
                        Cpmax = Cp.max()
                        iDF = np.argmin(Cp)
                        CpTE = 0.5 * (Cp[-1] + Cp[0]).item()
                        DCpmin = Cpmin - Cpmax
                        DCpTE = CpTE - Cpmax
                        DF = 1.0 - np.sqrt(DCpTE / DCpmin)
                        ax.plot(
                            np.abs(zeta_norm)[iDF],
                            Cp[iDF],
                            color=f"C{ispf}",
                            marker="o",
                        )
                        ax.annotate(
                            r"$\mathit{DF}=" + f"{DF:.2f}" + "$",
                            xy=(np.abs(zeta_norm)[iDF], Cp[iDF]),
                            xytext=(0.0, 10.0),
                            textcoords="offset points",
                        )

                if not lim:
                    lim = ax.get_ylim()
                ax.set_ylim(lim)
                Ntick = 8
                dtick = np.round(np.ptp(lim) / (Ntick - 1), decimals=1)
                ax.yaxis.set_major_locator(ticker.MultipleLocator(dtick))

                # Store the raw data
                key = f"row_{irow}_spf_{spf}_blade_{isurf}"
                raw_data[key] = np.stack((zeta_stag, Cp))

            if target:
                xpeak_target, DF_target = target
                Cp_stag = Cp.max()
                Cp_TE = 0.5 * (Cp[-1] + Cp[0]).item()
                print(Cp_stag, Cp_TE)
                Cp_target = Cp_TE - (Cp_stag - Cp_TE) * DF_target**2
                print(DF_target)
                ax.plot(xpeak_target, Cp_target, "r*")

        if note:
            axlim = ax.axis()
            ax.annotate(
                note,
                xytext=(-5, -5),
                xy=(axlim[1], axlim[3]),
                ha="right",
                va="top",
                textcoords="offset points",
                arrowprops=None,
            )

        if compare:
            compare_dat = np.load(compare)

            zeta_stag, Cp = compare_dat[key]
            zeta_max = zeta_stag.max(axis=0)
            zeta_min = np.abs(zeta_stag.min(axis=0))
            zeta_norm = zeta_stag.copy()
            zeta_norm[zeta_norm < 0.0] /= zeta_min
            zeta_norm[zeta_norm > 0.0] /= zeta_max

            ax.plot(np.abs(zeta_norm), Cp, "k-")

        ax.grid("on")

        plotname = os.path.join(postdir, f"pressure_distribution_row_{irow}.pdf")
        if nsectplt > 1:
            ax.legend()
        ax.grid(False)
        plt.savefig(plotname)
        plt.close()

        if write_raw:
            rawname = os.path.join(postdir, "pressure_distributions_raw")
            np.savez_compressed(rawname, **raw_data)
