"""Save plots of pressure distributions."""
import numpy as np
import os
import turbigen.util
import matplotlib.pyplot as plt

logger = turbigen.util.make_logger()


def post(grid, machine, meanline, postdir, row_spf):

    lnst = ["-", "--"]

    # Loop over rows
    for irow, spfrow in enumerate(row_spf):

        # Extract reference pressure from mean-line
        iin = irow * 2
        iout = iin + 1
        Po1, Po2 = meanline.Po_rel[
            (iin, iout),
        ]
        P1, P2 = meanline.P[
            (iin, iout),
        ]

        # Get all blades in this row
        surfs = grid.cut_blade_surfs()[irow]

        fig, ax = plt.subplots()
        ax.set_xlabel(r"Surface Distance, $\zeta/\zeta_\mathrm{TE}$")
        ax.set_xlim((0.0, 1.0))
        ax.set_ylabel(r"Static Pressure, $C_p$")

        # Loop over span fractions
        for ispf, spf in enumerate(spfrow):

            # Find the j-index corresponding to current span fraction on main blade
            jspf = np.argmin(np.abs(surfs[0].spf[1, :, 0] - spf))

            # Loop over main/splitter
            for isurf, surf in enumerate(surfs):

                snow = surf[:, jspf, :]

                # Extract pressure and non-dimensionalise
                P = snow.P
                if Po2 > Po1:
                    # Compressor
                    Cp = (P - Po1) / (Po1 - P1)
                else:
                    # Turbine
                    Cp = (P - Po1) / (Po1 - P2)

                # Extract surface distance and normalise
                zeta_stag = snow.zeta_stag
                # Calculate maximum zeta only on main blade
                if isurf == 0:
                    zeta_max = np.abs(zeta_stag).max(axis=0, keepdims=True)
                zeta_norm = zeta_stag / zeta_max

                if isurf == 0:
                    ax.plot(
                        np.abs(zeta_norm),
                        Cp,
                        label=f"spf={spf}",
                        color=f"C{ispf}",
                        linestyle=lnst[isurf],
                    )
                else:
                    ax.plot(
                        np.abs(zeta_norm), Cp, color=f"C{ispf}", linestyle=lnst[isurf]
                    )

        plotname = os.path.join(postdir, f"pressure_distribution_row_{irow}.pdf")
        ax.legend()
        plt.tight_layout()
        plt.savefig(plotname)
        plt.close()
