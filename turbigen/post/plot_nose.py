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

        lev_Cp = np.linspace(-1.5,0.1)

        # Loop over span fractions
        for ispf, spf in enumerate(spfrow):

            cut = grid.cut_span(spf)

            fig, ax = plt.subplots()

            for b in cut:

                P = b.P

                if Po2 > Po1:
                    # Compressor
                    Cp = (P - Po1) / (Po1 - P1)
                else:
                    # Turbine
                    Cp = (P - Po1) / (Po1 - P2)

                ax.contourf(b.y, b.z, Cp, lev_Cp)
            ax.axis('equal')
            plt.show()

        # plotname = os.path.join(postdir, f"pressure_distribution_row_{irow}.pdf")
        # ax.legend()
        # plt.tight_layout()
        # plt.savefig(plotname)
        # plt.close()
