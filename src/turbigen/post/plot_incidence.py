"""Save plots of incidence."""

import numpy as np
import os
import turbigen.util
import matplotlib.pyplot as plt

logger = turbigen.util.make_logger()


def post(grid, machine, meanline, _, postdir, fac_RLE=1.0):
    logger.info("Plotting incidence")
    data = turbigen.util.incidence(grid, machine, meanline, fac_RLE)

    for irow in range(len(data)):
        for jblade in range(len(data[irow])):
            spf, inc, chi_stag, chi_metal = data[irow][jblade]

            fig, ax = plt.subplots(1, 2)
            ax[0].set_xlabel("Angle/deg")
            ax[0].set_ylabel("Span Fraction")
            ax[0].plot(chi_stag, spf, label="Flow")
            ax[0].plot(chi_metal, spf, label="Metal")
            ax[0].legend()
            ax[1].set_xlabel("Incidence/deg")
            ax[1].plot(inc, spf)

            ax[1].set_xlim(np.nanquantile(inc, [0.01, 0.99]))

            pltname = os.path.join(postdir, f"incidence_row_{irow}_blade_{jblade}.pdf")
            plt.tight_layout(pad=0.1)
            plt.savefig(pltname)
            plt.close()
