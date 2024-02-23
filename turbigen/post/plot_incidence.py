"""Save plots of incidence."""
import os
import turbigen.util
import matplotlib.pyplot as plt

logger = turbigen.util.make_logger()


def post(grid, machine, meanline, postdir):

    logger.info("Plotting incidence")
    data = turbigen.util.incidence(grid, machine, meanline)

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
            ax[1].set_ylabel("Span Fraction")
            ax[1].plot(inc, spf)

            pltname = os.path.join(postdir, f"incidence_row_{irow}_blade_{jblade}.pdf")
            plt.tight_layout(pad=0.1)
            plt.savefig(pltname)
            plt.close()
