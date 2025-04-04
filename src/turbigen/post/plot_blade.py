"""Save plot of camber line."""

import os
import turbigen.util
import turbigen.camber
import matplotlib.pyplot as plt


logger = turbigen.util.make_logger()


def post(grid, machine, meanline, _, postdir, row_spf):
    # Meridional locations to plot at
    m = turbigen.util.cluster_cosine(50)

    # Loop over rows
    for irow, spfrow in enumerate(row_spf):
        # Set up axes
        fig, ax = plt.subplots(
            2, 1, layout="constrained", sharex=True, figsize=(3.5, 3.8)
        )
        ax[1].set_xlabel(r"Meridional Distance, $m/c_m$")
        ax[0].set_ylabel(r"Camber, $\hat{\chi}$")
        ax[1].set_ylabel(r"Thickness, $\hat{\tau}$")
        ax[0].set_xlim((0.0, 1.0))
        ax[0].set_ylim((0.0, 1.0))
        ax[1].set_ylim((0.0, 0.13))
        ax[0].set_xticks((0.0, 0.5, 1.0))

        # Loop over span fractions
        for ispf, spf in enumerate(spfrow):
            cam, thick = machine.bld[irow][0]._get_camber_thickness(spf)

            ax[0].plot(m, cam.chi_hat(m), "k-")
            ax[1].plot(m, thick.thick(m), "k-")

        plotname = os.path.join(postdir, f"blade_row_{irow}.pdf")
        plt.savefig(plotname)
        plt.close()
