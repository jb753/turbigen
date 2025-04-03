"""Save plot of thickness distribution."""

import os
import turbigen.util
import matplotlib.pyplot as plt


logger = turbigen.util.make_logger()


def post(grid, machine, meanline, _, postdir, row_spf):
    # Meridional locations to plot at
    m = turbigen.util.cluster_cosine(501)

    # Loop over rows
    for irow, spfrow in enumerate(row_spf):
        # Set up axes
        fig, ax = plt.subplots()
        ax.set_xlabel(r"Meridional Distance, $m/c_m$")
        ax.set_ylabel(r"Thickness")
        ax.set_xlim((0.0, 1.0))

        # Loop over span fractions
        for ispf, spf in enumerate(spfrow):
            _, thick = machine.bld[irow][0]._get_camber_thickness(spf)

            # title_str = (
            #     (r"$R_\mathrm{LE}=%.2f$, " % thick.q_thick[0])
            #     + (r"$t_\mathrm{max}=%.2f$, " % thick.q_thick[1])
            #     + (r"$x_{t_\mathrm{max}}=%.2f$, " % thick.q_thick[2])
            #     + (r"$\kappa_{t_\mathrm{max}}=%.2f$, " % thick.q_thick[3])
            #     + (r"$t_\mathrm{TE}=%.2f$, " % thick.q_thick[4])
            #     + (r"$\tan\zeta=%.1f^\circ$" % thick.q_thick[5])
            # )

            # ax.set_title(title_str, pad=10)
            plt.tight_layout()

            col = f"C{ispf}"
            ax.plot(m, thick.thick(m), "-", color=col)

            # ax.legend()
            # mctrl = (0.0, thick.s_tmax, 1.0)
            # ax.plot(mctrl, thick.tau(mctrl), "o", color=col, ms=10)
            ax.set_ylim(bottom=0)

        plotname = os.path.join(postdir, f"thickness_row_{irow}.pdf")
        # ax.legend()
        plt.savefig(plotname)
        plt.close()
