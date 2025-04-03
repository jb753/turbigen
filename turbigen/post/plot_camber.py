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
        fig, ax = plt.subplots(layout="constrained")
        ax.set_xlabel(r"Meridional Distance, $m/c_m$")
        ax.set_ylabel(r"Normalised Metal Angle, $\hat{\chi}$")
        ax.set_xlim((0.0, 1.0))

        # Loop over span fractions
        for ispf, spf in enumerate(spfrow):
            cam, _ = machine.bld[irow][0]._get_camber_thickness(spf)

            if isinstance(cam, turbigen.camber.Taylor):
                chi_str = (
                    r"$\hat{\chi} = "
                    r"\frac{\chi - \chi_\mathrm{in} }"
                    r"{\chi_\mathrm{out} - \chi_\mathrm{in}}$"
                )
            else:
                chi_str = (
                    r"$\hat{\chi} = "
                    r"\frac{\tan\chi - \tan\chi_\mathrm{in} }"
                    r"{\tan\chi_\mathrm{out} - \tan\chi_\mathrm{in}}$"
                )

            # title_str = (
            #     (r"$\chi_\mathrm{in}=%.1f^\circ$, " % cam.chi(0.0))
            #     + (r"$\chi_\mathrm{out}=%.1f^\circ$, " % cam.chi(1.0))
            #     + (r"$\hat{\chi}'(0)=%.2f$, " % cam.q_camber[2])
            #     + (r"$\hat{\chi}'(1)=%.2f$, " % cam.q_camber[3])
            #     + (r"$\hat{\chi}''(0.5)=%.2f$" % cam.q_camber[4])
            # )
            #
            # ax.set_title(title_str, pad=10)
            ax.text(0.025, 0.95, chi_str, va="top")

            col = f"C{ispf}"
            ax.plot(m, cam.chi_hat(m), "-", color=col, label=f"spf={spf}")

            mctrl = (0.0, 0.5, 1.0)
            ax.plot(mctrl, cam.chi_hat(mctrl), "o", color=col, ms=10)
            ax.set_ylim((0.0, 1.0))

        ax.legend(loc="lower right")

        plotname = os.path.join(postdir, f"camber_row_{irow}.pdf")
        plt.savefig(plotname)
        plt.close()
