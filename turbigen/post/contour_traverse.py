"""Contour loss coefficient over traverse plane."""
import os
import turbigen.util
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

logger = turbigen.util.make_logger()


def post(grid, machine, meanline, postdir, mnorm=None, coord_sys="yz", var=()):

    logger.info("Contouring traverse planes...")

    if not mnorm:
        logger.info("No cut locations specified.")

    # Loop over stations
    for i, ti in enumerate(mnorm):

        # Extract reference pressures
        Po1 = meanline.Po_rel[::2]
        P1 = meanline.P[::2]
        P2 = meanline.P[1::2]
        U = meanline.U[::2]

        # Get meridional coordinates of the cut planes
        xrc = machine.ann.get_cut_plane(ti)[0]

        # Take the cut
        C = grid.unstructured_cut_marching(xrc)

        _, triangles, iunique = C.get_triangulation()

        Cu = C.to_unstructured()

        # Choose coordinate system
        if coord_sys == "yz":
            c1 = Cu.y[iunique]
            c2 = Cu.z[iunique]
        else:
            raise Exception(f"Unrecognised coordinate system {coord_sys}")

        for vname in var:

            if vname == "Yp":

                dv = 0.1
                # Choose compressor or turbine definition
                Po_rel = Cu.Po_rel[iunique]
                PR = (P2 / P1)[i]
                if PR > 1.0:
                    v = (Po1 - Po_rel) / (Po1 - P1)
                else:
                    v = (Po1 - Po_rel) / (Po1 - P2)

                lab = "Stagnation Pressure Loss Coefficient, $Y_p$"

            elif vname == "Vm":

                if U[i] == 0.0:
                    Vref = Cu.Vm[iunique].mean()
                else:
                    Vref = U[i]
                dv = 0.1
                Vm = Cu.Vm[iunique] / Vref
                v = Vm

                lab = "Meridional Velocity, $V_m/U$"

            else:
                raise Exception(f"Unrecognised plot variable {vname}")

            lev = turbigen.util.clipped_levels(v, dv, thresh=0.01)

            fig, ax = plt.subplots()
            cm = ax.tricontourf(c1, c2, v, lev, cmap="cubehelix", linestyles="none")
            ax.axis("equal")
            # ax.axis('tight')
            ax.axis("off")
            hc = plt.colorbar(cm, label=lab)
            hc.ax.yaxis.set_major_locator(ticker.MultipleLocator(dv))
            plt.tight_layout()

            figname = os.path.join(postdir, f"traverse_{vname}_{i}.pdf")
            plt.savefig(figname)
            plt.close()
