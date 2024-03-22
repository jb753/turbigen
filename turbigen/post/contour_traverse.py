"""Contour loss coefficient over traverse plane."""
import os
import turbigen.util
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import warnings

logger = turbigen.util.make_logger()


def post(grid, machine, meanline, postdir, mnorm=None, coord_sys="yz", lim=None, var=()):

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
        s1 = meanline.s[::2]
        ho1 = meanline.ho_rel[::2]
        ho2 = meanline.ho_rel[1::2]
        h1 = meanline.h[::2]
        h2 = meanline.h[1::2]
        T2 = meanline.T[1::2]

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
        elif coord_sys == "rtx":
            c1 = Cu.rt[iunique]
            c2 = -Cu.x[iunique]
        else:
            raise Exception(f"Unrecognised coordinate system {coord_sys}")

        for iv, vname in enumerate(var):

            if vname == "Yp":

                dv = 0.1
                Po_rel = Cu.Po_rel[iunique]
                # Choose compressor or turbine definition
                PR = (P2 / P1)[i]
                if PR > 1.0:
                    v = (Po1 - Po_rel) / (Po1 - P1)
                else:
                    v = (Po1 - Po_rel) / (Po1 - P2)

                lab = "Stagnation Pressure Loss Coefficient, $Y_p$"

            if vname == "Ys":

                dv = 0.1
                s = Cu.s[iunique]

                # Choose compressor or turbine definition
                PR = (P2 / P1)[i]
                if PR > 1.0:
                    v = T2 * (s - s1[i]) / (ho1 - h1)
                else:
                    v = T2 * (s - s1[i]) / (ho2 - h2)

                lab = "Entropy Loss Coefficient, $Y_s$"

            elif vname == "Vm":

                if U[i] == 0.0:
                    Vref = Cu.Vm[iunique].mean()
                    lab = r"Meridional Velocity, $V_m/\overline{V_m}$"
                else:
                    Vref = U[i]
                    lab = r"Meridional Velocity, $V_m/U$"
                dv = 0.05
                v = Cu.Vm[iunique] / Vref


            else:
                raise Exception(f"Unrecognised plot variable {vname}")

            lev = turbigen.util.clipped_levels(v, dv, thresh=0.01)

            if lim:
                if lim[iv]:
                    lev = np.arange(*lim[iv], dv)

            v = np.clip(v, lev[0], lev[-1])

            fig, ax = plt.subplots()
            # It seems that we have to pass triangles as a kwarg to tricontour,
            # not positional, but this results in a UserWarning that contour
            # does not take it as a kwarg. So catch and hide this warning.
            with warnings.catch_warnings(action='ignore', category=UserWarning):
                cm = ax.tricontourf(c1, c2, v, lev, triangles=triangles, cmap="cubehelix", linestyles="none")

            cm.set_edgecolor("face")

            ax.axis("equal")
            # ax.axis('tight')
            ax.axis("off")
            hc = plt.colorbar(cm, label=lab)
            hc.ax.yaxis.set_major_locator(ticker.MultipleLocator(dv*2))
            plt.tight_layout()

            figname = os.path.join(postdir, f"traverse_{vname}_{i}.pdf")
            plt.savefig(figname)
            plt.close()
