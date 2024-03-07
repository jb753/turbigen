"""Contour loss coefficient over traverse plane."""
import os
import turbigen.util
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

logger = turbigen.util.make_logger()


def post(
    grid, machine, meanline, postdir, mnorm_traverse=None, lev=None, coord_sys="yz"
):

    logger.info("Contouring Yp...")

    if not mnorm_traverse:
        logger.info("No cut locations specified.")

    # Loop over stations
    for i, ti in enumerate(mnorm_traverse):

        # Extract reference pressures
        Po1 = meanline.Po_rel[::2]
        P1 = meanline.P[::2]
        P2 = meanline.P[1::2]

        # Get meridional coordinates of the cut planes
        xrc = machine.ann.get_cut_plane(ti)[0]

        # Take the cut
        C = grid.unstructured_cut_marching(xrc)

        (x, r, t), triangles, iunique = C.get_triangulation()

        Cu = C.to_unstructured()

        Po_rel = Cu.Po_rel[iunique]

        # Choose compressor or turbine definition
        PR = (P2 / P1)[i]
        if PR > 1.0:
            Yp = (Po1 - Po_rel) / (Po1 - P1)
        else:
            Yp = (Po1 - Po_rel) / (Po1 - P2)

        dYp = 0.1
        if not lev:
            lev = turbigen.util.clipped_levels(Yp, dYp, thresh=0.01)

        # Choose coordinate system
        if coord_sys == "yz":
            c1 = Cu.y[iunique]
            c2 = Cu.z[iunique]
        else:
            raise Exception(f"Unrecognised coordinate system {coord_sys}")

        fig, ax = plt.subplots()
        cm = ax.tricontourf(c1, c2, Yp, lev, cmap="cubehelix")
        ax.axis("equal")
        # ax.axis('tight')
        ax.axis("off")
        hc = plt.colorbar(cm, label="Stagnation Pressure Loss Coefficient, $Y_p$")
        hc.ax.yaxis.set_major_locator(ticker.MultipleLocator(dYp))
        plt.tight_layout()

        figname = os.path.join(postdir, f"traverse_Yp_{i}.pdf")
        plt.savefig(figname)
        plt.close()
