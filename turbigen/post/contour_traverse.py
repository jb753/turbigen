"""Contour loss coefficient over traverse plane."""
import os
import turbigen.util
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as patches
import warnings

logger = turbigen.util.make_logger()


def post(grid, machine, meanline, postdir, mnorm=None, coord_sys="yz", lim=None, var=(), step=None, horiz_offset=0., title=None):

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
        Npts = len(iunique)

        Cu = C.to_unstructured()

        # Choose coordinate system
        if coord_sys == "yz":
            c1 = Cu.y[iunique]
            c2 = Cu.z[iunique]
        elif coord_sys == "rtx":
            pitch = Cu.pitch
            rt_pitch = pitch * Cu.r.mean()
            xrt = Cu.xrt[:,iunique]
            tref = 0.5*(xrt[2].min()+xrt[2].max())
            xrt[2] -= tref
            xrt[0] *= -1.
            # Repeat by +- a pitch
            xrtp = xrt.copy()
            xrtp[2] += pitch
            xrtm = xrt.copy()
            xrtm[2] -= pitch
            xrt = np.concatenate((xrtm, xrt, xrtp),axis=-1)
            c1 = xrt[1]*xrt[2]
            c2 = xrt[0]

            trim = triangles.copy()
            tri = trim.copy()
            tri += Npts
            trip = tri.copy()
            trip += Npts
            triangles = np.concatenate((trim, tri, trip))


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

            if step:
                dv = step[iv]

            if coord_sys == "rtx":
                v = np.tile(v,(3,))

            lev = turbigen.util.clipped_levels(v, dv, thresh=0.01)

            if lim:
                if lim[iv]:
                    lev = np.arange(*lim[iv], dv)

            v = np.clip(v, lev[0], lev[-1])

            fig, ax = plt.subplots()
            # It seems that we have to pass triangles as a kwarg to tricontour,
            # not positional, but this results in a UserWarning that contour
            # does not take it as a kwarg. So catch and hide this warning.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cm = ax.tricontourf(c1, c2, v, lev, triangles=triangles, cmap="cubehelix", linestyles="none")

            cm.set_edgecolor("face")

            hc = plt.colorbar(cm, label=lab)
            hc.ax.yaxis.set_major_locator(ticker.MultipleLocator(dv*2))
            if title:
                ax.set_title(title)

            plt.tight_layout(pad=0.1)

            ax.axis("equal")
            ax.axis("off")


            if coord_sys == "rtx":

                rtlim = (np.array([-.5,.5])+horiz_offset)*rt_pitch
                xlim = np.array([c2.min(), c2.max()])


                # Hub and casing shading
                dr = xlim.ptp()*0.07

                # Add the rectangle patch to the axis
                # ax.add_patch(patches.Rectangle((rtlim[0], xlim[0]-dr), rt_pitch, dr, fill=True, hatch='/', color='grey', alpha=0.5, linestyle='none'))
                # ax.add_patch(patches.Rectangle((rtlim[0], xlim[1]), rt_pitch, dr, fill=True, hatch='/', color='grey', alpha=0.5, linestyle='none'))

                ax.text(rtlim.mean(), xlim[0]-dr, 'Hub', ha='center', va='center')
                ax.text(rtlim.mean(), xlim[1]+dr, 'Casing', ha='center', va='center')

                ax.set_xlim(rtlim)


            figname = os.path.join(postdir, f"traverse_{vname}_{i}.pdf")
            plt.savefig(figname)
            plt.close()
