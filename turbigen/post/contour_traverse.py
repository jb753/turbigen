"""Contour loss coefficient over traverse plane."""

import os
import turbigen.util
import numpy as np
import matplotlib.pyplot as plt

from turbigen.util import make_contour

logger = turbigen.util.make_logger()


def post(
    grid,
    machine,
    meanline,
    postdir,
    mnorm=[],
    coord_sys="yz",
    lim=None,
    var=(),
    step=None,
    theta_offset=0.0,
    title=None,
    N_passage=1,
):
    """contour_traverse(norm=[], coord_sys="yz", var=(), lim=None, step=None, theta_offset=0.0, title=None)
    Plot flow-field contours over a traverse cut at constant streamwise position.

    This is useful, for example, to give an indication of the distribution of loss downstream of blade rows.

    Parameters
    ----------
    mnorm_cut: list
        Normalised meridional coordinates at which to take the cuts. The coordinate is defined 0 at the inlet plane,
        1 at the first row LE, 2 at the first row TE, 3 at the second row LE, and so on. For example, to cut
        just upstream and downstream of the first row, use [0.95, 2.05]
    coord_sys: str
        Which coordinate system to plot in. `yz` for axials or `rtx` for radials.
    var: list of str
        Variable names to plot. Select from `Yp, Ys, Cho, Vm`.
    lim: list of (2,) list
        Upper and lower contour limits for each plot variable, omit to set automatically.
    step: list of float
        Contour steps for each plot variable, omit to set automatically.
    theta_offset: float
        Angular offset of the plot as a fraction of pitch. Use to line up wakes in different simulations.
    title: str
        String to add as a plot title, omit for no title.

    """

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
        C = grid.unstructured_cut_marching(xrc).repeat_pitchwise(N_passage)
        Ct, triangles = C.get_triangulation()

        ii = int(ti / 2 - 1)

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

            elif vname == "Cp":
                dv = 0.1
                P = Cu.P[iunique]
                v = (P - Po1) / (Po1 - P1)
                lab = "Static Pressure, $C_p$"

            elif vname == "Cp_rot":
                dv = 0.1
                P = Cu.P_rot[iunique]
                v = (P - Po1) / (Po1 - P1)
                lab = "Reduced Static Pressure, $C^*_p$"

            elif vname == "Ys":
                dv = 0.1
                s = Cu.s[iunique]

                # Choose compressor or turbine definition
                PR = (P2 / P1)[ii]
                if PR > 1.0:
                    v = T2[ii] * (s - s1[ii]) / (ho1[ii] - h1[ii])
                else:
                    v = T2[ii] * (s - s1[ii]) / (ho2[ii] - h2[ii])

                lab = "Entropy Loss Coefficient, $Y_s$"

            elif vname == "Cho":
                dv = 0.1
                ho = Cu.ho[iunique]

                v = (ho - ho1[ii]) / np.abs(ho2[ii] - ho1[ii])

                lab = "Work Coefficient, $C_{h_0}$"

            elif vname == "Vm":
                if U[ii] == 0.0:
                    Vref = Cu.Vm[iunique].mean()
                    lab = r"Meridional Velocity, $V_m/\overline{V_m}$"
                else:
                    Vref = U[ii]
                    lab = r"Meridional Velocity, $V_m/U$"
                dv = 0.05
                v = Cu.Vm[iunique] / Vref

            else:
                raise Exception(f"Unrecognised plot variable {vname}")

            if step:
                dv = step[iv]

            if coord_sys == "rtx":
                v = np.tile(v, (3,))

            lev = turbigen.util.clipped_levels(v, dv, thresh=0.01)
            if lim:
                if lim[iv]:
                    lev = np.arange(*lim[iv], dv)

            rtlim = (np.array([-0.5, 0.5]) + theta_offset) * rt_pitch
            fig, ax = make_contour(c1, c2, v, triangles, lev, lab, rtlim)
            if title:
                ax.set_title(title)

            figname = os.path.join(postdir, f"traverse_{vname}_{i}.pdf")
            plt.savefig(figname)
            plt.close()
