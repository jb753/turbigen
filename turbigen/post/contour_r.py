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
    r_cut=[],
    lim=None,
    var=(),
    step=None,
    title=None,
    irow_ref=0,
    theta_offset=0.3,
    Npass=1,
):
    """contour_r(r_cut=[], var=(), lim=None, step=None, title=None, irow_ref=0)
    Plot flow-field contours over a traverse cut at constant radius.
    """

    logger.info("Contouring constant r planes...")

    # Loop over stations
    for i in range(len(r_cut)):
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
        xrc = np.array([[-1.0, 1.0], [r_cut[i], r_cut[i]]])

        # Take the cut
        C = grid.unstructured_cut_marching(xrc)
        _, triangles, iunique = C.get_triangulation()
        Npts = len(iunique)
        Cu = C.to_unstructured()

        # Replicate +/- a pitch
        pitch = Cu.pitch
        rt_pitch = pitch * Cu.r.mean()
        assert np.ptp(Cu.r) / r_cut[i] < 1e-6
        xrt = Cu.xrt[:, iunique]
        tref = 0.5 * (xrt[2].min() + xrt[2].max())
        xrt[2] -= tref
        xrt[0] *= -1.0
        xrtp = xrt.copy()
        xrtp[2] += pitch
        xrtm = xrt.copy()
        xrtm[2] -= pitch
        xrt = np.concatenate((xrtm, xrt, xrtp), axis=-1)
        c1 = xrt[1] * xrt[2]
        c2 = xrt[0]
        trim = triangles.copy()
        tri = trim.copy()
        tri += Npts
        trip = tri.copy()
        trip += Npts
        triangles = np.concatenate((trim, tri, trip))

        ii = int(irow_ref)

        flip = False
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

            elif vname == "x":
                dv = 0.001
                v = Cu.x[iunique]
                lab = "Axial Coordinate"

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
                flip = True

            elif vname == "Vt":
                Vref = U[ii]
                lab = r"Circumferential Velocity, $V_\theta/U$"
                dv = 0.05
                v = Cu.Vt[iunique] / Vref
                flip = True

            else:
                raise Exception(f"Unrecognised plot variable {vname}")

            if step:
                dv = step[iv]

            v = np.tile(v, (3,))

            lev = turbigen.util.clipped_levels(v, dv, thresh=0.01)
            if lim:
                if lim[iv]:
                    lev = np.arange(*lim[iv], dv)

            rtlim = (np.array([-Npass / 2.0, Npass / 2.0]) + theta_offset) * rt_pitch
            fig, ax = make_contour(c1, c2, v, triangles, lev, lab, rtlim, flip)
            if title:
                ax.set_title(title, pad=18.0)

            figname = os.path.join(postdir, f"rcontour_{vname}_{i}.pdf")
            plt.savefig(figname)
            plt.close()
