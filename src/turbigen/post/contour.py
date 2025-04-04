"""Contour loss coefficient over traverse plane."""

import os
import turbigen.util
import turbigen.base
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
import warnings

logger = turbigen.util.make_logger()


class CoordMode(Enum):
    X = "x"
    R = "r"
    M = "m"
    SPF = "spf"


class PlotVars(Enum):
    VT = "Vt"
    VM = "Vm"
    YS = "Ys"
    CP = "Cp"
    CHO = "Cho"


def post(
    grid,
    machine,
    meanline,
    __,
    postdir,
    coord,
    value,
    variable,
    lim=None,
    step=None,
    theta_offset=0.0,
    title=None,
    N_passage=1,
    irow_ref=0,
    cmap="cubehelix",
    extend="max",
    show_mesh=False,
):
    """contour(coord, var=(), lim=None, step=None, theta_offset=0.0, title=None)
    Plot flow-field contours over a planar cut.

    Parameters
    ----------
    coord: str
        Which coordinate to take constant over the cut:
        - 'x' axial
        - 'r' radial
        - 'm' normalised meridional distance
    value: float
        Value of the coordinate to cut at.
    variable: str
        Variable name to plot. Select from `Yp, Ys, Cho, Vm`.
    lim: (2,) list
        Upper and lower contour limits for the plot variable.
    step: float
        Contour step for, omit to set automatically.
    theta_offset: float
        Angular offset of the plot as a fraction of pitch. Use to line up wakes in different simulations.
    title: str
        String to add as a plot title, omit for no title.
    N_passage: int
        Number of passages to repeat.
    irow_ref: int
        Which row to use to make things non-dimensional
    cmap: str
        A matplotlib colormap
    extend: str
        Extend mode for the contours and colorbar.
    """

    logger.info("Contouring...")

    # Process the input data
    try:
        coord_mode = CoordMode(coord)
    except KeyError:
        raise Exception(
            f"coord={coord} is not a valid cut, expecting one of {[c for c in CoordMode]}"
        )

    try:
        plot_var = PlotVars(variable)
    except KeyError:
        raise Exception(
            f"variable={variable} is not a valid name, expecting one of {[c for c in PlotVars]}"
        )

    if coord_mode == CoordMode.SPF:
        xrc = machine.ann.get_span_curve(value, n=101)

        # Cut and repeat each row separately
        Crow = grid.cut_span_unstructured(xrc)
        Crow = [Ci.repeat_pitchwise(N_passage) for Ci in Crow]

        # Combine the rows
        C = turbigen.base.concatenate(Crow)

    else:
        # Get an xr curve describing the cut plane.
        if coord_mode == CoordMode.X:
            xrc = np.array([[value, value], [0.1, 1.0]])
        elif coord_mode == CoordMode.R:
            xrc = np.array([[-1.0, 1.0], [value, value]])
        elif coord_mode == CoordMode.M:
            xrc = machine.ann.get_cut_plane(value)[0]
        else:
            raise Exception("Should not reach here")
        C = grid.unstructured_cut_marching(xrc)

        C = C.repeat_pitchwise(N_passage)

    # Matplotlib style triangulate, repeat if needed
    C_tri, triangles = C.get_mpl_triangulation()

    # Centre theta on zero
    C_tri.t -= 0.5 * (C_tri.t.min() + C_tri.t.max())

    # Add on offset
    C_tri.t += theta_offset * C_tri.pitch

    # Get the coordinates to plot
    if coord_mode == CoordMode.X:
        c1, c2 = C_tri.yz
    elif coord_mode == CoordMode.R:
        c1, c2 = C_tri.rt, C_tri.x
    elif coord_mode == CoordMode.SPF:
        # Now generate a mapping from xr to meridional distance
        mp_from_xr = machine.ann.get_mp_from_xr(value)
        c1 = mp_from_xr(C_tri.xr)
        c2 = C_tri.t
    elif coord_mode == CoordMode.M:
        if np.ptp(C_tri.r) > np.ptp(C_tri.x):
            c1, c2 = C_tri.yz
        else:
            c1, c2 = C_tri.rt, C_tri.r
    else:
        raise Exception("Should not reach here")

    if irow_ref is None:
        if coord_mode == CoordMode.M:
            irow_ref = int(value / 2 - 1)
        elif machine.Nrow > 1:
            raise Exception(
                "Need to set irow_ref if plotting at constant x, r, or span."
            )

    # Choose if this is compressor or turbine
    row = meanline.get_row(irow_ref)
    is_compressor = np.diff(row.P) > 0.0

    # Now non-dimensionalise the variable we want
    Uref = meanline.U.max()

    # Entropy loss coefficient
    if plot_var == PlotVars.YS:
        if is_compressor:
            v = row.T[1] * (C_tri.s - row.s[0]) / row.halfVsq_rel[0]
        else:
            v = row.T[1] * (C_tri.s - row.s[0]) / row.halfVsq_rel[1]

        label = "Entropy Loss Coefficient, $Y_s$"

    elif plot_var == PlotVars.VM:
        v = C_tri.Vm / Uref
        label = r"Meridional Velocity, $V_m/U$"

    elif plot_var == PlotVars.VT:
        v = C_tri.Vm / Uref
        label = r"Circumferential Velocity, $V_\theta/U$"

    elif plot_var == PlotVars.CHO:
        v = (C_tri.ho - row.ho[0]) / Uref**2
        label = r"Stagnation Enthalpy, $C_{h_0}$"

    elif plot_var == PlotVars.CP:
        Po1 = row.Po_rel[0]
        Po2 = row.Po_rel[1]
        P1 = row.P[0]
        P2 = row.P[1]
        if is_compressor:
            v = (C_tri.P - Po1) / (Po1 - P1)
        else:
            v = (C_tri.P - Po1) / (Po2 - P2)
        label = r"Static Pressure, $C_p$"

    else:
        raise Exception("Should not reach here.")

    # Now set contour levels
    if not step:
        step = 0.1
    if lim:
        levels = np.arange(*lim, step)
    else:
        # levels = turbigen.util.clipped_levels(v, step, thresh=0.01)
        levels = np.linspace(v.min(), v.max(), 20)

    eps = 1e-4 * np.diff(levels).mean()
    v = np.clip(v, levels[0] + eps, levels[-1] - eps)
    # v = np.clip(v, lev[-1] + eps, None)

    fig, ax = plt.subplots(layout="constrained")

    # It seems that we have to pass triangles as a kwarg to tricontour,
    # not positional, but this results in a UserWarning that contour
    # does not take it as a kwarg. So catch and hide this warning.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cm = ax.tricontourf(
            c1,
            c2,
            v,
            levels,
            triangles=triangles,
            cmap=cmap,
            linestyles="none",
            extend=extend,
        )

    ax.set_aspect("equal")  # Ensures equal scaling
    ax.axis("off")

    # Set x-limits tightly around data
    x = c1
    y = c2
    ax.set_xlim(x.min(), x.max())

    # Adjust y-limits based on the new equal aspect ratio
    ylim_range = (x.max() - x.min()) / ax.get_data_ratio()  # Calculate new y-range
    y_mid = (y.max() + y.min()) / 2  # Midpoint of y-data
    ax.set_ylim(y_mid - ylim_range / 2, y_mid + ylim_range / 2)  # Adjust y-limits

    # Make the colorbar
    cm.set_edgecolor("face")
    plt.colorbar(cm, label=label, shrink=0.8)

    # Show hub and casing for r=const cuts
    # These are easy to trim at constant theta either side
    # TODO make it so we can do this for yz cuts as well
    if coord_mode == CoordMode.R:
        xlim = np.array([-0.5, 0.5]) * N_passage * C_tri.pitch * C_tri.r.mean()
        ax.set_xlim(*xlim)

        # Remove box, grey backgroud for hub/casing/blades
        ax.set_aspect("equal", adjustable="box")
        ax.set_facecolor(np.ones((3,)) * 0.7)
        ax.set_xticks(())
        ax.set_yticks(())
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Hub and casing labels
        c2lim = np.array([c2.min(), c2.max()])
        dc2 = np.ptp(c2lim) * 0.07
        ax.text(xlim.mean(), c2lim[0] - dc2, "Hub", ha="center", va="center")
        ax.text(xlim.mean(), c2lim[1] + dc2, "Shroud", ha="center", va="center")
        ax.set_ylim(c2lim[0] - 2 * dc2, c2lim[1] + 2 * dc2)

    if title:
        ax.set_title(title)

    if show_mesh:
        ax.triplot(c1, c2, triangles, "k-", lw=0.1)

    figname = os.path.join(postdir, f"contour_{variable}_{coord}_{value:.3}.pdf")
    rawname = os.path.join(postdir, f"contour_{variable}_{coord}_{value:.3}.npz")
    np.savez_compressed(rawname, c1=c1, c2=c2, v=v, triangles=triangles)

    plt.savefig(figname)
    # plt.show()
    plt.close()
