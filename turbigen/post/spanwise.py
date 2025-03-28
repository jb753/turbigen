"""Contour loss coefficient over traverse plane."""

import os
import turbigen.util
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum

logger = turbigen.util.make_logger()


class PlotVars(Enum):
    VT = "Vt"
    VM = "Vm"
    YS = "Ys"
    CP = "Cp"
    CPO = "Cpo"
    CHO = "Cho"


def post(
    grid,
    machine,
    meanline,
    __,
    postdir,
    merid,
    variable,
    lim=None,
    title=None,
):
    """
    Plot a flow property in spanwise direction.

    """

    logger.info("Spanwise plotting...")

    try:
        plot_var = PlotVars(variable)
    except KeyError:
        raise Exception(
            f"variable={variable} is not a valid name, expecting one of {[c for c in PlotVars]}"
        )

    # Take an unstructured cut at given meridional position
    xrc = machine.ann.get_cut_plane(merid)[0]
    logger.info(f"merid={merid}, cut plane={xrc}")
    Cs = grid.unstructured_cut_marching(xrc).interpolate_to_structured()

    Csc = Cs.empty(shape=(Cs.ni, Cs.nj - 1, Cs.nk - 1))
    Csc._data[:] = turbigen.util.node_to_face(Cs._data)

    dAi = Cs.dAi
    mflux = Csc.rho * Csc.Vxrt
    mflow = turbigen.util.dot(mflux, dAi).sum(axis=-1).squeeze()
    dAmag = turbigen.util.vecnorm(dAi)
    area = dAmag.sum(axis=-1).squeeze()
    spf = Csc.spf[0, :, 0]

    def mass_avg(var):
        numer = turbigen.util.dot(mflux * var, dAi).sum(axis=-1).squeeze()
        return numer / mflow

    def area_avg(var):
        numer = (dAmag * var).sum(axis=-1).squeeze()
        return numer / area

    # Set the index into meanline for reference conditions
    irow_ref = int(merid / 2 - 1)

    # Choose if this is compressor or turbine
    row = meanline.get_row(irow_ref)
    is_compressor = np.diff(row.P) > 0.0
    logger.info(f"is_compressor={is_compressor}")

    # Now non-dimensionalise the variable we want

    # Entropy loss coefficient
    if plot_var == PlotVars.YS:
        s = mass_avg(Csc.s)
        if is_compressor:
            v = row.T[1] * (s - row.s[0]) / row.halfVsq_rel[0]
        else:
            v = row.T[1] * (s - row.s[0]) / row.halfVsq_rel[1]

        label = "Entropy Loss Coefficient, $Y_s$"
        Ys_mean = np.ptp(row.s) * row.T[1] / row.halfVsq_rel[0]
        logger.info(f"Mean Ys={Ys_mean}")

    elif plot_var == PlotVars.VM:
        if U := meanline.U.max():
            v = mass_avg(Csc.Vm) / U
        label = r"Meridional Velocity, $V_m/U$"

    elif plot_var == PlotVars.VT:
        v = mass_avg(Csc.Vt) / row.U[1]
        label = r"Circumferential Velocity, $V_\theta/U$"

    elif plot_var == PlotVars.CP:
        Po1 = row.Po_rel[0]
        Po2 = row.Po_rel[1]
        P1 = row.P[0]
        P2 = row.P[1]
        P = area_avg(Csc.P)
        if is_compressor:
            v = (P - Po1) / (Po1 - P1)
        else:
            v = (P - Po1) / (Po2 - P2)
        label = r"Static Pressure, $C_p$"

    elif plot_var == PlotVars.CPO:
        Po1 = row.Po_rel[0]
        Po2 = row.Po_rel[1]
        P1 = row.P[0]
        P2 = row.P[1]
        P = mass_avg(Csc.Po)
        if is_compressor:
            v = (P - Po1) / (Po1 - P1)
        else:
            v = (P - Po1) / (Po2 - P2)
        label = r"Stagnation Pressure, $C_{p0}$"

    elif plot_var == PlotVars.CHO:
        ho = mass_avg(Csc.ho)
        v = (ho - row.ho[0]) / row.U[1] ** 2
        label = r"Stagnation Enthalpy, $C_{h0}$"

    else:
        raise Exception("Should not reach here.")

    fig, ax = plt.subplots(layout="constrained")
    ax.plot(v, spf)
    ax.set_xlabel(label)
    ax.set_ylabel("Span Fraction")

    if lim:
        ax.set_xlim(lim)
    ax.set_ylim((0, 1))

    if title:
        ax.set_title(title)

    figname = os.path.join(postdir, f"spanwise_{variable}_{merid:.3}.pdf")
    plt.savefig(figname)
    plt.close()
