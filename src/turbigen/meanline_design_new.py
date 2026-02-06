"""Define the interface for mean-line designers."""

import numpy as np

from turbigen.plugins import register_mean_line


@register_mean_line
def turbine_cascade_forward(
    ml,
    span,
    Alpha,
    Ma2,
    Ys,
    htr=0.95,
    Po1=1e5,
    To1=300.0,
):
    """Calculate turbine cascade geometry from aerodynamic design variables."""

    # Check inputs
    assert ml.n_row == 1
    assert len(span) == 2
    assert len(Alpha) == 2

    # Get the inlet stagnation state
    inlet_stag = ml[0].copy().set_P_T(Po1, To1)
    V_ref = inlet_stag.a
    T_ref = inlet_stag.T
    s1 = inlet_stag.s
    ho1 = inlet_stag.h

    # Entropy from pseudo loss coefficient
    s2 = s1 + Ys * (0.5 * V_ref**2) / T_ref

    # Outlet state from known ho, s, Ma, angles
    ml[1].set_ho_s_Ma_Alpha_Beta(ho1, s2, Ma2, Alpha[1], Beta=0.0)

    # Set exit annulus geometry
    ml[1].set_span_htr(span[1], htr)

    # Now conserve mass to set inlet state
    rhoVx1 = ml[1].rhoVx * span[1] / span[0]
    ml[0].set_ho_s_rhoVm_Alpha_Beta(ho1, s1, rhoVx1, Alpha[0], Beta=0.0)

    # Set inlet annulus geometry (htr may vary, same r_mid)
    ml[0].set_span_r_mid(span[0], ml[1].r_mid)


@register_mean_line
def turbine_cascade_backward(ml):
    """Reverse a cascade mean-line to design variables.

    Parameters
    ----------
    ml: MeanLine
        A mean-line object specifying the flow in a cascade.

    Returns
    -------
    out : dict
        Dictionary of aerodynamic design parameters with fields:
        `span1`, `span2`, `Alpha1`, `Alpha2`, `Ma2`, `Yh`, `htr`, `RR`, `Beta`.
        The fields have the same meanings as in :func:`forward`.
    """

    # Pseudo loss coefficient
    V_ref = ml[0].a
    Ys = (ml[-1].s - ml[0].s) / ml[0].To * (0.5 * V_ref**2)

    out = {
        "span": ml.span,
        "Alpha": ml.Alpha,
        "Ma2": ml.Ma[1],
        "Ys": Ys,
        "htr": ml.htr[-1],
    }

    return out
