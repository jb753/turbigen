"""Define a custom mean-line plugin."""

# Need this decorator so that turbigen can find your functions
from turbigen.plugins import register_mean_line


@register_mean_line
def custom_forward(
    ml,
    span,
    Alpha,
    Ma2,
    Ys,
    htr,
    To1=300.0,
    Po1=1e5,
):
    """Calculate turbine cascade geometry from aerodynamic design variables."""

    # We need to set thermodynamic state and velocity at all stations, and
    # two geometry parameters from span, htr, r_mid
    # ml is a MeanLine object with n_row * stations
    # so ml[0] is the inlet station, ml[1] is the outlet station
    # We can manipulate the state by doing simply
    # ml[0].set_P_T(P, T) for thermodynamics
    # Or
    # ml[1].set_Vxrt(Vx, Vr, Vt) for velocity components
    # Or use the more complex setters to set all 5 at once,
    # ml[1].set_ho_s_Ma_Alpha_Beta(ho1, s2, Ma2, Alpha[1], Beta=0.0)
    # Which sets stagnation enthalpy, entropy, Mach number, flow angles

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

    # Can print the mean-line to check
    print("Vx=", ml.Vx)


@register_mean_line
def custom_backward(ml):
    """Reverse a mean-line flowfield back to design variables.

    Parameters
    ----------
    ml: MeanLine
        A mean-line object specifying the flow in a cascade.

    Returns
    -------
    out : dict
        Dictionary of aerodynamic design parameters with same
        fields as args to :func:`custom_forward`,
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
        "To1": ml[0].To,
        "Po1": ml[0].Po,
    }

    return out
