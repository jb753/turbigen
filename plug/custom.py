"""Define a custom mean-line plugin."""

# Need this decorator so that turbigen can find your functions
from turbigen.plugins import register_mean_line
import numpy as np


@register_mean_line
def custom_forward(
    ml,
    span,
    htr,
    phi,
    psi,
    Lam,
    Ma1_rel=0.6,
    To1=300.0,
    Po1=1e5,
    Ds=0.0,
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
    # assert len(Alpha) == 2

    # Use inlet conditions to evaluate EOS for state (01)
    fluid = ml[0].fluid
    rhoo1, uo1 = fluid.set_P_T(Po1, To1)
    ho1 = fluid.get_h(rhoo1, uo1)
    s1 = fluid.get_s(rhoo1, uo1)

    # Get flow angles from phi, psi, Lam assuming repeating stage
    Alpha1 = np.arctan((1 - Lam - psi / 2) / phi)
    Alpha2 = np.arctan(psi / phi + np.tan(Alpha1))
    Alpha_rel1 = np.arctan(np.tan(Alpha1) - 1 / phi)

    # We are going to iterate Vx1 to satisfy Ma1_rel and Alpha1
    # Initial guess for static state (1)
    ml[0].set_rho_s(rhoo1, s1)  # initial guess

    for _ in range(20):
        # Use current static state to eval a and hence V1_rel
        V1_rel = Ma1_rel * ml[0].a

        Vx1 = V1_rel * np.cos(Alpha_rel1)
        V1 = Vx1 / np.cos(Alpha1)

        # With V1 abs, set new guess for static state (1)
        h1 = ho1 - 0.5 * V1**2
        ml[0].set_h_s(h1, s1)

    # Apply velocities
    Vx = V1 * np.cos(Alpha1)
    Vt1 = V1 * np.sin(Alpha1)
    Vt2 = Vx * np.tan(Alpha2)
    ml[0].set_Vxrt(Vx, 0.0, Vt1)
    ml[1].set_Vxrt(Vx, 0.0, Vt2)

    # Set annulus geometry
    ml[0].set_span_htr(span, htr)

    # Find exit static enthalpy from psi and V2
    U = Vx / phi
    ho2 = ho1 + psi * U**2
    V2 = Vx / np.cos(Alpha2)
    h2 = ho2 - 0.5 * V2**2
    ml[1].set_h_s(h2, s1 + Ds)

    # Use cons of mass to get exit span
    span2 = span * ml[0].rhoVx / ml[1].rhoVx
    # Set inlet annulus geometry (htr may vary, same r_mid)
    ml[1].set_span_r_mid(span2, ml[0].r_mid)

    Omega = U / ml[0].r_mid
    ml.set_Omega(Omega)


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

    # # Pseudo loss coefficient
    # V_ref = ml[0].a
    # Ys = (ml[-1].s - ml[0].s) / ml[0].To * (0.5 * V_ref**2)

    out = {
        "span": ml.span[0],
        "htr": ml.htr[0],
        "phi": ml[0].Vx / ml[0].U,
        "psi": (ml[-1].ho - ml[0].ho) / (ml[0].U ** 2),
        "Lam": None,
        "Ma1": ml.Ma[0],
        "To1": ml[0].To,
        "Po1": ml[0].Po,
        "Ma1_rel": ml[0].Ma_rel,
    }

    return out
