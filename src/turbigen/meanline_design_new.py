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


@register_mean_line
def axial_turbine_forward(
    ml,
    psi,
    phi2,
    Ma2,
    fac_Ma3_rel,
    mdot,
    Ys,
    r_rms,
    zeta=(1.0, 1.0),
    Po1=1e5,
    To1=300.0,
):
    # Calculate some reference values for the iteration
    # Stagnation inlet speed of sound as reference velocity
    rhoo1, uo1 = ml.fluid.set_P_T(Po1, To1)
    ao1 = ml.fluid.get_a(rhoo1, uo1)
    s1 = ml.fluid.get_s(rhoo1, uo1)
    ho1 = ml.fluid.get_h(rhoo1, uo1)

    # Use pseudo entropy loss coefficient to set entropy
    # throughout the machine (update later based on CFD solution)
    # This is fixed during iteration for Alpha1 and U
    # Ys = To1*(s-s1)/(0.5*a01^2)
    dhead_ref = 0.5 * ao1**2
    s = np.concatenate(((0.0,), (Ys[0],), Ys)) * dhead_ref / To1 + s1

    # Define rotor Mach as offset from stator Mach
    Ma3_rel = fac_Ma3_rel * Ma2

    # Guesses for Alpha1 and blade speed U
    U_guess = ao1 * Ma2 * 0.5
    Alpha1 = 0.0
    Alpha3 = np.nan
    atol_Alpha = 0.1

    # Intialise the mean-line guess
    ml.set_h_s(ho1, s)
    ml.set_r_rms(r_rms)
    ml.set_Vr(0.0)
    ml.set_Omega(U_guess / r_rms * np.array([0, 0, 1, 1]))

    # Closure function to iterate U for a fixed Alpha1
    # Takes vars from outer scope
    def iter_U(
        Alpha1,
    ):
        # Preallocate and loop
        rf = 0.5
        conv_U = False
        for _ in range(500):
            #
            # Extract current blade speed
            U = ml.U[-1]
            #
            # Axial velocities
            Vx = np.array([zeta[0], 1.0, 1.0, zeta[1]]) * U * phi2

            # Inlet flow angle sets inlet tangential velocity
            Vt1 = Vx[0] * np.tan(np.radians(Alpha1))

            # Stator exit velocity from Mach
            V2 = Ma2 * ml.a[1]
            assert V2 > Vx[1]
            Vt2 = np.sqrt(V2**2 - Vx[1] ** 2)

            # Rotor exit relative velocity from rel Mach
            V3_rel = Ma3_rel * ml.a[3]
            Vt3_rel = -np.sqrt(V3_rel**2 - Vx[3] ** 2)
            Vt3 = Vt3_rel + U

            # Stagnation enthalpy using Euler work equation
            Vt = np.array([Vt1, Vt2, Vt2, Vt3])
            ho3 = ho1 + U * (Vt3 - Vt2)
            ho = np.array([ho1, ho1, ho1, ho3])
            h = ho - 0.5 * (Vx**2 + Vt**2)

            # Update the states
            ml.set_h_s(h, s)

            # New guess for blade speed
            U_new = np.sqrt((ho1 - ho3) / psi)

            # Check convergence
            dU = U_new - U
            err_rel_U = np.abs(dU) / U
            if err_rel_U < 1e-4:
                ml.set_Omega(U_new / r_rms * np.array([0, 0, 1, 1]))
                conv_U = True
                break
            else:
                U = U_new * rf + U * (1.0 - rf)
                ml.set_Omega(U / r_rms * np.array([0, 0, 1, 1]))

        if not conv_U:
            raise ValueError(f"U iteration did not converge: {U} -> {U_new}")

        # Conservation of mass to get areas
        ml.set_Am(mdot / ml.rho / Vx)

        # Assemble velocity components
        ml.set_Vx(Vx)
        ml.set_Vt(Vt)

        return ml.Alpha[3]

    # Iterate on Alpha1 to match the desired Ma2 and fac_Ma3_rel
    converged = False
    for _ in range(100):
        # Calculate and store next trial in ml
        Alpha3 = iter_U(Alpha1)
        err = np.abs(Alpha3 - Alpha1)

        if err < atol_Alpha:
            converged = True
            break
        else:
            Alpha1 = Alpha3

    if not converged:
        raise ValueError(f"Alpha1 iteration did not converge: {Alpha1} -> {Alpha3}")


@register_mean_line
def axial_turbine_backward(ml):
    """Reverse a turbine stage mean-line to design variables.

    Parameters
    ----------
    mean_line: MeanLine
        A mean-line object specifying the flow in an axial turbine.

    Returns
    -------
    out : dict
        Dictionary of aerodynamic design parameters with fields:
            - So1: State
            - PRtt: float
            - psi: float
            - phi2: float
            - zeta: float
            - Ma2: float
            - DMa3_rel: float
            - Alpha1: float
            - mdot: float
            - Ys: (float, float, float)

    """

    U2 = ml.U[2]
    Vx2 = ml.Vx[2]
    Ma2 = ml.Ma[2]

    # Calculate pseudo entropy loss coefficient
    Tref = ml.To[0]
    dhead_ref = 0.5 * ml.ao[0] ** 2
    sref = ml.s[0]
    s = ml.s[
        (1, 3),
    ]
    Ys = (s - sref) * Tref / dhead_ref

    # Calculate axial velocity ratios
    zeta = (
        ml.Vx[
            (0, 3),
        ]
        / Vx2
    )

    # Reaction
    h = ml.h
    Lam = (h[1] - h[0]) / (h[3] - h[0])

    phi2 = Vx2 / U2

    # Assemble the dict
    out = {
        "PR_ts": ml.PR_ts,
        "PR_tt": ml.PR_tt,
        "psi": (ml.ho[0] - ml.ho[3]) / U2**2,
        "phi2": phi2,
        "zeta": zeta,
        "Ma2": Ma2,
        "fac_Ma3_rel": ml.Ma_rel[3] / ml.Ma[1],
        "Ma3_rel": ml.Ma_rel[3],
        "Alpha1": ml.Alpha[0],
        "mdot": ml.mdot[0],
        "Lam": Lam,
        "Ys": np.array(Ys),
        "htr2": ml.htr[1],
        "r_rms": ml.r_rms[0],
        "eta_tt": ml.eta_tt,
        "eta_ts": ml.eta_ts,
        "Omega": ml.Omega[-1],
        "Po1": ml.Po[0],
        "To1": ml.To[0],
    }

    return out
