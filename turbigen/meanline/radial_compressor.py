"""Mean-line design of a radial impeller with vaneless diffuser"""
import numpy as np
import turbigen.flowfield
import turbigen.vtri
import turbigen.util


def forward(
    So1,
    PR_tt,
    eta_tt,
    mdot,
    phi1,
    Alpha1,
    Ma1_rel,
    htr1,
    Alpha2_rel,
    DH_rotor,
    DH_diff,
    VmR_diff,
    Ys,
    Beta1=0.0,
    Beta2=90.0,
    Beta3=90.0,
):
    """Design the mean-line for a radial compressor with vaneless diffuser."""

    MAXITER = 100

    # Calculate outlet state using duty and effy guess
    Po3 = So1.P * PR_tt
    So3s = So1.copy().set_P_s(Po3, So1.s)
    ho3 = So1.h + (So3s.h - So1.h) / eta_tt
    So3 = So1.copy().set_P_h(Po3, ho3)

    # We shall use the following notation for states
    # (1) impeller inlet
    # (2) impeller outlet
    # (3) diffuser outlet

    # # Precalculate some trig
    # tanAlpha1 = np.tan(np.radians(Alpha1))
    # cosAlpha1 = np.cos(np.radians(Alpha1))
    # tanAlpharel2 = np.tan(np.radians(Alpharel2))
    # tanBeta2 = np.tan(np.radians(Beta2))

    # Set rotor inlet state
    Maxrt1 = turbigen.vtri.resolve_rel_magnitude_abs_yaw(Ma1_rel, phi1, Alpha1, Beta1)
    Ma1 = turbigen.util.vecnorm(Maxrt1)
    S1 = So1.to_static(Ma1)
    Vxrt1 = S1.a * Maxrt1

    # Rotor inlet geometry using mdot, htr
    rrms1, A1, Omega = turbigen.vtri.annulus_geometry_from_flow(
        Vxrt1, mdot, S1.rho, phi1, htr1
    )
    U1 = rrms1 * Omega

    # Prescribe rotor DeHaller and angles to set rel frame velocities
    V1_rel = S1.a * Ma1_rel
    V2_rel = DH_rotor * V1_rel
    Vxrt2_rel = turbigen.vtri.resolve_magnitude(V2_rel, Alpha2_rel, Beta2)
    Vt2_rel = Vxrt2_rel[2]
    Vm2 = turbigen.util.vecnorm(Vxrt2_rel[:2])

    # Use a pseudo entropy loss coefficient to set rotor exit entropy
    s2 = So1.s + Ys / S1.T * (So1.h - S1.h)
    # Rotor exit has same total enthalpy as diffuser exit
    So2 = So1.copy().set_h_s(So3.h, s2)

    # Conserve rothalpy to set exit blade speed
    I1 = S1.h + 0.5 * (V1_rel**2.0 - U1**2.0)
    # No analytical solution so must iterate
    # Initial guesses
    h2 = So2.h
    U2 = U1
    converged = False
    rtol_U = 1e-4
    for _ in range(MAXITER):
        U2_new = np.sqrt(2.0 * (h2 + 0.5 * V2_rel**2 - I1))
        dU2 = np.abs(U2_new - U2)
        h2 = So2.h - 0.5 * (Vm2**2.0 + (Vt2_rel + U2_new) ** 2.0)
        U2 = U2_new
        if dU2 / U2_new < rtol_U:
            converged = True
            break
    if not converged:
        raise Exception("U2 iteration did not converge")

    # We now know rotor exit static state and absolute velocity at 2
    S2 = So2.copy().set_h_s(h2, So2.s)
    Vxrt2 = Vxrt2_rel.copy()
    Vxrt2[2] += U2
    Vt2 = Vxrt2[2]
    V2 = turbigen.util.vecnorm(Vxrt2)

    # Geometry of the outlet
    A2 = mdot / S2.rho / Vm2
    rrms2 = U2 / Omega

    # Prescribe diffuser velocities
    V3 = DH_diff * V2
    Vm3 = Vm2 * VmR_diff
    Vt3 = np.sqrt(V3**2 - Vm3**2)
    Vxrt3 = np.array(
        (
            Vm3 * turbigen.util.cosd(Beta3),
            Vm3 * turbigen.util.sind(Beta3),
            Vt3,
        )
    )

    # Diffuser exit state by cons of energy
    h3 = So2.h - 0.5 * V3**2
    S3 = So2.copy().set_h_s(h3, So3.s)

    # Select radius ratio for vaneless diffuser by
    # conserving mass and moment of momentum
    mom2 = rrms2 * S2.rho * Vt2
    DR_diff = S3.rho / S2.rho
    A3 = A2 / DR_diff / VmR_diff
    rrms3 = rrms2 / DR_diff * Vt2 / Vxrt3[2]

    # We now need a dummy state somewhere in the middle of the vaneless diffuser
    # Call this 2a
    rrms2a = rrms2 + 0.7 * (rrms3 - rrms2)
    A2a = A2 * rrms2a / rrms2

    # Guess density and iterate
    S2a = S2.copy()
    atol_h = 1e-4 * (So2.h - S2.h)
    converged = False
    for _ in range(MAXITER):
        Vm2a = mdot / S2a.rho / A2a
        Vt2a = mom2 / S2a.rho / rrms2a
        V2asq = Vm2a**2 + Vt2a**2
        h2a = So2.h - 0.5 * V2asq
        dh2a = np.abs(S2a.h - h2a)
        if dh2a < atol_h:
            converged = True
            break
        S2a.set_h_s(h2a, So2.s)
    if not converged:
        raise Exception("h2a iteration did not converge")

    Vxrt2a = np.array(
        (
            Vm2a * turbigen.util.cosd(Beta2),
            Vm2a * turbigen.util.sind(Beta2),
            Vt2a,
        )
    )

    S_all = S1.stack((S1, S2, S2a, S3))

    rrms_all = np.array([rrms1, rrms2, rrms2a, rrms3])
    A_all = np.array([A1, A2, A2a, A3])
    Omega_all = np.array([Omega, Omega, 0.0, 0.0])
    Vxrt = np.stack((Vxrt1, Vxrt2, Vxrt2a, Vxrt3), axis=-1)

    ml = turbigen.flowfield.make_mean_line(rrms_all, A_all, Omega_all, Vxrt, S_all)

    return ml


def inverse(ml):
    """Reverse a radial compressor mean-line to design variables."""

    # Generalised HTR
    K = ml.A[0] / 4.0 / np.pi / ml.rmid[0] ** 2
    htr = (1.0 - K) / (1.0 + K)

    out = {
        "So1": ml.stagnation[0],
        "PR_tt": ml.PR_tt,
        "eta_tt": ml.eta_tt,
        "mdot": ml.mdot[0],
        "phi1": ml.phi[0],
        "Alpha1": ml.Alpha[0],
        "Ma1_rel": ml.Ma_rel[0],
        "htr1": htr,
        "Alpha2_rel": ml.Alpha_rel[1],
        "DH_rotor": ml.V_rel[1] / ml.V_rel[0],
        "DH_diff": ml.V[-1] / ml.V[1],
        "VmR_diff": ml.Vm[3] / ml.Vm[1],
        "Ys": (ml.s[2] - ml.s[0]) * ml.T[0] / (ml.ho[0] - ml.h[0]),
        "Beta1": ml.Beta[0],
        "Beta2": ml.Beta[1],
        "Beta3": ml.Beta[3],
    }
    return out
