"""Mean-line design of a radial impeller with vaneless diffuser"""
import numpy as np
import turbigen.flowfield
import turbigen.vtri
from turbigen import util


def forward(
    So1,
    # Duty
    PR_tt,
    mdot,
    # Kinematic
    phi1,
    Ma1_rel,
    DH,
    VmR,
    # Geometric
    htr1a,
    RR_indu,
    Beta,
    # Loss
    Ys,
):
    """Design the mean-line for a radial compressor with vaneless diffuser."""

    # Verify input scalars
    util.check_scalar(
        PR_tt=PR_tt,
        mdot=mdot,
        phi1=phi1,
        Ma1_rel=Ma1_rel,
        htr1a=htr1a,
        RR_indu=RR_indu,
    )

    # Check shapes of vectors
    util.check_vector((4,), Ys=Ys, DH=DH, Beta=Beta)
    util.check_vector((3,), VmR=VmR)

    MAXITER = 100

    # We shall use the following notation for states
    # (1a) inducer inlet
    # (1b) inducer outlet
    # (1)  rotor inlet
    # (2)  rotor outlet
    # (3)  diffuser inlet
    # (4)  diffuser outlet

    # Assume inducer inlet is axial
    Beta1a = 0.0

    # Use pseudo entropy loss coefficient to guess entropy
    # throughout the machine (update later based on CFD solution)
    # Ys = To1*(s-s1)/(0.5*a01^2)
    s = np.concatenate(((0.0, 0.0), Ys)) * (0.5 * So1.a**2) / So1.T + So1.s

    # Calculate work using duty and loss guess
    Po3 = So1.P * PR_tt
    So3 = So1.copy().set_P_s(Po3, s[-1])

    # We can now define all stagnation states
    ho3 = So3.h
    ho1 = So1.h
    ho = np.array([ho1, ho1, ho1, ho3, ho3, ho3])
    So = So1.empty(shape=(6,)).set_h_s(ho, s)

    # We need a velocity scale. Use the inlet relative Mach.
    # But don't know static states yet
    # Plan is to guess static states, calc velocity everywhere
    # Then update static states based on known stagnation
    S = So.copy()
    hguess = ho - 0.5 * (So1.a) ** 2
    S.set_h_s(hguess, s)
    atol_h = 1e-6 * 0.5 * So1.a**2
    converged = False
    for _ in range(MAXITER):

        # Speed of sound sets relative velocity magnitude
        V1_rel = S.a[2] * Ma1_rel

        # Trigonometry for the rotor inlet velocity
        Vm1 = V1_rel * (1.0 + (phi1**-2)) ** -0.5

        # Flow coefficient sets blade speed at rotor LE
        U1 = Vm1 / phi1

        # Inducer inlet velocity from de Haller
        # Assuming no inlet swirl and Vt1a = Vt1 = 0
        Vm1a = Vm1 / DH[0]

        # Rotor exit rel velocity from prescribed de Haller
        V2_rel = DH[1] * V1_rel
        Vm2 = VmR[0] * Vm1
        Vt2_rel = -np.sqrt(V2_rel**2 - Vm2**2)

        # Conserve rothalpy to set exit blade speed
        I1 = S.h[2] + 0.5 * (V1_rel**2.0 - U1**2.0)

        # No analytical solution for rotor exit
        # Must iterate from initial guesses
        h2 = S.h[3]
        ho2 = So.h[3]
        U2 = U1
        converged = False
        rtol_U = 1e-4
        for _ in range(MAXITER):
            U2_new = np.sqrt(2.0 * (h2 + 0.5 * V2_rel**2 - I1))
            dU2 = np.abs(U2_new - U2)
            h2 = ho2 - 0.5 * (Vm2**2.0 + (Vt2_rel + U2_new) ** 2.0)
            U2 = U2_new
            if dU2 / U2_new < rtol_U:
                converged = True
                break
        if not converged:
            raise Exception("U2 iteration did not converge")

        # Absolute velocity at rotor exit
        Vt2 = Vt2_rel + U2
        V2 = np.sqrt(Vt2**2 + Vm2**2)

        # Prescribe diffuser velocities by de Haller
        V3 = V2 * DH[2]
        V4 = V3 * DH[3]

        # Assemble all the velocities and static enthalpy
        V = np.array([Vm1a, Vm1, Vm1, V2, V3, V4])
        h_new = ho - 0.5 * V**2

        # Convergence check
        dh = h_new - S.h
        if dh.max() < atol_h:
            converged = True
            break

        # Update static state
        S.set_h_s(S.h + 0.1 * dh, s)

    if not converged:
        raise Exception("Iteration for h did not converge")

    # Kinematic design is done
    # We now need to set the flow areas and radii to achieve these velocities

    # Inducer inlet radius from hub to tip ratio
    A1a = mdot / Vm1a / S.rho[0]
    rrms1a = turbigen.vtri.solve_rrms(A1a, htr1a, Beta1a)

    # Prescribed inducer radius ratio for rotor inlet
    rrms1 = RR_indu * rrms1a
    A1 = mdot / Vm1 / S.rho[2]

    # Can now set Omega and r2 using blade speeds
    Omega = U1 / rrms1
    rrms2 = U2 / Omega
    A2 = mdot / Vm2 / S.rho[3]

    # Set radius by conserving angular momentum across gap
    # Set area by conserving mass
    Vm3 = Vm2 * VmR[1]
    Vt3 = np.sqrt(V3**2 - Vm3**2)
    mom = rrms2 * Vt2
    rrms3 = mom / Vt3
    assert rrms3 > rrms2
    A3 = mdot / Vm3 / S.rho[4]

    Vm4 = Vm3 * VmR[2]
    Vt4 = np.sqrt(V4**2 - Vm4**2)
    mom = rrms3 * Vt3
    rrms4 = mom / Vt4
    assert rrms4 > rrms3
    A4 = mdot / Vm4 / S.rho[5]

    Vm = np.array([Vm1a, Vm1a, Vm1, Vm2, Vm3, Vm4])
    Vt = np.array([0.0, 0.0, 0.0, Vt2, Vt3, Vt4])
    Beta_all = np.array([Beta1a, Beta1a, Beta[0], Beta[1], Beta[2], Beta[3]])
    Vx = Vm * util.cosd(Beta_all)
    Vr = Vm * util.sind(Beta_all)

    rrms_all = np.array([rrms1a, rrms1a, rrms1, rrms2, rrms3, rrms4])
    A_all = np.array([A1a, A1a, A1, A2, A3, A4])
    Omega_all = np.array([0.0, 0.0, Omega, Omega, 0.0, 0.0])
    Vxrt = np.stack((Vx, Vr, Vt))

    ml = turbigen.flowfield.make_mean_line(rrms_all, A_all, Omega_all, Vxrt, S)

    return ml


def inverse(ml):
    """Reverse a radial compressor mean-line to design variables."""

    # Generalised HTR
    K = ml.A[0] / 4.0 / np.pi / ml.rmid[0] ** 2
    htr = (1.0 - K) / (1.0 + K)

    DH = (ml.V[1:] / ml.V[:-1])[1:]
    DH[1] = ml.V_rel[3] / ml.V_rel[2]

    VmR = (ml.Vm[1:] / ml.Vm[:-1])[2:]

    out = {
        "So1": ml.stagnation[0],
        "PR_tt": ml.PR_tt,
        "mdot": ml.mdot[0],
        "phi1": ml.phi[2],
        "Ma1_rel": ml.Ma_rel[2],
        "htr1a": htr,
        "DH": DH.tolist(),
        "VmR": VmR.tolist(),
        "Ys": ((ml.s[2:] - ml.s[0]) * ml.To[0] / (0.5 * ml.ao[0] ** 2)).tolist(),
        "Beta": ml.Beta[2:].tolist(),
        "RR_indu": ml.rrms[2] / ml.rrms[1],
    }
    return out
