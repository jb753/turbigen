import numpy as np
from turbigen import util
import turbigen.flowfield
from scipy.optimize import brentq


def _Vxr_from_Vm_Beta(Vm, Beta):

    tansqBeta = np.tan(np.radians(Beta)) ** 2

    # Branch on pitch angle to avoid infinities
    if np.abs(Beta) < 45.0:
        # Mostly axial, Vx~= 0, tan Beta ~= inf
        Vx = Vm / np.sqrt(1.0 + tansqBeta)
        Vr = np.sqrt(Vm**2 - Vx**2)
    else:
        # Mostly radial, Vr~=0, tan Beta ~= 0
        Vr = Vm / np.sqrt(1.0 + 1.0 / tansqBeta)
        Vx = np.sqrt(Vm**2 - Vr**2)

    return Vx, Vr


def _solve_static(So, mdot, A, Alpha, Beta):
    """Find static conditions for a given upstream stagnation, mass flow and area."""

    tanBeta = util.tand(Beta)
    tansqBeta = tanBeta**2.0
    tanAlpha = util.tand(Alpha)
    denom = np.sqrt(tanAlpha**2.0 + 1.0)
    S = So.copy()

    # Guess an enthalpy and iterate
    def _iter_h(h):
        V = np.sqrt(2.0 * (So.h - h))
        Vm = V / denom
        S.set_h_s(h, So.s)
        mdot_now = S.rho * A * Vm
        mdot_err = mdot_now / mdot - 1.0
        return mdot_err

    Mamax = 1.2
    Vmax = So.a * Mamax
    hlow = So.h - 0.5 * Vmax**2
    hhigh = So.h
    brentq(_iter_h, hlow, hhigh)

    # Recalculate velocity components
    V = np.sqrt(2.0 * (So.h - S.h))
    Vm = V / denom

    # Branch on pitch angle to avoid infinities
    if np.abs(Beta) < 45.0:
        # Mostly axial, Vx~= 0, tan Beta ~= inf
        Vx = Vm / np.sqrt(1.0 + tansqBeta)
        Vr = np.sqrt(Vm**2 - Vx**2)
    else:
        # Mostly radial, Vr~=0, tan Beta ~= 0
        Vr = Vm / np.sqrt(1.0 + 1.0 / tansqBeta)
        Vx = np.sqrt(Vm**2 - Vr**2)

    # Ensure correct sign of radial velocity
    # We assume Vx is always going +ve
    if Beta < 0.0 and Vr > 0.0:
        Vr *= -1.0

    Vt = tanAlpha * Vm
    Vxrt = np.array([Vx, Vr, Vt])

    return S, Vxrt


def forward(So1, rh, rt, rpm, mdot, Alpha_rel, Beta):
    r""" """

    # We want the length of input data to be a multiple of two
    # An inlet and an exit for each blade row
    N = len(rh)
    assert np.mod(N, 2) == 0
    nrow = N // 2

    util.check_scalar(mdot=mdot)

    util.check_vector(
        (N,),
        rh=rh,
        rt=rt,
        rpm=rpm,
        Alpha_rel=Alpha_rel,
        Beta=Beta,
    )

    # Make input data arrays
    rh = np.array(rh).reshape((N,))
    rt = np.array(rt).reshape((N,))
    rpm = np.array(rpm).reshape((N,))
    Alpha = np.array(Alpha_rel).reshape((N,))
    Beta = np.array(Beta).reshape((N,))

    # Get mean radii and annulus areas
    rrms = np.sqrt(0.5 * (rh**2 + rt**2))
    A = np.pi * (rt**2 - rh**2)

    # Blade speeds
    Omega = rpm / 60.0 * 2.0 * np.pi
    U = Omega * rrms

    # Preallocate thermodynamic states and velocity vectors
    S = So1.empty(shape=(N,))
    Vxrt = np.empty((3, N))

    # Find the inlet static state first
    Sin, Vxrt[:, 0] = _solve_static(So1, mdot, A[0], Alpha[0], Beta[0])

    # Use inlet static as initial guess for all stations
    S = [
        Sin,
    ]

    # Loop over rows
    for irow in range(nrow):
        ist = irow * 2
        ien = ist + 1

        # We always know the row inlet velocity vector and state
        # Outlet is an initial guess

        # Pull out thermodynamic states for inlet and exit of this row
        S1 = S[ist]
        S2 = S1.copy()

        # Get velocity components in relative frame
        U1 = U[ist]
        U2 = U[ien]
        Vt1rel = Vxrt[2, ist] - U1
        Vm1 = np.sqrt(np.sum(Vxrt[:2, ist] ** 2))
        V1rel = np.sqrt(Vm1**2 + Vt1rel**2)

        tanAlrel2 = np.tan(np.radians(Alpha[ien]))

        # Rothalpy at inlet
        I1 = S1.h + 0.5 * (V1rel**2 - U1**2)

        # Loop to converge exit density
        for _ in range(50):

            # Set exit meridional velocity by cons of mass
            Vm2 = mdot / S2.rho / A[ien]

            # Use rothalpy to set exit enthalpy
            h2 = I1 - 0.5 * (Vm2**2) * (1 + tanAlrel2**2) + 0.5 * U1**2

            # Update density assuming lossless
            S2.set_h_s(h2, S2.s)

        S.append(S2)

        # Resolve the velocity components
        Vt2rel = Vm2 * tanAlrel2
        Vt2 = Vt2rel + U2
        Vxrt[:2, ien] = _Vxr_from_Vm_Beta(Vm2, Beta[ien])
        Vxrt[2, ien] = Vt2

        # If not the last blade row, deal with the gap
        if irow < (nrow - 1):

            S2a = S2.copy()

            tanAlrel2a = np.tan(np.radians(Alpha[ien + 1]))

            # Loop to converge exit density
            for _ in range(10):

                # Set exit meridional velocity by cons of mass
                Vm2a = mdot / S2a.rho / A[ien + 1]

                # Use yaw to get rel tangential
                Vt2arel = tanAlrel2a * Vm2a
                Vt2a = Vt2arel + U[ien + 1]
                V2a = np.sqrt(Vt2a**2 + Vm2a**2)

                # Use total enthalpy to set exit enthalpy
                ho2 = S2.h + 0.5 * (Vt2**2 + Vm2**2)
                h2a = ho2 - 0.5 * V2a**2

                # Update density assuming lossless
                S2a.set_h_s(h2a, S2a.s)

                Vxrt[:2, ien + 1] = _Vxr_from_Vm_Beta(Vm2a, Beta[ien + 1])
                Vxrt[2, ien + 1] = Vt2a

            S.append(S2a)

    return turbigen.flowfield.make_mean_line(rrms, A, Omega, Vxrt, So1.stack(S))


def inverse(ml):

    out = {
        "So1": ml.stagnation[0],
        "rh": ml.rhub.tolist(),
        "rt": ml.rtip.tolist(),
        "rpm": ml.rpm.tolist(),
        "mdot": ml.mdot.mean(),
        "Alpha_rel": ml.Alpha_rel.tolist(),
        "Beta": ml.Beta.tolist(),
    }

    return out
