import numpy as np
from turbigen import util
import turbigen.flowfield
from scipy.optimize import brentq


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


def forward(So1, rh, rt, Omega, mdot, Alpha, Beta):
    r""" """

    nrow = len(rh)

    # Get mean radii and annulus areas
    rrms = np.sqrt(0.5 * (rh**2 + rt**2))
    A = np.pi * (rh**2 - rt**2)

    # Get inlet static conditions
    S1, Vxrt1 = _solve_static(So1, mdot, A[0], Alpha[0], Beta[0])

    S = So1.empty(shape=(nrow * 2,))

    # ml = turbigen.flowfield.make_mean_line(rrms, A, Omega, Vxrt, S)

    # # Check mass, energy conserved
    # assert np.ptp(ml.mdot) < ml.mdot[0] * 1e-3
    # assert np.ptp(ml.ho) / np.mean(ml.cp) < 0.1

    return ml


def inverse(ml):
    """Reverse a cascade mean-line to design variables.

    Parameters
    ----------
    ml: MeanLine
        A mean-line object specifying the flow in a cascade.

    Returns
    -------
    out : dict
        Dictionary of aerodynamic design parameters with fields: `So1`,
        `span1`, `span2`, `Alpha1`, `Alpha2`, `Ma2`, `Yh`, `htr`, `RR`, `Beta`.
        The fields have the same meanings as in :func:`forward`.
    """
    # Pull out states
    S2s = ml.empty().set_P_s(ml.P[-1], ml.s[0])

    # Loss coefficient
    horef = ml.ho[0]
    if ml.ARflow[0] >= 1.0:
        # Compressor
        href = ml.h[0]
    else:
        # Turbine
        href = ml.h[1]
    Yh_out = (ml.h[1] - S2s.h) / (horef - href)

    out = {
        "So1": ml.stagnation[0],
        "span1": ml.span[0],
        "span2": ml.span[1],
        "Alpha1": ml.Alpha[0],
        "Alpha2": ml.Alpha[1],
        "Ma2": ml.Ma[1],
        "Yh": Yh_out,
        "htr": ml.htr[0],
        "RR": ml.RR[0],
        "Beta": ml.Beta.tolist(),
    }

    return out
