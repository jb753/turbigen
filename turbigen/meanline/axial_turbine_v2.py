"""Generalised axial turbine mean-line design."""

from turbigen import util
import turbigen.flowfield
import turbigen.fluid
import numpy as np
from scipy.optimize import fsolve, bisect

logger = util.make_logger()


def forward(
    So1,
    PR_tt,
    phi2,
    zeta,
    Ma2,
    DMa3_rel,
    Alpha1,
    mdot,
    Ys,
    htr2,
):
    r"""Design the mean-line for an axial turbine stage.

    Parameters
    ----------
    So1: State
        Object specifing the working fluid and its state at inlet.


    Returns
    -------
    ml: MeanLine
        An object specifying the flow along the mean line.

    """

    # Verify input scalars
    util.check_scalar(
        PR_tt=PR_tt,
        phi2=phi2,
        Ma2=Ma2,
        DMa3_rel=DMa3_rel,
        Alpha1=Alpha1,
        mdot=mdot,
    )

    # Check shapes of vectors
    util.check_vector((2,), zeta=zeta, Ys=Ys)

    # Use pseudo entropy loss coefficient to guess entropy
    # throughout the machine (update later based on CFD solution)
    Tref = So1.T
    dhead_ref = 0.5 * So1.a**2
    # Ys = To1*(s-s1)/(0.5*a01^2)
    s = np.concatenate(((0.0,), (Ys[0],), Ys)) * dhead_ref / Tref + So1.s

    # Use pressure ratio to get exit stagnation state
    So3 = So1.copy().set_P_s(So1.P * PR_tt, s)

    # Can use enthalpy and entropy to fix all stagnation states
    ho = np.array([So1.h, So1.h, So1.h, So3.h])
    So = So1.empty(shape=(4,)).set_h_s(ho, s)

    # Define rotor Mach as offset from stator Mach
    Ma3_rel = DMa3_rel + Ma2

    # Euler work equation to get U
    U = So1.a * 0.5

    # Preallocate and loop
    S = So.copy()
    MAXITER = 100
    RTOL = 1e-6
    for i in range(MAXITER):
        # Axial velocities
        Vx2 = U * phi2
        Vx = np.array([zeta[0], 1.0, 1.0, zeta[1]]) * Vx2

        # Inlet flow angle sets inlet tangential velocity
        Vt1 = Vx[0] * np.tan(np.radians(Alpha1))

        # Stator exit velocity from Mach
        V2 = Ma2 * S.a[1]
        assert V2 > Vx2
        Vt2 = np.sqrt(V2**2 - Vx2**2)

        # Rotor exit relative velocity from rel Mach
        V3_rel = Ma3_rel * S.a[3]
        Vt3_rel = -np.sqrt(V3_rel**2 - Vx[3] ** 2)
        Vt3 = Vt3_rel + U

        # Stagnation enthalpy using Euler work equation
        Vt = np.array([Vt1, Vt2, Vt2, Vt3])
        ho1 = ho2 = So.h[0]
        ho3 = ho2 + U * (Vt3 - Vt2)
        ho = np.array([ho1, ho2, ho2, ho3])
        h = ho - 0.5 * (Vx**2 + Vt**2)

        # Update the states
        So.set_h_s(ho, s)
        S.set_h_s(h, s)

        # New guess for blade speed
        Unew = np.sqrt((ho1 - ho3) / psi)

        # Check convergence
        dU = Unew - U
        if np.abs(dU) < RTOL * U:
            print("breaking")
            break
        else:
            U = Unew

    # Conservation of mass to get areas
    A = mdot / S.rho / Vx

    # Mean radius from hub-to-tip ratio
    rrms2 = A[1] / 2 / np.pi * (1 + htr2**2) / (1 - htr2**2)
    rrms = np.full((4,), rrms2)

    # Angular velocity
    Omega = U / rrms * np.array([0, 0, 1, 1])

    # Assemble velocity components
    Vxrt = np.stack((Vx, np.zeros_like(Vx), Vt))

    ml = turbigen.flowfield.make_mean_line(rrms, A, Omega, Vxrt, S)

    return ml


def inverse(ml):
    """Extract design parameters from a turbine mean-line object.

    Parameters
    ----------
    ml: MeanLine
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
    Ys = (
        (
            ml.s[
                (1, 3),
            ]
            - ml.s[0]
        )
        * Tref
        / dhead_ref
    )

    # Calculate axial velocity ratios
    zeta = (
        ml.Vx[
            (0, 3),
        ]
        / Vx2
    )

    Lam = (ml.h[3] - ml.h[2]) / (ml.h[3] - ml.h[0])
    print(Lam)

    # Assemble the dict
    out = {
        "So1": ml.stagnation[0],
        "PR_tt": ml.PR_tt,
        "psi": (ml.ho[0] - ml.ho[3]) / U2**2,
        "phi2": Vx2 / U2,
        "zeta": zeta,
        "Ma2": Ma2,
        "DMa3_rel": ml.Ma_rel[3] - Ma2,
        "Alpha1": ml.Alpha[0],
        "mdot": ml.mdot[0],
        "Ys": tuple(Ys),
        "htr2": ml.htr[1],
        "MaU": U2 / ml.a[1],
        "Lam": Lam,
    }
    return out
