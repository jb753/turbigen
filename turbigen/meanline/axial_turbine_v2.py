"""Generalised axial turbine mean-line design."""

from turbigen import util
import turbigen.flowfield
import turbigen.fluid
import numpy as np
from scipy.optimize import fsolve

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
    rrms,
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

    # Calculate work using duty and loss guess
    Po3 = So1.P / PR_tt
    So3 = So1.copy().set_P_s(Po3, s[-1])

    # We can now define all stagnation states
    ho3 = So3.h
    ho1 = So1.h
    ho = np.array([ho1, ho1, ho1, ho3])
    So = So1.empty(shape=(4,)).set_h_s(ho, s)

    # Rotor Mach is defined by an offset to stator Mach
    Ma3_rel = DMa3_rel + Ma2
    logger.debug(f"Ma3_rel={Ma3_rel}")

    # Guess a blade speed
    Uguess = So.a[1] * 0.5
    # Guess static states
    # Only used for acoustic speed, does not need to be accurate
    h = ho - 0.5 * Uguess**2
    S = So.empty(shape=(4,)).set_h_s(h, s)
    Vx = np.zeros((4,))
    Vt = np.zeros((4,))

    def eval_U(U):
        """Get axial velocity error as function of U."""
        # Now we iterate to converge U and static states
        MAXITER = 100
        RTOL = 1e-6
        alast = np.inf
        U = U.item()
        logger.debug(f"Solving for static states at U={U}....")
        for i in range(MAXITER):
            # Use flow coefficient to get Vx2
            Vx2 = U * phi2

            # Axial velocity ratio for inlet Vx
            Vx1 = zeta[0] * Vx2

            # Inlet flow angle sets inlet tangential velocity
            Vt1 = Vx1 * np.tan(np.radians(Alpha1))

            # Stator exit velocity from Mach
            V2 = Ma2 * S.a[1]
            assert V2 > Vx2
            Vt2 = np.sqrt(V2**2 - Vx2**2)

            # Rotor exit relative velocity from rel Mach
            V3_rel = Ma3_rel * S.a[3]

            # Rotor exit tangential velocity from Euler work equation
            Vt3 = Vt2 + (ho3 - ho1) / U
            Vt3_rel = Vt3 - U
            if np.abs(Vt3_rel) > V3_rel:
                raise ValueError(
                    "Rotor Ma3_rel too low: increase DMa3_rel or reduce PR_tt"
                )

            # Rotor exit axial velocity
            Vx3 = np.sqrt(V3_rel**2 - Vt3_rel**2)

            # Update static states
            Vx[:] = np.array([Vx1, Vx2, Vx2, Vx3]).reshape(-1)
            Vt[:] = np.array([Vt1, Vt2, Vt2, Vt3]).reshape(-1)
            V = np.sqrt(Vx**2 + Vt**2)
            S.set_h_s(ho - 0.5 * V**2, s)

            # Check sound speed error
            da = np.abs(S.a[1] / alast - 1.0)
            if da < RTOL:
                logger.debug(f"a converged on iteration {i}")
                break
            else:
                alast = S.a[1]
                logger.debug(f"updating new a={alast}")

        # Calculate error wrt target axial velocity ratio at rotor exit
        return Vx[-1] / Vx[-2] - zeta[1]

    # Solve for U
    U = fsolve(eval_U, x0=Uguess)[0]

    # The kinematic design is complete
    # Now we must calculate the radii

    # Conservation of mass to get areas
    A = mdot / S.rho / Vx

    # Mean radius from hub-to-tip ratio
    # rrms2 = A[1] / 2 / np.pi * (1 + htr2**2) / (1 - htr2**2)
    # rrms = np.full((4,), rrms2)
    # Constant mean radius is input

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

    # Assemble the dict
    out = {
        "So1": ml.stagnation[0],
        "PR_tt": ml.PR_tt,
        "psi": (ml.ho[3] - ml.ho[0]) / U2**2,
        "phi2": Vx2 / U2,
        "zeta": zeta,
        "Ma2": Ma2,
        "DMa3_rel": ml.Ma_rel[3] - Ma2,
        "Alpha1": ml.Alpha[0],
        "mdot": ml.mdot[0],
        "Ys": tuple(Ys),
        "htr2": ml.htr[1],
        "rrms": ml.rrms[0],
    }
    return out
