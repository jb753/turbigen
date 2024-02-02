import turbigen.flowfield
import numpy as np


def forward(So1, PR, mdot, phi, psi, htr1, etatt):
    """Caluclate mean-line from inlet and design variables."""

    # Get the ideal exit state
    So2s = So1.copy()  # Duplicate the inlet state
    Po2 = So1.P * PR
    So2s.set_P_s(Po2, So1.s)  # Set pressure and entropy

    # Work from defn efficiency Eqn. (1)
    Dho = (So2s.h - So1.h) / etatt

    # Blade speed from defn psi Eqn. (2)
    U = np.sqrt(Dho / psi)

    # Axial velocity from defn phi Eqn. (3)
    Vx = phi * U

    # Circumferential velocity from Euler Eqn. (4)
    Vt2 = Dho / U

    # Assemble velocity vectors
    # shape (3 directions, 2 stations)
    Vxrt = np.stack(
        (
            (Vx, Vx),  # Constant axial velocity
            (0.0, 0.0),  # No radial velocity
            (0.0, Vt2),  # Zero inlet swirl
        )
    )

    # Outlet stagnation state from known total rises
    So2 = So1.copy().set_P_h(Po2, So1.h + Dho)

    # Assemble both stagnation states into a vector state
    So = So1.stack((So1, So2))

    # Get static states using velocity magnitude and same entropy
    Vmag = np.sqrt(np.sum(Vxrt**2, axis=0))
    h = So.h - 0.5 * Vmag**2  # Static enthalpy
    S = So.copy().set_h_s(h, So.s)

    # Conservation of mass for annulus area, Eqn. (5)
    A = mdot / S.rho / Vx

    # Constant mean radius at inlet from HTR Eqn. (6)
    rrms1 = np.sqrt(A[0] / np.pi / 2.0 * (1.0 + htr1**2) / (1.0 - htr1**2))
    rrms = np.tile(rrms1, 1)

    # Shaft angular velocity
    Omega = U / rrms

    # Return assembled mean-line object
    return turbigen.flowfield.make_mean_line(
        rrms,  # Mean radii
        A,  # Annulus areas
        Omega,  # Shaft angular velocity
        Vxrt,  # Velocity vectors
        S,  # Thermodynamic states
    )


def inverse(ml):
    """Calculate design variables from a mean-line object."""

    So1 = ml.stagnation[0]
    So2s = So1.copy().set_P_s(ml.Po[-1], ml.s[0])
    ho2s = So2s.h

    # The output is a dictionary keyed by the args to forward
    return {
        "So1": So1,
        "PR": ml.Po[-1] / ml.Po[0],
        "mdot": ml.mdot[0],
        "phi": ml.Vx[0] / ml.U[0],
        "psi": (ml.ho[-1] - ml.ho[0]) / (ml.U[0]) ** 2,
        "etatt": (ho2s - So1.h) / (ml.ho[-1] - ml.ho[0]),
        "htr1": ml.rhub[0] / ml.rtip[0],
    }
