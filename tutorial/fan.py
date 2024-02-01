import turbigen.flowfield
import numpy as np

def forward(So1, DPo, mdot, phi, psi, htr, etatt):
    """Caluclate mean-line from inlet and design variables."""

    # Get the ideal exit state
    So2s = So1.copy()  # Duplicate the inlet state
    So2s.set_P_s(So1.P + DPo, So1.s)  # Set pressure and entropy

    # Work from defn efficiency Eqn. (1)
    Dho = (So2s.h-So1.h)/etatt

    # Blade speed from defn psi Eqn. (2)
    U = np.sqrt(Dho/psi)

    # Axial velocity from defn phi Eqn. (3)
    Vx = phi*U

    # Circumferential velocity from Euler Eqn. (4)
    Vt2 = Dho/U

    # Assemble velocity vectors
    # shape (3 directions, 2 stations)
    Vxrt = np.stack(
        (
            (Vx, Vx),  # Constant axial velocity
            (0., 0.),  # No radial velocity
            (0., Vt2),  # Zero inlet swirl
        )
    )

    # Outlet stagnation state from known total rises
    So2 = So1.copy().set_P_h(So1.P + DPo, So1.h + Dho)

    # Assemble both stagnation states into a vector state
    So = So1.stack((So1,So2))

    # Get static states using velocity magnitude and same entropy
    Vmag = np.sqrt(np.sum(Vxrt**2,axis=0))
    h = So.h - 0.5*Vmag**2  # Static enthalpy
    S = So.copy().set_h_s(h , So.s)
    Po = S.P + 0.5*S.rho*Vmag**2

    # Conservation of mass for annulus area, Eqn. (5)
    A = mdot/S.rho/Vx

    # Mean radius from HTR Eqn. (6)
    rrms = np.sqrt(A/np.pi/2.*(1.+htr**2)/(1.-htr**2))

    # Shaft angular velocity
    Omega = U / rrms

    # Return assembled mean-line object
    return turbigen.flowfield.make_mean_line(
        rrms,  # Mean radii
        A,  # Annulus areas
        Omega,  # Shaft angular velocity
        Vxrt, # Velocity vectors
        S  # Thermodynamic states
    )


def inverse(ml):
    """Calculate design variables from a mean-line object."""

    So1 = ml.stagnation[0]
    So2s = So1.copy().set_P_s(ml.Po[-1], ml.s[0])
    ho2s = So2s.h

    # The output is a dictionary keyed by the args to forward
    return {
        "So1": So1,
        'DPo': ml.Po[-1] - ml.Po[0],
        'mdot': ml.mdot[0],
        'phi': ml.Vx[0]/ml.U[0],
        'psi': (ml.ho[-1]-ml.ho[0])/(ml.U[0])**2,
        'etatt': (ho2s-So1.h)/(ml.ho[-1]-ml.ho[0]),
        'htr': ml.rhub[0]/ml.rtip[0]
    }
