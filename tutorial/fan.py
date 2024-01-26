import turbigen.flowfield
import numpy as np


def forward(So1, DPo, mdot, phi, psi, etatt):
    """Caluclate mean-line from inlet and design variables."""

    # Get the ideal exit state
    So2s = So1.copy()  # Duplicate the inlet state
    So2s.set_P_s(So1.P + DPo, So1.s)  # Set pressure and entropy

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
            (Vx, Vx),
            (0.0, 0.0),
            (0.0, Vt2),
        )
    )
    print(Vxrt.shape)

    # Insert code to calculate rrms, A, Omega, Vxrt, states
    # ...
    raise NotImplementedError

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

    # The output is a dictionary keyed by the args to forward
    return {
        "So1": ml.stagnation[0],
        # 'DPo': ...,
        # 'mdot': ...,
        # 'phi': ...,
        # 'psi': ...,
        # 'etatt': ...,
    }
