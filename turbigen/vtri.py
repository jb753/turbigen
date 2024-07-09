"""Utility functions for manipulating velocity triangles."""
import numpy as np
import turbigen.util


def resolve_merid(Vm, Alpha, Beta):
    """Convert a meridional velocity and angles to an xrt vector."""
    return np.stack(
        (
            Vm * turbigen.util.cosd(Beta),
            Vm * turbigen.util.sind(Beta),
            Vm * turbigen.util.tand(Alpha),
        )
    )


def resolve_rel_magnitude_abs_yaw(V_rel, phi, Alpha, Beta):
    """Velocity components from relative magnitude and absolute yaw."""

    # The below equations can be found by combining:
    # Vt_rel = Vt - U
    # phi = Vm/U
    # tanAlpha = Vt/Vm
    # V_rel**2 = Vt_rel**2 + Vm**2

    Vm = V_rel * (1.0 + (turbigen.util.tand(Alpha) - 1.0 / phi) ** 2.0) ** -0.5
    Vxrt = resolve_merid(Vm, Alpha, Beta)
    return Vxrt


def resolve_magnitude(V, Alpha, Beta):
    """Velocity components from magnitude and angles."""
    Vm = V * turbigen.util.cosd(Alpha)
    return resolve_merid(Vm, Alpha, Beta)


def annulus_geometry_from_flow(Vxrt, mdot, rho, phi, htr):

    Vm = turbigen.util.vecnorm(Vxrt[:2])
    U = Vm / phi
    A = mdot / Vm / rho
    rrms = np.sqrt(A / np.pi * 0.5 * (1.0 + htr**2) / (1 - htr**2))
    Omega = U / rrms

    return rrms, A, Omega
