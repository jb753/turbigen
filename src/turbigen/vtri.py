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

    cosBeta = np.cos(np.arctan2(Vxrt[1], Vxrt[0]))

    # Use a generic definition of hub/tip ratio
    K = (1.0 - htr) / (1.0 + htr)
    rmid = np.sqrt(A / 4.0 / np.pi / K)
    H = A / 2.0 / np.pi / rmid
    rhub = rmid - H / 2.0 * cosBeta
    rcas = rmid + H / 2.0 * cosBeta
    rrms = np.sqrt(0.5 * (rhub**2 + rcas**2))
    Omega = U / rrms

    return rrms, A, Omega


def solve_rrms(A, htr, Beta=0.0):
    cosBeta = turbigen.util.cosd(Beta)
    K = (1.0 - htr) / (1.0 + htr)
    rmid = np.sqrt(A / 4.0 / np.pi / K)
    H = A / 2.0 / np.pi / rmid
    rhub = rmid - H / 2.0 * cosBeta
    rcas = rmid + H / 2.0 * cosBeta
    rrms = np.sqrt(0.5 * (rhub**2 + rcas**2))

    return rrms
