"""Functions for mixed-out averaging."""

import numpy as np
import compflow as cf
import matplotlib.pyplot as plt


def node_to_face(var):
    """For a (n,m) matrix of some property, average over the four corners of
    each face to produce an (n-1,m-1) matrix of face-centered properties."""
    return np.mean(
        np.stack(
            (var[:-1, :-1], var[1:, 1:], var[:-1, 1:], var[1:, :-1]),
            axis=0,
        )
    )

def face_length(c):
    """For (n,m) matrix of coordinates, get face length matrices."""
    return c[1:, 1:] - c[:-1, :-1], c[:-1, 1:] - c[1:, :-1]

def face_area(x, r, rt):
    """Calculate x and r areas for all cells in a cut."""

    # Lengths of each face
    dx1, dx2 = face_length(x)
    dr1, dr2 = face_length(r)
    drt1, drt2 = face_length(rt)

    # Cross lengths
    dAx = 0.5 * (dr1 * drt2 - dr2 * drt1)
    dAr = 0.5 * (dx2 * drt1 - dx1 * drt2)

    return dAx, dAr

def area_total(x, r, rt):
    dAx, dAr = face_area(x, r, rt)
    return np.sum(dAx), np.sum(dAr)

def area_integrate(x, r, rt, fx, fr):
    """Integrate variable over a y-z area and return total."""

    # Face areas and face-centered fluxes
    dAx, dAr = face_area(x, r, rt)
    fx_face = node_to_face(fx)
    fr_face = node_to_face(fr)

    # Perform integration
    return np.sum( fx_face * dAx ) + np.sum( fr_face * dAr)

def specific_heats( ga, rgas ):
    """Calculate specific heats from gas constant and specific heat ratio ."""
    cv = rgas / (ga-1.)
    cp = cv * ga
    return cp, cv

def primary_to_fluxes(r, ro, rovx, rovr, rorvt, roe, ga, rgas, Omega):
    """Convert CFD primary variables into fluxes of mass, momentum, energy."""

    cp, cv = specific_heats( ga, rgas )

    # Secondary variables
    vx, vr, vt, P, T = primary_to_secondary(r, ro, rovx, rovr, rorvt, roe, ga, rgas )

    # Calculate some secondary variables
    vsq = vx**2. + vr**2. + vt**2.
    ho = cp*T + 0.5*vsq
    rvt = vt*r

    # Mass fluxes
    mass_fluxes = np.stack((rovx, rovr))

    # Axial momentum fluxes
    xmom_fluxes = np.stack((rovx*vx + P, rovx*vr))

    # Moment of angular momentum fluxes
    rtmom_fluxes = np.stack((rovx*rvt, rovr*rvt))

    # Stagnation rothalpy fluxes
    ho_fluxes = np.stack((rovx*(ho - Omega*rvt), rovr*(ho - Omega*rvt) ))

    return mass_fluxes, xmom_fluxes, rtmom_fluxes, ho_fluxes




def mix_out(x, r, rt, ro, rovx, rovr, rorvt, roe, ga, rgas, Omega):
    """Perform mixed-out averaging."""

    cv = rgas / (ga-1.)
    cp = cv * ga

    # Get fluxes
    mass_fl, xmom_fl, rtmom_fl, ho_fl = primary_to_fluxes(
        r, ro, rovx, rovr, rorvt, roe, ga, rgas, Omega
    )

    # Get totals by integrating over area
    mass_tot = area_integrate(x, r, rt, *mass_fl)
    xmom_tot = area_integrate(x, r, rt, *xmom_fl)
    rtmom_tot = area_integrate(x, r, rt, *rtmom_fl)
    ho_tot = area_integrate(x, r, rt, *ho_fl)

    # Mix out at the mean radius
    rmid = np.mean((r.min(), r.max()))

    # The hypothetical mixed-out state is at constant x
    # So get the projected area in x-direction by integrating fx = 1, fr = 0
    Ax = area_integrate(x, r, rt, np.ones_like(x), np.zeros_like(x))

    # Guess for density
    ro_mix = np.mean(ro)

    # Fixed point iteration
    max_iter = 100
    tol_rel = 0.0001
    for _ in range(max_iter):

        # Axial velocity by conservation of mass
        vx_mix = mass_tot/ro_mix/Ax

        # Tangential velocity by conservation of moment of angular momentum
        vt_mix = rtmom_tot/ro_mix/vx_mix/rmid/Ax

        # Pressure by conservation of axial momentum
        P_mix = xmom_tot/Ax - ro_mix*vx_mix**2.

        # Stagnation enthalpy by conservation of energy
        ho_mix = ho_tot/ro_mix/vx_mix/Ax  + rmid*Omega*vt_mix

        # Mixed-out Mach
        vsq_mix = vx_mix**2. + vt_mix**2.
        V_cpTo_mix = np.sqrt(vsq_mix/ho_mix)
        Ma_mix = cf.Ma_from_V_cpTo(V_cpTo_mix, ga)

        # Static temperature
        T_mix = (ho_mix/cp)/cf.To_T_from_Ma(Ma_mix, ga)

        # New density_gess
        ro_new = P_mix / rgas / T_mix

        # Check convergence
        dro = np.abs(ro_new - ro_mix)/ro_mix
        ro_mix = ro_new

        if dro < tol_rel:
            break

    # Convert mixed state to primary variables
    rovx_mix = ro_mix * vx_mix
    rovr_mix = 0.  # Parallel streamlines, no radial velocity
    rorvt_mix = ro_mix * rmid * vt_mix
    roe_mix = ro_mix * ( cv*T_mix + 0.5 * vsq_mix )

    # Return the mixed state
    return rmid, ro_mix, rovx_mix, rovr_mix, rorvt_mix, roe_mix

def primary_to_secondary(r, ro, rovx, rovr, rorvt, roe, ga, rgas):
    """Convert CFD primary variables to pressure, temperature and velocity."""
    cp, cv = specific_heats( ga, rgas )
    # Divide out density
    vx = rovx/ro
    vr = rovr/ro
    vt = rorvt/ro/r
    e = roe/ro

    # Calculate secondary variables
    vsq = vx**2. + vr**2. + vt**2.
    T = (e - 0.5*vsq)/cv
    P = ro*rgas*T

    return vx, vr, vt, P, T

if __name__ == "__main__":

    # Coordinates
    xv = np.linspace(1.,2.)
    yv = np.linspace(1.,2.)
    zv = np.linspace(1.,2.)
    rv = yv**2. + zv**2.
    tv = np.arctan2(yv, zv)

    # Set up a non-uniform flow with uniform mass fluxes

