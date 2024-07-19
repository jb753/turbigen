"""Functions for mixed-out averaging."""

import numpy as np
import turbigen.util
import scipy.optimize

logger = turbigen.util.make_logger()


def face_length(c):
    """For (n,m) matrix of coordinates, get face length matrices in i- and j-dirs."""
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


def orient_grid(x, r, rt):
    """Return a closure that orients (n,m) matrix to be consistent with correct dA."""

    ax_flip = []

    logger.debug("Orienting grid")

    # Want k to be along increasing theta
    Dt_k = np.mean(rt[:, -1] - rt[:, 0])
    if Dt_k < 0.0:
        ax_flip.append(1)

    # Want j to be along increasing r
    # or failing that along decreasing x
    Dx_j = np.mean(x[-1, :] - x[0, :])
    Dr_j = np.mean(r[-1, :] - r[0, :])
    logger.debug(f"Dx_j: {Dx_j}")
    logger.debug(f"Dr_j: {Dr_j}")

    if np.abs(Dr_j) > np.abs(Dx_j):
        logger.debug("This cut is x~const")
        if Dr_j < 0.0:
            ax_flip.append(0)

    else:
        logger.debug("This cut is r~const")
        if Dx_j > 0.0:
            ax_flip.append(0)

    logger.debug(f"Flipping along axes {ax_flip}")

    def _orient(q):
        return np.flip(q, axis=ax_flip).copy()

    return _orient


def area_total(x, r, rt):
    dAx, dAr = face_area(x, r, rt)
    return np.sum(dAx), np.sum(dAr)


def area_integrate(x, r, rt, fx, fr):
    """Integrate variable over a y-z area and return total."""

    # Face areas and face-centered fluxes
    dAx, dAr = face_area(x, r, rt)
    fx_face = turbigen.util.node_to_face(fx)
    fr_face = turbigen.util.node_to_face(fr)

    # Perform integration
    return np.sum(fx_face * dAx) + np.sum(fr_face * dAr)


def specific_heats(ga, rgas):
    """Calculate specific heats from gas constant and specific heat ratio ."""
    cv = rgas / (ga - 1.0)
    cp = cv * ga
    return cp, cv


def primary_to_fluxes(r, ro, rovx, rovr, rorvt, P, ho, Omega):
    """Convert CFD primary variables into fluxes of mass, momentum, energy."""

    # Divide out velocities
    vx = rovx / ro
    vr = rovr / ro
    rvt = rorvt / ro

    # Mass fluxes in x and r dirns
    mass_fluxes = np.stack((rovx, rovr))

    # Axial momentum fluxes in x and r dirns
    xmom_fluxes = np.stack((rovx * vx + P, rovr * vx))

    # Radial momentum fluxes in x and r dirns
    rmom_fluxes = np.stack((rovx * vr, rovr * vr + P))

    # Moment of angular momentum fluxes in x and r dirns
    rtmom_fluxes = np.stack((rovx * rvt, rovr * rvt))

    # Stagnation rothalpy fluxes in x an r dirns
    ho_fluxes = np.stack((rovx * (ho - Omega * rvt), rovr * (ho - Omega * rvt)))

    return mass_fluxes, xmom_fluxes, rmom_fluxes, rtmom_fluxes, ho_fluxes


def solve_state(mass_tot, xmom_tot, rmom_tot, rtmom_tot, ho_tot, Ax, Ar, F_mix):
    # Normalise vars for minimise
    ro0 = F_mix.rho + 0.0
    Dro = ro0 + 0.0
    Beta0 = F_mix.Beta + 0.0
    DBeta = 1.0
    r_mix = F_mix.r
    Omega = F_mix.Omega
    vm_guess = np.abs(F_mix.Vm)
    V_ref = np.abs(vm_guess)
    ro_ref = F_mix.rho

    def _guess_roBeta(roBeta_norm):
        # Initialise a scalar flowfield for the mixed-out flow
        ro_mix = roBeta_norm[0] * Dro + ro0
        Beta_mix = roBeta_norm[1] * DBeta + Beta0

        # Trig
        tanBeta_mix = np.tan(np.radians(Beta_mix))

        # Choose which way round to solve for velocity components
        # if Ar == 0.0 or (cosBeta_mix > np.sqrt(2.0) / 2.0 and Ax != 0.0):
        if np.abs(Ar) < np.abs(Ax):
            # Beta < 45 and Vx > Vr, Ax > Ar, tanBeta small
            # Select a numerically robust reference area
            Aref = Ax + tanBeta_mix * Ar

            # Conservation of mass
            vx_mix = mass_tot / ro_mix / Aref

            # Conservation of x-momentum
            P_mix = (xmom_tot - ro_mix * vx_mix**2.0 * Aref) / Ax

            # Conservation of moment of angular momentum
            vt_mix = rtmom_tot / ro_mix / r_mix / vx_mix / Aref

            # New estimate of radial velocity
            vrsq_mix = np.abs((rmom_tot - P_mix * Ar) * tanBeta_mix / ro_mix / Aref)

            # Stagnation enthalpy by conservation of energy
            ho_mix = ho_tot / ro_mix / vx_mix / Aref + r_mix * Omega * vt_mix

            # Meridional velocity
            vm_mix = np.sqrt(vx_mix**2.0 + vrsq_mix)

            # Assign sign of Vr to give correct sign of rmom
            vr_mix = np.sqrt(vrsq_mix)
            rmom_mix = (
                ro_mix * vx_mix * vr_mix * Ax + (ro_mix * vr_mix**2.0 + P_mix) * Ar
            )
            if not np.sign(rmom_mix) == np.sign(rmom_tot):
                vr_mix *= -1.0

        else:
            # Beta > 45 and Vx < Vr, Ax < Ar, tanBeta large
            # Select a numerically robust reference area
            Aref = Ax / tanBeta_mix + Ar

            # Conservation of mass
            vr_mix = mass_tot / ro_mix / Aref

            # Pressure by conservation of radial momentum
            P_mix = (rmom_tot - ro_mix * vr_mix**2.0 * Aref) / Ar

            # Conservation of moment of angular momentum
            vt_mix = rtmom_tot / ro_mix / r_mix / vr_mix / Aref

            # New estimate of axial velocity
            cotBeta_mix = 1.0 / tanBeta_mix
            vxsq_mix = np.abs((xmom_tot - P_mix * Ax) * cotBeta_mix / ro_mix / Aref)

            # Stagnation enthalpy by conservation of energy
            ho_mix = ho_tot / ro_mix / vr_mix / Aref + r_mix * Omega * vt_mix

            # Meridional velocity
            vm_mix = np.sqrt(vxsq_mix + vr_mix**2.0)

            # Assign sign of Vx to give correct sign of xmom
            vx_mix = np.sqrt(vxsq_mix)
            xmom_mix = (
                ro_mix * vx_mix**2.0 + P_mix
            ) * Ax + ro_mix * vr_mix * vx_mix * Ar
            if not np.sign(xmom_mix) == np.sign(xmom_tot):
                vx_mix *= -1.0

        vsq_mix = vm_mix**2.0 + vt_mix**2.0

        # Static enthalpy
        h_mix = ho_mix - 0.5 * vsq_mix

        # New density guess from eqn of state
        F_mix.set_P_h(P_mix, h_mix)
        # ro_new = F_mix.rho

        # Insert velocities into flowfield
        F_mix.Vxrt = np.array([vx_mix, vr_mix, vt_mix])

        # Mixed-out fluxes
        rovx_mix = F_mix.rhoVx
        rovr_mix = F_mix.rhoVr
        vx_mix = F_mix.Vx
        vr_mix = F_mix.Vr
        vt_mix = F_mix.Vt
        P_mix = F_mix.P
        ho_mix = F_mix.ho

        # Check conservation
        mass_mix = rovx_mix * Ax + rovr_mix * Ar
        xmom_mix = (ro_mix * vx_mix**2.0 + P_mix) * Ax + ro_mix * vr_mix * vx_mix * Ar
        rmom_mix = ro_mix * vx_mix * vr_mix * Ax + (ro_mix * vr_mix**2.0 + P_mix) * Ar
        rtmom_mix = ro_mix * r_mix * vt_mix * (vx_mix * Ax + vr_mix * Ar)
        ho_mix = (
            ro_mix * (ho_mix - Omega * r_mix * vt_mix) * (vx_mix * Ax + vr_mix * Ar)
        )

        # Set absolute tolerances to rtol*reference to be more numerically robust
        # This handles with xmom or rmom ~ 0, and cases with low net mass flow
        rtol = 1e-2

        # Mass
        A_ref = np.sqrt(Ax**2.0 + Ar**2.0)
        mass_ref = ro_ref * V_ref * A_ref
        mass_tol = mass_ref * rtol

        # Momentum
        mom_ref = np.max((np.abs(xmom_tot), np.abs(rmom_tot)))
        mom_tol = rtol * mom_ref

        # Energy error is proportional to mass
        ho_tol = np.abs(ho_tot * mass_tol / mass_tot)

        err = np.sum(
            np.array(
                [
                    (mass_mix - mass_tot) / mass_tol,
                    (xmom_mix - xmom_tot) / mom_tol,
                    (rmom_mix - rmom_tot) / mom_tol,
                    (rtmom_mix - rtmom_tot) / mom_tol / r_mix,
                    (ho_mix - ho_tot) / ho_tol,
                ]
            )
            ** 2.0
        )

        return err

    x0 = np.array([0.0, 0.0])
    initial_simplex = np.array([[0.0, 0.0], [0.01, 0.01], [0.0, 0.01]])
    scipy.optimize.minimize(
        _guess_roBeta,
        x0,
        options={"initial_simplex": initial_simplex, "xatol": 1e-6, "fatol": 1e-6},
        method="nelder-mead",
        bounds=((-0.5, 0.5), (-60.0, 60)),
    )


def mix_out(F):
    """Perform mixed-out averaging on a flow field."""

    logger.debug("MIXING OUT A CUT")

    x = F.x.squeeze()
    r = F.r.squeeze()
    rt = F.rt.squeeze()
    t = F.t.squeeze()

    assert np.ptp(F.Omega) == 0.0
    Omega = np.float64(F.Omega.mean())
    s = F.s.squeeze()

    ro = F.rho.squeeze()
    rovx = F.rhoVx.squeeze()
    rovr = F.rhoVr.squeeze()

    # # Orient grid
    # flipper = orient_grid(x, r, rt)
    # x, r, rt, ro, rovx, rovr, rorvt, P, ho, s = [
    #     flipper(q).astype(np.float64)
    #     for q in (x, r, rt, ro, rovx, rovr, rorvt, P, ho, s)
    # ]

    # Mass fluxes in x and r dirns
    mass_fl = F.flux_mass.squeeze()

    # Axial momentum fluxes in x and r dirns
    xmom_fl = F.flux_xmom.squeeze()

    # Radial momentum fluxes in x and r dirns
    rmom_fl = F.flux_rmom.squeeze()

    # Moment of angular momentum fluxes in x and r dirns
    rtmom_fl = F.flux_rtmom.squeeze()

    # Stagnation rothalpy fluxes in x an r dirns
    ho_fl = F.flux_rothalpy.squeeze()

    # Get totals by integrating over area
    mass_tot = area_integrate(x, r, rt, *mass_fl[:2])
    xmom_tot = area_integrate(x, r, rt, *xmom_fl[:2])
    rmom_tot = area_integrate(x, r, rt, *rmom_fl[:2])
    rtmom_tot = area_integrate(x, r, rt, *rtmom_fl[:2])
    ho_tot = area_integrate(x, r, rt, *ho_fl[:2])

    # Assemble all total fluxes into a vector for convenience
    all_names = ["mass", "xmom", "rmom", "rtmom", "ho"]
    logger.debug("Total fluxes of conserved quantities are:")
    all_tot = np.array([mass_tot, xmom_tot, rmom_tot, rtmom_tot, ho_tot])
    for n, v in zip(all_names, all_tot):
        logger.debug(f"  {n}: {v}")

    if np.isnan(all_tot).any():
        raise Exception(f"Cannot average a NaN total flux: {all_tot}")

    # Mix out at the rms radius
    r_mix = np.sqrt(0.5 * (r.min() ** 2.0 + r.max() ** 2.0))
    x_mix = 0.5 * (x.max() + x.min())
    t_mix = 0.5 * (t.max() + t.min())
    xrt_mix = np.array([x_mix, r_mix, t_mix])

    # Get the projected area in x-direction by integrating fx = 1, fr = 0
    Ax = area_integrate(x, r, rt, np.ones_like(x), np.zeros_like(x))
    # Get the projected area in r-direction by integrating fx = 0, fr = 1
    Ar = area_integrate(x, r, rt, np.zeros_like(x), np.ones_like(x))

    # Initial guesses for pitch angle
    vr_guess = (rovr / ro).mean()
    vx_guess = (rovx / ro).mean()
    vm_guess = np.sqrt(vr_guess**2.0 + vx_guess**2.0)
    Beta_mix = np.degrees(np.arccos(vx_guess / vm_guess))

    # An initial guess of zero pitch can prevent convergence of radial momentum, so
    # override small pitch angles
    if np.abs(Beta_mix < 1.0):
        Beta_mix = 0.1

    # Set up initial guess state
    F_mix = F.copy().empty()
    F_mix.xrt = xrt_mix
    F_mix.Vx = vx_guess
    F_mix.Vr = vr_guess
    F_mix.Vt = F.Vt.mean()
    F_mix.Omega = Omega
    F_mix.set_rho_u(F.rho.mean(), F.u.mean())

    # Iterate on the guess state to match the desired total fluxes
    solve_state(mass_tot, xmom_tot, rmom_tot, rtmom_tot, ho_tot, Ax, Ar, F_mix)

    # Check conservation
    Axr = np.array([Ax, Ar])
    mass_mix = (F_mix.flux_mass[:2] * Axr).sum()
    xmom_mix = (F_mix.flux_xmom[:2] * Axr).sum()
    rmom_mix = (F_mix.flux_rmom[:2] * Axr).sum()
    rtmom_mix = (F_mix.flux_rtmom[:2] * Axr).sum()
    ho_mix = (F_mix.flux_rothalpy[:2] * Axr).sum()

    # Set absolute tolerances to rtol*reference to be more numerically robust
    # This handles with xmom or rmom ~ 0, and cases with low net mass flow
    rtol = 2e-2

    # Mass
    V_ref = np.abs(vm_guess)
    ro_ref = ro.mean()
    A_ref = np.sqrt(Ax**2.0 + Ar**2.0)
    mass_ref = ro_ref * V_ref * A_ref
    mass_tol = mass_ref * rtol

    # Momentum
    mom_ref = np.max((np.abs(xmom_tot), np.abs(rmom_tot)))
    mom_tol = rtol * mom_ref

    # Energy error is proportional to mass
    ho_tol = np.abs(ho_tot * mass_tol / mass_tot)

    if not np.isclose(mass_mix, mass_tot, atol=mass_tol):
        raise Exception(
            f"""Mixing out did not converge:
    mass {mass_mix, mass_tot, mass_tol}, rel_err={(mass_mix-mass_tot)/mass_tol:.3f}"""
        )

    if not np.isclose(xmom_mix, xmom_tot, atol=mom_tol):
        raise Exception(
            f"""Mixing out did not converge:
    xmom {xmom_mix, xmom_tot, mom_tol}, rel_err={(xmom_mix-xmom_tot)/mom_tol:.3f}"""
        )

    if not np.isclose(rmom_mix, rmom_tot, atol=mom_tol):
        raise Exception(
            f"""Mixing out did not converge:
    rmom {rmom_mix, rmom_tot, mom_tol}, rel_err={(rmom_mix-rmom_tot)/mom_tol:.3f}"""
        )

    if not np.isclose(rtmom_mix, rtmom_tot, atol=mom_tol * r_mix):
        raise Exception(
            f"""Mixing out did not converge:
            rtmom {rtmom_mix, rtmom_tot, mom_tol * r_mix},
            rel_err={(rtmom_mix-rtmom_tot)/mom_tol/r_mix:.3f}"""
        )

    if not np.isclose(ho_mix, ho_tot, atol=ho_tol):
        raise Exception(
            f"""Mixing out did not converge:
    energy {ho_mix, ho_tot, ho_tol}, rel_err={(ho_mix-ho_tot)/ho_tol:.3f}"""
        )

    try:
        Nb = F.Nb
    except AttributeError:
        Nb = 1

    # Quantify mixing loss
    ent_fl = np.stack((rovx * s, rovr * s))
    ent_tot = area_integrate(x, r, rt, *ent_fl)
    ent_mix = mass_mix * F_mix.s
    dsirrev = (ent_mix - ent_tot) / mass_tot
    Aann = A_ref * Nb
    # Return the mixed-out flow field state
    return F_mix, Aann, dsirrev


def primary_to_secondary(r, ro, rovx, rovr, rorvt, roe, ga, rgas):
    """Convert CFD primary variables to pressure, temperature and velocity."""
    cp, cv = specific_heats(ga, rgas)

    # Divide out density
    vx = rovx / ro
    vr = rovr / ro
    vt = rorvt / ro / r
    e = roe / ro

    # Calculate secondary variables
    vsq = vx**2.0 + vr**2.0 + vt**2.0
    T = (e - 0.5 * vsq) / cv
    P = ro * rgas * T

    return vx, vr, vt, P, T


def secondary_to_primary(r, vx, vr, vt, P, T, ga, rgas):
    """Convert secondary variables to CFD primary variables."""

    cp, cv = specific_heats(ga, rgas)

    vsq = vx**2.0 + vr**2.0 + vt**2.0

    ro = P / rgas / T
    rovx = ro * vx
    rovr = ro * vr
    rorvt = ro * r * vt
    roe = ro * (cv * T + 0.5 * vsq)

    return ro, rovx, rovr, rorvt, roe


def mix_out_unstructured(F):
    # Only works with triangles
    assert F.shape[1] == 3

    dA = F.tri_area[:2]

    mass_tot = (F.flux_mass[:2].mean(-1) * dA).sum()
    xmom_tot = (F.flux_xmom[:2].mean(-1) * dA).sum()
    rmom_tot = (F.flux_rmom[:2].mean(-1) * dA).sum()
    rtmom_tot = (F.flux_rtmom[:2].mean(-1) * dA).sum()
    ho_tot = (F.flux_rothalpy[:2].mean(-1) * dA).sum()
    ent_tot = (F.flux_entropy[:2].mean(-1) * dA).sum()

    Ax, Ar = dA.sum(-1)

    # Mix out at the rms radius
    r_mix = np.sqrt(0.5 * (F.r.min() ** 2.0 + F.r.max() ** 2.0))
    x_mix = 0.5 * (F.x.max() + F.x.min())
    t_mix = 0.5 * (F.t.max() + F.t.min())
    xrt_mix = np.array([x_mix, r_mix, t_mix])

    # Set up initial guess state
    F_mix = F.copy().empty()
    F_mix.xrt = xrt_mix
    F_mix.Vxrt = F.Vxrt.mean(axis=(1, 2))
    F_mix.Omega = F.Omega.mean()
    F_mix.set_rho_u(F.rho.mean(), F.u.mean())

    # Iterate on the guess state to match the desired total fluxes
    solve_state(mass_tot, xmom_tot, rmom_tot, rtmom_tot, ho_tot, Ax, Ar, F_mix)

    # Quantify mixing loss
    ent_mix = mass_tot * F_mix.s
    dsirrev = (ent_mix - ent_tot) / mass_tot

    A_ref = np.sqrt(Ax**2.0 + Ar**2.0)
    Aann = A_ref * F.Nb

    return F_mix, Aann, dsirrev
