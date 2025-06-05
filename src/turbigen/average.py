"""Functions for mixed-out averaging."""

import numpy as np
import turbigen.util
import scipy.optimize

logger = turbigen.util.make_logger()


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

    assert np.ptp(F.Omega) == 0.0
    Omega = np.float64(F.Omega.mean())

    # Get totals by integrating over area
    mass_tot = F.mass_integrate()
    xmom_tot = F.mass_integrate(F.Vx) + F.area_integrate((F.P, 0.0, 0.0))
    rmom_tot = F.mass_integrate(F.Vr) + F.area_integrate((0.0, F.P, 0.0))
    rtmom_tot = F.mass_integrate(F.rVt) + F.area_integrate((0.0, 0.0, F.r * F.P))
    I_tot = F.mass_integrate(F.I)
    s_tot = F.mass_integrate(F.s)

    # Mix out at the rms radius
    r_mix = np.sqrt(0.5 * (F.r.min() ** 2.0 + F.r.max() ** 2.0))
    x_mix = 0.5 * (F.x.max() + F.x.min())
    t_mix = 0.5 * (F.t.max() + F.t.min())
    xrt_mix = np.array([x_mix, r_mix, t_mix])

    # Get the projected areas in x and r directions
    Ax = F.area_integrate((np.ones(F.shape), 0.0, 0.0))
    Ar = F.area_integrate((0.0, np.ones(F.shape), 0.0))

    # An initial guess of zero pitch can prevent convergence of radial momentum, so
    # override small pitch angles
    Beta_mix = F.Beta.mean()
    if np.abs(Beta_mix < 1.0):
        Beta_mix = 0.1

    # Set up initial guess state
    Fm = F.copy().empty()
    Fm.xrt = xrt_mix
    Fm.Vx = F.Vx.mean()
    Fm.Vr = F.Vr.mean()
    Fm.Vt = F.Vt.mean()
    Fm.Omega = Omega
    Fm.set_rho_u(F.rho.mean(), F.u.mean())

    # Iterate on the guess state to match the desired total fluxes
    solve_state(mass_tot, xmom_tot, rmom_tot, rtmom_tot, I_tot, Ax, Ar, Fm)

    # Check conservation
    mass_mix = (Fm.rhoVx * Ax + Fm.rhoVr * Ar).sum()
    xmom_mix = ((Fm.rhoVx * Fm.Vx + Fm.P) * Ax + Fm.rhoVr * Fm.Vx * Ar).sum()
    rmom_mix = (Fm.rhoVx * Fm.Vr * Ax + (Fm.rhoVr * Fm.Vr + Fm.P) * Ar).sum()
    rtmom_mix = (Fm.rhoVx * Fm.r * Fm.Vt * Ax + Fm.rhoVr * Fm.r * Fm.Vt * Ar).sum()
    I_mix = Fm.I * mass_mix
    s_mix = Fm.s * mass_mix

    # Set absolute tolerances to rtol*reference to be more numerically robust
    # This handles with xmom or rmom ~ 0, and cases with low net mass flow
    rtol = 2e-2

    # Mass tolerance
    V_ref = F.V.mean()
    ro_ref = F.rho.mean()
    A_ref = np.sqrt(Ax**2.0 + Ar**2.0)
    mass_ref = ro_ref * V_ref * A_ref
    mass_tol = mass_ref * rtol
    assert np.isclose(
        mass_tot, mass_mix, atol=mass_tol
    ), f"Total mass {mass_tot} does not match mixed {mass_mix} within tolerance {mass_tol}"

    # Momentum
    mom_ref = np.max((np.abs(xmom_tot), np.abs(rmom_tot)))
    mom_tol = rtol * mom_ref
    assert np.isclose(
        xmom_tot, xmom_mix, atol=mom_tol
    ), f"Total xmom {xmom_tot} does not match mixed {xmom_mix} within tolerance {mom_tol}"
    assert np.isclose(
        rmom_tot, rmom_mix, atol=mom_tol
    ), f"Total rmom {rmom_tot} does not match mixed {rmom_mix} within tolerance {mom_tol}"

    # Angular momentum
    rtmom_tol = mom_ref * r_mix * rtol
    assert np.isclose(
        rtmom_tot, rtmom_mix, atol=rtmom_tol
    ), f"Total rtmom {rtmom_tot} does not match mixed {rtmom_mix} within tolerance {rtmom_tol}"

    # Rothalpy error is proportional to mass
    I_tol = np.abs(I_tot * mass_tol / mass_tot)
    assert np.isclose(
        I_tot, I_mix, atol=I_tol
    ), f"Total rothalpy {I_tot} does not match mixed {I_mix} within tolerance {I_tol}"

    try:
        Nb = F.Nb
    except (AttributeError, KeyError):
        Nb = 1

    # Quantify mixing loss
    dsirrev = (s_mix - s_tot) / mass_tot
    Aann = A_ref * Nb
    # Return the mixed-out flow field state
    return Fm, Aann, dsirrev


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
