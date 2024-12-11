"""Check averaging is conserving what we want."""

import json
import os
import numpy as np
import turbigen.compflow_native as cf
import turbigen.average
import turbigen.flowfield
from scipy.optimize import newton

MU = 1.8e-4

Tu0 = turbigen.flowfield.PerfectFlowField._Tu0_default


def test_grid():
    """Verify that area vectors have correct signs."""

    # Assemble radial and angular grids
    # r is j, rt is k
    nj = 3
    nk = 4
    rv = np.linspace(1.0, 1.1, nj)
    rtv = np.linspace(0.05, -0.05, nk)
    r, rt = np.meshgrid(rv, rtv, indexing="ij")

    # Now check that the sign of dAr is consistent with slope in meridional plane
    # if dx_dr < 0 then sloping backwards, so Vr>0 => mdot > 0 and dAr > 0
    # if dx_dr > 0 then sloping forwards, so Vr>0 => mdot < 0 and dAr < 0
    for dx_dr in (-0.1, 0.0, 0.1):
        x = 1 + dx_dr * r

        for ax_flip in ((), (0,), (1,), (0, 1)):
            print(dx_dr, ax_flip)

            xrtnow = np.stack([np.flip(c, ax_flip) for c in (x, r, rt)])
            flipper = turbigen.average.orient_grid(*xrtnow)
            xrtflip = np.stack([flipper(q) for q in xrtnow])

            dAx, dAr = turbigen.average.face_area(*xrtflip)
            assert (dAx > 0.0).all()
            assert (np.sign(dAr) == np.sign(-dx_dr)).all()

    # Test a constant-r grid
    # Note that this case is ambigous for the sign of dAr - is the flow
    # approaching from above or below? - so we let the direction of x dictate the sign of dAr
    # Where i in -x dirn => dAr > 0
    rtv = np.linspace(-0.05, 0.05, nk)
    xv = np.linspace(0.1, 0.0, nj)
    x, rt = np.meshgrid(xv, rtv, indexing="ij")
    r = np.ones_like(rt)
    for ax_flip in ((), (1,)):
        if ax_flip:
            xnow = np.flip(x, ax_flip)
            rnow = np.flip(r, ax_flip)
            rtnow = np.flip(rt, ax_flip)
        else:
            xnow = x.copy()
            rnow = r.copy()
            rtnow = rt.copy()

        flipper = turbigen.average.orient_grid(xnow, rnow, rtnow)

        xnow, rnow, rtnow = [flipper(q) for q in (xnow, rnow, rtnow)]

        dAx, dAr = turbigen.average.face_area(xnow, rnow, rtnow)
        assert np.isclose(dAx, 0.0).all()
        assert (dAr > 0.0).all()


def test_nonuniform_energy():
    """Run a test flow with nonuniform energy."""
    # Coordinates
    r0, r1 = 1.0, 2.0
    rv = np.linspace(r0, r1, 3)
    tv = np.linspace(-np.pi / 8.0, np.pi / 8.0, 4)
    r, t = np.meshgrid(rv, tv, indexing="ij")
    t = t + np.pi / 64 * r / r0  # Warp the grid
    rt = r * t
    r_norm = (r - r0) / (r1 - r0)
    x = np.ones_like(r) * r_norm * 0.2  # Angled cut plane

    # Define a base state that we want to conserve
    rgas = 287.0
    ga = 1.4
    cp = rgas * ga / (ga - 1.0)
    cv = cp / ga
    rovx_ref = 200.0
    xmom_ref = 1.1e5
    rtmom_ref = 1.5e4
    Omega = 200.0

    # Make the non-uniform flow
    ro0 = 1.0
    ro = ro0 * (1.0 + 0.5 * r_norm)
    vx = rovx_ref / ro
    vr = np.zeros_like(vx)
    P = xmom_ref - ro * vx**2.0
    T = P / rgas / ro
    vt = rtmom_ref / ro / vx / r

    # Initialise a flow-field object
    F = turbigen.flowfield.PerfectFlowField.from_properties(
        np.stack((x, r, t)), np.stack((vx, vr, vt)), np.stack((P, T)), cp, ga, MU, Omega
    )

    # Do the mixing
    F_mix = turbigen.average.mix_out(F)[0]

    # We used to manually check for conservation here, but now this is done
    # internally by `average.mix_out` which will raise an error if any of mass,
    # momentum, energy are not balanced


def test_nonuniform_xmom():
    """Run a test flow with nonuniform x-momentum."""
    # Coordinates
    r0, r1 = 1.0, 2.0
    rv = np.linspace(1.0, 2.0)
    tv = np.linspace(-np.pi / 8.0, np.pi / 8.0)
    r, t = np.meshgrid(rv, tv, indexing="ij")
    rt = r * t
    r_norm = (r - r0) / (r1 - r0)
    x = np.ones_like(r) * r_norm * 0.2  # Angled cut plane

    # Define a base state that we want to conserve
    rgas = 287.0
    ga = 1.4
    cp = rgas * ga / (ga - 1.0)
    cv = cp / ga
    rovx_ref = 200.0
    I_ref = 1e6
    rtmom_ref = 1.5e4
    Omega = 200.0

    # Make the non-uniform flow
    ro0 = 1.0
    ro = ro0 * (1.0 + 0.5 * r_norm)
    vx = rovx_ref / ro
    vr = np.zeros_like(vx)
    rmid = np.mean((r0, r1))
    vt = rtmom_ref / ro / vx / r

    hu0 = rgas * Tu0
    ho = I_ref + rmid * Omega * vt
    To = (ho - hu0) / cp + Tu0

    vsq = vx**2.0 + vr**2.0 + vt**2.0
    v_cpTo = np.sqrt(vsq / cp / To)
    Ma = cf.Ma_from_V_cpTo(v_cpTo, ga)
    T = To / cf.To_T_from_Ma(Ma, ga)
    P = ro * rgas * T

    # Make the flow field
    F = turbigen.flowfield.PerfectFlowField.from_properties(
        np.stack((x, r, t)), np.stack((vx, vr, vt)), np.stack((P, T)), cp, ga, MU, Omega
    )

    # Mix out
    F_mix = turbigen.average.mix_out(F)[0]

    # We used to manually check for conservation here, but now this is done
    # internally by `average.mix_out` which will raise an error if any of mass,
    # momentum, energy are not balanced

    # Rothalpy should be close to reference value
    # Not exact because we have a small radial span
    # I_mix = cp * To_mix - r_mix * Omega * vt_mix
    assert np.isclose(F_mix.I, I_ref, rtol=1e-3)


def test_nonuniform_grid():
    """Run a test flow with nonuniform x-momentum and grid."""

    # Coordinates
    r0, r1 = 1.0, 2.0
    rv = np.geomspace(1.0, 2.0)
    tv = np.linspace(-np.pi / 8.0, np.pi / 8.0)
    r, t = np.meshgrid(rv, tv, indexing="ij")
    rt = r * t
    r_norm = (r - r0) / (r1 - r0)
    x = np.ones_like(r) * r_norm * 0.2  # Angled cut plane

    # Define a base state that we want to conserve
    rgas = 287.0
    ga = 1.4
    cp = rgas * ga / (ga - 1.0)
    cv = cp / ga
    rovx_ref = 200.0
    I_ref = 1e6
    rtmom_ref = 1.5e4
    Omega = 200.0

    # Make the non-uniform flow
    ro0 = 1.0
    ro = ro0 * (1.0 + 0.5 * r_norm)
    vx = rovx_ref / ro
    vr = np.zeros_like(vx)
    rmid = np.mean((r0, r1))
    vt = rtmom_ref / ro / vx / r

    hu0 = rgas * Tu0
    ho = I_ref + rmid * Omega * vt
    To = (ho - hu0) / cp + Tu0

    vsq = vx**2.0 + vr**2.0 + vt**2.0
    v_cpTo = np.sqrt(vsq / cp / To)
    Ma = cf.Ma_from_V_cpTo(v_cpTo, ga)
    T = To / cf.To_T_from_Ma(Ma, ga)
    P = ro * rgas * T

    # Initialise a flow-field object
    F = turbigen.flowfield.PerfectFlowField.from_properties(
        np.stack((x, r, t)), np.stack((vx, vr, vt)), np.stack((P, T)), cp, ga, MU, Omega
    )

    # Do the mixing
    F_mix = turbigen.average.mix_out(F)[0]

    # We used to manually check for conservation here, but now this is done
    # internally by `average.mix_out` which will raise an error if any of mass,
    # momentum, energy are not balanced

    # Rothalpy should be close to reference value
    # Not exact because we have a small radial span
    assert np.isclose(F_mix.I, I_ref, rtol=1e-3)


def test_uniform():
    """Run a test uniform flow."""

    # Coordinates
    r0, r1 = 1000.0, 1001.0
    rv = np.linspace(r0, r1)
    tv = np.linspace(-np.pi / 8.0, np.pi / 8.0)
    r, t = np.meshgrid(rv, tv, indexing="ij")
    rt = r * t
    r_norm = (r - r0) / (r1 - r0)
    x = np.ones_like(r) * r_norm * 0.2

    # Define a base state that we want to conserve
    rgas = 287.0
    ga = 1.4
    cp = rgas * ga / (ga - 1.0)
    cv = cp / ga
    vx_ref = 100.0
    vr_ref = 0.0
    vt_ref = 100.0
    r_ref = 0.5 * (r0 + r1)
    ro_ref = 1.0
    T_ref = 300.0
    vsq_ref = vx_ref**2.0 + vr_ref**2.0 + vt_ref**2.0

    rovx_ref = ro_ref * vx_ref
    rovr_ref = ro_ref * vr_ref
    rorvt_ref = ro_ref * r_ref * vt_ref

    roe_ref = ro_ref * (cv * (T_ref - Tu0) + 0.5 * vsq_ref)

    Omega = 0.0

    ro = ro_ref * np.ones_like(x)
    rovx = rovx_ref * np.ones_like(x)
    rovr = rovr_ref * np.ones_like(x)
    rorvt = rorvt_ref * np.ones_like(x)

    P_ref = ro_ref * rgas * T_ref
    u_ref = cv * (T_ref - Tu0)
    h_ref = u_ref + P_ref / ro_ref
    ho_ref = h_ref + 0.5 * vsq_ref

    ho = ho_ref * np.ones_like(x)
    P = P_ref * np.ones_like(x)

    vx = vx_ref * np.ones_like(x)
    vr = vr_ref * np.ones_like(x)
    vt = vt_ref * np.ones_like(x)
    T = T_ref * np.ones_like(x)

    # Initialise a flow-field object
    F = turbigen.flowfield.PerfectFlowField.from_properties(
        np.stack((x, r, t)), np.stack((vx, vr, vt)), np.stack((P, T)), cp, ga, MU, Omega
    )

    # Do the mixing
    F_mix = turbigen.average.mix_out(F)[0]

    # Radius should be midspan
    r_mid = np.mean((r.min(), r.max()))
    assert np.isclose(F_mix.r, r_mid)

    # All primary vars should be same
    assert np.isclose(F_mix.rhoVx, rovx_ref)
    assert np.isclose(F_mix.rhoVr, rovr_ref, atol=rovx_ref * 1e-6)
    assert np.isclose(F_mix.rhorVt, rorvt_ref)
    assert np.isclose(F_mix.rhoe, roe_ref)


def test_supersonic():
    """Check we can return a supersonic cut."""

    # Coordinates
    r0, r1 = 1000.0, 1001.0
    rv = np.linspace(r0, r1)
    tv = np.linspace(-np.pi / 8.0, np.pi / 8.0)
    r, t = np.meshgrid(rv, tv, indexing="ij")
    rt = r * t
    r_norm = (r - r0) / (r1 - r0)
    x = np.ones_like(r)

    # Define a base state that we want to conserve
    rgas = 287.0
    ga = 1.4
    cp = rgas * ga / (ga - 1.0)
    cv = cp / ga
    r_ref = 0.5 * (r0 + r1)
    ro_ref = 1.0
    T_ref = 300.0

    Ma_ref = 2.00
    a_ref = np.sqrt(ga * rgas * T_ref)
    vx_ref = Ma_ref * a_ref
    vr_ref = 0.0
    vt_ref = 0.0

    vsq_ref = vx_ref**2.0 + vr_ref**2.0 + vt_ref**2.0

    rovx_ref = ro_ref * vx_ref
    rovr_ref = ro_ref * vr_ref
    rorvt_ref = ro_ref * r_ref * vt_ref
    roe_ref = ro_ref * (cv * (T_ref - Tu0) + 0.5 * vsq_ref)
    Omega = 0.0

    ro = ro_ref * np.ones_like(x)
    rovx = rovx_ref * np.ones_like(x)
    rovr = rovr_ref * np.ones_like(x)
    rorvt = rorvt_ref * np.ones_like(x)

    P_ref = ro_ref * rgas * T_ref
    u_ref = cv * (T_ref - Tu0)
    h_ref = u_ref + P_ref / ro_ref
    ho_ref = h_ref + 0.5 * vsq_ref

    ho = ho_ref * np.ones_like(x)
    P = P_ref * np.ones_like(x)

    vx = vx_ref * np.ones_like(x)
    vr = vr_ref * np.ones_like(x)
    vt = vt_ref * np.ones_like(x)
    T = T_ref * np.ones_like(x)

    # Initialise a flow-field object
    F = turbigen.flowfield.PerfectFlowField.from_properties(
        np.stack((x, r, t)), np.stack((vx, vr, vt)), np.stack((P, T)), cp, ga, MU, Omega
    )

    # Do the mixing
    F_mix = turbigen.average.mix_out(F)[0]

    # Radius should be midspan
    r_mid = np.mean((r.min(), r.max()))
    assert np.isclose(F_mix.r, r_mid)

    # All primary vars should be same
    assert np.isclose(F_mix.rhoVx, rovx_ref)
    assert np.isclose(F_mix.rhoVr, rovr_ref)
    assert np.isclose(F_mix.rhorVt, rorvt_ref)
    assert np.isclose(F_mix.rhoe, roe_ref)


def test_supersonic_radial():
    """Check we can return a supersonic cut."""

    # Coordinates
    xv = np.linspace(0.2, 0.1)
    r_ref = 1.0
    rtv = np.linspace(0.0, 0.1)
    x, rt = np.meshgrid(xv, rtv, indexing="ij")
    r = np.ones_like(x) * r_ref
    t = rt / r

    # Define a base state that we want to conserve
    rgas = 287.0
    ga = 1.4
    cp = rgas * ga / (ga - 1.0)
    cv = cp / ga
    ro_ref = 1.0
    T_ref = 300.0

    Ma_ref = 1.5
    a_ref = np.sqrt(ga * rgas * T_ref)
    vx_ref = 0.0
    vr_ref = Ma_ref * a_ref
    vt_ref = 0.0

    vsq_ref = vx_ref**2.0 + vr_ref**2.0 + vt_ref**2.0

    rovx_ref = ro_ref * vx_ref
    rovr_ref = ro_ref * vr_ref
    rorvt_ref = ro_ref * r_ref * vt_ref
    roe_ref = ro_ref * (cv * (T_ref - Tu0) + 0.5 * vsq_ref)
    Omega = 0.0

    ro = ro_ref * np.ones_like(x)
    rovx = rovx_ref * np.ones_like(x)
    rovr = rovr_ref * np.ones_like(x)
    rorvt = rorvt_ref * np.ones_like(x)

    P_ref = ro_ref * rgas * T_ref
    u_ref = cv * (T_ref - Tu0)
    h_ref = u_ref + P_ref / ro_ref
    ho_ref = h_ref + 0.5 * vsq_ref

    ho = ho_ref * np.ones_like(x)
    P = P_ref * np.ones_like(x)

    vx = vx_ref * np.ones_like(x)
    vr = vr_ref * np.ones_like(x)
    vt = vt_ref * np.ones_like(x)
    T = T_ref * np.ones_like(x)

    # Initialise a flow-field object
    F = turbigen.flowfield.PerfectFlowField.from_properties(
        np.stack((x, r, t)), np.stack((vx, vr, vt)), np.stack((P, T)), cp, ga, MU, Omega
    )

    # Do the mixing
    F_mix = turbigen.average.mix_out(F)[0]

    # Radius should be midspan
    r_mid = np.mean((r.min(), r.max()))
    assert np.isclose(F_mix.r, r_mid)

    assert np.isclose(F_mix.Vr, vr_ref)

    # All primary vars should be same
    assert np.isclose(F_mix.rhoVx, rovx_ref)
    assert np.isclose(F_mix.rhoVr, rovr_ref)
    assert np.isclose(F_mix.rhorVt, rorvt_ref)
    assert np.isclose(F_mix.rhoe, roe_ref)


def test_mixing():
    """Compare to analytical results for a mixed-out flow."""

    # Coordinates
    r0, r1 = 1.0, 2.0
    rv = np.linspace(r0, r1)
    tv = np.linspace(-np.pi / 16.0, np.pi / 32.0, 1000)
    r, t = np.meshgrid(rv, tv, indexing="ij")
    rt = r * t
    x = np.ones_like(r)

    # Gas props
    rgas = 287.0
    ga = 1.4
    cp = rgas * ga / (ga - 1.0)
    cv = cp / ga

    # Inlet conditions
    Po1 = 1e5
    To1 = 300.0
    Po2 = 2e5
    To2 = 400.0
    P12 = 0.8e5
    A1 = 1.0
    A2 = 0.5
    A3 = A1 + A2
    Omega = 0.0

    # Analytical solution
    M1 = cf.Ma_from_Po_P(Po1 / P12, ga)
    M2 = cf.Ma_from_Po_P(Po2 / P12, ga)
    mdot1 = cf.mcpTo_APo_from_Ma(M1, ga) * A1 * Po1 / np.sqrt(cp * To1)
    mdot2 = cf.mcpTo_APo_from_Ma(M2, ga) * A2 * Po2 / np.sqrt(cp * To2)
    mdot3 = mdot1 + mdot2
    V1 = cf.V_cpTo_from_Ma(M1, ga) * np.sqrt(cp * To1)
    V2 = cf.V_cpTo_from_Ma(M2, ga) * np.sqrt(cp * To2)
    mom = P12 * (A1 + A2) + mdot1 * V1 + mdot2 * V2
    To3 = (mdot1 * To1 + mdot2 * To2) / mdot3
    impulse = mom / mdot3 / np.sqrt(cp * To3)

    def F(Ma, F_target):
        To_T = 1.0 + (ga - 1.0) / 2.0 * Ma**2.0
        return (
            np.sqrt(ga - 1.0) / ga / Ma * (1.0 + ga * Ma**2.0) / np.sqrt(To_T)
            - F_target
        )

    M3 = newton(F, M1, args=(impulse,))
    Q3 = cf.mcpTo_APo_from_Ma(M3, ga)
    Po3 = mdot3 * np.sqrt(cp * To3) / A3 / Q3
    P3 = Po3 / cf.Po_P_from_Ma(M3, ga)
    T3 = To3 / cf.To_T_from_Ma(M3, ga)
    V3 = cf.V_cpTo_from_Ma(M3, ga) * np.sqrt(cp * To3)

    # Prepare 2D grid for 12 plane
    P1 = Po1 / cf.Po_P_from_Ma(M1, ga)
    P2 = Po2 / cf.Po_P_from_Ma(M2, ga)
    T1 = To1 / cf.To_T_from_Ma(M1, ga)
    T2 = To2 / cf.To_T_from_Ma(M2, ga)
    P = np.zeros_like(x)
    P[t <= 0.0] = P1
    P[t > 0.0] = P2
    T = np.zeros_like(x)
    T[t <= 0.0] = T1
    T[t > 0.0] = T2
    Vx = np.zeros_like(x)
    Vx[t <= 0.0] = V1
    Vx[t > 0.0] = V2
    Vr = np.zeros_like(Vx)
    Vt = np.zeros_like(Vx)

    # Initialise a flow-field object
    F = turbigen.flowfield.PerfectFlowField.from_properties(
        np.stack((x, r, t)), np.stack((Vx, Vr, Vt)), np.stack((P, T)), cp, ga, MU, Omega
    )

    # Do the mixing
    F_mix = turbigen.average.mix_out(F)[0]

    # Check the mixed-out state
    tol = 1e-3
    assert np.isclose(P3, F_mix.P, rtol=tol)
    assert np.isclose(T3, F_mix.T, rtol=tol)
    assert np.isclose(V3, F_mix.Vx, rtol=tol)


def test_radial():
    """Check that radial mass flux is conserved."""

    # Coordinates
    r0, r1 = 1000.0, 1001.0
    span = r1 - r0
    dt = span / r0 / 2.0
    rv = np.linspace(r0, r1)
    tv = np.linspace(-dt, dt)
    r, t = np.meshgrid(rv, tv, indexing="ij")
    rt = r * t
    r_norm = (r - r0) / (r1 - r0)
    x = np.ones_like(r) * r_norm * -span  # Slanted backwards

    # Define a base state that we want to conserve
    rgas = 287.0
    ga = 1.4
    cp = rgas * ga / (ga - 1.0)
    cv = cp / ga

    vx_ref = 100.0
    rvt_ref = 100.0
    ro_ref = 1.0
    T_ref = 300.0
    Omega = 0.0

    for vr_ref in (50.0, -10.0):
        # vr_ref = 50.0
        vt_ref = rvt_ref / r
        vsq_ref = vx_ref**2.0 + vr_ref**2.0 + vt_ref**2.0

        rovx_ref = ro_ref * vx_ref
        rovr_ref = ro_ref * vr_ref
        rorvt_ref = ro_ref * rvt_ref
        roe_ref = ro_ref * (cv * T_ref + 0.5 * vsq_ref)

        ro = ro_ref * np.ones_like(x)
        rovx = rovx_ref * np.ones_like(x)
        rovr = rovr_ref * np.ones_like(x)
        rorvt = rorvt_ref * np.ones_like(x)

        P_ref = ro_ref * rgas * T_ref
        P = P_ref * np.ones_like(x)

        u_ref = T_ref * cv
        ho_ref = u_ref + P_ref / ro_ref + 0.5 * vsq_ref
        ho = ho_ref * np.ones_like(x)

        vx = vx_ref * np.ones_like(x)
        vr = vr_ref * np.ones_like(x)
        vt = vt_ref * np.ones_like(x)
        T = T_ref * np.ones_like(x)

        # Initialise a flow-field object
        F = turbigen.flowfield.PerfectFlowField.from_properties(
            np.stack((x, r, t)),
            np.stack((vx, vr, vt)),
            np.stack((P, T)),
            cp,
            ga,
            MU,
            Omega,
        )

        # Do the mixing
        F_mix = F.mix_out()[0]  # average.mix_out(F)

        # The mixed-out mass flux should be sum of axial and radial contribs
        assert np.isclose(F_mix.rhoVx + F_mix.rhoVr, rovx_ref + rovr_ref, rtol=1e-4)

        # The mixed-out moment of momentum should be same
        rvt_mix = F_mix.Vt * F_mix.r
        assert np.isclose(rvt_mix, rvt_ref, rtol=1e-4)


def test_radial_inflow():
    """Check that negative radial velocities are handled."""

    # Coordinates
    xv = np.linspace(0.1, 0.2)
    r_ref = 1.0
    rtv = np.linspace(0.1, 0.0)
    x, rt = np.meshgrid(xv, rtv, indexing="ij")
    r = np.ones_like(x) * r_ref
    t = rt / r
    xnorm = (x - x.min()) / np.ptp(x)

    # Define a base state that we want to conserve
    rgas = 287.0
    ga = 1.4
    cp = rgas * ga / (ga - 1.0)
    cv = cp / ga
    ro_ref = 1.0
    T_ref = 300.0
    a_ref = np.sqrt(ga * rgas * T_ref)
    vx_ref = 0.0
    vt_ref = 0.0
    Omega = 0.0

    P_mix = []
    for Ma_ref in (-0.6, 0.6):
        vr_ref = -Ma_ref * a_ref

        vsq_ref = vx_ref**2.0 + vr_ref**2.0 + vt_ref**2.0

        rovx_ref = ro_ref * vx_ref
        rovr_ref = ro_ref * vr_ref
        rorvt_ref = ro_ref * r_ref * vt_ref
        roe_ref = ro_ref * (cv * (T_ref - Tu0) + 0.5 * vsq_ref)

        ro = ro_ref * np.ones_like(x)

        P_ref = ro_ref * rgas * T_ref
        u_ref = cv * (T_ref - Tu0)
        h_ref = u_ref + P_ref / ro_ref
        ho_ref = h_ref + 0.5 * vsq_ref

        ho = ho_ref * np.ones_like(x)
        P = P_ref * np.ones_like(x)

        vx = vx_ref * xnorm**2.0
        vr = vr_ref * np.ones_like(x)
        vt = vt_ref * xnorm
        T = T_ref * np.ones_like(x)

        # Initialise a flow-field object
        F = turbigen.flowfield.PerfectFlowField.from_properties(
            np.stack((x, r, t)),
            np.stack((vx, vr, vt)),
            np.stack((P, T)),
            cp,
            ga,
            MU,
            Omega,
        )

        # Do the mixing
        F_mix = turbigen.average.mix_out(F)[0]

        # Radius should be midspan
        r_mid = np.mean((r.min(), r.max()))
        assert np.isclose(F_mix.r, r_mid)

        assert np.isclose(F_mix.Vr, vr_ref)

        P_mix.append(F_mix.P)

        # All primary vars should be same
        assert np.isclose(F_mix.rhoVx, rovx_ref)
        assert np.isclose(F_mix.rhoVr, rovr_ref)
        assert np.isclose(F_mix.rhorVt, rorvt_ref)
        assert np.isclose(F_mix.rhoe, roe_ref)

    assert np.ptp(P_mix) < 1e-4 * P_mix[0]


def test_cfd():
    # Look for test data in a directory at same level as this script
    DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
    cut_json_path = os.path.join(DATA_DIR, "cfd_cut.json")

    # Load cut data
    with open(cut_json_path, "r") as f:
        cut = json.load(f)
    xrt = np.stack([cut[v] for v in ("x", "r", "t")])
    Vxrt = np.stack([cut[v] for v in ("Vx", "Vr", "Vt")])
    PT = np.stack([cut[v] for v in ("P", "T")])
    cp = cut["cp"]
    ga = cut["ga"]
    mu = cut["mu"]
    Omega = 0.0
    # Initialise a flow-field object
    F = turbigen.flowfield.PerfectFlowField.from_properties(
        xrt, Vxrt, PT, cp, ga, mu, Omega
    )
    F_mix, Aann, _ = F.mix_out()

    # Make sure we have obtained the supersonic solution
    assert F_mix.Mam > 1.0


if __name__ == "__main__":
    # test_supersonic_radial()
    # quit()
    test_cfd()
    # test_mixing()
    test_nonuniform_energy()
    test_nonuniform_xmom()
    test_nonuniform_grid()
    test_uniform()
    test_mixing()
    test_radial()
