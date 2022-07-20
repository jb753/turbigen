"""Check averaging is conserving what we want."""

import numpy as np
import turbigen.compflow as cf
from turbigen import average
from scipy.optimize import newton


def test_nonuniform_energy():
    """Run a test flow with nonuniform energy."""
    # Coordinates
    r0, r1 = 1.0, 2.0
    rv = np.linspace(1.0, 2.0)
    tv = np.linspace(-np.pi / 8.0, np.pi / 8.0)
    r, t = np.meshgrid(rv, tv, indexing="ij")
    t = t + np.pi / 64 * r / r0  # Warp the grid
    rt = r * t
    r_norm = (r - r0) / (r1 - r0)
    x = np.ones_like(r) * r_norm * 0.2  # Angled cut plane

    # Define a base state that we want to conserve
    rgas = 287.0
    ga = 1.4
    cp, cv = average.specific_heats(ga, rgas)
    rovx_ref = 200.0
    xmom_ref = 1.1e5
    rtmom_ref = 1.5e4
    Omega = 200.0

    # Make the non-uniform flow
    ro0 = 1.0
    ro = ro0 * (1.0 + 0.5 * r_norm)
    vx = rovx_ref / ro
    vr = np.zeros_like(vx)
    P = xmom_ref - ro * vx ** 2.0
    T = P / rgas / ro
    vt = rtmom_ref / ro / vx / r

    # Convert to primary vars
    ro, rovx, rovr, rorvt, roe = average.secondary_to_primary(
        r, vx, vr, vt, P, T, ga, rgas
    )

    # Do the mixing
    r_mix, ro_mix, rovx_mix, rovr_mix, rorvt_mix, roe_mix = average.mix_out(
        x, r, rt, ro, rovx, rovr, rorvt, roe, ga, rgas, Omega
    )

    # Secondary mixed vars
    vx_mix, vr_mix, vt_mix, P_mix, T_mix = average.primary_to_secondary(
        r_mix, ro_mix, rovx_mix, rovr_mix, rorvt_mix, roe_mix, ga, rgas
    )

    # Radius should be midspan
    r_mid = np.mean((r.min(), r.max()))
    assert np.isclose(r_mix, r_mid)

    # Mass flux should be reference value
    assert np.isclose(rovx_mix, rovx_ref)

    # Axial momentum should be reference value
    assert np.isclose(P_mix + ro_mix * vx_mix ** 2.0, xmom_ref)

    # Moment of angular momentum should be reference value
    assert np.isclose(ro_mix * vx_mix * r_mix * vt_mix, rtmom_ref)


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
    cp, cv = average.specific_heats(ga, rgas)
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
    To = (I_ref + rmid * Omega * vt) / cp
    vsq = vx ** 2.0 + vr ** 2.0 + vt ** 2.0
    v_cpTo = np.sqrt(vsq / cp / To)
    Ma = cf.Ma_from_V_cpTo(v_cpTo, ga)
    T = To / cf.To_T_from_Ma(Ma, ga)
    P = ro * rgas * T

    # Convert to primary vars
    ro, rovx, rovr, rorvt, roe = average.secondary_to_primary(
        r, vx, vr, vt, P, T, ga, rgas
    )

    # Do the mixing
    r_mix, ro_mix, rovx_mix, rovr_mix, rorvt_mix, roe_mix = average.mix_out(
        x, r, rt, ro, rovx, rovr, rorvt, roe, ga, rgas, Omega
    )

    # Secondary mixed vars
    vx_mix, vr_mix, vt_mix, P_mix, T_mix = average.primary_to_secondary(
        r_mix, ro_mix, rovx_mix, rovr_mix, rorvt_mix, roe_mix, ga, rgas
    )

    vsq_mix = vx_mix ** 2.0 + vr_mix ** 2.0 + vt_mix ** 2.0
    Ma_mix = np.sqrt(vsq_mix / ga / rgas / T_mix)
    To_mix = T_mix * cf.To_T_from_Ma(Ma_mix, ga)

    # Radius should be midspan
    r_mid = np.mean((r.min(), r.max()))
    assert np.isclose(r_mix, r_mid)

    # Mass flux should be reference value
    assert np.isclose(rovx_mix, rovx_ref)

    # Moment of angular momentum should be reference value
    assert np.isclose(ro_mix * vx_mix * r_mix * vt_mix, rtmom_ref)

    # Rothalpy should be close to reference value
    # Not exact because we have a small radial span
    I_mix = cp * To_mix - r_mix * Omega * vt_mix
    assert np.isclose(I_mix, I_ref, rtol=1e-3)


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
    cp, cv = average.specific_heats(ga, rgas)
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
    To = (I_ref + rmid * Omega * vt) / cp
    vsq = vx ** 2.0 + vr ** 2.0 + vt ** 2.0
    v_cpTo = np.sqrt(vsq / cp / To)
    Ma = cf.Ma_from_V_cpTo(v_cpTo, ga)
    T = To / cf.To_T_from_Ma(Ma, ga)
    P = ro * rgas * T

    # Convert to primary vars
    ro, rovx, rovr, rorvt, roe = average.secondary_to_primary(
        r, vx, vr, vt, P, T, ga, rgas
    )

    # Do the mixing
    r_mix, ro_mix, rovx_mix, rovr_mix, rorvt_mix, roe_mix = average.mix_out(
        x, r, rt, ro, rovx, rovr, rorvt, roe, ga, rgas, Omega
    )

    # Secondary mixed vars
    vx_mix, vr_mix, vt_mix, P_mix, T_mix = average.primary_to_secondary(
        r_mix, ro_mix, rovx_mix, rovr_mix, rorvt_mix, roe_mix, ga, rgas
    )

    vsq_mix = vx_mix ** 2.0 + vr_mix ** 2.0 + vt_mix ** 2.0
    Ma_mix = np.sqrt(vsq_mix / ga / rgas / T_mix)
    To_mix = T_mix * cf.To_T_from_Ma(Ma_mix, ga)

    # Radius should be midspan
    r_mid = np.mean((r.min(), r.max()))
    assert np.isclose(r_mix, r_mid)

    # Mass flux should be reference value
    assert np.isclose(rovx_mix, rovx_ref)

    # Moment of angular momentum should be reference value
    assert np.isclose(ro_mix * vx_mix * r_mix * vt_mix, rtmom_ref)

    # Rothalpy should be close to reference value
    # Not exact because we have a small radial span
    I_mix = cp * To_mix - r_mix * Omega * vt_mix
    assert np.isclose(I_mix, I_ref, rtol=1e-3)
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
    cp, cv = average.specific_heats(ga, rgas)

    vx_ref = 100.0
    vr_ref = 0.0
    vt_ref = 100.0
    r_ref = 1000.0
    ro_ref = 1.0
    T_ref = 300.0
    vsq_ref = vx_ref ** 2.0 + vr_ref ** 2.0 + vt_ref ** 2.0

    rovx_ref = ro_ref * vx_ref
    rovr_ref = ro_ref * vr_ref
    rorvt_ref = ro_ref * r_ref * vt_ref
    roe_ref = ro_ref * (cv * T_ref + 0.5 * vsq_ref)
    Omega = 0.0

    ro = ro_ref * np.ones_like(x)
    rovx = rovx_ref * np.ones_like(x)
    rovr = rovr_ref * np.ones_like(x)
    rorvt = rorvt_ref * np.ones_like(x)
    roe = roe_ref * np.ones_like(x)

    # Do the mixing
    r_mix, ro_mix, rovx_mix, rovr_mix, rorvt_mix, roe_mix = average.mix_out(
        x, r, rt, ro, rovx, rovr, rorvt, roe, ga, rgas, Omega
    )

    # Secondary mixed vars
    vx_mix, vr_mix, vt_mix, P_mix, T_mix = average.primary_to_secondary(
        r_mix, ro_mix, rovx_mix, rovr_mix, rorvt_mix, roe_mix, ga, rgas
    )

    # Radius should be midspan
    r_mid = np.mean((r.min(), r.max()))
    assert np.isclose(r_mix, r_mid)

    # All primary vars should be same
    assert np.isclose(rovx_mix, rovx_ref)
    assert np.isclose(rovr_mix, rovr_ref)
    assert np.isclose(rorvt_mix, rorvt_ref)
    assert np.isclose(roe_mix, roe_ref)


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
    cp, cv = average.specific_heats(ga, rgas)

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
        To_T = 1.0 + (ga - 1.0) / 2.0 * Ma ** 2.0
        return (
            np.sqrt(ga - 1.0) / ga / Ma * (1.0 + ga * Ma ** 2.0) / np.sqrt(To_T)
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
    P = np.empty_like(x)
    P[t <= 0.0] = P1
    P[t > 0.0] = P2
    T = np.empty_like(x)
    T[t <= 0.0] = T1
    T[t > 0.0] = T2
    Vx = np.empty_like(x)
    Vx[t <= 0.0] = V1
    Vx[t > 0.0] = V2
    Vr = np.zeros_like(Vx)
    Vt = np.zeros_like(Vx)

    # Convert to primarys
    ro, rovx, rovr, rorvt, roe = average.secondary_to_primary(
        r, Vx, Vr, Vt, P, T, ga, rgas
    )

    # Do the mixing
    r_mix, ro_mix, rovx_mix, rovr_mix, rorvt_mix, roe_mix = average.mix_out(
        x, r, rt, ro, rovx, rovr, rorvt, roe, ga, rgas, Omega
    )

    # Secondary mixed vars
    vx_mix, vr_mix, vt_mix, P_mix, T_mix = average.primary_to_secondary(
        r_mix, ro_mix, rovx_mix, rovr_mix, rorvt_mix, roe_mix, ga, rgas
    )

    # Check the mixed-out state
    tol = 1e-3
    assert np.isclose(P3, P_mix, rtol=tol)
    assert np.isclose(T3, T_mix, rtol=tol)
    assert np.isclose(V3, vx_mix, rtol=tol)


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
    cp, cv = average.specific_heats(ga, rgas)

    vx_ref = 100.0
    vr_ref = 50.0
    rvt_ref = 100.0
    vt_ref = rvt_ref / r
    ro_ref = 1.0
    T_ref = 300.0
    vsq_ref = vx_ref ** 2.0 + vr_ref ** 2.0 + vt_ref ** 2.0

    rovx_ref = ro_ref * vx_ref
    rovr_ref = ro_ref * vr_ref
    rorvt_ref = ro_ref * rvt_ref
    roe_ref = ro_ref * (cv * T_ref + 0.5 * vsq_ref)
    Omega = 0.0

    ro = ro_ref * np.ones_like(x)
    rovx = rovx_ref * np.ones_like(x)
    rovr = rovr_ref * np.ones_like(x)
    rorvt = rorvt_ref * np.ones_like(x)
    roe = roe_ref * np.ones_like(x)

    # Do the mixing
    r_mix, ro_mix, rovx_mix, rovr_mix, rorvt_mix, roe_mix = average.mix_out(
        x, r, rt, ro, rovx, rovr, rorvt, roe, ga, rgas, Omega
    )

    # Secondary mixed vars
    vx_mix, vr_mix, vt_mix, P_mix, T_mix = average.primary_to_secondary(
        r_mix, ro_mix, rovx_mix, rovr_mix, rorvt_mix, roe_mix, ga, rgas
    )

    # The mixed-out mass flux should be sum of axial and radial contribs
    assert np.isclose(rovx_mix, rovx_ref + rovr_ref)

    # The mixed-out moment of momentum should be same
    rvt_mix = vt_mix * r_mix
    assert np.isclose(rvt_mix, rvt_ref)
