"""Tests for thermodynamic properties of working fluids."""

from turbigen import fluid, flowfield, util
import numpy as np
import pytest

C = 273.15  # Celsius to Kelvin

tol_h = 1e3
tol_s = 1e0
tol_P = 0.01e5
tol_T = 0.1
epsQ = 1e-6

states = [
    fluid.PerfectState.from_properties(cp=1005.0, gamma=1.4, mu=1.8e-5),
    fluid.RealState.from_fluid_name("Air"),
    fluid.RealState.from_fluid_name("R134a"),
]


def test_databook():
    """Properties for R134a should match databook values."""

    fluid_name = "R134a"

    Tsat = np.array([-45.0, 0.0, 50.0]) + C
    hsat_f = np.array([141.9, 200.0, 271.6]) * 1e3
    hsat_g = np.array([370.8, 398.6, 423.4]) * 1e3
    ssat_f = np.array([0.7687, 1.0, 1.2374]) * 1e3
    ssat_g = np.array([1.7722, 1.7270, 1.7073]) * 1e3
    Psat = np.array([0.39, 2.93, 13.18]) * 1e5
    hsup = np.array([385.8, 416.4, 446.7]) * 1e3
    ssup = np.array([1.8348, 1.7900, 1.7772]) * 1e3
    DTsup = 20.0

    S = fluid.RealState.from_fluid_name(fluid_name)

    for (
        Tsat_i,
        hsat_f_i,
        hsat_g_i,
        ssat_f_i,
        ssat_g_i,
        Psat_i,
        hsup_i,
        ssup_i,
    ) in zip(Tsat, hsat_f, hsat_g, ssat_f, ssat_g, Psat, hsup, ssup):
        # Check saturated liquid properties
        S.set_T_chi(Tsat_i, 0.0)
        assert np.isclose(S.h, hsat_f_i, atol=tol_h)
        assert np.isclose(S.s, ssat_f_i, atol=tol_s)
        assert np.isclose(S.P, Psat_i, atol=tol_P)

        # Check saturated vapour properties
        S.set_T_chi(Tsat_i, 1.0 - epsQ)
        assert np.isclose(S.h, hsat_g_i, atol=tol_h)
        assert np.isclose(S.s, ssat_g_i, atol=tol_s)
        assert np.isclose(S.P, Psat_i, atol=tol_P)

        # Check T
        S.set_P_h(Psat_i, hsat_g_i)
        assert np.isclose(S.T, Tsat_i, atol=tol_T)

        # Check superheat
        S.set_P_T(Psat_i, Tsat_i + DTsup)
        assert np.isclose(S.h, hsup_i, atol=tol_h)
        assert np.isclose(S.s, ssup_i, atol=tol_s)
        assert np.isclose(S.DTsuperheat, DTsup, atol=tol_T)

        # Check saturation properties
        assert np.isclose(S.Tsat, Tsat_i, atol=tol_T)
        assert np.isclose(S.hsat_vapour, hsat_g_i, atol=tol_h)


def test_phase_indicators():
    """Check that the phase indicators work."""

    S = fluid.RealState.from_fluid_name("water")

    Pref = 1e5
    dh = 200e3

    # Two phase
    S.set_P_chi(Pref, 0.5)
    assert S.is_two_phase and (not S.is_liquid) and (not S.is_gas)
    assert not S.is_supercritical

    # Superheated vapour
    S.set_P_chi(Pref, 1.0)
    S.set_P_h(Pref, S.h + dh)
    assert S.is_gas
    assert (not S.is_two_phase) and (not S.is_liquid)
    assert not S.is_supercritical

    # Subcooled liquid
    S.set_P_chi(Pref, 0.0)
    S.set_P_h(Pref, S.h - dh)
    assert S.is_liquid
    assert (not S.is_two_phase) and (not S.is_gas)
    assert not S.is_supercritical

    # Supercritical
    S.set_P_h(250e5, 2500e3)
    assert (not S.is_liquid) and (not S.is_two_phase) and (not S.is_gas)
    assert S.is_supercritical


@pytest.mark.parametrize("S1", states)
def test_equals(S1):
    """Check that a state equals itself initialised with different props."""
    Tref = 360.0
    Pref = 1.6e5

    # S1 = fluid.RealState.from_fluid_name("Air")
    S1a = S1.copy()  # Can copy an empty state
    S1.set_P_T(Pref, Tref)

    S2 = S1.copy()
    S2.set_P_h(S1.P, S1.h)

    print(S1.P, S1.h)
    print(S2.P, S2.h)
    assert S1 == S2

    S3 = S1.copy()
    S3.set_P_h(S1.P, S1.h + 200e3)

    S4 = S1.copy()

    assert S1 == S4
    assert not (S1 == S3)

    assert not (S1 == None)
    S5 = fluid.RealState()
    S5.fluid_name = "Hydrogen"
    S5.set_P_T(Pref, Tref)
    assert not (S1 == S5)


def test_shape():
    """Ensure the properties of array states have consistent shapes."""

    Dh = 1000e3
    hmin = 2000e3
    Pref = 2e5
    shapes = [(), (1,), (6,), (4, 5), (1, 5, 6), (2, 3, 4)]

    for shape in shapes:
        h = hmin + Dh * np.random.random(shape)
        S = fluid.RealState.from_fluid_name("water", shape=shape)
        S.set_P_h(Pref, h)
        assert S.shape == shape

        # Chech shape is consistent
        for prop in (S.P, S.T, S.s):
            assert np.shape(prop) == shape

        # Check we actually have different properties in the array
        if np.size(S.h) > 1:
            assert np.isclose(np.ptp(S.h), np.ptp(h))
            assert np.ptp(S.s) > 0.0


def test_superheat():
    """Verify continuous superheat calculations."""

    Pref = 5e5
    s = np.linspace(6e3, 8e3)
    S = fluid.RealState.from_fluid_name("water", shape=s.shape)
    S.set_P_s(Pref, s)
    assert np.all(np.diff(S.DTsuperheat) > 0.0)
    assert np.all(S.DTsuperheat[S.is_two_phase] < 0.0)
    assert np.all(S.DTsuperheat[S.is_gas] > 0.0)

    S.set_P_chi(Pref, 1.0)
    assert np.allclose(S.DTsuperheat, 0.0)


def test_P_Tcrit():
    """Critcial temperature should match databook values."""
    Tc = [
        647.1,
    ]
    Pc = [2.2064e7]
    fluids = [
        "water",
    ]
    for f, T, P in zip(fluids, Tc, Pc):
        S = fluid.RealState.from_fluid_name(f)
        assert np.isclose(S.Tcrit, T, atol=tol_T)
        assert np.isclose(S.Pcrit, P, atol=tol_P)


def test_repr():
    """Make sure that we can always print the state successfully."""

    Pref = 1e5
    Tref = 300.0

    S = fluid.RealState()
    print(S)
    S.fluid_name = "water"
    print(S)
    S.set_P_T(Pref, Tref)
    print(S)
    S = fluid.RealState.from_fluid_name("water", shape=(1,))
    print(S)
    S.set_P_T(np.ones(1) * Pref, Tref)
    print(S)
    S = fluid.RealState.from_fluid_name("water", shape=(4,))
    print(S)
    S.set_P_T(np.ones(4) * Pref, Tref)
    print(S)

    dh = 100e3
    S.set_P_chi(Pref, 0.0)
    S.set_P_h(Pref, S.h - dh)
    print(S)  # Liquid
    S.set_P_chi(Pref, 0.5)
    print(S)  # 2-phase
    S.set_P_chi(Pref, 1.0)
    S.set_P_h(Pref, S.h + dh)
    print(S)  # Vapour
    S.set_P_h(S.Pcrit * 1.1, S.h)
    print(S)  # Supercritical


def test_props():
    """Check that thermodynamic properties are as expected."""

    fluid_name = "water"
    S = fluid.RealState.from_fluid_name(fluid_name)
    assert S.fluid_name == fluid_name
    S.set_P_T(1e5, 300.0)
    cp_ref = 4.18e3
    assert np.isclose(S.cp, cp_ref, atol=tol_s)


def test_chain():
    """We want to be able to use method chaining: all methods return self."""

    Sref = fluid.RealState.from_fluid_name("water")
    Sref.set_P_chi(1e5, 0.5)

    # Try method chaining on a 2-phase state
    S1 = (
        Sref.copy()
        .set_P_chi(Sref.P, Sref.chi)
        .set_P_h(Sref.P, Sref.h)
        .set_P_s(Sref.P, Sref.s)
        .set_T_chi(Sref.T, Sref.chi)
        .set_T_s(Sref.T, Sref.s)
    )
    assert S1 is not None

    # Try on a superheated
    # Try method chaining on a 2-phase state
    Sref.set_P_T(1e5, 500.0)
    S2 = (
        Sref.copy()
        .set_P_T(Sref.P, Sref.T)
        .set_P_h(Sref.P, Sref.h)
        .set_P_s(Sref.P, Sref.s)
        .set_T_s(Sref.T, Sref.s)
    )
    assert S2 is not None


@pytest.mark.parametrize("Sstag", states)
def test_static_stagnation(Sstag):
    """Check that static state conserves energy with correct Mach."""

    Sstag.set_P_T(2e5, 300.0)

    for Ma in (0.0, 1e-3, 0.1, 0.6, 1.0, 1.4):
        Sstat = Sstag.to_static(Ma)
        V = Sstat.a * Ma

        assert np.isclose(Sstat.h + 0.5 * V**2.0, Sstag.h)
        assert np.isclose(Sstat.s, Sstag.s)

    Sstag_out = Sstat.to_stagnation(Ma)
    assert Sstag_out == Sstag


def test_iter():
    Pref = 1e5
    T1 = np.array([400.0, 500.0, 600.0])
    T2 = T1.reshape(-1, 1)
    T3 = np.array(([400.0, 450.0], [500.0, 550.0]))

    # 1D row
    S1 = fluid.RealState.from_fluid_name("Air", shape=T1.shape).set_P_T(Pref, T1)
    for s, T in zip(S1, T1):
        assert np.isclose(T, s.T)

    # 1D col
    S2 = fluid.RealState.from_fluid_name("Air", shape=T2.shape).set_P_T(Pref, T2)
    for s, T in zip(S2, T2):
        assert np.isclose(T, s.T)

    # 2D matrix
    S3 = fluid.RealState.from_fluid_name("Air", shape=T3.shape).set_P_T(Pref, T3)
    for s, T in zip(S3, T3):
        assert np.isclose(T, s.T).all()


@pytest.mark.parametrize("S1", states)
def test_set_properties(S1):
    S1.set_P_T(2e5, 350.0)

    S2 = S1.copy()

    S2.set_P_T(S1.P, S1.T)
    assert S1 == S2

    S2.set_P_s(S1.P, S1.s)
    assert S1 == S2

    S2.set_h_s(S1.h, S1.s)
    assert S1 == S2

    S2.set_T_s(S1.T, S1.s)
    assert S1 == S2

    S2.set_P_h(S1.P, S1.h)
    assert S1 == S2

    S2.set_rho_u(S1.rho, S1.u)
    assert S1 == S2

    S2.set_rho_s(S1.rho, S1.s)
    assert S1 == S2

    S2.set_P_rho(S1.P, S1.rho)
    assert S1 == S2

    Ma_ref = 0.6
    So2 = S2.to_stagnation(Ma_ref)
    S2a = So2.to_static(Ma_ref)
    assert S2 == S2a


@pytest.mark.parametrize("S1", states)
def test_thermo_properties(S1):
    """Check that universal relations between thermodynamic properties are satisfied"""
    # print(S1.h)
    # print( S1.u + S1.P / S1.rho)
    assert np.isclose(S1.h, S1.u + S1.P / S1.rho)
    assert np.isclose(S1.gamma, S1.cp / S1.cv)


def test_perfect():
    """Check that perfect gas analytical relations are satisfied"""

    cp = 1105.0
    ga = 1.3
    S1 = fluid.PerfectState.from_properties(cp, ga, mu=1.8e-5)
    S1.set_P_T(1e5, 300.0)
    S2 = S1.copy().set_P_T(2e5, 400.0)

    assert np.isclose(S1.P, S1.rho * S1.rgas * S1.T)
    assert np.isclose(S1.rgas, S1.cp - S1.cv)
    assert np.isclose(S1.a, np.sqrt(S1.gamma * S1.rgas * S1.T))
    assert np.isclose(S2.u - S1.u, S1.cv * (S2.T - S1.T))
    assert np.isclose(S2.h - S1.h, cp * (S2.T - S1.T))
    assert np.isclose(
        S2.s - S1.s, cp * np.log(S2.T / S1.T) - S1.rgas * np.log(S2.P / S1.P)
    )

    S2s = S1.copy().set_P_s(S2.P, S1.s)
    gae = (ga - 1.0) / ga
    assert np.isclose(S2s.T, S1.T * (S2.P / S1.P) ** gae)


def test_Tu0():
    """Check that perfect gas Tu0 datum changes are transparent."""

    cp = 1105.0
    ga = 1.3
    T1 = 400.0
    P1 = 1e5

    S1 = fluid.PerfectState.from_properties(cp, ga, mu=1.8e-5)
    S1.set_P_T(T1, P1)
    S1c = S1.copy()
    for Tu0 in [0.0, 100.0, 300.0, 500.0]:
        S1c.set_Tu0(Tu0)
        assert np.isclose(S1.T, S1c.T)
        assert np.isclose(S1.P, S1c.P)


def test_perfect_deriv():
    """Check that perfect gas derivatives are correct by finite difference"""

    cp = 1105.0
    ga = 1.3
    rho1 = 2.0
    P1 = 2e5
    delta = np.linspace(0.8, 1.2)
    N = len(delta)
    Pv = delta * P1
    rhov = delta * rho1

    S1 = flowfield.PerfectFlowField(shape=delta.shape)
    S1.mu = 1.84e-5
    S1.cp = cp
    S1.gamma = ga
    S1.Vxrt = np.ones((3, N)) * 100.0
    S1.Vr = 0.0
    S1.xrt = np.ones((3, N))

    rtol = 1e-3

    # by rho first at constant P
    S1.set_P_rho(P1, rhov)
    dsdrho = np.gradient(S1.s, rhov)
    dhdrho = np.gradient(S1.h, rhov)
    dudrho = np.gradient(S1.u, rhov)
    drhoe_drho = np.gradient(S1.rhoe, rhov)
    assert np.allclose(S1.dsdrho_P[1:-1], dsdrho[1:-1], rtol=rtol)
    assert np.allclose(S1.dhdrho_P[1:-1], dhdrho[1:-1], rtol=rtol)
    assert np.allclose(S1.dudrho_P[1:-1], dudrho[1:-1], rtol=rtol)
    assert np.allclose(S1.drhoe_drho_P[1:-1], drhoe_drho[1:-1], rtol=rtol)

    # by P first at constant rho
    S1.set_P_rho(Pv, rho1)
    dsdP = np.gradient(S1.s, Pv)
    dhdP = np.gradient(S1.h, Pv)
    dudP = np.gradient(S1.u, Pv)
    drhoe_dP = np.gradient(S1.rhoe, Pv)
    assert np.allclose(S1.dsdP_rho[1:-1], dsdP[1:-1], rtol=rtol)
    assert np.allclose(S1.dhdP_rho[1:-1], dhdP[1:-1], rtol=rtol)
    assert np.allclose(S1.dudP_rho[1:-1], dudP[1:-1], rtol=rtol)
    assert np.allclose(S1.drhoe_dP_rho[1:-1], drhoe_dP[1:-1], rtol=rtol)


def test_matrices():
    # Set up two flow fields with a small perturbation between them

    mag = 1e-5
    tol = 1e-2
    perturbations = [
        [1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0, 1.0, 1.0],
        [-1.0, -1.0, -1.0, -1.0, -1.0],
        [-0.7, 0.9, -0.7, 0.8, 1.0],
        [-1.2, 3.5, -0.7, 4.1, 2.2],
    ]

    F = flowfield.PerfectFlowField(shape=(1,))
    F.cp = 1105.0
    F.gamma = 1.3
    F.mu = 1.8e-5
    F.xrt = 2 * np.ones((3, 1))
    F.Vxrt = [[[100.0], [80.0], [50.0]]]
    F.set_P_T(1.2e5, 295.0)
    F2 = F.copy()

    for fac_prim in perturbations:
        dprim = F.prim * np.array(fac_prim)[..., None] * mag
        F2.set_prim(F.prim + dprim)

        # Check conserved
        dcons = F2.conserved - F.conserved
        C = F.primitive_to_conserved
        Cinv = F.conserved_to_primitive
        assert np.allclose(dcons, C @ dprim, rtol=tol)
        assert np.allclose(np.diag((Cinv @ C).squeeze()), 1.0, rtol=1e-6)

        # Manually calculate chic vector
        dp = F2.P - F.P
        dVx = F2.Vx - F.Vx
        dVr = F2.Vr - F.Vr
        dVt = F2.Vt - F.Vt
        drho = F2.rho - F.rho
        a = 0.5 * (F2.a + F.a)
        rho = 0.5 * (F2.rho + F.rho)
        dchic = [
            dp - rho * a * dVx,
            dp + rho * a * dVx,
            rho * a * dVr,
            rho * a * dVt,
            dp - (a**2) * drho,
        ]

        # Check prim to chic
        B = F.primitive_to_chic
        Binv = F.chic_to_primitive
        assert np.allclose(np.diag((Binv @ B).squeeze()), 1.0, rtol=1e-6)
        assert np.allclose(dchic, B @ dprim, rtol=tol)

        # Check prim to fluxes
        dflux = F2.fluxes - F.fluxes
        A = F.primitive_to_flux
        Ainv = F.flux_to_primitive
        assert np.allclose(np.diag((Ainv @ A).squeeze()), 1.0, rtol=1e-6)
        assert np.allclose(dflux, A @ dprim, rtol=tol)

        # Check prim to bcond
        dbcond = F2.bcond - F.bcond
        Y = F.primitive_to_bcond
        Yinv = F.bcond_to_primitive
        assert np.allclose(np.diag((Yinv @ Y).squeeze()), 1.0, rtol=1e-6)
        assert np.allclose(dbcond, Y @ dprim, rtol=tol)

        # Check chic to conserved
        X = F.chic_to_conserved
        Xinv = F.conserved_to_chic
        assert np.allclose(np.diag((Xinv @ X).squeeze()), 1.0, rtol=1e-6)
        assert np.allclose(dcons, X @ dchic, rtol=tol)

        # Check inverses if no zeros in dprim
        if not (dprim.squeeze() == 0.0).any():
            assert np.allclose(dprim, Yinv @ dbcond, rtol=tol)
            assert np.allclose(dprim, Ainv @ dflux, rtol=tol)
            assert np.allclose(dprim, Binv @ dchic, rtol=tol)
            assert np.allclose(dprim, Cinv @ dcons, rtol=tol)

            # Check chic to conserved
            assert np.allclose(dchic, Xinv @ dcons, rtol=tol)

    print("ok")


def test_chic_waves():
    """Set up a flow field with travelling waves and check chics recovered."""

    cp = 1005.0
    ga = 1.4
    mu = 1.84e-5
    Vx = 100.0
    L = 1.0
    f = 500.0

    # Set up coordinates
    ni = 50
    nt = 100
    xv = np.linspace(0.0, L, ni)
    xrt = np.stack((xv, np.ones_like(xv), np.zeros_like(xv)))
    omega = 2 * np.pi * f
    t = np.linspace(0.0, 1 / f, nt, endpoint=False)[None, :]
    dt = np.diff(t)[0]
    x = xv[:, None]

    # Mean flow field first
    F = flowfield.PerfectFlowField(shape=(ni, nt))
    F.cp = cp
    F.gamma = ga
    F.mu = mu
    F.xrt = xrt[..., None]
    F.Vx = Vx
    F.Vr = 0.0
    F.Vt = 0.0
    F.set_P_T(1e5, 300.0)

    # Prescribe pressure wave
    Aup = 1e-3
    Adn = 2.2e-3
    a0 = F.a
    rho0 = F.rho
    dPdn = Adn * np.exp(1j * omega * (t - x / a0))
    dPup = Aup * np.exp(1j * omega * (t + x / a0))
    dP = np.real(dPdn + dPup)

    # Momentum for velocity
    # du/dt = -1/rho dp/dx
    dV = np.real(dPdn - dPup) / rho0 / a0

    # Apply to the flowfield
    F.set_P_s(F.P + dP, F.s)
    F.Vx = F.Vx + dV

    # Get changes over a time step
    dU = np.moveaxis(np.diff(F.conserved, axis=-1), 0, -1)[..., None]

    # Convert to chics
    dchic = F.conserved_to_chic[:, :-1, :, :] @ dU

    # The zero-to-peak amplitudes should match the ratio of up to down waves
    # Neglecting any scaling factors
    Amp_x = np.mean(np.ptp(dchic, axis=0)[:, :2, 0], axis=0)
    Amp_t = np.mean(np.ptp(dchic, axis=1)[:, :2, 0], axis=0)

    assert np.isclose(Amp_x[0] / Amp_x[1], Aup / Adn)
    assert np.isclose(Amp_t[0] / Amp_t[1], Aup / Adn)

    return


if __name__ == "__main__":
    # np.set_printoptions(precision=2)
    test_matrices()
