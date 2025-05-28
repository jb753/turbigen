"""Tests for new state class."""

from turbigen import state
import numpy as np
import pytest

C = 273.15  # Celsius to Kelvin

tol_h = 1e3
tol_s = 1e0
tol_P = 0.01e5
tol_T = 0.1
epsQ = 1e-6

STATES = [
    state.PerfectFluid(cp=1005.0, gamma=1.4, mu=1.8e-5),
]

METHODS = [k for k in state.BaseFluid.__dict__ if k.startswith("set_")]
print(METHODS)


def test_shape():
    """Ensure the properties of array states have consistent shapes."""

    Dh = 1000e3
    hmin = 2000e3
    Pref = 2e5
    shapes = [(), (1,), (6,), (4, 5), (1, 5, 6), (2, 3, 4)]

    for shape in shapes:
        h = hmin + Dh * np.random.random(shape)
        S = STATES[0].empty(shape=shape)
        S.set_P_h(Pref, h)
        assert S.shape == shape

        # Chech shape is consistent
        for prop in (S.P, S.T, S.s):
            assert np.shape(prop) == shape

        # Check we actually have different properties in the array
        if np.size(S.h) > 1:
            assert np.isclose(np.ptp(S.h), np.ptp(h))
            assert np.ptp(S.s) > 0.0
            for Si in S:
                print(Si.h)


@pytest.mark.parametrize("Sref", STATES)
def test_chain(Sref):
    """We want to be able to use method chaining: all methods return self."""

    S1 = (
        Sref.copy()
        .set_rho_u(Sref.rho, Sref.u)
        .set_h_s(Sref.h, Sref.s)
        .set_P_T(Sref.P, Sref.T)
        .set_P_s(Sref.P, Sref.s)
        .set_P_h(Sref.P, Sref.h)
        .set_P_rho(Sref.P, Sref.rho)
    )
    assert S1 is not None


@pytest.mark.parametrize("S1", STATES)
def test_set_properties(S1):
    S1.set_P_T(2e5, 350.0)

    S2 = S1.copy()

    S2.set_P_T(S1.P, S1.T)
    assert np.isclose(S1.P, S2.P)
    assert np.isclose(S1.T, S2.T)

    S2.set_P_s(S1.P, S1.s)
    assert np.isclose(S1.P, S2.P)
    assert np.isclose(S1.s, S2.s)

    S2.set_h_s(S1.h, S1.s)
    assert np.isclose(S1.h, S2.h)
    assert np.isclose(S1.s, S2.s)

    S2.set_P_h(S1.P, S1.h)
    assert np.isclose(S1.P, S2.P)
    assert np.isclose(S1.h, S2.h)

    S2.set_rho_u(S1.rho, S1.u)
    assert np.isclose(S1.rho, S2.rho)
    assert np.isclose(S1.u, S2.u)

    S2.set_P_rho(S1.P, S1.rho)
    assert np.isclose(S1.P, S2.P)
    assert np.isclose(S1.rho, S2.rho)


@pytest.mark.parametrize("S1", STATES)
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
    S1 = state.PerfectFluid(cp=cp, gamma=ga)
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
    rho1 = 2.8
    P1 = 2.1e5
    delta = np.linspace(0.8, 1.2, 50)
    Pv = delta * P1
    rhov = delta * rho1

    S1 = state.PerfectFluid(cp=cp, gamma=ga, shape=delta.shape)

    rtol = 1e-3

    # by rho first at constant P
    S1.set_P_rho(P1, rhov)
    S1.set_Vxrt(100, 0, 100)
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


if __name__ == "__main__":
    test_perfect_deriv()
