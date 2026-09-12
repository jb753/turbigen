"""Tests for isentropic efficiency (eta_tt and eta_ts) on MeanLine.

Verifies the total-to-total and total-to-static efficiencies against analytical
expressions for a perfect gas. Efficiency is defined between the machine
endpoints, ``ml.inlet`` and ``ml.outlet``, which are the first and last
stations in streamwise order regardless of how many rows lie between them.
"""

import ember.fluid
import numpy as np
import pytest
import turbigen_ref.meanline_new

GAMMA = 1.4
CP = 1005.0
KAPPA = (GAMMA - 1.0) / GAMMA  # (gamma - 1) / gamma


@pytest.fixture
def perfect_air():
    return ember.fluid.PerfectFluid(cp=CP, gamma=GAMMA, mu=1.8e-5, Pr=0.72)


def T_isentropic(T_in, PR):
    """Outlet temperature of an isentropic process at pressure ratio P_out/P_in."""
    return T_in * PR**KAPPA


def T_for_efficiency(T_in, PR, eta):
    """Outlet temperature of an expansion of the given isentropic efficiency."""
    return T_in - eta * (T_in - T_isentropic(T_in, PR))


def make_machine(fluid, states):
    """Build a mean line from (P, T, Vx) at each station in streamwise order.

    Two states make a single row, four make two rows, and so on. Axial flow
    throughout, constant radius and area, so nothing but the thermodynamic
    state and the axial velocity varies.
    """
    if len(states) % 2:
        raise ValueError("need an even number of stations, two per row")

    ml = turbigen_ref.meanline_new.MeanLine(len(states) // 2)
    ml.set_fluid(fluid)
    ml.set_r(0.5)
    ml.set_Am(1.0)

    P, T, Vx = (np.array(x, dtype=float) for x in zip(*states))

    # The flat view runs from machine inlet to machine outlet and writes
    # straight through to the parent.
    flat = ml.flat
    flat.set_P_T(P, T)
    flat.set_Vx(Vx)
    flat.set_Vr(0.0)
    flat.set_Vt(0.0)

    return ml


def make_turbine(fluid, P_in, T_in, V_in, P_out, T_out, V_out):
    """A single-row machine between two given states."""
    return make_machine(fluid, [(P_in, T_in, V_in), (P_out, T_out, V_out)])


#
# BASICS
#


@pytest.fixture
def simple_turbine(perfect_air):
    return make_turbine(perfect_air, 1e5, 300.0, 100.0, 0.8e5, 290.0, 110.0)


def test_efficiencies_are_in_range(simple_turbine):
    for eta in (simple_turbine.eta_tt, simple_turbine.eta_ts):
        assert not np.isnan(eta)
        assert 0.0 <= eta <= 1.0


def test_efficiencies_are_finite(simple_turbine):
    assert np.isfinite(simple_turbine.eta_tt)
    assert np.isfinite(simple_turbine.eta_ts)


def test_eta_tt_at_least_eta_ts(simple_turbine):
    """eta_ts charges the exit kinetic energy as a loss, so it cannot exceed eta_tt.

    Both share the same actual enthalpy drop, but the total-to-static ideal
    drop is taken to the lower outlet static pressure and so is larger.
    """
    assert simple_turbine.eta_tt >= simple_turbine.eta_ts


def test_zero_rows_is_rejected_at_construction():
    """A machine with no rows is refused when built, not when read."""
    with pytest.raises(ValueError, match="n_row must be >= 1"):
        turbigen_ref.meanline_new.MeanLine(0)


def test_efficiency_uses_machine_endpoints(perfect_air):
    """Efficiency spans first inlet to last outlet, ignoring intermediate rows."""
    T_in, P_in, P_out = 400.0, 1e5, 0.7e5
    T_out = T_for_efficiency(T_in, P_out / P_in, 0.85)
    T_mid, P_mid = 0.5 * (T_in + T_out), 0.85e5

    ml = make_machine(
        perfect_air,
        [
            (P_in, T_in, 80.0),
            (P_mid, T_mid, 90.0),
            (P_mid, T_mid, 90.0),
            (P_out, T_out, 110.0),
        ],
    )

    assert ml.inlet.P == pytest.approx(P_in, rel=1e-5)
    assert ml.outlet.P == pytest.approx(P_out, rel=1e-5)

    # Same endpoints in a single row must give the same efficiency, since
    # nothing in between enters the definition.
    single = make_turbine(perfect_air, P_in, T_in, 80.0, P_out, T_out, 110.0)

    assert ml.eta_tt == pytest.approx(single.eta_tt, rel=1e-4)
    assert ml.eta_ts == pytest.approx(single.eta_ts, rel=1e-4)


#
# ANALYTICAL VALUES
#


def test_isentropic_expansion_is_fully_efficient(perfect_air):
    """A constant-entropy expansion has unit efficiency."""
    P_in, T_in, P_out = 1e5, 300.0, 0.8e5
    T_out = T_isentropic(T_in, P_out / P_in)

    # Negligible kinetic energy, so total-to-total and total-to-static agree.
    ml = make_turbine(perfect_air, P_in, T_in, 1e-6, P_out, T_out, 1e-6)

    assert ml.eta_tt == pytest.approx(1.0, rel=1e-3)
    assert ml.eta_ts == pytest.approx(1.0, rel=1e-3)


@pytest.mark.parametrize("eta_target", [0.65, 0.75, 0.85, 0.95])
def test_expansion_recovers_its_target_efficiency(perfect_air, eta_target):
    """eta_tt is the actual stagnation enthalpy drop over the ideal one."""
    P_in, T_in, P_out = 1e5, 400.0, 0.7e5
    T_out = T_for_efficiency(T_in, P_out / P_in, eta_target)

    ml = make_turbine(perfect_air, P_in, T_in, 1e-6, P_out, T_out, 1e-6)

    assert ml.eta_tt == pytest.approx(eta_target, rel=1e-3)

    # Cross-check against the definition written out longhand.
    T_out_ideal = T_isentropic(T_in, P_out / P_in)
    expected = (CP * (T_in - T_out)) / (CP * (T_in - T_out_ideal))
    assert ml.eta_tt == pytest.approx(expected, rel=1e-3)


@pytest.mark.parametrize("eta_target", [0.7, 0.85, 0.95])
def test_compression_is_ideal_over_actual_work(perfect_air, eta_target):
    """For work input the ratio inverts, without the old 1/eta fix-up.

    A compressor's actual work exceeds the ideal, so the efficiency is the
    ideal work over the actual one. The implementation dispatches on the sign
    of the stagnation enthalpy change rather than computing the expansion
    form and inverting it whenever it came out above unity.
    """
    P_in, T_in, P_out = 1e5, 300.0, 1.5e5
    T_out_ideal = T_isentropic(T_in, P_out / P_in)
    T_out = T_in + (T_out_ideal - T_in) / eta_target

    ml = make_turbine(perfect_air, P_in, T_in, 1e-6, P_out, T_out, 1e-6)

    assert ml.eta_tt == pytest.approx(eta_target, rel=1e-3)
    assert ml.eta_tt <= 1.0


@pytest.mark.parametrize("PR", [0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
def test_efficiencies_finite_across_pressure_ratios(perfect_air, PR):
    P_in, T_in = 1e5, 300.0
    T_out = T_for_efficiency(T_in, PR, 0.85)

    ml = make_turbine(perfect_air, P_in, T_in, 100.0, P_in * PR, T_out, 120.0)

    assert np.isfinite(ml.eta_tt)
    assert np.isfinite(ml.eta_ts)


def test_high_pressure_ratio_expansion(perfect_air):
    P_in, T_in, P_out = 2e5, 400.0, 1e5
    T_out = T_for_efficiency(T_in, P_out / P_in, 0.90)

    ml = make_turbine(perfect_air, P_in, T_in, 100.0, P_out, T_out, 150.0)

    assert np.isfinite(ml.eta_tt)
    assert np.isfinite(ml.eta_ts)


#
# EDGE CASES
#


def test_zero_pressure_drop_gives_infinite_eta_tt(perfect_air):
    """A machine doing no work has a zero ideal drop too, reported as inf."""
    ml = make_turbine(perfect_air, 1e5, 300.0, 100.0, 1e5, 300.0, 100.0)

    # Total-to-total: both actual and ideal drops vanish, so 0/0 -> inf.
    assert np.isinf(ml.eta_tt)

    # Total-to-static: the ideal drop to the lower static pressure is finite,
    # so a zero actual drop is simply zero efficiency.
    assert ml.eta_ts == pytest.approx(0.0)


def test_small_temperature_difference(perfect_air):
    """A 0.5 K drop still gives finite efficiencies."""
    ml = make_turbine(perfect_air, 1e5, 300.0, 100.0, 0.95e5, 299.5, 101.0)

    assert np.isfinite(ml.eta_tt)
    assert np.isfinite(ml.eta_ts)


#
# CONSISTENCY
#


@pytest.mark.parametrize("scale", [0.5, 1.0, 2.0])
def test_efficiency_finite_across_velocity_scales(perfect_air, scale):
    ml = make_turbine(
        perfect_air, 1e5, 300.0, 100.0 * scale, 0.8e5, 290.0, 130.0 * scale
    )

    assert np.isfinite(ml.eta_tt)
    assert np.isfinite(ml.eta_ts)


def test_efficiency_depends_on_the_gas(perfect_air):
    """A different isentropic exponent gives a different ideal state."""
    noble = ember.fluid.PerfectFluid(cp=520.0, gamma=1.67, mu=1.8e-5, Pr=0.72)
    args = (1e5, 300.0, 100.0, 0.8e5, 290.0, 110.0)

    ml_air = make_turbine(perfect_air, *args)
    ml_noble = make_turbine(noble, *args)

    for ml in (ml_air, ml_noble):
        assert 0.0 <= ml.eta_tt <= 1.0
        assert 0.0 <= ml.eta_ts <= 1.0

    assert ml_air.eta_tt != pytest.approx(ml_noble.eta_tt, rel=0.01)


#
# eta_tt >= eta_ts, swept
#


@pytest.mark.parametrize("PR", [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])
@pytest.mark.parametrize("eta_target", [0.95, 0.85, 0.75, 0.65])
def test_eta_ordering_across_pressure_ratios(perfect_air, PR, eta_target):
    P_in, T_in = 1e5, 300.0
    T_out = T_for_efficiency(T_in, PR, eta_target)

    ml = make_turbine(perfect_air, P_in, T_in, 100.0, P_in * PR, T_out, 120.0)

    assert ml.eta_tt >= ml.eta_ts


@pytest.mark.parametrize(
    "P_in,T_in",
    [(1e5, 300.0), (2e5, 400.0), (0.5e5, 250.0), (1.5e5, 350.0)],
)
def test_eta_ordering_across_inlet_conditions(perfect_air, P_in, T_in):
    T_out = T_for_efficiency(T_in, 0.7, 0.85)

    ml = make_turbine(perfect_air, P_in, T_in, 100.0, P_in * 0.7, T_out, 120.0)

    assert ml.eta_tt >= ml.eta_ts


@pytest.mark.parametrize(
    "V_in,V_out",
    [(50.0, 60.0), (100.0, 120.0), (150.0, 180.0), (200.0, 250.0), (10.0, 15.0)],
)
def test_eta_ordering_across_velocities(perfect_air, V_in, V_out):
    P_in, T_in, P_out = 1e5, 300.0, 0.8e5
    T_out = T_for_efficiency(T_in, P_out / P_in, 0.85)

    ml = make_turbine(perfect_air, P_in, T_in, V_in, P_out, T_out, V_out)

    assert ml.eta_tt >= ml.eta_ts


@pytest.mark.parametrize("eta_target", [0.99, 0.98, 0.97, 0.96])
def test_eta_ordering_near_isentropic(perfect_air, eta_target):
    P_in, T_in, P_out = 1e5, 300.0, 0.8e5
    T_out = T_for_efficiency(T_in, P_out / P_in, eta_target)

    ml = make_turbine(perfect_air, P_in, T_in, 100.0, P_out, T_out, 110.0)

    assert ml.eta_tt >= ml.eta_ts


#
# MULTI-ROW
#


def test_two_row_turbine(perfect_air):
    T_in, P_in, P_out = 400.0, 1.0e5, 0.7e5
    T_out = T_for_efficiency(T_in, P_out / P_in, 0.85)
    T_mid, P_mid = 0.5 * (T_in + T_out), 0.85e5

    ml = make_machine(
        perfect_air,
        [
            (P_in, T_in, 80.0),
            (P_mid, T_mid, 90.0),
            (P_mid, T_mid, 90.0),
            (P_out, T_out, 110.0),
        ],
    )

    assert ml.n_row == 2
    assert np.isfinite(ml.eta_tt)
    assert np.isfinite(ml.eta_ts)
    assert 0.0 <= ml.eta_tt <= 1.0
    assert 0.0 <= ml.eta_ts <= 1.0


def test_multirow_and_single_row_both_in_range(perfect_air):
    """Splitting an expansion over two rows keeps the efficiencies physical."""
    T_in, P_in, P_out = 400.0, 1.0e5, 0.7e5
    T_out = T_for_efficiency(T_in, P_out / P_in, 0.85)
    T_mid, P_mid = 0.5 * (T_in + T_out), 0.85e5

    single = make_turbine(perfect_air, P_in, T_in, 80.0, P_out, T_out, 110.0)
    multi = make_machine(
        perfect_air,
        [
            (P_in, T_in, 80.0),
            (P_mid, T_mid, 95.0),
            (P_mid, T_mid, 95.0),
            (P_out, T_out, 110.0),
        ],
    )

    for ml in (single, multi):
        assert 0.0 <= ml.eta_tt <= 1.0
        assert 0.0 <= ml.eta_ts <= 1.0
