"""Tests for the MeanLine class in meanline_new.py.

MeanLine is an ember Block of shape (2, n_row): axis 0 is the station within a
row (0 inlet, 1 outlet), axis 1 is the row. Views and write-through are covered
separately in test_meanline_views.py; this file covers construction, the
annulus geometry built on the added Am data key, nodal Omega, and the overall
performance properties.
"""

import numpy as np
import pytest

import ember.block
import ember.fluid
import turbigen_ref.meanline_new


@pytest.fixture
def fluid():
    return ember.fluid.PerfectFluid(cp=1005.0, gamma=1.4, mu=1.8e-5, Pr=0.72)


@pytest.fixture
def station(fluid):
    """A single scalar station, taken as a view of a one-row mean line."""
    ml = turbigen_ref.meanline_new.MeanLine(1)
    ml.set_fluid(fluid)
    return ml[0, 0]


#
# CONSTRUCTION
#


def test_meanline_is_a_block(fluid):
    """MeanLine inherits Block, adding only the Am and Omega data keys."""
    ml = turbigen_ref.meanline_new.MeanLine(2)

    assert isinstance(ml, ember.block.Block)
    assert "Am" in ml._data_keys
    assert "Omega" in ml._data_keys

    # Everything ember defines is inherited rather than forwarded. Look the
    # names up on the class, so the descriptors are not evaluated here.
    for name in ("Po", "Ma", "Ma_rel", "s", "ho_rel", "conserved"):
        assert hasattr(turbigen_ref.meanline_new.MeanLine, name)
        assert getattr(turbigen_ref.meanline_new.MeanLine, name) is getattr(
            ember.block.Block, name
        )


def test_meanline_shape_is_station_by_row(fluid):
    """n_row rows give shape (2, n_row), station axis first."""
    for n_row in (1, 2, 5):
        ml = turbigen_ref.meanline_new.MeanLine(n_row)
        assert ml.shape == (2, n_row)
        assert ml.n_row == n_row
        assert ml.size == 2 * n_row


def test_meanline_rejects_zero_rows():
    with pytest.raises(ValueError, match="n_row must be >= 1"):
        turbigen_ref.meanline_new.MeanLine(0)


def test_meanline_n_row_readonly(fluid):
    ml = turbigen_ref.meanline_new.MeanLine(2)
    ml.set_fluid(fluid)

    with pytest.raises(AttributeError, match="property 'n_row'.*has no setter"):
        ml.n_row = 5


def test_row_view_reports_one_row(fluid):
    """A row view is a shape-(2,) mean line of one row."""
    ml = turbigen_ref.meanline_new.MeanLine(3)
    ml.set_fluid(fluid)

    row = ml.row(1)
    assert isinstance(row, turbigen_ref.meanline_new.MeanLine)
    assert row.shape == (2,)
    assert row.n_row == 1


def test_station_view_is_scalar(fluid):
    ml = turbigen_ref.meanline_new.MeanLine(3)
    ml.set_fluid(fluid)

    for i_row in range(3):
        for j in (0, 1):
            st = ml[j, i_row]
            assert isinstance(st, turbigen_ref.meanline_new.MeanLine)
            assert st.shape == ()


#
# API CONVENTIONS
#


def test_setters_return_none(fluid):
    """Setters follow ember and return None; they are not chainable."""
    ml = turbigen_ref.meanline_new.MeanLine(1)
    ml.set_fluid(fluid)

    assert ml.set_r(0.5) is None
    assert ml.set_Am(1.0) is None
    assert ml.set_Omega(0.0) is None


def test_properties_are_not_assignable(fluid):
    """Derived properties have no setter, so state changes go through set_*."""
    ml = turbigen_ref.meanline_new.MeanLine(2)
    ml.set_fluid(fluid)

    with pytest.raises(AttributeError, match="property 'Vx'.*has no setter"):
        ml.Vx = np.zeros((2, 2))


def test_setter_rejects_wrong_shape(fluid):
    """An array that does not broadcast to (2, n_row) is refused."""
    ml = turbigen_ref.meanline_new.MeanLine(2)
    ml.set_fluid(fluid)
    ml.set_r(0.5)
    ml.set_P_T(1e5, 300.0)

    with pytest.raises(ValueError):
        ml.set_Vx(np.array([100.0, 110.0, 120.0]))


def test_reading_uninitialised_data_raises(fluid):
    """An unset variable raises rather than quietly returning NaN."""
    ml = turbigen_ref.meanline_new.MeanLine(2)
    ml.set_fluid(fluid)

    for name in ("Vx", "Am", "P"):
        with pytest.raises(ValueError, match="not been initialised"):
            getattr(ml, name)


def test_r_is_the_rms_radius(station):
    """The block radius is the annulus RMS radius; there is no separate name."""
    station.set_r(0.5)
    assert station.r == pytest.approx(0.5)
    assert not hasattr(station, "r_rms")


#
# ANNULUS GEOMETRY
#


def test_set_and_get_annulus_area(station):
    station.set_Am(1.0)
    station.set_r(0.5)

    assert station.Am == pytest.approx(1.0)
    assert station.r == pytest.approx(0.5)


def test_computed_annulus_geometry(station):
    """Hub, casing, mid radius, span and hub-to-tip follow from Am and r."""
    Am, r_rms = 1.0, 0.5
    station.set_Am(Am)
    station.set_r(r_rms)

    # Pure axial flow, so the pitch angle Beta is zero and cosBeta is one.
    station.set_Vx(100.0)
    station.set_Vr(0.0)
    station.set_Vt(0.0)
    station.set_P_T(1e5, 300.0)

    assert station.Beta == pytest.approx(0.0)
    cosBeta = np.cos(np.radians(station.Beta))
    assert station.cosBeta == pytest.approx(cosBeta)

    expected_r_cas = np.sqrt(Am * cosBeta / 2.0 / np.pi + r_rms**2.0)
    expected_r_hub = np.sqrt(r_rms**2.0 - Am * cosBeta / 2.0 / np.pi)
    expected_r_mid = 0.5 * (expected_r_hub + expected_r_cas)

    assert station.r_cas == pytest.approx(expected_r_cas)
    assert station.r_hub == pytest.approx(expected_r_hub)
    assert station.r_mid == pytest.approx(expected_r_mid)
    assert station.span == pytest.approx(Am / 2.0 / np.pi / expected_r_mid)
    assert station.htr == pytest.approx(expected_r_hub / expected_r_cas)


def test_annulus_geometry_is_vectorised(fluid):
    """Geometry properties evaluate over the whole (2, n_row) block at once."""
    ml = turbigen_ref.meanline_new.MeanLine(2)
    ml.set_fluid(fluid)
    ml.set_r(0.5)
    ml.set_P_T(1e5, 300.0)
    ml.set_Vx(100.0)
    ml.set_Vr(0.0)
    ml.set_Vt(0.0)

    Am = np.array([[1.0, 1.2], [1.1, 1.3]])
    ml.set_Am(Am)

    assert ml.span.shape == (2, 2)
    np.testing.assert_allclose(ml.Am, Am, rtol=1e-5)

    # Each station agrees with the same quantity computed on its own view.
    for j in (0, 1):
        for i in range(2):
            assert ml.span[j, i] == pytest.approx(ml[j, i].span, rel=1e-5)


def test_set_span_htr_round_trip(station):
    """set_span_htr reproduces the span and hub-to-tip ratio it was given."""
    station.set_P_T(1e5, 300.0)
    station.set_Vx(100.0)
    station.set_Vr(0.0)
    station.set_Vt(0.0)

    span, htr = 0.1, 0.6
    station.set_span_htr(span, htr)

    assert station.span == pytest.approx(span, rel=1e-5)
    assert station.htr == pytest.approx(htr, rel=1e-5)


def test_set_span_htr_requires_zero_pitch_angle(station):
    """Radial flow invalidates the span/htr relation, so it is refused."""
    station.set_P_T(1e5, 300.0)
    station.set_Vx(100.0)
    station.set_Vr(100.0)  # Beta = 45 degrees
    station.set_Vt(0.0)

    with pytest.raises(ValueError, match="Beta must be set zero"):
        station.set_span_htr(0.1, 0.6)


def test_set_span_r_mid_round_trip(station):
    station.set_P_T(1e5, 300.0)
    station.set_Vx(100.0)
    station.set_Vr(0.0)
    station.set_Vt(0.0)

    span, r_mid = 0.1, 0.5
    station.set_span_r_mid(span, r_mid)

    assert station.span == pytest.approx(span, rel=1e-5)
    assert station.r_mid == pytest.approx(r_mid, rel=1e-5)


def test_mass_flow_rate(station):
    """mdot is rho * Vm * Am."""
    Am = 1.0
    station.set_Am(Am)
    station.set_r(0.5)
    station.set_P_T(1e5, 300.0)
    station.set_Vx(100.0)
    station.set_Vr(0.0)
    station.set_Vt(0.0)

    assert station.mdot == pytest.approx(station.rho * station.Vm * Am)
    assert station.mdot > 0.0


def test_area_rescales_with_reference_length(station):
    """Am is stored non-dimensionally, so it survives a change of L_ref."""
    station.set_r(0.5)
    station.set_Am(2.0)

    station.set_L_ref(0.3)

    assert station.Am == pytest.approx(2.0, rel=1e-5)


#
# THERMODYNAMIC AND KINEMATIC STATE
#


def test_station_holds_full_state(station):
    """Coordinates, thermodynamic state and velocity all round-trip."""
    station.set_x(0.0)
    station.set_r(0.5)
    station.set_t(0.0)
    station.set_Am(1.0)
    station.set_P_T(2e5, 400.0)
    station.set_Vx(150.0)
    station.set_Vr(0.0)
    station.set_Vt(50.0)

    assert station.x == pytest.approx(0.0)
    assert station.r == pytest.approx(0.5)
    assert station.P == pytest.approx(2e5)
    assert station.T == pytest.approx(400.0)
    assert station.Vx == pytest.approx(150.0)
    assert station.Vt == pytest.approx(50.0)
    assert station.mdot > 0.0


def test_vectorised_velocity_setters(fluid):
    """Setters take a (2, n_row) array, one value per station."""
    ml = turbigen_ref.meanline_new.MeanLine(2)
    ml.set_fluid(fluid)
    ml.set_r(np.array([[0.5, 0.6], [0.55, 0.65]]))
    ml.set_Am(np.array([[1.0, 1.2], [1.1, 1.3]]))
    ml.set_P_T(
        np.array([[1e5, 0.9e5], [0.95e5, 0.85e5]]),
        np.array([[300.0, 290.0], [295.0, 285.0]]),
    )

    Vx = np.array([[100.0, 120.0], [110.0, 130.0]])
    Vr = np.array([[5.0, 7.0], [6.0, 8.0]])
    Vt = np.array([[50.0, 60.0], [55.0, 65.0]])
    ml.set_Vx(Vx)
    ml.set_Vr(Vr)
    ml.set_Vt(Vt)

    np.testing.assert_allclose(ml.Vx, Vx, rtol=1e-5)
    np.testing.assert_allclose(ml.Vr, Vr, rtol=1e-5)
    np.testing.assert_allclose(ml.Vt, Vt, rtol=1e-5)

    # And the same values appear in streamwise order through the flat view.
    np.testing.assert_allclose(ml.flat.Vx, [100.0, 110.0, 120.0, 130.0], rtol=1e-5)


def test_unset_stations_read_as_nan_for_area(fluid):
    """Am is allocated NaN, so a station left unset reads NaN rather than junk.

    Note that ember tracks initialisation per variable, not per station: once
    any station sets a variable the whole block counts as initialised, so a
    partially built mean line no longer raises. Am is the one added variable
    where the unset value is a well-defined NaN.
    """
    ml = turbigen_ref.meanline_new.MeanLine(2)
    ml.set_fluid(fluid)

    ml[0, 0].set_Am(1.0)

    assert ml.Am[0, 0] == pytest.approx(1.0)
    assert np.isnan(ml.Am[1, 0])
    assert np.isnan(ml.Am[0, 1])
    assert np.isnan(ml.Am[1, 1])


#
# ROTATION
#


def test_omega_is_nodal_and_defaults_to_zero(fluid):
    """Omega is stored per station, unlike ember's scalar block metadata."""
    ml = turbigen_ref.meanline_new.MeanLine(2)
    ml.set_fluid(fluid)

    assert ml.Omega.shape == (2, 2)
    np.testing.assert_allclose(ml.Omega, 0.0)


def test_set_omega_row(fluid):
    """One angular velocity per row, applied to both of its stations."""
    ml = turbigen_ref.meanline_new.MeanLine(3)
    ml.set_fluid(fluid)

    ml.set_Omega_row([0.0, 1000.0, 2000.0])

    np.testing.assert_allclose(ml.Omega[0], [0.0, 1000.0, 2000.0])
    np.testing.assert_allclose(ml.Omega[1], [0.0, 1000.0, 2000.0])


def test_set_omega_row_needs_a_full_mean_line(fluid):
    ml = turbigen_ref.meanline_new.MeanLine(2)
    ml.set_fluid(fluid)

    with pytest.raises(ValueError, match="requires a full"):
        ml.row(0).set_Omega_row([1.0])


def test_rotation_gives_blade_speed_and_relative_frame(fluid):
    """set_rpm on a row view sets that row's blade speed only."""
    ml = turbigen_ref.meanline_new.MeanLine(2)
    ml.set_fluid(fluid)
    ml.set_r(0.5)
    ml.set_Am(1.0)
    ml.set_P_T(1e5, 300.0)
    ml.set_Vx(100.0)
    ml.set_Vr(0.0)
    ml.set_Vt(50.0)

    rpm = 3000.0
    ml.row(1).set_rpm(rpm)

    Omega_expected = rpm * np.pi / 30.0
    np.testing.assert_allclose(ml.row(1).Omega, Omega_expected, rtol=1e-5)
    np.testing.assert_allclose(ml.row(0).Omega, 0.0)

    # Blade speed and the relative frame follow Omega per row.
    np.testing.assert_allclose(ml.row(1).U, Omega_expected * 0.5, rtol=1e-5)
    np.testing.assert_allclose(ml.row(0).U, 0.0)

    # With no rotation the relative frame coincides with the absolute one.
    np.testing.assert_allclose(ml.row(0).Ma_rel, ml.row(0).Ma, rtol=1e-5)
    assert np.all(ml.row(1).Ma_rel != ml.row(1).Ma)


#
# OVERALL PERFORMANCE
#


@pytest.fixture
def expansion(fluid):
    """A two-row mean line expanding from 1 bar to 0.85 bar."""
    ml = turbigen_ref.meanline_new.MeanLine(2)
    ml.set_fluid(fluid)
    ml.set_r(0.5)
    ml.set_Am(1.0)
    ml.flat.set_P_T(
        np.array([1e5, 0.95e5, 0.9e5, 0.85e5]),
        np.array([400.0, 395.0, 390.0, 385.0]),
    )
    ml.set_Vx(100.0)
    ml.set_Vr(0.0)
    ml.set_Vt(0.0)
    return ml


def test_pressure_ratios_use_the_machine_endpoints(expansion):
    ml = expansion

    assert ml.PR_tt == pytest.approx(ml.inlet.Po / ml.outlet.Po, rel=1e-5)
    assert ml.PR_ts == pytest.approx(ml.inlet.Po / ml.outlet.P, rel=1e-5)

    # The endpoints are the first and last stations in streamwise order.
    assert ml.inlet.P == pytest.approx(ml.flat.P[0], rel=1e-5)
    assert ml.outlet.P == pytest.approx(ml.flat.P[-1], rel=1e-5)


def test_efficiencies_are_physical_for_an_expansion(expansion):
    ml = expansion

    assert 0.0 < ml.eta_tt <= 1.0
    assert 0.0 < ml.eta_ts <= 1.0

    # A total-to-static efficiency charges the exit kinetic energy as a loss,
    # so it can never exceed the total-to-total value.
    assert ml.eta_tt >= ml.eta_ts


#
# REPRESENTATION
#


def test_to_string_tabulates_stations_in_streamwise_order(expansion):
    out = expansion.to_string()

    assert "Mean line:" in out
    assert "Row 0" in out and "Row 1" in out
    assert "Inlet" in out and "Outlet" in out

    # Stagnation pressure falls monotonically through the machine, and the
    # table should read in that order.
    Po_bar = expansion.flat.Po / 1e5
    assert np.all(np.diff(Po_bar) < 0.0)
    for value in Po_bar:
        assert f"{value:.3f}" in out


def test_repr_reports_the_shape(fluid):
    ml = turbigen_ref.meanline_new.MeanLine(3)
    assert repr(ml) == "MeanLine(shape=(2, 3))"
