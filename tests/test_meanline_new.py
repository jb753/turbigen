"""Tests for meanline_new.py Station class."""

import pytest
import numpy as np
import turbigen.meanline_new
import ember.fluid
import ember.block


def test_station_initialization():
    """Test that Station can be created and inherits from Block."""
    station = turbigen.meanline_new.Station(shape=())

    # Should be an instance of both Station and Block
    assert isinstance(station, turbigen.meanline_new.Station)
    assert isinstance(station, ember.block.Block)

    # Should have the Am data key
    assert "Am" in station._data_keys


def test_station_must_be_scalar():
    """Test that Station must be initialized with scalar shape."""
    # Scalar shape should work
    station = turbigen.meanline_new.Station(shape=())
    assert station.shape == ()

    # Non-scalar shapes should raise ValueError
    with pytest.raises(ValueError, match="Station must be a scalar"):
        turbigen.meanline_new.Station(shape=(5,))

    with pytest.raises(ValueError, match="Station must be a scalar"):
        turbigen.meanline_new.Station(shape=(3, 4))


def test_station_set_geometry():
    """Test setting annulus area and radius."""
    fluid = ember.fluid.PerfectFluid(cp=1005.0, gamma=1.4, mu=1.8e-5, Pr=0.72)
    station = turbigen.meanline_new.Station(shape=())
    station.set_fluid(fluid)

    # Set annulus geometry
    Am = 1.0  # 1 m^2 meridional area
    r_rms = 0.5  # 0.5 m RMS radius

    station.set_Am(Am)
    station.set_r_rms(r_rms)

    assert station.Am == pytest.approx(Am)
    assert station.r_rms == pytest.approx(r_rms)


def test_station_computed_geometry_properties():
    """Test computed annulus geometry properties."""
    fluid = ember.fluid.PerfectFluid(cp=1005.0, gamma=1.4, mu=1.8e-5, Pr=0.72)
    station = turbigen.meanline_new.Station(shape=())
    station.set_fluid(fluid)

    # Set geometry
    Am = 1.0
    r_rms = 0.5
    station.set_Am(Am).set_r_rms(r_rms)

    # Set velocity to compute cosBeta
    station.set_Vx(100.0).set_Vr(0.0).set_Vt(0.0)  # Axial flow only
    station.set_P_T(1e5, 300.0)

    # Compute cosBeta from Beta angle
    # For pure axial flow (Vr=0), Beta=0, so cosBeta=1
    cosBeta = np.cos(station.Beta)

    expected_r_cas = np.sqrt(Am * cosBeta / 2.0 / np.pi + r_rms**2.0)
    expected_r_hub = np.sqrt(r_rms**2.0 - Am * cosBeta / 2.0 / np.pi)
    expected_r_mid = 0.5 * (expected_r_hub + expected_r_cas)
    expected_span = Am / 2.0 / np.pi / expected_r_mid
    expected_htr = expected_r_hub / expected_r_cas

    assert station.r_cas == pytest.approx(expected_r_cas)
    assert station.r_hub == pytest.approx(expected_r_hub)
    assert station.r_mid == pytest.approx(expected_r_mid)
    assert station.span == pytest.approx(expected_span)
    assert station.htr == pytest.approx(expected_htr)


def test_station_mass_flow_rate():
    """Test mass flow rate calculation."""
    fluid = ember.fluid.PerfectFluid(cp=1005.0, gamma=1.4, mu=1.8e-5, Pr=0.72)
    station = turbigen.meanline_new.Station(shape=())
    station.set_fluid(fluid)

    # Set geometry
    Am = 1.0  # 1 m^2
    station.set_Am(Am).set_r_rms(0.5)

    # Set thermodynamic state
    P, T = 1e5, 300.0
    station.set_P_T(P, T)

    # Set velocity
    Vx, Vr = 100.0, 0.0
    station.set_Vx(Vx).set_Vr(Vr).set_Vt(0.0)

    # Compute mass flow rate
    # mdot = rho * Vm * Am
    # For axial flow with small radial velocity, Vm ≈ Vx
    rho = station.rho
    Vm = station.Vm
    expected_mdot = rho * Vm * Am

    assert station.mdot == pytest.approx(expected_mdot)


def test_station_with_thermodynamics():
    """Test Station with full thermodynamic state."""
    fluid = ember.fluid.PerfectFluid(cp=1005.0, gamma=1.4, mu=1.8e-5, Pr=0.72)
    station = turbigen.meanline_new.Station(shape=())
    station.set_fluid(fluid)

    # Set coordinates (must set x, r_rms, t separately since set_xrt calls set_r)
    station.set_x(0.0).set_r_rms(0.5).set_t(0.0)

    # Set annulus area
    station.set_Am(1.0)

    # Set thermodynamic state
    P, T = 2e5, 400.0
    station.set_P_T(P, T)

    # Set velocity
    station.set_Vx(150.0).set_Vr(0.0).set_Vt(50.0)

    # Verify we can access thermodynamic properties
    assert station.P == pytest.approx(P)
    assert station.T == pytest.approx(T)
    assert station.Vx == pytest.approx(150.0)
    assert station.Vt == pytest.approx(50.0)

    # Verify mass flow is computed
    assert station.mdot > 0


def test_station_set_r_blocked():
    """Verify that set_r() raises NotImplementedError."""
    station = turbigen.meanline_new.Station(shape=())

    with pytest.raises(NotImplementedError, match="Use set_r_rms"):
        station.set_r(0.5)


def test_station_inherits_block_setters():
    """Verify Station can use Block's setter methods."""
    fluid = ember.fluid.PerfectFluid(cp=1005.0, gamma=1.4, mu=1.8e-5, Pr=0.72)
    station = turbigen.meanline_new.Station(shape=())
    station.set_fluid(fluid)

    # Should be able to chain Block's setters (but set coords separately)
    station.set_x(1.0).set_r_rms(0.5).set_t(0.1).set_Am(1.5).set_P_T(
        1e5, 300.0
    ).set_Vx(100.0).set_Vr(0.0).set_Vt(0.0)

    # Verify values were set
    assert station.x == pytest.approx(1.0)
    assert station.r_rms == pytest.approx(0.5)
    assert station.t == pytest.approx(0.1)
    assert station.Am == pytest.approx(1.5)
    assert station.P == pytest.approx(1e5)
    assert station.T == pytest.approx(300.0)
    assert station.Vx == pytest.approx(100.0)


def test_station_with_rotation():
    """Test Station with rotating frame."""
    fluid = ember.fluid.PerfectFluid(cp=1005.0, gamma=1.4, mu=1.8e-5, Pr=0.72)
    station = turbigen.meanline_new.Station(shape=())
    station.set_fluid(fluid)

    # Set up rotating station
    rpm = 3000.0
    station.set_rpm(rpm).set_r_rms(0.5).set_Am(1.0)

    # Set thermodynamic state and velocity
    station.set_P_T(1e5, 300.0).set_Vx(100.0).set_Vr(0.0).set_Vt(50.0)

    # Verify rotation was set
    Omega_expected = rpm / 30.0 * np.pi  # Convert RPM to rad/s
    assert station.Omega == pytest.approx(Omega_expected)

    # Verify we can compute relative velocities
    U = station.U
    assert U > 0  # Blade speed should be non-zero


def test_meanline_initialization():
    """Test that MeanLine can be created with specified number of rows."""
    fluid = ember.fluid.PerfectFluid(cp=1005.0, gamma=1.4, mu=1.8e-5, Pr=0.72)
    ml = turbigen.meanline_new.MeanLine(n_row=2).set_fluid(fluid)

    # Should have correct number of rows
    assert ml.n_row == 2

    # Should have 4 scalar stations (2 per row)
    assert len(ml._stations) == 4

    # Each station should be scalar
    for station in ml._stations:
        assert isinstance(station, turbigen.meanline_new.Station)
        assert station.shape == ()


def test_meanline_n_row_readonly():
    """Test that n_row property is read-only."""
    fluid = ember.fluid.PerfectFluid(cp=1005.0, gamma=1.4, mu=1.8e-5, Pr=0.72)
    ml = turbigen.meanline_new.MeanLine(n_row=2).set_fluid(fluid)

    # Attempting to assign should raise AttributeError
    with pytest.raises(AttributeError, match="property 'n_row'.*has no setter"):
        ml.n_row = 5


def test_meanline_scalar_indexing():
    """Test integer indexing returns a scalar Station at that flat index."""
    fluid = ember.fluid.PerfectFluid(cp=1005.0, gamma=1.4, mu=1.8e-5, Pr=0.72)
    ml = turbigen.meanline_new.MeanLine(n_row=3).set_fluid(fluid)

    # ml has 6 stations (2 per row); each index returns a scalar Station
    for i in range(6):
        s = ml[i]
        assert isinstance(s, turbigen.meanline_new.Station)
        assert s.shape == ()


def test_meanline_get_row():
    """Test get_row returns a MeanLine view of the two stations for that row."""
    fluid = ember.fluid.PerfectFluid(cp=1005.0, gamma=1.4, mu=1.8e-5, Pr=0.72)
    ml = turbigen.meanline_new.MeanLine(n_row=2).set_fluid(fluid)

    row0 = ml.get_row(0)
    assert isinstance(row0, turbigen.meanline_new.MeanLine)
    assert len(row0._stations) == 2

    row1 = ml.get_row(1)
    assert isinstance(row1, turbigen.meanline_new.MeanLine)
    assert len(row1._stations) == 2


def test_meanline_indexing_consistency():
    """Test flat indexing and get_row are consistent."""
    fluid = ember.fluid.PerfectFluid(cp=1005.0, gamma=1.4, mu=1.8e-5, Pr=0.72)
    ml = turbigen.meanline_new.MeanLine(n_row=2).set_fluid(fluid)

    # Set data via flat indexing
    ml[0].set_r_rms(0.5)
    ml[1].set_r_rms(0.75)

    # get_row(0) should give a view over stations 0 and 1
    row0 = ml.get_row(0)
    assert row0[0].r_rms == pytest.approx(0.5)
    assert row0[1].r_rms == pytest.approx(0.75)

    # Direct flat access agrees
    assert ml[0].r_rms == pytest.approx(0.5)
    assert ml[1].r_rms == pytest.approx(0.75)


def test_meanline_concatenated_property():
    """Test that concatenated properties work correctly using factory method."""
    fluid = ember.fluid.PerfectFluid(cp=1005.0, gamma=1.4, mu=1.8e-5, Pr=0.72)
    ml = turbigen.meanline_new.MeanLine(n_row=2).set_fluid(fluid)

    Vx = np.array([100.0, 110.0, 120.0, 130.0], dtype=np.float32)
    Vr = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    Vt = np.array([50.0, 55.0, 60.0, 65.0], dtype=np.float32)
    r_rms = np.array([0.5, 0.55, 0.6, 0.65], dtype=np.float32)
    Am = np.array([1.0, 1.1, 1.2, 1.3], dtype=np.float32)
    P = np.array([1e5, 0.95e5, 0.9e5, 0.85e5], dtype=np.float32)
    T = np.array([300.0, 295.0, 290.0, 285.0], dtype=np.float32)

    for i in range(4):
        ml[i].set_r_rms(r_rms[i])
        ml[i].set_Am(Am[i])
        ml[i].set_P_T(P[i], T[i])
        ml[i].set_Vx(Vx[i]).set_Vr(Vr[i]).set_Vt(Vt[i])

    Vx_concat = ml.Vx.squeeze()

    # Should be shape (4,) for 2 rows × 2 stations per row
    assert Vx_concat.shape == (4,)

    np.testing.assert_allclose(Vx_concat, Vx, rtol=1e-5)

    # Verify order: station 0, 1, 2, 3
    for i in range(4):
        assert Vx_concat[i] == pytest.approx(ml[i].Vx)


def test_meanline_property_docstring():
    """Test that factory-generated properties inherit docstrings from Block."""
    # Check that Vx has the docstring from Block
    vx_doc = turbigen.meanline_new.MeanLine.Vx.__doc__
    block_vx_doc = ember.block.Block.Vx.__doc__

    # Should inherit the Block's docstring
    assert vx_doc == block_vx_doc
    assert "Axial velocity" in vx_doc


def test_meanline_setter_method():
    """Test that factory-generated setter methods work correctly."""
    fluid = ember.fluid.PerfectFluid(cp=1005.0, gamma=1.4, mu=1.8e-5, Pr=0.72)
    ml = turbigen.meanline_new.MeanLine(n_row=2).set_fluid(fluid)

    r_rms = np.array([0.5, 0.55, 0.6, 0.65], dtype=np.float32)
    Am = np.array([1.0, 1.1, 1.2, 1.3], dtype=np.float32)
    P = np.array([1e5, 0.95e5, 0.9e5, 0.85e5], dtype=np.float32)
    T = np.array([300.0, 295.0, 290.0, 285.0], dtype=np.float32)

    for i in range(4):
        ml[i].set_r_rms(r_rms[i])
        ml[i].set_Am(Am[i])
        ml[i].set_P_T(P[i], T[i])

    Vx = np.array([100.0, 110.0, 120.0, 130.0], dtype=np.float32)
    Vr = np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32)
    Vt = np.array([50.0, 55.0, 60.0, 65.0], dtype=np.float32)

    ml.set_Vx(Vx).set_Vr(Vr).set_Vt(Vt)

    assert ml[0].Vx == pytest.approx(100.0)
    assert ml[1].Vx == pytest.approx(110.0)
    assert ml[2].Vx == pytest.approx(120.0)
    assert ml[3].Vx == pytest.approx(130.0)

    assert ml[0].Vr == pytest.approx(5.0)
    assert ml[1].Vr == pytest.approx(6.0)
    assert ml[2].Vr == pytest.approx(7.0)
    assert ml[3].Vr == pytest.approx(8.0)

    assert ml[0].Vt == pytest.approx(50.0)
    assert ml[1].Vt == pytest.approx(55.0)
    assert ml[2].Vt == pytest.approx(60.0)
    assert ml[3].Vt == pytest.approx(65.0)

    np.testing.assert_allclose(ml.Vx.squeeze(), Vx, rtol=1e-5)


def test_meanline_setter_shape_validation():
    """Test that setter methods validate input array shapes."""
    fluid = ember.fluid.PerfectFluid(cp=1005.0, gamma=1.4, mu=1.8e-5, Pr=0.72)
    ml = turbigen.meanline_new.MeanLine(n_row=2).set_fluid(fluid)

    # Should reject arrays that don't have length 4 (2 rows × 2 stations)
    Vx_wrong = np.array([100.0, 110.0, 120.0], dtype=np.float32)  # Only 3 values
    Vr = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    Vt = np.array([50.0, 55.0, 60.0, 65.0], dtype=np.float32)

    with pytest.raises(ValueError):
        ml.set_Vx(Vx_wrong).set_Vr(Vr).set_Vt(Vt)


def test_meanline_setter_returns_self():
    """Test that setter methods return self for chaining."""
    fluid = ember.fluid.PerfectFluid(cp=1005.0, gamma=1.4, mu=1.8e-5, Pr=0.72)
    ml = turbigen.meanline_new.MeanLine(n_row=2).set_fluid(fluid)

    for i in range(4):
        ml[i].set_r_rms(0.5).set_Am(1.0).set_P_T(1e5, 300.0)

    Vx = np.array([100.0, 110.0, 120.0, 130.0], dtype=np.float32)
    Vr = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    Vt = np.array([50.0, 55.0, 60.0, 65.0], dtype=np.float32)

    result = ml.set_Vx(Vx).set_Vr(Vr).set_Vt(Vt)
    assert result is ml


def test_meanline_setter_docstring():
    """Test that factory-generated setters inherit docstrings from Block."""
    set_vxrt_doc = turbigen.meanline_new.MeanLine.set_Vxrt.__doc__
    block_set_vxrt_doc = ember.block.Block.set_Vxrt.__doc__

    # Should inherit the Block's docstring
    assert set_vxrt_doc == block_set_vxrt_doc


def test_meanline_concatenated_property_readonly():
    """Test that concatenated properties are read-only."""
    fluid = ember.fluid.PerfectFluid(cp=1005.0, gamma=1.4, mu=1.8e-5, Pr=0.72)
    ml = turbigen.meanline_new.MeanLine(n_row=2).set_fluid(fluid)

    # Attempting to assign to a concatenated property should raise AttributeError
    with pytest.raises(AttributeError, match="property 'Vx'.*has no setter"):
        ml.Vx = np.array([100.0, 110.0, 120.0, 130.0], dtype=np.float32)

    # This ensures users must use the setter methods (set_Vxrt, etc.)
    # and prevents accidental overwrites of the property itself


def test_meanline_concatenated_array_values():
    """Test that concatenated property arrays return correct values."""
    fluid = ember.fluid.PerfectFluid(cp=1005.0, gamma=1.4, mu=1.8e-5, Pr=0.72)
    ml = turbigen.meanline_new.MeanLine(n_row=2).set_fluid(fluid)

    r_rms = np.array([0.5, 0.55, 0.6, 0.65], dtype=np.float32)
    Am = np.array([1.0, 1.1, 1.2, 1.3], dtype=np.float32)
    P = np.array([1e5, 0.95e5, 0.9e5, 0.85e5], dtype=np.float32)
    T = np.array([300.0, 295.0, 290.0, 285.0], dtype=np.float32)
    Vx = np.array([100.0, 110.0, 120.0, 130.0], dtype=np.float32)
    Vr = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    Vt = np.array([50.0, 55.0, 60.0, 65.0], dtype=np.float32)

    for i in range(4):
        ml[i].set_r_rms(r_rms[i]).set_Am(Am[i]).set_P_T(P[i], T[i])
    ml.set_Vx(Vx).set_Vr(Vr).set_Vt(Vt)

    np.testing.assert_allclose(ml.Vx.squeeze(), Vx, rtol=1e-5)
    np.testing.assert_allclose(ml.Vt.squeeze(), Vt, rtol=1e-5)


def test_meanline_uninitialized_stations():
    """Test that concatenated properties return NaN for uninitialized stations."""
    fluid = ember.fluid.PerfectFluid(cp=1005.0, gamma=1.4, mu=1.8e-5, Pr=0.72)
    ml = turbigen.meanline_new.MeanLine(n_row=2).set_fluid(fluid)

    # Don't initialize anything - all stations uninitialized
    vx = ml.Vx

    # Should have 4 entries for 2 rows × 2 stations
    assert vx.squeeze().shape == (4,)

    # All values should be NaN
    assert np.all(np.isnan(vx))


def test_meanline_partially_initialized():
    """Test that concatenated properties handle partially initialized meanlines."""
    fluid = ember.fluid.PerfectFluid(cp=1005.0, gamma=1.4, mu=1.8e-5, Pr=0.72)
    ml = turbigen.meanline_new.MeanLine(n_row=2).set_fluid(fluid)

    # Only initialize stations 0 and 1 (first row) using flat indexing
    ml[0].set_r_rms(0.5).set_Am(1.0).set_P_T(1e5, 300.0).set_Vx(100.0).set_Vr(0.0).set_Vt(50.0)
    ml[1].set_r_rms(0.55).set_Am(1.1).set_P_T(0.95e5, 295.0).set_Vx(110.0).set_Vr(0.0).set_Vt(55.0)

    # Stations 2, 3 remain uninitialized
    vx = ml.Vx.squeeze()

    assert vx.shape == (4,)
    assert vx[0] == pytest.approx(100.0)
    assert vx[1] == pytest.approx(110.0)
    assert np.isnan(vx[2])
    assert np.isnan(vx[3])


def test_meanline_single_station_initialized():
    """Test that concatenated properties handle single station initialization."""
    fluid = ember.fluid.PerfectFluid(cp=1005.0, gamma=1.4, mu=1.8e-5, Pr=0.72)
    ml = turbigen.meanline_new.MeanLine(n_row=2).set_fluid(fluid)

    # Only initialize flat station index 1
    ml[1].set_r_rms(0.55).set_Am(1.1).set_P_T(0.95e5, 295.0).set_Vx(110.0).set_Vr(0.0).set_Vt(55.0)

    vx = ml.Vx.squeeze()

    assert vx.shape == (4,)
    assert np.isnan(vx[0])
    assert vx[1] == pytest.approx(110.0)
    assert np.isnan(vx[2])
    assert np.isnan(vx[3])
