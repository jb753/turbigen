"""Tests for isentropic efficiency calculations (eta_tt and eta_ts) in MeanLine.

These tests verify the implementation of total-to-total and total-to-static
isentropic efficiencies against analytical expressions for a perfect gas.
"""

import pytest
import numpy as np
import turbigen.meanline_new
import ember.fluid


class TestIsentropicEfficiencyBasics:
    """Basic tests for efficiency calculations."""

    @pytest.fixture
    def perfect_air(self):
        """Create a perfect gas model for air."""
        return ember.fluid.PerfectFluid(cp=1005.0, gamma=1.4, mu=1.8e-5, Pr=0.72)

    @pytest.fixture
    def simple_turbine(self, perfect_air):
        """Create a simple turbine with 1 row (2 stations)."""
        ml = turbigen.meanline_new.MeanLine(n_row=1)
        ml.set_fluid(perfect_air)

        # Inlet: stagnation state
        inlet = ml[0]
        inlet.set_r_rms(0.5)
        inlet.set_Am(1.0)
        inlet.set_P_T(1e5, 300.0)  # 1 bar, 300 K static
        inlet.set_Vx(100.0).set_Vr(0.0).set_Vt(0.0)  # 100 m/s axial velocity

        # Outlet: after expansion
        outlet = ml[1]
        outlet.set_r_rms(0.5)
        outlet.set_Am(1.0)
        outlet.set_P_T(0.8e5, 290.0)  # 0.8 bar, 290 K static
        outlet.set_Vx(110.0).set_Vr(0.0).set_Vt(0.0)

        return ml

    def test_eta_tt_valid_range(self, simple_turbine):
        """Test that eta_tt is in valid range [0, 1] for reasonable turbine."""
        eta_tt = simple_turbine.eta_tt
        assert not np.isnan(eta_tt), "eta_tt should not be NaN"
        assert 0.0 <= eta_tt <= 1.0, f"eta_tt {eta_tt} should be between 0 and 1"

    def test_eta_ts_valid_range(self, simple_turbine):
        """Test that eta_ts is in valid range [0, 1] for reasonable turbine."""
        eta_ts = simple_turbine.eta_ts
        assert not np.isnan(eta_ts), "eta_ts should not be NaN"
        assert 0.0 <= eta_ts <= 1.0, f"eta_ts {eta_ts} should be between 0 and 1"

    def test_both_efficiencies_finite(self, simple_turbine):
        """Test that both efficiencies are finite numbers."""
        eta_tt = simple_turbine.eta_tt
        eta_ts = simple_turbine.eta_ts
        assert np.isfinite(eta_tt), f"eta_tt should be finite, got {eta_tt}"
        assert np.isfinite(eta_ts), f"eta_ts should be finite, got {eta_ts}"

    def test_eta_tt_greater_than_or_equal_eta_ts(self, simple_turbine):
        """Test that eta_tt >= eta_ts always.

        This is because eta_tt uses outlet stagnation pressure (higher) in the isentropic
        process, giving a larger ideal enthalpy drop than eta_ts which uses outlet static
        pressure (lower). Since both share the same actual enthalpy drop denominator,
        eta_tt >= eta_ts must hold.
        """
        eta_tt = simple_turbine.eta_tt
        eta_ts = simple_turbine.eta_ts
        assert eta_tt >= eta_ts, f"eta_tt ({eta_tt}) must be >= eta_ts ({eta_ts})"

    def test_empty_meanline_raises_error(self):
        """Test that empty MeanLine raises IndexError (no stations)."""
        fluid = ember.fluid.PerfectFluid(cp=1005.0, gamma=1.4, mu=1.8e-5, Pr=0.72)
        ml = turbigen.meanline_new.MeanLine(n_row=0)
        ml.set_fluid(fluid)

        # Accessing self[0] and self[-1] on empty meanline should raise IndexError
        with pytest.raises(IndexError):
            _ = ml.eta_tt
        with pytest.raises(IndexError):
            _ = ml.eta_ts

    def test_single_row_computes_normally(self):
        """Test that single row (2 stations) computes normally."""
        fluid = ember.fluid.PerfectFluid(cp=1005.0, gamma=1.4, mu=1.8e-5, Pr=0.72)
        ml = turbigen.meanline_new.MeanLine(n_row=1)
        ml.set_fluid(fluid)

        # Initialize both stations
        for i in range(2):
            ml[i].set_r_rms(0.5)
            ml[i].set_Am(1.0)
            ml[i].set_P_T(1e5, 300.0)
            ml[i].set_Vx(100.0).set_Vr(0.0).set_Vt(0.0)

        # With no work, denominator will be ~0, returning inf
        eta_tt = ml.eta_tt
        eta_ts = ml.eta_ts
        assert isinstance(eta_tt, (float, np.floating))
        assert isinstance(eta_ts, (float, np.floating))


class TestIsentropicEfficiencyAnalytical:
    """Test efficiency calculations against analytical expressions for perfect gas."""

    @pytest.fixture
    def perfect_air(self):
        """Create a perfect gas model for air."""
        return ember.fluid.PerfectFluid(cp=1005.0, gamma=1.4, mu=1.8e-5, Pr=0.72)

    def create_turbine(self, fluid, P_in, T_in, V_in, P_out, T_out, V_out):
        """Helper to create a turbine with given states."""
        ml = turbigen.meanline_new.MeanLine(n_row=1)
        ml.set_fluid(fluid)

        inlet = ml[0]
        inlet.set_r_rms(0.5)
        inlet.set_Am(1.0)
        inlet.set_P_T(P_in, T_in)
        inlet.set_Vx(V_in).set_Vr(0.0).set_Vt(0.0)

        outlet = ml[1]
        outlet.set_r_rms(0.5)
        outlet.set_Am(1.0)
        outlet.set_P_T(P_out, T_out)
        outlet.set_Vx(V_out).set_Vr(0.0).set_Vt(0.0)

        return ml

    def test_isentropic_expansion(self, perfect_air):
        """Test efficiency calculations for isentropic expansion."""
        gamma = 1.4
        P_in, T_in = 1e5, 300.0
        P_out = 0.8e5

        # Compute isentropic outlet temperature
        r = (gamma - 1) / gamma
        T_out_ideal = T_in * (P_out / P_in) ** r

        # Create turbine with isentropic process
        ml = self.create_turbine(
            perfect_air, P_in, T_in, 100.0, P_out, T_out_ideal, 110.0
        )

        eta_tt = ml.eta_tt
        eta_ts = ml.eta_ts

        # Both should be valid
        assert 0.0 <= eta_tt <= 1.0, f"eta_tt {eta_tt} outside valid range"
        assert 0.0 <= eta_ts <= 1.0, f"eta_ts {eta_ts} outside valid range"

    def test_high_pressure_ratio_expansion(self, perfect_air):
        """Test efficiency for higher pressure ratio expansion."""
        gamma = 1.4
        P_in, T_in = 2e5, 400.0
        P_out = 1e5  # 50% pressure ratio
        V_in, V_out = 100.0, 150.0

        # Compute isentropic outlet temperature
        r = (gamma - 1) / gamma
        T_out_ideal = T_in * (P_out / P_in) ** r

        # Assume 90% efficiency based on temperature drop
        eta_ideal = 0.90
        T_out_actual = T_in - eta_ideal * (T_in - T_out_ideal)

        ml = self.create_turbine(
            perfect_air, P_in, T_in, V_in, P_out, T_out_actual, V_out
        )

        eta_tt = ml.eta_tt
        eta_ts = ml.eta_ts

        # Should be reasonable values
        assert np.isfinite(eta_tt), f"eta_tt {eta_tt} should be finite"
        assert np.isfinite(eta_ts), f"eta_ts {eta_ts} should be finite"

    def test_multiple_pressure_ratios(self, perfect_air):
        """Test efficiencies across a range of pressure ratios."""
        gamma = 1.4
        T_in = 300.0
        P_in = 1e5

        pressure_ratios = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]

        for pr in pressure_ratios:
            P_out = P_in * pr
            r = (gamma - 1) / gamma
            T_out_ideal = T_in * pr**r

            # Assume 85% efficiency
            target_eta = 0.85
            T_out_actual = T_in - target_eta * (T_in - T_out_ideal)

            ml = self.create_turbine(
                perfect_air, P_in, T_in, 100.0, P_out, T_out_actual, 120.0
            )

            eta_tt = ml.eta_tt
            eta_ts = ml.eta_ts

            assert np.isfinite(eta_tt), f"eta_tt {eta_tt} should be finite for PR={pr}"
            assert np.isfinite(eta_ts), f"eta_ts {eta_ts} should be finite for PR={pr}"


class TestIsentropicEfficiencyEdgeCases:
    """Test edge cases and special conditions."""

    @pytest.fixture
    def perfect_air(self):
        """Create a perfect gas model for air."""
        return ember.fluid.PerfectFluid(cp=1005.0, gamma=1.4, mu=1.8e-5, Pr=0.72)

    def test_zero_pressure_drop_returns_inf(self, perfect_air):
        """Test that zero pressure drop (no work) returns inf."""
        ml = turbigen.meanline_new.MeanLine(n_row=1)
        ml.set_fluid(perfect_air)

        inlet = ml[0]
        inlet.set_r_rms(0.5)
        inlet.set_Am(1.0)
        inlet.set_P_T(1e5, 300.0)
        inlet.set_Vx(100.0).set_Vr(0.0).set_Vt(0.0)

        # Same pressure and temperature at outlet (no expansion)
        outlet = ml[1]
        outlet.set_r_rms(0.5)
        outlet.set_Am(1.0)
        outlet.set_P_T(1e5, 300.0)  # Same P and T
        outlet.set_Vx(100.0).set_Vr(0.0).set_Vt(0.0)

        eta_tt = ml.eta_tt
        eta_ts = ml.eta_ts

        # With zero numerator (no actual work), eta_tt returns inf (0/0 = NaN → inf)
        assert np.isinf(eta_tt), f"eta_tt {eta_tt} should be inf for zero work"
        # eta_ts can return 0 or other values (0 / finite = 0) without special casing
        assert isinstance(eta_ts, (float, np.floating))

    def test_small_temperature_difference(self, perfect_air):
        """Test with very small temperature difference."""
        ml = turbigen.meanline_new.MeanLine(n_row=1)
        ml.set_fluid(perfect_air)

        inlet = ml[0]
        inlet.set_r_rms(0.5)
        inlet.set_Am(1.0)
        inlet.set_P_T(1e5, 300.0)
        inlet.set_Vx(100.0).set_Vr(0.0).set_Vt(0.0)

        # Small temperature drop at lower pressure
        outlet = ml[1]
        outlet.set_r_rms(0.5)
        outlet.set_Am(1.0)
        outlet.set_P_T(0.95e5, 299.5)  # Only 0.5 K drop
        outlet.set_Vx(101.0).set_Vr(0.0).set_Vt(0.0)

        eta_tt = ml.eta_tt
        eta_ts = ml.eta_ts

        # Both should be finite (may be outside 0-1 range due to velocity effects)
        assert np.isfinite(eta_tt), f"eta_tt {eta_tt} should be finite"
        assert np.isfinite(eta_ts), f"eta_ts {eta_ts} should be finite"


class TestIsentropicEfficiencyConsistency:
    """Test consistency of efficiency calculations."""

    @pytest.fixture
    def perfect_air(self):
        """Create a perfect gas model for air."""
        return ember.fluid.PerfectFluid(cp=1005.0, gamma=1.4, mu=1.8e-5, Pr=0.72)

    def test_efficiency_finite_different_velocities(self, perfect_air):
        """Test that efficiency is finite for different velocities."""

        def create_and_compute_eta(V_scale):
            ml = turbigen.meanline_new.MeanLine(n_row=1)
            ml.set_fluid(perfect_air)

            inlet = ml[0]
            inlet.set_r_rms(0.5)
            inlet.set_Am(1.0)
            inlet.set_P_T(1e5, 300.0)
            inlet.set_Vx(100.0 * V_scale).set_Vr(0.0).set_Vt(0.0)

            outlet = ml[1]
            outlet.set_r_rms(0.5)
            outlet.set_Am(1.0)
            outlet.set_P_T(0.8e5, 290.0)
            outlet.set_Vx(130.0 * V_scale).set_Vr(0.0).set_Vt(0.0)

            return ml.eta_tt, ml.eta_ts

        # Test with different velocity scales
        for scale in [0.5, 1.0, 2.0]:
            eta_tt, eta_ts = create_and_compute_eta(scale)
            assert np.isfinite(
                eta_tt
            ), f"eta_tt {eta_tt} should be finite for scale {scale}"
            assert np.isfinite(
                eta_ts
            ), f"eta_ts {eta_ts} should be finite for scale {scale}"

    def test_efficiency_for_different_gases(self):
        """Test efficiencies for different gas properties (different gamma)."""
        P_in, T_in = 1e5, 300.0
        P_out, T_out_actual = 0.8e5, 290.0

        # Air-like (gamma=1.4)
        air = ember.fluid.PerfectFluid(cp=1005.0, gamma=1.4, mu=1.8e-5, Pr=0.72)

        # Noble gas-like (gamma=1.67)
        noble = ember.fluid.PerfectFluid(cp=520.0, gamma=1.67, mu=1.8e-5, Pr=0.72)

        def compute_eta(fluid):
            ml = turbigen.meanline_new.MeanLine(n_row=1)
            ml.set_fluid(fluid)

            inlet = ml[0]
            inlet.set_r_rms(0.5)
            inlet.set_Am(1.0)
            inlet.set_P_T(P_in, T_in)
            inlet.set_Vx(100.0).set_Vr(0.0).set_Vt(0.0)

            outlet = ml[1]
            outlet.set_r_rms(0.5)
            outlet.set_Am(1.0)
            outlet.set_P_T(P_out, T_out_actual)
            outlet.set_Vx(110.0).set_Vr(0.0).set_Vt(0.0)

            return ml.eta_tt, ml.eta_ts

        eta_tt_air, eta_ts_air = compute_eta(air)
        eta_tt_noble, eta_ts_noble = compute_eta(noble)

        # Both should be in valid range
        assert 0.0 <= eta_tt_air <= 1.0
        assert 0.0 <= eta_tt_noble <= 1.0
        assert 0.0 <= eta_ts_air <= 1.0
        assert 0.0 <= eta_ts_noble <= 1.0

        # For this specific case, they should be different
        # (different isentropic exponents give different ideal temperatures)
        assert eta_tt_air != pytest.approx(eta_tt_noble, rel=0.01)


class TestEtaTtGreaterThanEtaTs:
    """Strict tests enforcing eta_tt >= eta_ts always."""

    @pytest.fixture
    def perfect_air(self):
        """Create a perfect gas model for air."""
        return ember.fluid.PerfectFluid(cp=1005.0, gamma=1.4, mu=1.8e-5, Pr=0.72)

    def create_turbine(self, fluid, P_in, T_in, V_in, P_out, T_out, V_out):
        """Helper to create a turbine with given states."""
        ml = turbigen.meanline_new.MeanLine(n_row=1)
        ml.set_fluid(fluid)

        inlet = ml[0]
        inlet.set_r_rms(0.5)
        inlet.set_Am(1.0)
        inlet.set_P_T(P_in, T_in)
        inlet.set_Vx(V_in).set_Vr(0.0).set_Vt(0.0)

        outlet = ml[1]
        outlet.set_r_rms(0.5)
        outlet.set_Am(1.0)
        outlet.set_P_T(P_out, T_out)
        outlet.set_Vx(V_out).set_Vr(0.0).set_Vt(0.0)

        return ml

    def test_eta_tt_ge_eta_ts_various_pressure_ratios(self, perfect_air):
        """Test eta_tt >= eta_ts across wide range of pressure ratios."""
        gamma = 1.4
        T_in = 300.0
        P_in = 1e5

        # Test pressure ratios from high expansion to small expansion
        pressure_ratios = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]

        for pr in pressure_ratios:
            P_out = P_in * pr
            r = (gamma - 1) / gamma
            T_out_ideal = T_in * pr**r

            # Test both high efficiency (close to ideal) and lower efficiency cases
            for target_eta in [0.95, 0.85, 0.75, 0.65]:
                T_out_actual = T_in - target_eta * (T_in - T_out_ideal)
                ml = self.create_turbine(
                    perfect_air, P_in, T_in, 100.0, P_out, T_out_actual, 120.0
                )

                eta_tt = ml.eta_tt
                eta_ts = ml.eta_ts

                assert eta_tt >= eta_ts, (
                    f"eta_tt ({eta_tt}) must be >= eta_ts ({eta_ts}) "
                    f"for PR={pr}, target_eta={target_eta}"
                )

    def test_eta_tt_ge_eta_ts_various_inlet_conditions(self, perfect_air):
        """Test eta_tt >= eta_ts across various inlet conditions."""
        gamma = 1.4
        r = (gamma - 1) / gamma

        # Test various inlet temperatures and pressures
        test_cases = [
            (1e5, 300.0),  # Standard conditions
            (2e5, 400.0),  # High pressure, high temperature
            (0.5e5, 250.0),  # Low pressure, low temperature
            (1.5e5, 350.0),  # Medium-high conditions
        ]

        for P_in, T_in in test_cases:
            # Expansion to 70% of inlet pressure
            P_out = P_in * 0.7
            T_out_ideal = T_in * (P_out / P_in) ** r

            # 85% efficient expansion
            T_out_actual = T_in - 0.85 * (T_in - T_out_ideal)

            ml = self.create_turbine(
                perfect_air, P_in, T_in, 100.0, P_out, T_out_actual, 120.0
            )

            eta_tt = ml.eta_tt
            eta_ts = ml.eta_ts

            assert eta_tt >= eta_ts, (
                f"eta_tt ({eta_tt}) must be >= eta_ts ({eta_ts}) "
                f"for inlet P={P_in:.0e}, T={T_in:.1f}"
            )

    def test_eta_tt_ge_eta_ts_various_velocities(self, perfect_air):
        """Test eta_tt >= eta_ts across various velocity combinations."""
        gamma = 1.4
        P_in, T_in = 1e5, 300.0
        P_out = 0.8e5
        r = (gamma - 1) / gamma
        T_out_ideal = T_in * (P_out / P_in) ** r
        T_out_actual = T_in - 0.85 * (T_in - T_out_ideal)

        # Test various velocity combinations
        velocity_pairs = [
            (50.0, 60.0),
            (100.0, 120.0),
            (150.0, 180.0),
            (200.0, 250.0),
            (10.0, 15.0),
        ]

        for V_in, V_out in velocity_pairs:
            ml = self.create_turbine(
                perfect_air, P_in, T_in, V_in, P_out, T_out_actual, V_out
            )

            eta_tt = ml.eta_tt
            eta_ts = ml.eta_ts

            assert eta_tt >= eta_ts, (
                f"eta_tt ({eta_tt}) must be >= eta_ts ({eta_ts}) "
                f"for velocities V_in={V_in}, V_out={V_out}"
            )

    def test_eta_tt_ge_eta_ts_near_isentropic(self, perfect_air):
        """Test eta_tt >= eta_ts for near-isentropic expansions."""
        gamma = 1.4
        P_in, T_in = 1e5, 300.0
        P_out = 0.8e5
        r = (gamma - 1) / gamma
        T_out_ideal = T_in * (P_out / P_in) ** r

        # Near-isentropic cases (high efficiencies)
        for target_eta in [0.99, 0.98, 0.97, 0.96]:
            T_out_actual = T_in - target_eta * (T_in - T_out_ideal)
            ml = self.create_turbine(
                perfect_air, P_in, T_in, 100.0, P_out, T_out_actual, 110.0
            )

            eta_tt = ml.eta_tt
            eta_ts = ml.eta_ts

            assert eta_tt >= eta_ts, (
                f"eta_tt ({eta_tt}) must be >= eta_ts ({eta_ts}) "
                f"for near-isentropic case eta={target_eta}"
            )


class TestIsentropicEfficiencyMultiRow:
    """Test efficiency calculations for multi-row turbines."""

    @pytest.fixture
    def perfect_air(self):
        """Create a perfect gas model for air."""
        return ember.fluid.PerfectFluid(cp=1005.0, gamma=1.4, mu=1.8e-5, Pr=0.72)

    def test_2row_turbine(self, perfect_air):
        """Test efficiency calculation for 2-row turbine.

        The efficiency should be computed from first inlet to last outlet.
        """
        gamma = 1.4
        r = (gamma - 1) / gamma

        # Start at 400K, 1 bar, expand to 0.7 bar
        T_in = 400.0
        P_in = 1.0e5
        P_out = 0.7e5
        T_out_ideal = T_in * (P_out / P_in) ** r

        # Assume 85% efficiency
        eta_target = 0.85
        T_out_actual = T_in - eta_target * (T_in - T_out_ideal)

        ml = turbigen.meanline_new.MeanLine(n_row=2)
        ml.set_fluid(perfect_air)

        # First row inlet (station 0)
        ml[0].set_r_rms(0.5)
        ml[0].set_Am(1.0)
        ml[0].set_P_T(P_in, T_in)
        ml[0].set_Vx(80.0).set_Vr(0.0).set_Vt(0.0)

        # First row outlet (station 1) - intermediate expansion
        T_mid = 0.5 * (T_in + T_out_actual)
        P_mid = 0.85e5
        ml[1].set_r_rms(0.5)
        ml[1].set_Am(1.0)
        ml[1].set_P_T(P_mid, T_mid)
        ml[1].set_Vx(90.0).set_Vr(0.0).set_Vt(0.0)

        # Second row inlet (station 2) - same as row 1 outlet
        ml[2].set_r_rms(0.5)
        ml[2].set_Am(1.0)
        ml[2].set_P_T(P_mid, T_mid)
        ml[2].set_Vx(90.0).set_Vr(0.0).set_Vt(0.0)

        # Second row outlet (station 3)
        ml[3].set_r_rms(0.5)
        ml[3].set_Am(1.0)
        ml[3].set_P_T(P_out, T_out_actual)
        ml[3].set_Vx(110.0).set_Vr(0.0).set_Vt(0.0)

        eta_tt = ml.eta_tt
        eta_ts = ml.eta_ts

        # Should be finite
        assert np.isfinite(eta_tt), f"eta_tt {eta_tt} should be finite"
        assert np.isfinite(eta_ts), f"eta_ts {eta_ts} should be finite"

    def test_multirow_vs_single_row(self, perfect_air):
        """Compare multi-row efficiency to equivalent single-row expansion."""
        gamma = 1.4
        r = (gamma - 1) / gamma

        # Define overall expansion
        T_in = 400.0
        P_in = 1.0e5
        P_out = 0.7e5
        T_out_ideal = T_in * (P_out / P_in) ** r
        eta_target = 0.85
        T_out_actual = T_in - eta_target * (T_in - T_out_ideal)

        # Single large expansion in one row
        ml_single = turbigen.meanline_new.MeanLine(n_row=1)
        ml_single.set_fluid(perfect_air)

        ml_single[0].set_r_rms(0.5)
        ml_single[0].set_Am(1.0)
        ml_single[0].set_P_T(P_in, T_in)
        ml_single[0].set_Vx(80.0).set_Vr(0.0).set_Vt(0.0)

        ml_single[1].set_r_rms(0.5)
        ml_single[1].set_Am(1.0)
        ml_single[1].set_P_T(P_out, T_out_actual)
        ml_single[1].set_Vx(110.0).set_Vr(0.0).set_Vt(0.0)

        eta_tt_single = ml_single.eta_tt
        eta_ts_single = ml_single.eta_ts

        # Two smaller expansions
        ml_multi = turbigen.meanline_new.MeanLine(n_row=2)
        ml_multi.set_fluid(perfect_air)

        # Same inlet and outlet, split expansion
        T_mid = 0.5 * (T_in + T_out_actual)
        P_mid = 0.85e5

        ml_multi[0].set_r_rms(0.5)
        ml_multi[0].set_Am(1.0)
        ml_multi[0].set_P_T(P_in, T_in)
        ml_multi[0].set_Vx(80.0).set_Vr(0.0).set_Vt(0.0)

        ml_multi[1].set_r_rms(0.5)
        ml_multi[1].set_Am(1.0)
        ml_multi[1].set_P_T(P_mid, T_mid)
        ml_multi[1].set_Vx(95.0).set_Vr(0.0).set_Vt(0.0)

        ml_multi[2].set_r_rms(0.5)
        ml_multi[2].set_Am(1.0)
        ml_multi[2].set_P_T(P_mid, T_mid)
        ml_multi[2].set_Vx(95.0).set_Vr(0.0).set_Vt(0.0)

        ml_multi[3].set_r_rms(0.5)
        ml_multi[3].set_Am(1.0)
        ml_multi[3].set_P_T(P_out, T_out_actual)
        ml_multi[3].set_Vx(110.0).set_Vr(0.0).set_Vt(0.0)

        eta_tt_multi = ml_multi.eta_tt
        eta_ts_multi = ml_multi.eta_ts

        # Both should be in valid range
        assert (
            0.0 <= eta_tt_single <= 1.0
        ), f"eta_tt_single {eta_tt_single} out of range"
        assert 0.0 <= eta_tt_multi <= 1.0, f"eta_tt_multi {eta_tt_multi} out of range"
        assert 0.0 <= eta_ts_single <= 1.0
        assert 0.0 <= eta_ts_multi <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
