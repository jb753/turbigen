"""Test annulus geometry classes."""

import pytest
import numpy as np
from turbigen.annulus import Smooth


@pytest.fixture
def smooth_annulus_two_row():
    """Create a Smooth annulus with two blade rows."""
    # Two rows means 4 stations: inlet/exit for each row
    nrow = 2
    npt = nrow * 2  # 4 points

    # Define geometric parameters
    rmid = np.array([0.5, 0.55, 0.6, 0.65])  # Mid-span radii
    span = np.array([0.2, 0.22, 0.24, 0.26])  # Spans
    Beta = np.array([-20.0, 10.0, 20.0, 30.0])  # Pitch angles [deg]

    # Aspect ratios
    AR_chord = np.array([1.5, 1.8])  # For 2 rows
    AR_gap = np.array([2.0, 2.2, 2.4])  # For 3 gaps (inlet, middle, exit)

    # Create the annulus with design variables
    design_vars = {
        "AR_chord": AR_chord,
        "AR_gap": AR_gap,
    }
    ann = Smooth(design_vars)
    ann.forward(rmid, span, Beta, AR_chord, AR_gap)

    return ann, rmid, span, Beta


def test_smooth_evaluate_xr_at_midspan(smooth_annulus_two_row):
    """Test that evaluate_xr returns rmid at span fraction 0.5."""
    ann, rmid, span, Beta = smooth_annulus_two_row

    # Evaluate at all row stations
    # The meridional coordinate m maps as:
    # 0=inlet, 1=row1_LE, 2=row1_TE, 3=row2_LE, 4=row2_TE, 5=exit
    m_values = np.array([1.0, 2.0, 3.0, 4.0])
    spf = 0.5

    # Evaluate xr coordinates at mid-span
    xr = ann.evaluate_xr(m_values, spf)

    # Extract radii (second component of xr)
    r_evaluated = xr[1, :]

    # Expected radii at these stations
    # rmid is indexed as: 0=row1_inlet, 1=row1_exit, 2=row2_inlet, 3=row2_exit
    # m=1 (row1 LE) should match rmid[0]
    # m=2 (row1 TE) should match rmid[1]
    # m=3 (row2 LE) should match rmid[2]
    # m=4 (row2 TE) should match rmid[3]
    r_expected = rmid

    # Verify that evaluated radii match input rmid at mid-span
    np.testing.assert_allclose(r_evaluated, r_expected, rtol=1e-3)


def test_smooth_span_distance(smooth_annulus_two_row):
    """Test that distance between hub and casing matches input span."""
    ann, rmid, span, Beta = smooth_annulus_two_row

    # Evaluate at all row stations
    # The meridional coordinate m maps as:
    # 0=inlet, 1=row1_LE, 2=row1_TE, 3=row2_LE, 4=row2_TE, 5=exit
    m_values = np.array([1.0, 2.0, 3.0, 4.0])

    # Evaluate xr coordinates at hub (spf=0) and casing (spf=1)
    xr_hub = ann.evaluate_xr(m_values, spf=0.0)
    xr_cas = ann.evaluate_xr(m_values, spf=1.0)

    # Calculate arc length distance between hub and casing at each station
    # The distance is perpendicular to the pitch angle
    span_evaluated = np.linalg.norm(xr_cas - xr_hub, axis=0)

    # Expected span at these stations
    # span is indexed as: 0=row1_inlet, 1=row1_exit, 2=row2_inlet, 3=row2_exit
    span_expected = span

    # Verify that evaluated span matches input span
    np.testing.assert_allclose(span_evaluated, span_expected, rtol=1e-3)


def test_smooth_pitch_angle(smooth_annulus_two_row):
    """Test that pitch angle Beta matches the span vector orientation."""
    ann, rmid, span, Beta = smooth_annulus_two_row

    # Evaluate at all row stations
    # The meridional coordinate m maps as:
    # 0=inlet, 1=row1_LE, 2=row1_TE, 3=row2_LE, 4=row2_TE, 5=exit
    m_values = np.array([1.0, 2.0, 3.0, 4.0])

    # Evaluate xr coordinates at hub (spf=0) and casing (spf=1)
    xr_hub = ann.evaluate_xr(m_values, spf=0.0)
    xr_cas = ann.evaluate_xr(m_values, spf=1.0)

    # Calculate the span vector from hub to casing
    delta_xr = xr_cas - xr_hub

    # Extract axial (x) and radial (r) components
    dx = delta_xr[0, :]  # axial component
    dr = delta_xr[1, :]  # radial component

    # Calculate pitch angle from geometry
    # Beta is the angle from the radial direction (not axial)
    # From the implementation: dx = -span*sinBeta, dr = span*cosBeta
    # So tan(Beta) = -dx/dr
    Beta_evaluated = np.degrees(np.arctan2(-dx, dr))

    # Expected pitch angles at these stations
    # Beta is indexed as: 0=row1_inlet, 1=row1_exit, 2=row2_inlet, 3=row2_exit
    Beta_expected = Beta

    # Verify that evaluated pitch angle matches input Beta
    np.testing.assert_allclose(Beta_evaluated, Beta_expected, atol=0.1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
