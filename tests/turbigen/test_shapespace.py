"""Tests for the shape space arithmetic camber and thickness share.

Free functions over arrays, so these are about the maths rather than about a
design: that the Bernstein basis is the one everybody else means by it, that
degree elevation is exact and leaves the ends alone, and that the class
function and its endpoint values invert each other.
"""

import numpy as np
import pytest

from turbigen import shapespace

M = np.linspace(0.0, 1.0, 51)
"""Positions to compare curves at, ends included."""


#
# DOMAIN
#


@pytest.mark.parametrize("m", (-1e-9, 1.0 + 1e-9, -1.0, 2.0))
def test_domain_rejects_positions_off_the_chord(m):
    with pytest.raises(ValueError, match="range"):
        shapespace.validate_domain(m)


def test_domain_accepts_the_whole_chord():
    shapespace.validate_domain(M)


#
# BERNSTEIN CURVES
#


def test_bernstein_coefficients_sum_to_one():
    """Partition of unity: all-ones coefficients give the constant one.

    The defining property of the basis, and the one that says the binomial
    weights are right.
    """
    for order in range(1, 8):
        value = shapespace.evaluate_bernstein(np.ones(order + 1), M)
        np.testing.assert_allclose(value, 1.0, atol=1e-14)


def test_bernstein_reproduces_a_straight_line():
    """Linear precision: evenly spaced coefficients give the straight line.

    Exactly, not approximately --- which is what lets a curve be written as a
    line plus a perturbation, with the line added wherever is convenient.
    """
    for order in range(1, 8):
        coeff = np.linspace(-0.3, 1.7, order + 1)
        value = shapespace.evaluate_bernstein(coeff, M)
        np.testing.assert_allclose(value, -0.3 + 2.0 * M, atol=1e-14)


def test_bernstein_ends_at_its_end_coefficients():
    """The first and last coefficients are the values at m=0 and m=1.

    What makes it possible to write a leading edge radius or a wedge angle
    into a curve and know it will still be there.
    """
    coeff = (0.2, -1.0, 3.0, 0.5)
    value = shapespace.evaluate_bernstein(coeff, M)
    assert value[0] == pytest.approx(coeff[0], abs=1e-14)
    assert value[-1] == pytest.approx(coeff[-1], abs=1e-14)


def test_bernstein_matches_the_cubic_written_out():
    """Against the textbook polynomial, spelled out term by term."""
    b0, b1, b2, b3 = 0.2, -1.0, 3.0, 0.5
    expected = (
        b0 * (1.0 - M) ** 3.0
        + 3.0 * b1 * M * (1.0 - M) ** 2.0
        + 3.0 * b2 * M**2.0 * (1.0 - M)
        + b3 * M**3.0
    )
    np.testing.assert_allclose(
        shapespace.evaluate_bernstein((b0, b1, b2, b3), M), expected, atol=1e-14
    )


def test_bernstein_returns_a_scalar_for_a_scalar():
    value = shapespace.evaluate_bernstein((0.0, 1.0, 0.0), 0.5)
    assert np.isscalar(value) or value.ndim == 0
    assert value == pytest.approx(0.5)


@pytest.mark.parametrize("coeff", ((1.0,), (), np.zeros((2, 2))))
def test_bernstein_needs_a_row_of_at_least_two_coefficients(coeff):
    with pytest.raises(ValueError, match="at least two"):
        shapespace.evaluate_bernstein(coeff, M)


#
# DEGREE ELEVATION
#


def test_elevation_leaves_the_curve_alone():
    """The whole point: a higher basis, the same shape."""
    coeff = (0.2, -1.0, 3.0, 0.5)
    low = shapespace.evaluate_bernstein(coeff, M)

    for order in range(3, 12):
        high = shapespace.evaluate_bernstein(
            shapespace.elevate_bernstein(coeff, order), M
        )
        np.testing.assert_allclose(high, low, atol=1e-13)


def test_elevation_gives_the_order_asked_for():
    for order in range(2, 9):
        assert len(shapespace.elevate_bernstein((0.0, 1.0, 0.0), order)) == order + 1


def test_elevation_keeps_the_ends_pinned():
    """A perturbation pinned at zero stays pinned, however far it is raised.

    What lets a camber line keep its end angles and a thickness keep its
    leading edge radius while an iterator moves the interior.
    """
    for order in range(3, 12):
        elevated = shapespace.elevate_bernstein((0.0, 0.7, -0.4, 0.0), order)
        assert elevated[0] == 0.0
        assert elevated[-1] == 0.0


def test_elevation_carries_a_line_to_the_same_line():
    """A straight line elevates to the straight line at the new order.

    The invariant behind reading a low-order design as a high-order one: the
    line and the perturbation can be raised separately and added back at
    either end of the operation.
    """
    for order in range(3, 12):
        elevated = shapespace.elevate_bernstein(np.linspace(0.0, 1.0, 4), order)
        np.testing.assert_allclose(
            elevated, np.linspace(0.0, 1.0, order + 1), atol=1e-14
        )


def test_elevation_to_its_own_order_changes_nothing():
    coeff = (0.2, -1.0, 3.0, 0.5)
    np.testing.assert_array_equal(shapespace.elevate_bernstein(coeff, 3), coeff)


def test_elevation_cannot_lower_the_order():
    with pytest.raises(ValueError, match="only raises"):
        shapespace.elevate_bernstein((0.2, -1.0, 3.0, 0.5), 2)


def test_elevation_needs_a_row_of_at_least_two_coefficients():
    with pytest.raises(ValueError, match="at least two"):
        shapespace.elevate_bernstein((1.0,), 4)


#
# THE CLASS FUNCTION
#


def test_thickness_closes_the_leading_edge():
    """Zero thickness at m=0, whatever the shape space curve says."""
    for tau in (0.1, 1.0, 5.0):
        t = shapespace.thickness_from_tau(0.0, tau, t_TE=0.02)
        assert t == pytest.approx(0.0, abs=1e-14)


def test_thickness_leaves_half_the_trailing_edge():
    """The trailing edge thickness is the total due to both sides."""
    t_TE = 0.03
    t = shapespace.thickness_from_tau(1.0, 2.0, t_TE)
    assert t == pytest.approx(t_TE / 2.0, abs=1e-14)


def test_thickness_has_a_square_root_nose():
    """Near the leading edge the half-thickness grows as the square root.

    Which is the reason for the class function: a nose with a radius, rather
    than a wedge.
    """
    tau = 0.3
    m = np.array([1e-8, 4e-8, 9e-8])
    t = shapespace.thickness_from_tau(m, tau, t_TE=0.0)
    np.testing.assert_allclose(t / np.sqrt(m), tau, rtol=1e-6)


def test_shape_space_inverts_the_class_function():
    """Round trip, at every position the inverse is defined at."""
    m = M[1:-1]
    tau = 0.5 + 0.4 * np.sin(3.0 * m)
    t = shapespace.thickness_from_tau(m, tau, t_TE=0.02)
    np.testing.assert_allclose(
        shapespace.tau_from_thickness(m, t, t_TE=0.02), tau, rtol=1e-12
    )


@pytest.mark.parametrize("m", (0.0, 1.0, np.array([0.5, 1.0])))
def test_shape_space_is_undefined_at_the_ends(m):
    """Where the class function vanishes, every curve gives the same thickness."""
    with pytest.raises(ValueError, match="undefined at the ends"):
        shapespace.tau_from_thickness(m, 0.01, t_TE=0.02)


def test_shape_space_rejects_positions_off_the_chord():
    with pytest.raises(ValueError, match="range"):
        shapespace.tau_from_thickness(1.5, 0.01, t_TE=0.02)


#
# ENDPOINT VALUES
#


def test_leading_edge_value_gives_back_its_radius():
    for R_LE in (0.002, 0.04, 0.5):
        tau = shapespace.tau_LE(R_LE)
        assert shapespace.R_LE_from_tau(tau) == pytest.approx(R_LE, rel=1e-14)


def test_leading_edge_radius_is_the_nose_of_the_thickness():
    """`tau_LE` is what makes the thickness match a circle at the nose.

    Checked against the circle itself: a nose of radius `R_LE` has
    half-thickness `sqrt(2 R_LE m)` near `m=0`, so this is the statement that
    the coefficient is the one that lands on it.
    """
    R_LE = 0.01
    m = np.array([1e-8, 4e-8, 9e-8])
    t = shapespace.thickness_from_tau(m, shapespace.tau_LE(R_LE), t_TE=0.0)
    np.testing.assert_allclose(t, np.sqrt(2.0 * R_LE * m), rtol=1e-6)


def test_trailing_edge_value_gives_back_its_wedge():
    for t_TE, tanwedge in ((0.0, 0.25), (0.02, 0.25), (0.05, 0.0)):
        tau = shapespace.tau_TE(t_TE, tanwedge)
        assert shapespace.tanwedge_from_tau(tau, t_TE) == pytest.approx(
            tanwedge, abs=1e-14
        )


def test_trailing_edge_wedge_is_the_slope_of_the_thickness():
    """`tau_TE` is what makes the thickness close at the wedge angle.

    On a closed trailing edge the half-thickness arrives with slope
    `-tanwedge` against `m`, which is what that coefficient buys. A finite
    `t_TE` steepens it by a further `t_TE / 2`, the ramp that holds the
    trailing edge open arriving alongside it --- so the wedge is read on a
    closed trailing edge, where it is the only thing setting the slope.
    """
    tanwedge = 0.25
    tau = shapespace.tau_TE(0.0, tanwedge)

    m = 1.0 - np.array([1e-7, 2e-7])
    t = shapespace.thickness_from_tau(m, tau, t_TE=0.0)
    slope = np.diff(t) / np.diff(m)
    np.testing.assert_allclose(slope, -tanwedge, rtol=1e-5)
