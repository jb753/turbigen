"""Tests for camber line shapes.

A :class:`~turbigen.camber.CamberDesign` states the camber line slope between
the end tangents it is handed; these check the built-in shapes evaluate
correctly, that the ``bernstein`` shape reduces to ``quadratic`` when it is not
perturbed, matching the Bernstein camber line carried by the package this
replaces, and that ``circular_arc`` is the constant-curvature line the end
angles alone fix.

The polynomial shapes are evaluated between tangents of 0 and 1, where the
slope is exactly the normalised camber the old protocol stated, so the
reference comparison is the same comparison it always was.

Test cases:
- test_zero_coefficients_match_quadratic: an unperturbed Bernstein is quadratic
- test_endpoints_are_pinned: the slope is the end tangent at either end
- test_short_coefficients_is_an_error: a short coeff tuple is refused
- test_upsample_matches_the_lower_order_curve: upsample raises it exactly
- test_too_many_coefficients_is_an_error: a long one is refused
- test_bad_order_is_an_error: order below 2 is refused
- test_selected_from_a_config_dict: it round-trips through the Node protocol
- test_matches_reference_bernstein: the port agrees with the old maths
- test_wraps_into_a_camber_line: it drives a CamberLine to the end tangents
- test_arc_reaches_the_end_tangents: the arc starts and ends where it is told
- test_arc_turns_uniformly: sin of the arc's camber angle is linear in m
- test_arc_of_equal_angles_is_straight: no turning gives a straight line
- test_arc_selected_from_a_config_dict: the arc round-trips with no parameters
- test_arc_wraps_into_a_camber_line: its angle runs monotonically between ends
"""

import numpy as np
import pytest
import turbigen_ref.new_geometry

from turbigen import Bernstein, CircularArc, Quadratic
from turbigen.camber import CamberDesign, CamberLine

M = np.linspace(0.0, 1.0, 51)


def test_zero_coefficients_match_quadratic():
    """All-zero coefficients recover a quadratic camber line."""
    quad = Quadratic(aft_loading=0.0).dydm(M, 0.0, 1.0)
    assert np.allclose(Bernstein(order=3).dydm(M, 0.0, 1.0), quad)
    assert np.allclose(
        Bernstein(order=4, coeff=(0.0, 0.0, 0.0)).dydm(M, 0.0, 1.0), quad
    )


def test_endpoints_are_pinned():
    """The camber line ends stay put whatever the coefficients."""
    shape = Bernstein(order=5, coeff=(0.3, -0.2, 0.4, 0.1))
    assert shape.dydm(0.0, 0.2, 1.3) == pytest.approx(0.2)
    assert shape.dydm(1.0, 0.2, 1.3) == pytest.approx(1.3)


def test_short_coefficients_is_an_error():
    """A non-empty coeff shorter than order - 1 is refused without upsample."""
    with pytest.raises(ValueError):
        Bernstein(order=4, coeff=(0.3,))


def test_upsample_matches_the_lower_order_curve():
    """upsample=True raises a short coeff to order, leaving the curve unchanged."""
    coeff = (0.3, -0.2)
    low = Bernstein(order=len(coeff) + 1, coeff=coeff)
    high = Bernstein(order=6, coeff=coeff, upsample=True)
    assert len(high.coeff) == 5
    assert np.allclose(high.dydm(M, 0.0, 1.0), low.dydm(M, 0.0, 1.0))


def test_too_many_coefficients_is_an_error():
    with pytest.raises(ValueError):
        Bernstein(order=3, coeff=(0.1, 0.2, 0.3)).dydm(M, 0.0, 1.0)


def test_bad_order_is_an_error():
    with pytest.raises(ValueError):
        Bernstein(order=1).dydm(M, 0.0, 1.0)


def test_selected_from_a_config_dict():
    """A config mapping builds a Bernstein and dumps back to the same mapping."""
    data = {"type": "bernstein", "order": 4, "coeff": [0.1, -0.2, 0.05]}
    shape = CamberDesign.from_dict(data)
    assert isinstance(shape, Bernstein)
    assert shape.order == 4
    assert shape.coeff == (0.1, -0.2, 0.05)
    assert shape.to_dict() == {**data, "upsample": False}


@pytest.mark.parametrize(
    "coeff",
    [(), (0.25,), (0.3, -0.1, 0.2), (0.0, 0.5, 0.0, -0.4)],
)
def test_matches_reference_bernstein(coeff):
    """The port agrees with the old package's Bernstein camber line."""
    order = len(coeff) + 1 if coeff else 3
    q = list(coeff) + [0.0] * (order - 1 - len(coeff))
    reference = turbigen_ref.new_geometry.Camber.from_design_vector(q).evaluate(M)
    assert np.allclose(Bernstein(order=order, coeff=coeff).dydm(M, 0.0, 1.0), reference)


def test_wraps_into_a_camber_line():
    """Placed between end tangents, it reaches them at the ends."""
    line = CamberLine(Bernstein(order=4, coeff=(0.2, -0.1, 0.3)), 0.1, 1.4)
    assert line.dydm(0.0) == pytest.approx(0.1)
    assert line.dydm(1.0) == pytest.approx(1.4)
    assert np.all(np.isfinite(line.chi(M)))


def test_arc_reaches_the_end_tangents():
    """The arc is placed between the end angles like any other shape."""
    shape = CircularArc()
    assert shape.dydm(0.0, -0.4, 1.7) == pytest.approx(-0.4)
    assert shape.dydm(1.0, -0.4, 1.7) == pytest.approx(1.7)


def test_arc_turns_uniformly():
    """Constant curvature: the sine of the camber angle is linear in m."""
    tanchi = (-0.4, 1.7)
    s = np.sin(np.arctan(CircularArc().dydm(M, *tanchi)))
    s_ends = [t / np.hypot(1.0, t) for t in tanchi]
    assert np.allclose(s, s_ends[0] + M * (s_ends[1] - s_ends[0]))


def test_arc_of_equal_angles_is_straight():
    """Equal end angles give a straight line, not a divided-by-zero one."""
    dydm = CircularArc().dydm(M, 0.8, 0.8)
    assert np.allclose(dydm, 0.8)


def test_arc_selected_from_a_config_dict():
    """A config mapping builds an arc, which carries no parameters at all."""
    shape = CamberDesign.from_dict({"type": "circular_arc"})
    assert isinstance(shape, CircularArc)
    assert shape.to_dict() == {"type": "circular_arc"}


def test_arc_wraps_into_a_camber_line():
    """Its angle runs monotonically from one metal angle to the other."""
    line = CamberLine(
        CircularArc(), np.tan(np.radians(-20.0)), np.tan(np.radians(65.0))
    )
    chi = line.chi(M)
    assert chi[0] == pytest.approx(-20.0)
    assert chi[-1] == pytest.approx(65.0)
    assert np.all(np.diff(chi) > 0.0)
