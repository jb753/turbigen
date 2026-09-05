"""Tests for camber line shapes.

A :class:`~turbigen.camber.CamberDesign` is only the shape between the end
angles; these check the two built-in shapes evaluate correctly and that the
``bernstein`` shape reduces to ``quadratic`` when it is not perturbed, matching
the Bernstein camber line carried by the package this replaces.

Test cases:
- test_zero_coefficients_match_quadratic: an unperturbed Bernstein is quadratic
- test_endpoints_are_pinned: chi_hat is 0 at the LE and 1 at the TE
- test_short_coefficients_is_an_error: a short coeff tuple is refused
- test_upsample_matches_the_lower_order_curve: upsample raises it exactly
- test_too_many_coefficients_is_an_error: a long one is refused
- test_bad_order_is_an_error: order below 2 is refused
- test_selected_from_a_config_dict: it round-trips through the Node protocol
- test_matches_reference_bernstein: the port agrees with the old maths
- test_wraps_into_a_camber_line: it drives a CamberLine to the end tangents
"""

import numpy as np
import pytest
import turbigen_ref.new_geometry

from turbigen import Bernstein, Quadratic
from turbigen.camber import CamberDesign, CamberLine

M = np.linspace(0.0, 1.0, 51)


def test_zero_coefficients_match_quadratic():
    """All-zero coefficients recover a quadratic camber line."""
    quad = Quadratic(aft_loading=0.0).chi_hat(M)
    assert np.allclose(Bernstein(order=3).chi_hat(M), quad)
    assert np.allclose(Bernstein(order=4, coeff=(0.0, 0.0, 0.0)).chi_hat(M), quad)


def test_endpoints_are_pinned():
    """The camber line ends stay put whatever the coefficients."""
    shape = Bernstein(order=5, coeff=(0.3, -0.2, 0.4, 0.1))
    assert shape.chi_hat(0.0) == pytest.approx(0.0)
    assert shape.chi_hat(1.0) == pytest.approx(1.0)


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
    assert np.allclose(high.chi_hat(M), low.chi_hat(M))


def test_too_many_coefficients_is_an_error():
    with pytest.raises(ValueError):
        Bernstein(order=3, coeff=(0.1, 0.2, 0.3)).chi_hat(M)


def test_bad_order_is_an_error():
    with pytest.raises(ValueError):
        Bernstein(order=1).chi_hat(M)


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
    assert np.allclose(Bernstein(order=order, coeff=coeff).chi_hat(M), reference)


def test_wraps_into_a_camber_line():
    """Placed between end tangents, it reaches them at the ends."""
    line = CamberLine(Bernstein(order=4, coeff=(0.2, -0.1, 0.3)), 0.1, 1.4)
    assert line.dydm(0.0) == pytest.approx(0.1)
    assert line.dydm(1.0) == pytest.approx(1.4)
    assert np.all(np.isfinite(line.chi(M)))
