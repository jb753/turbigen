"""Tests for re-centring a Clark section on its camber line.

Test cases:
- test_a_balanced_section_is_left_alone: nothing to equalise, nothing moves
- test_an_unbalanced_section_is_equalised: the thickness evens out
- test_the_aerofoil_does_not_move: the surfaces trace the same curves
- test_the_exit_angle_and_rise_are_held: only the exponent and inlet move
- test_a_non_clark_camber_is_refused: there is no exponent to move
"""

import numpy as np
import pytest

from turbigen import recentre
from turbigen.camber import CamberLine, ClarkCamber, Quadratic
from turbigen.thickness import ClarkThickness

TANCHI = (-1.1, 2.5)
"""End tangents of a heavily turning stator, like the one that prompted this."""

UNBALANCED = ClarkThickness(
    R_LE=0.024,
    coeff=((-0.31, 0.10, 0.17, -0.50), (0.16, 0.64, 0.27, 0.22)),
    tanwedge=0.66,
    t_TE=0.025,
)
"""A pressure surface about twice as thick as the suction surface."""


def _recentred(thickness=UNBALANCED):
    camber = CamberLine(ClarkCamber(exponent=2.5), *TANCHI)
    return camber, recentre.recentre(camber, thickness, 0.8, False)


@pytest.fixture(scope="module")
def unbalanced():
    return _recentred()


def test_a_balanced_section_is_left_alone():
    """Equal thickness already, so no exponent can improve on it."""
    balanced = ClarkThickness(
        R_LE=0.024, coeff=((0.1, 0.1), (0.1, 0.1)), tanwedge=0.3, t_TE=0.025
    )

    camber, out = _recentred(balanced)

    assert out.camber is camber
    assert out.thickness is balanced
    assert out.error == 0.0


def test_an_unbalanced_section_is_equalised(unbalanced):
    _, out = unbalanced

    assert recentre.imbalance(out.thickness) < 0.1 * recentre.imbalance(UNBALANCED)
    t_s, t_p = out.thickness.thick_both(np.linspace(0.0, 1.0, 201))
    assert t_s.max() == pytest.approx(t_p.max(), abs=0.01)
    assert out.camber.shape.exponent > 2.5


def test_the_aerofoil_does_not_move(unbalanced):
    """Within a small fraction of chord, which is what an order-5 curve can do."""
    _, out = unbalanced

    assert out.error < 0.01


def test_the_exit_angle_and_rise_are_held(unbalanced):
    camber, out = unbalanced

    assert out.camber.tanchi_TE == camber.tanchi_TE
    assert out.thickness.tanwedge == pytest.approx(UNBALANCED.tanwedge)
    rise = np.sum(TANCHI) / 2.5
    assert (
        out.camber.tanchi_LE + out.camber.tanchi_TE
    ) / out.camber.shape.exponent == pytest.approx(rise)


def test_a_non_clark_camber_is_refused():
    camber = CamberLine(Quadratic(), *TANCHI)

    with pytest.raises(ValueError, match="Clark camber exponent"):
        recentre.recentre(camber, UNBALANCED, 0.8, False)
