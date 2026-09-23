"""Tests for the Clark loading curve.

Test cases:
- test_both_surfaces_start_at_stagnation: Ma / Ma_te is zero at z = 0
- test_both_surfaces_end_at_the_trailing_edge_value: and exactly one at z = 1
- test_the_nose_factor_recovers_the_plateau: the parabola hands over downstream
- test_the_suction_ramp_is_approached_from_below: and never overshoots the line
- test_a_ramp_negative_at_the_nose_is_refused: no negative Mach out of stagnation
"""

import numpy as np
import pytest

from turbigen import clark

STYLE = {"Ma_peak": 1.3, "z_peak": 0.55, "Ma_LE": 0.62, "Ma_PS": 0.2}


def test_both_surfaces_start_at_stagnation():
    ss, ps = clark.loading(0.0, **STYLE)
    assert ss == 0.0
    assert ps == 0.0


def test_both_surfaces_end_at_the_trailing_edge_value():
    ss, ps = clark.loading(1.0, **STYLE)
    assert ss == pytest.approx(1.0, abs=1e-12)
    assert ps == pytest.approx(1.0, abs=1e-12)


def test_the_nose_factor_recovers_the_plateau():
    """A few nose radii downstream the pressure surface is on its plateau."""
    ps = clark.pressure(10.0 * clark.R_NOSE_PS, **STYLE)
    assert ps == pytest.approx(STYLE["Ma_PS"], rel=0.01)


def test_the_suction_ramp_is_approached_from_below():
    sigma = (
        clark.C_RAMP
        * (STYLE["Ma_peak"] - STYLE["Ma_LE"])
        / (STYLE["z_peak"] - clark.Z_LE)
    )
    z = np.linspace(0.0, clark.Z_LE, 101)
    ramp = STYLE["Ma_LE"] + sigma * (z - clark.Z_LE)
    ss = clark.suction(z, **STYLE)
    assert np.all(ss <= ramp)
    assert ss[-1] == pytest.approx(ramp[-1], rel=0.03)


def test_a_ramp_negative_at_the_nose_is_refused():
    with pytest.raises(ValueError, match="positive at the nose"):
        clark.suction(0.5, Ma_peak=1.6, z_peak=0.2, Ma_LE=0.1, Ma_PS=0.2)
