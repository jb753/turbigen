"""Tests for placing a measured point back onto a blade.

`turbigen.loading` mostly needs a solved grid to say anything, and the tests
that exercise that live with the iterators and metrics that consume it. This is
the part that does not: `locate_arc_length` takes coordinates and a blade, so
what it promises can be checked against a point whose arc length is known by
construction, with no flow anywhere.
"""

import numpy as np
import pytest
from test_blade import SPF, build

import turbigen.util
from turbigen.blade import to_xrrt
from turbigen.loading import locate_arc_length

NCHORD = 2001
"""Coarse enough to be quick, fine enough for a nearest node to land close."""


@pytest.fixture(scope="module")
def machine():
    return build().design()


def _point_at(blade, spf, i_surf, m_point):
    """Return a point on one surface, and its true arc length there.

    Sampled off the same dense curve the search walks, so what comes back is a
    point that is exactly on the blade rather than one merely near it --- which
    is the case the docstring's "within microns" claim is about.
    """
    m = turbigen.util.cluster_cosine(NCHORD)
    surfaces = blade.evaluate_section(spf, m=m)
    _, s = blade.evaluate_arc_length(spf, m=m)

    j = int(np.argmin(np.abs(m - m_point)))
    return to_xrrt(surfaces[i_surf])[:, j], float(s[i_surf][j])


@pytest.mark.parametrize("i_row", [0, 1])
@pytest.mark.parametrize("m_point", [0.1, 0.5, 0.9])
def test_a_point_on_the_suction_surface_keeps_its_arc_length(machine, i_row, m_point):
    """Positive, and the length it was sampled at."""
    blade = machine.rows[i_row].blade
    xrt, s_true = _point_at(blade, 0.5, 0, m_point)

    assert locate_arc_length(blade, 0.5, xrt, nchord=NCHORD) == pytest.approx(
        s_true, rel=1e-9
    )
    assert s_true > 0.0


@pytest.mark.parametrize("i_row", [0, 1])
@pytest.mark.parametrize("m_point", [0.1, 0.5, 0.9])
def test_a_point_on_the_pressure_surface_comes_back_negated(machine, i_row, m_point):
    """The sign is the whole point of searching both surfaces.

    A stagnation point at any positive incidence sits on the pressure side of
    the nose, and it is `sigma` from the leading edge the other way round it.
    So a suction-surface point at geometric `s` stands `s + sigma` from it
    along the blade, which is what subtracting a negative answer gives. A
    search restricted to the suction surface answers with approximately the
    leading edge instead --- silently, and wrong by exactly the distance that
    matters most.

    Checked on both rows, which turn opposite ways, so the sign is a statement
    about the surface rather than about which way this particular blade bends.
    """
    blade = machine.rows[i_row].blade
    xrt, s_true = _point_at(blade, 0.5, 1, m_point)

    assert locate_arc_length(blade, 0.5, xrt, nchord=NCHORD) == pytest.approx(
        -s_true, rel=1e-9
    )
    assert s_true > 0.0


@pytest.mark.parametrize("i_row", [0, 1])
def test_the_leading_edge_is_the_origin_either_way(machine, i_row):
    """Where the two surfaces meet, so the sign has nothing to choose between."""
    blade = machine.rows[i_row].blade
    for i_surf in (0, 1):
        xrt, _ = _point_at(blade, 0.5, i_surf, 0.0)
        assert locate_arc_length(blade, 0.5, xrt, nchord=NCHORD) == pytest.approx(
            0.0, abs=1e-12
        )


@pytest.mark.parametrize("spf", SPF)
def test_a_point_off_the_surface_lands_on_the_nearest_part_of_it(machine, spf):
    """Nudged into the passage, it still reports where it came from.

    What a measurement actually offers: a node one cell off the wall, or a
    stagnation point found on a grid rather than on the design. Displaced here
    by a hundredth of the surface length, which is far more than either.
    """
    blade = machine.rows[0].blade
    xrt, s_true = _point_at(blade, spf, 0, 0.4)

    length = blade.evaluate_surface_length(spf)[0]
    nudged = xrt + np.array([0.0, 0.0, 0.01 * length])

    assert locate_arc_length(blade, spf, nudged, nchord=NCHORD) == pytest.approx(
        s_true, rel=0.05
    )
