"""Tests for placing a measured point back onto a blade.

`turbigen.loading` mostly needs a solved grid to say anything, and the tests
that exercise that live with the iterators and metrics that consume it. This is
the part that does not: `locate_arc_length` takes coordinates and a blade, so
what it promises can be checked against a point whose arc length is known by
construction, with no flow anywhere.
"""

import dataclasses

import numpy as np
import pytest
import yaml
from test_blade import SPF, build
from test_cli import RUN_CASE

import turbigen.util
from turbigen import Config, Result, cli
from turbigen.blade import to_xrrt
from turbigen.loading import locate_arc_length, measure_clark_profile

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


#
# BOTH SURFACES, AGAINST A SOLUTION
#
# `measure_clark_profile` needs a flow field, so unlike the placement above it
# cannot be checked against geometry alone. The fixture is the fast cascade,
# marched briefly -- enough for a real distribution, and nowhere near enough
# for the *shape* of one, which is why what follows checks the contract rather
# than the aerodynamics.
#

CLARK = {
    "type": "clark",
    "R_LE": 0.05,
    "tanwedge": [0.18, 0.18],
    "t_TE": 0.03,
    "coeff": [[0.1, 0.05, 0.02], [0.03, -0.05, 0.01]],
}


@pytest.fixture(scope="module")
def solved():
    """The fast cascade, given a two-sided thickness and marched briefly."""
    case = yaml.safe_load(RUN_CASE)
    for section in case["blades"][0]["sections"]:
        section["thickness"] = dict(CLARK)

    config = Config.from_dict(case)
    _, machine, grid = cli.prepare(config)
    config.solver.solve(grid)
    return config, Result(machine=machine, grid=grid, converged=True)


def _thickness(config):
    return config.blades[0].sections[0].thickness


def test_both_surfaces_are_measured(solved):
    """One row of samples per surface, suction first."""
    config, result = solved
    m_ctl = _thickness(config).m_ctl

    z, fac = measure_clark_profile(result, 0, 0.5, m_ctl)

    assert z.shape == (2, len(m_ctl))
    assert fac.shape == (2, len(m_ctl))
    assert np.all(np.isfinite(z)) and np.all(np.isfinite(fac))


def test_each_surface_is_its_own_fraction(solved):
    """`z` runs from nose to trailing edge along each side separately.

    Clark's independent variable is `l / L_surf`, so a surface fraction on the
    suction side and one on the pressure side are fractions of *different*
    lengths --- which is why one `m` does not give one `z`, and why these are
    measured rather than shared.
    """
    config, result = solved
    z, _ = measure_clark_profile(result, 0, 0.5, _thickness(config).m_ctl)

    assert np.all(z > 0.0) and np.all(z < 1.0)
    assert np.all(np.diff(z, axis=1) > 0.0)

    # The two surfaces have different lengths, so the same `m` lands at
    # different fractions along each.
    assert not np.allclose(z[0], z[1])


def test_the_trailing_edge_is_the_reference(solved):
    """`fac` is `Ma / Ma_TE`, so both surfaces arrive at one there.

    Not exactly one on either side: `ma_TE` is the mean of the cut's two ends,
    which is the one value the two surfaces share, so they straddle it.
    """
    config, result = solved
    _, fac = measure_clark_profile(result, 0, 0.5, np.array([1.0]))

    assert np.mean(fac) == pytest.approx(1.0, abs=1e-6)


def test_the_leading_edge_is_the_origin(solved):
    """Measured from the geometric nose, where the two surfaces meet.

    Not from the stagnation point, which is where the flow attached rather
    than where the surface starts -- and which, with no incidence iterator
    holding it, moves as the very thickness being driven changes.
    """
    config, result = solved
    z, _ = measure_clark_profile(result, 0, 0.5, np.array([0.0]))

    np.testing.assert_allclose(z.ravel(), 0.0, atol=1e-12)


def test_a_section_above_a_gap_is_unmeasured(solved):
    """No surface to cut is not an error, only nothing to say."""
    config, result = solved
    assert measure_clark_profile(result, 5, 0.5, np.array([0.5])) is None


@pytest.mark.parametrize("missing", ["grid", "machine"])
def test_an_unsolved_run_is_unmeasured(solved, missing):
    config, result = solved
    stripped = dataclasses.replace(result, **{missing: None})

    assert measure_clark_profile(stripped, 0, 0.5, np.array([0.5])) is None
