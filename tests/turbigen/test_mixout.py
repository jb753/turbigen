"""Tests for reducing a solved grid back to a mean line.

The mixing itself is ember's, and it verifies its own conservation: the Newton
solve balances mass, both momentum components, angular momentum and energy to a
tolerance, and raises if it cannot. So what is worth testing here is the part
that is ours -- where the cuts are taken, how the mixed states are assembled
into a mean line, and that the equation of state is not lost on the way.

The solution these run on is a short march, so it is a transient rather than a
converged answer. That is enough: none of this depends on the flow being
settled, only on it being a valid field.

Test cases:
- test_cut_planes_land_in_the_gaps: not inside a row, where there is no plane
- test_cut_planes_span_hub_to_casing: two points, the shape ember wants
- test_actual_has_the_shape_of_the_nominal: assembled per station
- test_actual_keeps_the_design_annulus_area: what the AR contraction is for
- test_actual_reports_the_speed_the_grid_ran_at: and Omega, which does not
- test_actual_is_a_plausible_flow: finite, positive, roughly the design
- test_actual_is_not_reinterpreted_by_the_datum: the P/T/V transfer
- test_a_cut_that_misses_the_grid_is_reported: names the station
"""

import numpy as np
import pytest

from test_blade import build
from turbigen import cli, mixout
from turbigen.config import Config

CASCADE = {
    "fluid": {"type": "perfect", "cp": 1005.0, "gamma": 1.4, "mu": 1.8e-5},
    "mean_line": {
        "type": "turbine_cascade",
        "span": [0.01, 0.011],
        "Alpha": [40.0, -65.0],
        "Ma2": 0.6,
        "Ys": 0.029,
        "htr": 0.99,
    },
    "annulus": {
        "type": "fixed_axial_chord",
        "cx_row": [0.00525],
        "cx_gap": [0.0105, 0.0105],
    },
    "blades": [
        {
            "count": {"type": "Co", "Co": 0.7},
            "sections": [
                {
                    "spf": 0.5,
                    "dchi_LE": 10.0,
                    "dchi_TE": -2.0,
                    "camber": {"type": "quadratic"},
                    "thickness": {
                        "type": "taylor",
                        "R_LE": 0.05,
                        "t_max": 0.12,
                        "m_tmax": 0.3,
                        "t_TE": 0.03,
                        "tanwedge": 0.18,
                    },
                }
            ],
        }
    ],
    "mesh": {
        "type": "h",
        "resolution_factor": 0.25,
        "dm_TE": 0.0,
        "AR_cusp": 2.0,
        "ni_cusp": 5,
    },
    "solver": {"type": "ember", "n_step": 10, "n_step_log": 10},
}
"""The single-row cascade from examples/turbine_cascade.yaml.

One row, so no mixing planes, and it holds together for long enough to give a
valid field -- unlike the two-row case elsewhere in these tests, which diverges
within a handful of steps.
"""


@pytest.fixture(scope="module")
def solved():
    """A machine and a short march on it, enough to have a valid field."""
    config = Config.from_dict(CASCADE)
    _, machine, grid = cli.prepare(config)
    config.solver.solve(grid)
    return machine, grid


#
# WHERE THE CUTS GO
#


def test_cut_planes_land_in_the_gaps():
    """Cutting at a station exactly would put the plane inside the blade.

    Rows occupy the odd segments of the annulus coordinate, so a leading-edge
    cut has to sit just below an odd integer and a trailing-edge cut just above
    the next one.
    """
    annulus = build().design().annulus
    n_row = annulus.n_row

    # Recover the m of each cut from its axial position, via the mid-span line.
    m_dense = np.linspace(0.0, annulus.mmax, 20001)
    x_dense = annulus.evaluate_xr(m_dense, 0.5)[0]

    for i_station, xr in enumerate(mixout.cut_planes(annulus)):
        x_cut = float(xr.mean(axis=0)[0])
        m_cut = float(np.interp(x_cut, x_dense, m_dense))

        station = 1.0 + i_station
        if i_station % 2 == 0:
            assert m_cut < station, "a leading edge cut must sit upstream"
        else:
            assert m_cut > station, "a trailing edge cut must sit downstream"
        assert abs(m_cut - station) < 0.5, "and still in the adjacent gap"

    assert len(mixout.cut_planes(annulus)) == 2 * n_row


def test_cut_planes_span_hub_to_casing():
    """Two (x, r) points, which is the curve ember.cut.unstructured takes."""
    annulus = build().design().annulus

    for xr in mixout.cut_planes(annulus):
        assert xr.shape == (2, 2)
        r_hub, r_cas = xr[0][1], xr[1][1]
        assert r_cas > r_hub


#
# WHAT COMES BACK
#


def test_actual_has_the_shape_of_the_nominal(solved):
    machine, grid = solved

    actual = mixout.mean_line(grid, machine)

    assert actual.shape == machine.mean_line.shape
    assert actual.n_row == machine.mean_line.n_row


def test_actual_keeps_the_design_annulus_area(solved):
    """Which is what the AR contraction in the mix-out is for.

    Each cut is offset into a gap, where the annulus area differs from the
    station's, so the mixed state is contracted back to the design area. That
    makes the actual mean line comparable with the nominal station by station.
    """
    machine, grid = solved

    actual = mixout.mean_line(grid, machine)

    np.testing.assert_allclose(
        np.asarray(actual.flat.Am, dtype=float),
        np.asarray(machine.mean_line.flat.Am, dtype=float),
        rtol=1e-12,
    )


def test_actual_reports_the_speed_the_grid_ran_at(solved):
    """Shaft speed comes from the grid, not from the design it was meshed from.

    It is the one quantity here that a cut genuinely cannot measure but the
    blocks were told, and what they were told is what the solver used. Copied
    from the design instead, an operating point that changed the speed would
    archive the speed it did *not* run at -- and `Ma_rel`, `Alpha_rel` and
    `V_rel` are all derived from it, so the whole relative frame would be wrong
    while every number stayed plausible.
    """
    machine, grid = solved

    # Stand in for an operating point: spin the grid after it was meshed.
    for block in grid:
        block.set_Omega(100.0)

    actual = mixout.mean_line(grid, machine)

    assert np.all(np.asarray(actual.flat.Omega, dtype=float) == pytest.approx(100.0))
    # And the design it came from is untouched, this being a cascade at rest.
    assert np.all(np.asarray(machine.mean_line.flat.Omega, dtype=float) == 0.0)

    for block in grid:
        block.set_Omega(0.0)


def test_actual_is_a_plausible_flow(solved):
    machine, grid = solved

    actual = mixout.mean_line(grid, machine)

    assert np.all(np.isfinite(np.asarray(actual.flat.P)))
    assert np.all(np.asarray(actual.flat.P) > 0.0)
    assert np.all(np.asarray(actual.flat.T) > 0.0)

    # Loosely in the region of the design; this is ten steps of a transient,
    # so the assertion is that it is the same machine, not that it is right.
    nominal = machine.mean_line.flat
    np.testing.assert_allclose(
        np.asarray(actual.flat.P, dtype=float),
        np.asarray(nominal.P, dtype=float),
        rtol=0.5,
    )


def test_actual_is_not_reinterpreted_by_the_datum(solved):
    """The cut carries the grid's fluid, the mean line carries the design's.

    Those have different datums, so transferring the conserved variables would
    silently shift the temperature. State is moved as P, T and velocity, which
    crosses unchanged -- so the actual temperature must be a real temperature,
    not one displaced by the gap between the two datums.
    """
    machine, grid = solved
    assert grid[0].fluid.T_dtm != machine.mean_line.fluid.T_dtm

    actual = mixout.mean_line(grid, machine)

    T = np.asarray(actual.flat.T, dtype=float)
    T_nominal = np.asarray(machine.mean_line.flat.T, dtype=float)
    assert np.all(np.abs(T - T_nominal) < 0.25 * T_nominal)


def test_a_cut_that_misses_the_grid_is_reported(solved):
    """Which station failed, since a bare `None` from ember says nothing."""
    machine, grid = solved

    with pytest.raises(ValueError, match="station 0"):
        # An offset of many chords puts the first cut far upstream of the inlet.
        mixout.mean_line(grid, machine, offset=50.0)
