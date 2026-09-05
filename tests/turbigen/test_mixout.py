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
- test_actual_has_the_shape_of_the_nominal: assembled per station
- test_actual_keeps_the_design_annulus_area: what the AR contraction is for
- test_actual_reports_the_speed_the_grid_ran_at: and Omega, which does not
- test_actual_is_a_plausible_flow: finite, positive, roughly the design
- test_actual_is_not_reinterpreted_by_the_datum: the P/T/V transfer
- test_mixing_loss_has_the_shape_of_the_mean_line: one value per station
- test_mixing_loss_is_a_real_entropy_rise: positive through the wake
- test_mixing_loss_builds_through_the_row: exit lossier than inlet
- test_a_cut_that_misses_the_grid_is_reported: names the station
"""

import numpy as np
import pytest

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
# WHAT COMES BACK
#


def test_actual_has_the_shape_of_the_nominal(solved):
    machine, grid = solved

    actual, _ = mixout.mean_line(grid, machine)

    assert actual.shape == machine.mean_line.shape
    assert actual.n_row == machine.mean_line.n_row


def test_actual_keeps_the_design_annulus_area(solved):
    """Which is what the AR contraction in the mix-out is for.

    Each cut is offset into a gap, where the annulus area differs from the
    station's, so the mixed state is contracted back to the design area. That
    makes the actual mean line comparable with the nominal station by station.
    """
    machine, grid = solved

    actual, _ = mixout.mean_line(grid, machine)

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

    actual, _ = mixout.mean_line(grid, machine)

    assert np.all(np.asarray(actual.flat.Omega, dtype=float) == pytest.approx(100.0))
    # And the design it came from is untouched, this being a cascade at rest.
    assert np.all(np.asarray(machine.mean_line.flat.Omega, dtype=float) == 0.0)

    for block in grid:
        block.set_Omega(0.0)


def test_actual_is_a_plausible_flow(solved):
    machine, grid = solved

    actual, _ = mixout.mean_line(grid, machine)

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
    """State crosses from the grid as P, T and velocity, not as conserved.

    A design and the grid meshed from it share one datum, so today the transfer
    would survive being written the wrong way. That is a property of the
    pipeline rather than of `mean_line`, and it can be taken away by any change
    upstream --- so the grid is put on a deliberately different datum here and
    the mixed-out temperature has to come across unchanged anyway. Conserved
    energy is measured from a datum and would arrive displaced by the gap
    between the two; pressure, temperature and velocity are not and do not.
    """
    machine, grid = solved

    grid = grid.copy()
    fluid = grid[0].fluid
    grid.set_fluid(
        fluid.change_datum(P_dtm=0.5 * fluid.P_dtm, T_dtm=0.5 * fluid.T_dtm)
    )
    assert grid[0].fluid.T_dtm != machine.mean_line.fluid.T_dtm

    actual, _ = mixout.mean_line(grid, machine)

    T = np.asarray(actual.flat.T, dtype=float)
    T_nominal = np.asarray(machine.mean_line.flat.T, dtype=float)
    assert np.all(np.abs(T - T_nominal) < 0.25 * T_nominal)


#
# THE MIXING LOSS
#


def test_mixing_loss_has_the_shape_of_the_mean_line(solved):
    """One entropy rise per station, laid out like every other mean-line field."""
    machine, grid = solved

    actual, Ds_mix = mixout.mean_line(grid, machine)

    assert Ds_mix.shape == actual.shape
    assert np.all(np.isfinite(Ds_mix))


def test_mixing_loss_is_a_real_entropy_rise(solved):
    """Mixing a non-uniform wake to uniformity generates entropy.

    The trailing-edge cuts carry a wake, so their loss is unambiguously
    positive. The inlet cuts sit in a nearly uniform inflow whose fluxes a
    ten-step transient has not made conservative, so the mixing inequality can
    be off by a small amount there --- bounded well below the wake's rise.
    """
    machine, grid = solved

    _, Ds_mix = mixout.mean_line(grid, machine)

    assert np.all(Ds_mix[1] > 0.0)
    assert np.all(np.abs(Ds_mix[0]) < Ds_mix[1])


def test_mixing_loss_builds_through_the_row(solved):
    """The trailing-edge cut carries a wake; the inlet cut is nearly uniform.

    So the outlet station's mixing loss must exceed the inlet's, even on a
    ten-step transient -- the wake is a geometric feature, not a converged one.
    """
    machine, grid = solved

    _, Ds_mix = mixout.mean_line(grid, machine)

    assert np.all(Ds_mix[1] > Ds_mix[0])


def test_a_cut_that_misses_the_grid_is_reported(solved):
    """Which station failed, since a bare `None` from ember says nothing."""
    machine, grid = solved

    with pytest.raises(ValueError, match="station 0"):
        # An offset of many chords puts the first cut far upstream of the inlet.
        mixout.mean_line(grid, machine, offset=50.0)
