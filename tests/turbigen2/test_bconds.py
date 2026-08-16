"""Tests for putting the design's operating point onto a meshed grid.

No solve anywhere: `bconds.apply` is a pure function of a grid and the machine
it was meshed from, so every assertion here reads a patch or a block back.

Rotation gets most of the attention because it is the half that fails quietly.
A wall takes its block's angular velocity unless a `RotatingPatch` overrides
it, so a rotor with no patches is a *shrouded* rotor and one whose casing is
meant to stand still needs that saying. Get it backwards and the wall boundary
condition is wrong while every number on the page looks reasonable, which is
why the ember semantics this rests on are pinned here directly rather than
assumed.

Test cases:
- test_inlet_gets_the_design_stagnation_state: Po, To and both angles
- test_outlet_gets_the_design_static_pressure: the other end
- test_a_grid_without_an_inlet_is_refused: and one without an outlet
- test_each_row_turns_at_its_design_speed: the stator stationary, the rotor not
- test_a_shrouded_rotor_turns_every_wall: no patch means the block speed
- test_a_tip_gap_leaves_the_casing_standing_still: five walls turn, not six
- test_a_stator_turns_nothing: including its casing patch, if it has one
- test_speed_can_be_changed_without_re-meshing: what bconds is for
- test_an_unvalued_rotating_patch_is_refused: nan reaches the solver otherwise
- test_the_row_count_must_match: a mean line and a grid that disagree
"""

import dataclasses

import numpy as np
import pytest

from test_blade import blade, build
from turbigen2 import bconds

MESH = {"type": "h", "dm_TE": 0.05, "resolution_factor": 0.5, "dspf_mid": 0.1}
"""Coarse, because none of this depends on the mesh being fine."""

TIP = [blade(), blade(dchi_LE=2.0, tip_span=0.02)]
"""A two-row machine whose rotor has tip clearance."""


def meshed(blades=None):
    """Return the machine and grid for a two-row turbine, meshed but bare."""
    config = build(blades=blades, mesh=MESH)
    machine = config.design()
    return machine, config.mesh.mesh(machine)


@pytest.fixture(scope="module")
def shrouded():
    return meshed()


@pytest.fixture(scope="module")
def gapped():
    return meshed(blades=TIP)


def wall_speeds(block):
    """Return the wall angular velocity on each face, non-dimensional."""
    return {
        key.replace("omega_wall", "").replace("_nd", ""): float(np.max(value))
        for key, value in block.Omega_wall_nd.items()
    }


#
# INLET AND OUTLET
#


def test_inlet_gets_the_design_stagnation_state(shrouded):
    machine, grid = shrouded
    bconds.apply(grid, machine)

    inlet = machine.mean_line.inlet
    for patch in grid.patches.inlet:
        assert patch.Po == pytest.approx(float(inlet.Po), rel=1e-6)
        assert patch.To == pytest.approx(float(inlet.To), rel=1e-6)
        assert patch.Alpha == pytest.approx(float(inlet.Alpha), abs=1e-4)


def test_outlet_gets_the_design_static_pressure(shrouded):
    machine, grid = shrouded
    bconds.apply(grid, machine)

    for patch in grid.patches.outlet:
        assert patch.P == pytest.approx(float(machine.mean_line.outlet.P), rel=1e-6)


def test_a_grid_without_an_inlet_is_refused(shrouded):
    machine, grid = shrouded

    class Bare:
        patches = type("P", (), {"inlet": (), "outlet": ()})()

    with pytest.raises(ValueError, match="at least one of each"):
        bconds.apply(Bare(), machine)


#
# ROTATION
#


def test_each_row_turns_at_its_design_speed(shrouded):
    machine, grid = shrouded
    bconds.apply(grid, machine)

    Omega = np.asarray(machine.mean_line.Omega, dtype=float)[0]
    for blocks, Omega_row in zip(grid.rows, Omega):
        for block in blocks:
            assert float(block.Omega) == pytest.approx(Omega_row, rel=1e-6)

    # The stator stands still and the rotor does not, or nothing above is
    # testing anything.
    assert float(grid[0].Omega) == 0.0
    assert float(grid[1].Omega) > 1.0


def test_a_shrouded_rotor_turns_every_wall(shrouded):
    """The ember semantics this design rests on, asserted rather than assumed.

    Every face defaults to the block's own angular velocity, so a rotor needs
    no rotating patch at all to be shrouded. If that ever stopped being true,
    every rotor here would silently acquire a stationary casing.
    """
    machine, grid = shrouded
    bconds.apply(grid, machine)

    assert not grid.patches.rotating

    rotor = grid[1]
    speeds = wall_speeds(rotor)
    assert list(speeds.values()) == pytest.approx([float(rotor.Omega_nd)] * 6)


def test_a_tip_gap_leaves_the_casing_standing_still(gapped):
    """Five walls turn with the blade; the casing over the gap does not."""
    machine, grid = gapped
    bconds.apply(grid, machine)

    rotor = grid[1]
    speeds = wall_speeds(rotor)

    assert speeds["nj"] == pytest.approx(0.0)
    turning = [value for face, value in speeds.items() if face != "nj"]
    assert turning == pytest.approx([float(rotor.Omega_nd)] * 5)


def test_a_stator_turns_nothing(gapped):
    machine, grid = gapped
    bconds.apply(grid, machine)

    assert list(wall_speeds(grid[0]).values()) == pytest.approx([0.0] * 6)


def test_speed_can_be_changed_without_re_meshing(gapped):
    """What putting this in bconds rather than the mesher buys.

    A speedline varies one number over a fixed grid; if rotation were applied
    while meshing it would cost a mesh per point.
    """
    machine, grid = gapped

    slower = dataclasses.replace(
        machine, mean_line=machine.mean_line.copy()
    )
    Omega = np.asarray(machine.mean_line.Omega, dtype=float)[0]
    slower.mean_line.set_Omega_row(0.5 * Omega)

    bconds.apply(grid, slower)

    assert float(grid[1].Omega) == pytest.approx(0.5 * Omega[1], rel=1e-5)
    # The casing is still stationary, not scaled with the shaft.
    assert wall_speeds(grid[1])["nj"] == pytest.approx(0.0)


def test_a_placed_patch_is_unvalued_until_bconds_runs():
    """Placement and value are separate steps, and the gap between them is
    loud.

    `RotatingPatch` defaults to `Omega = nan` rather than zero, so a grid that
    skipped this module would turn the solution to NaN rather than quietly run
    a rotor as though it were stationary -- which is exactly the failure the
    package had while nothing set rotation at all. There is no check for it in
    `apply_rotation` because there is no way to miss a patch: `grid.rows` puts
    every block in a row, so the loop reaches all of them.
    """
    machine, grid = meshed(blades=TIP)

    casing = grid.patches.rotating
    assert len(casing) == 1
    assert np.isnan(casing[0].Omega)

    bconds.apply(grid, machine)

    assert np.isfinite(grid.patches.rotating[0].Omega)


def test_the_row_count_must_match(shrouded):
    machine, grid = shrouded

    one_row = machine.mean_line.copy()[:, :1]

    with pytest.raises(ValueError, match="one speed per row"):
        bconds.apply_rotation(grid, dataclasses.replace(machine, mean_line=one_row))
