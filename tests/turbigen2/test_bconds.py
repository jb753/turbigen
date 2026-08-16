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
- test_a_placed_patch_is_unvalued_until_bconds_runs: nan, not a silent zero
- test_the_row_count_must_match: a mean line and a grid that disagree
- test_no_operating_point_is_the_design_point: what an absent section means
- test_the_pressure_change_scales_linearly: what DP_adjust exists for
- test_a_machine_that_raises_pressure_gets_more_rise: the sign convention
- test_a_turbine_and_a_compressor_move_oppositely: in pressure, not in duty
- test_an_adjustment_past_vacuum_is_refused: an exit pressure that is not one
- test_the_operating_point_reaches_the_outlet_patches: and nothing else moves
- test_the_operating_point_round_trips: an ordinary config node
- test_the_operating_point_is_not_a_design_variable: outside database.SUBTREE
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


#
# THE OPERATING POINT
#
# A design states one condition; a machine has a whole characteristic. Reaching
# the rest of it changes the boundary conditions and nothing else, which is why
# this lives here and not in any design stage.
#


class Stub:
    """The two numbers `exit_pressure` reads, and nothing else.

    Lets the compressor case be tested without a compressor design: only the
    *sign* of the design pressure change distinguishes one, and that is exactly
    what the formula turns on.
    """

    def __init__(self, Po_in, P_out):
        end = lambda value: type("S", (), {"Po": value, "P": value})()  # noqa: E731
        self.mean_line = type(
            "M", (), {"inlet": end(Po_in), "outlet": end(P_out)}
        )()


def test_no_operating_point_is_the_design_point(shrouded):
    machine, _ = shrouded

    assert bconds.exit_pressure(machine, None) == pytest.approx(
        float(machine.mean_line.outlet.P)
    )
    assert bconds.exit_pressure(
        machine, bconds.OperatingPoint()
    ) == pytest.approx(float(machine.mean_line.outlet.P))


@pytest.mark.parametrize("adjust", [0.05, -0.1, 0.5])
def test_the_pressure_change_scales_linearly(shrouded, adjust):
    """The property the field exists for, and the one a pressure ratio does not
    have: the same number means the same fractional change at any Mach number.
    """
    machine, _ = shrouded
    Po_in = float(machine.mean_line.inlet.Po)
    DP_design = Po_in - float(machine.mean_line.outlet.P)

    P_out = bconds.exit_pressure(
        machine, bconds.OperatingPoint(DP_adjust=adjust)
    )

    assert (Po_in - P_out) / DP_design == pytest.approx(1.0 + adjust)


def test_a_machine_that_raises_pressure_gets_more_rise():
    """The sign convention, which is the subtle half.

    A compressor's design pressure change is negative, so scaling it makes it
    more negative and the exit pressure higher. One formula, both machine
    types, and no machine type named anywhere.
    """
    compressor = Stub(Po_in=1.0e5, P_out=1.3e5)

    raised = bconds.exit_pressure(
        compressor, bconds.OperatingPoint(DP_adjust=0.5)
    )

    assert raised > 1.3e5
    assert (1.0e5 - raised) / (1.0e5 - 1.3e5) == pytest.approx(1.5)


def test_a_turbine_and_a_compressor_move_oppositely():
    """Positive always means *more* pressure change, which is opposite
    directions in exit pressure and the same direction in duty."""
    turbine = bconds.exit_pressure(Stub(1.0e5, 0.7e5), bconds.OperatingPoint(DP_adjust=0.1))
    compressor = bconds.exit_pressure(Stub(1.0e5, 1.3e5), bconds.OperatingPoint(DP_adjust=0.1))

    assert turbine < 0.7e5
    assert compressor > 1.3e5


def test_an_adjustment_past_vacuum_is_refused():
    with pytest.raises(ValueError, match="not a pressure"):
        bconds.exit_pressure(Stub(1.0e5, 0.7e5), bconds.OperatingPoint(DP_adjust=3.0))


def test_the_operating_point_reaches_the_outlet_patches(shrouded):
    machine, grid = shrouded
    point = bconds.OperatingPoint(DP_adjust=0.1)

    bconds.apply(grid, machine, point)

    expected = bconds.exit_pressure(machine, point)
    for patch in grid.patches.outlet:
        assert patch.P == pytest.approx(expected, rel=1e-6)

    # And nothing else moved: the inlet is the design's whatever the exit does.
    for patch in grid.patches.inlet:
        assert patch.Po == pytest.approx(float(machine.mean_line.inlet.Po), rel=1e-6)


def test_the_operating_point_round_trips():
    from turbigen2 import Config  # noqa: PLC0415

    config = build(mesh=MESH)
    config = dataclasses.replace(
        config, operating_point=bconds.OperatingPoint(DP_adjust=0.05)
    )

    assert Config.from_dict(config.to_dict()) == config
    assert config.to_dict()["operating_point"]["DP_adjust"] == 0.05


def test_the_operating_point_is_not_a_design_variable():
    """Outside `database.SUBTREE`, so two runs of one machine at different back
    pressures are one design run twice rather than two designs."""
    from turbigen2 import database  # noqa: PLC0415

    assert "operating_point" not in database.SUBTREE
