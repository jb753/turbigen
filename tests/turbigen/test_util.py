"""Tests for the small helpers with no home of their own.

The cuts, which are the only part of the module with a mesh behind it rather
than arithmetic. A tip gap is the case worth meshing for: it is what tells the
blade surface apart from the k faces above it, and what the endwalls have to
be indifferent to.
"""

import numpy as np
import pytest

import ember.cut
import ember.patch
from test_blade import build
from test_mesh import MESH, TIP
from turbigen import H, bconds, guess, util


@pytest.fixture(scope="module")
def machine():
    """A two-row design whose second row has a tip gap."""
    return build(blades=TIP, mesh=MESH).design()


@pytest.fixture(scope="module")
def grid(machine):
    """Meshed and turning, the case that could break the hub-to-casing convention.

    Rotation is applied because a `RotatingPatch` leaves `Omega` at NaN until
    `bconds` values it, and the meridional guess because a cut of a wall with
    no field in it has no state to read. A grid that reaches any of this has
    been through both.
    """
    grid = H(**{k: v for k, v in MESH.items() if k != "type"}).mesh(machine)
    bconds.apply_rotation(grid, machine)
    guess.apply(grid, machine)
    return grid


def test_endwalls_are_two_per_block(grid):
    """One list per row, hub and casing for each block in it."""
    walls = util.cut_endwalls(grid)
    assert len(walls) == grid.n_row
    for row_block, row_walls in zip(grid.rows, walls):
        assert len(row_walls) == 2 * len(row_block)


def test_endwalls_span_the_passage(grid):
    """Hub below casing, and both at the extremes of the block's radii."""
    for row_block, row_walls in zip(grid.rows, util.cut_endwalls(grid)):
        for block, hub, casing in zip(row_block, row_walls[::2], row_walls[1::2]):
            np.testing.assert_allclose(hub.r, block.r[:, 0, :])
            np.testing.assert_allclose(casing.r, block.r[:, -1, :])
            assert hub.r.max() < casing.r.min()


def test_an_endwall_cut_has_an_area(grid):
    """Returned 2D, which is what `dA_quad` wants, with no squeeze in the way."""
    for cut in util.cut_endwalls(grid)[0]:
        assert cut.ndim == 2
        assert cut.dA_quad.shape == (cut.shape[0] - 1, cut.shape[1] - 1, 3)


def test_an_offset_reads_off_the_wall(grid):
    """One cell in from each surface, on the side the surface is on."""
    walls = util.cut_endwalls(grid)
    off = util.cut_endwalls(grid, offset=1)
    for row_walls, row_off in zip(walls, off):
        for i, (wall, inner) in enumerate(zip(row_walls, row_off)):
            if i % 2:
                assert (inner.r < wall.r).all()
            else:
                assert (inner.r > wall.r).all()


def _wall_cells(block):
    """Spanwise cells of one blade side that are wall, from ember's own mask.

    A ``k`` face is wall wherever no patch covers it, so a cell is blade if it
    is wall at any chordwise position: the periodics upstream and downstream of
    the blade span the whole span and mark nothing, and the tip patch marks the
    gap. Derived from the patches rather than from the trim's own arithmetic,
    which is the point of using it to check the trim.
    """
    _, _, kwall = block._get_face_wall_arrays()
    return (kwall[:, :, 0] == 0).any(axis=0)


def test_a_clearance_gap_is_trimmed_off_the_blade(grid):
    """The gap is flow, not wall, so it is not part of the blade surface."""
    shrouded, gapped = util.cut_blade_surfs(grid)

    # The shrouded row runs wall to wall, so its surface keeps the whole span.
    assert shrouded[0].shape[1] == grid[0].shape[1]
    assert gapped[0].shape[1] < grid[1].shape[1]


def test_the_trim_keeps_exactly_the_wall(grid):
    """Every cell kept is wall and every cell dropped is not.

    The bound is a node count and the mask is a cell count, so this is where an
    off-by-one would show: a blade node belongs to the surface if it touches a
    wall cell, which puts the last node one past the last wall cell -- the tip
    edge is blade, the gap above it is not.
    """
    for block, surface in zip(grid, [row[0] for row in util.cut_blade_surfs(grid)]):
        wall = _wall_cells(block)
        n_cell = surface.shape[1] - 1

        assert wall[:n_cell].all(), "a cell that is not wall was kept"
        assert not wall[n_cell:].any(), "a wall cell was trimmed away"


def test_no_blade_surface_above_a_clearance_gap(grid, machine):
    """A section over the tip finds nothing, rather than the flow passing over."""
    surface = util.cut_blade_surfs(grid)[1][0][:, :, None]
    m = np.linspace(3.0, 4.0, 41)

    def n_cut(spf):
        xr = machine.annulus.evaluate_xr(m, spf)
        return len(ember.cut.structured_meridional(surface, xr.T))

    assert n_cut(0.5) == 1
    assert n_cut(1.0) == 0


#
# WHICH FRAME A WALL CUT IS IN
#


def _with_rotating(grid, *specs):
    """The stator block of a copy of `grid`, given extra rotating patches.

    The stator is chosen because it carries none of its own, so whatever the
    test adds is the whole story on that face.
    """
    grid = grid.copy()
    block = grid[0]
    for Omega, kwargs in specs:
        patch = ember.patch.RotatingPatch(**kwargs)
        block.patches.append(patch)
        patch.set_Omega(Omega)
    return block


def test_a_wall_turns_with_its_block(grid):
    """No rotating patch on a face means it takes the block's own speed."""
    for block in grid:
        assert util._wall_Omega(block, 1, False) == float(block.Omega)
        assert util._wall_Omega(block, 2, False) == float(block.Omega)
        assert util._wall_Omega(block, 2, True) == float(block.Omega)


def test_a_casing_over_a_gap_stands_still(grid):
    """The one wall a turbigen mesh gives a speed of its own."""
    shrouded, gapped = grid

    assert util._wall_Omega(shrouded, 1, True) == float(shrouded.Omega)

    assert float(gapped.Omega) > 0.0
    assert util._wall_Omega(gapped, 1, True) == 0.0


def test_a_face_with_two_speeds_is_refused(grid):
    """Two patches disagreeing cannot be reduced to the one number a cut wants."""
    block = _with_rotating(
        grid,
        (100.0, dict(i=(0, 20), j=-1)),
        (200.0, dict(i=(20, 80), j=-1)),
    )

    with pytest.raises(ValueError, match="more than one wall speed"):
        util._wall_Omega(block, 1, True)


def test_a_partly_covered_face_is_refused(grid):
    """What a patch leaves uncovered still turns with the block, so that is two
    speeds as surely as two patches are."""
    block = _with_rotating(grid, (100.0, dict(i=(0, 20), j=-1)))

    with pytest.raises(ValueError, match="more than one wall speed"):
        util._wall_Omega(block, 1, True)


def test_a_fully_covered_face_takes_the_patch_speed(grid):
    """One patch spanning the face is the case that resolves."""
    block = _with_rotating(grid, (100.0, dict(j=-1)))

    assert util._wall_Omega(block, 1, True) == 100.0


def test_cuts_carry_the_speed_of_their_own_wall(grid):
    """Hub and blade turn with the block; the casing over a gap does not."""
    for i_row, block in enumerate(grid):
        hub, casing = util.cut_endwalls(grid)[i_row]
        assert float(hub.Omega) == float(block.Omega)
        assert float(util.cut_blade_surfs(grid)[i_row][0].Omega) == float(block.Omega)

        expected = 0.0 if block.patches.rotating else float(block.Omega)
        assert float(casing.Omega) == expected


def test_a_stationary_casing_is_measured_in_the_absolute_frame(grid):
    """`ho_rel` on a cut is the stagnation enthalpy its own boundary layer sees.

    The mechanism rather than the label: a cut carrying zero angular velocity
    must report the absolute stagnation enthalpy, which is what lets a metric
    read `ho_rel` without asking which wall it has.
    """
    _, casing = util.cut_endwalls(grid)[1]

    assert float(casing.Omega) == 0.0
    # ho_rel and ho are separate float32 reductions of what is analytically the
    # same sum for a stationary cut, so they agree to a rounding step, not to
    # assert_allclose's 1e-7 default.
    np.testing.assert_allclose(casing.ho_rel, casing.ho, rtol=1e-6)


def test_setting_a_cut_speed_does_not_re_time_the_grid(grid):
    """The cuts are copies, so writing to one cannot reach the solution.

    A slice would share its block's angular velocity, and setting the wall speed
    on one would silently turn the row it was cut from.
    """
    before = [float(block.Omega) for block in grid]

    for cuts in util.cut_endwalls(grid) + util.cut_blade_surfs(grid):
        for cut in cuts:
            cut.set_Omega(-999.0)

    assert [float(block.Omega) for block in grid] == before
