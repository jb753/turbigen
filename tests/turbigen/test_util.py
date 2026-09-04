"""Tests for the small helpers with no home of their own.

The cuts, which are the only part of the module with a mesh behind it rather
than arithmetic. A tip gap is the case worth meshing for: it is what tells the
blade surface apart from the k faces above it, and what the endwalls have to
be indifferent to.
"""

import numpy as np
import pytest

import ember.block
import ember.cut
import ember.fluid
import ember.patch
import ember.util
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


#
# THE STAGNATION POINT
#
# Arithmetic on a planted pressure peak, so that what is asserted is the search
# and the refinement rather than a solution someone has to trust.
#


def stagnation_block(ni=41, nj=5, i_peak=20.0, flat=False):
    """A 2D cut carrying a Gaussian pressure peak at `i_peak`.

    Straight and uniformly spaced, so that arc length is a linear map of index
    and a fractional index can be read back out of a `zeta` unambiguously.
    """
    shape = (ni, nj, 1)
    block = ember.block.Block(shape=shape)
    block.set_xrt(ember.util.linmesh3([0.0, 1.0], [1.0, 1.5], [0.0, 0.0], shape))
    block.set_fluid(ember.fluid.PerfectFluid(cp=1005.0, gamma=1.4, mu=1e-5, Pr=0.72))
    block.set_Omega(0.0)

    i = np.arange(ni)[:, None, None]
    if flat:
        # Rising all the way across, so there is no interior maximum to find.
        P = 1e5 + 1e3 * i * np.ones(shape)
    else:
        P = 1e5 + 1e5 * np.exp(-((i - i_peak) ** 2) / 20.0) * np.ones(shape)

    block.set_P_rho(P, np.full(shape, 1.2))
    block.set_Vx(np.full(shape, 50.0))
    block.set_Vr(np.zeros(shape))
    block.set_Vt(np.zeros(shape))

    return block.squeeze()


def test_the_stagnation_point_is_the_pressure_peak():
    """A peak planted on a node is found on that node, and reported as found."""
    i_stag, found = util.get_i_stag(stagnation_block(i_peak=20.0))
    assert (i_stag == 20).all()
    assert found.all()


def test_a_stagnation_point_between_nodes_is_refined_onto_it():
    """The parabola recovers a peak that no node sits on.

    The integer index alone is a step function of the flow, so a leading edge
    that moves by less than a cell would show no change in incidence at all.
    """
    block = stagnation_block(i_peak=20.4)
    i_stag, found = util.get_i_stag(block)
    assert found.all()

    zeta = util.get_zeta(block)
    zeta_stag = util.get_zeta_stag(block, i_stag)

    # Back to a fractional index, which the uniform spacing makes linear.
    lower, upper = zeta[20, :], zeta[21, :]
    i_recovered = 20.0 + (zeta_stag - lower) / (upper - lower)

    assert (lower < zeta_stag).all() and (zeta_stag < upper).all()
    np.testing.assert_allclose(i_recovered, 20.4, atol=1e-2)


def test_the_refinement_stands_on_the_node_it_is_given():
    """With no maximum, `get_zeta_stag` falls back rather than extrapolating."""
    block = stagnation_block(flat=True)
    i_stag, _ = util.get_i_stag(block)
    zeta = util.get_zeta(block)
    np.testing.assert_allclose(
        util.get_zeta_stag(block, i_stag), zeta[i_stag, np.arange(block.shape[1])]
    )


def test_a_stagnation_point_that_was_not_found_says_so():
    """A monotonic surface still returns an index, flagged as a guess.

    The incidence iterator drops a section it could not measure, and would
    otherwise step on whichever end of the blade happened to be at the higher
    pressure.
    """
    i_stag, found = util.get_i_stag(stagnation_block(flat=True))
    assert not found.any()
    assert (i_stag >= 0).all()


#
# THE SURFACE NORMAL
#
# A circular arc standing for a leading edge, because the answer is then known
# in closed form: the normal at a point swept an angle from the nose is that
# angle away from the camber direction, which is what an incidence is.
#


def nose_arc(a_deg, R=0.05, r_hub=1.0, nj=3, reverse=False):
    """An arc of radius `R` about the origin, as a cut of a leading edge.

    At constant radius, so that `(m, r * theta)` is `(x, r * theta)` and the
    geometry can be written down. `a = 180 deg` is the nose, with the blade
    interior towards positive x.
    """
    a = np.radians(np.asarray(a_deg, dtype=float))
    if reverse:
        a = a[::-1]

    shape = (len(a), nj, 1)
    x = (R * np.cos(a))[:, None, None] * np.ones(shape)
    r = np.full(shape, r_hub)
    t = (R * np.sin(a) / r_hub)[:, None, None] * np.ones(shape)

    block = ember.block.Block(shape=shape)
    block.set_xrt(np.stack((x, r, t), axis=-1))

    return block.squeeze()


ARC = np.linspace(90.0, 270.0, 361)
"""Half a turn about the nose, at half a degree a node."""

MERIDIONAL = np.array([1.0, 0.0])
"""Downstream, for an arc drawn at constant radius."""


@pytest.mark.parametrize("swept", [-40.0, 0.0, 40.0])
def test_the_normal_yaw_is_the_angle_swept_from_the_nose(swept):
    """What the reference subtends at the centre of the leading edge circle.

    Recovered without the circle's radius anywhere in the arithmetic, which is
    what lets this work for a thickness distribution that does not define one.
    """
    cut = nose_arc(ARC)
    node = int(np.argmin(np.abs(ARC - (180.0 + swept))))

    yaw = util.surface_normal_yaw(
        cut, util.get_zeta(cut)[node, :], MERIDIONAL, 0.0
    )

    # Differenced centrally on a finely drawn arc, so it is recovered to a
    # hundred-thousandth of a degree. Nothing here needs it that tight.
    np.testing.assert_allclose(yaw, swept, atol=1e-3)


def test_the_normal_yaw_does_not_depend_on_which_way_the_cut_runs():
    """`cut_blade_sides` joins the two k faces in an order the H-mesh and
    O-mesh branches do not share, so the direction of travel round a nose is a
    property of the mesh. The answer must not be."""
    node = int(np.argmin(np.abs(ARC - 220.0)))

    forward = nose_arc(ARC)
    backward = nose_arc(ARC, reverse=True)

    yaw_forward = util.surface_normal_yaw(
        forward, util.get_zeta(forward)[node, :], MERIDIONAL, 0.0
    )
    # The same point of the arc, which the reversal moved to the other end.
    yaw_backward = util.surface_normal_yaw(
        backward, util.get_zeta(backward)[len(ARC) - 1 - node, :], MERIDIONAL, 0.0
    )

    np.testing.assert_allclose(yaw_forward, yaw_backward, atol=1e-3)


def test_the_metal_angle_only_chooses_between_the_two_normals():
    """A surface has two normals and `chi` says which is inward. It must not
    otherwise enter the answer, which is measured, not assumed."""
    cut = nose_arc(ARC)
    zeta = util.get_zeta(cut)[int(np.argmin(np.abs(ARC - 200.0))), :]

    yaw = [util.surface_normal_yaw(cut, zeta, MERIDIONAL, chi) for chi in (-30.0, 30.0)]

    np.testing.assert_allclose(yaw[0], yaw[1], atol=1e-9)


def test_the_normal_yaw_moves_smoothly_between_nodes():
    """Read at a sub-cell arc length, as `get_zeta_stag` returns one.

    The components are interpolated rather than the angle they make, so a
    stagnation point crossing a cell boundary moves the answer by a cell's
    worth of angle and not by a step.
    """
    cut = nose_arc(ARC)
    zeta_line = util.get_zeta(cut)[:, 0]

    node = int(np.argmin(np.abs(ARC - 200.0)))
    fractions = np.linspace(0.0, 1.0, 11)
    zeta = zeta_line[node] + fractions * (zeta_line[node + 1] - zeta_line[node])

    yaw = np.array(
        [
            util.surface_normal_yaw(cut, np.full(3, z), MERIDIONAL, 0.0)[0]
            for z in zeta
        ]
    )

    # A cell is one node spacing of sweep, and it is traversed evenly.
    step = ARC[1] - ARC[0]
    np.testing.assert_allclose(yaw, yaw[0] + fractions * step, atol=1e-3)
