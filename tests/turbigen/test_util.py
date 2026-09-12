"""Tests for the small helpers with no home of their own.

The cuts, which are the only part of the module with a mesh behind it rather
than arithmetic. A tip gap is the case worth meshing for: it is what tells the
blade surface apart from the k faces above it, and what the endwalls have to
be indifferent to.
"""

import ember.block
import ember.cut
import ember.fluid
import ember.patch
import ember.util
import numpy as np
import pytest
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


def test_a_spanwise_cut_is_the_one_ember_would_have_made(grid, machine):
    """Cropping to the band is exact, not an approximation.

    The cut surface lies inside the band by construction, so the nodes left
    out could only ever have been one sign of distance from it --- which is
    what the uncropped call is asked to confirm, node for node, rather than to
    within a tolerance.
    """
    import ember.cut

    annulus = machine.annulus
    m = np.linspace(0.0, annulus.m_max, annulus.n_segment * 25 + 1)

    for spf in (0.1, 0.5, 0.9):
        xr_cut = annulus.evaluate_xr(m, spf).T

        cropped = util.cut_spanwise(grid, xr_cut)
        whole = ember.cut.structured_meridional(grid, xr_cut)

        assert len(cropped) == len(whole) > 0
        for a, b in zip(cropped, whole):
            assert np.array_equal(np.asarray(a.x), np.asarray(b.x))
            assert np.array_equal(np.asarray(a.r), np.asarray(b.r))
            np.testing.assert_allclose(np.asarray(a.P), np.asarray(b.P))


def test_a_spanwise_cut_that_misses_everything_is_empty(grid):
    """A curve nowhere near the machine cuts nothing, rather than raising.

    The scan finds nothing to bracket and hands the block on whole, which is
    what leaves the decision about what intersects where it always was.
    """
    far = np.stack((np.linspace(0.0, 1.0, 51), np.full(51, 99.0)), axis=1)

    assert len(util.cut_spanwise(grid, far)) == 0


def test_a_cut_landing_on_a_gridline_is_still_bracketed(grid, machine):
    """The case a strict sign change misses, and it is the common one.

    A cut that passes exactly through a node leaves a distance of exactly
    zero there, whose sign multiplies to zero against either neighbour and
    never to less than it. A mesh with an odd number of spanwise nodes puts a
    gridline on mid-span, so this is what asking for `spf=0.5` usually does.
    """
    block = grid[0]
    nj = block.shape[1]

    # The span curve through a node of this block, rather than a span
    # fraction that might happen to fall between two.
    xr_node = np.asarray(block.xrt[:, nj // 2, 0, :2])

    band = util.cut_band(block, xr_node)

    assert band.start < nj // 2 < band.stop
    assert band.stop - band.start < nj


def test_a_cut_band_is_padded_around_what_the_scan_found(grid, machine):
    """A coarse scan can only miss by the gridlines it stepped over.

    So the band is widened, and the test that it was is that the cut still
    lands strictly inside it --- with room to spare at both ends rather than
    on the boundary, where a stride that missed the crossing would put it.
    """
    annulus = machine.annulus
    xr_cut = annulus.evaluate_xr(
        np.linspace(0.0, annulus.m_max, annulus.n_segment * 25 + 1), 0.5
    ).T

    band = util.cut_band(grid[0], xr_cut)

    assert band.stop - band.start > 2 * util.BAND_PAD
    assert band.start >= 0 and band.stop <= grid[0].shape[1]
    assert band.stop - band.start < grid[0].shape[1]


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
        (100.0, {"i": (0, 20), "j": -1}),
        (200.0, {"i": (20, 80), "j": -1}),
    )

    with pytest.raises(ValueError, match="more than one wall speed"):
        util._wall_Omega(block, 1, True)


def test_a_partly_covered_face_is_refused(grid):
    """What a patch leaves uncovered still turns with the block, so that is two
    speeds as surely as two patches are."""
    block = _with_rotating(grid, (100.0, {"i": (0, 20), "j": -1}))

    with pytest.raises(ValueError, match="more than one wall speed"):
        util._wall_Omega(block, 1, True)


def test_a_fully_covered_face_takes_the_patch_speed(grid):
    """One patch spanning the face is the case that resolves."""
    block = _with_rotating(grid, (100.0, {"j": -1}))

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


def stretched_stagnation_block(ni=11, nj=5, ratio=1.2, i_peak=5, offset=0.4):
    """A cut whose spacing grows geometrically, carrying an exact parabola.

    The pressure is a parabola *in arc length*, so the three-point fit is exact
    and its vertex is recoverable to round-off however the nodes are spaced ---
    which is what tells a refinement that respects the spacing from one that
    assumes it uniform.

    Returns the block, the index the peak sits in, and the arc length it was
    planted at.
    """
    shape = (ni, nj, 1)
    block = ember.block.Block(shape=shape)

    # Geometric spacing at the expansion ratio a mesh actually carries, so no
    # two cells about the peak are the same size.
    ds = ratio ** np.arange(ni - 1)
    x = np.concatenate(([0.0], np.cumsum(ds)))
    x = x / x[-1]

    xrt = np.zeros(shape + (3,))
    xrt[..., 0] = x[:, None, None]
    xrt[..., 1] = 1.0
    block.set_xrt(xrt)
    block.set_fluid(ember.fluid.PerfectFluid(cp=1005.0, gamma=1.4, mu=1e-5, Pr=0.72))
    block.set_Omega(0.0)

    # Between `i_peak` and its neighbour, in the units of that cell.
    x_peak = x[i_peak] + offset * (x[i_peak + 1] - x[i_peak])

    # At constant radius and zero angle, arc length is the axial coordinate, so
    # a parabola in one is a parabola in the other. Shallow enough to stay
    # positive across a domain of unit length.
    P = 2e5 - 1e5 * (xrt[..., 0] - x_peak) ** 2

    block.set_P_rho(P, np.full(shape, 1.2))
    block.set_Vx(np.full(shape, 50.0))
    block.set_Vr(np.zeros(shape))
    block.set_Vt(np.zeros(shape))

    return block.squeeze(), i_peak, x_peak


def test_the_refinement_uses_the_spacing_it_is_given():
    """A peak on a stretched mesh is found where it is, not a fraction of a
    cell off.

    The vertex of a parabola through three unevenly spaced points is not the
    midpoint slope over twice the curvature: that form is out by
    ``(d01 - d12) / 4``, which vanishes only on a uniform mesh. What it leaves
    is a bias that moves as the stagnation point crosses cells, on the one
    quantity that exists to slide smoothly between them.
    """
    block, i_peak, x_peak = stretched_stagnation_block()
    nj = block.shape[1]

    zeta = util.get_zeta(block)
    zeta_stag = util.get_zeta_stag(block, np.full((nj,), i_peak))

    # Arc length is the axial coordinate here, zeroed at i = 0.
    zeta_peak = x_peak - float(block.x[0, 0])

    # Measured as a fraction of the cell the peak sits in, so the tolerance
    # means the same thing wherever on the stretched mesh it landed.
    cell = zeta[i_peak + 1, 0] - zeta[i_peak, 0]
    np.testing.assert_allclose(zeta_stag, zeta_peak, atol=1e-3 * cell)

    # Asserted as absent rather than merely small: on this mesh the bias the
    # uniform form carries is a measurable fraction of a cell.
    bias = 0.25 * (
        (zeta[i_peak, 0] - zeta[i_peak - 1, 0])
        - (zeta[i_peak + 1, 0] - zeta[i_peak, 0])
    )
    assert abs(bias) > 1e-2 * cell


def test_the_refinement_stands_on_the_node_it_is_given():
    """With no maximum, `get_zeta_stag` falls back rather than extrapolating."""
    block = stagnation_block(flat=True)
    i_stag, _ = util.get_i_stag(block)
    zeta = util.get_zeta(block)
    np.testing.assert_allclose(
        util.get_zeta_stag(block, i_stag), zeta[i_stag, np.arange(block.shape[1])]
    )


def test_a_stagnation_point_swept_round_the_nose_is_still_found():
    """The window holds the sweep, not just the nose it is centred on.

    The stagnation point moves round the leading edge with incidence, and a
    window narrow enough to cut into that sweep reports not-found on a section
    whose stagnation point is perfectly well defined --- or worse, keeps a
    flank of the flat pressure plateau around it and reads the ripples there
    as the peak.
    """
    block = stagnation_block(i_peak=22.0)

    # The geometric nose is the midpoint, so the peak is two nodes away, which
    # on this uniform mesh is 0.10 in normalised arc length.
    xrt_LE = np.asarray(block.xrt)[20, 0, :]
    i_stag, found = util.get_i_stag(block, xrt_LE=xrt_LE)

    assert found.all()
    assert (i_stag == 22).all()


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

    yaw = util.surface_normal_yaw(cut, util.get_zeta(cut)[node, :], MERIDIONAL, 0.0)

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
        [util.surface_normal_yaw(cut, np.full(3, z), MERIDIONAL, 0.0)[0] for z in zeta]
    )

    # A cell is one node spacing of sweep, and it is traversed evenly.
    step = ARC[1] - ARC[0]
    np.testing.assert_allclose(yaw, yaw[0] + fractions * step, atol=1e-3)


#
# SURFACE DISTRIBUTIONS
#
# Arithmetic, so no mesh below here. What the two-line fit has to survive is a
# flat top, a noisy curve and a curve with no peak at all -- the first being a
# design style rather than a pathology, and the last being what an unconverged
# march looks like.
#

TARGET = {
    "zeta_front": 0.1,
    "zeta_peak": 0.55,
    "ma_front": 0.585,
    "ma_peak": 1.3,
    "ma_TE": 1.0,
}
"""A distribution in the middle of the family, to perturb away from."""


def distribution(n=200, **overrides):
    """Return ``(zeta, ma)`` sampled from an exact two-line target."""
    zeta = np.linspace(0.1, 0.98, n)
    return zeta, util.loading_target(zeta, **{**TARGET, **overrides})


def test_the_target_hits_the_points_it_is_built_from():
    """Both anchors and the peak, which is the whole definition of the shape."""
    zeta = np.array([TARGET["zeta_front"], TARGET["zeta_peak"], 1.0])
    ma = util.loading_target(zeta, **TARGET)

    assert ma[0] == pytest.approx(TARGET["ma_front"])
    assert ma[1] == pytest.approx(TARGET["ma_peak"])
    assert ma[2] == pytest.approx(TARGET["ma_TE"])


def test_the_target_says_nothing_ahead_of_the_front_anchor():
    """NaN rather than an extrapolation, the window being where the claim is.

    Ahead of it the distribution belongs to the leading edge, and a line drawn
    there would read as a target nobody set.
    """
    zeta = np.linspace(0.0, 0.98, 99)
    ma = util.loading_target(zeta, **TARGET)

    assert np.all(np.isnan(ma[zeta < TARGET["zeta_front"]]))
    assert np.all(np.isfinite(ma[zeta >= TARGET["zeta_front"]]))


def test_the_fit_recovers_an_exact_two_line_curve():
    """The round trip: build a curve from four numbers, read them back."""
    zeta_peak, ma_peak, slope_front, slope_aft = util.fit_two_lines(*distribution())

    assert zeta_peak == pytest.approx(TARGET["zeta_peak"], abs=1e-3)
    assert ma_peak == pytest.approx(TARGET["ma_peak"], abs=1e-3)
    assert slope_front > 0.0 > slope_aft


@pytest.mark.parametrize("zeta_peak", [0.3, 0.45, 0.6, 0.75])
def test_the_fit_finds_the_peak_anywhere_in_the_window(zeta_peak):
    """Front-loaded through aft-loaded, which is the range of styles asked for."""
    measured, _, _, _ = util.fit_two_lines(*distribution(zeta_peak=zeta_peak))
    assert measured == pytest.approx(zeta_peak, abs=1e-3)


def test_the_fit_survives_noise():
    """Mesh noise on the surface must not move the peak much.

    The peak comes back as the intersection of two lines each fitted over many
    points, so noise averages out of it -- where an argmax would follow
    whichever node it happened to lift.
    """
    zeta, ma = distribution()
    noisy = ma + np.random.default_rng(0).normal(0.0, 0.01, ma.shape)

    zeta_peak, _, _, _ = util.fit_two_lines(zeta, noisy)
    assert zeta_peak == pytest.approx(TARGET["zeta_peak"], abs=0.02)


def test_the_fit_survives_a_flat_top():
    """A flat-topped profile is a design style, not a degenerate case.

    Where the peak sits is genuinely ill-defined on one, so this asks only that
    the answer stay finite and inside the window: a NaN here would stall the
    iteration on exactly the blades it was built for.
    """
    zeta, _ma = util.fit_two_lines(*distribution(ma_peak=1.02, ma_front=0.918))[:2]

    assert np.isfinite(zeta)
    assert 0.1 < zeta < 0.98


@pytest.mark.parametrize(
    "ma", [lambda z: z, lambda z: 2.0 - z, lambda z: 0.0 * z + 1.0]
)
def test_the_fit_refuses_a_curve_with_no_peak(ma):
    """Rising, falling and flat all have no interior maximum to place.

    NaN rather than the least-bad breakpoint, because the caller drops an
    unmeasured knob and steps on a measured one --- so a number invented here
    would move a design.
    """
    zeta = np.linspace(0.1, 0.98, 200)
    assert np.all(np.isnan(util.fit_two_lines(zeta, ma(zeta))))


def test_the_fit_needs_enough_points():
    """Three coefficients, so three points constrain nothing."""
    assert np.all(np.isnan(util.fit_two_lines([0.1, 0.5, 0.9], [0.5, 1.0, 0.8])))


def test_the_reduction_recovers_what_the_target_was_built_from():
    """`loading_from_distribution` against the numbers it is meant to read."""
    zeta_peak, ma_peak, ma_front = util.loading_from_distribution(*distribution())

    assert zeta_peak == pytest.approx(TARGET["zeta_peak"], abs=1e-3)
    assert ma_peak == pytest.approx(TARGET["ma_peak"], abs=1e-3)
    assert ma_front == pytest.approx(TARGET["ma_front"], abs=1e-3)


def test_the_reduction_scales_with_the_distribution():
    """Every number it reads is a Mach number, so all of them scale together.

    What the caller divides them by decides what survives a change of level:
    referred to the peak the ratio is untouched, and referred to the trailing
    edge -- which the duty pins while the peak floats -- it is not. That choice
    is `turbigen.iterate.measure_loading`'s and deliberately not made here.
    """
    zeta, ma = distribution()
    reference = np.array(util.loading_from_distribution(zeta, ma))

    for factor in (0.5, 2.0):
        scaled = np.array(util.loading_from_distribution(zeta, factor * ma))
        # The peak position is a position and does not scale; the two Mach
        # numbers do, exactly.
        assert scaled[0] == pytest.approx(reference[0], abs=1e-6)
        np.testing.assert_allclose(scaled[1:], factor * reference[1:], rtol=1e-6)


def test_the_reduction_reports_nothing_from_a_curve_with_no_peak():
    zeta = np.linspace(0.1, 0.98, 200)
    assert np.all(np.isnan(util.loading_from_distribution(zeta, zeta)))


def test_the_reduction_reports_nothing_from_an_empty_window():
    """A cut too coarse to have four points in the window measures nothing."""
    zeta = np.array([0.0, 0.05, 0.5, 1.0])
    assert np.all(np.isnan(util.loading_from_distribution(zeta, np.ones(4))))


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_the_suction_side_is_the_faster_one(sign):
    """Read off the flow, not off a mesh convention.

    `normalise_surface_distance` signs the two surfaces by the direction its
    cut loops in, which says nothing about which one the flow accelerates over.
    """
    zeta = np.linspace(-1.0, 1.0, 201)
    ma = np.where(np.sign(zeta) == sign, 1.0 + np.abs(zeta), 0.2 * np.abs(zeta))

    folded, kept = util.suction_side(zeta, ma)

    assert np.all(folded >= 0.0)
    assert np.all(np.diff(folded) >= 0.0)
    assert kept.max() == pytest.approx(2.0)


def test_the_suction_side_of_a_one_sided_cut_is_all_of_it():
    """Nothing to choose between, so folding is all there is to do."""
    zeta = np.linspace(0.0, 1.0, 51)
    folded, ma = util.suction_side(zeta, zeta**2)

    np.testing.assert_allclose(folded, zeta)
    np.testing.assert_allclose(ma, zeta**2)
