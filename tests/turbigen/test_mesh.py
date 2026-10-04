"""Tests for meshing.

The mesher is the first stage whose result is not ours --- it produces an ember
`Grid` --- and the first with a framework method that does substantial work of
its own: wall spacings on the way in, and reference length, volume checking and
wall distance on the way out. The package this replaces leaves all of that to
the caller, so these check the framework as well as the mesh.
"""

import itertools
import sys
from pathlib import Path

import ember.patch
import numpy as np
import pytest
import turbigen.clusterfunc
import turbigen_ref.annulus
import turbigen_ref.geometry
import turbigen_ref.hmesh
from test_blade import ANNULUS, blade, build

from turbigen import H, Mesher, WallSpacing
from turbigen.config import Config

MESH = {
    "type": "h",
    "dm_TE": 0.05,
    "resolution_factor": 0.5,
    "dspf_mid": 0.1,
    # Stated rather than defaulted, because the reference mesher hard-codes
    # nine and takes no argument for it: comparing against it means asking
    # both for the same number. Our own default is lower, so what the golden
    # test pins is the algorithm rather than the choice of default -- which is
    # the right division, the default being ours to move and the algorithm not.
    "njtip_min": 9,
}
"""A deliberately coarse mesh, so that the tests run in about a second each."""

CUSP = {**MESH, "dm_TE": 0.0, "AR_cusp": 1.0, "ni_cusp": 8}
TIP = [blade(), blade(dchi_LE=2.0, tip_span=0.02)]


@pytest.fixture(scope="module")
def machine():
    return build(mesh=MESH).design()


@pytest.fixture(scope="module")
def grid(machine):
    return build(mesh=MESH).mesh.mesh(machine)


class AsOldBlade:
    """One of this package's blades, under the names the old mesher calls.

    The old mesher reads a blade through `evaluate_section` and `get_chi` and
    nothing else, and both mean here what they meant there.
    """

    def __init__(self, blade):
        self.blade = blade

    def evaluate_section(self, spf, nchord=10000, m=None):
        # The old mesher wants the higher-angle surface first, which is what
        # this package's `evaluate_section` used to promise and no longer does
        # -- it orders by surface now, and `hmesh` reorders for the same
        # reason this does.
        upper, lower = self.blade.evaluate_section(spf, nchord=nchord, m=m)
        if upper[2].mean() < lower[2].mean():
            upper, lower = lower, upper
        return upper, lower

    def get_chi(self, spf):
        return self.blade.evaluate_chi(spf)


def old_grid(machine, mesh, spacing):
    """The same mesh, generated through the package this replaces."""
    flat = machine.mean_line.flat
    cx_row = np.array(ANNULUS["cx_row"])
    cx_gap = np.array(ANNULUS["cx_gap"])
    annulus = turbigen_ref.annulus.MergedFixedAxialChord(
        {"cx_row": cx_row, "cx_gap": cx_gap}
    )
    annulus.forward(
        np.asarray(flat.r_mid, dtype=float),
        np.asarray(flat.span, dtype=float),
        np.asarray(flat.Beta, dtype=float),
        cx_row=cx_row,
        cx_gap=cx_gap,
        merge_weight=0.0,
    )

    # The old mesher is driven by *this* package's blades, through the two
    # methods it asks of one. The blade the old package would have built is a
    # fifth of a degree away at the endwalls, since it interpolates resolved
    # metal angles where this evaluates the vortex distribution wherever it is
    # asked --- a difference in the blade, which `test_blade` pins on its own,
    # and one that would otherwise leak into a comparison of meshers.
    blades = [[AsOldBlade(row.blade)] for row in machine.rows]
    mac = turbigen_ref.geometry.Machine(
        annulus,
        blades,
        np.array([r.n_blade for r in machine.rows]),
        np.array([r.tip_gap for r in machine.rows]),
        None,
    )

    # `type` names the mesher rather than configuring it, and `njtip_min` is a
    # setting the reference does not have -- it hard-codes nine, which is what
    # the fixture states so that both sides are asked for the same thing. A
    # key this mesher never took cannot be part of what the comparison pins.
    unknown = {"type", "njtip_min"}
    mesher = turbigen_ref.hmesh.H(**{k: v for k, v in mesh.items() if k not in unknown})
    reference = mesher.make_grid(
        None, mac, spacing.hub, spacing.casing, spacing.surface
    )

    # Old turbigen sets a reference length on the grid too, at config.py:709.
    # Without it the reference would skip a float32 rescale of the coordinates
    # that turbigen's mesher performs, and the two would differ by an epsilon
    # that has nothing to do with the mesh mathematics being compared.
    reference.set_L_ref(
        H(**{k: v for k, v in mesh.items() if k != "type"}).L_ref(machine)
    )

    return reference


#
# WALL SPACING
#


def test_characteristic_station_is_the_end_with_the_higher_relative_velocity(machine):
    for i_row in range(machine.mean_line.n_row):
        row = machine.mean_line[:, i_row]
        station = machine.mean_line.get_characteristic_station(i_row)

        i_expected = int(np.argmax(row.V_rel))
        assert station.V_rel == pytest.approx(row.V_rel[i_expected])
        assert station.rho == pytest.approx(row.rho[i_expected])


def test_surface_reynolds_number_matches_its_definition(machine):
    """On the machine, not the mesher: it needs no mesh to compute, and the
    Re_surf iterator wants it before there is one."""
    Re_surf = machine.Re_surf()

    for i_row, row in enumerate(machine.rows):
        station = machine.mean_line.get_characteristic_station(i_row)
        expected = (
            row.blade.evaluate_surface_length(0.5)[0]
            * station.rho
            * station.V_rel
            / station.mu
        )
        assert Re_surf[i_row] == pytest.approx(expected)


def test_wall_spacing_scales_with_yplus(machine):
    coarse = H(yplus=30.0).wall_spacing(machine)
    fine = H(yplus=1.0).wall_spacing(machine)

    np.testing.assert_allclose(fine.surface * 30.0, coarse.surface, rtol=1e-12)
    assert fine.hub * 30.0 == pytest.approx(coarse.hub)


def test_annulus_spacings_are_the_mean_of_the_rows(machine):
    spacing = H().wall_spacing(machine)

    assert spacing.hub == spacing.casing
    assert spacing.hub == pytest.approx(np.mean(spacing.surface))


def test_wall_spacing_is_a_small_fraction_of_the_chord(machine):
    """A sanity bound: a y+ of 30 is microns, not millimetres."""
    spacing = H().wall_spacing(machine)
    chord = machine.annulus.evaluate_chords(0.5)[1]

    assert np.all(spacing.surface > 0.0)
    assert np.all(spacing.surface < 0.01 * chord)


#
# THE FRAMEWORK
#


def test_mesh_finishes_the_grid_the_mesher_returns(machine, grid):
    """Scales, equation of state and wall distance are the framework's job.

    In the package this replaces they are steps every caller of `make_grid` has
    to remember, spread across `config.setup_mesh`, `config.adjust_ref` and two
    lines of `config.run`.
    """
    # The longest row chord at mid-span, off the annulus. Not the mean line,
    # which carries no length of its own.
    assert grid[0].L_ref == pytest.approx(
        machine.annulus.evaluate_chords(0.5)[1::2].max()
    )
    assert np.isfinite(grid[0].wdist).all()
    assert (grid[0].wdist >= 0.0).all()


def test_mesh_sets_the_scales_before_there_is_a_flow_to_scale(machine, grid):
    """The grid leaves the mesher with an equation of state ready for a solver.

    The scales have to be in place before any flow state is written, or the
    initial guess would be stored against unit references and the whole field
    would need rescaling afterwards. A mean line, read dimensionally and never
    iterated on, does not care about its own scales; a grid does.
    """
    reference = machine.mean_line.get_referenced_fluid()

    # The grid carries its references as float32, so the mean line's float64
    # values come back a rounding step away rather than bit-identical.
    assert grid[0].fluid.rho_ref == pytest.approx(reference.rho_ref, rel=1e-6)
    assert grid[0].fluid.V_ref == pytest.approx(reference.V_ref, rel=1e-6)
    assert grid[0].fluid.Rgas_ref == pytest.approx(reference.Rgas_ref, rel=1e-6)

    # Order one, which is the point of setting them at all.
    assert 0.1 < reference.rho_ref < 10.0
    assert 10.0 < reference.V_ref < 1000.0


def test_a_machine_without_blades_cannot_be_meshed():
    config = build(mesh=MESH)
    machine_no_blades = type(config.design())(mean_line=config.design().mean_line)

    with pytest.raises(ValueError, match="needs blades"):
        config.mesh.mesh(machine_no_blades)


def test_negative_volumes_are_raised_not_exited(grid):
    """The old code calls sys.exit(1) here, which a caller cannot catch."""

    class Collapsed(Mesher):
        """A mesher that returns a block with a collapsed cell."""

        def forward(self, machine, spacing):
            del machine, spacing
            return grid

    block = grid[0]
    x = block.x.copy()
    try:
        block.set_x(np.zeros_like(x))
        with pytest.raises(ValueError, match="negative volume"):
            Collapsed().check_volumes(grid)
    finally:
        block.set_x(x)


#
# THE MESH
#


def test_one_block_per_row_with_meshable_sizes(machine, grid):
    assert len(grid) == len(machine.rows)

    for block in grid:
        assert block.Nb in [r.n_blade for r in machine.rows]
        for n in block.shape:
            # Multigrid needs each direction to coarsen three times
            assert (n - 1) % 8 == 0


def test_every_cell_has_positive_volume(grid):
    for block in grid:
        assert (block.vol_nd > 0.0).all()


def test_mixing_planes_have_matching_coordinates(grid):
    for upstream, downstream in itertools.pairwise(grid):
        np.testing.assert_allclose(
            upstream.xrt[-1, :, 0, :2],
            downstream.xrt[0, :, 0, :2],
            atol=1e-12,
        )


def test_the_mesh_advances_downstream_at_midspan(grid):
    for block in grid:
        x_mid = block.x[:, block.shape[1] // 2, block.shape[2] // 2]
        assert (np.diff(x_mid) > 0.0).all()


def test_the_passage_runs_from_the_first_periodic_face_to_the_second(grid):
    """k increases with theta, so the two periodic faces cannot cross.

    Pairing them is the mesher's own last step and would raise here if they
    did not match, but nothing in that pairing says which way round they are:
    a block wound the other way pairs just as happily and then has negative
    volumes everywhere.
    """
    for block in grid:
        assert (block.xrt[..., -1, 2] > block.xrt[..., 0, 2]).all()


#
# AGREEMENT WITH THE PACKAGE THIS REPLACES
#

bit_exact = pytest.mark.skipif(
    sys.platform != "linux",
    reason=(
        "bit-exact agreement with the frozen reference holds on the platform "
        "it was frozen on, and is a regression check rather than a portable "
        "property: the two implementations reach the same coordinate by "
        "different expressions, which other platforms may evaluate to a "
        "different last bit"
    ),
)
"""Restrict a coordinate-for-coordinate comparison to where it means something.

The two meshers agree exactly here --- 0 of 571950 coordinates differ --- so
the tolerance below asserts equality rather than closeness, and that is worth
keeping: it catches any drift at all in a mesher being migrated. It is not a
cross-platform claim. On Windows one node in 298275 came back 8.3e-08 apart,
which is float32 epsilon territory and says only that a compiler ordered an
expression differently, not that either mesh is wrong.

The tests that compare patches rather than coordinates are unaffected and run
everywhere, as does everything else in this file.
"""


@pytest.fixture
def vinokur(monkeypatch):
    """Put back the pitchwise and spanwise clustering the reference uses.

    This mesher spaces the pitch and the span on plateaus, where the package
    it replaces used Vinokur stretching, so no coordinate of a passage would
    otherwise agree. Mapping the plateau calls back onto the clustering they
    replaced keeps everything else --- the streamwise grid, the blocks, the
    patches --- compared exactly, which is what these tests are for. The
    clustering itself is pinned in `test_clusterfunc_plateau.py`.
    """
    cf = turbigen.clusterfunc

    def single_free(dmin, dmax, ERmax, x0=0.0, x1=1.0, mult=8, width=None):
        return cf.single.free(dmin, dmax, ERmax, x0, x1, mult)

    def double_free(dx0, dx1, dmax, ERmax, x0=0.0, x1=1.0, mult=8, width=None):
        return cf.double.free(dx0, dx1, dmax, ERmax, x0, x1, mult)

    def symmetric_free(dmin, dmax, ERmax, x0=0.0, x1=1.0, mult=8, width=None):
        return cf.symmetric.free(dmin, dmax, ERmax, x0, x1, mult)

    def symmetric_fixed(dmin, dmax, ERmax, N, x0=0.0, x1=1.0, width=None):
        return cf.symmetric.fixed(dmin, N, x0, x1)

    monkeypatch.setattr(cf.single, "plateau_free", single_free)
    monkeypatch.setattr(cf.double, "plateau_free", double_free)
    monkeypatch.setattr(cf.symmetric, "plateau_free", symmetric_free)
    monkeypatch.setattr(cf.symmetric, "plateau_fixed", symmetric_fixed)


@bit_exact
def test_matches_the_turbigen_implementation(machine, vinokur):
    grid = build(mesh=MESH).mesh.mesh(machine)
    reference = old_grid(
        machine,
        MESH,
        H(**{k: v for k, v in MESH.items() if k != "type"}).wall_spacing(machine),
    )

    assert len(grid) == len(reference)
    for block, block_ref in zip(grid, reference):
        assert block.shape == block_ref.shape
        assert block.Nb == block_ref.Nb
        np.testing.assert_allclose(block.xrt, block_ref.xrt, rtol=1e-12, atol=1e-12)
        assert [repr(p) for p in block.patches] == [repr(p) for p in block_ref.patches]


@bit_exact
@pytest.mark.parametrize(
    "mesh, blades",
    [(CUSP, None)],
    ids=["cusp"],
)
def test_optional_features_match_the_turbigen_implementation(mesh, blades, vinokur):
    """The cusp reshapes the block, so it is compared exactly too.

    The tip gap used to be compared here as well, and is no longer: the old
    mesher sized the gap's end spacing as the clearance over the node count,
    which leaves four cells to fill five spacings and a jump of one and a half
    across the gap. This mesher divides by the cell count instead, which moves
    every node of a tipped row, so there is nothing exact left to compare.
    `test_the_tip_gap_spanwise_spacing_stays_within_the_expansion_ratio` pins
    what replaced it.

    Nor is the cusp compared all the way down. This mesher takes the slope
    into the trailing-edge corner from the section rather than from the grid,
    and scales the corner windows with the resolution factor, so the trailing
    edge it builds is deliberately not the old one. Upstream of the old
    corner blend, which starts `2 * ni_TE` points before the cusp, nothing has
    changed, and that part is still compared exactly.

    Rotating patches are compared separately, below. The old mesher places
    none: there, rotation arrives later from `Grid.apply_rotation` at
    boundary-condition time, which is the arrangement this deliberately
    departs from. Filtering rather than loosening keeps every patch the old
    mesher does make pinned exactly.
    """
    config = build(blades=blades, mesh=mesh)
    machine = config.design()

    grid = config.mesh.mesh(machine)
    reference = old_grid(machine, mesh, config.mesh.wall_spacing(machine))

    for block, block_ref in zip(grid, reference):
        assert block.shape == block_ref.shape
        ite = next(p.ist for p in block.patches if isinstance(p, ember.patch.CuspPatch))
        i_blend = ite - 2 * config.mesh.ni_TE
        np.testing.assert_allclose(
            block.xrt[: i_blend + 1],
            block_ref.xrt[: i_blend + 1],
            rtol=1e-12,
            atol=1e-12,
        )
        shared = [
            p for p in block.patches if not isinstance(p, ember.patch.RotatingPatch)
        ]
        assert [repr(p) for p in shared] == [repr(p) for p in block_ref.patches]


@pytest.mark.parametrize("njtip_min", [5, 9])
@pytest.mark.parametrize("tip", [0.0025, 0.005, 0.03])
def test_the_tip_gap_spanwise_spacing_stays_within_the_expansion_ratio(tip, njtip_min):
    """No jump across a clearance too small for the free clustering to fill.

    Below a few casing spacings the gap falls back to a fixed count of nodes,
    and sizing its ends as the clearance over the node count made the middle
    cells half as large again as the ends. Every clearance in a typical sweep
    takes that path, and the jump sat where a mixing-plane blow-up began.
    """
    mesher = H(njtip_min=njtip_min)
    dspf_wall = 3.4e-3

    spf = mesher.spanwise_grid(dspf_wall, dspf_wall, tip)

    # The gap, plus the last cell of the main passage it joins onto.
    gap = np.diff(spf[spf >= 1.0 - tip - 1e-12])
    joined = np.diff(spf)[-(len(gap) + 1) :]
    ratio = joined[1:] / joined[:-1]
    assert np.maximum(ratio, 1.0 / ratio).max() <= mesher.ER_span
    assert len(gap) >= njtip_min - 1


@pytest.mark.parametrize("resolution_factor", [0.5, 2.0])
def test_the_tip_gap_does_not_scale_with_resolution(resolution_factor):
    """Only the main passage takes the resolution factor.

    Resampling the gap with a multigrid multiple rounded its cells up to eight
    at any factor but one, halving the casing spacing with it, so a resolution
    study meshed a different wall at every level.
    """
    tip, dspf_wall = 0.005, 3.4e-3
    base = H().spanwise_grid(dspf_wall, dspf_wall, tip)
    scaled = H(resolution_factor=resolution_factor).spanwise_grid(
        dspf_wall, dspf_wall, tip
    )

    def gap(spf):
        return spf[spf >= 1.0 - tip - 1e-12]

    np.testing.assert_allclose(gap(scaled), gap(base))
    assert (len(scaled) - 1) % 8 == 0


def test_half_span_clusters_at_the_hub_only():
    mesher = H(half_span=True)
    dspf_hub = 1e-3

    spf = mesher.spanwise_grid(dspf_hub, 1e-4, 0.0)

    assert spf[0] == 0.0
    assert np.isclose(spf[-1], 1.0)
    assert (len(spf) - 1) % 8 == 0
    ds = np.diff(spf)
    assert (ds > 0.0).all()
    assert ds[0] <= dspf_hub * 1.1
    # Grows away from the hub and is never clustered again at the casing.
    assert (ds[1:] / ds[:-1] <= mesher.ER_span * 1.01).all()
    assert ds[-1] > 10.0 * dspf_hub
    # The casing spacing is ignored.
    np.testing.assert_array_equal(spf, mesher.spanwise_grid(dspf_hub, 1e-2, 0.0))


def test_half_span_is_fewer_points_than_two_sided():
    dspf_hub = 1e-3
    one = H(half_span=True).spanwise_grid(dspf_hub, H().dspf_mid, 0.0)
    two = H().spanwise_grid(dspf_hub, dspf_hub, 0.0)
    assert len(one) < len(two)


def test_half_span_refuses_a_tip_gap():
    with pytest.raises(ValueError, match="half_span"):
        H(half_span=True).spanwise_grid(1e-3, 1e-3, 0.01)


def test_ni_down_extra_must_keep_the_multigrid_alignment():
    with pytest.raises(ValueError, match="multiple of eight"):
        H.from_dict({"type": "h", "ni_down_extra": 4})


def test_ni_down_extra_adds_points_downstream_of_the_cusp():
    machine = build(mesh=CUSP).design()
    base = build(mesh=CUSP).mesh.mesh(machine)
    extra = build(mesh={**CUSP, "ni_down_extra": 8}).mesh.mesh(machine)

    for a, b in zip(extra, base):
        assert a.shape == (b.shape[0] + 8, *b.shape[1:])
        # Upstream of the cusp the grid is the same.
        ite = next(p.ist for p in b.patches if isinstance(p, ember.patch.CuspPatch))
        np.testing.assert_array_equal(a.xrt[: ite + 1], b.xrt[: ite + 1])


def test_a_tip_gap_is_the_one_patch_the_old_mesher_does_not_place(machine):
    """The deliberate divergence, stated rather than tolerated.

    A wall takes its block's angular velocity unless a rotating patch overrides
    it, so a casing over a tip gap is the only wall that needs saying, and
    saying it is placement rather than value -- which is why it belongs to the
    mesher and its speed does not.
    """
    with_gap = build(blades=TIP, mesh=MESH).design()
    grid = H(**{k: v for k, v in MESH.items() if k != "type"}).mesh(with_gap)

    labels = [p.label for p in grid.patches.rotating]
    assert labels == ["casing"]

    # And none at all without a gap, the shrouded row needing no override.
    plain = H(**{k: v for k, v in MESH.items() if k != "type"}).mesh(machine)
    assert not plain.patches.rotating


#
# SERIALISATION
#


def test_config_with_a_mesh_round_trips():
    config = build(mesh=MESH)

    assert type(config).from_dict(config.to_dict()) == config


def test_mesh_defaults_are_written_out():
    data = build(mesh=MESH).to_dict()["mesh"]

    assert data["type"] == "h"
    assert data["yplus"] == 30.0
    assert data["ER_span"] == 1.2


def test_a_cusp_needs_the_trailing_edge_at_the_true_trailing_edge():
    with pytest.raises(ValueError, match="ni_cusp requires"):
        H.from_dict({"type": "h", "ni_cusp": 8, "dm_TE": 0.05})


def _stream_ends(grid):
    """Return (inlet cell, exit cell) streamwise spacing of each block, midspan."""
    ends = []
    for block in grid:
        m = np.hypot(
            np.diff(block.x[:, block.shape[1] // 2, block.shape[2] // 2]),
            np.diff(block.r[:, block.shape[1] // 2, block.shape[2] // 2]),
        )
        ends.append((m[0], m[-1]))
    return ends


def test_ar_mix_defaults_to_ar_stream():
    """No behaviour change until it is lowered."""
    assert H().AR_mix == H().AR_stream

    machine = build(mesh=MESH).design()
    same = build(mesh={**MESH, "AR_mix": H().AR_stream}).mesh.mesh(machine)
    base = build(mesh=MESH).mesh.mesh(machine)

    for a, b in zip(same, base):
        assert a.shape == b.shape


def test_ar_mix_tightens_the_mixing_plane_and_not_the_true_boundaries():
    machine = build(mesh=MESH).design()
    wide = build(mesh=MESH).mesh.mesh(machine)  # AR_mix == AR_stream
    tight = build(mesh={**MESH, "AR_mix": 1.0}).mesh.mesh(machine)

    (in0_w, ex0_w), (in1_w, ex1_w) = _stream_ends(wide)
    (in0_t, ex0_t), (in1_t, ex1_t) = _stream_ends(tight)

    # Row 0 exit and row 1 inlet are the mixing plane: finer with AR_mix = 1.
    assert ex0_t < 0.9 * ex0_w
    assert in1_t < 0.9 * in1_w
    # Row 0 inlet is the true machine inlet, row 1 exit the true outlet: untouched.
    assert in0_t == pytest.approx(in0_w, rel=0.02)
    assert ex1_t == pytest.approx(ex1_w, rel=0.02)


def _knife_edged(blade):
    """Return `blade` with every section closed to a point at the trailing edge."""
    for section in blade["sections"]:
        section["thickness"] = {**section["thickness"], "t_TE": 0.0}
    return blade


def test_a_cusp_needs_a_trailing_edge_to_be_built_on():
    """A knife edge has no width for `AR_cusp` to be a multiple of.

    Left to run, the sides cross and an assertion on the pitchwise ordering of
    the block coordinates fails several steps later, naming nothing a designer
    wrote. Measured off the section rather than read off a thickness parameter,
    so a distribution that closes to a point without saying so is caught too.
    """
    config = build(
        blades=[_knife_edged(blade()), _knife_edged(blade(dchi_LE=2.0))], mesh=CUSP
    )
    machine = config.design()

    with pytest.raises(ValueError, match="cusp is built on the width"):
        config.mesh.mesh(machine)


def test_a_square_trailing_edge_does_not_need_one():
    """The check is the cusp's, so `AR_cusp = 0` meshes a knife edge as before."""
    config = build(
        blades=[_knife_edged(blade()), _knife_edged(blade(dchi_LE=2.0))], mesh=MESH
    )
    config.mesh.mesh(config.design())


def test_wall_spacing_is_not_a_config_node():
    """It is a result: computed from the machine, never written to a file."""
    assert not issubclass(WallSpacing, Mesher)
    assert not hasattr(WallSpacing, "to_dict")


NARROW_WEDGE = Path(__file__).parents[1] / "data" / "cusp_narrow_wedge.yaml"


def test_a_narrow_trailing_edge_wedge_cusps_without_crossing():
    """Both sides of a cusp are spaced on one meridional coordinate.

    The H-mesh keeps one (x, r) per streamwise index for both surfaces, so a
    cusp spaced separately on each side's arc length is put back at a
    meridional position that is not its own, with its own theta. This stator
    has a narrow enough wedge, and so a long enough cusp, that nine points
    geometrically clustered from the trailing edge reached the tip on one side
    and not the other. The side that could not fell back to uniform spacing,
    and the two surfaces crossed over on every span.
    """
    config = Config.from_file(NARROW_WEDGE)
    grid = config.mesh.mesh(config.design())

    for block in grid:
        ite = next(p.ist for p in block.patches if isinstance(p, ember.patch.CuspPatch))
        j = block.shape[1] // 2
        dm = np.hypot(np.diff(block.x[:, j, 0]), np.diff(block.r[:, j, 0]))
        # No spacing jump where the blade's last cell meets the cusp's first.
        assert dm[ite] == pytest.approx(dm[ite - 1], rel=0.02)
