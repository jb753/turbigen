"""Tests for meshing.

The mesher is the first stage whose result is not ours --- it produces an ember
`Grid` --- and the first with a framework method that does substantial work of
its own: wall spacings on the way in, and reference length, volume checking and
wall distance on the way out. The package this replaces leaves all of that to
the caller, so these check the framework as well as the mesh.
"""

import itertools
import sys

import ember.patch
import numpy as np
import pytest
import turbigen_ref.annulus
import turbigen_ref.geometry
import turbigen_ref.hmesh
from test_blade import ANNULUS, blade, build

from turbigen import H, Mesher, WallSpacing

MESH = {
    "type": "h",
    "dm_TE": 0.05,
    "resolution_factor": 0.5,
    "dspf_mid": 0.1,
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
        return self.blade.evaluate_section(spf, nchord=nchord, m=m)

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

    mesher = turbigen_ref.hmesh.H(**{k: v for k, v in mesh.items() if k != "type"})
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
            row.blade.evaluate_surface_length(0.5)
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


@bit_exact
def test_matches_the_turbigen_implementation(machine, grid):
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
    [(CUSP, None), (MESH, TIP)],
    ids=["cusp", "tip_gap"],
)
def test_optional_features_match_the_turbigen_implementation(mesh, blades):
    """The cusp and the tip gap are the two paths that reshape the block.

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
        np.testing.assert_allclose(block.xrt, block_ref.xrt, rtol=1e-12, atol=1e-12)
        shared = [
            p for p in block.patches if not isinstance(p, ember.patch.RotatingPatch)
        ]
        assert [repr(p) for p in shared] == [repr(p) for p in block_ref.patches]


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


def test_wall_spacing_is_not_a_config_node():
    """It is a result: computed from the machine, never written to a file."""
    assert not issubclass(WallSpacing, Mesher)
    assert not hasattr(WallSpacing, "to_dict")
