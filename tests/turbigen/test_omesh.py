"""Tests for O meshing with a shifted periodic boundary.

The O block is periodic with itself, its lower periodic face paired with its
upper the other way round and a pitch along, so besides what any mesher must
promise these check that ember pairs it, that the pairs are where the blade
normals put them, and that the wall gets the spacing and angle asked for.
"""

import numpy as np
import pytest
from scipy.spatial import cKDTree
from test_blade import blade, build
from test_ohmesh import _wall_metrics

from turbigen import OPeriodic, guess, util
from turbigen.omesh import LABELS, _ray_fractions

MESH = {
    "type": "o",
    "dspf_mid": 0.1,
    "resolution_factor": 0.5,
    "n_span_smooth": 3,
}
"""A deliberately coarse mesh, smoothed on few planes, so the tests run in
reasonable time."""


@pytest.fixture(scope="module")
def config():
    return build(mesh=MESH)


@pytest.fixture(scope="module")
def machine(config):
    return config.design()


@pytest.fixture(scope="module")
def spacing(config, machine):
    return config.mesh.wall_spacing(machine)


@pytest.fixture(scope="module")
def grid(config, machine):
    return config.mesh.mesh(machine)


def blocks_of_row(grid, i_row):
    return {
        b.label.split("_", 1)[1]: b for b in grid if b.label.startswith(f"row{i_row}_")
    }


#
# STRUCTURE
#


def test_three_blocks_per_row_with_meshable_sizes(machine, grid):
    assert len(grid) == 3 * len(machine.rows)
    for i_row, row in enumerate(machine.rows):
        blocks = blocks_of_row(grid, i_row)
        assert set(blocks) == set(LABELS)
        for block in blocks.values():
            assert block.Nb == row.n_blade
            for n in block.shape:
                assert (n - 1) % 8 == 0


def test_every_cell_has_positive_volume(grid):
    for block in grid:
        assert (block.vol_nd > 0.0).all(), block.label


def test_every_periodic_and_mixing_patch_pairs(grid):
    grid.connectivity.periodic.pair()
    grid.connectivity.mixing.pair()


def test_the_o_block_is_periodic_with_itself(config, machine, spacing, grid):
    # Lower periodic node A + j and upper node F - j are one pitch apart.
    for i_row, row in enumerate(machine.rows):
        lay = config.mesh.layout(machine, i_row, spacing)
        O = blocks_of_row(grid, i_row)["O"]
        j = np.arange(lay.i_C - lay.i_A + 1)
        lower = O.xrt[lay.i_A + j, :, -1]
        upper = O.xrt[lay.i_F - j, :, -1]
        np.testing.assert_allclose(lower[..., :2], upper[..., :2], atol=1e-10)
        np.testing.assert_allclose(
            upper[..., 2] - lower[..., 2], 2.0 * np.pi / row.n_blade, atol=1e-10
        )


def test_the_o_block_wraps_with_the_blade_on_k0(grid):
    for i_row in range(2):
        O = blocks_of_row(grid, i_row)["O"]
        np.testing.assert_allclose(O.xrt[0], O.xrt[-1], atol=1e-12)


def test_mixing_planes_have_matching_coordinates(grid):
    dn = blocks_of_row(grid, 0)["dn"]
    up = blocks_of_row(grid, 1)["up"]
    np.testing.assert_allclose(dn.xrt[-1, :, 0, :2], up.xrt[0, :, 0, :2], atol=1e-12)


#
# GEOMETRY
#


def test_the_wall_has_the_spacing_and_angle_asked_for(machine, spacing, grid):
    for i_row in range(len(machine.rows)):
        O = blocks_of_row(grid, i_row)["O"]
        j = O.shape[1] // 2
        r_ref = O.r[:, j, 0].mean()
        height, lean = _wall_metrics(O, j, r_ref)
        ratio = height / spacing.surface[i_row]
        assert np.mean(np.abs(ratio - 1.0) < 0.1) > 0.95
        assert np.mean(lean < 5.0) > 0.95


def test_the_blade_surface_is_the_designed_section(config, machine, spacing, grid):
    span_ref = machine.annulus.evaluate_span(1)
    spf = config.mesh.spanwise_grid(spacing.hub / span_ref, spacing.casing / span_ref)
    for i_row, row in enumerate(machine.rows):
        O = blocks_of_row(grid, i_row)["O"]
        chord = row.blade.evaluate_chord(0.5)
        # Nodes across the trailing edge face lie between the two surfaces'
        # end points rather than on either, so only the surfaces proper.
        i_lo, i_hi = config.mesh.layout(machine, i_row, spacing).i_te_corners
        for j in range(0, len(spf), 3):
            surfaces = row.blade.evaluate_section(spf[j], nchord=20000)
            r_ref = surfaces[0][1].mean()
            pts = np.concatenate([util.to_xrrt_ref(s, r_ref).T for s in surfaces])
            wall = util.to_xrrt_ref(O.xrt[i_lo : i_hi + 1, j, 0].T, r_ref).T
            dist, _ = cKDTree(pts).query(wall)
            assert dist.max() < 1e-4 * chord


#
# POST-PROCESSING
#


def test_the_blade_surface_cut_wraps_once_per_row(grid):
    surfs = util.cut_blade_surfs(grid)
    assert [len(s) for s in surfs] == [1, 1]


def test_the_initial_guess_is_finite_round_the_blade(machine, grid):
    guess.apply(grid, machine)
    for block in grid:
        assert np.isfinite(block.conserved).all(), block.label


#
# ROUND TRAILING EDGE
#


@pytest.fixture(scope="module")
def round_grid():
    from test_blade import ROUND

    config = build(blades=[blade(section={"thickness": ROUND})] * 2, mesh=MESH)
    machine = config.design()
    return machine, config.mesh.wall_spacing(machine), config.mesh.mesh(machine)


def test_a_round_trailing_edge_meshes_with_the_wall_square_all_round(round_grid):
    machine, spacing, grid = round_grid
    grid.connectivity.periodic.pair()
    for i_row in range(len(machine.rows)):
        O = blocks_of_row(grid, i_row)["O"]
        assert (O.vol_nd > 0.0).all()
        j = O.shape[1] // 2
        height, lean = _wall_metrics(O, j, O.r[:, j, 0].mean())
        # No trailing edge corners to leave alone: held everywhere.
        assert lean.max() < 5.0
        assert np.all(np.abs(height / spacing.surface[i_row] - 1.0) < 0.1)


#
# CONFIGURATION AND HELPERS
#


def test_a_tip_gap_is_refused_rather_than_shrouded():
    config = build(blades=[blade(), blade(tip_span=0.02)], mesh=MESH)
    with pytest.raises(ValueError, match="tip gap"):
        config.mesh.mesh(config.design())


def test_config_with_an_o_mesh_round_trips():
    config = build(mesh=MESH)
    assert type(config).from_dict(config.to_dict()) == config
    assert config.to_dict()["mesh"]["type"] == "o"


@pytest.mark.parametrize(
    "kwargs", [{"ratio_LE": 0.0}, {"ER_surf": 1.0}, {"L_Z": 0.0}, {"end_shear": 1.5}]
)
def test_inconsistent_settings_are_refused(kwargs):
    with pytest.raises(ValueError):
        OPeriodic.from_dict({"type": "o", **kwargs})


def test_rays_grow_from_the_wall_and_fill():
    D = np.array([0.5, 1.0, 2.0, 0.05])
    f = _ray_fractions(D, 0.01, 1.2, 16)
    cells = np.diff(f, axis=1) * D[:, None]
    np.testing.assert_allclose(f[:, [0, -1]], [[0, 1]] * 4)
    # The wall cell kept, whatever the ray, and the cells only growing.
    np.testing.assert_allclose(cells[:3, 0], 0.01, rtol=1e-6)
    assert (np.diff(cells[:3], axis=1) >= -1e-12).all()
    # At ER, capped, on a ray it fills; faster than ER on one it does not.
    assert (cells[0, 1:] / cells[0, :-1] <= 1.2 + 1e-9).all()
    assert cells[2, 1] / cells[2, 0] > 1.2
    # Too short to grow: uniform.
    np.testing.assert_allclose(cells[3], 0.05 / 16)
