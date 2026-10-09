"""Tests for O-H meshing.

The mesher builds a mesh on surfaces of constant span fraction, each in its own
conformal coordinates, so these check what that promises on the way back out:
nodes on the surface they were meshed on, the blade where the blade is, the
wall spacing and angle the wall control asks for, and blocks that ember will
accept as one mesh.
"""

import numpy as np
import pytest
from scipy.spatial import cKDTree
from test_blade import blade, build

from turbigen import OH, guess, util
from turbigen.ohmesh import LABELS, ConformalMap

MESH = {
    "type": "oh",
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
def grid(config, machine):
    return config.mesh.mesh(machine)


@pytest.fixture(scope="module")
def spf(config, machine):
    """Span fraction of each spanwise plane, as the mesher chose them."""
    mesher = config.mesh
    spacing = mesher.wall_spacing(machine)
    span_ref = machine.annulus.evaluate_span(1)
    return mesher.spanwise_grid(spacing.hub / span_ref, spacing.casing / span_ref)


def blocks_of_row(grid, i_row):
    """Return the seven blocks of a row, by name."""
    return {
        b.label.split("_", 1)[1]: b for b in grid if b.label.startswith(f"row{i_row}_")
    }


#
# STRUCTURE
#


def test_seven_blocks_per_row_with_meshable_sizes(machine, grid):
    assert len(grid) == 7 * len(machine.rows)
    for i_row, row in enumerate(machine.rows):
        blocks = blocks_of_row(grid, i_row)
        assert set(blocks) == set(LABELS)
        for block in blocks.values():
            assert block.Nb == row.n_blade
            for n in block.shape:
                assert (n - 1) % 8 == 0


def test_every_block_spans_the_same_planes(grid):
    assert len({block.shape[1] for block in grid}) == 1


def test_every_cell_has_positive_volume(grid):
    for block in grid:
        assert (block.vol_nd > 0.0).all(), block.label


def test_every_periodic_and_mixing_patch_pairs(grid):
    # Pairing raises if any patch is left without a partner.
    grid.connectivity.periodic.pair()
    grid.connectivity.mixing.pair()


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


def test_every_node_lies_on_the_surface_it_was_meshed_on(machine, grid, spf):
    annulus = machine.annulus
    m = np.linspace(0.0, annulus.m_max, 20001)
    for j in range(0, len(spf), 3):
        line = annulus.evaluate_xr(m, spf[j]).T
        tree = cKDTree(line)
        h = np.median(np.linalg.norm(np.diff(line, axis=0), axis=1))
        for block in grid:
            xr = np.stack((block.x[:, j, :], block.r[:, j, :]), -1).reshape(-1, 2)
            dist, _ = tree.query(xr)
            # To within the chord of the sampled line, which is far finer than
            # any cell.
            assert dist.max() < h, block.label


def test_the_blade_surface_is_the_designed_section(config, machine, grid, spf):
    spacing = config.mesh.wall_spacing(machine)
    for i_row, row in enumerate(machine.rows):
        O = blocks_of_row(grid, i_row)["O"]
        chord = row.blade.evaluate_chord(0.5)
        # Nodes across the trailing edge face lie between the two surfaces'
        # end points rather than on either, so only the surfaces proper.
        i_lo, i_hi = config.mesh.layout(machine, i_row, spacing).i_te_corners
        surface = slice(i_lo, i_hi + 1)
        for j in range(0, len(spf), 3):
            surfaces = row.blade.evaluate_section(spf[j], nchord=20000)
            r_ref = surfaces[0][1].mean()
            pts = np.concatenate(
                [util.to_xrrt_ref(s, r_ref).T for s in surfaces], axis=0
            )
            tree = cKDTree(pts)
            wall = util.to_xrrt_ref(O.xrt[surface, j, 0].T, r_ref).T
            dist, _ = tree.query(wall)
            assert dist.max() < 1e-4 * chord


def _wall_metrics(O, j, r_ref):
    """First-cell height and lean off the wall of an O block, at plane `j`."""
    wall = util.to_xrrt_ref(O.xrt[:-1, j, 0].T, r_ref).T
    first = util.to_xrrt_ref(O.xrt[:-1, j, 1].T, r_ref).T
    tangent = np.roll(wall, -1, axis=0) - np.roll(wall, 1, axis=0)
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
    off = first - wall
    along = np.sum(off * tangent, axis=1)
    height = np.linalg.norm(off - along[:, None] * tangent, axis=1)
    return height, np.degrees(np.arctan(np.abs(along) / height))


def test_the_wall_has_the_spacing_and_angle_asked_for(config, machine, grid, spf):
    spacing = config.mesh.wall_spacing(machine)
    for i_row in range(len(machine.rows)):
        O = blocks_of_row(grid, i_row)["O"]
        j = len(spf) // 2
        r_ref = O.r[:, j, 0].mean()
        height, lean = _wall_metrics(O, j, r_ref)
        ratio = height / spacing.surface[i_row]
        # The two trailing edge corners have no normal to be square to, and
        # are left alone; everywhere else is held.
        assert np.mean(np.abs(ratio - 1.0) < 0.1) > 0.95
        assert np.mean(lean < 5.0) > 0.95


#
# POST-PROCESSING
#


def test_the_blade_surface_cut_wraps_once_per_row(grid):
    surfs = util.cut_blade_surfs(grid)
    assert [len(s) for s in surfs] == [1, 1]
    for row in surfs:
        assert np.allclose(row[0].xrt[0], row[0].xrt[-1])


def test_endwalls_are_a_hub_and_casing_face_of_each_block(grid):
    walls = util.cut_endwalls(grid)
    assert [len(w) for w in walls] == [14, 14]
    assert util.cut_blade_tips(grid) == [[], []]


def test_the_initial_guess_is_finite_round_the_blade(machine, grid):
    # Only reads the flow it writes, so the shared grid can take it.
    guess.apply(grid, machine)
    for block in grid:
        assert np.isfinite(block.conserved).all(), block.label


#
# CONFIGURATION
#


def test_a_tip_gap_is_refused_rather_than_shrouded():
    config = build(blades=[blade(), blade(tip_span=0.02)], mesh=MESH)
    with pytest.raises(ValueError, match="tip gap"):
        config.mesh.mesh(config.design())


def test_config_with_an_oh_mesh_round_trips():
    config = build(mesh=MESH)
    assert type(config).from_dict(config.to_dict()) == config
    assert config.to_dict()["mesh"]["type"] == "oh"


@pytest.mark.parametrize("kwargs", [{"ni_TE": 8}, {"LE_split": 0.96}, {"frac_Z": 1.0}])
def test_inconsistent_settings_are_refused(kwargs):
    with pytest.raises(ValueError):
        OH.from_dict({"type": "oh", **kwargs})


#
# CONFORMAL MAP
#


def test_the_conformal_map_round_trips(machine):
    cmap = ConformalMap(machine.annulus, 0.3, 0.5, 4.5)
    mp = np.linspace(cmap.mp[0], cmap.mp[-1], 101)
    np.testing.assert_allclose(cmap.mp_from_xr(cmap.xr_from_mp(mp)), mp, atol=1e-9)


def test_the_conformal_map_keeps_angles(machine):
    # A line at 45 degrees in (m', theta) is at 45 degrees on the surface,
    # whatever the radius, because dm' = dm / r.
    cmap = ConformalMap(machine.annulus, 0.7, 0.5, 4.5)
    mp = np.linspace(cmap.mp[0], cmap.mp[-1], 2001)
    theta = mp.copy()
    xr = cmap.xr_from_mp(mp)
    dm = np.linalg.norm(np.diff(xr, axis=1), axis=0)
    r = 0.5 * (xr[1, 1:] + xr[1, :-1])
    angle = np.degrees(np.arctan2(r * np.diff(theta), dm))
    np.testing.assert_allclose(angle, 45.0, atol=1e-3)
