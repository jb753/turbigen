"""Tests for the tip gap of the passage-block O-H mesher.

A two-row machine with a clearance over the second row only, meshed coarse so
the tests run in reasonable time. The gap is two blocks inside the outline the
blade would have had, an O block and an H block, so these check that they fit
the rest of the mesh node for node, that the blade surface and endwalls are
read off it as they would be off an H mesh's gridded tip, and that the cells
in the gap are fit to solve on. The rest guard what has gone wrong before:
steps in spacing along the blade, across the O block and spanwise into
the clearance, poor cells in the
main blocks, and a guessed flow turned the wrong way in the blocks whose `i`
does not run downstream.
"""

import numpy as np
import pytest
from test_blade import ROUND, blade, build

from turbigen import OHPassage, guess, util
from turbigen.ohpmesh import _cell_angles

MESH = {
    "type": "ohp",
    "dspf_mid": 0.1,
    "resolution_factor": 0.5,
    "n_span_smooth": 3,
}
"""A deliberately coarse mesh, swept on few planes."""

TIP = 0.02
"""Clearance of the second row, as a fraction of span."""


@pytest.fixture(scope="module")
def machine():
    config = build(
        blades=[
            blade(section={"thickness": ROUND}),
            blade(dchi_LE=2.0, section={"thickness": ROUND}, tip_span=TIP),
        ],
        mesh=MESH,
    )
    return config.design()


@pytest.fixture(scope="module")
def grid(machine):
    return OHPassage.from_dict(MESH).mesh(machine)


def blocks_of_row(grid, i_row):
    """Return the blocks of a row, by name."""
    return {
        b.label.split("_", 1)[1]: b for b in grid if b.label.startswith(f"row{i_row}_")
    }


def test_the_row_with_a_gap_has_two_more_blocks(grid):
    assert len(blocks_of_row(grid, 0)) == 8
    row = blocks_of_row(grid, 1)
    assert len(row) == 10
    assert {"tipO", "tipH"} <= set(row)
    for name in ("tipO", "tipH"):
        assert [len(r) for r in grid.rows] == [8, 10]
        assert row[name].shape[1] < row["O"].shape[1]


def test_every_cell_has_positive_volume(grid):
    for block in grid:
        assert (block.vol > 0.0).all(), block.label


def test_every_periodic_patch_pairs(grid):
    pairs = grid.connectivity.periodic.pair()
    n_periodic = sum(len(block.patches.periodic) for block in grid)
    assert len(pairs) == n_periodic


def test_the_gap_outline_is_the_o_block_over_the_gap(grid):
    # Renoded, so joined by a non-matching interface rather than node for
    # node: the two faces pair, unpermuted and unflipped, and each lies on
    # the other to well within the size of a cell.
    pairs = grid.connectivity.nonmatch.pair()
    assert len(pairs) == 2
    for _, transform in pairs.values():
        assert transform == ((0, 1, 2), ())

    row = blocks_of_row(grid, 1)
    O, tO = row["O"], row["tipO"]
    jtip = O.shape[1] - tO.shape[1]
    assert tO.shape[0] == O.shape[0]
    outline = tO.xrt[:, :, -1]
    face = O.xrt[:, jtip:, 0]
    np.testing.assert_array_equal(outline[[0, -1]], face[[0, -1]])
    xy = lambda xrt: np.stack((xrt[..., 0], xrt[..., 1] * xrt[..., 2]), axis=-1)
    chord = np.ptp(face[:, 0, 0])
    for j in range(face.shape[1]):
        P, W = xy(outline[:, j]), xy(face[:, j])
        seg = W[1:] - W[:-1]
        t = np.einsum("pij,ij->pi", P[:, None] - W[None, :-1], seg)
        t = np.clip(t / np.sum(seg**2, axis=-1), 0.0, 1.0)
        foot = W[None, :-1] + t[..., None] * seg
        dist = np.linalg.norm(P[:, None] - foot, axis=-1).min(axis=1)
        assert dist.max() < 1e-4 * chord


def test_the_gap_spans_the_clearance(grid):
    row = blocks_of_row(grid, 1)
    O, tO = row["O"], row["tipO"]
    jtip = O.shape[1] - tO.shape[1]
    # The top of the blade the clearance below the casing, as the fraction of
    # span the blade's clearance was given as. Measured between nodes of one
    # i, which sit at slightly different meridional positions on each plane,
    # so only the middle of the spread is exact.
    xr = np.stack((O.x[:, :, 0], O.r[:, :, 0]))
    height = np.linalg.norm(xr[:, :, -1] - xr[:, :, jtip], axis=0)
    span = np.linalg.norm(xr[:, :, -1] - xr[:, :, 0], axis=0)
    assert np.isclose(np.median(height / span), TIP, rtol=1e-2)
    assert np.allclose(height / span, TIP, rtol=5e-2)
    for block in row.values():
        assert not (block.shape[1] - 1) % 8


def test_the_blade_surface_stops_at_the_tip(grid):
    surfs = util.cut_blade_surfs(grid)
    row = blocks_of_row(grid, 1)
    jtip = row["O"].shape[1] - row["tipO"].shape[1]
    assert surfs[0][0].shape[1] == row["O"].shape[1]
    assert len(surfs[1]) == 1
    assert surfs[1][0].shape[1] == jtip + 1


def test_the_tip_is_both_blocks_of_the_gap(grid):
    tips = util.cut_blade_tips(grid)
    assert tips[0] == []
    assert len(tips[1]) == 2


def test_the_casing_stands_still_over_the_gap_only(grid):
    for block in blocks_of_row(grid, 0).values():
        assert not list(block.patches.rotating)
    for block in blocks_of_row(grid, 1).values():
        (patch,) = block.patches.rotating
        assert patch.label == "casing"


def test_the_cells_in_the_gap_are_not_badly_skewed(grid):
    row = blocks_of_row(grid, 1)
    for name in ("tipO", "tipH"):
        block = row[name]
        rt = block.r * block.t
        worst = min(
            _cell_angles(np.stack((block.x[:, j, :], rt[:, j, :]))).min()
            for j in range(block.shape[1])
        )
        assert worst > 10.0, name


def test_rows_with_different_clearances_are_refused():
    config = build(
        blades=[
            blade(section={"thickness": ROUND}, tip_span=0.01),
            blade(dchi_LE=2.0, section={"thickness": ROUND}, tip_span=0.02),
        ],
        mesh=MESH,
    )
    with pytest.raises(ValueError, match="same one as a fraction of span"):
        config.mesh.mesh(config.design())


@pytest.mark.parametrize(
    "kwargs",
    [{"nk_tip": 10}, {"nk_tip_O": 5}, {"njtip_min": 4}, {"t_tip_O": 0.5}],
)
def test_inconsistent_tip_settings_are_refused(kwargs):
    with pytest.raises(ValueError):
        OHPassage.from_dict({"type": "ohp", **kwargs})


def _cartesian(block):
    xrt = block.xrt.astype(float)
    r, t = xrt[..., 1], xrt[..., 2]
    return np.stack((xrt[..., 0], r * np.cos(t), r * np.sin(t)), axis=-1)


def _steps(d, axis):
    """Ratio of each cell length `d` to the next along `axis`, either way up."""
    n = d.shape[axis]
    r = np.take(d, range(1, n), axis) / np.take(d, range(n - 1), axis)
    return np.maximum(r, 1.0 / r)


def test_the_spacing_along_the_blade_does_not_step(grid):
    # Up to four times where a round trailing edge met the surfaces, once.
    for i_row in (0, 1):
        wall = _cartesian(blocks_of_row(grid, i_row)["O"])[:, :, 0]
        d = np.linalg.norm(np.diff(wall, axis=0), axis=-1)
        d = np.concatenate((d[-1:], d, d[:1]))  # round the cut
        assert _steps(d, 0).max() < 2.0, i_row


def test_no_block_steps_sharply_in_any_direction(grid):
    # Across the O block on short lines, and where the blocks behind the
    # blade meet the passage's, up to four times once; and spanwise, where
    # the main span met a clearance held finer by its least count, twice.
    for block in grid:
        xyz = _cartesian(block)
        for axis in range(3):
            d = np.linalg.norm(np.diff(xyz, axis=axis), axis=-1)
            assert _steps(d, axis).max() < 3.0, (block.label, "ijk"[axis])


def test_the_cells_in_the_main_blocks_are_not_badly_skewed(grid):
    for block in grid:
        if "tip" in block.label:
            continue
        rt = block.r * block.t
        worst = min(
            _cell_angles(np.stack((block.x[:, j, :], rt[:, j, :]))).min()
            for j in range(block.shape[1])
        )
        assert worst > 25.0, block.label


def test_the_guess_follows_the_mean_line_in_every_block(machine, grid):
    # Aligned with the grid lines of blocks whose `i` runs across the pitch
    # or round the blade, the guess once took its tangential velocity of
    # either sign there, and started the rotor supersonic.
    guess.apply(grid, machine)
    Vt = guess.meridional(machine, grid[0].fluid).Vt
    lo, hi = Vt.min(), Vt.max()
    slack = 0.15 * (hi - lo)
    for block in grid:
        assert lo - slack < block.Vt.min(), block.label
        assert block.Vt.max() < hi + slack, block.label
        W = np.sqrt(block.Vx**2 + block.Vr**2 + (block.Vt - block.U) ** 2)
        assert (W / block.a).max() < 1.05, block.label
