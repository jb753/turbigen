import ember.block
import ember.grid
import ember.patch
import ember.util
import numpy as np
import pytest
import turbigen_ref.util_post


def row_block(shape, i_le, i_te, di_cusp=None):
    """Create an H row block."""

    # Setup the block geometry
    Nb = 100
    pitch_2 = np.pi / Nb
    xrt = ember.util.linmesh3([0.0, 0.2], [0.9, 1.1], [-pitch_2, pitch_2], shape)

    # Initialise patches
    patches = []

    if di_cusp is not None:
        i_cusp = i_te - di_cusp
        patches.extend(
            [
                ember.patch.CuspPatch(i=(i_cusp, i_te), k=0),
                ember.patch.CuspPatch(i=(i_cusp, i_te), k=-1),
            ]
        )
        i_te += di_cusp  # Downstream periodics now start after cusp

    patches.extend(
        [
            ember.patch.PeriodicPatch(i=(0, i_le), k=0),
            ember.patch.PeriodicPatch(i=(0, i_le), k=-1),
            ember.patch.PeriodicPatch(i=(i_te, -1), k=0),
            ember.patch.PeriodicPatch(i=(i_te, -1), k=-1),
        ]
    )

    # Squeeze theta inside the blade
    x_norm = xrt[i_le : i_te + 1, 0, 0, 0]
    x_norm = (x_norm - x_norm.min()) / (x_norm.max() - x_norm.min())
    frac_squeeze = np.interp(x_norm, [0.0, 0.5, 1.0], [1.0, 0.8, 1.0])
    xrt[i_le : i_te + 1, :, :, 2] *= frac_squeeze[:, None, None]

    block = ember.block.Block(shape=shape)
    block.set_xrt(xrt)
    block.set_Nb(Nb)
    block.patches.extend(patches)

    # Create grid and set periodic connectivity
    grid = ember.grid.Grid([block])
    grid.connectivity.periodic.pair()

    return grid


def test_cut_blade_sides_basic():
    """Test cut_blade_sides on a simple H-mesh grid."""
    from turbigen_ref.util_post import cut_blade_sides

    # Create a single-row H-mesh with shape (30, 20, 10)
    # Leading edge at i=5, trailing edge at i=15
    shape = (30, 20, 10)
    i_le = 5
    i_te = 15
    grid = row_block(shape, i_le, i_te)

    # Call cut_blade_sides
    cuts = cut_blade_sides(grid, offset=0)

    # Should return a list with one element (one row)
    assert len(cuts) == 1

    # The cut should not be None
    assert cuts[0] is not None

    # Should have two sides [Ck0, Cnk]
    sides = cuts[0]
    assert len(sides) == 2

    # Check shape of cuts: should be (i_te - i_le + 1, nj, 1)
    expected_ni = i_te - i_le + 1
    expected_nj = shape[1]
    assert sides[0].shape == (expected_ni, expected_nj, 1)
    assert sides[1].shape == (expected_ni, expected_nj, 1)

    # Verify that one side has had pitch subtracted (theta adjustment)
    # One side should have lower theta values than the other
    t_max_0 = sides[0].t.max()
    t_max_1 = sides[1].t.max()
    assert not np.isclose(t_max_0, t_max_1), "Expected theta adjustment on one side"

    # Check that first i index (leading edge) is coincident on both sides
    # x, r, and t should all match at i=0 for both pressure and suction sides
    np.testing.assert_allclose(sides[0][0, :, 0].x, sides[1][0, :, 0].x, rtol=1e-6)
    np.testing.assert_allclose(sides[0][0, :, 0].r, sides[1][0, :, 0].r, rtol=1e-6)
    np.testing.assert_allclose(sides[0][0, :, 0].t, sides[1][0, :, 0].t, rtol=1e-6)

    # Check that last i index (trailing edge) is coincident on both sides
    # x, r, and t should all match at i=-1 for both pressure and suction sides
    np.testing.assert_allclose(sides[0][-1, :, 0].x, sides[1][-1, :, 0].x, rtol=1e-6)
    np.testing.assert_allclose(sides[0][-1, :, 0].r, sides[1][-1, :, 0].r, rtol=1e-6)
    np.testing.assert_allclose(sides[0][-1, :, 0].t, sides[1][-1, :, 0].t, rtol=1e-6)

    # Check that interior points (i=1 and i=-2) are NOT coincident
    # These are in the blade passage where pressure and suction sides are separated
    # Check x and r should be similar (same meridional position) but not exact
    # Check theta should definitely differ (this is the blade thickness)
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(sides[0][1, :, 0].t, sides[1][1, :, 0].t, rtol=1e-6)
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(
            sides[0][-2, :, 0].t, sides[1][-2, :, 0].t, rtol=1e-6
        )

    # Verify theta actually differs (blade has thickness)
    t_diff_near_le = np.abs(sides[0][1, 0, 0].t - sides[1][1, 0, 0].t)
    t_diff_near_te = np.abs(sides[0][-2, 0, 0].t - sides[1][-2, 0, 0].t)
    assert t_diff_near_le > 1e-6, f"Expected theta separation near LE: {t_diff_near_le}"
    assert t_diff_near_te > 1e-6, f"Expected theta separation near TE: {t_diff_near_te}"


def test_cut_blade_sides_cusped():
    """Test cut_blade_sides on an H-mesh grid with a cusp at trailing edge."""
    from turbigen_ref.util_post import cut_blade_sides

    # Create a single-row H-mesh with a cusp
    # Leading edge at i=5, trailing edge at i=15, cusp extends 2 cells upstream
    shape = (30, 20, 10)
    i_le = 5
    i_te = 15
    di_cusp = 2
    grid = row_block(shape, i_le, i_te, di_cusp=di_cusp)

    # Call cut_blade_sides
    cuts = cut_blade_sides(grid, offset=0)

    # Should return a list with one element (one row)
    assert len(cuts) == 1

    # The cut should not be None
    assert cuts[0] is not None

    # Should have two sides [Ck0, Cnk]
    sides = cuts[0]
    assert len(sides) == 2

    # Both sides should have the same number of points
    assert sides[0].shape == sides[1].shape

    # Check that first i index (leading edge) is coincident on both sides
    # LE should still be a sharp edge with matching coordinates
    np.testing.assert_allclose(sides[0][0, :, 0].x, sides[1][0, :, 0].x, rtol=1e-6)
    np.testing.assert_allclose(sides[0][0, :, 0].r, sides[1][0, :, 0].r, rtol=1e-6)
    np.testing.assert_allclose(sides[0][0, :, 0].t, sides[1][0, :, 0].t, rtol=1e-6)

    # Check that i=1 (near LE interior) is NOT coincident
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(sides[0][1, :, 0].t, sides[1][1, :, 0].t, rtol=1e-6)

    # Verify theta differs at i=1
    t_diff_near_le = np.abs(sides[0][1, 0, 0].t - sides[1][1, 0, 0].t)
    assert t_diff_near_le > 1e-6, f"Expected theta separation near LE: {t_diff_near_le}"

    # Check that last i index (trailing edge/cusp) is NOT coincident
    # The cusp creates a separation between the two sides at the TE
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(
            sides[0][-1, :, 0].t, sides[1][-1, :, 0].t, rtol=1e-6
        )

    # Verify theta differs significantly at the cusp
    t_diff_at_cusp = np.abs(sides[0][-1, 0, 0].t - sides[1][-1, 0, 0].t)
    assert t_diff_at_cusp > 1e-6, f"Expected theta separation at cusp: {t_diff_at_cusp}"


def test_cut_blade_surfs_basic():
    """Test cut_blade_surfs on a simple H-mesh grid."""
    from turbigen_ref.util_post import cut_blade_surfs

    # Create a single-row H-mesh with shape (30, 20, 10)
    # Leading edge at i=5, trailing edge at i=15
    shape = (30, 20, 10)
    i_le = 5
    i_te = 15
    grid = row_block(shape, i_le, i_te)

    # Call cut_blade_surfs
    surfs = cut_blade_surfs(grid, offset=0)

    # Should return a list with one element (one row)
    assert len(surfs) == 1

    # The surf should not be None
    assert surfs[0] is not None

    # For H-mesh, should have one surface per row (list of one element)
    assert len(surfs[0]) == 1

    # Get the concatenated surface
    surf = surfs[0][0]

    # Shape should be (2 * (i_te - i_le + 1) - 1, nj, 1)
    # Because we concatenate both sides with flip, removing duplicate first point
    # sides[0].flip(axis=0) has length (i_te - i_le + 1)
    # sides[1][1:, ...] has length (i_te - i_le)
    # Total: (i_te - i_le + 1) + (i_te - i_le) = 2*(i_te - i_le + 1) - 1
    expected_ni = 2 * (i_te - i_le + 1) - 1
    expected_nj = shape[1]
    assert surf.shape == (expected_ni, expected_nj, 1)

    # Check that the surface forms a loop - first and last i should be at TE
    # After flip and concatenate: starts at TE (from side 0), goes to LE, returns to TE (from side 1)
    # After pitch adjustment in cut_blade_sides, TE is coincident in all coordinates
    np.testing.assert_allclose(surf[0, :, 0].x, surf[-1, :, 0].x, rtol=1e-6)
    np.testing.assert_allclose(surf[0, :, 0].r, surf[-1, :, 0].r, rtol=1e-6)
    np.testing.assert_allclose(surf[0, :, 0].t, surf[-1, :, 0].t, rtol=1e-6)

    # Check that theta varies along the blade surface (not constant)
    # The middle of each side should have different theta values
    # Find approximate quarter points (should be in middle of each side)
    i_quarter = surf.shape[0] // 4
    i_three_quarter = 3 * surf.shape[0] // 4
    t_quarter = surf[i_quarter, 0, 0].t
    t_three_quarter = surf[i_three_quarter, 0, 0].t
    # They should differ (blade has thickness in theta direction)
    assert not np.isclose(
        t_quarter, t_three_quarter
    ), f"Expected theta variation along blade: {t_quarter} vs {t_three_quarter}"


if __name__ == "__main__":
    blk = row_block((30, 20, 10), 5, 15, di_cusp=2)
    import matplotlib.pyplot as plt

    C01 = turbigen_ref.util_post.cut_blade_sides(blk)[0]

    fig, ax = plt.subplots()
    for C in C01:
        C = C.squeeze()[:, 0]
        ax.plot(C.x, C.r * C.t, "k-x", lw=0.5)
        ax.plot(C.x.T, (C.r * C.t).T, "k-x", lw=0.5)
    ax.axis("equal")
    plt.show()
