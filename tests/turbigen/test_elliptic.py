"""Tests for elliptic smoothing of a plane of blocks.

Analytic grids only: a uniform mesh is a solution of the equations with no
control, so it is both a fixed point and what a disturbed mesh with uniform
boundaries returns to. That separates the numerics from any turbomachinery.
"""

import numpy as np
import pytest

from turbigen import elliptic


def rectangle(ni, nk, x0=0.0, x1=1.0, y0=0.0, y1=1.0):
    """Uniform block on a rectangle, shape (2, ni, nk)."""
    x = np.linspace(x0, x1, ni)
    y = np.linspace(y0, y1, nk)
    return np.stack(np.meshgrid(x, y, indexing="ij"))


def edges(shape):
    """Boolean mask of the boundary nodes of a block."""
    mask = np.zeros(shape, dtype=bool)
    mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = True
    return mask


def disturb(xy, mask, amp, seed=0):
    """Return `xy` with nodes where `mask` is set moved at random."""
    rng = np.random.default_rng(seed)
    out = xy.copy()
    out[:, mask] += amp * rng.uniform(-1.0, 1.0, (2, mask.sum()))
    return out


def test_a_uniform_rectangle_is_a_fixed_point():
    xy = rectangle(17, 9)
    topo = elliptic.PlaneTopology([xy], fixed=[edges(xy.shape[1:])])
    X0 = topo.gather([xy])
    X, _, n_iter = elliptic.smooth(topo, X0, scale=1.0 / 16, tol=1e-12)
    assert np.allclose(X, X0, atol=1e-12)
    assert n_iter == 1


def test_a_disturbed_interior_returns_to_uniform():
    xy = rectangle(17, 9)
    interior = ~edges(xy.shape[1:])
    topo = elliptic.PlaneTopology([xy], fixed=[~interior])
    X0 = topo.gather([disturb(xy, interior, 0.02)])
    X, _, _ = elliptic.smooth(topo, X0, scale=1.0 / 16, tol=1e-10, max_iter=500)
    assert np.allclose(topo.scatter(X)[0], xy, atol=1e-8)


def test_a_floating_interface_between_blocks_returns_to_uniform():
    # Two blocks side by side, meeting at x = 0.5, with the interface free.
    left = rectangle(9, 9, 0.0, 0.5)
    right = rectangle(9, 9, 0.5, 1.0)
    fixed = []
    for blk, face in ((left, 0), (right, -1)):
        mask = np.zeros(blk.shape[1:], dtype=bool)
        mask[face, :] = mask[:, 0] = mask[:, -1] = True
        fixed.append(mask)
    topo = elliptic.PlaneTopology([left, right], fixed=fixed)
    assert len(topo.regular) == 15 * 7

    # Move the interface nodes, which belong to both blocks at once.
    X0 = topo.gather([left, right])
    interface = topo.gid[0][-1, 1:-1]
    assert np.array_equal(interface, topo.gid[1][0, 1:-1])
    rng = np.random.default_rng(0)
    X0[interface] += 0.02 * rng.uniform(-1.0, 1.0, (len(interface), 2))

    X, _, _ = elliptic.smooth(topo, X0, scale=1.0 / 16, tol=1e-10, max_iter=500)
    out = topo.scatter(X)
    assert np.allclose(out[0], left, atol=1e-8)
    assert np.allclose(out[1], right, atol=1e-8)


def test_cell_areas_are_positive_and_flip_when_a_cell_folds():
    xy = rectangle(5, 3, 0.0, 4.0, 0.0, 2.0)
    topo = elliptic.PlaneTopology([xy], fixed=[edges(xy.shape[1:])])
    (area,) = elliptic.cell_areas(topo, topo.gather([xy]))
    assert area.shape == (4, 2)
    assert np.allclose(area, 1.0)

    # Push an interior node past its neighbours to fold the cell after it.
    folded = xy.copy()
    folded[0, 2, 1] = 5.0
    (area,) = elliptic.cell_areas(topo, topo.gather([folded]))
    assert (area[2:, :] < 0.0).any()


@pytest.mark.parametrize("shear", [0.0, 0.3])
def test_a_periodic_channel_with_sliding_ends_returns_to_uniform(shear):
    # A channel periodic in y, its ends sliding on lines of constant x. With a
    # shear the starting mesh is skewed, and the answer is still uniform.
    pitch = 0.5
    xy = rectangle(17, 9, 0.0, 1.0, 0.0, pitch)
    slide = np.zeros(xy.shape[1:], dtype=bool)
    slide[(0, -1), :] = True
    topo = elliptic.PlaneTopology([xy], pitch=pitch, slide=[slide])

    # The two periodic faces are one set of nodes.
    assert topo.n_node == 17 * 8
    assert len(topo.irregular) == 0

    start = xy.copy()
    start[1] += shear * np.sin(np.pi * start[0])
    start = disturb(start, ~slide, 0.01)
    start[1, :, -1] = start[1, :, 0] + pitch
    start[0, :, -1] = start[0, :, 0]

    X, _, _ = elliptic.smooth(
        topo, topo.gather([start]), scale=1.0 / 16, tol=1e-10, max_iter=500
    )
    out = topo.scatter(X)[0]

    # The ends kept their x, and the periodic faces stayed a pitch apart.
    assert np.allclose(out[0, (0, -1)], xy[0, (0, -1)])
    assert np.allclose(out[1, :, -1] - out[1, :, 0], pitch)

    # Grid lines leave the ends square to them.
    assert np.allclose(out[1, 0], out[1, 1], atol=1e-8)
    assert np.allclose(out[1, -1], out[1, -2], atol=1e-8)

    # Uniform again, to within a rigid shift in y, which nothing pins.
    dy = out[1] - xy[1]
    assert np.allclose(out[0], xy[0], atol=1e-8)
    assert np.allclose(dy, dy.mean(), atol=1e-8)


def test_sliding_ends_can_keep_the_starting_angle():
    # The same channel started sheared, with its ends told to keep the offset
    # from the line inside them. A uniform shear is an affine map, which the
    # equations hold exactly, so it comes back sheared rather than square.
    pitch = 0.5
    shear = 0.4
    xy = rectangle(17, 9, 0.0, 1.0, 0.0, pitch)
    slide = np.zeros(xy.shape[1:], dtype=bool)
    slide[(0, -1), :] = True
    topo = elliptic.PlaneTopology([xy], pitch=pitch, slide=[slide])

    sheared = xy.copy()
    sheared[1] += shear * sheared[0]
    # Nothing pins y here, so the held offsets act as Neumann data, and those
    # at the two ends have to balance for the plane to have an answer at all.
    # Leaving the lines inside the ends undisturbed makes them the shear's own.
    inner = np.zeros_like(slide)
    inner[2:-2, :] = True
    start = disturb(sheared, inner, 0.01)
    start[1, :, -1] = start[1, :, 0] + pitch
    start[0, :, -1] = start[0, :, 0]

    X, _, _ = elliptic.smooth(
        topo,
        topo.gather([start]),
        scale=1.0 / 16,
        tol=1e-10,
        max_iter=500,
        keep_slide_angle=True,
    )
    out = topo.scatter(X)[0]

    # The ends kept their x and their offset from the line inside.
    assert np.allclose(out[0, (0, -1)], xy[0, (0, -1)])
    assert np.allclose(out[1, 0] - out[1, 1], start[1, 0] - start[1, 1], atol=1e-8)
    assert np.allclose(out[1, -1] - out[1, -2], start[1, -1] - start[1, -2], atol=1e-8)

    # The uniform shear, to within a rigid shift in y.
    dy = out[1] - sheared[1]
    assert np.allclose(out[0], xy[0], atol=1e-8)
    assert np.allclose(dy, dy.mean(), atol=1e-8)


def annulus(ni, nk, r0, r1, spacing=None):
    """O block round a circle of radius `r0`, i clockwise, k outward."""
    phi = -np.linspace(0.0, 2.0 * np.pi, ni)
    if spacing is None:
        rad = np.linspace(r0, r1, nk)
    else:
        # Geometric from the wall, to give the smoother a fair starting mesh.
        n = nk - 1
        ER = 1.15
        d = spacing * ER ** np.arange(n)
        rad = r0 + (r1 - r0) * np.concatenate(([0.0], np.cumsum(d))) / d.sum()
    R, PHI = np.meshgrid(rad, phi)
    return np.stack((R * np.cos(PHI), R * np.sin(PHI)))


def test_an_o_grid_cut_is_one_set_of_nodes():
    xy = annulus(33, 9, 1.0, 2.0)
    fixed = np.zeros(xy.shape[1:], dtype=bool)
    fixed[:, (0, -1)] = True
    topo = elliptic.PlaneTopology([xy], fixed=[fixed])
    assert topo.n_node == 32 * 9
    assert len(topo.regular) == 32 * 7


def test_wall_control_sets_spacing_and_angle_on_a_cylinder():
    ni, nk = 65, 25
    d_wall = 0.01
    xy = annulus(ni, nk, 1.0, 3.0, spacing=0.03)

    # Skew the interior so that there is an angle at the wall to correct.
    rad = np.hypot(*xy)
    twist = 0.3 * (rad - 1.0) * (3.0 - rad)
    c, s = np.cos(twist), np.sin(twist)
    xy = np.stack((c * xy[0] - s * xy[1], s * xy[0] + c * xy[1]))

    fixed = np.zeros(xy.shape[1:], dtype=bool)
    fixed[:, (0, -1)] = True
    control = elliptic.WallControl(block=0, spacing=np.full(ni, d_wall), closed=True)
    topo = elliptic.PlaneTopology([xy], fixed=[fixed], control=control)
    X, pq, _ = elliptic.smooth(
        topo, topo.gather([xy]), scale=d_wall, tol=1e-4, max_iter=400
    )
    out = topo.scatter(X)[0]

    wall = out[:, :-1, 0]
    first = out[:, :-1, 1]
    off = first - wall
    height = np.linalg.norm(off, axis=0)
    assert np.allclose(height, d_wall, rtol=0.05)

    # Angle between the first line off the wall and the radial direction.
    radial = wall / np.linalg.norm(wall, axis=0)
    cosang = np.sum(off * radial, axis=0) / height
    angle = np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))
    assert angle.max() < 2.0

    assert pq.shape == (2, ni)
