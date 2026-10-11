"""Tests for the topology and local optimisation of a plane of blocks.

Analytic grids only, which separates the numerics from any turbomachinery.
"""

import numpy as np

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


def annulus(ni, nk, r0, r1):
    """O block round a circle of radius `r0`, i clockwise, k outward."""
    phi = -np.linspace(0.0, 2.0 * np.pi, ni)
    rad = np.linspace(r0, r1, nk)
    R, PHI = np.meshgrid(rad, phi)
    return np.stack((R * np.cos(PHI), R * np.sin(PHI)))


def test_an_o_grid_cut_is_one_set_of_nodes():
    xy = annulus(33, 9, 1.0, 2.0)
    fixed = np.zeros(xy.shape[1:], dtype=bool)
    fixed[:, (0, -1)] = True
    topo = elliptic.PlaneTopology([xy], fixed=[fixed])
    assert topo.n_node == 32 * 9
    assert len(topo.regular) == 32 * 7


def _min_angle(blk):
    """Smallest corner angle of any cell, degrees."""
    p = blk.transpose(1, 2, 0)
    c = [p[:-1, :-1], p[1:, :-1], p[1:, 1:], p[:-1, 1:]]
    out = 180.0
    for a in range(4):
        u = c[(a + 1) % 4] - c[a]
        v = c[(a - 1) % 4] - c[a]
        cs = np.sum(u * v, -1) / np.linalg.norm(u, axis=-1) / np.linalg.norm(v, axis=-1)
        out = min(out, np.degrees(np.arccos(np.clip(cs, -1, 1))).min())
    return out


def test_polish_squares_up_cells_near_a_node_and_leaves_the_rest():
    xy = rectangle(17, 17)
    interior = ~edges(xy.shape[1:])
    bad = disturb(xy, interior, 0.025)
    topo = elliptic.PlaneTopology([bad], fixed=[~interior])
    X0 = topo.gather([bad])
    centre = topo.gid[0][8, 8]
    # Targets from the uniform mesh, as the algebraic mesh is in a mesher.
    X = elliptic.polish(topo, X0, [centre], rings=3, X_target=topo.gather([xy]))
    moved = np.abs(X - X0).max(axis=1) > 0.0
    near = np.zeros(xy.shape[1:], dtype=bool)
    near[5:12, 5:12] = True
    # Only nodes within three cells of the centre move, and none fixed.
    assert not moved[topo.gid[0][~near]].any()
    assert not moved[topo.fixed].any()
    window = (slice(None), slice(5, 12), slice(5, 12))
    assert _min_angle(topo.scatter(X)[0][window]) > _min_angle(bad[window]) + 10.0


def test_a_higher_polish_power_works_on_the_worst_cell():
    xy = rectangle(17, 17)
    interior = ~edges(xy.shape[1:])
    bad = disturb(xy, interior, 0.025, seed=1)
    topo = elliptic.PlaneTopology([bad], fixed=[~interior])
    X0 = topo.gather([bad])
    target = topo.gather([xy])
    centre = [topo.gid[0][8, 8]]
    plain, worst_first = (
        elliptic.polish(topo, X0, centre, rings=8, X_target=target, power=p)
        for p in (1.0, 4.0)
    )
    assert _min_angle(topo.scatter(worst_first)[0]) >= _min_angle(
        topo.scatter(plain)[0]
    )
