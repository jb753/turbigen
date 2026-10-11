"""Topology and local optimisation of a plane of structured blocks.

A set of two-dimensional structured blocks that tile a region of a plane, some
of whose faces meet each other and some of which are periodic, is treated as
one mesh. Nothing here knows about blades. The coordinates are a pair
`(x, y)`, with `y` optionally periodic over `pitch`, and a mesher hands over
blocks and says which nodes are pinned and which slide along `y`.

Block faces that meet are not boundaries. The blocks are merged into one set
of nodes, and every node whose surroundings are four quadrilaterals gets a
nine-point stencil assembled from those four cells, whichever blocks they
belong to and whichever side of a periodic boundary, which the compiled
Winslow sweeps of :func:`ember.elliptic.winslow_sweeps` smooth with. So
interfaces and periodic lines float: a block boundary is where the bookkeeping
changes, not where the mesh does. A node with other than four cells round it,
the corner of an O block in an H block for instance, is a singular point, put
where the parallelograms of the cells round it would put it.

What the elliptic equations leave poor round the singular points is improved
by :func:`polish`, a local target-matrix optimisation.
"""

import logging

import ember.elliptic
import numpy as np
import scipy.optimize
import scipy.sparse
import scipy.sparse.csgraph
from scipy.spatial import cKDTree

logger = logging.getLogger("turbigen")

MERGE_RTOL = 1e-9
"""Distance below which two nodes are the same, as a fraction of the extent."""

MERGE_ULP = 16
"""Distance below which two nodes are the same, in units of the rounding of
the coordinates' precision at their size, where that is the larger: in single
precision, a node shifted by a pitch rounds off its partner by about one."""


class PlaneTopology:
    """Node numbering and stencils for a set of blocks, fixed once.

    Built from one instance of the coordinates, which fixes which nodes are the
    same node and which cells surround each. Every plane of a mesh with the same
    block structure reuses it, so the setup cost is paid once per topology
    rather than once per plane.

    Parameters
    ----------
    xy : list of ndarray
        Coordinates of each block, shape (2, ni, nk). Nodes shared between
        blocks must coincide, or be one `pitch` apart in `y`.
    pitch : float, optional
        Period in `y`. None for a mesh with no periodic boundary.
    fixed : list of ndarray, optional
        Per block, a boolean mask (ni, nk) of nodes that do not move.
    slide : list of ndarray, optional
        Per block, a boolean mask (ni, nk) of nodes on a boundary of constant
        `x`, which move in `y` only, to sit level with their neighbour inside
        the mesh.
    stencils : bool, optional
        Whether to classify the nodes and assemble the stencils the sweeps
        need. Without them, the topology numbers the nodes and flags the
        fixed and sliding ones, which is all :func:`polish` needs, much
        sooner.

    """

    def __init__(self, xy, pitch=None, fixed=None, slide=None, stencils=True):
        self.shapes = [blk.shape[1:] for blk in xy]
        self.pitch = pitch

        points = np.concatenate([blk.reshape(2, -1).T for blk in xy])
        self._offsets = np.cumsum([0] + [ni * nk for ni, nk in self.shapes])

        gid, shift, rep = self._merge(points)
        self.n_node = len(rep)
        self.gid = self._split(gid)
        self.shift = self._split(shift)

        is_fixed = self._flag(fixed)
        is_slide = self._flag(slide) & ~is_fixed

        if stencils:
            self._build_stencils(is_fixed, is_slide)
        else:
            self.fixed = np.flatnonzero(is_fixed)
            self.slide = np.flatnonzero(is_slide)

    #
    # SETUP
    #

    def _split(self, flat):
        """Return a flat per-instance array as one (ni, nk) array per block."""
        return [
            flat[self._offsets[b] : self._offsets[b + 1]].reshape(shape)
            for b, shape in enumerate(self.shapes)
        ]

    def _flag(self, masks):
        """Return a per-node flag set wherever any instance of the node is."""
        flag = np.zeros(self.n_node, dtype=bool)
        if masks is None:
            return flag
        for b, mask in enumerate(masks):
            if mask is not None:
                flag[self.gid[b][mask]] = True
        return flag

    def _merge(self, points):
        """Number the distinct nodes and the pitch shift of each instance.

        Two instances are the same node if they coincide, or if they are one
        pitch apart in `y`. Connected components of that relation are the
        nodes, and the shift of each instance is counted from the first
        instance of its node.
        """
        n = len(points)
        tol = max(
            MERGE_RTOL * np.ptp(points, axis=0).max(),
            MERGE_ULP * np.finfo(points.dtype).eps * np.abs(points).max(),
        )
        tree = cKDTree(points)

        pairs = [tree.query_pairs(tol, output_type="ndarray")]
        if self.pitch:
            dist, j = tree.query(points + [0.0, self.pitch], distance_upper_bound=tol)
            hit = np.isfinite(dist)
            pairs.append(np.stack((np.flatnonzero(hit), j[hit]), axis=1))
        pairs = np.concatenate(pairs)

        graph = scipy.sparse.coo_matrix(
            (np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n)
        )
        _, label = scipy.sparse.csgraph.connected_components(graph, directed=False)

        # Renumber so the nodes are in order of first appearance, and take each
        # node's coordinates from its first instance.
        _, first, gid = np.unique(label, return_index=True, return_inverse=True)
        order = np.argsort(first)
        rank = np.empty_like(order)
        rank[order] = np.arange(len(order))
        gid = rank[gid]
        rep = first[order]

        if self.pitch:
            shift = np.round((points[:, 1] - points[rep[gid], 1]) / self.pitch)
            shift = shift.astype(int)
        else:
            shift = np.zeros(n, dtype=int)

        err = points - points[rep[gid]]
        err[:, 1] -= shift * (self.pitch or 0.0)
        if np.abs(err).max() > tol:
            raise ValueError(
                "Merged nodes do not coincide: a node is the same as itself "
                "shifted by more than one route."
            )

        return gid, shift, rep

    def _build_stencils(self, is_fixed, is_slide):
        """Classify every node, and assemble the stencil of each regular one."""

        # Every cell, its corners counter-clockwise in index space.
        cg, cs = [], []
        for b, (ni, nk) in enumerate(self.shapes):
            g, s = self.gid[b], self.shift[b]
            corners = [(0, 0), (1, 0), (1, 1), (0, 1)]
            cg.append(
                np.stack(
                    [g[di : ni - 1 + di, dk : nk - 1 + dk] for di, dk in corners],
                    axis=-1,
                ).reshape(-1, 4)
            )
            cs.append(
                np.stack(
                    [s[di : ni - 1 + di, dk : nk - 1 + dk] for di, dk in corners],
                    axis=-1,
                ).reshape(-1, 4)
            )
        cg = np.concatenate(cg)
        cs = np.concatenate(cs)
        n_cell = len(cg)

        # One entry per (cell, corner): the node at that corner, and the next,
        # opposite and previous corners as seen from it, with their shifts
        # relative to it. Going round the node counter-clockwise, a cell's
        # previous corner is the next cell's next corner.
        rows = []
        for r in range(4):
            node = cg[:, r]
            nb = [cg[:, (r + d) % 4] for d in (1, 2, 3)]
            rs = [cs[:, (r + d) % 4] - cs[:, r] for d in (1, 2, 3)]
            rows.append(
                np.stack(
                    [node, *nb, *rs, np.full(n_cell, r), np.arange(n_cell)], axis=1
                )
            )
        ent = np.concatenate(rows)

        # Sorted by node, and within a node with the preferred starting cell
        # first: one the node is the origin corner of, earliest block first.
        # That cell's next corner is then the node's +i neighbour in its own
        # block, so a node inside a block gets a stencil in that block's
        # orientation.
        ent = ent[np.lexsort((ent[:, 8], ent[:, 7] != 0, ent[:, 0]))]
        count = np.bincount(ent[:, 0], minlength=self.n_node)
        start = np.concatenate(([0], np.cumsum(count)))

        stencil_g, stencil_s, regular = [], [], []
        irregular, irregular_nb = [], []
        slide_nodes, slide_nb = [], []

        for node in range(self.n_node):
            e = ent[start[node] : start[node + 1]]
            nxt = [(row[1], row[4]) for row in e]
            prv = [(row[3], row[6]) for row in e]
            edge_nb = sorted(set(nxt) | set(prv))

            if is_fixed[node]:
                continue

            if is_slide[node]:
                inward = _crossing(nxt, prv, is_slide)
                if not inward:
                    inward = [nb for nb in edge_nb if not is_slide[nb[0]]]
                if not inward:
                    raise ValueError(f"Sliding node {node} has no neighbour inside.")
                slide_nodes.append(node)
                slide_nb.append(inward)
                continue

            closed = sorted(nxt) == sorted(prv)
            if not closed:
                raise ValueError(
                    f"Node {node} is on the edge of the mesh but neither fixed "
                    f"nor sliding."
                )

            if len(e) != 4:
                irregular.append(node)
                irregular_nb.append(_parallelogram(e))
                continue

            # Walk round the node counter-clockwise from the starting cell.
            order = [0]
            for _ in range(3):
                want = prv[order[-1]]
                order.append(nxt.index(want))
            if prv[order[-1]] != nxt[0]:
                raise ValueError(f"Cells round node {node} do not close.")

            g = [node]
            s = [0]
            for q in order:
                g += [e[q, 1], e[q, 2]]
                s += [e[q, 4], e[q, 5]]
            stencil_g.append(g)
            stencil_s.append(s)
            regular.append(node)

        self.regular = np.array(regular, dtype=int)
        self.stencil_g = np.array(stencil_g, dtype=int).reshape(-1, 9)
        self.stencil_s = np.array(stencil_s, dtype=int).reshape(-1, 9)

        self.irregular = np.array(irregular, dtype=int)
        self.irregular_nb = irregular_nb
        self.slide = np.array(slide_nodes, dtype=int)
        self.slide_nb = slide_nb
        self.fixed = np.flatnonzero(is_fixed)

        logger.debug(
            f"Plane topology: {self.n_node} nodes, {len(self.regular)} regular, "
            f"{len(self.irregular)} irregular, {len(self.slide)} sliding, "
            f"{len(self.fixed)} fixed."
        )

    #
    # COORDINATES
    #

    def gather(self, xy):
        """Return node coordinates (n_node, 2) from per-block coordinates, in
        their precision."""
        X = np.empty((self.n_node, 2), dtype=np.result_type(*xy))
        for b, blk in enumerate(xy):
            pts = blk.reshape(2, -1).T.copy()
            if self.pitch:
                pts[:, 1] -= self.shift[b].ravel() * self.pitch
            X[self.gid[b].ravel()] = pts
        return X

    def scatter(self, X):
        """Return per-block coordinates (2, ni, nk) from node coordinates."""
        out = []
        for b in range(len(self.shapes)):
            blk = X[self.gid[b]].copy()
            if self.pitch:
                blk[..., 1] += self.shift[b] * self.pitch
            out.append(np.moveaxis(blk, -1, 0))
        return out

    def _values(self, X, g, s):
        """Return coordinates of nodes `g` as seen with shifts `s`."""
        V = X[g].copy()
        if self.pitch:
            V[..., 1] += s * self.pitch
        return V


def _crossing(nxt, prv, is_slide):
    """Return the neighbours on the middle grid line(s) into a boundary node.

    The cells round a node on the edge of the mesh form an open fan, from the
    boundary on one side to the boundary on the other. The edges between
    consecutive cells are the grid lines that come into the node from inside;
    the middle one of them, or the middle two, are those that would meet the
    boundary square were the cells equal. Taking only these, rather than every
    neighbour off the boundary, keeps a node where the boundary has a corner
    from being pulled along the grid lines that run beside it. Empty, leaving
    the node to every neighbour off the boundary, unless at least two grid
    lines come in: an ordinary boundary node has one, and next to a fixed node
    the fixed neighbour is what holds the first cell open. Also empty when the
    middle line leads to another sliding node.
    """
    # The successor of a cell, going counter-clockwise, is the one whose next
    # corner is its previous corner. The fan starts at the cell with none
    # before it.
    n = len(nxt)
    order = [next(a for a in range(n) if nxt[a] not in prv)]
    while prv[order[-1]] in nxt and len(order) < n:
        order.append(nxt.index(prv[order[-1]]))
    lines = [prv[a] for a in order[:-1]]
    if len(lines) < 2:
        return []
    mid = len(lines) // 2
    pick = [lines[mid]] if len(lines) % 2 else lines[mid - 1 : mid + 1]
    if any(is_slide[g] for g, _ in pick):
        return []
    return pick


def _parallelogram(e):
    """Return the weighted neighbours (g, shift, w) of a singular point.

    Each cell round the point would put it at the sum of its two edge
    neighbours less the opposite corner, were the cell a parallelogram; the
    point sits at the mean of those. For n cells that tends to corners of 360/n
    degrees, where the mean of the neighbours alone would pull the point
    towards whichever of them lie furthest away.
    """
    m = len(e)
    w = {}
    for row in e:
        for g, s, sign in (
            (row[1], row[4], 1.0),
            (row[3], row[6], 1.0),
            (row[2], row[5], -1.0),
        ):
            w[(g, s)] = w.get((g, s), 0.0) + sign / m
    return [(g, s, wt) for (g, s), wt in sorted(w.items())]


#
# LOCAL OPTIMISATION
#

POLISH_EPS = 1e-3
"""Regularisation of the corner Jacobian in :func:`polish`, so a folded cell
has a finite, steep cost rather than none."""


def polish(
    topo,
    X,
    centres,
    rings=4,
    w_size=1.0,
    max_iter=500,
    X_target=None,
    power=1.0,
    hold=None,
):
    """Return `X` with the nodes near `centres` moved to square up their cells.

    The elliptic equations leave poor cells round the singular points of a
    multi-block mesh, where more or fewer than four cells meet and no
    neighbour stencil can be satisfied by right angles. Here the nodes within
    `rings` cells of each centre are moved to minimise, over every cell they
    touch, a measure of each cell corner against a target --- the
    target-matrix optimisation of Knupp, a discrete form of Brackbill and
    Saltzman's variational grid generation. At a corner with edges `a` along
    `i` and `b` along `k`, and target lengths `la` and `lb` read off the mesh
    as it is, `T = [a / la, b / lb]`; the shape term `|T|^2 / (2 det T)` is
    least for edges square to each other in the target ratio, and the size
    term `(det T + 1 / det T) / 2` for the target area. Only the shape of the
    cells changes, then, not the spacing the mesh was built with.

    Fixed nodes stay put and sliding nodes move only in `y`. The rest of the
    mesh is untouched, so the elliptic solution elsewhere is kept exactly.

    Parameters
    ----------
    topo : PlaneTopology
        Structure of the plane.
    X : ndarray
        Node coordinates, shape (n_node, 2).
    centres : array_like of int
        Nodes to polish round, typically ``topo.irregular``.
    rings : int
        How many cells out from each centre nodes are moved.
    w_size : float
        Weight of the size term against the shape term.
    max_iter : int
        Most iterations of the optimiser.
    X_target : ndarray, optional
        Mesh to read the target edge lengths off, shape (n_node, 2); `X` if
        not given.
    power : float
        Power each corner's shape term is raised to. Above one, the worst
        corners dominate the sum, so the optimiser works on them rather than
        trading them for small gains everywhere else.
    hold : array_like of int, optional
        Nodes to leave where they are, as well as the fixed ones.

    Returns
    -------
    ndarray
        Polished node coordinates, shape (n_node, 2).

    """
    pitch = topo.pitch or 0.0

    # Every cell as its four corner nodes and their periodic shifts, in the
    # order (i, k), (i + 1, k), (i + 1, k + 1), (i, k + 1).
    ids, shifts = [], []
    for g, s in zip(topo.gid, topo.shift):
        corners = [
            (slice(None, -1), slice(None, -1)),
            (slice(1, None), slice(None, -1)),
            (slice(1, None), slice(1, None)),
            (slice(None, -1), slice(1, None)),
        ]
        ids.append(np.stack([g[c].ravel() for c in corners], axis=1))
        shifts.append(np.stack([s[c].ravel() for c in corners], axis=1))
    ids = np.concatenate(ids)
    shifts = np.concatenate(shifts).astype(float) * pitch

    # Nodes within `rings` cells of a centre, by growing through the cells.
    near = np.zeros(topo.n_node, dtype=bool)
    near[np.asarray(centres, dtype=int)] = True
    for _ in range(rings):
        near[ids[near[ids].any(axis=1)].ravel()] = True
    free = np.zeros((topo.n_node, 2), dtype=bool)
    free[near] = True
    free[topo.fixed] = False
    free[topo.slide, 0] = False
    if hold is not None:
        free[np.asarray(hold, dtype=int)] = False
    if not free.any():
        return X
    cells = near[ids].any(axis=1)
    ids, shifts = ids[cells], shifts[cells]

    def corners(Xc):
        P = Xc[ids].copy()  # (n_cell, 4, 2)
        P[..., 1] += shifts
        # (node, along i, along k) at each corner, as offsets
        return [
            (P[:, 1] - P[:, 0], P[:, 3] - P[:, 0]),
            (P[:, 1] - P[:, 0], P[:, 2] - P[:, 1]),
            (P[:, 2] - P[:, 3], P[:, 2] - P[:, 1]),
            (P[:, 2] - P[:, 3], P[:, 3] - P[:, 0]),
        ]

    # Edge lengths of the mesh as it is, the mean of the cell's two edges in
    # each direction, as the targets.
    a0 = corners(X if X_target is None else X_target)
    la = 0.5 * (np.linalg.norm(a0[0][0], axis=1) + np.linalg.norm(a0[2][0], axis=1))
    lb = 0.5 * (np.linalg.norm(a0[0][1], axis=1) + np.linalg.norm(a0[1][1], axis=1))
    scale = np.sqrt(np.mean(la * lb))
    n_corner = 4.0 * len(ids)
    X0 = X.copy()

    # The cost and its gradient by ember's compiled kernel, in double
    # precision whatever the mesh is held in.
    X64 = X.astype(float)

    def fun(z):
        Xc = X64.copy()
        Xc[free] = z * scale
        f, g = ember.elliptic.polish_cost(
            Xc, ids, shifts, la, lb, POLISH_EPS, power, w_size
        )
        return f / n_corner, g[free] * scale / n_corner

    res = scipy.optimize.minimize(
        fun,
        X0[free] / scale,
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": max_iter, "gtol": 1e-10, "ftol": 0.0},
    )
    out = X0.copy()
    out[free] = res.x * scale
    logger.debug(
        f"Polished {free.any(axis=1).sum()} nodes in {res.nit} iterations: "
        f"{fun(X0[free] / scale)[0]:.4g} to {res.fun:.4g}, {res.message}"
    )
    return out
