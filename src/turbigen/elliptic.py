"""Elliptic smoothing of a plane of structured blocks.

A set of two-dimensional structured blocks that tile a region of a plane, some
of whose faces meet each other and some of which are periodic, is smoothed as
one mesh by solving the Thompson-Thames-Mastin equations in the form of Thomas
and Middlecoff,

    alpha (r_xixi + P r_xi) - 2 beta r_xieta + gamma (r_etaeta + Q r_eta) = 0

with `alpha = |r_eta|^2`, `beta = r_xi . r_eta`, `gamma = |r_xi|^2`, and
control functions `P`, `Q` that pull the mesh onto a wall at a prescribed
spacing and angle. They are set at the wall by feedback on the spacing and
angle the mesh actually has, after Hilgenstock and White, and decay
exponentially away from it.

This form rather than Sorenson's, which carries the sources as
`-J^2 (P r_xi + Q r_eta)`, because here `P` and `Q` are dimensionless: in one
dimension `Q = -r_etaeta / r_eta`, the logarithm of the local expansion ratio
of the mesh. So one limit and one decay rate mean the same on any mesh, where
the other form's sources scale with the square of the number of nodes per unit
length and would need retuning with resolution.

Nothing here knows about blades. The coordinates are a pair `(x, y)`, with `y`
optionally periodic over `pitch`, and a mesher hands over blocks, says which
nodes are pinned, which slide along `y`, and which face is the wall. That
keeps the numerical part testable on analytic grids, and reusable on a plane
of different topology --- a tip gap, say --- without change.

Block faces that meet are not boundaries. The blocks are merged into one set
of nodes and every node whose surroundings are four quadrilaterals is
smoothed with a nine-point stencil assembled from those four cells, whichever
blocks they belong to and whichever side of a periodic boundary. So interfaces
and periodic lines float: a block boundary is where the bookkeeping changes,
not where the mesh does. A node with other than four cells round it, the
corner of an O block in an H block for instance, is put at the mean of its
neighbours, which is the usual treatment of a singular point.

The equations are solved by Picard iteration: coefficients and source terms are
frozen at the current mesh, the linear nine-point system is solved directly for
each coordinate, and the change is under-relaxed. A direct solve converges the
whole plane in tens of iterations rather than the thousands a point iteration
takes, which is what makes smoothing every spanwise plane affordable.
"""

import dataclasses
import logging

import ember.elliptic
import numpy as np
import scipy.optimize
import scipy.sparse
import scipy.sparse.csgraph
import scipy.sparse.linalg
from scipy.spatial import cKDTree

logger = logging.getLogger("turbigen")

MERGE_RTOL = 1e-9
"""Distance below which two nodes are the same, as a fraction of the extent."""

DECAY = 0.3
"""Rate at which the wall control functions decay away from the wall, per node."""

PQ_RELAX = 0.1
"""Gain on the feedback that sets the wall control functions."""

SHIFT = 1e-8
"""Diagonal added to the factorised operator, against a singular plane."""

INNER_RTOL = 1e-3
"""Relative residual each linear solve is converged to. Loose, because each
is one step of an iteration that recomputes its residual anyway."""

INNER_MAX = 10
"""Passes on a factorisation before it is judged stale and refreshed."""

SIGNAL_MAX = 0.5
"""Largest lean, or logarithm of the ratio of first-cell height to its
target, that the control feedback acts on in one iteration."""

PQ_LIMIT = 5.0
"""Largest magnitude a control function may take. Well beyond any expansion
ratio, `Q` being its logarithm; it is `P` that needs the room, to shear the
grid lines of a thin wall block square against the pull of the mesh beyond."""


@dataclasses.dataclass(frozen=True)
class WallControl:
    """Spacing and orthogonality control on the `k = 0` face of one block.

    The face is the wall; `k` increases away from it and `i` runs along it.
    """

    block: int
    """Index of the block whose `k = 0` face is the wall."""

    spacing: np.ndarray
    """Target wall-normal distance of the first node off the wall, at each
    `i`, shape (ni,)."""

    closed: bool = False
    """Whether the wall is a closed loop, with `i = 0` and `i = ni - 1` the same
    node, as round an aerofoil."""

    weight: np.ndarray = None
    """Gain on the control at each `i`, shape (ni,), defaulting to one. Lower it
    where the wall has a corner, at which no single direction is normal."""


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
    control : WallControl, optional
        Which face to hold at a spacing and angle.

    """

    def __init__(self, xy, pitch=None, fixed=None, slide=None, control=None):
        self.shapes = [blk.shape[1:] for blk in xy]
        self.pitch = pitch
        self.control = control

        points = np.concatenate([blk.reshape(2, -1).T for blk in xy])
        self._offsets = np.cumsum([0] + [ni * nk for ni, nk in self.shapes])

        gid, shift, rep = self._merge(points)
        self.n_node = len(rep)
        self.gid = self._split(gid)
        self.shift = self._split(shift)

        is_fixed = self._flag(fixed)
        is_slide = self._flag(slide) & ~is_fixed

        self._build_stencils(is_fixed, is_slide)

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
        tol = MERGE_RTOL * np.ptp(points, axis=0).max()
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
        cg, cs, corner0 = [], [], []
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
            # Which block and index the origin corner of each cell is, for
            # orienting a stencil along the block it is in.
            ii, kk = np.meshgrid(np.arange(ni - 1), np.arange(nk - 1), indexing="ij")
            corner0.append(np.stack((np.full(ii.size, b), ii.ravel(), kk.ravel()), 1))
        cg = np.concatenate(cg)
        cs = np.concatenate(cs)
        corner0 = np.concatenate(corner0)
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
        # orientation, which is what control functions are expressed in.
        ent = ent[np.lexsort((ent[:, 8], ent[:, 7] != 0, ent[:, 0]))]
        count = np.bincount(ent[:, 0], minlength=self.n_node)
        start = np.concatenate(([0], np.cumsum(count)))

        stencil_g, stencil_s, ctrl_ik, regular = [], [], [], []
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

            b0, i0, k0 = corner0[e[0, 8]]
            ctrl_ik.append(
                (i0, k0)
                if e[0, 7] == 0 and control_owns(self.control, b0)
                else (-1, -1)
            )

        self.regular = np.array(regular, dtype=int)
        self.stencil_g = np.array(stencil_g, dtype=int).reshape(-1, 9)
        self.stencil_s = np.array(stencil_s, dtype=int).reshape(-1, 9)
        self.ctrl_ik = np.array(ctrl_ik, dtype=int).reshape(-1, 2)

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
        """Return node coordinates (n_node, 2) from per-block coordinates."""
        X = np.empty((self.n_node, 2))
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


def control_owns(control, block):
    """Whether `block` is the one whose wall is controlled."""
    return control is not None and control.block == block


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
# THE SOLVER
#


def smooth(
    topo,
    X0,
    pq=None,
    scale=1.0,
    omega=0.8,
    tol=1e-3,
    max_iter=200,
    keep_spacing=False,
    keep_slide_angle=False,
):
    """Smooth a plane of blocks.

    Each iteration assembles the equations at the current mesh and moves the
    mesh by the correction that would satisfy them, found by iterating on a
    factorisation of the operator that is only refreshed when it stops
    converging. A stale operator changes how fast the iteration goes and not
    where it ends, because the correction is always driven by the residual of
    the equations as they are now; and the operator changes slowly once the
    mesh is close, so a factorisation costing fifteen solves pays for itself
    many times over.

    Parameters
    ----------
    topo : PlaneTopology
        Structure of the plane.
    X0 : ndarray
        Starting node coordinates, shape (n_node, 2), as from
        :meth:`PlaneTopology.gather`. Fixed nodes stay where they are here, and
        sliding nodes keep their `x`.
    pq : ndarray, optional
        Starting wall control functions, shape (2, ni) for the controlled block,
        as returned from a previous call. Zero if not given.
    scale : float
        Length that changes in position are measured against for convergence,
        typically the smallest spacing in the mesh.
    omega : float
        Under-relaxation of each update.
    tol : float
        Converged when no node moves more than this times `scale` in an
        iteration, and no wall control function changes by more than this.
    max_iter : int
        Most iterations to take.
    keep_spacing : bool
        Hold the spacing of the starting mesh along each grid line of the
        controlled block, with :func:`inherent_sources`, tapering to none at its
        outer edge, rather than let the equations distribute points as they
        would on their own. Elsewhere the equations are left to themselves,
        which spreads out clustering carried in from the block's edges.
    keep_slide_angle : bool
        Hold the first grid line off each sliding node at the angle it has in
        the starting mesh, rather than sit the node level with its neighbours
        inside and meet the boundary square. The pitchwise offset from those
        neighbours is their current distance in `x` times the starting slope,
        lagged by an iteration in `x`. Holding the offset itself would not do:
        the smoother is free to squeeze the first cell in `x`, and the same
        offset across a shorter cell is a steeper line, whose shear then
        carries on through the mesh beyond.

    Returns
    -------
    X : ndarray
        Smoothed node coordinates, shape (n_node, 2).
    pq : ndarray or None
        Final wall control functions, to start a neighbouring plane from.
    n_iter : int
        Number of iterations taken.

    """
    X = X0.copy()
    ctrl = topo.control
    if ctrl is not None:
        ni = topo.shapes[ctrl.block][0]
        pq = np.zeros((2, ni)) if pq is None else pq.copy()

    # Sources that make the starting mesh's spacing along each grid line its
    # own, so smoothing straightens and squares the lines without washing out
    # the clustering it was built with.
    if keep_spacing and ctrl is not None:
        PQ0 = inherent_sources(topo, X0) * _taper(topo)[:, None]
    else:
        PQ0 = np.zeros((len(topo.regular), 2))

    if keep_slide_angle:
        dx0, dy0 = _slide_offsets(topo, X0)
        slide_slope = dy0 / dx0
    slide_offset = 0.0

    lu = None
    for n_iter in range(1, max_iter + 1):
        dpq = 0.0
        if ctrl is not None:
            pq_new = _update_control(topo, X, pq)
            dpq = np.abs(pq_new - pq).max()
            pq = pq_new
        if keep_slide_angle:
            slide_offset = slide_slope * _slide_offsets(topo, X)[0]
        A, rhs = _system(topo, X, PQ0 + _source(topo, pq), slide_offset)

        # The correction that satisfies the equations frozen here, iterating
        # on the last factorisation. While the operator has changed little that
        # takes a pass or two; once it has changed a lot it stops converging,
        # which is the signal to refactorise.
        # Rows are normalised to a unit diagonal, so a residual is a distance,
        # and one far below what convergence is judged on is enough.
        atol = 0.1 * tol * scale
        dX = np.empty_like(X)
        for d in range(2):
            res = rhs[:, d] - A[d] @ X[:, d]
            if lu is not None:
                dX[:, d], ok = _inner_solve(A[d], res, lu[d], atol)
                if ok:
                    continue
            lu = _factorise(topo, A, lu, d)
            dX[:, d], _ = _inner_solve(A[d], res, lu[d], atol)

        X += omega * dX
        change = omega * np.abs(dX).max() / scale
        if change < tol and dpq < tol:
            break
    else:
        logger.debug(
            f"Smoothing did not converge: change={change:.3g}, dpq={dpq:.3g}, tol={tol}"
        )

    logger.debug(f"Smoothed in {n_iter} iterations, change={change:.3g}")
    return X, pq, n_iter


def inherent_sources(topo, X):
    """Return the sources (n_regular, 2) that keep the spacing of mesh `X`.

    After Thomas and Middlecoff: in one dimension `r_xixi + P r_xi = 0` holds
    any distribution of points if `P = -r_xi . r_xixi / |r_xi|^2`, so with
    these the equations leave the spacing along each grid line as it was and
    act on the shape of the lines. Independent of the orientation of a
    stencil, as the equations are, so defined the same at a block interface
    as inside a block.
    """
    V = topo._values(X, topo.stencil_g, topo.stencil_s)
    E, N, W, S, C = V[:, 1], V[:, 3], V[:, 5], V[:, 7], V[:, 0]
    r_xi, r_eta = 0.5 * (E - W), 0.5 * (N - S)
    r_xixi, r_etaeta = E - 2.0 * C + W, N - 2.0 * C + S
    P = -np.sum(r_xi * r_xixi, axis=1) / np.sum(r_xi**2, axis=1)
    Q = -np.sum(r_eta * r_etaeta, axis=1) / np.sum(r_eta**2, axis=1)
    return np.clip(np.stack((P, Q), axis=1), -PQ_LIMIT, PQ_LIMIT)


def _taper(topo):
    """Return a weight at each regular node, one at the wall to zero outside.

    Zero away from the controlled block, and falling quadratically across it,
    so the sources it carries meet the rest of the mesh without a step.
    """
    i, k = topo.ctrl_ik.T
    nk = topo.shapes[topo.control.block][1]
    weight = np.where(i >= 0, 1.0 - (k / (nk - 1)) ** 2, 0.0)
    return np.clip(weight, 0.0, 1.0)


def _source(topo, pq):
    """Return the control functions (n_regular, 2) at every regular node."""
    PQ = np.zeros((len(topo.regular), 2))
    if pq is None:
        return PQ
    i, k = topo.ctrl_ik.T
    on = i >= 0
    nk = topo.shapes[topo.control.block][1]
    # Off below the last layer, so the outer edge of the block is unforced and
    # whatever lies beyond it meets an unforced mesh.
    on &= k < nk - 1
    decay = np.exp(-DECAY * k[on])
    PQ[on] = (pq[:, i[on]] * decay).T
    return PQ


def _factorise(topo, A, lu=None, d=None):
    """Return a factorisation of each coordinate's operator.

    Shifted off singular, for a plane free to slide bodily in `y`: it is only a
    preconditioner, so it need not be exact. The two coordinates share one
    operator, and so one factorisation, when no node slides. When they have
    their own, passing the last factorisations `lu` and a coordinate `d`
    refreshes only that coordinate's, the other being as good as it was.
    """
    _, patterns, _ = _static(topo)

    def one(a, pat):
        shifted = a.copy()
        shifted.data[pat["diag"]] += SHIFT
        return scipy.sparse.linalg.splu(shifted, permc_spec="MMD_AT_PLUS_A")

    if A[1] is A[0]:
        f = one(A[0], patterns[0])
        return [f, f]
    if lu is None or d is None:
        return [one(a, pat) for a, pat in zip(A, patterns)]
    lu = list(lu)
    lu[d] = one(A[d], patterns[d])
    return lu


def _inner_solve(A, b, lu, atol):
    """Return the solution of `A x = b` by iterating on `lu`, and success.

    Preconditioned Richardson iteration: with a factorisation of a nearby
    operator, each pass cuts the residual by orders of magnitude. It fails,
    asking for a fresh factorisation, as soon as a pass does not reduce the
    residual or it has taken too many.
    """
    target = max(INNER_RTOL * np.abs(b).max(), atol)
    x = lu.solve(b)
    res_last = np.inf
    for _ in range(INNER_MAX):
        r = b - A @ x
        res = np.abs(r).max()
        if res <= target:
            return x, True
        if res >= res_last:
            return x, False
        res_last = res
        x += lu.solve(r)
    return x, False


def _static(topo):
    """Return the parts of the equations that do not depend on the mesh.

    The constant the pitch shifts of the singular and sliding nodes contribute
    to the right-hand side, and for each coordinate the sparsity pattern of its
    operator with the fixed, singular and sliding rows already filled in, so
    that each iteration only has to add in the stencils of the regular nodes.
    The pattern is shared by both coordinates when no node slides, which is
    what lets them share a factorisation. Built once and cached on the
    topology.
    """
    if getattr(topo, "_static_cache", None) is not None:
        return topo._static_cache

    pitch = topo.pitch or 0.0
    rows, cols, vals = [], [], []
    shift_rhs = np.zeros(topo.n_node)

    # Singular points sit at the mean of where each cell round them would put
    # them, were it a parallelogram.
    for node, nbs in zip(topo.irregular, topo.irregular_nb):
        rows.append(np.full(len(nbs) + 1, node))
        cols.append(np.array([node] + [g for g, _, _ in nbs]))
        vals.append(np.array([1.0] + [-w for _, _, w in nbs]))
        shift_rhs[node] = sum(s * w for _, s, w in nbs) * pitch

    # Fixed nodes stay put.
    rows.append(topo.fixed)
    cols.append(topo.fixed)
    vals.append(np.ones(len(topo.fixed)))

    common = [np.concatenate(a) for a in (rows, cols, vals)]

    # Sliding nodes keep their x, and in y sit level with the inside.
    ry, cy, vy = [], [], []
    for node, nbs in zip(topo.slide, topo.slide_nb):
        m = len(nbs)
        ry.append(np.full(m + 1, node))
        cy.append(np.array([node] + [g for g, _ in nbs]))
        vy.append(np.array([1.0] + [-1.0 / m] * m))
        shift_rhs[node] = sum(s for _, s in nbs) * pitch / m
    slide_x = (topo.slide, topo.slide, np.ones(len(topo.slide)))
    slide_y = tuple(np.concatenate(a) if a else np.zeros(0) for a in (ry, cy, vy))

    reg_rows = np.repeat(topo.regular, 9)
    reg_cols = topo.stencil_g.ravel()
    patterns = []
    for extra in (slide_x, slide_y):
        if patterns and not len(topo.slide):
            patterns.append(patterns[0])
            continue
        r, c, v = (np.concatenate((a, e)) for a, e in zip(common, extra))
        patterns.append(_pattern(topo.n_node, reg_rows, reg_cols, r, c, v))

    # The stencils as the compiled kernel takes them, converted once here
    # rather than on every call.
    stencils = tuple(
        np.asfortranarray(a, dtype=np.int32) for a in (topo.stencil_g, topo.stencil_s)
    )

    topo._static_cache = (shift_rhs, patterns, stencils)
    return topo._static_cache


def _pattern(n, reg_rows, reg_cols, rows, cols, vals):
    """Return the compressed-column pattern of one operator, and how to fill it.

    The regular stencils come first in the triplets and the constant rows after.
    Entries that land on the same place are summed, as a sparse constructor
    would, so filling is a weighted bincount through `inv` and the constant
    part is summed here once.
    """
    n_reg = len(reg_rows)
    rows = np.concatenate((reg_rows, rows)).astype(np.int64)
    cols = np.concatenate((reg_cols, cols)).astype(np.int64)
    # Column-major keys sort into compressed-column order.
    key, inv = np.unique(cols * n + rows, return_inverse=True)
    indices = key % n
    indptr = np.concatenate(([0], np.cumsum(np.bincount(key // n, minlength=n))))
    nnz = len(key)
    base = np.bincount(inv[n_reg:], weights=vals, minlength=nnz)
    diag = np.searchsorted(key, np.arange(n) * (n + 1))
    if not np.array_equal(key[diag], np.arange(n) * (n + 1)):
        raise ValueError("Every row of the operator needs a diagonal entry.")
    return {
        "indices": indices,
        "indptr": indptr,
        "inv": inv[:n_reg],
        "base": base,
        "diag": diag,
    }


def _slide_offsets(topo, X):
    """Return how far each sliding node of `X` is from its neighbours inside.

    Measured from the mean of those neighbours, each seen across any periodic
    boundary between them, which is where the sliding equations put it in `y`.

    Returns
    -------
    dx, dy : ndarray
        Offsets in `x` and `y`, one per sliding node.
    """
    if getattr(topo, "_slide_flat", None) is None:
        # Neighbours of every sliding node in one array, with which node owns
        # each, built once rather than walked on every call.
        owner = np.repeat(np.arange(len(topo.slide)), [len(n) for n in topo.slide_nb])
        g, sh = np.array([nb for nbs in topo.slide_nb for nb in nbs]).reshape(-1, 2).T
        topo._slide_flat = (owner, g, sh, np.bincount(owner))
    owner, g, sh, count = topo._slide_flat

    pitch = topo.pitch or 0.0
    nb = X[g].copy()
    nb[:, 1] += sh * pitch
    mean = np.stack(
        [
            np.bincount(owner, weights=nb[:, d], minlength=len(count)) / count
            for d in (0, 1)
        ]
    )
    return X[topo.slide].T - mean


def _system(topo, X, PQ, slide_offset=0.0):
    """Return the linear equations for each coordinate, frozen at `X`.

    `slide_offset` is added to where the sliding equations put each sliding
    node in `y`, one value per sliding node or one for all.

    Returns
    -------
    A : list of scipy.sparse.csc_matrix
        Operator for each coordinate.
    rhs : ndarray, shape (n_node, 2)
        Right-hand sides.

    """
    n = topo.n_node
    pitch = topo.pitch or 0.0
    shift_rhs, patterns, stencils = _static(topo)

    # Coefficients in stencil order, normalised to a unit diagonal, with the
    # lagged source and the constants the periodic shifts contribute.
    w, src = ember.elliptic.ttm_stencil(X, *stencils, PQ, pitch)

    rhs = np.zeros((n, 2))
    rhs[:, 1] = shift_rhs
    rhs[topo.regular] = src
    rhs[topo.fixed] = X[topo.fixed]
    rhs[topo.slide, 0] = X[topo.slide, 0]
    rhs[topo.slide, 1] += slide_offset

    A = []
    for pat in patterns:
        if A and pat is patterns[0]:
            A.append(A[0])
            continue
        data = pat["base"] + np.bincount(
            pat["inv"], weights=w.ravel(), minlength=len(pat["base"])
        )
        A.append(
            scipy.sparse.csc_matrix((data, pat["indices"], pat["indptr"]), shape=(n, n))
        )
    return A, rhs


def cell_areas(topo, X):
    """Return the signed area of every cell, one (ni-1, nk-1) array per block.

    Positive for a cell whose `i` and `k` directions turn anticlockwise, which
    every block of a valid mesh shares; a negative cell has folded.
    """
    out = []
    for blk in topo.scatter(X):
        a = blk[:, 1:, 1:] - blk[:, :-1, :-1]
        b = blk[:, :-1, 1:] - blk[:, 1:, :-1]
        out.append(0.5 * (a[0] * b[1] - a[1] * b[0]))
    return out


def wall_metrics(topo, X):
    """Return first-cell height and lean at each node of the controlled wall.

    Returns
    -------
    height : ndarray, shape (ni,)
        Distance of the first node off the wall, normal to the wall.
    lean : ndarray, shape (ni,)
        Tangent of the angle between the first grid line off the wall and the
        wall normal, positive leaning towards increasing `i`.
    idx : ndarray
        The wall nodes these are defined at: all but the repeated node of a
        closed wall, all but the two ends of an open one.

    """
    ctrl = topo.control
    blk = topo.scatter(X)[ctrl.block]
    ni = blk.shape[1]
    r0, r1 = (blk[:, :, k].T for k in range(2))

    # Neighbours along the wall; a closed wall wraps past the repeated node.
    if ctrl.closed:
        idx = np.arange(ni - 1)
        ip = np.roll(idx, -1)
        im = np.roll(idx, 1)
    else:
        idx = np.arange(1, ni - 1)
        ip, im = idx + 1, idx - 1

    tangent = r0[ip] - r0[im]
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
    off = r1[idx] - r0[idx]

    # Unit normal, turned to point into the mesh.
    normal = np.stack((-tangent[:, 1], tangent[:, 0]), axis=1)
    normal *= np.sign(np.sum(normal * off, axis=1, keepdims=True))

    height = np.full(ni, np.nan)
    lean = np.full(ni, np.nan)
    height[idx] = np.sum(off * normal, axis=1)
    lean[idx] = np.sum(off * tangent, axis=1) / height[idx]
    return height, lean, idx


def _update_control(topo, X, pq):
    """Return wall control functions moved to correct the wall spacing and angle.

    A feedback rather than Sorenson's estimate. Sorenson solves the governing
    equation at the wall for the `P` and `Q` that would give the target
    derivatives there, but no equation is solved at the wall --- its nodes are
    fixed --- and the sources act only from the first node off it, already
    decayed, so the estimate settles short of the target: a quarter too coarse
    on a cylinder. Correcting each by what is measured, after Hilgenstock and
    White, has the target as its only fixed point.

    `Q` is the logarithmic rate of expansion away from the wall, so it is
    moved by the logarithm of how far the first cell is from the target
    height. `P` moves nodes along the wall, and is moved against the lean of
    the first grid line off it.
    """
    ctrl = topo.control
    height, lean, idx = wall_metrics(topo, X)
    ni = len(height)

    weight = np.ones(ni) if ctrl.weight is None else ctrl.weight
    # Each signal limited, so that a mesh still far from settled, where a first
    # cell can be nearly flat, nudges the control rather than throwing it.
    ratio = np.clip(height[idx] / ctrl.spacing[idx], np.exp(-SIGNAL_MAX), None)
    dpq = np.zeros_like(pq)
    dpq[0, idx] = -np.clip(lean[idx], -SIGNAL_MAX, SIGNAL_MAX)
    dpq[1, idx] = -np.clip(np.log(ratio), -SIGNAL_MAX, SIGNAL_MAX)
    dpq *= PQ_RELAX * weight

    pq = np.clip(pq + dpq, -PQ_LIMIT, PQ_LIMIT)
    if ctrl.closed:
        pq[:, -1] = pq[:, 0]
    return pq


#
# LOCAL OPTIMISATION
#

POLISH_EPS = 1e-3
"""Regularisation of the corner Jacobian in :func:`polish`, so a folded cell
has a finite, steep cost rather than none."""


def polish(topo, X, centres, rings=4, w_size=1.0, max_iter=500, X_target=None):
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
    # Which nodes each corner's edges run between, for the gradient.
    edge_nodes = [
        ((1, 0), (3, 0)),
        ((1, 0), (2, 1)),
        ((2, 3), (2, 1)),
        ((2, 3), (3, 0)),
    ]
    n_corner = 4.0 * len(ids)
    X0 = X.copy()

    def fun(z):
        Xc = X0.copy()
        Xc[free] = z * scale
        f = 0.0
        G = np.zeros((len(ids), 4, 2))
        for (a, b), ((pa, ma), (pb, mb)) in zip(corners(Xc), edge_nodes):
            ta, tb = a / la[:, None], b / lb[:, None]
            det = ta[:, 0] * tb[:, 1] - ta[:, 1] * tb[:, 0]
            root = np.sqrt(det**2 + POLISH_EPS**2)
            dreg = 0.5 * (det + root)
            dd = 0.5 * (1.0 + det / root)
            nrm2 = np.sum(ta**2, 1) + np.sum(tb**2, 1)
            f += np.sum(nrm2 / (2.0 * dreg) + w_size * 0.5 * (dreg + 1.0 / dreg))
            dF = (-nrm2 / (2.0 * dreg**2) + w_size * 0.5 * (1.0 - 1.0 / dreg**2)) * dd
            ga = ta / dreg[:, None] + dF[:, None] * np.stack((tb[:, 1], -tb[:, 0]), 1)
            gb = tb / dreg[:, None] + dF[:, None] * np.stack((-ta[:, 1], ta[:, 0]), 1)
            ga /= la[:, None]
            gb /= lb[:, None]
            G[:, pa] += ga
            G[:, ma] -= ga
            G[:, pb] += gb
            G[:, mb] -= gb
        g = np.zeros_like(X)
        np.add.at(g, ids.ravel(), G.reshape(-1, 2))
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
