"""OH-topology meshing with a passage block from blade to blade.

Eight blocks per blade row: an O block wrapping the blade; a passage block
filling the space between the O block of one blade and that of the next; a
block ahead of the leading edge and one behind the trailing edge; one each
ahead of and behind the passage block; and an H block from each end of those
to the inlet, outlet or mixing plane::

     inlet/MP   Z_in                                      Z_out   outlet/MP
       |        |  LE  C1----------- O -----------C2  TE  |        |
       |   up   +------C4--------------------------C3-----+   dn   |
       |        |  Pin |     P (to the blade below)   | Pout |        |
       |        +------+------------------------------+------+        |

The only periodic boundaries are ahead of the corners C1 and C4 and behind C2
and C3, between the block at each end of the blade and the one at each end of
the passage, and the edges of the inlet and outlet H blocks. The corners are
the points of the O block's outline nearest the corners of its bounding box,
the box turned to the stagger of the blade, up to a limit.

Every block runs from hub to casing in `j`, and every constant-`j` plane is a
surface of constant span fraction of the annulus --- the surface the blade
section at that span fraction lies on. Each is meshed in its conformal
coordinates `(m', theta)`, `dm' = dm / r`, in which the surface of revolution
is a plane and angles are angles on the surface: a mesh orthogonal there is
orthogonal on the stream surface, and the blade and its metal angles appear
undistorted, `dtheta/dm' = tan(chi)`, at any radius. Node counts are fixed
once per row, at mid-span, and reused at every span. A plane is built
and smoothed coarse, with cells the size of those the free stream needs, and
fewer only where the blade turns too sharply for them. It is smoothed by
point-Jacobi sweeps of Winslow's equations. Each sweep, the corners take the
mean of where each cell round them would put them, and the nodes near them are
drawn in towards them; the nodes on the blade slide along it to the foot of
the perpendicular from the node outside them, which squares the cells there;
and those on the inlet and outlet sit level with the node inside, which
squares the ends.

Only once smoothed is it refined, where the flow needs it: along the blade,
to the leading and trailing edges; across the passage, towards the O blocks;
and along the blocks ahead of and behind the blade, towards the O block. New
nodes go at fractional indices of the coarse ones, on cubic splines through
them, each spline running only between the junctions of blocks, so blocks
that share a stretch of edge still match. The O block is clustered to the
wall spacing at the blade, respaced along its smoothed lines.

A row with a tip gap has two more blocks, from a spanwise node at the top of
the blade to the casing, inside the outline the blade would have had: an O
block, whose outer face is the O block's inner one over the gap, node for
node, and an H block filling it from the leading edge to the trailing edge,
its lines normal to the camber line, meeting the O block node for node all
round, or along the lower side and the rear only, by non-matching interfaces
along the front and the upper side, as :attr:`_Tune.tip_match` chooses.
See :meth:`OHPassage._tip_gap`.
"""

import dataclasses
import functools
import itertools
import logging
from typing import ClassVar

import ember.block
import ember.elliptic
import ember.grid
import ember.patch
import numpy as np
import scipy.interpolate
import scipy.ndimage
import scipy.sparse
import scipy.sparse.linalg
from scipy.spatial import cKDTree

from turbigen import clusterfunc, elliptic, util
from turbigen.blade import CHORD_FRACTION
from turbigen.hmesh import _by_theta
from turbigen.mesh import Mesher
from turbigen.node import Node

logger = logging.getLogger("turbigen")

LABELS = ("O", "P", "LE", "Pin", "TE", "Pout", "up", "dn")
"""Block names, in the order blocks are held."""


CORNER_COS, CORNER_SIN = np.cos(np.pi / 3.0), np.sin(np.pi / 3.0)
"""Direction of an edge out of a corner of the O block, against the O block's
face and its outward normal there."""


DTYPE = np.float32
"""Precision the planes are held, swept, refined and polished in: the
finished grid is stored in single precision, and the sweeps run faster in
it. Only their geometry is worked out in double before."""

THETA_BISECT = 10
"""Halvings :func:`_theta_even` finds each station's weight by, to a 2^10th
of the range it searches."""


class ConformalMap:
    """Conformal coordinates on one surface of constant span fraction.

    Parameters
    ----------
    annulus : Annulus
        The annulus the surface is in.
    spf : float
        Span fraction of the surface.
    m_lo, m_hi : float
        Range of annulus meridional coordinate to cover.

    """

    def __init__(self, annulus, spf, m_lo, m_hi):
        self.annulus = annulus
        self.spf = spf
        # With the joins between the annulus's segments among the samples:
        # `m` runs on at a different rate either side of one, so `m'`, linear
        # in `m` between samples, would cut the corner there.
        # Less any sample a rounding error from one, which would leave a
        # segment of no length.
        joins = np.arange(np.ceil(m_lo), np.floor(m_hi) + 1.0)
        m = np.linspace(m_lo, m_hi, _Tune.n_merid)
        near = np.isclose(m[:, None], joins[None, :], rtol=0.0, atol=1e-9).any(axis=1)
        self.m = np.union1d(m[~near], joins)
        self.xr = annulus.evaluate_xr(self.m, spf)
        ds = np.linalg.norm(np.diff(self.xr, axis=1), axis=0)
        r_mid = 0.5 * (self.xr[1, 1:] + self.xr[1, :-1])
        self.mp = np.concatenate(([0.0], np.cumsum(ds / r_mid)))

    def m_from_mp(self, mp):
        """Return annulus meridional coordinate at conformal coordinate `mp`."""
        return np.interp(mp, self.mp, self.m)

    def mp_from_m(self, m):
        """Return conformal coordinate at annulus meridional coordinate `m`."""
        return np.interp(m, self.m, self.mp)

    def xr_from_mp(self, mp):
        """Return axial and radial coordinates (2, ...) on the surface at `mp`.

        Evaluated on the annulus itself rather than interpolated, so every node
        lies exactly on the surface it was meshed on.
        """
        mp = np.asarray(mp)
        # Allowed past the ends by the rounding of its precision there, in
        # which a node put on an end can come out a little beyond it.
        eps = np.finfo(mp.dtype).eps if mp.dtype.kind == "f" else 0.0
        tol = max(1e-9, 16 * eps * np.abs(self.mp).max())
        if mp.min() < self.mp[0] - tol or mp.max() > self.mp[-1] + tol:
            raise ValueError("Conformal coordinate outside the mapped range.")
        mp = np.clip(mp, self.mp[0], self.mp[-1])
        return self.annulus.evaluate_xr(self.m_from_mp(mp), self.spf)

    @functools.cached_property
    def _tree(self):
        """A search tree of the samples, built only if asked for."""
        return cKDTree(self.xr.T)

    def mp_from_xr(self, xr):
        """Return the conformal coordinate of points (2, n) on the surface.

        Each point is projected onto the nearer of the two sampled segments
        either side of its nearest sample.
        """
        pts = np.reshape(xr, (2, -1)).T
        mp = _project(self.xr, self.mp, self._tree, pts)
        return mp.reshape(np.shape(xr)[1:])


@dataclasses.dataclass(frozen=True)
class _Layout:
    """Counts of one row on the coarse mesh, and how it is refined, fixed at
    mid-span."""

    pitch: float
    m_in: float
    m_Zin: float
    """Where the block ahead of the blade ends, at mid-span; each plane's is as
    far from its own leading edge, see :meth:`OHPassage.plane_Z`."""
    m_Zout: float
    """Where the block behind the blade ends, likewise."""
    m_out: float
    m_edges: tuple
    """The blade's leading and trailing edges at mid-span, annulus `m`."""
    d_wall: float
    """Wall spacing [m]."""
    t_O: float
    """Thickness of the O block, in conformal units."""
    h: float
    """Cell size next to the O block once refined, in conformal units."""
    n_wall: tuple
    """Coarse cells along each piece of the blade, from the middle of the
    trailing edge round the lower side, the leading edge and the upper side."""
    f_wall: tuple
    """Where the coarse nodes lie along each of those pieces, as fractions of
    its length."""
    corners: tuple
    """Where the corners C3, C4, C1, C2 lie, as fractions of the way round the
    blade from the middle of the trailing edge."""
    i_wall: tuple
    """Which ends of those pieces are the corners C3, C4, C1, C2."""
    n_O: int
    """Coarse cells across the O block: twice as many as the free stream
    size would give, so that it holds more of its thickness against the
    passage block's through the sweeps."""
    n_P: int
    """Coarse cells across the passage block."""
    n_in: int
    n_out: int
    n_up: int
    n_dn: int
    """Coarse cells along the blocks ahead of and behind the blade, and the
    inlet and outlet blocks."""
    nk_O: int
    """Nodes across the O block once clustered to the blade: for its
    nominal thickness here, and for its thickness once swept by
    :meth:`OHPassage._O_count`."""
    wall_density: tuple
    """Refined cells per unit length round the blade, as fractions of its
    length from the middle of the trailing edge, and the density there,
    for the O block's nodes on the blade."""
    sigma_LE: float
    """Where the leading edge lies in `wall_density`, as a fraction of the
    way round the blade, at mid-span. On another plane the density is moved
    round the blade to put this on its own leading edge."""
    chord_frac: float
    """Meridional chord as a fraction of the way round the blade, both in
    conformal units, at mid-span. On another plane the cells along the blade
    are scaled by its own over this, so they are the same fractions of its
    own chord."""
    nf_wall: tuple
    """Refined cells along each piece of the blade, in the order of
    `n_wall`: the rear of the lower side from the cut, the lower side
    between corners, the leading edge, the upper side between corners, and
    the rear of the upper side to the cut. Counted again once mid-span is
    swept, by :meth:`OHPassage._recount_wall`."""
    u: dict
    """Where each refined node lies, in fractional coarse indices from the
    start of its stretch: across the passage, `P`; and along the blocks
    ahead of and behind the blade, `in` and `out`. Once refined along the
    blade by :meth:`OHPassage._wall_u`, also along the rear of the lower and
    upper sides of the blade from the cut, `rl` and `ru`; along the lower
    and upper sides between corners, upstream to downstream, `side_lo` and
    `side_up`; and round the leading edge, `front`."""

    H: float = 0.0
    """Free stream cell size, in conformal units."""

    H_coarse: float = 0.0
    """Cell size of the coarse mesh in the free stream, in conformal units."""

    r_free: float = 1.0
    """Radius at which :attr:`H` and :attr:`H_coarse` hold [m]: the least on
    the row's mid-span surface. Along the blocks ahead of and behind the blade,
    and the ducts beyond, a free stream cell is that many metres long, so
    shorter in conformal units, by the ratio of radii, wherever the radius is
    larger."""

    extrude_in: bool = False
    extrude_out: bool = False
    """Whether the inlet block ahead of the first row, and the outlet block
    behind the last, are extruded from a cut rather than swept; if so, they
    are one coarse cell long beyond the cut through the sweeps."""

    symmetric: bool = False
    """Whether the blade nodes are alike on both halves of the blade, a row
    with a tip gap matched node for node all round
    (:attr:`_Tune.tip_match`)."""

    def corners_at(self, spf):
        """Where the corners C3, C4, C1, C2 lie at span fraction `spf`, as
        fractions of the way round the blade from the middle of the
        trailing edge: `corners`, the same at every span."""
        return self.corners

    def i_corners_coarse(self):
        """O block `i` of the corners C3, C4, C1, C2 on the coarse mesh."""
        ends = np.cumsum((0, *self.n_wall))
        return tuple(int(ends[i]) for i in self.i_wall)

    def i_corners(self):
        """O block `i` of the corners C3, C4, C1, C2 once refined."""
        return tuple(int(i) for i in np.cumsum(self.nf_wall[:4]))


class _Tune(Node):
    """Every number the passage mesher is tuned by, which a user has no need
    to set: first the settings it was tuned with, then its internals.

    Each can be set under `tune` in the mesh config, to experiment, and only
    those changed are written back out. See :class:`OHPassage`."""

    ER_span: float = 1.2
    """Expansion ratio of spanwise grid away from hub and casing."""

    n_span_smooth: int = 17
    """Number of spanwise planes to smooth, hub and casing included. The rest
    are interpolated between them. Zero smooths every plane."""

    turn_max: float = 15.0
    """Most the blade turns across one cell of the coarse mesh, degrees."""

    ER_surf: float = 1.15
    """Expansion ratio along the blade away from its edges."""

    ER_stream: float = 1.2
    """Expansion ratio away from the O block, across the passage and along
    the blocks ahead of and behind the blade, except the block from a blade
    to the mixing plane behind it, which grows at `ER_mix`."""

    ER_mix: float = 1.1
    """Expansion ratio along the block from a blade to the mixing plane
    behind it."""

    AR_mix: float = 2.0
    """Largest aspect ratio, length along over width across the pitch, of
    the cells of the block from a blade to the mixing plane behind it, at
    mid-span, which sets how many it has. Along each line its first cell
    within `ER_mix` of the last of the blocks behind the blade where it
    meets them, growing at `ER_mix` to a cap that fills the line in that
    count, on every plane, so the aspect ratio varies with span."""

    ER_O: float = 1.2
    """Expansion ratio off the blade in the O block."""

    n_sweep: int = 5000
    """Smoothing sweeps on the coarse mesh."""

    relax: float = 0.8
    """Fraction of the way each node moves each smoothing sweep, the blade
    nodes aside. One is plain point-Jacobi; below one damps the node-to-node
    oscillation it leaves where the coarse mesh is skewed, and converges to
    the same mesh more slowly."""

    correct_every: int = 0
    """Sweeps between corrections of the smooth error the sweeps are slow to
    remove, by :func:`elliptic.winslow_correction`, once the blade nodes are
    held; zero for none. The held sweeps converge to the same mesh either
    way, in far fewer with them, so `n_sweep_held` can be cut with it. Not
    while the blade nodes slide: how far they slide is set by how many
    sweeps they slide for."""

    correct_refactor: float = 0.1
    """Strain of the plane by a correction, as a fraction of the lengths of
    its cells when its map was made, below which the map is kept whatever:
    the plane is then near enough converged that the map no longer matters.
    Above it, the map is made again for the next correction unless this one
    strained the plane less than `correct_contract` of the last. From a
    plane far from smooth, the coefficients change as it is corrected, and a
    map made early stops converging it and then overshoots, by more than the
    sweeps take back."""

    correct_contract: float = 0.8
    """Least factor each correction must reduce the strain by, while it is
    above `correct_refactor`, to keep its map."""

    n_sweep_held: int | None = None
    """Sweeps once the blade nodes are held, if they slide first; None for
    what `slide_share` leaves of `n_sweep`."""

    orth_taper: float = 8.0
    """With it, the sweeps draw square to the blade not only the first line
    of the O block off it, by `orth`, but every line `k` out after, from the
    line inside it, by `orth` times (1 - (k - 1) / (K - 1)) to this power,
    `K` the O block's edge, which is left alone. Only the first if zero."""

    t_O: float = 0.1
    """Thickness of the O block, as a fraction of pitch, before smoothing."""

    o_hold: float = 0.3
    """Least length of each coarse cell along each O block line through the
    sweeps, as a fraction of its length on the algebraic plane they start
    from, so the block keeps its thickness round the edges, where the sweeps
    draw its outer face in. Zero for no floor."""

    o_coarse_factor: float = 2.0
    """Coarse cells across the O block, as a multiple of what the free stream
    size gives across its thickness: more hold more of it through the sweeps."""

    cut_mixing: float = 0.3
    """Distance from a blade's edge on the side of a mixing plane at which
    the block ahead of or behind it ends, in meridional chords of that row;
    the block beyond runs on to the mixing plane. Held to at most
    `cut_mixing_frac` of the way there where the gap is too short for it."""

    cut_mixing_frac: float = 0.5
    """The furthest the block ahead of or behind a blade ends towards a
    mixing plane, as a fraction of the way from the blade's edge to it."""

    straighten: float = 1.0
    """Distance over which the lines of the extruded inlet and outlet blocks
    turn from the angle of the grid at the cut to meridional, in meridional
    chords."""

    even_theta: float = 2.0
    """Distance over which the nodes of the extruded inlet and outlet blocks
    relax from the spacing across the pitch at the cut to even in theta, in
    meridional chords."""

    spf_corners: float = 0.5
    """Span fraction at which the corners of the O block are found, by the
    rules of :meth:`_wall`, and kept at every span as the same fractions of
    the way round the blade."""

    corner_ref: str = "ends"
    """Where the corners every span starts from are found: "spf", at
    `spf_corners`; "ends", the mean of those found at the hub and at the
    casing, as fractions of the way round the blade."""

    stagger_max: float = 45.0
    """Most the bounding box that places the corners is turned, degrees."""

    cor_fac: float = 0.5
    """How hard the nodes near each corner are drawn in towards it each
    sweep: lower draws them in harder."""

    cor_fac_mid: float | None = 0.5
    """`cor_fac` of the first row's corners part way along the blade, more
    than `corner_mid` of the way round it from both edges, rather than by one
    of them; null for `cor_fac`."""

    n_corner: int = 16
    """Nodes along each side of the patch drawn in towards a corner."""

    polish_below: float = 32.0
    """Once swept, the nodes of each control plane near any cell whose worst
    corner angle, folded onto 0 to 90 degrees, is under this are polished:
    within a few cells of it, short of the O block's first layers; and again
    once refined, where one is under `polish_fine_below`. None if zero."""

    polish_fine_below: float = 30.0
    """Worst cell corner, degrees, under which a refined control plane is
    polished again, with `polish_below`; the polish of the swept plane
    leaves most above 30. None if zero."""

    nk_tip_O_min: int = 9
    """Least number of nodes across the O block of a tip gap, from the H block
    to the outline of the blade, one more than a multiple of eight. The count
    is otherwise as many as grow from the wall spacing at `ER_O` across the
    block's thickness, :func:`_gap_O_count`."""

    t_tip_O: float = 0.2
    """Thickness of the O block of a tip gap, as a fraction of the blade's
    greatest thickness on the section at its tip, capped at `gap_cap` of the
    radius of the largest circle inside the blade touching it at each node."""

    tip_match: int = 0
    """How a tip gap's H block meets its O block: 1 node for node all round;
    2 node for node along the lower side and the rear, by non-matching
    interfaces along the upper side and, unless as many cells round the nose
    as the tail, the front; 0 to choose for each row, 1 where nodes at equal
    fractions of each surface's length lie across the camber line from each
    other at the top of the blade, within `tip_match_drift`, else 2.

    Matched all round, a row with a tip gap has its blade nodes alike on
    both halves of the blade, from the middle of the trailing edge to the
    leading edge, coarse and refined, by the greater of the two sides'
    densities, so node `h - k` lies across from node `h + k`; held where they
    are put through the sweeps from `tip_match_from` up; and the H block's
    corners paired across the blade. The coarse blade nodes are then spaced
    at half the refined density, and each coarse cell split in two. On a
    blade whose surfaces differ much in length the pairs then lie far apart
    along the camber line, and the H block's lines between them run nearly
    along the blade. See :func:`_symmetric_wall`,
    :meth:`OHPassage._doubled_wall`, :func:`_gap_corners` and
    :func:`_pairing_drift`."""

    tip_match_drift: float = 0.08
    """With `tip_match` 0, the most that points at equal fractions of the
    two surfaces' lengths may lie apart along the camber line, as a fraction
    of its length, on the section at the top of the blade, for the gap to be
    matched node for node all round."""

    tip_match_from: float = 0.7
    """With a tip gap matched node for node all round, the span fraction
    from which the blade nodes are
    held where the algebraic plane put them through the sweeps, rather than
    slid along the blade: not at all there, wholly from the tip of the blade
    up, and between, each node put a smoothstep of the way from where the
    sweeps slid it to where it started, and swept again held there. See
    :meth:`OHPassage._freeze_weight`."""

    tip_end: float = 1.2
    """How far the front of a tip gap's H block stands back along the camber
    line from the nose of the O block's inner face, in half widths across
    it there."""

    tip_end_rear: float = 2.0
    """As `tip_end`, for the rear of the H block from the tail."""

    tip_tail_outline: float = 0.75
    """How far the nodes of a tip gap's inner face round the tail, from A
    back through the cut to D, are spaced along the squared-off tail line as
    the nodes they are offset from lie along the inner face before it is
    squared, rather than evenly: 0 evenly, 1 wholly so, which runs the O
    block's lines there along the blade's normal rather than fanning off it
    where the outline's nodes bunch round the trailing edge, but bunches the
    H block's rear face, those same nodes, as much. See :func:`_gap_square`."""

    tip_O_square: float = 0.25
    """How far each line of a tip gap's O block arrives at the blade's
    outline along its normal rather than straight from the inner face: 0
    straight, 1 along the normal, the line a cubic leaving the inner face
    straight either way. See :func:`_gap_O_lines`."""

    tip_nose_outline: float = 0.75
    """As `tip_tail_outline`, for the nodes round the nose, from B to C,
    which the H block's front face shares."""

    tip_bow: float = 0.2
    """How far the rear of a tip gap's H block bows aft at its middle, as a
    fraction of the way to where the O block's inner face crosses the cut."""

    tip_front_lean: float = 20.0
    """Most the O block's lines from a tip gap's front corners B and C lean
    off the blade's normal, degrees, so that the corners slide along the
    inner face towards each other's place along the camber, and the H
    block's front runs more nearly across it. None if zero."""

    tip_bow_front: float = 0.1
    """How far the front of a tip gap's H block bows forward at its middle,
    normal to the line between its front corners, as a fraction of the way
    to the O block's inner face round the nose."""

    dw_max: float = 0.15
    """Most the blend of the blocks ahead of and behind the blade towards
    nodes even in theta across the pitch (:func:`_theta_even`) grows from
    one station to the next towards where the blocks end, so the shear is
    spread along them."""

    theta_lean: float = 10.0
    """With it, the blend of :func:`_theta_even` found on every control plane
    first, and each station's weight then held, plane to plane up the span,
    to no more than leans the grid lines between neighbouring control planes
    this much further, degrees, by how far a node moves with the weight; the
    planes blended again under those. Zero blends each plane alone."""

    geometric_O_match: float = 3.0
    """How far each refined O block line's outer cell, grown geometrically
    from the wall spacing, may move from the
    geometric towards the first cell beyond the O block: within this factor
    of it either way, smoothed round the blade over `o_match_smooth` lines,
    the line then double-sided stretched between the two. One, purely
    geometric."""

    tip_rear_multiple: int = 4
    """What the cells round a tip gap's tail, the H
    block's count across, are a multiple of."""

    tip_rear_lean: float = 30.0
    """Most the O block's lines from a tip gap's
    rear corners A and D lean off the blade's normal, degrees, so that each
    corner slides along the inner face to where its line bisects the turn
    between the side and the rear line. None if zero."""

    tip_rear_max: int = 37
    """The most nodes round a tip gap's tail, from
    A through the cut to D, the H block's count of nodes across. No cap if
    zero."""

    tip_blend: float = 0.05
    """Length, as a fraction of the camber line, over which the spacing
    across a tip gap's H block eases from each end face's to even, being
    even in between. 0 blends from the front's to the rear's along the whole
    block instead."""

    # Internals: sample counts, smoothing widths and limits, best left alone

    n_merid: int = 4001
    """Samples along a span line from which its conformal coordinate is integrated."""

    n_smooth_normal: int = 10
    """Passes of a three-point average over the blade normals the O block is
    extruded along, so they fan round corners rather than cross."""

    slope_max: float = 1.0
    """Largest `dtheta/dm'` the initial periodic line is extended at behind the
    blade, being the tangent of 45 degrees. Only a starting guess: the line floats."""

    slope_max_le: float = 0.5
    """Largest `dtheta/dm'` the initial periodic line is extended at ahead of the
    blade, about 27 degrees: at a steeper inlet metal angle, it skews the blocks
    there more than the sweeps undo."""

    end_fit: float = 0.05
    """Fraction of chord at each end of the camber line its end slope is fitted
    over, to average out an uneven parameterisation of the annulus."""

    n_end_fit: int = 101
    """Samples, evenly spaced in chord, over which each end slope is fitted."""

    polish_power: float = 4.0
    """Power the polish raises each cell corner's shape measure to, so that the
    worst cells dominate rather than being traded for small gains elsewhere."""

    n_match_span: int = 3
    """Corrections of the main span's end spacing to meet a tip gap's."""

    n_match_rear: int = 4
    """Corrections of the blade's spacing at the corners behind it, to match
    the cells across the passage there."""

    n_wall: int = 5001
    """Samples along the blade, which nodes on it are placed on."""

    corner_mid: float = 0.1
    """How far round the blade from either edge a corner must be to count as part
    way along it, for :attr:`_Tune.cor_fac_mid`; the leading edge taken half way
    round."""

    polish_coarse_rings: int = 2
    """How many cells out from each poor cell of a swept plane are polished."""

    hold_coarse: int = 1
    """Layers of cells off the blade the polish of a swept plane leaves be."""

    polish_fine_rings: int = 3
    """How many cells out from each poor cell of the refined plane are polished."""

    polish_taper: float = 30.0
    """Weight of the polish's penalty on how far each node moves, growing to
    this at the edge of the polished region, so the spacing along the grid
    lines does not step where it stops. See :func:`elliptic.polish`."""

    hold_fine: int = 3
    """Layers of cells off the blade the polish of the refined plane leaves be."""

    dn_smooth_spf: float = 0.02
    """Width, in span fraction, of the Gaussian the spacing along each plane's
    block to the mixing plane is smoothed over across span, so neighbouring
    planes' stations agree. See :meth:`OHPassage.mesh_row`."""

    tip_o_turn: float = 0.5
    """Most a tip gap's O line is squared to the outline, `tip_O_square`, times
    its length and the outline's curvature there, so that where the outline
    turns fast, round the nose, neighbouring lines bent towards its normals do
    not cross. See :func:`_gap_O_lines`."""

    tip_o_ease: int = 8
    """Nodes round the blade the squaring of a tip gap's O lines is eased over,
    so neighbouring lines bend alike."""

    o_carry_sound: float = 30.0
    """Worst corner of a carried plane's O block cell, rebuilt from offsets,
    under which its nodes are taken from their neighbours round the blade
    instead [deg]. See :func:`_O_from_offsets`."""

    o_carry_margin: int = 2
    """Nodes either side of each such stretch taken from their neighbours too."""

    o_match_smooth: float = 8.0
    """Lines round the blade over which `geometric_O_match` smooths how far the
    O block's outer cells move, the deviation of a Gaussian."""

    o_corner_width: float = 4.0
    """Lines either side of each corner of the O block over which its outer
    cell is drawn, within `geometric_O_match`, to the geometric mean of the
    first cells of the other lines out of the corner. Zero leaves it as
    smoothed."""

    side_corner_cells: float = 0.0
    """Fraction of the cells along each side of the blade over which the
    first cell along the O block's outer face at each corner is scaled to
    the one round the front or rear there. Zero refines the face as the
    blade. See :meth:`_face_u`."""

    passage_end_cells: float = 0.33
    """Fraction of the cells across the passage over which its first cells
    ahead of and behind the blade are scaled to those of the blocks round
    the leading and trailing edges they meet. Zero leaves them as laid out.
    See :meth:`_passage_u`."""

    extrude_match_cells: float = 0.33
    """Fraction of the stations of an extruded inlet or outlet block over
    which each line's first cell, matched to the last inside the cut on the
    line it continues, fades back to the median's. Zero spaces every line
    alike. See :meth:`OHPassage._extrude_fractions`."""

    end_corner_cells: float = 0.33
    """Fraction of the cells along the blocks ahead of and behind the blade
    over which their first cell at the blade is scaled to the geometric
    mean of the other first cells at the corners there. Zero leaves it as
    laid out. See :meth:`_ends_u`."""

    extrude_gap: float = 0.3
    """Most a line of an extruded inlet or outlet block departs from the turn of
    the pitch-mean line, as a fraction of the gap to its nearer neighbour."""

    n_ends_min: int = 8
    """Least coarse cells along the blocks ahead of and behind the blade."""

    n_side_min: int = 4
    """Least coarse cells along each side of the passage block."""

    coarse_factor: float = 2.0
    """How many times finer than `ds_free` the coarse mesh is everywhere, which
    leaves the refined mesh's cell sizes as they are."""

    er_coarse: float = 1.3
    """Expansion ratio along the blade of the coarse mesh, away from where
    `turn_max` clusters it round the edges."""

    coarse_ends_factor: float = 1.5
    """How many times more coarse cells than the rest of the coarse mesh the
    blocks ahead of and behind the blade have along them, at least `n_ends_min`."""

    orth: float = 1.0
    """How far, as a fraction of the way, each sweep draws the coarse nodes one
    cell off the blade towards the blade's normal at their blade node, at the
    distance they lie. Sliding the blade nodes squares the grid there only where
    the nodes off the blade lie over distinct points of it; round an edge much
    smaller than a coarse cell, this squares it. See
    :func:`ember.elliptic.winslow_sweeps`."""

    slide_mid: float = 1.0
    """How far each sweep slides the blade nodes midway between the leading and
    trailing edges towards the foot of the perpendicular from the node outside
    them, as a fraction of the way; none at the edges, so their clustering stays
    put. See :meth:`OHPassage._slide_factors`."""

    turn_fine: float = 10.0
    """Most the blade turns across one refined cell along it, degrees: an edge
    whose radius is small against `ds_LE` or `ds_TE` is clustered further, to
    resolve it."""

    ends_factor: int = 2
    """How many times more refined cells than `ds` and `ER_stream` give the
    blocks ahead of and behind the blade have along them, each split evenly."""

    n_smooth_turn: int = 5
    """Samples of the blade, of `n_wall`, its turning is averaged over to set
    the coarse spacing round its edges: few, so the curvature of an edge much
    smaller than a cell is not averaged away."""

    n_spline_o: int = 16
    """Samples per coarse cell of the spline along each line of the O block that
    its refined nodes are respaced along."""

    freeze_le: float = 0.05
    """Fraction of the blade's perimeter either side of the leading edge over
    which the blade nodes are moved back to where they started after the sliding
    sweeps and held through the sweeps after: the sweeps slide them round the
    nose, all one way, which leaves the clustering there to one side of it and
    the cells past it growing twice as fast as they should. Zero for none."""

    freeze_le_ramp: float = 0.15
    """Fraction of the perimeter beyond :data:`freeze_le` over which that eases
    off."""

    edge_gap: float = 0.02
    """Least fraction of the blade's perimeter between C4 and C1 of the O block
    and the leading edge either side of them: on a twisted rotor the corner
    rules can put C4 on the nose, which folds the block ahead of the blade.
    Zero for none."""

    theta_skew_min: float = 45.0
    """Least cell corner angle, degrees, the blend of the blocks ahead of and
    behind the blade towards nodes even in theta may take a cell to, unless it
    was already less, when it may not be made worse (:func:`_theta_even`)."""

    slide_share: float = 0.3
    """Share of the sweeps given to sliding where the blade nodes are then moved
    back and held, the rest swept held, so the two passes cost one; None for the
    full count each. More to the held pass evens the O block's outer face
    better."""

    min_gap: float = 0.1
    """Least fraction of its length as built a gap between blade nodes keeps as
    they slide: only a guard against a run of them collapsing, the crowding the
    sliding wants at the trailing edge being well above it."""

    gap_cap: float = 0.5
    """Most the O block of a tip gap reaches into the blade, as a fraction of the
    radius of the largest circle inside it touching it at a node: the two sides'
    inner faces then stay apart, and inside the radius of curvature, so the
    normals do not cross before them."""

    gap_bow_decay: float = 2.0
    """Length over which each end face's bow eases out along a tip gap's H
    block, in widths of that face."""

    n_gap_scan: int = 2001
    """Points along the camber line at which the width of a tip gap's inner face
    is measured, to place the ends of its H block."""

    def __post_init__(self):
        def positive(v):
            return v > 0.0

        def fraction(v):
            return 0.0 <= v <= 1.0

        checks = {
            "relax": (lambda v: 0.0 < v <= 1.0, "in (0, 1]"),
            "spf_corners": (fraction, "from 0 to 1"),
            "corner_ref": (lambda v: v in ("spf", "ends"), "spf or ends"),
            "polish_below": (lambda v: 0.0 <= v < 90.0, "from zero to 90 degrees"),
            "t_tip_O": (lambda v: 0.0 < v < 0.5, "between zero and 0.5"),
            "tip_match": (lambda v: v in (0, 1, 2), "0, 1 or 2"),
            "tip_match_from": (lambda v: 0.0 <= v < 1.0, "in [0, 1)"),
            "tip_nose_outline": (fraction, "in [0, 1]"),
            "tip_tail_outline": (fraction, "in [0, 1]"),
            "tip_blend": (lambda v: v >= 0.0, "zero or more"),
            "correct_every": (lambda v: v >= 0, "zero or more"),
            "correct_refactor": (positive, "positive"),
            "correct_contract": (lambda v: 0.0 < v < 1.0, "in (0, 1)"),
        }
        for name in ("t_O", "cut_mixing", "tip_match_drift", "tip_end", "tip_end_rear"):
            checks[name] = (positive, "positive")
        for name, (ok, text) in checks.items():
            v = getattr(self, name)
            if not ok(v):
                raise ValueError(f"{name}={v!r} must be {text}.")
        _check_count("nk_tip_O_min", self.nk_tip_O_min)


def _check_count(name, n):
    """Refuse count `n` of setting `name` unless a multigrid level can halve
    it: nine or more, one above a multiple of eight."""
    if n < 9 or (n - 1) % 8:
        raise ValueError(
            f"{name}={n} is not a count a multigrid level can halve; "
            f"give it nine or more, one above a multiple of eight."
        )


class OHPassage(Mesher):
    """Generate a mesh with an O block round each blade and a passage block
    between neighbouring O blocks, smoothed coarse and refined where needed."""

    type: ClassVar[str] = "ohp"

    labels: ClassVar[tuple] = LABELS
    """Names of the blocks of a row, in the order :meth:`plane` returns them."""

    meshes_sweep: ClassVar[bool] = True

    tune: _Tune = dataclasses.field(default_factory=_Tune)
    """The numbers the mesher is tuned by; see :class:`_Tune`."""

    dspf_mid: float = 0.03
    """Spanwise grid spacing at midspan, as a fraction of span."""

    resolution_factor: float = 1.0
    """Multiply the number of points in each direction, keeping relative spacings."""

    ds: float = 0.02
    """Cell size along the blade between its edges, next to the O block and
    across the passage there, as a fraction of meridional chord."""

    ds_LE: float = 0.01
    """Cell size along the blade at the leading edge, as a fraction of
    meridional chord."""

    ds_TE: float = 0.001
    """Cell size along the blade at the middle of the trailing edge, as a
    fraction of meridional chord: a round trailing edge is much smaller than
    the leading edge, and too few nodes round it fan the O block's lines
    sharply off it."""

    ds_free: float = 0.05
    """Cell size in the free stream, as a fraction of meridional chord; the
    coarse mesh that is smoothed is `coarse_factor` times finer."""

    cut: float = 0.3
    """Distance from the leading edge of the first row and the trailing edge
    of the last at which the blocks ahead of and behind the blade end, and
    beyond which the inlet and outlet blocks are extruded rather than swept,
    in meridional chords."""

    njtip_min: int = 9
    """Least number of spanwise nodes across a tip gap, one more than a
    multiple of eight."""

    def __post_init__(self):
        for name in ("ds", "ds_LE", "ds_TE", "ds_free", "cut"):
            if not getattr(self, name) > 0.0:
                raise ValueError(f"{name}={getattr(self, name)!r} must be positive.")
        if not max(self.ds_LE, self.ds_TE) <= self.ds <= self.ds_free:
            raise ValueError(
                f"Need ds_LE and ds_TE <= ds <= ds_free, got {self.ds_LE}, "
                f"{self.ds_TE}, {self.ds}, {self.ds_free}."
            )
        _check_count("njtip_min", self.njtip_min)

    def to_dict(self):
        """Return the settings as :meth:`Node.to_dict` does, with only the
        numbers in `tune` that differ from their defaults, and none of it if
        none do."""
        data = super().to_dict()
        default = _Tune().to_dict()
        tune = {k: v for k, v in data.pop("tune").items() if v != default[k]}
        if tune:
            data["tune"] = tune
        return data

    #
    # THE STAGE
    #

    def forward(self, machine, spacing):
        """Generate a Grid for a designed machine."""
        for i_row, row in enumerate(machine.rows):
            if not row.blade.round_TE:
                raise ValueError(
                    f"Row {i_row} has a blunt trailing edge, which the "
                    f"{type(self).__name__} mesher does not grid: its blade nodes "
                    f"slide along the blade, and would gather at the corners. "
                    f"Round it off with round_TE: true."
                )
            if row.tip_gap < 0.0:
                raise ValueError(
                    f"Row {i_row} has a negative tip gap of {row.tip_gap:.3g} m; "
                    f"the {type(self).__name__} mesher grids casing clearances only."
                )

        self._check_endwalls()
        spf, jtip = self._span_grid(machine, spacing)

        blocks = []
        for i_row, row in enumerate(machine.rows):
            j = jtip if row.tip_gap else None
            blocks.append(self.mesh_row(machine, i_row, spacing, spf, j))

        _stitch_mixing_planes(
            [row[: len(self.labels)] for row in blocks],
            self.labels,
        )
        self._relax_outlet([b for row in blocks for b in row], machine.annulus, spf)
        grid = ember.grid.Grid([b for row in blocks for b in row])
        grid.connectivity.periodic.pair()
        return grid

    def _span_grid(self, machine, spacing):
        """Return the spanwise grid every row is meshed on, and the index of
        the node at the top of the blade, None without a tip gap."""
        tip = self._tip_fraction(machine)
        span_ref = machine.annulus.evaluate_span(1)
        spf = self.spanwise_grid(spacing.hub / span_ref, spacing.casing / span_ref, tip)
        logger.debug(f"{type(self).__name__} spanwise points: {len(spf)}")

        # The node at the top of the blade, which the blocks in a clearance
        # start from. Found by value rather than counted out of
        # `spanwise_grid`, which puts it there exactly.
        jtip = None
        if tip:
            jtip = int(np.argmin(np.abs(spf - (1.0 - tip))))
            assert spf[jtip] == 1.0 - tip, f"spf={spf[jtip]} against {1.0 - tip}"
        return spf, jtip

    def _tip_fraction(self, machine):
        """Return the tip gap of the rows that have one, as a fraction of span.

        Every row is meshed on one spanwise grid, because a mixing plane
        matches its two sides span for span, and the blocks in a clearance
        meet the rest at a node at the top of the blade. So the rows that
        have a gap must have the same one as a fraction of span, the span of
        a row being its mean over its leading and trailing edges, as the blade
        has it when its clearance is given as a fraction of span.
        """
        annulus = machine.annulus
        frac = np.array(
            [
                row.tip_gap / np.mean(annulus.evaluate_span([2 * i + 1, 2 * i + 2]))
                for i, row in enumerate(machine.rows)
            ]
        )
        gaps = frac[frac > 0.0]
        if not gaps.size:
            return 0.0
        if not np.allclose(gaps, gaps[0], rtol=1e-6):
            raise ValueError(
                f"The {type(self).__name__} mesher puts a grid line at the top "
                f"of the blade, and every row is meshed on one spanwise grid, so "
                f"the rows that have a clearance must have the same one as a "
                f"fraction of span. This machine has {frac}."
            )
        return float(gaps[0])

    def spanwise_grid(self, dspf_hub, dspf_casing, tip=0.0):
        """Return the spanwise grid vector given hub and casing spacings.

        With a tip gap, a fraction `tip` of span, two pieces joined at a node
        at the top of the blade: the main span, and at least
        :attr:`njtip_min` nodes across the gap, spaced at the casing spacing
        or finer, the main span ending as finely as the gap begins. Evenly
        spaced, the spacings ignored, with :attr:`inviscid_endwalls`.
        """
        if self.inviscid_endwalls:
            return self._uniform_span(self.dspf_mid / self.resolution_factor, tip)
        if not tip:
            return util.resample(
                clusterfunc.double.free(
                    dspf_hub, dspf_casing, self.dspf_mid, self.tune.ER_span
                ),
                self.resolution_factor,
                mult=8,
            )

        L_main = 1.0 - tip
        # Growing from the walls at ER_span, no cell coarser than the gap
        # spread evenly over its least count
        dspf_even = tip / (self.njtip_min - 1)
        dspf_tip = min(dspf_casing, dspf_even)
        try:
            spf_tip = clusterfunc.double.free(
                dspf_tip, dspf_tip, dspf_even, self.tune.ER_span, L_main, 1.0
            )
        except clusterfunc.exceptions.ClusteringException:
            spf_tip = clusterfunc.double.fixed(
                dspf_tip, dspf_tip, self.njtip_min, L_main, 1.0
            )
        spf_tip = util.resample(spf_tip, self.resolution_factor, mult=8)
        # Each piece is a block's span in the clearance, so each a count a
        # multigrid level can halve, and the gap no fewer than asked.
        spf_tip = util.resample_to(
            spf_tip, max(_up8(len(spf_tip) - 1) + 1, self.njtip_min)
        )

        # The main span ends as finely as the gap begins once resampled. Its
        # end spacing grows as the resolution falls, but the gap's is held by
        # its least count, and the two once met at a step of twice.
        d_end = spf_tip[1] - spf_tip[0]
        d_main = d_end * self.resolution_factor
        for _ in range(self.tune.n_match_span):
            spf_main = clusterfunc.double.free(
                dspf_hub, d_main, self.dspf_mid, self.tune.ER_span, 0.0, L_main
            )
            spf_main = util.resample(spf_main, self.resolution_factor, mult=8)
            spf_main = util.resample_to(spf_main, _up8(len(spf_main) - 1) + 1)
            d_main *= d_end / (spf_main[-1] - spf_main[-2])
        spf_main[-1] = spf_tip[0] = L_main
        return np.concatenate((spf_main[:-1], spf_tip))

    def control_planes(self, spf):
        """Return the indices of the spanwise planes that are smoothed.

        :attr:`n_span_smooth` distinct planes, as near even steps in span
        fraction as the grid allows, rather than even steps in index, which
        would follow the spanwise clustering and crowd them at the endwalls;
        always including the hub and the casing. Where the grid is coarser
        than the steps, the planes nearest them would coincide, so each is
        moved on to the next free plane. Every plane if
        :attr:`n_span_smooth` is zero.
        """
        nj = len(spf)
        n = self.tune.n_span_smooth
        if not n or n >= nj:
            return np.arange(nj)
        target = np.linspace(spf[0], spf[-1], n)
        j = np.rint(np.interp(target, spf, np.arange(nj))).astype(int)
        # Distinct and in order, the hub and casing held: each at least one
        # past the one before, and leaving room for those after
        for k in range(1, n):
            j[k] = max(j[k], j[k - 1] + 1)
        for k in range(n - 2, -1, -1):
            j[k] = min(j[k], j[k + 1] - 1)
        return j

    #
    # GEOMETRY
    #

    def _wall(self, blade, cmap, t_O, corners=None, pitch=None, section=None):
        """Return the blade and where the corners of the O block sit on it.

        The blade as one closed line (2, n_wall), evenly spaced in arc length,
        in the O block's order, from the middle of the trailing edge round the
        lower side; the arc lengths along it of the ends of its pieces, from
        the middle of the trailing edge through each corner of the O block;
        and which of those ends are C3, C4, C1, C2. The corners at `corners`,
        fractions of the way round, if given; else C3 and C1 from the bounding
        box, C2 at the throat, the point of the upper side's outline nearest
        C3 on the next blade `pitch` above, and C4 where the normal out of C1
        on the blade below lands on the lower side's; or, where theta rises
        along the chord, mirrored: C2 and C4 from the box, C3 at the point of
        the lower side's outline nearest C2 on the blade below, and C1 where
        the normal out of C4 on the blade above lands on the upper side's.
        """
        wall, half, s_LE = _closed_wall(blade, cmap, section)
        S = util.arc_length(wall)
        s_lo, s_hi = half, S - half
        s = np.linspace(0.0, S, self.tune.n_wall)
        wall = _at_arc(wall, s)
        if corners is not None:
            marks = np.array([0.0, *(np.asarray(corners) * S), S])
            return wall, marks, (1, 2, 3, 4), s_LE

        # The outline of the O block, turned by the stagger about the leading
        # edge; its corners are where it comes nearest those of its box.
        outline = wall + t_O * _outward_normals(wall, tune=self.tune)
        # The box turned about the furthest point upstream, which sets the
        # corners as they were tuned; the leading edge proper, `s_LE`, sets
        # the spacing along the blade and the tip gap's sides.
        i_LE = int(np.argmin(wall[0]))
        s_up = s[i_LE]
        chord = wall[:, 0] - wall[:, i_LE]
        stag = np.clip(
            np.arctan2(chord[1], chord[0]),
            -np.radians(self.tune.stagger_max),
            np.radians(self.tune.stagger_max),
        )
        c, sn = np.cos(stag), np.sin(stag)
        rel = outline - wall[:, i_LE : i_LE + 1]
        rot = np.stack((c * rel[0] + sn * rel[1], -sn * rel[0] + c * rel[1]))
        lo, hi = rot.min(axis=1), rot.max(axis=1)
        box = {
            "C3": (hi[0], lo[1]),
            "C4": (lo[0], lo[1]),
            "C1": (lo[0], hi[1]),
            "C2": (hi[0], hi[1]),
        }
        s_C = {
            k: s[int(np.argmin(np.hypot(rot[0] - b[0], rot[1] - b[1])))]
            for k, b in box.items()
        }
        gap = 0.02 * S
        normals = _outward_normals(wall, tune=self.tune)

        def nearest(name, side, other, shift):
            """Corner `name` at the point of the outline in `side` nearest
            corner `other` on the blade `shift` pitches above."""
            i = int(np.argmin(np.abs(s - s_C[other])))
            to = outline - (outline[:, i] + shift * pitch * _J)[:, None]
            s_C[name] = s[side][np.argmin(np.hypot(*to[:, side]))]

        def normal_lands(name, side, other, shift):
            """Corner `name` where the normal out of corner `other` on the
            blade `shift` pitches above lands on the outline in `side`, if it
            does."""
            i = int(np.argmin(np.abs(s - s_C[other])))
            A = outline[:, i] + shift * pitch * _J
            n = normals[:, i]
            rel = outline[:, side] - A[:, None]
            ahead = rel[0] * n[0] + rel[1] * n[1] > 0.0
            off = np.abs(rel[0] * n[1] - rel[1] * n[0])
            if ahead.any():
                s_C[name] = s[side][ahead][np.argmin(off[ahead])]

        # The passage block's edges, corner to corner across the passage,
        # leave the O block square, where a box corner far along a staggered
        # blade would start the blocks round it as slivers: at the rear, one
        # side's corner nearest the other's on the next blade; at the front,
        # one side's where the normal out of the other's on the next blade
        # lands. With theta falling along the chord, or unstaggered, these
        # place the upper rear and lower front corners; rising, mirrored.
        if np.arctan2(chord[1], chord[0]) <= 0.0:
            # Clear of the trailing edge, and in order round the blade.
            s_C["C3"] = np.clip(s_C["C3"], s_lo + gap, s_up - 3 * gap)
            s_C["C4"] = np.clip(s_C["C4"], s_C["C3"] + gap, s_up - gap)
            nearest("C2", (s > s_up + 3 * gap) & (s < s_hi - gap), "C3", 1)
            s_C["C1"] = np.clip(s_C["C1"], s_up + gap, s_C["C2"] - gap)
            normal_lands("C4", (s > s_C["C3"] + gap) & (s < s_up - gap), "C1", -1)
        else:
            s_C["C2"] = np.clip(s_C["C2"], s_up + 3 * gap, s_hi - gap)
            s_C["C1"] = np.clip(s_C["C1"], s_up + gap, s_C["C2"] - gap)
            nearest("C3", (s > s_lo + gap) & (s < s_up - 3 * gap), "C2", -1)
            s_C["C4"] = np.clip(s_C["C4"], s_C["C3"] + gap, s_up - gap)
            normal_lands("C1", (s > s_up + gap) & (s < s_C["C2"] - gap), "C4", 1)

        if self.tune.edge_gap > 0.0:
            # Clear of the leading edge proper, s_LE
            s_C["C4"] = min(s_C["C4"], s_LE - self.tune.edge_gap * S)
            s_C["C1"] = max(s_C["C1"], s_LE + self.tune.edge_gap * S)
        corners = [s_C[k] for k in ("C3", "C4", "C1", "C2")]
        return wall, np.array([0.0, *corners, S]), (1, 2, 3, 4), s_LE

    def _geometry(self, blade_wall, t_O, f_wall):
        """Return the blade, its coarse nodes, and the O block outline.

        The blade and its corners as :meth:`_wall` returns them, and the
        nodes at fractions `f_wall` of each piece of it between them, the
        same at every span.
        """
        wall, marks, i_C, s_LE = blade_wall
        s = np.linspace(0.0, marks[-1], self.tune.n_wall)
        n_wall = [len(f) - 1 for f in f_wall]
        s_loop = np.concatenate(
            [
                (a + f * (b - a))[(0 if j == 0 else 1) :]
                for j, (a, b, f) in enumerate(zip(marks[:-1], marks[1:], f_wall))
            ]
        )
        loop = _interp_rows(s_loop, s, wall)
        loop[:, -1] = loop[:, 0]
        outer = loop + t_O * _outward_normals(loop, tune=self.tune)
        ends = np.cumsum((0, *n_wall))
        i3, i4, i1, i2 = (int(ends[i]) for i in i_C)
        return {
            "wall": wall,
            "marks": marks,
            "i_C": i_C,
            "s_loop": s_loop / marks[-1],
            "sigma_LE": s_LE / marks[-1],
            "loop": loop,
            "outer": outer,
            "C1": outer[:, i1],
            "C2": outer[:, i2],
            "C3": outer[:, i3],
            "C4": outer[:, i4],
        }

    #
    # LAYOUT
    #

    def layout(self, machine, i_row, spacing, spf):
        """Fix the coarse counts of one row, and how it is refined, at mid-span.

        The blocks ahead of and behind the blade end as far from its edges at
        every span fraction `spf` meshed as at mid-span, so where the edges
        are swept they end swept alike, and the counts and clustering fixed
        here mean the same at every span.
        """
        row = machine.rows[i_row]
        blade = row.blade
        n_row = len(machine.rows)
        pitch = 2.0 * np.pi / row.n_blade

        L_up = 1.0 if i_row == 0 else 0.5
        L_dn = 1.0 if i_row == n_row - 1 else 0.5
        m_row = 2.0 * i_row + 1.0
        m_in, m_out = m_row - L_up, m_row + 1.0 + L_dn
        # The edges, swept or not, at mid-span
        m_LE, m_TE = (float(m) for m in m_row + blade.evaluate_swept_m([0.0, 1.0], 0.5))
        extrude_in, extrude_out = i_row == 0, i_row == n_row - 1
        chord = blade.evaluate_chord(0.5)

        # Both a distance in chords from the edge: `cut` to the inlet or
        # outlet, beyond which the blocks are extruded; `cut_mixing` to a
        # mixing plane, the block beyond running on to it, held short of a
        # mixing plane close to the blade
        def to_mixing(m0, m1):
            return min(
                self.tune.cut_mixing * chord,
                self.tune.cut_mixing_frac * _distance(machine.annulus, m0, m1),
            )

        d_in = self.cut * chord if extrude_in else to_mixing(m_LE, m_in)
        d_out = self.cut * chord if extrude_out else to_mixing(m_TE, m_out)
        m_Zin = _m_at_distance(machine.annulus, m_LE, m_in, d_in)
        m_Zout = _m_at_distance(machine.annulus, m_TE, m_out, d_out)

        # Every plane's blocks must end short of its inlet and outlet
        Z = np.array(
            [self.plane_Z(blade, s, m_row, (m_LE, m_TE), (m_Zin, m_Zout)) for s in spf]
        )
        if Z[:, 0].min() <= m_in or Z[:, 1].max() >= m_out:
            raise ValueError(
                f"Row {i_row} has edges swept so far that the blocks ahead of and "
                f"behind its blade, as long as at mid-span, reach m="
                f"{Z[:, 0].min():.4g} and {Z[:, 1].max():.4g}, which leaves no "
                f"room for the blocks between them and the ends of its domain "
                f"at m={m_in:.4g} and {m_out:.4g}."
            )

        cmap = ConformalMap(machine.annulus, 0.5, m_in, m_out)
        r_ref = float(np.mean(_by_theta(blade, 0.5)[0][1]))
        to_c = _chord(blade, 0.5) / self.resolution_factor
        h, h_LE, H = self.ds * to_c, self.ds_LE * to_c, self.ds_free * to_c
        h_TE = self.ds_TE * to_c
        t_O = self.tune.t_O * pitch

        # Coarse along the blade: the free stream size, or less where it
        # turns sharply; the two sides of the passage block alike.
        # The corners found at spf_corners, and kept as fractions of the way
        # round at every span: found at each, the nearest point to a corner of
        # the box can jump from one part of the blade to another between
        # spans, and the planes either side no longer fit together.
        def found_at(x):
            """The corners the rules find at span fraction `x`."""
            cmap_C = ConformalMap(machine.annulus, x, m_in, m_out)
            _, marks_C, _, _ = self._wall(blade, cmap_C, t_O, pitch=pitch)
            return marks_C[1:5] / marks_C[-1]

        if self.tune.corner_ref == "ends":
            corners = np.mean([found_at(0.0), found_at(1.0)], axis=0)
        else:
            corners = found_at(self.tune.spf_corners)
        corners = tuple(float(x) for x in corners)
        blade_wall = self._wall(blade, cmap, t_O, corners)
        wall, marks, i_C, s_LE = blade_wall
        S = marks[-1]
        s = np.linspace(0.0, S, self.tune.n_wall)
        # The coarse mesh, uniformly finer than the free stream.
        Hc = H / self.tune.coarse_factor
        density = _coarse_density(
            wall,
            Hc,
            np.radians(self.tune.turn_max),
            self.tune.er_coarse,
            tune=self.tune,
        )
        i3, _, i1, _ = i_C
        n = [max(1, round(_integral(s, density, a, b))) for a, b in _pairs(marks)]
        n[i3] = n[i1] = max(n[i3], n[i1], self.tune.n_side_min)
        # Where the coarse nodes lie along each piece, by the curvature at
        # mid-span; each plane places its own, see `_coarse_wall`.
        f_wall = tuple(
            (_by_density(s, density, a, b, k) - a) / (b - a)
            for (a, b), k in zip(_pairs(marks), n)
        )
        geo = self._geometry(blade_wall, t_O, f_wall)

        # Refined along the blade: down to h_LE at the leading edge and h_TE
        # at the middle of the trailing edge, growing at ER_surf to h between.
        # Only the counts are fixed here; `_wall_u` places the nodes once the
        # sweeps have moved the coarse ones.
        grow = self.tune.ER_surf - 1.0
        fine = np.minimum.reduce(
            [
                np.full_like(s, h),
                h_LE + grow * np.abs(s - s_LE),
                h_TE + grow * np.minimum(s, S - s),
            ]
        )
        nf_wall = _wall_counts(
            [_integral(s, 1.0 / fine, a, b) for a, b in _pairs(marks)], row.tip_gap
        )

        # Refined across the passage and along the blocks ahead of and behind
        # the blade: h at the O block, growing at ER_stream to the free stream.
        C1, C2, C3, C4 = (geo[k] for k in ("C1", "C2", "C3", "C4"))
        mp_in, mp_Zin, mp_Zout, mp_out = cmap.mp_from_m([m_in, m_Zin, m_Zout, m_out])
        periodic = _periodic_line(blade, cmap, pitch, tune=self.tune)
        d = 1e-6 * (mp_out - mp_in)
        stretch_in = np.hypot(1.0, (periodic(mp_Zin) - periodic(mp_Zin - d)) / d)
        stretch_out = np.hypot(1.0, (periodic(mp_Zout + d) - periodic(mp_Zout)) / d)
        L_P = max(
            np.linalg.norm(C4 - C1 + pitch * _J), np.linalg.norm(C3 - C2 + pitch * _J)
        )
        # Streamwise, in conformal lengths at `r_free`, so free stream cells
        # are as long in metres at any radius; across the passage, where the
        # count per pitch is fixed, in conformal units as they are.
        r_free = float(cmap.xr[1].min())
        k_in, to_mp_in, w_in = _radius_metric(cmap, min(C1[0], C4[0]), mp_Zin, r_free)
        k_out, to_mp_out, w_out = _radius_metric(
            cmap, max(C2[0], C3[0]), mp_Zout, r_free
        )
        L_in = (min(C1[0], C4[0]) - mp_Zin) * stretch_in * k_in
        L_out = (mp_Zout - max(C2[0], C3[0])) * stretch_out * k_out
        n_P, n_in, n_out = (max(2, round(L / Hc)) for L in (L_P, L_in, L_out))
        # The blocks ahead of and behind the blade finer along them still,
        # so the corners of the O block there have room to open out.
        n_in, n_out = (
            max(self.tune.n_ends_min, round(self.tune.coarse_ends_factor * n))
            for n in (n_in, n_out)
        )
        f_P = _mult(_double(h / L_P, h / L_P, H / L_P, self.tune.ER_stream), 8)
        # Ahead of and behind the blade, ends_factor times the cells each
        # way, the clustering kept.
        k = self.tune.ends_factor
        # From the O block, where a cell is `h` in conformal units there,
        # out to the free stream size in metres; laid out in the weighted
        # length and put back at the conformal fractions it lands at.
        f_in = _mult(
            to_mp_in(_single(h * w_in / L_in, H / L_in, self.tune.ER_stream)), 8
        )
        f_out = _mult(
            to_mp_out(_single(h * w_out / L_out, H / L_out, self.tune.ER_stream)), 8
        )
        f_in = 1.0 - np.flip(util.resample_to(f_in, k * (len(f_in) - 1) + 1))
        f_out = util.resample_to(f_out, k * (len(f_out) - 1) + 1)

        k_dn = _radius_metric(cmap, mp_Zout, mp_out, r_free)[0]
        k_up = _radius_metric(cmap, mp_Zin, mp_in, r_free)[0]
        n_dn = (
            1
            if extrude_out
            else _up8(round((mp_out - mp_Zout) * stretch_out * k_dn / Hc))
        )

        dw = float(spacing.surface[i_row]) / r_ref
        f_O = clusterfunc.single.free(dw / t_O, max(h, dw) / t_O, self.tune.ER_O)

        layout = _Layout(
            pitch=pitch,
            m_in=m_in,
            m_Zin=m_Zin,
            m_Zout=m_Zout,
            m_edges=(m_LE, m_TE),
            m_out=m_out,
            d_wall=float(spacing.surface[i_row]),
            t_O=t_O,
            h=h,
            n_wall=tuple(int(x) for x in n),
            f_wall=f_wall,
            corners=corners,
            i_wall=i_C,
            n_O=max(2, round(self.tune.o_coarse_factor * t_O / Hc)),
            n_P=n_P,
            n_in=n_in,
            n_out=n_out,
            n_up=1
            if extrude_in
            else _up8(round((mp_Zin - mp_in) * stretch_in * k_up / Hc)),
            n_dn=n_dn,
            nk_O=len(_mult(f_O, 8)),
            H=H,
            H_coarse=Hc,
            r_free=r_free,
            extrude_in=extrude_in,
            extrude_out=extrude_out,
            wall_density=(s / S, S / fine),
            sigma_LE=s_LE / S,
            chord_frac=_chord(blade, 0.5) / S,
            nf_wall=nf_wall,
            u={
                "P": f_P * n_P,
                "in": f_in * n_in,
                "out": f_out * n_out,
            },
        )
        logger.debug(
            f"Row {i_row} OH passage layout, coarse cells: blade {layout.n_wall}, "
            f"O {layout.n_O}, passage {n_P}, in {n_in}, out {n_out}, up "
            f"{layout.n_up}, dn {layout.n_dn}; refined: blade {layout.nf_wall}, "
            f"P {len(layout.u['P']) - 1}, in {len(layout.u['in']) - 1}, out "
            f"{len(layout.u['out']) - 1}, dn "
            f"{len(layout.u['dn']) - 1 if 'dn' in layout.u else layout.n_dn}, "
            f"O {layout.nk_O} nodes"
        )
        return layout

    #
    # ONE PLANE
    #

    def _wall_coarse_density(self, blade_wall, layout):
        """Return the density of coarse cells along the blade on one plane, as
        :meth:`_wall` returns it, at its `n_wall` samples.

        By :attr:`turn_max` round the blade's own edges on this plane, which
        sharpen and move round it along the span. See :func:`_coarse_density`.
        """
        return _coarse_density(
            blade_wall[0],
            layout.H_coarse,
            np.radians(self.tune.turn_max),
            self.tune.er_coarse,
            tune=self.tune,
        )

    def _wall_half_density(self, blade_wall, layout, blade, spf, section):
        """Return the density of coarse cells along the blade on one plane, as
        :meth:`_wall` returns it, at its `n_wall` samples, for a row whose
        blade nodes are alike on both halves: half the refined density
        :meth:`_plane_density` asks for, which each coarse cell is then split
        in two to make (:meth:`_doubled_wall`)."""
        wall, _, _, s_LE = blade_wall
        S = util.arc_length(wall)
        plane = {
            "wall": wall,
            "sigma_LE": s_LE / S,
            "chord_frac": _chord(blade, spf, section) / S,
        }
        _, dens = self._plane_density(layout, plane)
        return 0.5 * dens / S

    def _coarse_wall(self, blade_wall, density, counts=None):
        """Return coarse cells along each piece of the blade on one plane, as
        :meth:`_wall` returns it, as many as `density` asks for, or with
        `counts`, where they lie along each piece as fractions of its length.
        """
        _, marks, _, _ = blade_wall
        s = np.linspace(0.0, marks[-1], self.tune.n_wall)
        if counts is None:
            return [_integral(s, density, a, b) for a, b in _pairs(marks)]
        return tuple(
            (_by_density(s, density, a, b, k) - a) / (b - a)
            for (a, b), k in zip(_pairs(marks), counts)
        )

    def _coarse_counts(self, i_row, layout, blade_walls, densities):
        """Return `layout` with the coarse cells along each piece of the blade
        the most any plane of `blade_walls`, at `densities`, asks for; the two
        sides of the passage block alike."""
        if layout.symmetric:
            n = list(
                _symmetric_counts(blade_walls, densities, layout.i_wall, tune=self.tune)
            )
            i3, _, i1, _ = layout.i_wall
            n[i3] = n[i1] = max(n[i3], n[i1], self.tune.n_side_min)
            logger.debug(
                f"Row {i_row} coarse cells along the blade, alike on both sides: "
                f"{tuple(n)}"
            )
            return dataclasses.replace(layout, n_wall=tuple(n))
        need = np.zeros(len(layout.n_wall))
        for blade_wall, density in zip(blade_walls, densities):
            need = np.maximum(need, self._coarse_wall(blade_wall, density))
        n = [max(1, round(x)) for x in need]
        i3, _, i1, _ = layout.i_wall
        n[i3] = n[i1] = max(n[i3], n[i1], self.tune.n_side_min)
        logger.debug(
            f"Row {i_row} coarse cells along the blade: {layout.n_wall} at "
            f"mid-span, {tuple(n)} the most any plane asks for"
        )
        return dataclasses.replace(layout, n_wall=tuple(n))

    @staticmethod
    def plane_Z(blade, spf, m_row, m_edges, m_Z):
        """Return where the blocks ahead of and behind the blade end at `spf`,
        annulus `m`: as far from its edges there as `m_Z` is from `m_edges`
        at mid-span. Exactly `m_Z` where the blade is not swept."""
        if not blade.swept:
            return m_Z
        here = m_row + blade.evaluate_swept_m([0.0, 1.0], spf)
        return tuple(float(z + e - e_mid) for z, e, e_mid in zip(m_Z, here, m_edges))

    def plane(self, machine, i_row, layout, cmap, blade_wall, density, section):
        """Return the coarse algebraic mesh of the constant-span plane of
        `cmap`, its blade as :meth:`_wall` returns it and the density of
        coarse cells along it."""
        blade = machine.rows[i_row].blade
        spf = cmap.spf
        pitch = layout.pitch
        pJ = pitch * _J
        m_Zin, m_Zout = self.plane_Z(
            blade, spf, 2.0 * i_row + 1.0, layout.m_edges, (layout.m_Zin, layout.m_Zout)
        )
        mp_in, mp_Zin, mp_Zout, mp_out = cmap.mp_from_m(
            [layout.m_in, m_Zin, m_Zout, layout.m_out]
        )
        if layout.symmetric:
            sym_wall, f_sym = _symmetric_wall(
                blade_wall, density, layout.n_wall, tune=self.tune
            )
            w = self._sym_weight(spf)
            if w < 1.0:
                # Free below, alike near the tip, blended between
                f_free = self._coarse_wall(blade_wall, density, layout.n_wall)
                wall, marks, i_C, s_LE = blade_wall
                marks = (1.0 - w) * np.asarray(marks) + w * np.asarray(sym_wall[1])
                blade_wall = (wall, marks, i_C, s_LE)
                f_wall = tuple((1.0 - w) * a + w * b for a, b in zip(f_free, f_sym))
            else:
                blade_wall, f_wall = sym_wall, f_sym
        else:
            f_wall = self._coarse_wall(blade_wall, density, layout.n_wall)
        g = self._geometry(blade_wall, layout.t_O, f_wall)
        loop, outer = g["loop"], g["outer"]
        C1, C2, C3, C4 = (g[k] for k in ("C1", "C2", "C3", "C4"))
        i3, i4, i1, i2 = layout.i_corners_coarse()
        periodic = _periodic_line(blade, cmap, pitch, tune=self.tune)

        def uniform(n):
            return np.linspace(0.0, 1.0, n + 1)

        # Ahead of and behind the blade, the lines from the corners run at the
        # slope of the periodic line there.
        d = 1e-6 * (mp_out - mp_in)
        slope_in = (periodic(mp_Zin) - periodic(mp_Zin - d)) / d
        slope_out = (periodic(mp_Zout + d) - periodic(mp_Zout)) / d

        def extend(P, mp, slope):
            return np.array([mp, P[1] + slope * (mp - P[0])])

        Q_top, Q_mid = extend(C1, mp_Zin, slope_in), extend(C4, mp_Zin, slope_in)
        R_top, R_mid = extend(C2, mp_Zout, slope_out), extend(C3, mp_Zout, slope_out)

        # The two edges out of each front corner leave it at a third and two
        # thirds of the half turn outside the O block, so the three blocks
        # there start with a third each rather than a sliver between an edge
        # and the O block. Each bends round to where it is going: to its split
        # line along the periodic line, or across the passage to the corner of
        # the next O block, arriving there as it leaves that one. Behind the
        # blade the edges stay straight: the smoothing pulls bent ones back
        # against the O block there, and leaves worse cells than straight.
        along = {}
        for name, i in zip(("C4", "C1", "C2"), (i4, i1, i2)):
            t = outer[:, i + 1] - outer[:, i - 1]
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])
            along[name] = (t, n)

        def leave(name, sense):
            """Direction out of a corner, a third of the way round from its
            O block face going forwards (`sense` 1) or backwards (-1)."""
            t, n = along[name]
            return sense * CORNER_COS * t + CORNER_SIN * n

        to_in = np.array([-1.0, -slope_in]) / np.hypot(1.0, slope_in)
        top_in = _bend(Q_top, -to_in, C1, -leave("C1", -1), layout.n_in)
        mid_in = _bend(Q_mid, -to_in, C4, -leave("C4", 1), layout.n_in)
        # Out of C2, the passage block's edge across to C3 leaves square, so
        # the edge to the outlet takes half of what is left downstream.
        t, n = along["C2"]
        to_out = np.array([1.0, slope_out]) / np.hypot(1.0, slope_out)
        top_out = _bend(C2, (t + n) / np.sqrt(2.0), R_top, to_out, layout.n_out)
        mid_out = _line(C3, R_mid, np.linspace(0.0, 1.0, layout.n_out + 1))
        across_in = _bend(C1 - pJ, leave("C1", 1), C4, -leave("C4", -1), layout.n_P)
        across_out = _line(C2 - pJ, C3, np.linspace(0.0, 1.0, layout.n_P + 1))

        f_O = uniform(layout.n_O)[None, None]
        O = loop[..., None] + f_O * (outer - loop)[..., None]
        f_P = uniform(layout.n_P)
        P = _tfi(
            outer[:, i1 : i2 + 1] - pJ[:, None],
            np.flip(outer[:, i3 : i4 + 1], axis=1),
            across_in,
            across_out,
        )
        front = outer[:, i4 : i1 + 1]
        LE = _tfi(
            mid_in,
            top_in,
            _line(Q_mid, Q_top, _norm_arc(front)),
            front,
        )
        Pin = _tfi(
            top_in - pJ[:, None],
            mid_in,
            _line(Q_top - pJ, Q_mid, f_P),
            P[:, 0, :],
        )
        rear = np.concatenate((outer[:, i3::-1], outer[:, -2 : i2 - 1 : -1]), axis=1)
        TE = _tfi(
            mid_out,
            top_out,
            rear,
            _line(R_mid, R_top, _norm_arc(rear)),
        )
        Pout = _tfi(
            top_out - pJ[:, None],
            mid_out,
            P[:, -1, :],
            _line(R_top - pJ, R_mid, f_P),
        )
        # An inlet or outlet block extruded after the sweeps is one free
        # stream cell of the coarse mesh long through them, its outer face
        # sliding as theirs do.
        if layout.extrude_in:
            r_cut = float(cmap.xr_from_mp(mp_Zin)[1])
            mp_in = mp_Zin - layout.H_coarse * layout.r_free / r_cut
        if layout.extrude_out:
            r_cut = float(cmap.xr_from_mp(mp_Zout)[1])
            mp_out = mp_Zout + layout.H_coarse * layout.r_free / r_cut
        Z_in = np.concatenate((Pin[:, 0, :], LE[:, 0, 1:]), axis=1)
        T_in = np.array([mp_in, Q_top[1]])
        f_up = uniform(layout.n_up)
        up = _tfi(
            _line(T_in - pJ, Q_top - pJ, f_up),
            _line(T_in, Q_top, f_up),
            _line(T_in - pJ, T_in, _norm_arc(Z_in)),
            Z_in,
        )
        Z_out = np.concatenate((Pout[:, -1, :], TE[:, -1, 1:]), axis=1)
        T_out = np.array([mp_out, R_top[1]])
        f_dn = uniform(layout.n_dn)
        dn = _tfi(
            _line(R_top - pJ, T_out - pJ, f_dn),
            _line(R_top, T_out, f_dn),
            Z_out,
            _line(T_out - pJ, T_out, _norm_arc(Z_out)),
        )

        xy = [O, P, LE, Pin, TE, Pout, up, dn]
        for name, blk in zip(LABELS, xy):
            assert np.isfinite(blk).all(), f"non-finite coordinates in {name}"

        return {
            "xy": [blk.astype(DTYPE) for blk in xy],
            "cmap": cmap,
            "wall": g["wall"],
            # The section's own outline, clustered round the leading edge,
            # which the refined nodes there are put on: see `_refined`
            "wall_section": _closed_wall(blade, cmap, section)[0],
            "s_loop": g["s_loop"],
            "sigma_LE": g["sigma_LE"],
            "chord_frac": _chord(blade, spf, section) / util.arc_length(g["wall"]),
        }

    #
    # SMOOTHING
    #

    def _sweep_operators(self, topo, corners=()):
        """Return `topo` with the operators the sweeps need attached."""
        n = topo.n_node
        pitch = topo.pitch or 0.0

        def matrix(rows_nbs, weight):
            rows, cols, vals = [], [], []
            shift = np.zeros(len(rows_nbs))
            for r, nbs in enumerate(rows_nbs):
                for nb in nbs:
                    w = weight(nb, len(nbs))
                    rows.append(r)
                    cols.append(nb[0])
                    vals.append(w)
                    shift[r] += nb[1] * w * pitch
            A = scipy.sparse.csr_matrix((vals, (rows, cols)), shape=(len(shift), n))
            return A, shift

        # Singular points: the mean of where each cell round them would put
        # them, were it a parallelogram. Inlet and outlet nodes: level with
        # the mean of those inside.
        topo.sweep_irr = matrix(topo.irregular_nb, lambda nb, m: nb[2])
        topo.sweep_slide = matrix(topo.slide_nb, lambda nb, m: 1.0 / m)
        topo.sweep_corner = self._corner_patches(topo, corners)
        return topo

    def _corner_patches(self, topo, corners=()):
        """Return the nodes drawn in towards each corner, and how far, a corner
        of `corners`, fractions of the way round the blade, in order, that is
        part way along the blade by `cor_fac_mid` if set.

        For each cell round each singular point, the patch of nodes in its
        block within `n_corner` of the point along each side, or half way
        across the O block; each node with
        the fraction it keeps of its distance from the point. The fraction
        rises as a hyperbolic tangent along the two sides of the patch, and
        inside it is smoothed once from them.
        """
        free = np.ones(topo.n_node, dtype=bool)
        free[topo.fixed] = False
        free[topo.slide] = False
        free[topo.irregular] = False
        out = {"node": [], "corner": [], "shift": [], "f": []}
        # The corners in order round the O block's outer face, as `corners` is
        outer = [int(c) for c in topo.gid[0][:, -1]]
        round_blade = sorted((int(c) for c in topo.irregular), key=outer.index)
        fac = {c: self.tune.cor_fac for c in round_blade}
        if self.tune.cor_fac_mid is not None:
            for c, x in zip(round_blade, corners):
                if min(x, 1.0 - x, abs(x - 0.5)) > self.tune.corner_mid:
                    fac[c] = self.tune.cor_fac_mid
        for c in topo.irregular:
            for b, (g, s) in enumerate(zip(topo.gid, topo.shift)):
                ni, nk = g.shape
                # In the O block, no nearer the blade than half way across it,
                # so the nodes there stay free to square up to it. In any
                # block, no more than half way along or across it, so the
                # patches of corners at either end of a short block do not
                # overlap: claimed node by node by one corner or the other,
                # they draw its lines into a zigzag through the sweeps.
                most = self.tune.n_corner if b else (nk - 1) // 2
                most = min(most, (ni + 1) // 2, (nk + 1) // 2)
                for i0, k0 in np.argwhere(g == c):
                    di_s = [d for d in (1, -1) if 0 <= i0 + d < ni]
                    dk_s = [d for d in (1, -1) if 0 <= k0 + d < nk]
                    for di, dk in itertools.product(di_s, dk_s):
                        m = min(
                            most,
                            ni - i0 if di > 0 else i0 + 1,
                            nk - k0 if dk > 0 else k0 + 1,
                        )
                        if m < 2:
                            continue
                        f = np.tanh(np.arange(1, m + 1) * fac[int(c)])
                        fc = np.ones((m, m))
                        fc[:, 0] = f
                        fc[0, :] = f
                        for a in range(1, m - 1):
                            for b in range(1, m - 1):
                                fc[a, b] = 0.25 * (
                                    fc[a + 1, b]
                                    + fc[a - 1, b]
                                    + fc[a, b + 1]
                                    + fc[a, b - 1]
                                )
                        for a, b in itertools.product(range(m), range(m)):
                            ia, kb = i0 + di * a, k0 + dk * b
                            node = g[ia, kb]
                            if not free[node]:
                                continue
                            free[node] = False
                            out["node"].append(node)
                            out["corner"].append(c)
                            out["shift"].append(s[i0, k0] - s[ia, kb])
                            out["f"].append(fc[a, b])
        return {k: np.array(v) for k, v in out.items()}

    def _freeze_weight(self, topo, plane):
        """Return how far plane `plane`'s blade nodes are held where its
        algebraic plane put them, from 0, slid, to 1, held, rising as a
        smoothstep over `topo.freeze`, the span fractions from which they are
        held and to which wholly, or None to slide them everywhere."""
        if getattr(topo, "freeze", None) is None:
            return 0.0
        a, b = topo.freeze
        return float(_smoothstep(1.0 if b <= a else (plane["cmap"].spf - a) / (b - a)))

    def _sym_weight(self, spf):
        """How far plane `spf`'s blade nodes are laid alike on both halves
        with a tip gap matched all round: only where the gap needs it, none
        below `tip_match_from`, where the corners sit at their own places and
        the nodes by density, as without a gap, rising as a smoothstep to
        wholly at the casing."""
        return float(
            _smoothstep(
                (spf - self.tune.tip_match_from) / (1.0 - self.tune.tip_match_from)
            )
        )

    def _freeze_LE(self, topo, X, plane):
        """Return how far each blade node of plane `X` is held where it
        starts for being near the leading edge: one within `freeze_le` of the
        perimeter of it, easing to none over `freeze_le_ramp` beyond."""
        wall = plane["wall"]
        S = util.arc_length(wall)
        sigma = _wall_arc(plane, topo.scatter(X)[0][:, :, 0].astype(float)) / S
        d = np.abs(sigma - plane["sigma_LE"])
        return 1.0 - _smoothstep(
            (d - self.tune.freeze_le) / max(self.tune.freeze_le_ramp, 1e-12)
        )

    def _sweeps(self, topo, X, n_sweep, plane):
        """Return `X` after `n_sweep` sweeps by :meth:`_sweeps_once`, its blade
        nodes held as far as :meth:`_freeze_weight` says: slid, held where they
        are, or slid, moved back that far towards where they started, and
        swept again held there."""
        floors = self._O_floors(topo, X)
        w = self._freeze_weight(topo, plane)
        if self.tune.freeze_le > 0.0:
            w = np.maximum(w, self._freeze_LE(topo, X, plane))
        if np.all(w <= 0.0):
            return self._sweeps_once(topo, X, n_sweep, plane, False, floors)
        if np.any(w < 1.0):
            # Slid, moved back that far towards where they started, and the
            # plane swept again held there
            n_slide = n_sweep
            if self.tune.slide_share is not None:
                n_slide = max(1, round(self.tune.slide_share * n_sweep))
                n_sweep = max(1, n_sweep - n_slide)
            X0 = X
            X = self._sweeps_once(topo, X, n_slide, plane, False, floors)
            X = self._blade_toward(topo, X, X0, plane, w)
        if self.tune.n_sweep_held is not None:
            n_sweep = self.tune.n_sweep_held
        return self._sweeps_once(topo, X, n_sweep, plane, True, floors)

    def _O_floors(self, topo, X):
        """Return the least length of each cell of each O block line, from
        the blade out, `o_hold` of what it is on plane `X`, the algebraic
        plane the sweeps start from: the first cells (n_wall,) and the rest
        (n_wall, K - 1), for :func:`ember.elliptic.winslow_sweeps`; None
        without `o_hold`."""
        if not self.tune.o_hold:
            return None
        gid, shift = topo.gid[0], topo.shift[0]
        P = topo._values(np.asarray(X, dtype=float), gid, shift)  # (n_wall, nk, 2)
        L = self.tune.o_hold * np.linalg.norm(np.diff(P, axis=1), axis=-1)
        return L[:, 0], L[:, 1:]

    @staticmethod
    def _blade_toward(topo, X, X0, plane, w):
        """Return plane `X` with each of its blade nodes moved `w` of the way
        along the blade, by arc length round it, from where `X` has it to
        where `X0` has it."""
        wall = plane["wall"]
        S = util.arc_length(wall)
        xy = topo.scatter(X)
        s0 = _wall_arc(plane, topo.scatter(X0)[0][:, :, 0].astype(float))
        s1 = _wall_arc(plane, xy[0][:, :, 0].astype(float))
        s = (1.0 - w) * s1 + w * s0
        xy[0][:, :, 0] = _at_arc(wall, np.mod(s, S)).astype(xy[0].dtype)
        return topo.gather(xy)

    def _sweeps_once(self, topo, X, n_sweep, plane, frozen, floors=None):
        """Return `X` after `n_sweep` point-Jacobi sweeps of Winslow's equations.

        By :func:`ember.elliptic.winslow_sweeps`: each sweep the regular and
        singular nodes from their stencils, the inlet and outlet level with
        the nodes inside, the nodes near the corners drawn in, the blade
        nodes of `plane` slid along the blade towards the foot of the
        perpendicular from the node outside them, by :meth:`_slide_factors`,
        and the nodes outside them drawn square to the blade by `orth`, with
        the lines beyond by `orth_taper`.
        """
        pitch = topo.pitch
        W_irr, c_irr = topo.sweep_irr
        W_sl, c_sl = topo.sweep_slide
        cp = topo.sweep_corner
        args = (
            X,
            topo.stencil_g,
            topo.stencil_s,
            pitch,
            (topo.irregular, W_irr.indptr, W_irr.indices, W_irr.data, c_irr),
            (topo.slide, W_sl.indptr, W_sl.indices, W_sl.data, c_sl),
            (cp["node"], cp["corner"], cp["shift"] * pitch, cp["f"]),
        )
        gid, shift = topo.gid[0], topo.shift[0] * pitch
        # The blade as a closed line, and where on it each blade node and the
        # node outside it start.
        line = plane["wall"].T
        s_line = util.cum_arc_length(plane["wall"])
        tree = cKDTree(line[:-1])
        w0 = topo._values(X, gid[:, 0], topo.shift[0][:, 0])
        w1 = topo._values(X, gid[:, 1], topo.shift[0][:, 1])

        # The least gaps between blade nodes, a fraction of those the plane
        # was built with, so a run of nodes cannot gather onto one point.
        s_ref = plane["s_loop"]
        n = gid.shape[0]
        s_ref = np.interp(
            np.linspace(0, len(s_ref) - 1, n), np.arange(len(s_ref)), s_ref
        )
        floor = np.concatenate(
            ([0.0], np.cumsum(self.tune.min_gap * np.diff(s_ref) * s_line[-1]))
        )

        lines = {}
        if self.tune.orth_taper:
            K = gid.shape[1] - 1
            # With floors, out to the outer face too, which is drawn square
            # by none, so the floors hold the whole line
            end = K + 1 if floors is not None else K
            k = np.arange(2, end)
            lines["orth_lines"] = (
                gid[:, 2:end],
                shift[:, 2:end],
                (1.0 - (k - 1) / (K - 1)) ** self.tune.orth_taper,
            )
            if floors is not None:
                lines["floors"] = floors
        slide = 0.0 if frozen else self._slide_factors(plane, line, s_line, tree, w0)

        def sweep(X, n, seg, relax):
            return ember.elliptic.winslow_sweeps(
                X,
                *args[1:],
                (
                    gid[:, 0],
                    gid[:, 1],
                    shift[:, 0],
                    shift[:, 1],
                    line,
                    s_line,
                    floor,
                    *seg,
                ),
                n,
                self.tune.orth,
                slide,
                relax=relax,
                **lines,
            )

        seg = (tree.query(w0)[1], tree.query(w1)[1])
        # Only once the blade nodes are held: how far they slide is set by
        # how many sweeps they slide for
        every = self.tune.correct_every if frozen else 0
        if not every:
            return sweep(X, n_sweep, seg, self.tune.relax)[0]

        # In chunks, each followed by a correction from the residual of one
        # plain sweep, the blade nodes and those outside them, which the
        # sweeps set by sliding and drawing square, held.
        held = np.concatenate((gid[:, 0], gid[:, 1]))
        correct = None
        last = np.inf
        done = 0
        while done < n_sweep:
            n = min(every, n_sweep - done)
            X, *seg = sweep(X, n, seg, self.tune.relax)
            done += n
            # Ended on sweeps, which leave the projections as they want them
            if done >= n_sweep:
                break
            if correct is None:
                correct = elliptic.winslow_correction(topo, X, held)
            R = sweep(X, 1, seg, 1.0)[0].astype(float) - X
            d, strain = correct(R)
            X = (X + d).astype(X.dtype)
            # Still straining the plane, and by not much less than the last:
            # the map no longer fits it, so a new one for the next
            if strain > self.tune.correct_refactor and (
                strain > self.tune.correct_contract * last
            ):
                correct = None
                strain = np.inf
            last = strain
        return X

    def _slide_factors(self, plane, line, s_line, tree, w0):
        """Return how far each blade node at `w0` slides each sweep.

        The nodes off the blade are drawn square to it, by `orth`, and sliding
        the blade nodes under them round the leading and trailing edges only
        drags the clustering there onto the surfaces as the sweeps even the
        first layer. So none there, rising as the square of a sine along each
        side to `slide_mid` midway between them, where the nodes are free to
        find their places along the blade.
        """
        S = s_line[-1]
        sigma = _project(line.T, s_line, tree, np.asarray(w0, dtype=float)) / S
        a = plane["sigma_LE"]
        # From the middle of the trailing edge (0) to the leading edge (1).
        t = np.where(sigma <= a, sigma / a, (1.0 - sigma) / (1.0 - a))
        return self.tune.slide_mid * np.sin(np.pi * np.clip(t, 0.0, 1.0)) ** 2

    def _match_blade(self, topo, X, plane, lay, frac):
        """Return the fractions of each piece of the blade the coarse blade
        nodes of plane `X` lie at, and `X` with them moved to `frac` of their
        pieces if given."""
        xy = topo.scatter(np.asarray(X))
        wall = plane["wall"]
        S = util.arc_length(wall)
        s_c = _wall_arc(plane, xy[0][:, :, 0].astype(float))
        ends = (0, *lay.i_corners_coarse(), len(s_c) - 1)
        own = []
        for a, b in itertools.pairwise(ends):
            own.append((s_c[a : b + 1] - s_c[a]) / (s_c[b] - s_c[a]))
        if frac is None:
            return own, X
        for (a, b), f in zip(itertools.pairwise(ends), frac):
            s_c[a : b + 1] = s_c[a] + f * (s_c[b] - s_c[a])
        xy[0][:, :, 0] = _at_arc(wall, np.mod(s_c, S)).astype(xy[0].dtype)
        return frac, topo.gather(xy)

    def mesh_row(self, machine, i_row, spacing, spf, jtip=None):
        """Return the ember blocks of one row, with a tip gap from `jtip`."""
        lay = self.layout(machine, i_row, spacing, spf)
        # Each plane's blade evaluated once, for its coarse counts and nodes
        blade = machine.rows[i_row].blade
        cmaps = [ConformalMap(machine.annulus, s, lay.m_in, lay.m_out) for s in spf]
        sections = [_section(blade, s) for s in spf]
        if jtip is not None:
            lay = dataclasses.replace(
                lay,
                symmetric=self._full_match(i_row, blade, cmaps[jtip], sections[jtip]),
            )
        walls = [
            self._wall(blade, c, lay.t_O, lay.corners_at(c.spf), section=sec)
            for c, sec in zip(cmaps, sections)
        ]
        if lay.symmetric:
            dens = [
                self._wall_half_density(w, lay, blade, c.spf, sec)
                for w, c, sec in zip(walls, cmaps, sections)
            ]
        else:
            dens = [self._wall_coarse_density(w, lay) for w in walls]
        lay = self._coarse_counts(i_row, lay, walls, dens)
        planes = [
            self.plane(machine, i_row, lay, *cwds)
            for cwds in zip(cmaps, walls, dens, sections)
        ]
        j_mid = int(np.argmin(np.abs(spf - 0.5)))

        topo = self._sweep_operators(
            _plane_topology(planes[j_mid]["xy"], lay.pitch),
            lay.corners if i_row == 0 else (),
        )
        topo.freeze = (
            (self.tune.tip_match_from, float(spf[jtip])) if lay.symmetric else None
        )
        hold = topo.gid[0][:, : self.tune.hold_coarse + 1].ravel()
        X_alg = [topo.gather(p["xy"]) for p in planes]

        # Swept on the control planes, and the correction that made carried
        # to the planes between.
        j_ctrl = np.union1d(self.control_planes(spf), [j_mid])
        correction = [None] * len(spf)
        # Each slides its blade nodes along the blade, to the foot of the
        # perpendicular from the node outside them, which squares the O block
        # there; where they come to rest is kept, as fractions of each piece.
        fracs = [None] * len(spf)
        # With the blade nodes alike on both halves, how far each control
        # plane's lie from its algebraic plane's instead, as fractions of the
        # way round the blade
        slips = [None] * len(spf)
        offsets = [None] * len(spf)
        swept = [None] * len(spf)

        def control(j):
            """Sweep plane j, keeping its correction, blade fractions and O
            block offsets to carry to the planes between, or, with
            `theta_lean`, the plane as swept for :func:`finish`."""
            # Started from the algebraic plane as it is: the sweeps unfold
            # what folds in it without untangling first.
            X = self._sweeps(topo, X_alg[j], self.tune.n_sweep, planes[j])
            # Polished before it is refined too: what the refined plane's
            # polish is then left to do steps the spacing less across the O
            # block's outer face.
            if self.tune.polish_below:
                X += self._polish_shift(
                    topo, hold, topo.scatter(X), self.tune.polish_coarse_rings
                )
            if self.tune.theta_lean:
                swept[j] = X
                logger.debug(f"Row {i_row} spf={spf[j]:.3f} swept")
                return
            xy = topo.scatter(X)
            for side in ("in", "out"):
                xy = _theta_even(
                    xy, lay.pitch, self.tune.theta_skew_min, self.tune.dw_max, side
                )
            X = topo.gather(xy).astype(X.dtype)
            keep(j, X)
            logger.debug(f"Row {i_row} spf={spf[j]:.3f} swept")

        def keep(j, X):
            """Keep control plane j's correction, blade fractions and O block
            offsets."""
            own, _ = self._match_blade(topo, X, planes[j], lay, None)
            fracs[j] = np.concatenate(own)
            if lay.symmetric:
                slips[j] = _wall_frac(topo, X, planes[j]) - _wall_frac(
                    topo, X_alg[j], planes[j]
                )
            correction[j] = X - X_alg[j]
            offsets[j] = _O_offsets(
                topo, X_alg[j] + correction[j], planes[j], tune=self.tune
            )

        def finish():
            """With `theta_lean`, blend every control plane as swept, the
            weights held in step up the span, and keep each."""
            if not self.tune.theta_lean:
                return
            xy = self._theta_even_span(
                [topo.scatter(swept[j]) for j in j_ctrl],
                [planes[j] for j in j_ctrl],
                lay.pitch,
            )
            for j, B in zip(j_ctrl, xy):
                keep(j, topo.gather(B).astype(swept[j].dtype))

        def between(values):
            """Fill in `values` at the planes between the control planes."""
            V = np.stack([values[j] for j in j_ctrl])
            for j in range(len(spf)):
                if values[j] is None:
                    k = np.searchsorted(j_ctrl, j)
                    a, b = j_ctrl[k - 1], j_ctrl[k]
                    w = float((spf[j] - spf[a]) / (spf[b] - spf[a]))
                    values[j] = (1.0 - w) * V[k - 1] + w * V[k]
            return values

        cuts = np.cumsum([n + 1 for n in lay.n_wall])[:-1]

        def carry():
            """Return every plane, the control planes as swept and the rest
            carried from them."""
            corr = between(list(correction))
            X_smooth = [X_alg[j] + corr[j] for j in range(len(spf))]
            # A correction carried between planes leaves their blade nodes
            # off the blade, and round a thin edge out of order along it: put
            # back on it at fractions of each piece between those of the
            # control planes either side.
            if lay.symmetric:
                # Each plane's blade nodes clustered round its own edges, which
                # move along the blade with span as fractions carried between
                # control planes do not: its algebraic plane's, slipped as far
                # as the control planes either side slipped from theirs
                sl = between(list(slips))
                X_smooth = [
                    X
                    if j in j_ctrl
                    else _wall_at_frac(
                        topo,
                        X,
                        planes[j],
                        _wall_frac(topo, X_alg[j], planes[j]) + sl[j],
                    )
                    for j, X in enumerate(X_smooth)
                ]
            else:
                fr = between(list(fracs))
                X_smooth = [
                    X
                    if j in j_ctrl
                    else self._match_blade(
                        topo, X, planes[j], lay, np.split(fr[j], cuts)
                    )[1]
                    for j, X in enumerate(X_smooth)
                ]
            # The O block of each plane between rebuilt from its lines'
            # offsets along and normal to the blade from their blade nodes,
            # interpolated between the control planes: displacements
            # interpolated in the plane would skew lines that leave each
            # control plane's blade square where the blade between turns, as
            # it does where its stagger changes.
            off = between(list(offsets))
            return [
                X
                if j in j_ctrl
                else _O_from_offsets(topo, X, planes[j], off[j], tune=self.tune)
                for j, X in enumerate(X_smooth)
            ]

        for j in j_ctrl:
            control(j)
        finish()
        X_smooth = carry()
        # The refined cells along each piece of the blade counted again
        # between the corners as the sweeps left them, and the refinement
        # along the blade taken from where the coarse nodes lie at mid-span.
        xy_mid = topo.scatter(X_smooth[j_mid])
        lay = self._recount_wall(
            lay, planes, [topo.scatter(X) for X in X_smooth], machine.rows[i_row]
        )
        lay = self._wall_u(lay, planes[j_mid], xy_mid)
        refine = self._refine_matrices(lay, planes[j_mid]["xy"])
        mid = self._refined(xy_mid, planes[j_mid], lay, refine)
        if self.tune.side_corner_cells > 0.0:
            lay = self._face_u(lay, mid)
            refine = self._refine_matrices(lay, planes[j_mid]["xy"])
            mid = self._refined(xy_mid, planes[j_mid], lay, refine)
        if self.tune.passage_end_cells > 0.0:
            lay = self._passage_u(lay, mid)
            refine = self._refine_matrices(lay, planes[j_mid]["xy"])
            mid = self._refined(xy_mid, planes[j_mid], lay, refine)
        if self.tune.end_corner_cells > 0.0:
            lay = self._ends_u(lay, mid)
            refine = self._refine_matrices(lay, planes[j_mid]["xy"])
            mid = self._refined(xy_mid, planes[j_mid], lay, refine)
        lay = self._O_count(lay, planes[j_mid], mid[0], _beyond_O(mid, lay))
        if not lay.extrude_out:
            lay = self._dn_u(lay, mid)
            refine = self._refine_matrices(lay, planes[j_mid]["xy"])
        per_plane = [
            self._refined(topo.scatter(X), plane, lay, refine, respace_dn=False)
            for X, plane in zip(X_smooth, planes)
        ]
        # The block behind the blade to the mixing plane respaced along `i`
        # as `_dn_lines` would each plane, but by its fractions smoothed
        # across span over `dn_smooth_spf`: each plane's alone step from one
        # to the next by more than the cells by the endwalls are high, and
        # lean the spanwise lines there over
        if "dn" in lay.u:
            F = np.stack(
                [
                    _dn_fractions(xy[7], xy[4], xy[5], self.tune.ER_mix)
                    for xy in per_plane
                ]
            )
            w = np.exp(
                -0.5 * ((spf[:, None] - spf[None]) / self.tune.dn_smooth_spf) ** 2
            )
            F = np.tensordot(w / w.sum(axis=1, keepdims=True), F, axes=1)
            for xy, f in zip(per_plane, F):
                xy[7] = _dn_apply(xy[7], f).astype(xy[7].dtype)

        # The control planes are polished once refined, and how far that
        # moves each node carried to the planes between, as the smoothing is.
        # Only those with a cell under `polish_fine_below`; the rest still
        # gathered and scattered, which makes the nodes the blocks share
        # agree to the last bit.
        poor = [
            j
            for j in j_ctrl
            if min(_cell_angles(B).min() for B in per_plane[j])
            < self.tune.polish_fine_below
        ]
        if self.tune.polish_below:
            fine = _plane_topology(per_plane[j_mid], lay.pitch, stencils=False)
            hold = fine.gid[0][:, : self.tune.hold_fine + 1].ravel()
            shift = [None] * len(spf)
            for j in j_ctrl:
                if j in poor:
                    shift[j] = self._polish_shift(
                        fine, hold, per_plane[j], self.tune.polish_fine_rings
                    )
                else:
                    shift[j] = np.zeros_like(fine.gather(per_plane[j]))
            for j, d in enumerate(between(shift)):
                per_plane[j] = fine.scatter(fine.gather(per_plane[j]) + d)

        # The inlet and outlet blocks extruded from the cut, with the same
        # stations at every span, set at mid-span.
        for inlet, use in ((True, lay.extrude_in), (False, lay.extrude_out)):
            if use:
                f = self._extrude_fractions(per_plane[j_mid], planes[j_mid], lay, inlet)
                for xy, plane in zip(per_plane, planes):
                    self._extrude(xy, plane, lay, inlet, f)
        return self.assemble(machine, i_row, lay, planes, per_plane, jtip)

    @staticmethod
    def _cut_face(xy, inlet):
        """Return the cut face (2, nk) of a refined plane `xy`, and the nodes
        one cell inside it, ahead of or behind the blade."""
        if inlet:
            _, _, LE, Pin = xy[:4]
            face = np.concatenate((Pin[:, 0, :], LE[:, 0, 1:]), axis=1)
            inner = np.concatenate((Pin[:, 1, :], LE[:, 1, 1:]), axis=1)
        else:
            TE, Pout = xy[4:6]
            face = np.concatenate((Pout[:, -1, :], TE[:, -1, 1:]), axis=1)
            inner = np.concatenate((Pout[:, -2, :], TE[:, -2, 1:]), axis=1)
        return face.astype(float), inner.astype(float)

    def _theta_even_span(self, xy, planes, pitch):
        """Return the blocks `xy` of control planes `planes`, in order up the
        span, each blended by :func:`_theta_even` on each side, its weights
        no more than those it finds on the plane alone, nor than those of the
        planes either side by more than leans the lines between them
        `theta_lean` further: a node moved by the weight's step from where
        it is towards even in theta, against how far the planes lie apart
        in the meridional plane."""
        tan = np.tan(np.radians(self.tune.theta_lean))
        xy = [list(B) for B in xy]
        for side in ("in", "out"):
            found = [
                _theta_even(
                    B,
                    pitch,
                    self.tune.theta_skew_min,
                    self.tune.dw_max,
                    side,
                    weights=True,
                )
                for B in xy
            ]
            if any(f[1] is None for f in found):
                xy = [f[0] for f in found]
                continue
            W = np.array([f[1] for f in found])
            # Each band in the meridional plane, and how far each station's
            # nodes move per unit weight, scaled by radius
            P, move = [], []
            for (_, _, band, full), plane in zip(found, planes):
                xr = plane["cmap"].xr_from_mp(band[0])
                P.append(xr)
                move.append((xr[1] * np.linalg.norm(full - band, axis=0)).max(axis=1))
            for a in range(1, len(W)):
                apart = np.linalg.norm(P[a] - P[a - 1], axis=0).min(axis=1)
                step = tan * apart / np.maximum(np.maximum(move[a], move[a - 1]), 1e-30)
                W[a] = np.minimum(W[a], W[a - 1] + step)
            for a in range(len(W) - 1, 0, -1):
                apart = np.linalg.norm(P[a] - P[a - 1], axis=0).min(axis=1)
                step = tan * apart / np.maximum(np.maximum(move[a], move[a - 1]), 1e-30)
                W[a - 1] = np.minimum(W[a - 1], W[a] + step)
            logger.debug(
                f"Theta-even blend, side {side}, weights held up the span by "
                f"{np.round((W - np.array([f[1] for f in found])).min(axis=0), 2)} at most"
            )
            xy = [
                _theta_even(
                    B, pitch, self.tune.theta_skew_min, self.tune.dw_max, side, cap=w
                )
                for B, w in zip(xy, W)
            ]
        return xy

    def _extrude_fractions(self, xy, plane, lay, inlet):
        """Return fractions of the way from the cut to the inlet or outlet
        the extruded block's stations lie at, on refined plane `xy`.

        From the size of the cells inside the cut, growing at `ER_stream`
        to the free stream size, in a count a multigrid level can halve: one
        set (n,) for every line, or with `extrude_match_cells`, one for each
        line across the face (n, nk), its first cell its own.
        """
        face, inner = self._cut_face(xy, inlet)
        mp_end = plane["cmap"].mp_from_m(lay.m_in if inlet else lay.m_out)
        mp_cut = float(np.median(face[0]))
        # In conformal lengths at `r_free`, as the blocks inside the cut, so
        # the free stream cell is as long in metres out to the end
        k, to_mp, w = _radius_metric(plane["cmap"], mp_cut, float(mp_end), lay.r_free)
        D = float(np.median(np.abs(mp_end - face[0]))) * k
        h0 = float(np.median(np.linalg.norm(face - inner, axis=0))) * w
        # No cell longer than half the way, which may be short where the
        # inlet or outlet lies not far beyond the cut.
        h1 = min(max(lay.H, h0), 0.5 * D)
        try:
            f = _single(h0 / D, h1 / D, self.tune.ER_stream)
        except clusterfunc.exceptions.ClusteringException:
            f = _uniform(min(h0, h1) / D)
        f = _mult(to_mp(f), 8)
        if self.tune.extrude_match_cells <= 0.0:
            return f
        # Each line's first cell, along `m'`, as long as the last inside the
        # cut on the line it continues, the median's faded back in over
        # `extrude_match_cells` of the stations; the blocks inside the cut
        # are spaced alike along every line no more than the median is
        c = self.tune.geometric_O_match
        own = np.abs(face[0] - inner[0]) / np.abs(mp_end - face[0])
        F = np.clip(own / f[1], 1.0 / c, c)
        n_fade = self.tune.extrude_match_cells * (len(f) - 1)
        return np.stack([_scale_end(f, Fk, n_fade, at_end=False) for Fk in F], axis=1)

    def _extrude(self, xy, plane, lay, inlet, f):
        """Replace the inlet or outlet block of refined plane `xy` by its cut
        face extruded to the inlet or outlet, at fractions `f` of the way.

        Each line leaves the cut along the grid line it continues, and turns
        to meridional with the pitch-mean slope `dtheta/dm'`, which falls
        linearly to nothing over :attr:`straighten` chords, its own
        departure from that mean fading within :data:`extrude_gap` of the
        gap to its nearer neighbour; the nodes across the pitch relax from
        their spacing at the cut to even in theta over :attr:`even_theta`
        chords. The lines at either side of the pitch, one pitch apart at
        the cut, turn alike and stay so.
        """
        face, inner = self._cut_face(xy, inlet)
        mp_end = plane["cmap"].mp_from_m(lay.m_in if inlet else lay.m_out)
        chord = float(np.ptp(plane["wall"][0]))
        slope = (face[1] - inner[1]) / (face[0] - inner[0])
        f = f if np.ndim(f) == 2 else f[:, None]
        mp = face[0][None] + f * (mp_end - face[0])[None]
        d = np.abs(mp - face[0][None])
        sense = np.sign(mp_end - face[0])[None]
        # Every line turns as the pitch-mean slope does, straightening over
        # `straighten` chords; its own departure from that mean, with which
        # it leaves the cut along its grid line, fades within a fraction of
        # the gap to its nearer neighbour, so that lines leaving at different
        # angles cannot cross. The lines at either side of the pitch, images
        # of each other, see the gaps either side of the cut alike.
        R = self.tune.straighten * chord
        run = np.where(d < R, d - d**2 / (2.0 * R), 0.5 * R)
        mean = float(np.mean(slope[:-1]))
        gap = np.diff(face[1])
        near = np.minimum(np.r_[gap[-1], gap], np.r_[gap, gap[0]])
        excess = np.abs(slope - mean)
        reach = self.tune.extrude_gap * near / np.maximum(excess, 1e-300)
        own = reach * -np.expm1(-d / np.maximum(reach, 1e-300)[None])
        theta = face[1][None] + sense * (mean * run + (slope - mean)[None] * own)
        nk = face.shape[1]
        even = theta[:, :1] + lay.pitch * np.linspace(0.0, 1.0, nk)[None]
        w = _smoothstep(d / (self.tune.even_theta * chord))
        theta = (1.0 - w) * theta + w * even
        B = np.stack((mp, theta)).astype(xy[0].dtype)
        B[:, 0] = face
        if inlet:
            xy[6] = B[:, ::-1]
        else:
            xy[7] = B

    def _recount_wall(self, lay, planes, xys, row):
        """Return `lay` with the refined cells along each piece of the blade
        counted between the corners where the sweeps left them, on every
        plane, `xys` the coarse blocks of each of `planes`.

        The layout counts them between the corners as built at mid-span,
        which the sweeps slide along the blade, and on a blade whose edges
        sharpen along the span the curvature asks for more on some planes
        than mid-span: each piece takes the most any plane asks for, so none
        is coarsened by :meth:`_wall_u` to fit. See :func:`_wall_counts`.
        With `lay.symmetric`, each coarse cell split in two instead, by
        :meth:`_doubled_wall`.
        """
        if lay.symmetric:
            return self._doubled_wall(lay)
        need = np.zeros(len(lay.n_wall))
        for plane, xy in zip(planes, xys):
            S = util.arc_length(plane["wall"])
            s_c = _wall_arc(plane, xy[0][:, :, 0])
            c = s_c[np.cumsum((0, *lay.n_wall))] / S
            sigma, dens = _round_twice(*self._plane_density(lay, plane))
            here = [_integral(sigma, dens, a, b) for a, b in itertools.pairwise(c)]
            need = np.maximum(need, here)
        nf_wall = _wall_counts(need, row.tip_gap)
        logger.debug(
            f"Refined cells along the blade: {np.round(need, 1)} asked for, "
            f"{lay.nf_wall} as laid out, {nf_wall} once swept"
        )
        return dataclasses.replace(lay, nf_wall=nf_wall)

    @staticmethod
    def _doubled_wall(lay):
        """Return `lay` refined along the blade by splitting each coarse cell
        in two, for `lay.symmetric`: the coarse nodes already spaced at half
        the refined density, the refined spacing is theirs as the sweeps left
        them, alike on both halves of the blade. Also `sym_kappa`, the refined
        nodes along each half from the middle of the trailing edge, in
        fractional coarse indices."""
        u = dict(lay.u)
        for name, n in zip(("rl", "side_lo", "front", "side_up", "ru"), lay.n_wall):
            u[name] = np.linspace(0.0, n, 2 * n + 1)
        n_half = sum(lay.n_wall) // 2
        u["sym_kappa"] = np.linspace(0.0, n_half, 2 * n_half + 1)
        return dataclasses.replace(lay, nf_wall=tuple(2 * n for n in lay.n_wall), u=u)

    def _plane_density(self, lay, plane):
        """Return the density of refined cells round the blade on `plane`, as
        `_Layout.wall_density`, moved to its own leading edge and raised where
        the blade turns sharply.

        Mid-span's, its fractions of the way round from the middle of the
        trailing edge stretched piecewise linearly either side of the leading
        edge, so the clustering there lands on this plane's, wherever twist
        has moved it, and the trailing edge's stays put. Then at least enough
        that the blade turns by no more than `turn_fine` across a cell, the
        cells growing from there at no more than :attr:`ER_surf`.
        """
        sigma, dens = lay.wall_density
        a, b = lay.sigma_LE, plane["sigma_LE"]
        moved = np.where(
            sigma < a, sigma * b / a, b + (sigma - a) * (1.0 - b) / (1.0 - a)
        )
        wall = plane["wall"]
        sw = _norm_arc(wall)
        # As fractions of this plane's own chord, which the blade's length
        # round it is not in proportion to along the span.
        h = plane["chord_frac"] / lay.chord_frac / np.interp(sw, moved, dens)
        # Cell lengths, as fractions of the way round, to turn by turn_fine.
        d = np.gradient(wall, axis=1)
        phi = np.unwrap(np.arctan2(d[1], d[0]))
        turn = np.abs(np.gradient(phi, sw))
        turn = scipy.ndimage.uniform_filter1d(
            turn, self.tune.n_smooth_turn, mode="wrap"
        )
        h = np.minimum(h, np.radians(self.tune.turn_fine) / np.maximum(turn, 1e-300))
        # Growing away from each fine cell at ER_surf at most.
        h = _grow_limited(h, sw, self.tune.ER_surf - 1.0)
        return sw, 1.0 / h

    def _wall_u(self, lay, plane, xy):
        """Return `lay` refined along the blade to its density on plane `xy`.

        The layout fixes only how many refined cells each piece of the blade
        has, since the sweeps slide the coarse nodes along it. Here they are
        placed at fractional coarse indices, the inverse of the spline
        :meth:`_refined` runs through the swept coarse nodes, at the nodes
        the layout's density puts along each piece, so the spacing along the
        blade at this plane is the density's, scaled to fit each piece's
        count of cells. The scale runs linearly along each piece and
        continuously from one to the next, so the spacing does not step where
        they meet, as it would scaling each piece alone. At the corners
        behind the blade, C3 and C2, the scale is instead what makes the
        rear pieces' first cells as long as the passage block's last cells
        across there, at the downstream end of the trailing edge block, which
        the blocks behind the blade continue along `k`. The two sides are
        refined apart, and the passage block between them blends from one to
        the other. With `lay.symmetric`, as :meth:`_doubled_wall` left it.
        """
        if lay.symmetric:
            return lay
        wall = plane["wall"]
        S = util.arc_length(wall)
        s_c = _wall_arc(plane, xy[0][:, :, 0])
        e = np.cumsum((0, *lay.n_wall))
        m = dict(zip(("rl", "side_lo", "front", "side_up", "ru"), lay.nf_wall))
        counts = np.array(lay.nf_wall)

        # The density round the whole blade, from the first corner to where
        # it closes on itself.
        sigma, dens = _round_twice(*self._plane_density(lay, plane))
        c = s_c[e] / S
        t = np.linspace(c[0], c[-1], self.tune.n_wall)
        d = np.interp(t, sigma, dens)

        # Each piece scales the density to its count; at each corner, the
        # geometric mean of the scales of the pieces either side, so the
        # spacing does not step there; and along each piece, linearly from
        # one corner's to the next with a parabolic bow to make its count.
        n_piece = len(counts)
        pieces = []
        for j in range(n_piece):
            inside = (t >= c[j]) & (t <= c[j + 1])
            w = (t[inside] - c[j]) / (c[j + 1] - c[j])
            pieces.append((inside, w))
        k = np.array(
            [
                counts[j] / np.trapezoid(d[ins], t[ins])
                for j, (ins, _) in enumerate(pieces)
            ]
        )
        g = np.sqrt(k * np.roll(k, 1))

        def scaled(g):
            """The density scaled by `g` at the corners, or None if the bow
            that makes the counts turns it negative anywhere."""
            g = np.append(g, g[0])
            scale = np.empty(len(t))
            for j, (inside, w) in enumerate(pieces):
                lin = g[j] * (1.0 - w) + g[j + 1] * w
                # On the scale, not over the cells' length: weighted to the
                # long cells, the bow of a short piece whose density changes
                # sharply lands at its coarse end and steps the spacing there.
                bow = w * (1.0 - w)
                q = (
                    counts[j] - np.trapezoid(d[inside] * lin, t[inside])
                ) / np.trapezoid(d[inside] * bow, t[inside])
                scale[inside] = lin + q * bow
            return d * scale if (scale > 0.0).all() else None

        def along(d_eff, j, m):
            a, b = e[j], e[j + 1]
            s_f = S * _by_density(t, d_eff, c[j], c[j + 1], m)
            q = np.linspace(0.0, b - a, 64 * (b - a) + 1)
            s_q = _pchip_unit(s_c[a : b + 1], q)
            u = np.interp(s_f, s_q, q)
            u[0], u[-1] = 0.0, b - a
            return u

        # Behind the blade, the blocks there continue across the passage
        # block's last cells from C2 and C3 along `k` into the rear pieces'
        # first, so the scale at C3 and C2 is corrected until the two match
        # at the downstream end of the trailing edge block, which the block
        # behind it carries on unchanged. A scale too uneven to bow positive
        # leaves each piece scaling alone in `_by_density`.
        n_rl, n_ru = lay.n_wall[0], lay.n_wall[4]
        dtype = xy[1].dtype
        R_P = _refine_matrix(lay.n_P + 1, [(0, lay.n_P, lay.u["P"])], dtype)
        across = xy[5][:, -1, :] @ R_P.T
        h_across = np.linalg.norm(np.diff(across, axis=1), axis=0)
        d_eff = d
        for _ in range(self.tune.n_match_rear):
            trial = scaled(g)
            if trial is None:
                break
            d_eff = trial
            rear = [
                (0, n_rl, _reverse(along(d_eff, 0, m["rl"]), n_rl)),
                (n_rl, n_rl + n_ru, _reverse(along(d_eff, 4, m["ru"]), n_ru)),
            ]
            R_rear = _refine_matrix(n_rl + n_ru + 1, rear, dtype)
            behind = xy[4][:, -1, :] @ R_rear.T
            h_rear = np.linalg.norm(np.diff(behind, axis=1), axis=0)
            logger.debug(
                f"Rear over across at C3 {h_rear[0] / h_across[-1]:.3f}, "
                f"at C2 {h_rear[-1] / h_across[0]:.3f}"
            )
            # Rear spacing goes as one over the scale at its corner.
            g[1] *= h_rear[0] / h_across[-1]
            g[4] *= h_rear[-1] / h_across[0]

        u = dict(lay.u)
        n_sd = lay.n_wall[1]
        u["rl"] = along(d_eff, 0, m["rl"])
        u["side_lo"] = _reverse(along(d_eff, 1, m["side_lo"]), n_sd)
        u["side_up"] = along(d_eff, 3, m["side_up"])
        u["front"] = along(d_eff, 2, m["front"])
        u["ru"] = along(d_eff, 4, m["ru"])
        return dataclasses.replace(lay, u=u)

    def _face_u(self, lay, mid):
        """Return `lay` with the sides of the blade refined apart along the
        O block's outer face, `u["side_lo_face"]` and `u["side_up_face"]`:
        each side's first cell along the face at each corner scaled to the
        first cell of the piece round the front or rear there, as refined
        plane `mid` has them, within `geometric_O_match` either way, and
        faded to none over `side_corner_cells` of the way along the side.
        The passage block is refined along the sides so, and the O block
        blended across from the spacing along the blade to it."""
        c = self.tune.geometric_O_match
        n_sd = lay.n_wall[1]
        spokes = dict(_corner_spokes(mid, lay, mid[0][:, :, -1]))
        i3, i4, i1, i2 = lay.i_corners()
        u = dict(lay.u)
        # In the O block's direction, the lower side from C3 to C4 and the
        # upper from C1 to C2; the side is after C3 and C1, before C4 and C2
        lo = _reverse(np.asarray(u["side_lo"], float), n_sd)
        up = np.asarray(u["side_up"], float)
        for name, v, (a, b) in (("side_lo", lo, (i3, i4)), ("side_up", up, (i1, i2))):
            n_fade = self.tune.side_corner_cells * (len(v) - 1)
            Fa = np.clip(spokes[a]["face-"] / spokes[a]["face+"], 1.0 / c, c)
            Fb = np.clip(spokes[b]["face+"] / spokes[b]["face-"], 1.0 / c, c)
            v = _scale_end(v, Fa, n_fade, at_end=False)
            v = _scale_end(v, Fb, n_fade, at_end=True)
            u[name + "_face"] = _reverse(v, n_sd) if name == "side_lo" else v
            logger.debug(f"{name} along the face scaled {Fa:.2f} and {Fb:.2f}")
        return dataclasses.replace(lay, u=u)

    def _passage_u(self, lay, mid):
        """Return `lay` with the refinement across the passage ahead of and
        behind the blade set apart, `u["P_in"]` and `u["P_out"]`, each end
        scaled so the first cells across the passage block ahead of (behind)
        the blade are, in the geometric mean along each face it shares with
        the block round the leading (trailing) edge, as long as that block's
        first cells there, as refined plane `mid` has them; within
        `geometric_O_match` either way, and faded to none over
        `passage_end_cells` of the way across."""
        c = self.tune.geometric_O_match
        LE, Pin, TE, Pout = (np.asarray(mid[b], float) for b in (2, 3, 4, 5))

        def first(B, k0):
            """First cells along `k` from face `k0` (0 or -1), each `i`."""
            k1 = 1 if k0 == 0 else -2
            return np.linalg.norm(B[:, :, k1] - B[:, :, k0], axis=0)

        def factor(edge, passage, k_edge, k_passage, near_blade):
            """In the geometric mean over the quarter of the face nearest
            the blade, or nearest the far end."""
            r = np.log(first(edge, k_edge)) - np.log(first(passage, k_passage))
            r = r[1:-1]
            n = max(1, len(r) // 4)
            # Ahead of the blade, i runs towards it; behind, away
            r = r[-n:] if near_blade else r[:n]
            return float(np.clip(np.exp(np.mean(r)), 1.0 / c, c))

        u = dict(lay.u)
        n_fade = self.tune.passage_end_cells * (len(u["P"]) - 1)
        for name, edge, passage in (("P_in", LE, Pin), ("P_out", TE, Pout)):
            for far, suffix in ((False, ""), (True, "_far")):
                near_blade = (name == "P_in") != far
                # The passage's k = 0 meets the edge block's k = -1, and its
                # k = -1 the edge block's k = 0
                F0 = factor(edge, passage, -1, 0, near_blade)
                F1 = factor(edge, passage, 0, -1, near_blade)
                v = _scale_end(np.asarray(u["P"], float), F0, n_fade, at_end=False)
                u[name + suffix] = _scale_end(v, F1, n_fade, at_end=True)
                logger.debug(f"{name}{suffix} ends scaled by {F0:.2f} and {F1:.2f}")
        return dataclasses.replace(lay, u=u)

    def _ends_u(self, lay, mid):
        """Return `lay` with the refinement along the blocks ahead of and
        behind the blade, `u["in"]` and `u["out"]`, scaled at the blade so
        their first cells there are, in the geometric mean over the two
        corners each meets, the geometric mean of the first cells along the
        O block's outer face and into the passage block at those corners, as
        refined plane `mid` has them; within `geometric_O_match` either way,
        and faded to none over `end_corner_cells` of the way along."""
        c = self.tune.geometric_O_match
        spokes = _corner_spokes(mid, lay, mid[0][:, :, -1])
        u = dict(lay.u)
        for name, key, at_end in (("in", "LE", True), ("out", "TE", False)):
            r = [
                np.mean(np.log([cells["face-"], cells["face+"], cells["P"]]))
                - np.log(cells[key])
                for _, cells in spokes
                if key in cells
            ]
            F = float(np.clip(np.exp(np.mean(r)), 1.0 / c, c))
            n = len(u[name]) - 1
            u[name] = _scale_end(u[name], F, self.tune.end_corner_cells * n, at_end)
            logger.debug(f"First cells {name} at the blade scaled by {F:.2f}")
        return dataclasses.replace(lay, u=u)

    def _dn_u(self, lay, mid):
        """Return `lay` with the refinement along the block from behind the
        blade to the mixing plane, from mid-span's refined plane `mid`.

        Only its count, the fewest cells, a multiple of eight, that fill the
        block with a first cell within `ER_mix` of the last of the blocks
        behind the blade where they meet it, growing at no more than
        `ER_mix`, to no longer than `AR_mix` times the block's mean width
        across the pitch. Each plane's own lines are respaced to it in
        :meth:`_refined`, so here they are even.
        """
        TE, Pout, dn = (np.asarray(mid[b], dtype=float) for b in (4, 5, 7))
        d0 = float(
            np.mean(
                [np.linalg.norm(B[:, -1] - B[:, -2], axis=0).mean() for B in (TE, Pout)]
            )
        )
        L = float(np.linalg.norm(np.diff(dn, axis=1), axis=0).sum(axis=0).mean())
        width = float(np.linalg.norm(np.diff(dn, axis=2), axis=0).mean())
        ER = self.tune.ER_mix
        cap = max(self.tune.AR_mix * width, d0 * ER)
        n = 8
        # Grown from the longest first cell allowed, as far as the cap, these
        # cells must reach the end; uniform at the shortest, not overshoot it
        while np.minimum(d0 * ER * ER ** np.arange(n), cap).sum() < L:
            n += 8
        if n * d0 / ER > L:
            logger.debug(
                f"Downstream block: {n} cells no shorter than {d0 / ER:.4g} "
                f"overfill it, {L:.4g} long; its first cell is shorter"
            )
        logger.debug(f"Downstream block: {n} cells, first near {d0:.4g}")
        return dataclasses.replace(
            lay, u={**lay.u, "dn": np.linspace(0.0, lay.n_dn, n + 1)}
        )

    def _O_count(self, lay, plane, xy, beyond):
        """Return `lay` with the nodes across the O block counted for its
        lines as swept and refined round the blade on `plane`, not its
        nominal thickness.

        Enough that each line, from the wall spacing at the blade to the
        first cell `beyond` it, can grow steadily the whole way: a line
        growing geometrically from one to the other fills its length `L`
        with `ln(h / d) / ln(r)` cells, `r = (L - d) / (L - h)`, and with
        fewer the double-sided stretching must grow past `h` and shrink back.
        As many as all but the longest tenth of lines need, and no fewer than
        grow at `ER_O`, rounded up to a count a multigrid level can halve.
        `xy` is the O block (2, ni, nk) refined round the blade.
        """
        O = np.asarray(xy, dtype=float)
        L = np.linalg.norm(np.diff(O, axis=2), axis=0).sum(axis=1)
        d = lay.d_wall / plane["cmap"].xr_from_mp(O[0, :, 0])[1]
        h = np.maximum(np.asarray(beyond, dtype=float), d * (1.0 + 1e-6))
        r = (L - d) / np.maximum(L - h, 1e-30)
        steady = np.where(
            L > h, np.log(h / d) / np.log(np.maximum(r, 1.0 + 1e-9)), L / d
        )
        n_ER = np.log(h / d) / np.log(self.tune.ER_O)
        n = max(float(np.percentile(steady, 90)), float(np.median(n_ER)), 1.0)
        nk = _up8(int(np.ceil(n))) + 1
        logger.debug(
            f"O block nodes across: {lay.nk_O} as laid out, {nk} to grow "
            f"steadily from wall to the cells beyond"
        )
        return dataclasses.replace(lay, nk_O=nk)

    @classmethod
    def _front_nodes(cls, lay):
        """Return the first and last refined blade node of the piece of the O
        block round the leading edge, as :meth:`_refined` numbers them."""
        sizes = [
            len(u) - (0 if j == 0 else 1)
            for j, (_, _, u) in enumerate(cls._wall_pieces(lay))
        ]
        start = int(np.sum(sizes[:2]))
        return start - 1, start + sizes[2] - 1

    @staticmethod
    def _wall_pieces(lay):
        """Return the stretches of the O block along the blade, `(a, b, u)`
        from coarse node `a` to `b` with refined nodes at `u` from `a`."""
        u = lay.u
        e = np.cumsum((0, *lay.n_wall))
        return [
            (e[0], e[1], u["rl"]),
            (e[1], e[2], _reverse(u["side_lo"], lay.n_wall[1])),
            (e[2], e[3], u["front"]),
            (e[3], e[4], u["side_up"]),
            (e[4], e[5], u["ru"]),
        ]

    def _refine_matrices(self, lay, xy):
        """Return the matrices refining each block of a coarse plane `xy`.

        For each block, one along `i` and one along `k`, or None if it is not
        refined that way: the splines' values for each coarse node alone,
        built once for the row, the refinement being linear in the coarse
        nodes and its pieces fixed. Also `P_lo`, along `i` of the passage
        block as the O block is refined along the lower side.
        """
        u = lay.u
        n_rl, _, n_fr, _, n_ru = lay.n_wall
        n_sd = lay.n_wall[1]
        n_P = lay.n_P
        # The sides as along the O block's outer face, which the passage
        # block meets, if `_face_u` set them apart from along the blade
        side = [(0, n_sd, u.get("side_up_face", u["side_up"]))]
        side_lo = [(0, n_sd, u.get("side_lo_face", u["side_lo"]))]
        # Across the passage ahead of and behind the blade, if `_passage_u`
        # set them apart to meet the blocks round the leading and trailing
        # edges; the passage block blends from one to the other along `i`
        across_in = [(0, n_P, u.get("P_in", u["P"]))]
        across_out = [(0, n_P, u.get("P_out", u["P"]))]
        # At the far ends of the blocks ahead of and behind the blade, which
        # each blends to from its own next to the blade
        across_in_far = [(0, n_P, u.get("P_in_far", u.get("P_in", u["P"])))]
        across_out_far = [(0, n_P, u.get("P_out_far", u.get("P_out", u["P"])))]
        front = [(0, n_fr, u["front"])]
        rear = [
            (0, n_rl, _reverse(u["rl"], n_rl)),
            (n_rl, n_rl + n_ru, _reverse(u["ru"], n_ru)),
        ]

        def shifted(pieces, by):
            return [(a + by, b + by, v) for a, b, v in pieces]

        plan = {
            "O": (self._wall_pieces(lay), None),
            "P": (side, across_in),
            "LE": ([(0, lay.n_in, u["in"])], front),
            "Pin": ([(0, lay.n_in, u["in"])], across_in),
            "TE": ([(0, lay.n_out, u["out"])], rear),
            "Pout": ([(0, lay.n_out, u["out"])], across_out),
            "up": (None, across_in_far + shifted(front, n_P)),
            "dn": (
                [(0, lay.n_dn, u["dn"])] if "dn" in u else None,
                across_out_far + shifted(rear, n_P),
            ),
        }
        refine = {
            name: tuple(
                None if pieces is None else _refine_matrix(n, pieces, B.dtype)
                for n, pieces in zip(B.shape[1:], plan[name])
            )
            for name, B in zip(LABELS, xy)
        }
        refine["P_lo"] = _refine_matrix(n_sd + 1, side_lo, xy[1].dtype)
        refine["P_out"] = _refine_matrix(n_P + 1, across_out, xy[1].dtype)
        refine["Pin_far"] = _refine_matrix(n_P + 1, across_in_far, xy[1].dtype)
        refine["Pout_far"] = _refine_matrix(n_P + 1, across_out_far, xy[1].dtype)
        refine["O_face"] = _refine_matrix(
            xy[0].shape[1], self._wall_pieces(_face_lay(lay)), xy[0].dtype
        )
        return refine

    def _refined(self, xy, plane, lay, refine, respace_dn=True):
        """Return the blocks of a smoothed coarse plane, refined.

        Each block refined along each of its directions on cubic splines
        through the coarse nodes, one between each pair of block junctions,
        by the matrices `refine`; the O block's blade nodes put back on the
        blade; each line off it respaced along a cubic spline through its
        coarse nodes, leaving the blade along its normal, from the wall
        spacing at the blade to the first cell of the block beyond it, by
        double-sided stretching whose expansion ratio varies linearly.
        """
        # The O block's blade nodes refined through where they lie along the
        # blade, monotonically, so they stay on it and in order round it.
        wall = plane["wall"]
        S = util.arc_length(wall)
        s_c = _wall_arc(plane, xy[0][:, :, 0])
        s_f = np.concatenate(
            [
                _pchip_unit(s_c[a : b + 1], v)[(0 if j == 0 else 1) :]
                for j, (a, b, v) in enumerate(self._wall_pieces(lay))
            ]
        )
        if lay.symmetric:
            # Each half on one spline, from the middle of the trailing edge to
            # the leading edge, so the two run alike through their corners
            kappa = lay.u["sym_kappa"]
            nh = (len(s_c) - 1) // 2
            s_lo = _pchip_unit(s_c[: nh + 1], kappa)
            s_up = _pchip_unit(s_c[len(s_c) - 1 - nh :][::-1], kappa)
            s_sym = np.concatenate((s_lo, s_up[::-1][1:]))
            w = self._sym_weight(float(plane["cmap"].spf))
            s_f = (1.0 - w) * s_f + w * s_sym

        out = []
        for name, B in zip(LABELS, xy):
            R_i, R_k = refine[name]
            if name == "P":
                # Along the upper side of the blade at k = 0 and the lower at
                # k = -1, each as the O block it meets is refined, and
                # blended linearly across.
                w = np.linspace(0.0, 1.0, B.shape[2], dtype=B.dtype)
                B = (R_i @ B) * (1.0 - w) + (refine["P_lo"] @ B) * w
                # Across, from as ahead of the blade at i = 0 to as behind
                # it at i = -1, smoothly
                t = np.linspace(0.0, 1.0, B.shape[1])
                t = (t * t * (3.0 - 2.0 * t)).astype(B.dtype)[:, None]
                B = (B @ R_k.T) * (1.0 - t) + (B @ refine["P_out"].T) * t
                out.append(B)
                continue
            if name in ("Pin", "Pout"):
                # Across, from as next to the blade to as at the far end,
                # smoothly along `i`: ahead of the blade, i = -1 is next to
                # it; behind, i = 0
                B = R_i @ B
                t = np.linspace(0.0, 1.0, B.shape[1])
                t = t * t * (3.0 - 2.0 * t)
                t = (1.0 - t if name == "Pin" else t).astype(B.dtype)[:, None]
                far = refine[f"{name}_far"]
                out.append((B @ R_k.T) * (1.0 - t) + (B @ far.T) * t)
                continue
            elif name == "O":
                # Along the blade as the blade nodes are spaced, blended
                # across the block to the outer face as the passage block
                # beyond it is
                w = np.linspace(0.0, 1.0, B.shape[2])
                w = (w * w * (3.0 - 2.0 * w)).astype(B.dtype)
                B = (R_i @ B) * (1.0 - w) + (refine["O_face"] @ B) * w
            elif R_i is not None:
                B = R_i @ B
            if R_k is not None:
                B = B @ R_k.T
            out.append(B)

        # The O block to the blade and clustered off it, its last cells as
        # long as the first ones beyond them.
        O = out[0]
        O[:, :, 0] = _at_arc(wall, np.mod(s_f, S))
        normal = _wall_normals(wall, s_f, tune=self.tune)
        # Round the leading edge, on the section's own outline: `wall` is even
        # in arc length, coarser there than the refined cells round a sharp
        # nose, and nodes on its straight pieces would facet it, the normal
        # jumping at each of its points. At the same fraction of the way round.
        i0, i1 = self._front_nodes(lay)
        section = plane["wall_section"]
        s_sec = np.mod(s_f[i0 : i1 + 1], S) * (util.arc_length(section) / S)
        O[:, i0 : i1 + 1, 0] = _at_arc(section, s_sec)
        normal[:, i0 : i1 + 1] = _wall_normals(section, s_sec, tune=self.tune)
        dtype = O.dtype
        # Each line a cubic spline through its coarse nodes, leaving the
        # blade along its normal, sampled finely, so the nodes respaced along
        # it leave the blade square and do not cut the corners where the
        # lines between coarse nodes bend.
        O = O.astype(float)
        speed = np.linalg.norm(O[:, :, 1] - O[:, :, 0], axis=0)
        start = normal * speed
        O = (
            np.concatenate((O, start[:, :, None]), axis=2)
            @ _spline_O(O.shape[2], tune=self.tune).T
        )
        nk = lay.nk_O
        d_wall = lay.d_wall / plane["cmap"].xr_from_mp(O[0, :, 0])[1]
        d = np.diff(O, axis=2)
        d *= d
        s = np.zeros(O.shape[1:])
        np.cumsum(np.sqrt(d[0] + d[1]), axis=1, out=s[:, 1:])
        L = s[:, -1]
        # Geometric growth along each line from the wall spacing, no coarser
        # than even, the same expansion ratio all along it, so neighbouring
        # lines of different lengths keep their nodes at like fractions.
        f0 = np.clip(d_wall / L, 1e-12, 1.0 / (nk - 1))
        T = _geometric_many(f0, nk)
        if self.tune.geometric_O_match > 1.0:
            # The outer cell towards the first beyond, by a factor
            # capped and smoothed round the blade, its last line the
            # first again.
            g1 = T[:, -1] - T[:, -2]
            c = np.log(self.tune.geometric_O_match)
            q = np.clip(np.log(_beyond_O(out, lay) / L / g1), -c, c)[:-1]
            w = np.exp(
                -0.5
                * (
                    np.arange(
                        -3 * self.tune.o_match_smooth, 3 * self.tune.o_match_smooth + 1
                    )
                    / self.tune.o_match_smooth
                )
                ** 2
            )
            q = np.convolve(
                np.concatenate((q[-len(w) :], q, q[: len(w)])), w / w.sum(), "same"
            )[len(w) : -len(w)]
            if self.tune.o_corner_width > 0.0:
                # At each corner the outer cell to the geometric mean of the
                # first cells of the other four lines there, the smoothed
                # factor corrected over `o_corner_width` lines either side
                n_ring = len(q)
                i_ring = np.arange(n_ring)
                for i_c, cells in _corner_spokes(out, lay, O[:, :, -1]):
                    target = np.exp(np.mean(np.log(list(cells.values()))))
                    dq = np.clip(np.log(target / L[i_c] / g1[i_c]), -c, c) - q[i_c]
                    t = np.abs((i_ring - i_c + n_ring // 2) % n_ring - n_ring // 2)
                    t = np.clip(t / self.tune.o_corner_width, 0.0, 1.0)
                    q = q + dq * (1.0 - t * t * (3.0 - 2.0 * t))
                q = np.clip(q, -c, c)
            f1 = g1 * np.exp(np.append(q, q[0]))
            T = clusterfunc.double.smooth_many(f0, f1, nk)
        T = L[:, None] * T
        # Every line at once: which segment each new node falls in, and how
        # far along it, searched in one sorted run with each line lifted
        # clear of the one before.
        inner = s[:, 1:-1]
        lift = (2.0 * float(s[:, -1].max()) + 1.0) * np.arange(len(s))[:, None]
        q = np.searchsorted((inner + lift).ravel(), T + lift, side="right")
        q = np.clip(q - inner.shape[1] * np.arange(len(s))[:, None], 0, len(s[0]) - 2)
        s_lo = np.take_along_axis(s, q, axis=1)
        s_hi = np.take_along_axis(s, q + 1, axis=1)
        w = (T - s_lo) / np.maximum(s_hi - s_lo, 1e-30)
        lo = np.take_along_axis(O, q[None], axis=2)
        hi = np.take_along_axis(O, q[None] + 1, axis=2)
        out[0] = (lo + w[None] * (hi - lo)).astype(dtype)
        if "dn" in lay.u and respace_dn:
            out[7] = _dn_lines(out[7], out[4], out[5], self.tune.ER_mix).astype(dtype)
        return out

    def _polish_shift(self, topo, hold, xy, rings):
        """Return how far polishing moves each node of blocks `xy`.

        The nodes within `rings` cells of every cell whose worst corner is
        under :attr:`polish_below`, against cells the size they are; the
        nodes `hold`, the O block's first layers of cells off the blade, held
        so its spacing and squareness there stay as clustered.
        """
        poor = []
        for B, g in zip(xy, topo.gid):
            bad = _cell_angles(B) < self.tune.polish_below
            for corner in (g[:-1, :-1], g[1:, :-1], g[1:, 1:], g[:-1, 1:]):
                poor.append(corner[bad])
        poor = np.unique(np.concatenate(poor))
        X = topo.gather(xy)
        if not len(poor):
            return np.zeros_like(X)
        polished = elliptic.polish(
            topo,
            X,
            poor,
            rings=rings,
            power=self.tune.polish_power,
            hold=hold,
            taper=self.tune.polish_taper,
        )
        return polished - X

    #
    # TIP GAP
    #

    def _full_match(self, i_row, blade, cmap, section):
        """Whether row `i_row`'s tip gap meets its H block node for node all
        round, by :attr:`tip_match`, choosing from the drift of the blade's
        section `section` at the top of the blade, on `cmap`, with 0."""
        if self.tune.tip_match != 0:
            return self.tune.tip_match == 1
        drift = _pairing_drift(blade, cmap, section)
        full = drift <= self.tune.tip_match_drift
        logger.info(
            f"Row {i_row} tip gap: pairing drift {drift:.3f} of the camber "
            f"{'within' if drift <= self.tune.tip_match_drift else 'over'} "
            f"{self.tune.tip_match_drift}, so "
            f"{'matched node for node all round' if full else 'non-matching along the upper side'}"
        )
        return full

    def _tip_gap(self, blade, layout, planes, per_plane):
        """Return the blocks of a tip gap on each plane, and where they meet.

        Two blocks, inside the outline the blade would have had: an O block
        whose outer face is the O block round the blade's inner one, node for
        node, and an H block inside it, running `i` from the leading edge to
        the trailing edge and `k` from the lower side of the blade to the
        upper.

        The O block's inner face is the outline offset inwards along the
        blade's normal at each node, by `t_tip_O` of the blade's greatest
        thickness, less round the edges (:func:`_gap_offset`), its lines
        straight and respaced to the wall spacing at the outline. Its nose
        and tail are then squared off by bowed lines, the H block's ends,
        between its corners, nodes of the inner face at `iA` and `iB` along
        the lower side, `iC` and `iD` along the upper (:func:`_gap_square`):
        B and C standing back from the nose by `tip_end`, A and D moved onto
        the rear line, back from the tail by `tip_end_rear`
        (:func:`_gap_corners`).
        The H block's lines run across the blade normal to the camber line
        from each inner node on the lower side, which meets the O block
        node for node, as does the rear, round the tail through the cut,
        whose nodes set the H block's count across; the front, of as many
        nodes, and the upper side meet it by non-matching interfaces
        (:func:`_gap_H`); with `layout.symmetric` (:attr:`tip_match`), from
        the nodes paired across, every face meeting node for node.

        Every plane is built so, algebraically, on its own outline and
        camber line, with the corners, the rear line's fraction of the way
        along the camber and the H block's node on the cut found on the
        plane at the tip of the blade, so that both sides of
        each non-matching interface lie on one curve on every plane.

        Parameters
        ----------
        blade : Blade
            The row's blade.
        layout : _Layout
            The row's layout.
        planes : list of dict
            The planes in the gap, from the tip of the blade to the casing.
        per_plane : list of list
            The finished blocks (2, ni, nk) of each of `planes`.

        Returns
        -------
        gap : list of list
            The O and H blocks (2, ni, nk) of each plane.
        corners : tuple of int
            `iA`, `iB`, `iC`, `iD`.
        k_cut : int
            The `k` of the H block's rear face on the cut.

        """
        walls = [np.asarray(xy[0][:, :, 0], dtype=float) for xy in per_plane]
        i_LE = int(np.argmin(walls[0][0]))
        # Sides of the H block meeting the O block node for node: 1 the
        # lower, 3 both
        match = 3 if layout.symmetric else 1
        corners = k_cut = nk_O = None
        gap = []
        for W, plane in zip(walls, planes):
            cmap = plane["cmap"]
            cam = _Camber(*_surfaces(blade, cmap, cmap.spf))
            inner = _gap_offset(W, plane["wall"], self.tune.t_tip_O, tune=self.tune)
            if corners is None:
                corners = _gap_corners(inner, i_LE, cam, self, match)
            # The rear line through the corners' own nodes, which so stay
            # where they are, once leant level along the camber
            if self.tune.tip_rear_lean:
                inner = _gap_rear_bisect(
                    inner, W, corners, self, self.tune.tip_rear_lean
                )
            rear = (inner[:, corners[0]], inner[:, corners[3]])
            if self.tune.tip_front_lean:
                inner = _gap_lean(inner, W, corners, cam, self.tune.tip_front_lean)
            # Both sides, each rear corner landing where the rear line meets
            # it, so the O block's lines lean progressively towards it on
            # either side rather than cross by the corner it is moved to
            # The far side's slide faded out over `tip_blend` of the camber
            # from its corner, so the rest of that side stays square to the
            # blade
            fade = (
                self.tune.tip_blend * cam.s[-1] if self.tune.tip_blend > 0.0 else None
            )
            for side in (1, 2):
                inner = _gap_spread(
                    inner,
                    corners,
                    rear,
                    i_LE,
                    side,
                    fade=None if side & match else fade,
                )
            inner, front = _gap_square(inner, corners, rear, self)
            H, k_cut = _gap_H(
                inner,
                i_LE,
                corners,
                front,
                cam,
                match,
                self.tune.tip_blend,
                tune=self.tune,
            )
            O = _gap_O_lines(inner, W, self.tune.tip_O_square, tune=self.tune)
            d_wall = layout.d_wall / cmap.xr_from_mp(W[0])[1]
            if nk_O is None:
                # Counted once, at the tip of the blade, as the corners are:
                # a block has one count on every plane
                nk_O = _gap_O_count(O, d_wall, self.tune)
            O = _respace_gap_O(O, d_wall, nk_O)
            O = O.astype(DTYPE)
            O[:, :, -1] = W
            gap.append([O, H.astype(DTYPE)])
            for B in gap[-1]:
                assert np.isfinite(B).all(), "non-finite coordinates in tip gap"

        q_tip = min(_cell_angles(B).min() for xy in gap for B in xy)
        logger.debug(
            f"Tip gap: O {gap[0][0].shape[1:]}, H {gap[0][1].shape[1:]}, "
            f"corners {corners}, rear cut at k={k_cut}, "
            f"worst cell corner {q_tip:.1f} deg"
        )
        return gap, corners, k_cut

    def blocks(self, machine, i_row, planes, per_plane, labels):
        """Return bare ember blocks, without patches, from finished planes.

        `per_plane` holds the blocks (2, ni, nk) of each of `planes` in its
        conformal coordinates, in the order `labels` names them.
        """
        nj = len(planes)
        blocks = []
        for b, name in enumerate(labels):
            ni, nk = per_plane[0][b].shape[1:]
            xrt = np.empty((3, ni, nj, nk), dtype=per_plane[0][b].dtype)
            for j, (plane, xy) in enumerate(zip(planes, per_plane)):
                xrt[:2, :, j, :] = plane["cmap"].xr_from_mp(xy[b][0])
                xrt[2, :, j, :] = xy[b][1]
            block = ember.block.Block(shape=(ni, nj, nk))
            block.set_label(f"row{i_row}_{name}")
            block.set_Nb(machine.rows[i_row].n_blade)
            block.set_x(xrt[0])
            block.set_r(xrt[1])
            block.set_t(xrt[2])
            for dim, n in zip("ijk", block.shape):
                # The tip gap's H block is as long as the blade has nodes
                # between its corners, and as wide as round its tail, not
                # counts multigrid can halve
                if name == "tipH" and dim in "ik":
                    continue
                assert not (n - 1) % 8, f"{name}: n{dim}-1={n - 1} not divisible by 8"
            blocks.append(block)
        return blocks

    def assemble(self, machine, i_row, layout, planes, per_plane, jtip=None):
        """Return ember blocks from the finished planes of a row, with the two
        blocks of a tip gap from `jtip` after the rest."""
        n_row = len(machine.rows)
        blocks = self.blocks(machine, i_row, planes, per_plane, self.labels)
        O, P, LE, Pin, TE, Pout, up, dn = blocks
        i3, i4, i1, i2 = layout.i_corners()
        nkP = P.shape[2] - 1
        Per = ember.patch.PeriodicPatch

        O.patches.extend(
            [
                Per(i=0, label="cut0"),
                Per(i=-1, label="cut1"),
                Per(i=(0, i3), k=-1, label="rear_lower"),
                Per(i=(i3, i4), k=-1, label="lower"),
                Per(i=(i4, i1), k=-1, label="front"),
                Per(i=(i1, i2), k=-1, label="upper"),
                Per(i=(i2, -1), k=-1, label="rear_upper"),
            ]
        )
        for blk in (P, LE, Pin, Pout):
            blk.patches.extend([Per(i=0), Per(i=-1), Per(k=0), Per(k=-1)])
        TE.patches.extend(
            [Per(i=0, k=(0, i3)), Per(i=0, k=(i3, -1)), Per(i=-1), Per(k=0), Per(k=-1)]
        )
        up.patches.extend(
            [
                ember.patch.InletPatch(i=0)
                if i_row == 0
                else ember.patch.MixingPatch(i=0),
                Per(i=-1, k=(0, nkP)),
                Per(i=-1, k=(nkP, -1)),
                Per(k=0),
                Per(k=-1),
            ]
        )
        dn.patches.extend(
            [
                ember.patch.OutletPatch(i=-1)
                if i_row == n_row - 1
                else ember.patch.MixingPatch(i=-1),
                Per(i=0, k=(0, nkP)),
                Per(i=0, k=(nkP, -1)),
                Per(k=0),
                Per(k=-1),
            ]
        )
        for blk in blocks:
            blk.patches.extend(self._endwall_patches())
        casing = self._duct_endwalls(machine, i_row, up, dn)
        if jtip is None:
            return blocks

        # Over the gap the blade's outline is the face the O block of the gap
        # shares with the O block round the blade, node for node. Inside it,
        # the gap's O block meets its H block node for node along the lower
        # side and the rear, either side of the cut, and by non-matching
        # interfaces along the front and the upper side, or, matched all
        # round (`layout.symmetric`), node for node everywhere.
        NonMatch = ember.patch.NonMatchPatch
        O.patches.append(Per(k=0, j=(jtip, -1), label="tip_gap"))
        gap, (iA, iB, iC, iD), k_cut = self._tip_gap(
            machine.rows[i_row].blade,
            layout,
            planes[jtip:],
            per_plane[jtip:],
        )
        tO, tH = self.blocks(machine, i_row, planes[jtip:], gap, ("tipO", "tipH"))
        # The front matched node for node if as many round the nose as the tail
        Front = Per if iC - iB == iA + (tO.shape[0] - 1) - iD else NonMatch
        Lower, Upper = (Per, NonMatch)
        if layout.symmetric:
            # The halves of the gap's inner face alike: both sides node for node
            Lower = Upper = Per
        tO.patches.extend(
            [
                Per(i=0, label="tip_cut0"),
                Per(i=-1, label="tip_cut1"),
                Per(k=-1, label="tip_outline"),
                Lower(i=(iA, iB), k=0, label="tip_lower"),
                Front(i=(iB, iC), k=0, label="tip_front"),
                Upper(i=(iC, iD), k=0, label="tip_upper"),
                Per(i=(iD, -1), k=0, label="tip_rear_upper"),
                Per(i=(0, iA), k=0, label="tip_rear_lower"),
            ]
        )
        tH.patches.extend(
            [
                Lower(k=0, label="tip_lower"),
                Front(i=0, label="tip_front"),
                Upper(k=-1, label="tip_upper"),
                Per(i=-1, k=(k_cut, -1), label="tip_rear_upper"),
                Per(i=-1, k=(0, k_cut), label="tip_rear_lower"),
            ]
        )

        # Walls turn with their block unless a `RotatingPatch` overrides
        # them, so the casing over a tip gap needs one on every block; `bconds`
        # sets its speed. The j = 0 faces of the gap's blocks are the tip of
        # the blade, a wall turning with the row, which is what an unpatched
        # face already means.
        # Not over a stretch of the casing in a duct, which a rotating patch
        # would overlap.
        blocks += [tO, tH]
        for blk in blocks:
            i_rot = casing.get(id(blk), (0, -1))
            blk.patches.append(ember.patch.RotatingPatch(i=i_rot, j=-1, label="casing"))
        return blocks


def _dn_lines(dn, TE, Pout, ER):
    """Return the block from behind the blade to the mixing plane, `dn`
    (2, ni, nk), respaced along each of its lines in `i` by
    :func:`_dn_fractions`."""
    return _dn_apply(dn, _dn_fractions(dn, TE, Pout, ER))


def _dn_fractions(dn, TE, Pout, ER):
    """Return where the nodes of each line in `i` of the block from behind
    the blade to the mixing plane, `dn` (2, ni, nk), go, as fractions of the
    line's length (nk, ni).

    Each line's first cell as near the last cell of the line of `TE` or
    `Pout` it meets as `ER` either way allows, growing at `ER` to the cap
    that fills the line in its count of cells; a line too short for its
    count at the shortest first cell allowed is spaced evenly.
    """
    dn = np.asarray(dn, dtype=float)
    ni, nk = dn.shape[1:]
    n = ni - 1
    # The last cell of the line meeting each of dn's at its start
    faces = np.concatenate(
        [np.asarray(B, dtype=float)[:, -1] for B in (TE, Pout)], axis=1
    )
    lasts = np.concatenate(
        [
            np.linalg.norm(
                np.asarray(B, dtype=float)[:, -1] - np.asarray(B, dtype=float)[:, -2],
                axis=0,
            )
            for B in (TE, Pout)
        ]
    )
    near = cKDTree(faces.T).query(dn[:, 0].T)[1]
    growth = ER ** np.arange(n)
    # Every line at once, (nk, ni): its arc length, length and first cell
    s = np.zeros((nk, ni))
    np.cumsum(np.linalg.norm(np.diff(dn, axis=1), axis=0).T, axis=1, out=s[:, 1:])
    L, d0 = s[:, -1], lasts[near]
    grown = d0[:, None] * growth
    # Too short to grow from it: as long a first cell as fits, evenly
    short = n * d0 >= L
    # Too long to reach growing from it at ER: the first cell longer, as far
    # as ER allows, then growing faster if it must
    long = ~short & (grown.sum(axis=1) < L)
    with np.errstate(divide="ignore", invalid="ignore"):
        first = np.minimum(L * (ER - 1.0) / (ER**n - 1.0), d0 * ER)
    # Otherwise growing at ER to the cap that fills the line, found by
    # bisection
    lo, hi = d0.copy(), grown.max(axis=1)
    for _ in range(60):
        top = 0.5 * (lo + hi)
        under = np.minimum(grown, top[:, None]).sum(axis=1) < L
        lo, hi = np.where(under, top, lo), np.where(under, hi, top)
    cells = np.minimum(grown, 0.5 * (lo + hi)[:, None])
    cells[long] = first[long, None] * growth
    cells *= (L / cells.sum(axis=1))[:, None]
    cells[short] = (L[short] / n)[:, None]
    f = np.zeros((nk, ni))
    np.cumsum(cells, axis=1, out=f[:, 1:])
    return f / f[:, -1:]


def _dn_apply(dn, f):
    """Return `dn` (2, ni, nk) with the nodes of each of its lines in `i`
    moved by arc length along it to fractions `f` (nk, ni) of its length,
    its ends staying put."""
    dn = np.asarray(dn, dtype=float)
    nk = dn.shape[2]
    s = np.zeros(f.shape)
    np.cumsum(np.linalg.norm(np.diff(dn, axis=1), axis=0).T, axis=1, out=s[:, 1:])
    out = dn.copy()
    for k in range(nk):
        for c in range(2):
            out[c, :, k] = np.interp(f[k] * s[k, -1], s[k], dn[c, :, k])
    return out


def _theta_even(xy, pitch, skew_min, dw_max, side, cap=None, weights=False):
    """Return blocks `xy` (2, ni, nk) with the blocks round one edge of the
    blade, LE and Pin ahead of it for `side` "in" or TE and Pout behind it
    for "out", blended towards nodes even in theta across the pitch.

    The two stacked into one band across the pitch, from where the blocks
    round the edge end, Z_in or Z_out, to the blade, its ends across the
    pitch held; behind the blade, its blocks taken the other way round in
    `i`, so the same holds. Each node's place along its `k` line, as a fraction of the
    line's length, goes `w` of the way from where it is to where the line
    reaches its share of the theta between its ends. `w` is set station by
    station from the blade, where it is nought, towards the block ahead's
    end: the most, rising by no more than `dw_max` a station and not
    falling unless it must, that leaves every cell between that station and
    the one before with its least corner angle no less than `skew_min`, or
    than it had if that was less. The block ahead follows the band where
    they meet: each of its lines along it moved in theta as its node there
    is, all along, and in m' there only, fading to none at its other end.

    With `cap`, each station's weight is no more than it. With `weights`,
    also return the weights, the band as it was, and the band fully even
    in theta, each (2, ni, nk), or None for all three if there is no band.
    """
    out = [np.asarray(B, dtype=float) for B in xy]
    # The blocks round this edge, ordered with where they end at i = 0 and
    # the block beyond with it at i = -1, as ahead of the blade they are
    b_edge, b_pass, b_beyond = (2, 3, 6) if side == "in" else (4, 5, 7)
    flip = (lambda B: B[:, ::-1]) if side == "out" else (lambda B: B)
    LE, Pin, up = (flip(out[b]) for b in (b_edge, b_pass, b_beyond))
    tol = 1e-9 * max(np.ptp(LE[0]), np.ptp(Pin[0]))

    # The band: Pin then LE, joined where they meet, a pitch apart if so
    def end(B, k):
        return B[:, :, k]

    join = None
    for kp in (0, -1):
        for kl in (0, -1):
            for d in (-1.0, 0.0, 1.0):
                gap = end(Pin, kp) - end(LE, kl) - np.array([[0.0], [d * pitch]])
                if np.abs(gap).max() < tol:
                    join = (kp, kl, d)
    if join is None:
        return (out, None, None, None) if weights else out
    kp, kl, d = join
    A = Pin if kp == -1 else Pin[:, :, ::-1]
    B = (LE if kl == 0 else LE[:, :, ::-1]) + np.array([0.0, d * pitch])[:, None, None]
    band = np.concatenate((A, B[:, :, 1:]), axis=2)
    ni, nk = band.shape[1:]

    def arc(line):
        return np.concatenate(
            ([0.0], np.cumsum(np.linalg.norm(np.diff(line, axis=1), axis=0)))
        )

    def station(i):
        """The `k` line at station i, its arc length and, if it runs one way
        in theta, the arc length at which it would be even in theta."""
        line = band[:, i, :]
        s = arc(line)
        y = line[1]
        if not (np.all(np.diff(y) > 0) or np.all(np.diff(y) < 0)):
            return line, s, None
        y_even = np.linspace(y[0], y[-1], nk)
        order = np.argsort(y)
        return line, s, np.interp(y_even, y[order], s[order])

    def blended(line, s, s_even, w):
        """The `k` line of a station, `w` of the way to even in theta."""
        if w == 0.0 or s_even is None:
            return line
        s_new = (1.0 - w) * s + w * s_even
        return _interp_rows(s_new, s, line)

    new = band.copy()
    full = band.copy()
    w = np.zeros(ni)
    cap = np.ones(ni) if cap is None else np.asarray(cap, dtype=float)
    # Each station's line, and the floor of the cells between it and the one
    # after, nearer the blade
    stations = [None] * ni

    def holds(k, a, b):
        """Whether the cells between lines `a` at station k and `b` at k + 1
        keep to their floor."""
        return np.all(_strip_angles(b)(a) >= floors[k] - 1e-9)

    # The floors of every strip of the band as it was, at once
    floors = list(np.minimum(_strip_angles(band[:, 1:])(band[:, :-1]), skew_min))
    for i in range(ni - 2, -1, -1):
        angles = _strip_angles(new[:, i + 1])
        floor = floors[i]
        hi = min(1.0, w[i + 1] + dw_max, cap[i])
        lo = min(w[i + 1], hi)
        line, s, s_even = stations[i] = station(i)
        full[:, i] = blended(line, s, s_even, 1.0)

        def ok(wi):
            return np.all(angles(blended(line, s, s_even, wi)) >= floor - 1e-9)

        # Whether the line put at this station is known to keep to the floor
        # against the one after, by the same test `holds` would make
        kept = ok(hi)
        if kept:
            lo = hi
        else:
            if not ok(lo):
                # Holding the weight of the station before would break the
                # floor too: the most below it that does not, or none
                lo, hi = 0.0, lo
            for _ in range(THETA_BISECT):
                mid = 0.5 * (lo + hi)
                lo, hi = (mid, hi) if ok(mid) else (lo, mid)
        w[i] = lo
        new[:, i] = blended(line, s, s_even, lo)
        # Where none at this station keeps the floor, the stations nearer
        # the blade took too much: each lowered in turn, as little as keeps
        # the cells either side of it, until the floor holds. All of them at
        # none is the band as it was, which does.
        k = i
        while (
            not (kept and k == i)
            and not holds(k, new[:, k], new[:, k + 1])
            and k + 1 < ni - 1
        ):
            m = k + 1

            def both(wm):
                line_m = blended(*stations[m], wm)
                return holds(k, new[:, k], line_m) and holds(m, line_m, new[:, m + 1])

            lo, hi = 0.0, w[m]
            for _ in range(THETA_BISECT):
                mid = 0.5 * (lo + hi)
                lo, hi = (mid, hi) if both(mid) else (lo, mid)
            w[m] = lo
            new[:, m] = blended(*stations[m], lo)
            k = m
    logger.debug(
        f"Theta-even blend, side {side}, from where the blocks end to the "
        f"blade: {np.round(w, 2)}"
    )

    # Back into the two blocks
    nA = A.shape[2]
    A_new, B_new = new[:, :, :nA], new[:, :, nA - 1 :]
    B_new = B_new - np.array([0.0, d * pitch])[:, None, None]
    out[b_pass] = flip(A_new if kp == -1 else A_new[:, :, ::-1])
    out[b_edge] = flip(B_new if kl == 0 else B_new[:, :, ::-1])

    # The block ahead follows where they meet: its face there on the band's
    # first station, found node by node, a pitch either way if so
    shift = new[:, 0] - band[:, 0]
    tree = cKDTree(band[:, 0].T)
    best = None
    for dd in (0.0, -1.0, 1.0):
        dist, idx = tree.query((up[:, -1] - np.array([0.0, dd * pitch])[:, None]).T)
        if best is None or dist.max() < best[0].max():
            best = (dist, idx)
    idx = best[1]
    ni_up = up.shape[1]
    up_new = up.copy()
    for i in range(ni_up):
        # Each line along the block moved in theta as its node on the face
        # is, the same all along, so its cells keep their shape; in m' only
        # where it meets the band, fading to none at its other end, the
        # inlet or outlet, which slides in theta only
        step = shift[:, idx].copy()
        step[0] *= i / (ni_up - 1)
        up_new[:, i] = up[:, i] + step
    out[b_beyond] = flip(up_new)
    return (out, w, band, full) if weights else out


def _plane_topology(xy, pitch, stencils=True):
    """Return the topology of a plane of a row's main blocks `xy`, coarse or
    refined: the O block's blade nodes fixed, the inlet and outlet sliding,
    and where the blocks ahead of and behind the blade end; with the
    sweeps' `stencils`, or only what the polish needs."""
    shapes = [B.shape[1:] for B in xy]
    fixed = [np.zeros(sh, dtype=bool) for sh in shapes]
    slide = [np.zeros(sh, dtype=bool) for sh in shapes]
    fixed[0][:, 0] = True
    slide[6][0, :] = True
    slide[7][-1, :] = True
    # Where the blocks ahead of and behind the blade end, held where `cut`
    # and `cut_mixing` put them rather than drawn along by the smoothing: the
    # faces of the inlet and outlet blocks nearer the blade, unless a block
    # is no more than its two faces, to be extruded once refined
    if shapes[6][0] > 2:
        slide[6][-1, :] = True
    if shapes[7][0] > 2:
        slide[7][0, :] = True
    return elliptic.PlaneTopology(
        xy, pitch=pitch, fixed=fixed, slide=slide, stencils=stencils
    )


def _project(wall, s_wall, tree, pts):
    """Return `s_wall`, sampled along `wall` (2, n), at the point of `wall`
    nearest each of `pts` (m, 2).

    On the segment either side of the nearest sample, so between samples
    rather than snapped to one.
    """
    k = np.clip(tree.query(pts)[1], 1, wall.shape[1] - 2)
    best_d = np.full(len(pts), np.inf)
    best_s = np.zeros(len(pts))
    for a, b in ((k - 1, k), (k, k + 1)):
        A, B = wall[:, a].T, wall[:, b].T
        AB = B - A
        t = np.clip(np.sum((pts - A) * AB, axis=1) / np.sum(AB * AB, axis=1), 0, 1)
        d = np.linalg.norm(A + t[:, None] * AB - pts, axis=1)
        better = d < best_d
        best_d[better] = d[better]
        best_s[better] = (s_wall[a] + t * (s_wall[b] - s_wall[a]))[better]
    return best_s


def _wall_arc(plane, nodes):
    """Return where `nodes` (2, n) lie along the closed line of `plane`'s
    blade, unwrapped to increase from the first, which the last closes on."""
    wall = plane["wall"]
    # Kept with the plane, which asks for them several times
    if "wall_tree" not in plane:
        plane["wall_s"] = util.cum_arc_length(wall)
        plane["wall_tree"] = cKDTree(wall.T)
    s_wall = plane["wall_s"]
    S = s_wall[-1]
    s_c = _unwrap(_project(wall, s_wall, plane["wall_tree"], nodes.T), S)
    s_c[-1] = s_c[0] + S
    return s_c


def _unwrap(s, S):
    """Return arc lengths `s` round a closed line of length `S`, one per node
    from its start to its end, each on the side of the cut its node is.

    A node near either end can land just across the cut, at the far end of
    the line; it is put back by a turn of `S`, past the start or the end.
    """
    s = s.copy()
    half = len(s) // 2
    s[:half] = np.where(s[:half] > 0.75 * S, s[:half] - S, s[:half])
    s[half:] = np.where(s[half:] < 0.25 * S, s[half:] + S, s[half:])
    return s


def _bend(A, dA, B, dB, n):
    """Return `n + 1` points (2, n + 1) on a cubic from `A` to `B`, leaving `A`
    along `dA` and arriving at `B` along `dB`, evenly in its parameter."""
    L = np.linalg.norm(B - A) / 3.0
    t = np.linspace(0.0, 1.0, n + 1)[None]
    P1, P2 = A + L * dA, B - L * dB
    return (
        (1 - t) ** 3 * A[:, None]
        + 3 * (1 - t) ** 2 * t * P1[:, None]
        + 3 * (1 - t) * t**2 * P2[:, None]
        + t**3 * B[:, None]
    )


def _pairs(marks):
    """Return the stretches between consecutive `marks`."""
    return list(itertools.pairwise(marks))


def _wall_counts(need, tip_gap):
    """Return refined cells along each piece of the blade, in the order of
    `_Layout.n_wall`, for the cells `need` the density along each asks for.

    Each count rounded up, never down, so no piece is coarser than asked.
    The two sides between corners are the opposite edges of the passage
    block, so they share the larger count; it and the leading edge's are
    multiples of eight, and so is the sum of the two rear pieces, the edge
    of the trailing edge block, the remainder going on the rear upper side.
    """
    n = [int(np.ceil(x - 1e-6)) for x in need]
    nf_side = _up8(max(n[1], n[3]))
    nf_front = _up8(n[2])
    nf_rl = max(n[0], 2)
    nf_ru = max(n[4], 2)
    nf_ru += -(nf_rl + nf_ru) % 8
    # The H block of a tip gap takes half the cells round the blade along
    # each side, and its counts must halve under multigrid too.
    if tip_gap:
        nf_ru += -(nf_rl + nf_ru + 2 * nf_side + nf_front) % 16
    return (nf_rl, nf_side, nf_front, nf_side, nf_ru)


def _up8(n):
    """Return `n` rounded up to a multiple of eight, at least eight."""
    return max(8, 8 * int(np.ceil(n / 8)))


def _face_lay(lay):
    """Return `lay` with the sides of the blade refined as along the O
    block's outer face, if :meth:`_face_u` has set them apart, rather than
    as along the blade."""
    u = dict(lay.u)
    for name in ("side_lo", "side_up"):
        if name + "_face" in u:
            u[name] = u[name + "_face"]
    return dataclasses.replace(lay, u=u)


def _corner_spokes(out, lay, face):
    """Return, for each corner of the O block of refined blocks `out`, its
    `i` and the first cells of the lines out of it but the O block's own
    line in: along the O block's outer face `face` (2, ni) either way, into
    the passage block, and into the block ahead of or behind the blade,
    keyed "face-", "face+", "P" and "LE" or "TE"."""
    P, LE, TE = out[1], out[2], out[4]
    i3, i4, i1, i2 = lay.i_corners()

    def cell(a, b):
        return float(np.linalg.norm(np.asarray(b, float) - np.asarray(a, float)))

    def along(i):
        return {
            "face-": cell(face[:, i - 1], face[:, i]),
            "face+": cell(face[:, i], face[:, i + 1]),
        }

    return [
        (
            i1,
            {
                **along(i1),
                "P": cell(P[:, 0, 0], P[:, 0, 1]),
                "LE": cell(LE[:, -1, -1], LE[:, -2, -1]),
            },
        ),
        (
            i2,
            {
                **along(i2),
                "P": cell(P[:, -1, 0], P[:, -1, 1]),
                "TE": cell(TE[:, 0, -1], TE[:, 1, -1]),
            },
        ),
        (
            i4,
            {
                **along(i4),
                "P": cell(P[:, 0, -1], P[:, 0, -2]),
                "LE": cell(LE[:, -1, 0], LE[:, -2, 0]),
            },
        ),
        (
            i3,
            {
                **along(i3),
                "P": cell(P[:, -1, -1], P[:, -1, -2]),
                "TE": cell(TE[:, 0, 0], TE[:, 1, 0]),
            },
        ),
    ]


def _scale_end(u, F, n_fade, at_end):
    """Return refined fractions `u` with the step at one end scaled by `F`,
    the scaling faded smoothly to none over `n_fade` steps and the other
    steps scaled alike to keep the total: at the last step if `at_end`,
    else the first."""
    du = np.diff(np.asarray(u, dtype=float))
    if at_end:
        du = du[::-1]
    j = np.arange(len(du))
    t = np.clip(j / max(n_fade, 1.0), 0.0, 1.0)
    F = min(F, 0.5 * du.sum() / du[0])
    du = du * F ** (1.0 - t * t * (3.0 - 2.0 * t))
    rest = du[1:].sum()
    du[1:] *= (u[-1] - u[0] - du[0]) / rest
    if at_end:
        du = du[::-1]
    return u[0] + np.concatenate(([0.0], np.cumsum(du)))


def _beyond_O(out, lay):
    """Return the length of the first cell beyond each node of the O block's
    outer face, in the refined blocks `out` of a plane.

    Taken from the passage block on the sides of the blade, the block ahead
    of the leading edge round the front, and the one behind the trailing
    edge round the rear; at a corner, the mean of the two blocks that meet
    there. Not `h`: those blocks are refined as fractions of their lengths
    set at mid-span, and on a swept plane their first cells differ from it
    by up to twice either way.
    """
    O, P, LE, _, TE = out[:5]
    n = O.shape[1]
    i3, i4, i1, i2 = lay.i_corners()
    total = np.zeros(n)
    count = np.zeros(n)
    rear = np.concatenate((np.arange(i3, -1, -1), np.arange(n - 2, i2 - 1, -1)))
    for idx, face, next_ in (
        (np.arange(i1, i2 + 1), P[:, :, 0], P[:, :, 1]),
        (np.arange(i4, i3 - 1, -1), P[:, :, -1], P[:, :, -2]),
        (np.arange(i4, i1 + 1), LE[:, -1, :], LE[:, -2, :]),
        (rear, TE[:, 0, :], TE[:, 1, :]),
    ):
        assert len(idx) == face.shape[1], "O block outer face out of step"
        total[idx] += np.linalg.norm(next_ - face, axis=0)
        count[idx] += 1
    count[-1], total[-1] = count[0], total[0]
    return total / count


def _cross(a, b):
    """Return the 2D cross product of `a` and `b`, (2, ...) alike."""
    return a[0] * b[1] - a[1] * b[0]


def _intersect(Q, P, n):
    """Return where the line through `P` along `n` (2,) crosses polyline `Q`
    (2, M) nearest `P`, or None if it does not."""
    A, d = Q[:, :-1], np.diff(Q, axis=1)
    den = _cross(n[:, None], d)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = _cross(A - P[:, None], d) / den
        u = _cross(A - P[:, None], n[:, None]) / den
    ok = np.isfinite(t) & (u >= 0.0) & (u <= 1.0)
    if not ok.any():
        return None
    t = t[ok][np.argmin(np.abs(t[ok]))]
    return P + t * n


def _inscribed_radius(P, n_in, wall):
    """Return the radius of the largest circle inside the blade, closed line
    `wall`, that touches it at each of points `P` (2, n) on it, whose inward
    normals are `n_in` (2, n).

    Through the finely sampled blade rather than the mesh's nodes, so that
    it varies smoothly from node to node however unevenly the nodes are
    spread between the two sides.
    """
    Q = wall.T
    h2 = (util.arc_length(wall) / (wall.shape[1] - 1)) ** 2
    r = np.empty(P.shape[1])
    # The circle tangent at a through b has radius |b - a|^2 / (2 (b - a) . n_a);
    # the largest inside is the least of these over the points on its inward
    # side, those within a sample of a left out, where the slightest error in
    # the normal makes it vanish. A few nodes at a time, to bound the memory.
    for a in range(0, P.shape[1], 16):
        rel = Q[None] - P[:, a : a + 16].T[:, None]
        along = np.einsum("abc,ac->ab", rel, n_in[:, a : a + 16].T)
        dist2 = np.sum(rel**2, axis=-1)
        with np.errstate(divide="ignore", invalid="ignore"):
            rad = dist2 / (2.0 * along)
        ok = (along > 0.0) & (dist2 > h2)
        r[a : a + 16] = np.where(ok, rad, np.inf).min(axis=1)
    return r


def _gap_offset(W, wall, frac, *, tune):
    """Return outline `W` (2, N + 1) offset inwards along the blade's normal.

    Along the normal of the blade, closed line `wall`, at each node, by
    `frac` of the blade's greatest thickness, the diameter of the largest
    circle inside it, capped at `gap_cap` of the radius of the largest circle
    inside it that touches it at that node, round the edges where the blade
    is thinner.
    """
    n = _wall_normals(wall, _wall_arc({"wall": wall}, W), tune=tune)[:, :-1]
    r = _inscribed_radius(W[:, :-1], -n, wall)
    d = np.minimum(frac * 2.0 * r.max(), tune.gap_cap * r)
    inner = W - np.append(d, d[0]) * np.append(n, n[:, :1], axis=1)
    inner[:, -1] = inner[:, 0]
    return inner


class _Camber:
    """A blade's camber line on a section, the midpoint of its two surfaces,
    and the lines normal to it."""

    def __init__(self, upper, lower):
        self.xy = 0.5 * (upper + lower)
        self.s = util.cum_arc_length(self.xy)
        t = np.gradient(self.xy, self.s, axis=1)
        self.t = t / np.linalg.norm(t, axis=0)
        self.tree = cKDTree(self.xy.T)

    def at(self, pts):
        """Return the arc length along the camber nearest each of `pts` (2, n)."""
        return _project(self.xy, self.s, self.tree, np.atleast_2d(pts.T))

    def tangent(self, s):
        """Return the unit tangent (2,) at arc length `s`, towards the tail."""
        t = _interp_rows(s, self.s, self.t)
        return t / np.linalg.norm(t)

    def normal(self, s):
        """Return the point (2,) at arc length `s` and the unit normal there."""
        P = _interp_rows(s, self.s, self.xy)
        t = self.tangent(s)
        return P, np.array([-t[1], t[0]])

    def across(self, s, sides):
        """Return where the normal at `s` meets each of `sides`, or None."""
        P, n = self.normal(s)
        return [_intersect(Q, P, n) for Q in sides]


def _pairing_drift(blade, cmap, section=None):
    """Return how far apart along the camber line points at equal fractions
    of the two surfaces' lengths lie, at most, as a fraction of its length,
    on `blade`'s section at `cmap`'s span fraction (`section` if given): near
    nothing on a thin blade whose surfaces are about as long, a fifth on a
    thick, highly cambered one."""
    upper, lower = _surfaces(blade, cmap, cmap.spf, section)
    cam = _Camber(upper, lower)
    t = np.linspace(0.0, 1.0, 401)

    def at_fractions(P):
        s = util.cum_arc_length(P)
        return _interp_rows(t * s[-1], s, P)

    u = cam.at(at_fractions(upper)) - cam.at(at_fractions(lower))
    return float(np.abs(u).max() / cam.s[-1])


class _GapScan:
    """Lines normal to the camber line of a tip gap's inner face `inner`
    (2, N + 1), at `n_gap_scan` points from its nose to its tail: where each
    meets the two sides, the width across there, and the H block's corners
    on them."""

    def __init__(self, inner, i_LE, cam, mesher):
        self.inner, self.i_LE = inner, i_LE
        self.sides = (inner[:, : i_LE + 1], inner[:, i_LE:])
        s_nose, s_tail = cam.at(inner[:, [i_LE, 0]])
        self.s = np.linspace(s_nose, s_tail, mesher.tune.n_gap_scan)
        self.u = np.linspace(0.0, 1.0, mesher.tune.n_gap_scan)
        self.across = [cam.across(s, self.sides) for s in self.s]
        self.ok = np.array([p is not None and q is not None for p, q in self.across])
        self.width = np.array(
            [
                np.linalg.norm(q - p) if k else np.nan
                for (p, q), k in zip(self.across, self.ok)
            ]
        )
        # Distances from the nose and to the tail, in half widths
        self.from_nose = (self.s - s_nose) / (0.5 * self.width)
        front = self.ok & (self.from_nose >= mesher.tune.tip_end)
        if not front.any():
            raise ValueError(
                "The tip gap's H block has no room behind its front: reduce tip_end."
            )
        self.j_F = int(np.argmax(front))
        self.iB, self.iC = self._nearest(self.j_F)

    def _nearest(self, j):
        """Return the nodes of each side nearest where line `j` meets it."""
        return tuple(
            offset + int(np.argmin(np.linalg.norm(Q - P[:, None], axis=0)))
            for Q, P, offset in zip(self.sides, self.across[j], (0, self.i_LE))
        )


def _gap_rear_search(
    inner, i_LE, cam, scan, to_tail, multiple=8, most=0, symmetric=False
):
    """Return the rear corners (iA, iD) of a tip gap's H block, nodes of
    inner face `inner` (2, N + 1), whose line is nearest normal to the
    camber.

    Over every A on side 1 behind the front corner B of `scan`, and every
    D on side 2 behind C a multiple of `multiple` cells round the tail from
    it, the H block's count across, its nodes no more than `most` if that
    is not nought, whose line's middle lies from `to_tail` to twice that
    from the tail, in half widths across there; failing any, those at least
    `to_tail` from it. The rear line runs through the two
    corners' own nodes, so neither side slides onto it and both O lines
    leave the outline square; the pair kept is the one whose line leans
    least off the camber's normal at its middle, then the one with fewest
    cells round the tail. With `symmetric`, only pairs as far either way round
    from the cut, D the `N - A`-th node, the halves of the inner face being
    alike.
    """
    N = inner.shape[1] - 1
    iB, iC = scan.iB, scan.iC
    s_tail = cam.at(inner[:, [0]])[0]
    ok = scan.ok
    best = (None, None)
    for iA in range(1, iB):
        counts = range(multiple * (iA // multiple + 1), N, multiple)
        if symmetric:
            counts = [2 * iA] if (2 * iA) % multiple == 0 else []
        for count in counts:
            if most and count + 1 > most:
                break
            iD = iA + N - count
            if iD <= iC:
                break
            v = inner[:, iD] - inner[:, iA]
            s_mid = float(cam.at(0.5 * (inner[:, iA] + inner[:, iD])[:, None])[0])
            half = 0.5 * np.interp(s_mid, scan.s[ok], scan.width[ok])
            back = (s_tail - s_mid) / half
            if back < to_tail:
                continue
            _, n = cam.normal(s_mid)
            tilt = np.degrees(np.arccos(min(1.0, abs(v @ n) / np.linalg.norm(v))))
            key = (tilt, count)
            near = int(back > 2.0 * to_tail)
            if best[near] is None or key < best[near][0]:
                best = (*best[:near], (key, iA, iD, s_mid), *best[near + 1 :])
    best = best[0] or best[1]
    if best is None:
        raise ValueError(
            f"The tip gap's H block has no rear corners a multiple of {multiple} "
            "cells round the tail behind its front corners."
        )
    (tilt, _), iA, iD, _ = best
    logger.debug(f"Tip gap rear corners A={iA}, D={iD}, {tilt:.2f} deg off normal")
    return iA, iD


def _on_polyline(Q, p):
    """Return the fractional index along polyline `Q` (2, n) of point `p` on
    it, or nearest it."""
    a, seg = Q[:, :-1], np.diff(Q, axis=1)
    t = np.sum((p[:, None] - a) * seg, axis=0) / np.sum(seg**2, axis=0)
    t = np.clip(t, 0.0, 1.0)
    k = int(np.argmin(np.linalg.norm(a + t * seg - p[:, None], axis=0)))
    return k + float(t[k])


def _gap_spread(inner, corners, rear, i_LE, match, fade=None):
    """Return inner face `inner` (2, N + 1) with the nodes of side `match`
    between the H block's corners spread along it, so that its rear corner
    lands where the rear line meets that side, at `rear` (pA, pD), and its
    front corner stays: each keeps its fraction of the arc length between.
    The rear corner is a node a multiple of eight cells from the front one;
    on the side the H block matches, as far as half that from the crossing,
    and on the other at or behind it, by less than eight cells (see
    :func:`_gap_corners`). Spread, the O block's lines lean a little along
    the side rather than crush the cells by it.
    With `fade`, a length, each node's slide eases smoothly out to nothing
    that far along the side from the rear corner, the rest staying put."""
    iA, iB, iC, iD = corners
    if match == 1:
        i0, i1, u0 = iA, iB, _on_polyline(inner[:, : i_LE + 1], rear[0])
    else:
        i0, i1, u0 = iD, iC, i_LE + _on_polyline(inner[:, i_LE:], rear[1])
    s = util.cum_arc_length(inner)
    s0 = np.interp(u0, np.arange(len(s)), s)
    step = 1 if i1 > i0 else -1
    idx = np.arange(i0, i1 + step, step)
    f = (s[idx] - s[i0]) / (s[i1] - s[i0])
    s_new = s0 + f * (s[i1] - s0)
    if fade is not None:
        s_new = s[idx] + (1.0 - _smoothstep(np.abs(s[idx] - s[i0]) / fade)) * (
            s_new - s[idx]
        )
    out = inner.copy()
    out[:, idx] = _interp_rows(s_new, s, inner)
    return out


def _gap_lean(inner, W, corners, cam, lean, rear=False):
    """Return inner face `inner` (2, N + 1) with the H block's front corners
    B and C, or with `rear` its rear corners A and D, slid along it towards
    the place along camber `cam` midway between them, each no further than
    leans the O block's line from it to outline `W` `lean` degrees off the
    normal; the nodes on its side towards the other end of the block
    following it less and less, to none half way to the corner there.
    Round the nose, the nodes between B and C are spread in proportion
    between where the two now sit, so none is left bunched against a corner
    that slid towards it: :func:`_gap_square` spaces the nose part way to
    them. Round the tail they are spread again when it is squared off."""
    iA, iB, iC, iD = corners
    N = inner.shape[1] - 1
    s = util.cum_arc_length(inner)
    pairs = ((iA, iB), (iD, iC)) if rear else ((iB, iA), (iC, iD))
    u = cam.at(inner[:, [pairs[0][0], pairs[1][0]]])
    target = 0.5 * (u[0] + u[1])
    out = inner.copy()
    tan = np.tan(np.radians(lean))
    s_corner = {}
    for i, i_rear in pairs:
        # Where along the face, within reach of the node, is nearest the
        # target along the camber; its side behind it eased from the whole
        # slide to none
        a, pts = _within_reach(inner, s, W, i, tan)
        slide = a[int(np.argmin(np.abs(cam.at(pts) - target)))] - s[i]
        s_corner[i] = s[i] + slide
        idx, s_new = _slid_arcs(s, i, i_rear, slide)
        out[:, idx] = _interp_rows(s_new, s, inner)
    if not rear:
        iB, iC = pairs[0][0], pairs[1][0]
        idx = np.arange(iB + 1, iC)
        x = (s[idx] - s[iB]) / (s[iC] - s[iB])
        s_new = s_corner[iB] + x * (s_corner[iC] - s_corner[iB])
        out[:, idx] = _interp_rows(s_new, s, inner)
    out[:, N] = out[:, 0]
    return out


def _slid_arcs(s, i, i_far, slide):
    """Return the nodes from `i` to `i_far` and their arc lengths, `s` round
    a face, with node `i` slid `slide` along it and those towards `i_far`
    following less and less, to none half way there."""
    step = 1 if i_far > i else -1
    idx = np.arange(i, i_far + step, step)
    x = np.abs(s[idx] - s[i]) / (0.5 * abs(s[i_far] - s[i]))
    return idx, s[idx] + (1.0 - _smoothstep(x)) * slide


def _within_reach(inner, s, W, i, tan):
    """Return arc lengths round face `inner`, `s` along it, within reach of
    node `i`, as far as leans the line from it to outline `W` by an angle of
    tangent `tan`, and the points there."""
    reach = tan * np.linalg.norm(W[:, i] - inner[:, i])
    a = s[i] + np.linspace(-reach, reach, 201)
    return a, _interp_rows(a, s, inner)


def _gap_slide(inner, s, i, i_far, slide):
    """Return inner face `inner` (2, N + 1), arc lengths `s` along it, with
    node `i` slid `slide` along it (:func:`_slid_arcs`)."""
    idx, s_new = _slid_arcs(s, i, i_far, slide)
    out = inner.copy()
    out[:, idx] = _interp_rows(s_new, s, inner)
    return out


def _gap_rear_bisect(inner, W, corners, mesher, lean, passes=4):
    """Return inner face `inner` (2, N + 1) with the H block's rear corners
    A and D each slid along it to where the O block's line from it to
    outline `W` best bisects the turn between its side and the rear line
    through both, on the face squared off (:func:`_gap_square`), each no
    further than leans that line `lean` degrees off the normal from where it
    was; the nodes towards the front corner on its side following it less
    and less, to none half way there. The rear line moving with the
    corners, found again over `passes`."""
    iA, iB, iC, iD = corners
    N = inner.shape[1] - 1
    s = util.cum_arc_length(inner)
    tan = np.tan(np.radians(lean))
    slides = {iA: 0.0, iD: 0.0}
    ends = {iA: (iB, iA + 1, iA - 1), iD: (iC, iD - 1, iD + 1)}

    def slid():
        out = inner
        for i, (i_far, _, _) in ends.items():
            out = _gap_slide(out, s, i, i_far, slides[i])
        out = out.copy()
        out[:, N] = out[:, 0]
        return out

    for _ in range(passes):
        now = slid()
        sq, _ = _gap_square(now, corners, (now[:, iA], now[:, iD]), mesher)
        for i, (i_far, i_side, i_tail) in ends.items():
            t1, t2 = (sq[:, k] - sq[:, i] for k in (i_side, i_tail))
            b = -(t1 / np.linalg.norm(t1) + t2 / np.linalg.norm(t2))
            a, pts = _within_reach(inner, s, W, i, tan)
            o = W[:, i, None] - pts
            cos = (b @ o) / np.linalg.norm(b) / np.linalg.norm(o, axis=0)
            slides[i] = a[int(np.argmax(cos))] - s[i]
    return slid()


def _gap_corners(inner, i_LE, cam, mesher, match=1):
    """Return the corners (iA, iB, iC, iD) of a tip gap's H block, nodes of
    inner face `inner` (2, N + 1).

    The rear corners A and D by :func:`_gap_rear_search`, the pair whose
    line through their own nodes is nearest normal to the camber, a multiple
    of `tip_rear_multiple` cells round the tail apart. The front corners B
    and C as many cells round the nose, so the H block's front meets the O
    block node for node, centred where a line normal to the camber, back
    from the inner face's nose by `tip_end` of half the width across there,
    meets each side (:class:`_GapScan`).

    With `match` 3, both sides node for node, the corners are instead
    paired across the halves of the inner face, which are alike.
    """
    scan = _GapScan(inner, i_LE, cam, mesher)
    iA, iD = _gap_rear_search(
        inner,
        i_LE,
        cam,
        scan,
        mesher.tune.tip_end_rear,
        mesher.tune.tip_rear_multiple,
        mesher.tune.tip_rear_max,
        symmetric=match == 3,
    )
    N = inner.shape[1] - 1
    if match == 3:
        # The halves of the inner face alike, node h - k across from h + k:
        # the rear corners as far round from the cut either way, and the front
        # as far round from the leading edge, so the rear and front have as
        # many cells, and the sides too
        h = N // 2
        if not iA < h - iA:
            raise ValueError(
                f"The tip gap's rear corners {(iA, iD)} are past its front ones "
                f"{(h - iA, h + iA)}, the halves of its inner face {h} cells."
            )
        return iA, h - iA, h + iA, iD
    # As many cells round the nose as the tail, centred where the scan put
    # the front corners, so only the H block's unmatched side does not match
    n = iA + N - iD
    iB = int(round(0.5 * (scan.iB + scan.iC - n)))
    corners = (iA, iB, iB + n, iD)
    if not 0 < iA < iB < iB + n < iD < N:
        raise ValueError(
            f"The tip gap's front corners, {n} cells round the nose to match "
            f"the tail, overlap its rear corners {(iA, iD)}."
        )
    return corners


def _gap_square(inner, corners, rear, mesher):
    """Return inner face `inner` (2, N + 1) squared off across the nose and
    the tail between the H block's `corners`, and its front face (2, nk), as
    many nodes across as the inner face has from A back through the cut to
    D.

    Nodes A and D are moved to `rear`, the points (2,) of each side its rear
    line meets; B and C stay where they are.

    The nodes beyond the corners, round the nose or the tail, are spread
    evenly along the line between. The H block's
    front is the nose's line too, through as many nodes as round the tail,
    spaced as those round the nose,
    meeting them by a non-matching interface. The tail's line is bowed aft, its middle
    `tip_bow` of the way to where the inner face crosses the cut; the
    nose's forward, normal to the line from B to C, away from the tail,
    `tip_bow_front` of the way to the inner face round the nose.
    """
    iA, iB, iC, iD = corners
    N = inner.shape[1] - 1
    pA, pD = rear
    pB, pC = inner[:, iB], inner[:, iC]

    # Quadratics through control points: the rear's middle moved aft
    # towards the cut, the front's forward normal to B-C
    M_rear = 0.5 * (pA + pD)
    tail = 2.0 * (M_rear + mesher.tune.tip_bow * (inner[:, 0] - M_rear)) - M_rear
    M = 0.5 * (pB + pC)
    n = np.array([pB[1] - pC[1], pC[0] - pB[0]]) / np.linalg.norm(pC - pB)
    n *= np.sign(n @ (M - M_rear))
    hit = _intersect(inner[:, iB : iC + 1], M, n)
    nose = 2.0 * (M + mesher.tune.tip_bow_front * (hit - M)) - M

    out = inner.copy()
    front = None
    # Round the nose, from B to C, and round the tail, from A back through
    # the cut to D, each spaced evenly along its line, or part way to as the
    # nodes before squaring by `tip_nose_outline` and `tip_tail_outline`. The H block's front
    # on the nose's curve too, as many nodes spaced as the inner face's from
    # B to C. Graded to the side cells at the corners instead, which differ
    # several times over where the corners sit at unlike distances from the
    # edge, the finest cells bunch to the side of the finer.
    for idx, P, Q, C in (
        (np.arange(iB, iC + 1), pB, pC, nose),
        (np.r_[np.arange(iA, -1, -1), np.arange(N - 1, iD - 1, -1)], pA, pD, tail),
    ):
        t = np.linspace(0.0, 1.0, 401)
        curve = (
            (1 - t) ** 2 * P[:, None] + 2 * t * (1 - t) * C[:, None] + t**2 * Q[:, None]
        )
        g = util.cum_arc_length(curve)
        f = np.linspace(0.0, 1.0, len(idx))
        w = (
            mesher.tune.tip_nose_outline
            if front is None
            else mesher.tune.tip_tail_outline
        )
        if w:
            # Part way to as the nodes before squaring lie along it
            a = util.cum_arc_length(inner[:, idx])
            f = (1.0 - w) * f + w * a / a[-1]
        out[:, idx] = _interp_rows(f, g / g[-1], curve)
        if front is None:
            u = np.linspace(0.0, 1.0, iA + N - iD + 1)
            fH = np.interp(u, np.linspace(0.0, 1.0, len(idx)), f)
            front = _interp_rows(fH, g / g[-1], curve)
    out[:, N] = out[:, 0]
    if iC - iB == iA + N - iD:
        # As many round the nose as the tail: the front meets them node for node
        front = out[:, iB : iC + 1]
    return out, front


def _gap_H(inner, i_LE, corners, front, cam, match, blend=0.0, *, tune):
    """Return the H block (2, ni, nk) inside squared inner face `inner`
    (2, N + 1), and the `k` at which its rear face crosses the cut.

    `i` runs from the front to the rear, `k` from the lower side, `inner`'s
    nodes from the cut to the leading edge at `i_LE`, to the upper. Its lines
    run straight from each of the inner nodes between the corners on side
    `match`, which it meets node for node, to the same fraction of the way
    along the other side between its corners, which it meets by a
    non-matching interface. Its front face is
    `front` (2, nk), across the nose; its rear face runs from A to D with a
    node on the cut, at `k_cut`, or with its cells either side in
    proportion to the lengths.
    Across, each line's nodes are at fractions blended from the
    front's to the rear's by the matched side's arc length; or, with `blend`,
    even, easing into each end face's own over `blend` of the camber line's
    length from it. Each end
    face's bow eases out along the block over `gap_bow_decay` of its widths.
    With `match` 3, the far side's nodes, as many, are the lines' ends, so it
    meets that side node for node too.
    """
    iA, iB, iC, iD = corners
    N = inner.shape[1] - 1
    # The matched side's nodes, and the far side, which the block meets by a
    # non-matching interface, so the lines may end anywhere along it: at the
    # same fraction of its length between the corners as they start along
    # the matched side, spread as it is, so they neither fold nor bunch
    lo_nodes, up_nodes = inner[:, iB : iA - 1 : -1], inner[:, iC : iD + 1]
    own, far = (lo_nodes, up_nodes) if match == 1 else (up_nodes, lo_nodes)
    f_own = util.cum_arc_length(own)
    s_far = util.cum_arc_length(far)
    ends = _interp_rows(f_own / f_own[-1] * s_far[-1], s_far, far)
    if match == 3:
        if far.shape[1] != own.shape[1]:
            raise ValueError(
                f"The tip gap's H block has {own.shape[1]} nodes along one side "
                f"and {far.shape[1]} along the other."
            )
        ends = far
    lower, upper = (own, ends) if match == 1 else (ends, own)
    nk = front.shape[1]

    # The rear face the inner face's nodes from A back through the cut to D,
    # node for node, the cut at the node `iA` along it
    rear = np.concatenate((inner[:, iA::-1], inner[:, N - 1 : iD - 1 : -1]), axis=1)
    if rear.shape[1] != nk:
        raise ValueError(
            f"The tip gap's H block has {nk} nodes across its front and "
            f"{rear.shape[1]} round the tail."
        )
    k_cut = iA

    def fractions(Q):
        f = util.cum_arc_length(Q)
        return f / f[-1]

    s_own = util.cum_arc_length(own)
    f_front, f_rear = fractions(front), fractions(rear)
    if blend > 0.0:
        # Each end's weight easing smoothly from one at its face to nothing a
        # `blend` of the chord in, shared out where the two overlap
        def ease(d):
            return 1.0 - _smoothstep(d / (blend * cam.s[-1]))

        w_front = ease(s_own - s_own[0])[:, None]
        w_rear = ease(s_own[-1] - s_own)[:, None]
        total = np.maximum(w_front + w_rear, 1.0)
        w_front, w_rear = w_front / total, w_rear / total
        even = np.linspace(0.0, 1.0, nk)
        v = w_front * f_front + w_rear * f_rear + (1.0 - w_front - w_rear) * even
    else:
        t = fractions(own)[:, None]
        v = (1.0 - t) * f_front + t * f_rear
    H = (1.0 - v) * lower[:, :, None] + v * upper[:, :, None]

    # Each end face's bow off the straight line between its corners, eased
    # out along the block, so the lines bow gradually
    for face, dist in ((front, s_own - s_own[0]), (rear, s_own[-1] - s_own)):
        P0, P1 = face[:, 0], face[:, -1]
        bow = face - (P0[:, None] + fractions(face) * (P1 - P0)[:, None])
        x = dist / (tune.gap_bow_decay * np.linalg.norm(P1 - P0))
        H += (1.0 - _smoothstep(x))[None, :, None] * bow[:, None, :]
    H[:, 0], H[:, -1] = front, rear
    return H, k_cut


def _gap_O_lines(inner, W, square, n=129, *, tune):
    """Return the lines (2, ni, n) of a tip gap's O block from its inner face
    `inner` to the blade's outline `W` (2, ni), sampled finely: cubics
    leaving the inner face straight towards the outline and arriving at it
    `square` of the way from straight to along its outward normal, but no
    further than `tip_o_turn` over the line's length times the outline's
    curvature there."""
    chord = W - inner
    L = np.linalg.norm(chord, axis=0)
    nrm = _outward_normals(W, tune=tune)
    # How fast the outline's normal turns along it, per unit length
    turn = np.arccos(np.clip(np.sum(nrm[:, 1:] * nrm[:, :-1], axis=0), -1.0, 1.0))
    ds = np.linalg.norm(np.diff(W, axis=1), axis=0)
    kappa = turn / np.maximum(ds, 1e-300)
    kappa = np.maximum(np.append(kappa, kappa[0]), np.insert(kappa, 0, kappa[-1]))
    w = np.minimum(square, tune.tip_o_turn / np.maximum(L * kappa, 1e-300))
    # Eased round the blade, so neighbouring lines bend alike
    g = np.exp(
        -0.5
        * (np.arange(-3 * tune.tip_o_ease, 3 * tune.tip_o_ease + 1) / tune.tip_o_ease)
        ** 2
    )
    ring = np.concatenate((w[-len(g) - 1 : -1], w[:-1], w[: len(g)]))
    w = np.convolve(ring, g / g.sum(), "same")[len(g) : -len(g)]
    w = np.append(w, w[0])[None]
    end = (1.0 - w) * chord / L + w * nrm
    end *= L / np.linalg.norm(end, axis=0)
    t = np.linspace(0.0, 1.0, n)[None, None]
    h00, h10 = 2 * t**3 - 3 * t**2 + 1, t**3 - 2 * t**2 + t
    h01, h11 = -2 * t**3 + 3 * t**2, t**3 - t**2
    return (
        h00 * inner[..., None]
        + h10 * chord[..., None]
        + h01 * W[..., None]
        + h11 * end[..., None]
    )


def _gap_O_count(O, d_wall, tune):
    """Return the nodes across the O block of a tip gap, lines `O` (2, ni, m).

    As many cells as grow from the wall spacing `d_wall` (ni,) at the outline
    at `ER_O` to fill a line, in the median over the lines, as the O block
    round the blade is laid out: a geometric series from `d` at ratio `r`
    fills `L` in `ln(1 + (r - 1) L / d) / ln(r)` cells. Rounded up to a count
    a multigrid level can halve, and no fewer than `nk_tip_O_min`. The block
    is a fixed fraction of the blade's thickness, so this follows the wall
    spacing: a few times it thick takes the least count, and a finer
    spacing more.
    """
    L = np.linalg.norm(np.diff(O, axis=2), axis=0).sum(axis=1)
    ER = tune.ER_O
    n = float(np.median(np.log1p((ER - 1.0) * L / d_wall))) / np.log(ER)
    nk = max(_up8(int(np.ceil(n))) + 1, tune.nk_tip_O_min)
    logger.debug(f"Tip gap O block: {nk} nodes across, {n:.1f} cells to grow")
    return nk


def _respace_gap_O(O, d_wall, nk=None):
    """Return the O block of a tip gap, its lines `O` (2, ni, m) respaced to
    `nk` nodes each, `m` if not given.

    Along itself from the H block, where its cells are as long as even
    spacing would make them, to the outline, where they are `d_wall` (ni,),
    or as long as even spacing would make them if that is less.
    """
    nk = nk or O.shape[2]
    seg = np.linalg.norm(np.diff(O, axis=2), axis=0)
    s = np.concatenate((np.zeros((len(seg), 1)), np.cumsum(seg, axis=1)), axis=1)
    L = s[:, -1]
    even = 1.0 / (nk - 1)
    T = L[:, None] * clusterfunc.double.fixed_many(
        np.full(len(L), even), np.minimum(d_wall / L, even), nk
    )
    out = np.empty((2, O.shape[1], nk), dtype=O.dtype)
    for c in range(2):
        out[c] = [np.interp(Ti, si, Oi) for Ti, si, Oi in zip(T, s, O[c])]
    out[:, :, 0] = O[:, :, 0]
    out[:, :, -1] = O[:, :, -1]
    return out


def _reverse(u, n):
    """Return fractional indices `u` along a stretch of `n` cells, run the
    other way."""
    return n - u[::-1]


def _integral(s, density, a, b):
    """Return the integral of `density` sampled at `s` from `a` to `b`."""
    t = np.linspace(a, b, 2001)
    return float(np.trapezoid(np.interp(t, s, density), t))


def _by_density(s, density, a, b, n):
    """Return `n + 1` points from `a` to `b` evenly in the integral of
    `density`, sampled at `s`."""
    t = np.linspace(a, b, 4001)
    d = np.interp(t, s, density)
    c = np.concatenate(([0.0], np.cumsum(0.5 * (d[1:] + d[:-1]) * np.diff(t))))
    out = np.interp(np.linspace(0.0, c[-1], n + 1), c, t)
    out[0], out[-1] = a, b
    return out


def _symmetric_halves(blade_wall, density, *, tune):
    """Return the halves of the blade of `blade_wall`, as :meth:`_wall`
    returns it, from the middle of the trailing edge to the leading edge, at
    one density, `density` sampled at its `n_wall` points.

    Returns `t`, fractions of the way along each half; `c`, cells from the
    trailing edge at `t`, by the greater of the two sides' densities there;
    and `s_lo`, `s_up`, where `t` lies on each side, as arc lengths round the
    blade from the middle of the trailing edge.
    """
    _, marks, _, s_LE = blade_wall
    S = marks[-1]
    s = np.linspace(0.0, S, tune.n_wall)
    t = np.linspace(0.0, 1.0, 4001)
    s_lo = t * s_LE
    s_up = S - t * (S - s_LE)
    d = np.maximum(
        np.interp(s_lo, s, density) * s_LE, np.interp(s_up, s, density) * (S - s_LE)
    )
    c = np.concatenate(([0.0], np.cumsum(0.5 * (d[1:] + d[:-1]) * np.diff(t))))
    return t, c, s_lo, s_up


def _symmetric_counts(blade_walls, densities, i_wall, *, tune):
    """Return coarse cells along each piece of the blade, alike on both
    sides, for :func:`_symmetric_wall` on every plane of `blade_walls`.

    Each half as many cells as the most any plane asks for, and each corner
    the node nearest its place on the planes on average, the two sides of the
    passage block moved apart or together about their middles to as many
    cells each. Each piece split in two once refined, the two sides and the
    front are multiples of four, as are the two rear pieces together and
    each half, so the refined counts are the multiples of eight, and sixteen
    round the blade, that :func:`_wall_counts` makes them.
    """
    if tuple(i_wall) != (1, 2, 3, 4):
        raise ValueError(
            f"a tip gap matched all round needs the corners in order, not {i_wall}."
        )
    halves, ks = [], []
    for blade_wall, density in zip(blade_walls, densities):
        t, c, _, _ = _symmetric_halves(blade_wall, density, tune=tune)
        _, marks, _, s_LE = blade_wall
        S = marks[-1]
        # Each corner as a fraction of its half from the trailing edge
        f = [
            marks[1] / s_LE,
            marks[2] / s_LE,
            (S - marks[3]) / (S - s_LE),
            (S - marks[4]) / (S - s_LE),
        ]
        halves.append(c[-1])
        ks.append(np.interp(f, t, c / c[-1]))
    n_half = max(8, 4 * int(np.ceil(max(halves) / 4)))
    k3, k4, k1, k2 = n_half * np.mean(ks, axis=0)
    side = max(4, 4 * round(0.5 * ((k4 - k3) + (k1 - k2)) / 4))
    k3 = max(1, round(0.5 * (k3 + k4) - 0.5 * side))
    m2 = 0.5 * (k1 + k2) - 0.5 * side
    # The rear pieces together, and so the front, a multiple of four
    k2 = min(
        (k for k in range(1, n_half) if (k3 + k) % 4 == 0), key=lambda k: abs(k - m2)
    )
    k4, k1 = k3 + side, k2 + side
    if max(k4, k1) > n_half - 2:
        raise ValueError(
            f"a tip gap matched all round puts a corner at the leading edge: {n_half} cells "
            f"each half, corners {k3}, {k4} from the trailing edge on one side "
            f"and {k2}, {k1} on the other."
        )
    return [k3, side, 2 * n_half - k4 - k1, side, k2]


def _symmetric_wall(blade_wall, density, counts, *, tune):
    """Return `blade_wall` with its corners moved onto the coarse nodes, and
    where those lie along each piece, as fractions of its length, placed
    alike on both sides.

    The `k`-th node from the middle of the trailing edge is as far along
    either half of the blade, to the leading edge, both evenly in the
    integral of the greater of the two sides' densities, `density`, there;
    the corners the nodes `counts` puts them at.
    """
    wall, marks, i_C, s_LE = blade_wall
    N = int(sum(counts))
    if N % 2:
        raise ValueError(
            f"a tip gap matched all round needs an even count round the blade, not {N}."
        )
    n_half = N // 2
    t, c, s_lo, s_up = _symmetric_halves(blade_wall, density, tune=tune)
    t_k = np.interp(np.linspace(0.0, c[-1], n_half + 1), c, t)
    s_nodes = np.empty(N + 1)
    s_nodes[: n_half + 1] = np.interp(t_k, t, s_lo)
    s_nodes[N - n_half :][::-1] = np.interp(t_k, t, s_up)
    s_nodes[0], s_nodes[n_half], s_nodes[N] = 0.0, s_LE, marks[-1]
    ends = np.cumsum((0, *counts))
    marks = s_nodes[ends]
    f_wall = tuple(
        (s_nodes[a : b + 1] - s_nodes[a]) / (s_nodes[b] - s_nodes[a])
        for a, b in itertools.pairwise(ends)
    )
    return (wall, marks, i_C, s_LE), f_wall


def _coarse_density(wall, h_free, turn, ER, *, tune):
    """Return nodes per unit length along a closed line (2, n), evenly spaced,
    for cells of `h_free` or less, across which it turns by no more than
    `turn` radians, growing from one to the next by `ER` at most."""
    d = np.gradient(wall, axis=1)
    phi = np.unwrap(np.arctan2(d[1], d[0]))
    ds = util.arc_length(wall) / (wall.shape[1] - 1)
    kappa = np.abs(np.gradient(phi)) / ds
    kappa = scipy.ndimage.uniform_filter1d(kappa, tune.n_smooth_turn, mode="wrap")
    h = 1.0 / np.maximum(kappa / turn, 1.0 / h_free)
    return 1.0 / _grow_limited(h, util.cum_arc_length(wall), ER - 1.0)


def _grow_limited(h, s, g):
    """Return cell lengths `h` at `s` round a closed line, its last point its
    first, growing away from each short cell at `g` per unit length at most:
    the least over every point of its length plus the rise from it, as running
    minima forwards and back over three turns. A cell growing so is `1 + g`
    times the one before."""
    P = s[-1] - s[0]
    x = np.concatenate((s[:-1] - P, s[:-1], s + P))
    H = np.concatenate((h[:-1], h[:-1], h))
    fwd = g * x + np.minimum.accumulate(H - g * x)
    back = -g * x + np.minimum.accumulate((H + g * x)[::-1])[::-1]
    n = len(s) - 1
    return np.minimum(fwd, back)[n : 2 * n + 1]


def _pchip_unit(y, t):
    """Return the PCHIP through `y` at knots 0, 1, ..., n - 1 at `t`.

    As :class:`scipy.interpolate.PchipInterpolator` makes it, to rounding,
    without the cost of building one, which on a few knots is most of it.
    """
    m = np.diff(y)
    if len(y) == 2:
        d = np.array([m[0], m[0]])
    else:
        # Interior slopes the harmonic mean of the chords either side, or
        # flat at a turn; each end's a one-sided estimate kept in shape.
        d = np.zeros_like(y)
        with np.errstate(divide="ignore", invalid="ignore"):
            whmean = (3.0 / m[:-1] + 3.0 / m[1:]) / 6.0
        ok = (np.sign(m[1:]) == np.sign(m[:-1])) & (m[1:] != 0) & (m[:-1] != 0)
        d[1:-1][ok] = 1.0 / whmean[ok]
        for i, m0, m1 in ((0, m[0], m[1]), (-1, m[-1], m[-2])):
            e = (3.0 * m0 - m1) / 2.0
            if np.sign(e) != np.sign(m0):
                e = 0.0
            elif np.sign(m0) != np.sign(m1) and abs(e) > 3.0 * abs(m0):
                e = 3.0 * m0
            d[i] = e
    k = np.clip(np.floor(t).astype(int), 0, len(y) - 2)
    x = t - k
    c2 = 3.0 * m[k] - 2.0 * d[k] - d[k + 1]
    c3 = d[k] + d[k + 1] - 2.0 * m[k]
    return ((c3 * x + c2) * x + d[k]) * x + y[k]


@functools.cache
def _spline_O(nc, *, tune):
    """Return the samples (n, nc + 1), `n_spline_o` per coarse cell, of the
    cubic spline through `nc` evenly spaced values with a given slope at the
    first and not-a-knot at the last, per unit of each value and the slope.

    The same at every plane, so each O block's lines are splined by one
    product with their coarse nodes and the slopes off the blade.
    """
    u = np.linspace(0.0, nc - 1.0, tune.n_spline_o * (nc - 1) + 1)
    e = np.eye(nc + 1)
    start = ((1, e[nc]), "not-a-knot")
    spline = scipy.interpolate.CubicSpline(np.arange(nc), e[:nc], bc_type=start)
    return np.ascontiguousarray(spline(u))


def _refine_matrix(n, pieces, dtype):
    """Return the matrix (n_fine, n) refining values at `n` coarse nodes.

    `pieces` are stretches `(a, b, u)` from coarse node `a` to `b`, and where
    the refined nodes lie in fractional indices from `a`; each on a cubic
    spline through its own coarse nodes only, so a stretch of edge comes out
    the same in every block that shares it. The matrix is the splines through
    each coarse node alone, the refinement being linear in their values.
    """
    E = np.eye(n)
    parts = []
    for j, (a, b, u) in enumerate(pieces):
        x = np.arange(a, b + 1, dtype=float)
        seg = E[a : b + 1]
        if b - a >= 3:
            vals = scipy.interpolate.CubicSpline(x, seg, axis=0)(a + u)
        else:
            vals = scipy.interpolate.interp1d(x, seg, axis=0)(a + u)
        parts.append(vals if j == 0 else vals[1:])
    return np.concatenate(parts).astype(dtype)


def _strip_angles(b):
    """Return a function of a line `a` (2, nk) giving the worst corner of
    each cell of the strip from `a` to line `b` (2, nk), as
    :func:`_cell_angles` of the block (2, 2, nk) stacking them, to the last
    bit, but with what depends on `b` alone found once."""

    def corner(u0, u1, nu, v0, v1, nv):
        cos = (u0 * v0 + u1 * v1) / np.maximum(nu * nv, 1e-300)
        ang = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
        return 90.0 - np.abs(90.0 - ang)

    def norm(x0, x1):
        return np.sqrt(x0 * x0 + x1 * x1)

    # The cell's corners a_k, b_k, b_k+1, a_k+1 in turn, each between the
    # edges to the corner after it and before it
    # Either line may carry axes between its first and last, for several
    # lines at once
    bx, by = b[0], b[1]
    b0x, b0y, bkx, bky = bx[..., :-1], by[..., :-1], bx[..., 1:], by[..., 1:]
    b1x, b1y = bkx - b0x, bky - b0y
    b2x, b2y = b0x - bkx, b0y - bky

    def angles(a):
        ax, ay = a[0], a[1]
        a0x, a0y, a1x, a1y = ax[..., :-1], ay[..., :-1], ax[..., 1:], ay[..., 1:]
        # The four corners' edges stacked, so each step is done once for all
        ux = np.stack(np.broadcast_arrays(b0x - a0x, b1x, a1x - bkx, a0x - a1x))
        uy = np.stack(np.broadcast_arrays(b0y - a0y, b1y, a1y - bky, a0y - a1y))
        vx = np.stack(np.broadcast_arrays(a1x - a0x, a0x - b0x, b2x, bkx - a1x))
        vy = np.stack(np.broadcast_arrays(a1y - a0y, a0y - b0y, b2y, bky - a1y))
        out = corner(ux, uy, norm(ux, uy), vx, vy, norm(vx, vy)).min(axis=0)
        return np.minimum(90.0, out)

    return angles


def _geometric_many(f0, n):
    """Return `n` points from 0 to 1 for each first cell `f0`, the cells
    growing by the same ratio all the way, found by bisection; even where
    `f0` is no less than even."""
    f0 = np.asarray(f0, dtype=float)[:, None]
    m = n - 1
    lo = np.ones_like(f0)
    hi = np.full_like(f0, 10.0)
    for _ in range(60):
        r = 0.5 * (lo + hi)
        # r reaches 1 exactly where f0 is even, 0/0, which compares False
        with np.errstate(invalid="ignore", divide="ignore"):
            over = f0 * np.expm1(m * np.log(r)) / (r - 1.0) > 1.0
        hi = np.where(over, r, hi)
        lo = np.where(over, lo, r)
    r = 0.5 * (lo + hi)
    k = np.arange(n)[None, :]
    with np.errstate(invalid="ignore", divide="ignore"):
        x = np.expm1(k * np.log(r)) / np.expm1(m * np.log(r))
    x = np.where(f0 * m >= 1.0 - 1e-12, k / m, x)
    x[:, -1] = 1.0
    return x


def _cell_angles(B):
    """Return the worst corner of each cell of a block (2, ni, nk), folded
    onto 0 to 90 degrees, 90 square and 0 degenerate."""
    p = B.transpose(1, 2, 0)
    c = [p[:-1, :-1], p[1:, :-1], p[1:, 1:], p[:-1, 1:]]
    out = np.full(p[:-1, :-1, 0].shape, 90.0)
    for a in range(4):
        u = c[(a + 1) % 4] - c[a]
        v = c[(a - 1) % 4] - c[a]
        cos = np.sum(u * v, -1) / np.maximum(
            np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1), 1e-300
        )
        ang = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
        out = np.minimum(out, 90.0 - np.abs(90.0 - ang))
    return out


#
# GEOMETRY HELPERS
#

_J = np.array([0.0, 1.0])
"""Unit vector in the direction of increasing `theta` in the conformal plane."""


def _section(blade, spf):
    """Return the two surfaces (4, `N_CHORD`) at `spf`, higher-theta first:
    `x`, `r`, `theta`, and the annulus meridional coordinate."""
    upper, lower = blade.evaluate_section(spf)
    if upper[2].mean() < lower[2].mean():
        upper, lower = lower, upper
    return upper, lower


def _surfaces(blade, cmap, spf, section=None):
    """Return the two surfaces in conformal coordinates, higher-theta first.

    Each (2, n), from the leading edge to its trailing edge corner; of
    `section`, as :func:`_section` returns it, if given. Mapped from where
    the blade put each point along the annulus, not searched for.
    """
    upper, lower = _section(blade, spf) if section is None else section
    return tuple(np.stack((cmap.mp_from_m(s[3]), s[2])) for s in (upper, lower))


def _blade_pieces(blade, cmap, spf, section=None):
    """Return the two surfaces and the trailing edge in conformal coordinates.

    Each surface (2, n) from the leading edge to its trailing edge corner,
    higher-theta first, and the trailing edge (2, n) from the upper corner to
    the lower. On a round trailing edge, the corners are where each surface's
    arc begins, and the trailing edge is the arc.
    """
    upper, lower = _surfaces(blade, cmap, spf, section)
    caps = blade.evaluate_TE_cap(spf)
    if caps is None:
        t = np.linspace(0.0, 1.0, 101)[None]
        return upper, lower, upper[:, -1:] + t * (lower[:, -1:] - upper[:, -1:])
    # Split each surface at its sample nearest where the arc begins; the
    # samples are those `_surfaces` evaluates at.
    m = CHORD_FRACTION
    m_upper, m_lower = caps if blade.suction_is_upper else caps[::-1]
    i_u, i_l = (int(np.argmin(np.abs(m - m_cap))) for m_cap in (m_upper, m_lower))
    te = np.concatenate((upper[:, i_u:], lower[:, i_l:][:, ::-1][:, 1:]), axis=1)
    return upper[:, : i_u + 1], lower[:, : i_l + 1], te


def _at_arc(curve, s):
    """Return points on a polyline (2, n) at arc lengths `s` from its start."""
    return _interp_rows(s, util.cum_arc_length(curve), curve)


def _round_twice(sigma, dens):
    """Return density `dens` at fractions `sigma` of the way round the blade,
    from 0 to 1, repeated from -1 to 2, so it can be read past either end."""
    sigma = np.concatenate((sigma[:-1] - 1.0, sigma[:-1], sigma + 1.0))
    return sigma, np.concatenate((dens[:-1], dens[:-1], dens))


def _interp_rows(x, xp, P):
    """Return each row of `P` (m, n), given at `xp` (n,), at `x`."""
    return np.stack([np.interp(x, xp, c) for c in P])


def _smoothstep(x):
    """Return the cubic smoothstep of `x`, clipped to [0, 1]."""
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _distance(annulus, m0, m1):
    """Return the distance along mid-span from annulus meridional coordinate
    `m0` to `m1`."""
    m = np.linspace(m0, m1, 2001)
    return float(util.cum_arc_length(annulus.evaluate_xr(m, 0.5))[-1])


def _m_at_distance(annulus, m0, m1, dist):
    """Return the annulus meridional coordinate `dist` along mid-span from
    `m0` towards `m1`."""
    m = np.linspace(m0, m1, 2001)
    s = util.cum_arc_length(annulus.evaluate_xr(m, 0.5))
    if dist >= s[-1]:
        raise ValueError(
            f"The end of the blocks round the blade, {dist:.3g} m from it, lies "
            f"beyond the inlet, outlet or mixing plane, {s[-1]:.3g} m from it; "
            "make cut or cut_mixing smaller."
        )
    return float(np.interp(dist, s, m))


def _O_frame(plane, O, *, tune):
    """Return the blade node, and the blade's unit tangent and outward normal
    there, (2, ni) each, at the blade end of each line of O block `O`."""
    wall = plane["wall"]
    n = _wall_normals(wall, _wall_arc(plane, O[:, :, 0]), tune=tune)
    return O[:, :, 0], np.stack((n[1], -n[0])), n


def _wall_frac(topo, X, plane):
    """Return where the O block's blade nodes of plane `X` lie, as fractions
    of the way round its blade from the middle of the trailing edge."""
    wall = plane["wall"]
    return _wall_arc(
        plane, topo.scatter(X)[0][:, :, 0].astype(float)
    ) / util.arc_length(wall)


def _wall_at_frac(topo, X, plane, f):
    """Return plane `X` with the O block's blade nodes put at fractions `f`
    of the way round its blade."""
    wall = plane["wall"]
    S = util.arc_length(wall)
    xy = topo.scatter(X)
    xy[0][:, :, 0] = _at_arc(wall, np.mod(f * S, S)).astype(xy[0].dtype)
    return topo.gather(xy)


def _O_offsets(topo, X, plane, *, tune):
    """Return the offsets (2, ni, nk) of the O block's nodes in plane `X`
    along and normal to the blade from their lines' blade nodes."""
    O = np.asarray(topo.scatter(X)[0], dtype=float)
    b, t, n = _O_frame(plane, O, tune=tune)
    d = O - b[:, :, None]
    return np.stack(
        (np.sum(d * t[:, :, None], axis=0), np.sum(d * n[:, :, None], axis=0))
    )


def _O_from_offsets(topo, X, plane, offsets, *, tune):
    """Return plane `X` with the O block's nodes between its blade and outer
    faces put at `offsets` (2, ni, nk) along and normal to its own blade from
    their lines' blade nodes.

    Each line then moved by where its outer end misses the outer face, in
    proportion to how far along it each node is, so it ends on the outer face
    as `X` has it and leaves the blade as the offsets do: round a nose,
    where the blade's frame turns fast from node to node, offsets carried
    from planes either side can otherwise put the line's last nodes past the
    outer face and fold its outermost cell.

    Then round the blade, along each layer of the O block: where the lines so
    rebuilt shear past one another, cells under `o_carry_sound` or turned
    inside out, the nodes are moved as far from where `X` has them as their
    sound neighbours either side are, interpolated linearly round the blade,
    over the stretch and `o_carry_margin` nodes either side of it. `X`, from
    the carried correction, does not fold there, and the offsets keep the
    rest of the layer as they made it.
    """
    O = np.asarray(topo.scatter(X)[0], dtype=float)
    b, t, n = _O_frame(plane, O, tune=tune)
    new = (
        b[:, :, None]
        + offsets[0][None] * t[:, :, None]
        + offsets[1][None] * n[:, :, None]
    )
    miss = O[:, :, -1] - new[:, :, -1]
    d = np.linalg.norm(np.diff(new, axis=2), axis=0)
    s = np.concatenate((np.zeros((d.shape[0], 1)), np.cumsum(d, axis=1)), axis=1)
    new += (s / np.maximum(s[:, -1:], 1e-300))[None] * miss[:, :, None]

    # Round the blade, where the rebuilt lines shear, each layer moved from
    # `X` as far as its sound nodes either side are
    p = np.moveaxis(new, 0, -1)
    e, f = p[1:, 1:] - p[:-1, :-1], p[:-1, 1:] - p[1:, :-1]
    area = e[..., 0] * f[..., 1] - e[..., 1] * f[..., 0]
    unsound = (_cell_angles(new) < tune.o_carry_sound) | (
        area * np.sign(np.median(area)) <= 0.0
    )
    if unsound.any():
        ni, nk = new.shape[1:]
        bad = np.zeros((ni, nk), dtype=bool)
        for di, dk in ((0, 0), (1, 0), (0, 1), (1, 1)):
            bad[di : ni - 1 + di, dk : nk - 1 + dk] |= unsound
        wide = np.ones(2 * tune.o_carry_margin + 1)
        idx = np.arange(ni)
        move = new - O
        for k in range(1, nk - 1):
            here = np.convolve(bad[:, k], wide, "same") > 0
            if here.any() and (~here).sum() >= 2:
                for c in range(2):
                    move[c, here, k] = np.interp(
                        idx[here], idx[~here], move[c, ~here, k]
                    )
        new = O + move
    g = topo.gid[0][:, 1:-1]
    pts = new[:, :, 1:-1]
    if topo.pitch:
        pts = pts - np.stack(
            (np.zeros_like(pts[0]), topo.shift[0][:, 1:-1] * topo.pitch)
        )
    X = np.array(X)
    X[g.ravel()] = np.moveaxis(pts, 0, -1).reshape(-1, 2).astype(X.dtype)
    return X


def _closed_wall(blade, cmap, section=None):
    """Return the blade as one closed line (2, n), from the middle of the
    trailing edge round the lower side, its surfaces as the section evaluates
    them, and so clustered at its leading edge.

    Also half the trailing edge's length, and the arc length to the leading
    edge where the two surfaces meet, at the start of the camber line they
    hang off; not the furthest point upstream, which on a staggered blade lies
    round the nose on one side.
    """
    upper, lower, te = _blade_pieces(blade, cmap, cmap.spf, section)
    s_te = util.cum_arc_length(te)
    half = 0.5 * s_te[-1]
    wall = np.concatenate(
        (
            _at_arc(te, np.linspace(half, s_te[-1], 201)),
            lower[:, ::-1][:, 1:],
            upper[:, 1:],
            _at_arc(te, np.linspace(0.0, half, 201))[:, 1:],
        ),
        axis=1,
    )
    return wall, half, (s_te[-1] - half) + util.arc_length(lower)


def _wall_normals(wall, s, *, tune):
    """Return the outward unit normals (2, n) of the blade, closed line
    `wall`, at arc lengths `s` along it, as :func:`_outward_normals` makes
    them."""
    S = util.arc_length(wall)
    eps = S / tune.n_wall
    t = _at_arc(wall, np.mod(s + eps, S)) - _at_arc(wall, np.mod(s - eps, S))
    t /= np.linalg.norm(t, axis=0)
    return np.stack((-t[1], t[0]))


def _chord(blade, spf, section=None):
    """Return the meridional chord of the section at `spf`, in conformal
    units, over the mean radius of the section; of `section`, as
    :func:`_section` returns it, if given."""
    # One section for both, as `Blade.evaluate_chord` measures it
    upper, lower = _by_theta(blade, spf) if section is None else section
    r_ref = float(np.mean(upper[1]))
    chord = util.arc_length(np.stack((upper, lower)).mean(axis=0)[:2])
    return chord / r_ref


def _norm_arc(curve):
    """Return normalised arc length along a polyline (2, n)."""
    d = np.diff(curve, axis=1)
    s = np.empty(curve.shape[1])
    s[0] = 0.0
    np.cumsum(np.sqrt(d[0] * d[0] + d[1] * d[1]), out=s[1:])
    s /= s[-1]
    return s


def _outward_normals(loop, *, tune):
    """Return smoothed unit normals out of a closed clockwise loop (2, n)."""
    pts = loop[:, :-1]
    t = np.roll(pts, -1, axis=1) - np.roll(pts, 1, axis=1)
    t /= np.linalg.norm(t, axis=0)
    n = np.stack((-t[1], t[0]))
    for _ in range(tune.n_smooth_normal):
        n = 0.25 * np.roll(n, 1, axis=1) + 0.5 * n + 0.25 * np.roll(n, -1, axis=1)
        n /= np.linalg.norm(n, axis=0)
    return np.concatenate((n, n[:, :1]), axis=1)


def _periodic_line(blade, cmap, pitch, *, tune):
    """Return `theta(m')` of the initial periodic line above the blade.

    Half a pitch above the camber line, extended straight beyond it at the
    end slopes of the camber line, limited to `slope_max_le` ahead and
    `slope_max` behind.
    """
    # The camber line along the chord and over each end evaluated together:
    # evaluated alone, an end would be stretched over the whole chord.
    m_line = util.cluster_cosine(400)
    m_ends = (
        np.linspace(0.0, tune.end_fit, tune.n_end_fit),
        np.linspace(1.0 - tune.end_fit, 1.0, tune.n_end_fit),
    )
    m_all, inverse = np.unique(np.concatenate((m_line, *m_ends)), return_inverse=True)
    cam = blade.evaluate_camber(cmap.spf, m=m_all)
    mp_all = cmap.mp_from_xr(cam[:2])
    i_line, i_LE, i_TE = np.split(inverse, np.cumsum([len(m_line), tune.n_end_fit]))
    mp, theta = mp_all[i_line], cam[2][i_line] + 0.5 * pitch
    i_TE = i_TE[::-1]
    slope = (
        np.clip(
            _end_slope(mp_all[i_LE], cam[2][i_LE]),
            -tune.slope_max_le,
            tune.slope_max_le,
        ),
        np.clip(
            _end_slope(mp_all[i_TE], cam[2][i_TE]), -tune.slope_max, tune.slope_max
        ),
    )

    def periodic(mq):
        mq = np.asarray(mq, dtype=float)
        out = np.interp(mq, mp, theta)
        out = np.where(mq < mp[0], theta[0] + slope[0] * (mq - mp[0]), out)
        out = np.where(mq > mp[-1], theta[-1] + slope[1] * (mq - mp[-1]), out)
        return out

    return periodic


def _end_slope(mp, theta):
    """Return `dtheta/dm'` of the camber line at the end its samples `mp`,
    `theta` over the last part of chord start from.

    From a quadratic fitted over them, not differenced between the
    end's first two samples: an annulus whose `m` runs on unevenly over a
    small fraction of chord puts those a few percent out of true in `m'`, by
    an amount that jumps from span to span. Extended a chord beyond the blade,
    that shears the periodic surface by a degree where the planes crowd the
    endwalls.
    """
    return np.polyfit(mp - mp[0], theta, 2)[1]


def _line(a, b, f):
    """Return points (2, n) on the straight line from `a` to `b` at fractions `f`."""
    pts = a[:, None] + f[None, :] * (b - a)[:, None]
    pts[:, 0], pts[:, -1] = a, b
    return pts


def _tfi(bottom, top, left, right):
    """Return a block (2, ni, nk) by transfinite interpolation of its edges.

    `bottom` and `top` run along `i`, `left` and `right` along `k`, and the
    corners must agree. Blended on normalised arc length rather than index, so
    a clustered edge pulls the interior with it.
    """
    c00, c10, c01, c11 = bottom[:, 0], bottom[:, -1], top[:, 0], top[:, -1]
    for a, b in (
        (c00, left[:, 0]),
        (c10, right[:, 0]),
        (c01, left[:, -1]),
        (c11, right[:, -1]),
    ):
        assert np.abs(a - b).max() <= 1e-12, f"TFI corners disagree {a} {b}"

    sb = _norm_arc(bottom)[:, None]
    st = _norm_arc(top)[:, None]
    tl = _norm_arc(left)[None, :]
    tr = _norm_arc(right)[None, :]
    dst = st - sb
    u = (sb + dst * tl) / (1.0 - dst * (tr - tl))
    v = (tr - tl) * u
    v += tl

    # The blend of the four edges less the bilinear one of the corners,
    # gathered by what each multiplies: `v` varies along `k`, `u` along `i`.
    Q = (top - bottom - (c01 - c00)[:, None])[:, :, None]
    R = (right - left - (c10 - c00)[:, None])[:, None, :]
    K = (c11 - c10 - c01 + c00)[:, None, None]
    xy = v * Q
    xy += (bottom - c00[:, None])[:, :, None]
    xy += left[:, None, :]
    xy += u * (R - v * K)

    # Edges exactly as given, so shared edges are bit-identical between blocks.
    xy[:, :, 0], xy[:, :, -1] = bottom, top
    xy[:, 0, :], xy[:, -1, :] = left, right
    return xy


def _mult(x, mult):
    """Return `x` resampled up to a number of intervals divisible by `mult`."""
    n_cell = len(x) - 1
    return util.resample_to(x, n_cell + (-n_cell) % mult + 1)


def _radius_metric(cmap, mp_a, mp_b, r_free, n=401):
    """Lengths from `mp_a` to `mp_b` on the surface of `cmap`, in conformal
    units at radius `r_free`: `r / r_free` times the conformal length.

    Returns the ratio of that length to the conformal one, a function taking
    fractions of it from `mp_a` to the fractions of the conformal length they
    land at, and `r / r_free` at `mp_a`. Either way round, `mp_b` upstream of
    `mp_a` or downstream.
    """
    mp = np.linspace(mp_a, mp_b, n)
    w = cmap.xr_from_mp(mp)[1] / r_free
    d = np.abs(np.diff(mp))
    q = np.concatenate(([0.0], np.cumsum(0.5 * (w[1:] + w[:-1]) * d)))
    p = np.concatenate(([0.0], np.cumsum(d)))
    if not p[-1] > 0.0:
        return 1.0, lambda f: np.asarray(f), float(w[0])

    def to_mp(f):
        out = np.interp(f, q / q[-1], p / p[-1])
        out[0], out[-1] = 0.0, 1.0
        return out

    return float(q[-1] / p[-1]), to_mp, float(w[0])


def _uniform(d):
    """Uniform points on the unit interval at no more than spacing `d`."""
    return np.linspace(0.0, 1.0, max(int(np.ceil(1.0 / d)), 1) + 1)


def _single(dmin, dmax, ER):
    """Single-sided clustering on the unit interval, coping with a short one."""
    if dmin * ER >= dmax or dmin >= 0.5:
        return _uniform(min(dmin, dmax))
    try:
        return clusterfunc.single.free(dmin, dmax, ER)
    except clusterfunc.exceptions.ClusteringException:
        pass
    try:
        return clusterfunc.single.free(dmin, dmax, ER, mult=1)
    except clusterfunc.exceptions.ClusteringException:
        # Too short to grow at this ratio in the fewest points it takes:
        # evenly, at the first spacing.
        return _uniform(dmin)


def _double(d0, d1, dmax, ER):
    """Double-sided clustering on the unit interval, coping with a short one."""
    if min(d0, d1) * ER >= dmax or min(d0, d1) >= 0.5:
        return _uniform(min(d0, d1, dmax))
    dmax = max(dmax, d0, d1)
    try:
        return clusterfunc.double.free(d0, d1, dmax, ER)
    except clusterfunc.exceptions.ClusteringException:
        pass
    try:
        return clusterfunc.double.free(d0, d1, dmax, ER, mult=1)
    except clusterfunc.exceptions.ClusteringException:
        # Too short to grow at this ratio: the end spacings, whatever the
        # growth between them.
        n = max(int(np.ceil(2.0 / (d0 + d1))), 2) + 1
        return clusterfunc.double.fixed(d0, d1, n)


def _stitch_mixing_planes(rows, labels=LABELS):
    """Force xr coordinates to match exactly at mixing planes."""
    for i_row in range(len(rows) - 1):
        dn = rows[i_row][labels.index("dn")]
        up = rows[i_row + 1][labels.index("up")]
        x = 0.5 * (dn.x[-1, :, 0] + up.x[0, :, 0])[:, None]
        r = 0.5 * (dn.r[-1, :, 0] + up.r[0, :, 0])[:, None]
        dn[-1].set_x(x)
        up[0].set_x(x)
        dn[-1].set_r(r)
        up[0].set_r(r)
