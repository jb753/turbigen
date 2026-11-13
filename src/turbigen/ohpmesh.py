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
its lines normal to the camber line, meeting the O block node for node along
one side and the rear and by non-matching interfaces along the front and the
other side. See :meth:`OHPassage._tip_gap`.
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

logger = logging.getLogger("turbigen")

LABELS = ("O", "P", "LE", "Pin", "TE", "Pout", "up", "dn")
"""Block names, in the order blocks are held."""

N_MERID = 4001
"""Samples along a span line from which its conformal coordinate is integrated."""

N_SMOOTH_NORMAL = 10
"""Passes of a three-point average over the blade normals the O block is
extruded along, so they fan round corners rather than cross."""

SLOPE_MAX = 1.0
"""Largest `dtheta/dm'` the initial periodic line is extended at behind the
blade, being the tangent of 45 degrees. Only a starting guess: the line floats."""

SLOPE_MAX_LE = 0.5
"""Largest `dtheta/dm'` the initial periodic line is extended at ahead of the
blade, about 27 degrees: at a steeper inlet metal angle, it skews the blocks
there more than the sweeps undo."""

END_FIT = 0.05
"""Fraction of chord at each end of the camber line its end slope is fitted
over, to average out an uneven parameterisation of the annulus."""

N_END_FIT = 101
"""Samples, evenly spaced in chord, over which each end slope is fitted."""

POLISH_POWER = 4.0
"""Power the polish raises each cell corner's shape measure to, so that the
worst cells dominate rather than being traded for small gains elsewhere."""

N_MATCH_SPAN = 3
"""Corrections of the main span's end spacing to meet a tip gap's."""

N_MATCH_REAR = 4
"""Corrections of the blade's spacing at the corners behind it, to match
the cells across the passage there."""

N_WALL = 5001
"""Samples along the blade, which nodes on it are placed on."""

CORNER_COS, CORNER_SIN = np.cos(np.pi / 3.0), np.sin(np.pi / 3.0)
"""Direction of an edge out of a corner of the O block, against the O block's
face and its outward normal there."""

INTERFACES = (
    (2, "k0", 3, "k1"),
    (2, "k1", 3, "k0"),
    (3, "i1", 1, "i0"),
    (1, "i1", 5, "i0"),
    (5, "k0", 4, "k1"),
    (5, "k1", 4, "k0"),
)
"""The faces shared round the passage, (block, face, block, face), the blocks
in the order O, P, LE, Pin, TE, Pout: LE and Pin, and Pout and TE, meet on both
sides, a pitch apart on one."""

EVEN_RATIO = 1.4
"""Most a spoke's first cell may differ from the geometric mean of those round
its singular point, either way, after :func:`_even_corners`."""

CORNER_MID = 0.1
"""How far round the blade from either edge a corner must be to count as part
way along it, for :attr:`OHPassage.cor_fac_mid`; the leading edge taken half way
round."""

MATCH_EASE = 4
"""Lines from each end of a face over which :func:`_match_interfaces` eases from
holding the block's edge to its factor."""

MATCH_RATIO = 1.2
"""Most one cell may be of the next across an interface after
:func:`_match_interfaces`."""

CORNER_REF_SPANS = 21
"""Spans, evenly from hub to casing, across which `corner_ref="span"` finds the
corners of the O block."""

N_CORNER_SPANS = 51
"""Spans across which :attr:`OHPassage.corner_cap` finds the corners by the
rules, evenly in span fraction from hub to casing."""

CORNER_SMOOTH = 0.0
"""Width, in span fraction, of the Gaussian the corners found across the
span are smoothed with; none if zero."""

CORNER_ANCHOR = 1.0
"""Step in span fraction between the anchors the corners are interpolated
between, `spf_corners` one too: one, the hub, `spf_corners` and the
casing."""

POLISH_COARSE_RINGS = 2
"""How many cells out from each poor cell of a swept plane are polished."""

HOLD_COARSE = 1
"""Layers of cells off the blade the polish of a swept plane leaves be."""

POLISH_FINE_RINGS = 3
"""How many cells out from each poor cell of the refined plane are polished."""

HOLD_FINE = 3
"""Layers of cells off the blade the polish of the refined plane leaves be."""

DN_SMOOTH_SPF = 0.02
"""Width, in span fraction, of the Gaussian the spacing along each plane's
block to the mixing plane is smoothed over across span, so neighbouring
planes' stations agree. See :meth:`OHPassage.mesh_row`."""

MIX_BLEND = 8
"""Cells along each line from a mixing plane over which
:attr:`OHPassage.straight_mixing` blends its end cell to the common one. See
:func:`_straighten_mixing`."""

TIP_O_TURN = 0.5
"""Most a tip gap's O line is squared to the outline, `tip_O_square`, times
its length and the outline's curvature there, so that where the outline
turns fast, round the nose, neighbouring lines bent towards its normals do
not cross. See :func:`_gap_O_lines`."""

TIP_O_EASE = 8
"""Nodes round the blade the squaring of a tip gap's O lines is eased over,
so neighbouring lines bend alike."""

O_CARRY_SOUND = 30.0
"""Worst corner of a carried plane's O block cell, rebuilt from offsets,
under which its nodes are taken from their neighbours round the blade
instead [deg]. See :func:`_O_from_offsets`."""

O_CARRY_MARGIN = 2
"""Nodes either side of each such stretch taken from their neighbours too."""

O_MATCH_SMOOTH = 8.0
"""Lines round the blade over which `geometric_O_match` smooths how far the
O block's outer cells move, the deviation of a Gaussian."""

DTYPE = np.float32
"""Precision the planes are held, swept, refined and polished in: the
finished grid is stored in single precision, and the sweeps run faster in
it. Only their geometry is worked out in double before."""

EXTRUDE_GAP = 0.3
"""Most a line of an extruded inlet or outlet block departs from the turn of
the pitch-mean line, as a fraction of the gap to its nearer neighbour."""

N_ENDS_MIN = 8
"""Least coarse cells along the blocks ahead of and behind the blade."""

COARSE_FACTOR = 2.0
"""How many times finer than `ds_free` the coarse mesh is everywhere, which
leaves the refined mesh's cell sizes as they are."""

ER_COARSE = 1.3
"""Expansion ratio along the blade of the coarse mesh, away from where
`turn_max` clusters it round the edges."""

COARSE_ENDS_FACTOR = 1.5
"""How many times more coarse cells than the rest of the coarse mesh the
blocks ahead of and behind the blade have along them, at least `N_ENDS_MIN`."""

ORTH = 1.0
"""How far, as a fraction of the way, each sweep draws the coarse nodes one
cell off the blade towards the blade's normal at their blade node, at the
distance they lie. Sliding the blade nodes squares the grid there only where
the nodes off the blade lie over distinct points of it; round an edge much
smaller than a coarse cell, this squares it. See
:func:`ember.elliptic.winslow_sweeps`."""

SLIDE_MID = 1.0
"""How far each sweep slides the blade nodes midway between the leading and
trailing edges towards the foot of the perpendicular from the node outside
them, as a fraction of the way; none at the edges, so their clustering stays
put. See :meth:`OHPassage._slide_factors`."""

TURN_FINE = 10.0
"""Most the blade turns across one refined cell along it, degrees: an edge
whose radius is small against `ds_LE` or `ds_TE` is clustered further, to
resolve it."""

ENDS_FACTOR = 2
"""How many times more refined cells than `ds` and `ER_stream` give the
blocks ahead of and behind the blade have along them, each split evenly."""

N_SMOOTH_TURN = 5
"""Samples of the blade, of `N_WALL`, its turning is averaged over to set
the coarse spacing round its edges: few, so the curvature of an edge much
smaller than a cell is not averaged away."""

N_SPLINE_O = 16
"""Samples per coarse cell of the spline along each line of the O block that
its refined nodes are respaced along."""

MIN_GAP = 0.1
"""Least fraction of its length as built a gap between blade nodes keeps as
they slide: only a guard against a run of them collapsing, the crowding the
sliding wants at the trailing edge being well above it."""


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
        m = np.linspace(m_lo, m_hi, N_MERID)
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


GAP_CAP = 0.5
"""Most the O block of a tip gap reaches into the blade, as a fraction of the
radius of the largest circle inside it touching it at a node: the two sides'
inner faces then stay apart, and inside the radius of curvature, so the
normals do not cross before them."""

GAP_BOW_DECAY = 2.0
"""Length over which each end face's bow eases out along a tip gap's H
block, in widths of that face."""

N_GAP_SCAN = 2001
"""Points along the camber line at which the width of a tip gap's inner face
is measured, to place the ends of its H block."""


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
    extrude_in: bool = False
    extrude_out: bool = False
    """Whether the inlet block ahead of the first row, and the outlet block
    behind the last, are extruded from a cut rather than swept; if so, they
    are one coarse cell long beyond the cut through the sweeps."""
    corner_anchors: tuple = ()
    """With :attr:`OHPassage.corner_cap`, (spf, corners) pairs in rising
    span fraction that the corners at each span are interpolated between;
    empty, `corners` at every span."""
    symmetric: bool = False
    """Whether the blade nodes are alike on both halves of the blade, a row
    with a tip gap under :attr:`OHPassage.tip_match` 3."""

    def corners_at(self, spf):
        """Where the corners C3, C4, C1, C2 lie at span fraction `spf`, as
        fractions of the way round the blade from the middle of the
        trailing edge: piecewise linear between `corner_anchors`, else
        `corners`."""
        return _corners_at(self.corners, self.corner_anchors, spf)

    def i_corners_coarse(self):
        """O block `i` of the corners C3, C4, C1, C2 on the coarse mesh."""
        ends = np.cumsum((0, *self.n_wall))
        return tuple(int(ends[i]) for i in self.i_wall)

    def i_corners(self):
        """O block `i` of the corners C3, C4, C1, C2 once refined."""
        return tuple(int(i) for i in np.cumsum(self.nf_wall[:4]))


class OHPassage(Mesher):
    """Generate a mesh with an O block round each blade and a passage block
    between neighbouring O blocks, smoothed coarse and refined where needed."""

    type: ClassVar[str] = "ohp"

    labels: ClassVar[tuple] = LABELS
    """Names of the blocks of a row, in the order :meth:`plane` returns them."""

    meshes_sweep: ClassVar[bool] = True

    dspf_mid: float = 0.03
    """Spanwise grid spacing at midspan, as a fraction of span."""

    ER_span: float = 1.2
    """Expansion ratio of spanwise grid away from hub and casing."""

    resolution_factor: float = 1.0
    """Multiply the number of points in each direction, keeping relative spacings."""

    n_span_smooth: int = 17
    """Number of spanwise planes to smooth, hub and casing included. The rest
    are interpolated between them. Zero smooths every plane."""

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
    coarse mesh that is smoothed is `COARSE_FACTOR` times finer."""

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

    n_sweep: int = 2000
    """Smoothing sweeps on the coarse mesh."""

    orth_taper: float = 8.0
    """With it, the sweeps draw square to the blade not only the first line
    of the O block off it, by `ORTH`, but every line `k` out after, from the
    line inside it, by `ORTH` times (1 - (k - 1) / (K - 1)) to this power,
    `K` the O block's edge, which is left alone. Only the first if zero."""

    t_O: float = 0.1
    """Thickness of the O block, as a fraction of pitch, before smoothing."""

    cut_mixing: float = 0.3
    """Distance from a blade's edge on the side of a mixing plane at which
    the block ahead of or behind it ends, in meridional chords of that row;
    the block beyond runs on to the mixing plane. Held to at most
    `cut_mixing_frac` of the way there where the gap is too short for it."""

    cut_mixing_frac: float = 0.5
    """The furthest the block ahead of or behind a blade ends towards a
    mixing plane, as a fraction of the way from the blade's edge to it."""

    cut: float = 0.3
    """Distance from the leading edge of the first row and the trailing edge
    of the last at which the blocks ahead of and behind the blade end, and
    beyond which the inlet and outlet blocks are extruded rather than swept,
    in meridional chords."""

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
    casing, as fractions of the way round the blade; "span", midway between
    the least and greatest of those found every 0.05 of span, each logged."""

    corner_cap: float = 0.0
    """With it, the corners move with span: every `CORNER_ANCHOR` of span,
    from where they are found at `spf_corners` towards where the same rules
    find them there, smoothed in span over `CORNER_SMOOTH`, as far as leaves
    every piece of the blade between them within this factor of its length
    at `spf_corners`; and piecewise linear in span fraction between. Every
    piece's length is then within the factor at every span, so the counts
    along it, the same at every span, still fit. Zero, or one, the corners
    found at `spf_corners` kept at every span."""

    stagger_max: float = 30.0
    """Most the bounding box that places the corners is turned, degrees."""

    cor_fac: float = 0.6
    """How hard the nodes near each corner are drawn in towards it each
    sweep: lower draws them in harder."""

    cor_fac_mid: float | None = 0.4
    """`cor_fac` of the first row's corners part way along the blade, more
    than `CORNER_MID` of the way round it from both edges, rather than by one
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

    carry_below: float = 0.0
    """A plane carried between control planes whose worst cell corner, as
    carried and before it is refined, is under this [deg] is swept as a
    control plane itself, the worst between each pair of control planes at a
    time, until none is. Off if zero."""

    njtip_min: int = 9
    """Least number of spanwise nodes across a tip gap, one more than a
    multiple of eight."""

    nk_tip_O: int = 17
    """Nodes across the O block of a tip gap, from the H block to the outline
    of the blade, one more than a multiple of eight."""

    t_tip_O: float = 0.2
    """Thickness of the O block of a tip gap, as a fraction of the blade's
    greatest thickness on the section at its tip, capped at `GAP_CAP` of the
    radius of the largest circle inside the blade touching it at each node."""

    tip_match: int = 3
    """Side of a tip gap's H block, 1 the lower or 2 the upper, whose lines
    start from the O block's inner nodes, meeting it node for node, the other
    side meeting it by a non-matching interface; or 3, both (1 | 2), node for
    node all round. With 3, a row with a tip gap has its blade nodes alike on
    both halves of the blade, from the middle of the trailing edge to the
    leading edge, coarse and refined, by the greater of the two sides'
    densities, so node `h - k` lies across from node `h + k`; held where they
    are put through the sweeps from `tip_match_from` up; and the H block's
    corners paired across the blade. The coarse blade nodes are then spaced
    at half the refined density, and each coarse cell split in two. See
    :func:`_symmetric_wall`, :meth:`OHPassage._doubled_wall` and
    :func:`_gap_corners`."""

    tip_match_from: float = 0.7
    """With `tip_match` 3, the span fraction from which the blade nodes are
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

    tip_front_lean: float = 30.0
    """Most the O block's lines from a tip gap's front corners B and C lean
    off the blade's normal, degrees, so that the corners slide along the
    inner face towards each other's place along the camber, and the H
    block's front runs more nearly across it. None if zero."""

    tip_bow_front: float = 0.1
    """How far the front of a tip gap's H block bows forward at its middle,
    normal to the line between its front corners, as a fraction of the way
    to the O block's inner face round the nose."""

    n_sweep_tip: int = 0
    """Point-Jacobi sweeps of Winslow's equations evening out the inside of
    each of a tip gap's blocks on each plane, their faces fixed, the O
    block's lines respaced to the wall spacing after; 0 for none."""

    even_corners: bool = True
    """Even the coarse cells round each singular point of a swept plane,
    where five grid lines meet: every one of them within `EVEN_RATIO` of the
    spokes' geometric mean, blended away over `EVEN_O_CELLS` along each
    block edge, and into each block's interior. See :func:`_even_corners`."""

    cluster_P: bool = True
    """Respace the swept coarse passage block along its mid-pitch line by
    double-sided clustering, its end cells kept as they are, blended to no
    change at its pitchwise boundaries. See :func:`_cluster_P`."""

    match_interfaces: bool = True
    """Keep the coarse spacing close to continuous across the interfaces
    between the blocks round the passage, LE to Pin, Pin to P, P to Pout and
    Pout to TE: each line crossing one has its cells either side brought
    within `MATCH_RATIO` of each other, blended away along the line. See
    :func:`_match_interfaces`."""

    straight_mixing: bool = True
    """Keep the grid lines straight across each mixing plane: the lines
    either side of it respaced so their cell at the plane is the same length
    in (x, r) across the pitch and on both sides, and level in theta, blended
    away over `MIX_BLEND` cells, then each node on the plane put midway
    between its neighbours either side, endwalls included. See
    :func:`_straighten_mixing`."""

    theta_even: bool = True
    """Blend the swept coarse blocks ahead of and behind the blade towards
    nodes even in theta across the pitch, fully where they end if the skew
    allows, none where they meet the blade's O block and the passage block.
    See :func:`_theta_even`."""

    theta_skew_min: float = 45.0
    """Least cell corner angle the blend of `theta_even` may take a cell to,
    unless it was already less, when it may not be made worse [deg]."""

    dw_max: float = 0.15
    """Most the blend of `theta_even` grows from one station to the next
    towards where the blocks end, so the shear is spread along them."""

    theta_lean: float = 10.0
    """With it, the blend of `theta_even` found on every control plane
    first, and each station's weight then held, plane to plane up the span,
    to no more than leans the grid lines between neighbouring control planes
    this much further, degrees, by how far a node moves with the weight; the
    planes blended again under those. Zero blends each plane alone."""

    cluster_O: bool = False
    """Respace the swept coarse O block along each line off the blade by
    double-sided clustering, its end cells kept as they are. See
    :func:`_cluster_O`."""

    geometric_O: bool = True
    """Respace the refined O block's lines off the blade by geometric growth
    from the wall spacing, the same expansion ratio all along each line,
    rather than double-sided stretching to the first cell beyond the O
    block: neighbouring lines of different lengths then keep their nodes at
    like fractions along them, at the cost of a step in cell size at the O
    block's edge."""

    geometric_O_match: float = 3.0
    """With `geometric_O`, how far each line's outer cell may move from the
    geometric towards the first cell beyond the O block: within this factor
    of it either way, smoothed round the blade over `O_MATCH_SMOOTH` lines,
    the line then double-sided stretched between the two. One, purely
    geometric."""

    tip_rear_search: bool = True
    """Place a tip gap's rear corners by a search instead of `tip_end_rear`:
    the pair a multiple of `tip_rear_multiple` cells round the tail apart
    whose line through their own nodes is nearest normal to the camber, so
    that both leave the outline square, rather than one moved onto a line
    normal to the camber through the other; and the front corners as many
    cells round the nose, so the H block's front meets the O block node for
    node. See :func:`_gap_rear_search`."""

    tip_rear_multiple: int = 4
    """With `tip_rear_search`, what the cells round a tip gap's tail, the H
    block's count across, are a multiple of."""

    tip_rear_lean: float = 30.0
    """With `tip_rear_search`, most the O block's lines from a tip gap's
    rear corners A and D lean off the blade's normal, degrees, so that each
    corner slides along the inner face to where its line bisects the turn
    between the side and the rear line. None if zero."""

    tip_rear_max: int = 37
    """With `tip_rear_search`, the most nodes round a tip gap's tail, from
    A through the cut to D, the H block's count of nodes across. No cap if
    zero."""

    tip_blend: float = 0.05
    """Length, as a fraction of the camber line, over which the spacing
    across a tip gap's H block eases from each end face's to even, being
    even in between. 0 blends from the front's to the rear's along the whole
    block instead."""

    def __post_init__(self):
        if min(self.ds, self.ds_LE, self.ds_TE, self.ds_free, self.t_O) <= 0.0:
            raise ValueError("ds, ds_LE, ds_TE, ds_free and t_O must be positive.")
        if not max(self.ds_LE, self.ds_TE) <= self.ds <= self.ds_free:
            raise ValueError(
                f"Need ds_LE and ds_TE <= ds <= ds_free, got {self.ds_LE}, "
                f"{self.ds_TE}, {self.ds}, {self.ds_free}."
            )
        if min(self.cut, self.cut_mixing) <= 0.0:
            raise ValueError(
                f"cut={self.cut} and cut_mixing={self.cut_mixing} must be positive."
            )
        if not 0.0 <= self.spf_corners <= 1.0:
            raise ValueError(f"spf_corners={self.spf_corners} must be from 0 to 1.")
        if self.corner_ref not in ("spf", "ends", "span"):
            raise ValueError(
                f"corner_ref={self.corner_ref!r} must be 'spf', 'ends' or 'span'."
            )
        if self.corner_cap and self.corner_cap < 1.0:
            raise ValueError(
                f"corner_cap={self.corner_cap} must be zero or at least one."
            )
        if not 0.0 <= self.polish_below < 90.0:
            raise ValueError(
                f"polish_below={self.polish_below} must be from zero to 90 degrees."
            )
        for name in ("njtip_min", "nk_tip_O"):
            n = getattr(self, name)
            if n < 9 or (n - 1) % 8:
                raise ValueError(
                    f"{name}={n} is not a count a multigrid level can halve; "
                    f"give it nine or more, one above a multiple of eight."
                )
        if not 0.0 < self.t_tip_O < 0.5:
            raise ValueError(f"t_tip_O={self.t_tip_O} must be between zero and 0.5.")
        if self.tip_match not in (1, 2, 3):
            raise ValueError(f"tip_match={self.tip_match} must be 1, 2 or 3.")
        if not 0.0 <= self.tip_match_from < 1.0:
            raise ValueError(f"tip_match_from={self.tip_match_from} must be in [0, 1).")
        if self.tip_match == 3 and not self.tip_rear_search:
            raise ValueError("tip_match=3 needs tip_rear_search.")
        if min(self.tip_end, self.tip_end_rear) <= 0.0:
            raise ValueError(
                f"tip_end={self.tip_end} and tip_end_rear={self.tip_end_rear} "
                "must be positive."
            )
        if self.n_sweep_tip < 0:
            raise ValueError(f"n_sweep_tip={self.n_sweep_tip} must be zero or more.")
        for name in ("tip_nose_outline", "tip_tail_outline"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name}={getattr(self, name)} must be in [0, 1].")
        if self.tip_blend < 0.0:
            raise ValueError(f"tip_blend={self.tip_blend} must not be negative.")

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

        self._check_inviscid_ducts(machine.annulus)
        spf, jtip = self._span_grid(machine, spacing)

        blocks = []
        for i_row, row in enumerate(machine.rows):
            j = jtip if row.tip_gap else None
            blocks.append(self.mesh_row(machine, i_row, spacing, spf, j))

        _stitch_mixing_planes(
            [row[: len(self.labels)] for row in blocks],
            self.labels,
            self.straight_mixing,
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
                    dspf_hub, dspf_casing, self.dspf_mid, self.ER_span
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
                dspf_tip, dspf_tip, dspf_even, self.ER_span, L_main, 1.0
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
        for _ in range(N_MATCH_SPAN):
            spf_main = clusterfunc.double.free(
                dspf_hub, d_main, self.dspf_mid, self.ER_span, 0.0, L_main
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
        n = self.n_span_smooth
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

        The blade as one closed line (2, N_WALL), evenly spaced in arc length,
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
        s = np.linspace(0.0, S, N_WALL)
        wall = _at_arc(wall, s)
        if corners is not None:
            marks = np.array([0.0, *(np.asarray(corners) * S), S])
            return wall, marks, (1, 2, 3, 4), s_LE

        # The outline of the O block, turned by the stagger about the leading
        # edge; its corners are where it comes nearest those of its box.
        outline = wall + t_O * _outward_normals(wall)
        # The box turned about the furthest point upstream, which sets the
        # corners as they were tuned; the leading edge proper, `s_LE`, sets
        # the spacing along the blade and the tip gap's sides.
        i_LE = int(np.argmin(wall[0]))
        s_up = s[i_LE]
        chord = wall[:, 0] - wall[:, i_LE]
        stag = np.clip(
            np.arctan2(chord[1], chord[0]),
            -np.radians(self.stagger_max),
            np.radians(self.stagger_max),
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
        normals = _outward_normals(wall)

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

        corners = [s_C[k] for k in ("C3", "C4", "C1", "C2")]
        return wall, np.array([0.0, *corners, S]), (1, 2, 3, 4), s_LE

    def _geometry(self, blade_wall, t_O, f_wall):
        """Return the blade, its coarse nodes, and the O block outline.

        The blade and its corners as :meth:`_wall` returns them, and the
        nodes at fractions `f_wall` of each piece of it between them, the
        same at every span.
        """
        wall, marks, i_C, s_LE = blade_wall
        s = np.linspace(0.0, marks[-1], N_WALL)
        n_wall = [len(f) - 1 for f in f_wall]
        s_loop = np.concatenate(
            [
                (a + f * (b - a))[(0 if j == 0 else 1) :]
                for j, (a, b, f) in enumerate(zip(marks[:-1], marks[1:], f_wall))
            ]
        )
        loop = np.stack([np.interp(s_loop, s, c) for c in wall])
        loop[:, -1] = loop[:, 0]
        outer = loop + t_O * _outward_normals(loop)
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
                self.cut_mixing * chord,
                self.cut_mixing_frac * _distance(machine.annulus, m0, m1),
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
        t_O = self.t_O * pitch

        # Coarse along the blade: the free stream size, or less where it
        # turns sharply; the two sides of the passage block alike.
        # The corners found at spf_corners, and kept as fractions of the way
        # round at every span: found at each, the nearest point to a corner of
        # the box can jump from one part of the blade to another between
        # spans, and the planes either side no longer fit together.
        if self.corner_ref == "ends":
            ends = []
            for x in (0.0, 1.0):
                cmap_C = ConformalMap(machine.annulus, x, m_in, m_out)
                _, marks_C, _, _ = self._wall(blade, cmap_C, t_O, pitch=pitch)
                ends.append(marks_C[1:5] / marks_C[-1])
            corners = tuple(float(x) for x in np.mean(ends, axis=0))
        elif self.corner_ref == "span":
            found = []
            for x in np.linspace(0.0, 1.0, CORNER_REF_SPANS):
                cmap_C = ConformalMap(machine.annulus, x, m_in, m_out)
                _, marks_C, _, _ = self._wall(blade, cmap_C, t_O, pitch=pitch)
                found.append(marks_C[1:5] / marks_C[-1])
                logger.info(
                    f"Row {i_row} corners found at spf={x:.2f}: {np.round(found[-1], 4)}"
                )
            corners = tuple(
                float(x) for x in 0.5 * (np.min(found, axis=0) + np.max(found, axis=0))
            )
            logger.info(f"Row {i_row} corners, midway: {np.round(corners, 4)}")
        else:
            cmap_C = ConformalMap(machine.annulus, self.spf_corners, m_in, m_out)
            _, marks_C, _, _ = self._wall(blade, cmap_C, t_O, pitch=pitch)
            corners = tuple(float(x) for x in marks_C[1:5] / marks_C[-1])
        anchors = ()
        if self.corner_cap > 1.0:
            # The corners as the rules find them across the span, smoothed
            # in span fraction, which turns where the rules jump from one
            # part of the blade to another into a ramp; then every
            # CORNER_ANCHOR of span, each moved from those at spf_corners
            # only as far as the cap allows, those at spf_corners held
            x_found = np.linspace(0.0, 1.0, N_CORNER_SPANS)
            found = []
            for x in x_found:
                cmap_x = ConformalMap(machine.annulus, x, m_in, m_out)
                _, marks_x, _, _ = self._wall(blade, cmap_x, t_O, pitch=pitch)
                found.append(marks_x[1:5] / marks_x[-1])
            smooth = np.array(found)
            if CORNER_SMOOTH:
                w = np.exp(-0.5 * ((x_found[:, None] - x_found) / CORNER_SMOOTH) ** 2)
                smooth = (w @ smooth) / w.sum(axis=1)[:, None]
            x_anchor = np.linspace(0.0, 1.0, round(1.0 / CORNER_ANCHOR) + 1)
            anchors = {
                float(x): _capped_corners(
                    corners,
                    [np.interp(x, x_found, smooth[:, k]) for k in range(4)],
                    self.corner_cap,
                )
                for x in x_anchor
                if abs(x - self.spf_corners) > 1e-9
            }
            anchors[self.spf_corners] = corners
            anchors = tuple(sorted(anchors.items()))
            logger.debug(f"Row {i_row} corners by span: {anchors}")
        blade_wall = self._wall(blade, cmap, t_O, _corners_at(corners, anchors, 0.5))
        wall, marks, i_C, s_LE = blade_wall
        S = marks[-1]
        s = np.linspace(0.0, S, N_WALL)
        # The coarse mesh, uniformly finer than the free stream.
        Hc = H / COARSE_FACTOR
        density = _coarse_density(wall, Hc, np.radians(self.turn_max), ER_COARSE)
        i3, _, i1, _ = i_C
        n = [max(1, round(_integral(s, density, a, b))) for a, b in _pairs(marks)]
        n[i3] = n[i1] = max(n[i3], n[i1])
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
        grow = self.ER_surf - 1.0
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
        periodic = _periodic_line(blade, cmap, pitch)
        d = 1e-6 * (mp_out - mp_in)
        stretch_in = np.hypot(1.0, (periodic(mp_Zin) - periodic(mp_Zin - d)) / d)
        stretch_out = np.hypot(1.0, (periodic(mp_Zout + d) - periodic(mp_Zout)) / d)
        L_P = max(
            np.linalg.norm(C4 - C1 + pitch * _J), np.linalg.norm(C3 - C2 + pitch * _J)
        )
        L_in = (min(C1[0], C4[0]) - mp_Zin) * stretch_in
        L_out = (mp_Zout - max(C2[0], C3[0])) * stretch_out
        n_P, n_in, n_out = (max(2, round(L / Hc)) for L in (L_P, L_in, L_out))
        # The blocks ahead of and behind the blade finer along them still,
        # so the corners of the O block there have room to open out.
        n_in, n_out = (
            max(N_ENDS_MIN, round(COARSE_ENDS_FACTOR * n)) for n in (n_in, n_out)
        )
        f_P = _mult(_double(h / L_P, h / L_P, H / L_P, self.ER_stream), 8)
        # Ahead of and behind the blade, ENDS_FACTOR times the cells each
        # way, the clustering kept.
        k = ENDS_FACTOR
        f_in = _mult(_single(h / L_in, H / L_in, self.ER_stream), 8)
        f_out = _mult(_single(h / L_out, H / L_out, self.ER_stream), 8)
        f_in = 1.0 - np.flip(util.resample_to(f_in, k * (len(f_in) - 1) + 1))
        f_out = util.resample_to(f_out, k * (len(f_out) - 1) + 1)

        n_dn = 1 if extrude_out else _up8(round((mp_out - mp_Zout) * stretch_out / Hc))

        dw = float(spacing.surface[i_row]) / r_ref
        f_O = clusterfunc.single.free(dw / t_O, max(h, dw) / t_O, self.ER_O)

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
            corner_anchors=anchors,
            i_wall=i_C,
            n_O=max(2, round(2.0 * t_O / Hc)),
            n_P=n_P,
            n_in=n_in,
            n_out=n_out,
            n_up=1 if extrude_in else _up8(round((mp_Zin - mp_in) * stretch_in / Hc)),
            n_dn=n_dn,
            nk_O=len(_mult(f_O, 8)),
            H=H,
            H_coarse=Hc,
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
        :meth:`_wall` returns it, at its `N_WALL` samples.

        By :attr:`turn_max` round the blade's own edges on this plane, which
        sharpen and move round it along the span. See :func:`_coarse_density`.
        """
        return _coarse_density(
            blade_wall[0], layout.H_coarse, np.radians(self.turn_max), ER_COARSE
        )

    def _wall_half_density(self, blade_wall, layout, blade, spf, section):
        """Return the density of coarse cells along the blade on one plane, as
        :meth:`_wall` returns it, at its `N_WALL` samples, for a row whose
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
        s = np.linspace(0.0, marks[-1], N_WALL)
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
            n = _symmetric_counts(blade_walls, densities, layout.i_wall)
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
        n[i3] = n[i1] = max(n[i3], n[i1])
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
            blade_wall, f_wall = _symmetric_wall(blade_wall, density, layout.n_wall)
        else:
            f_wall = self._coarse_wall(blade_wall, density, layout.n_wall)
        g = self._geometry(blade_wall, layout.t_O, f_wall)
        loop, outer = g["loop"], g["outer"]
        C1, C2, C3, C4 = (g[k] for k in ("C1", "C2", "C3", "C4"))
        i3, i4, i1, i2 = layout.i_corners_coarse()
        periodic = _periodic_line(blade, cmap, pitch)

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
            mp_in = mp_Zin - layout.H_coarse
        if layout.extrude_out:
            mp_out = mp_Zout + layout.H_coarse
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
        fac = {c: self.cor_fac for c in round_blade}
        if self.cor_fac_mid is not None:
            for c, x in zip(round_blade, corners):
                if min(x, 1.0 - x, abs(x - 0.5)) > CORNER_MID:
                    fac[c] = self.cor_fac_mid
        for c in topo.irregular:
            for b, (g, s) in enumerate(zip(topo.gid, topo.shift)):
                ni, nk = g.shape
                # In the O block, no nearer the blade than half way across it,
                # so the nodes there stay free to square up to it. In any
                # block, no more than half way along or across it, so the
                # patches of corners at either end of a short block do not
                # overlap: claimed node by node by one corner or the other,
                # they draw its lines into a zigzag through the sweeps.
                most = self.n_corner if b else (nk - 1) // 2
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
        x = (
            1.0
            if b <= a
            else np.clip((float(plane["cmap"].spf) - a) / (b - a), 0.0, 1.0)
        )
        return float(x * x * (3.0 - 2.0 * x))

    def _sweeps(self, topo, X, n_sweep, plane):
        """Return `X` after `n_sweep` sweeps by :meth:`_sweeps_once`, its blade
        nodes held as far as :meth:`_freeze_weight` says: slid, held where they
        are, or slid, moved back that far towards where they started, and
        swept again held there."""
        w = self._freeze_weight(topo, plane)
        if w <= 0.0:
            return self._sweeps_once(topo, X, n_sweep, plane, False)
        if w < 1.0:
            # Slid, moved back that far towards where they started, and the
            # plane swept again held there
            X0 = X
            X = self._sweeps_once(topo, X, n_sweep, plane, False)
            X = self._blade_toward(topo, X, X0, plane, w)
        return self._sweeps_once(topo, X, n_sweep, plane, True)

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

    def _sweeps_once(self, topo, X, n_sweep, plane, frozen):
        """Return `X` after `n_sweep` point-Jacobi sweeps of Winslow's equations.

        By :func:`ember.elliptic.winslow_sweeps`: each sweep the regular and
        singular nodes from their stencils, the inlet and outlet level with
        the nodes inside, the nodes near the corners drawn in, the blade
        nodes of `plane` slid along the blade towards the foot of the
        perpendicular from the node outside them, by :meth:`_slide_factors`,
        and the nodes outside them drawn square to the blade by `ORTH`, with
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
            ([0.0], np.cumsum(MIN_GAP * np.diff(s_ref) * s_line[-1]))
        )

        lines = {}
        if self.orth_taper:
            K = gid.shape[1] - 1
            k = np.arange(2, K)
            lines["orth_lines"] = (
                gid[:, 2:K],
                shift[:, 2:K],
                (1.0 - (k - 1) / (K - 1)) ** self.orth_taper,
            )
        X, _, _ = ember.elliptic.winslow_sweeps(
            *args,
            (
                gid[:, 0],
                gid[:, 1],
                shift[:, 0],
                shift[:, 1],
                line,
                s_line,
                floor,
                tree.query(w0)[1],
                tree.query(w1)[1],
            ),
            n_sweep,
            ORTH,
            0.0 if frozen else self._slide_factors(plane, line, s_line, tree, w0),
            **lines,
        )
        return X

    def _slide_factors(self, plane, line, s_line, tree, w0):
        """Return how far each blade node at `w0` slides each sweep.

        The nodes off the blade are drawn square to it, by `ORTH`, and sliding
        the blade nodes under them round the leading and trailing edges only
        drags the clustering there onto the surfaces as the sweeps even the
        first layer. So none there, rising as the square of a sine along each
        side to `SLIDE_MID` midway between them, where the nodes are free to
        find their places along the blade.
        """
        S = s_line[-1]
        sigma = _project(line.T, s_line, tree, np.asarray(w0, dtype=float)) / S
        a = plane["sigma_LE"]
        # From the middle of the trailing edge (0) to the leading edge (1).
        t = np.where(sigma <= a, sigma / a, (1.0 - sigma) / (1.0 - a))
        return SLIDE_MID * np.sin(np.pi * np.clip(t, 0.0, 1.0)) ** 2

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
        lay = dataclasses.replace(
            lay, symmetric=self.tip_match == 3 and jtip is not None
        )
        # Each plane's blade evaluated once, for its coarse counts and nodes
        blade = machine.rows[i_row].blade
        cmaps = [ConformalMap(machine.annulus, s, lay.m_in, lay.m_out) for s in spf]
        sections = [_section(blade, s) for s in spf]
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
        topo.freeze = (self.tip_match_from, float(spf[jtip])) if lay.symmetric else None
        hold = topo.gid[0][:, : HOLD_COARSE + 1].ravel()
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
            X = self._sweeps(topo, X_alg[j], self.n_sweep, planes[j])
            # Polished before it is refined too: what the refined plane's
            # polish is then left to do steps the spacing less across the O
            # block's outer face.
            if self.polish_below:
                X += self._polish_shift(
                    topo, hold, topo.scatter(X), POLISH_COARSE_RINGS
                )
            if self.even_corners:
                X = topo.gather(_even_corners(topo.scatter(X), topo)).astype(X.dtype)
            if self.cluster_P:
                X = topo.gather(_cluster_P(topo.scatter(X))).astype(X.dtype)
            if self.cluster_O:
                X = topo.gather(_cluster_O(topo.scatter(X))).astype(X.dtype)
            if self.match_interfaces:
                X = topo.gather(_match_interfaces(topo.scatter(X))).astype(X.dtype)
            if self.theta_even and self.theta_lean:
                swept[j] = X
                logger.debug(f"Row {i_row} spf={spf[j]:.3f} swept")
                return
            if self.theta_even:
                xy = topo.scatter(X)
                for side in ("in", "out"):
                    xy = _theta_even(
                        xy, lay.pitch, self.theta_skew_min, self.dw_max, side
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
            offsets[j] = _O_offsets(topo, X_alg[j] + correction[j], planes[j])

        def finish():
            """With `theta_lean`, blend every control plane as swept, the
            weights held in step up the span, and keep each."""
            if not (self.theta_even and self.theta_lean):
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
                X if j in j_ctrl else _O_from_offsets(topo, X, planes[j], off[j])
                for j, X in enumerate(X_smooth)
            ]

        for j in j_ctrl:
            control(j)
        finish()
        X_smooth = carry()
        # A carried plane with a cell under `carry_below` made a control
        # plane, the worst between each pair, and the rest carried again,
        # until none is
        while self.carry_below:
            poor = {}
            for j, X in enumerate(X_smooth):
                if j in j_ctrl:
                    continue
                a = min(_cell_angles(B).min() for B in topo.scatter(X))
                k = int(np.searchsorted(j_ctrl, j))
                if a < self.carry_below and a < poor.get(k, (None, np.inf))[1]:
                    poor[k] = (j, a)
            if not poor:
                break
            for j, a in poor.values():
                logger.debug(
                    f"Row {i_row} spf={spf[j]:.4f} carried to {a:.1f} deg, swept"
                )
                control(j)
            j_ctrl = np.union1d(j_ctrl, [j for j, _ in poor.values()])
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
        # across span over `DN_SMOOTH_SPF`: each plane's alone step from one
        # to the next by more than the cells by the endwalls are high, and
        # lean the spanwise lines there over
        if "dn" in lay.u:
            F = np.stack(
                [_dn_fractions(xy[7], xy[4], xy[5], self.ER_mix) for xy in per_plane]
            )
            w = np.exp(-0.5 * ((spf[:, None] - spf[None]) / DN_SMOOTH_SPF) ** 2)
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
            if min(_cell_angles(B).min() for B in per_plane[j]) < self.polish_fine_below
        ]
        if self.polish_below:
            fine = _plane_topology(per_plane[j_mid], lay.pitch, stencils=False)
            hold = fine.gid[0][:, : HOLD_FINE + 1].ravel()
            shift = [None] * len(spf)
            for j in j_ctrl:
                if j in poor:
                    shift[j] = self._polish_shift(
                        fine, hold, per_plane[j], POLISH_FINE_RINGS
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
        tan = np.tan(np.radians(self.theta_lean))
        xy = [list(B) for B in xy]
        for side in ("in", "out"):
            found = [
                _theta_even(
                    B, pitch, self.theta_skew_min, self.dw_max, side, weights=True
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
                _theta_even(B, pitch, self.theta_skew_min, self.dw_max, side, cap=w)
                for B, w in zip(xy, W)
            ]
        return xy

    def _extrude_fractions(self, xy, plane, lay, inlet):
        """Return fractions of the way from the cut to the inlet or outlet
        the extruded block's stations lie at, on refined plane `xy`.

        From the size of the cells inside the cut, growing at `ER_stream`
        to the free stream size, in a count a multigrid level can halve.
        """
        face, inner = self._cut_face(xy, inlet)
        mp_end = plane["cmap"].mp_from_m(lay.m_in if inlet else lay.m_out)
        D = float(np.median(np.abs(mp_end - face[0])))
        h0 = float(np.median(np.linalg.norm(face - inner, axis=0)))
        # No cell longer than half the way, which may be short where the
        # inlet or outlet lies not far beyond the cut.
        h1 = min(max(lay.H, h0), 0.5 * D)
        try:
            f = _single(h0 / D, h1 / D, self.ER_stream)
        except clusterfunc.exceptions.ClusteringException:
            f = _uniform(min(h0, h1) / D)
        return _mult(f, 8)

    def _extrude(self, xy, plane, lay, inlet, f):
        """Replace the inlet or outlet block of refined plane `xy` by its cut
        face extruded to the inlet or outlet, at fractions `f` of the way.

        Each line leaves the cut along the grid line it continues, and turns
        to meridional with the pitch-mean slope `dtheta/dm'`, which falls
        linearly to nothing over :attr:`straighten` chords, its own
        departure from that mean fading within :data:`EXTRUDE_GAP` of the
        gap to its nearer neighbour; the nodes across the pitch relax from
        their spacing at the cut to even in theta over :attr:`even_theta`
        chords. The lines at either side of the pitch, one pitch apart at
        the cut, turn alike and stay so.
        """
        face, inner = self._cut_face(xy, inlet)
        mp_end = plane["cmap"].mp_from_m(lay.m_in if inlet else lay.m_out)
        chord = float(np.ptp(plane["wall"][0]))
        slope = (face[1] - inner[1]) / (face[0] - inner[0])
        mp = face[0][None] + f[:, None] * (mp_end - face[0])[None]
        d = np.abs(mp - face[0][None])
        sense = np.sign(mp_end - face[0])[None]
        # Every line turns as the pitch-mean slope does, straightening over
        # `straighten` chords; its own departure from that mean, with which
        # it leaves the cut along its grid line, fades within a fraction of
        # the gap to its nearer neighbour, so that lines leaving at different
        # angles cannot cross. The lines at either side of the pitch, images
        # of each other, see the gaps either side of the cut alike.
        R = self.straighten * chord
        run = np.where(d < R, d - d**2 / (2.0 * R), 0.5 * R)
        mean = float(np.mean(slope[:-1]))
        gap = np.diff(face[1])
        near = np.minimum(np.r_[gap[-1], gap], np.r_[gap, gap[0]])
        excess = np.abs(slope - mean)
        reach = EXTRUDE_GAP * near / np.maximum(excess, 1e-300)
        own = reach * -np.expm1(-d / np.maximum(reach, 1e-300)[None])
        theta = face[1][None] + sense * (mean * run + (slope - mean)[None] * own)
        nk = face.shape[1]
        even = theta[:, :1] + lay.pitch * np.linspace(0.0, 1.0, nk)[None]
        x = np.clip(d / (self.even_theta * chord), 0.0, 1.0)
        w = x * x * (3.0 - 2.0 * x)
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
            sigma, dens = self._plane_density(lay, plane)
            sigma = np.concatenate((sigma[:-1] - 1.0, sigma[:-1], sigma + 1.0))
            dens = np.concatenate((dens[:-1], dens[:-1], dens))
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
        that the blade turns by no more than `TURN_FINE` across a cell, the
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
        # Cell lengths, as fractions of the way round, to turn by TURN_FINE.
        d = np.gradient(wall, axis=1)
        phi = np.unwrap(np.arctan2(d[1], d[0]))
        turn = np.abs(np.gradient(phi, sw))
        turn = scipy.ndimage.uniform_filter1d(turn, N_SMOOTH_TURN, mode="wrap")
        h = np.minimum(h, np.radians(TURN_FINE) / np.maximum(turn, 1e-300))
        # Growing away from each fine cell at ER_surf at most.
        h = _grow_limited(h, sw, self.ER_surf - 1.0)
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
        sigma, dens = self._plane_density(lay, plane)
        sigma = np.concatenate((sigma[:-1] - 1.0, sigma[:-1], sigma + 1.0))
        dens = np.concatenate((dens[:-1], dens[:-1], dens))
        c = s_c[e] / S
        t = np.linspace(c[0], c[-1], N_WALL)
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
        for _ in range(N_MATCH_REAR):
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
        ER = self.ER_mix
        cap = max(self.AR_mix * width, d0 * ER)
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
        n_ER = np.log(h / d) / np.log(self.ER_O)
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
        side = [(0, n_sd, u["side_up"])]
        side_lo = [(0, n_sd, u["side_lo"])]
        across = [(0, n_P, u["P"])]
        front = [(0, n_fr, u["front"])]
        rear = [
            (0, n_rl, _reverse(u["rl"], n_rl)),
            (n_rl, n_rl + n_ru, _reverse(u["ru"], n_ru)),
        ]

        def shifted(pieces, by):
            return [(a + by, b + by, v) for a, b, v in pieces]

        plan = {
            "O": (self._wall_pieces(lay), None),
            "P": (side, across),
            "LE": ([(0, lay.n_in, u["in"])], front),
            "Pin": ([(0, lay.n_in, u["in"])], across),
            "TE": ([(0, lay.n_out, u["out"])], rear),
            "Pout": ([(0, lay.n_out, u["out"])], across),
            "up": (None, across + shifted(front, n_P)),
            "dn": (
                [(0, lay.n_dn, u["dn"])] if "dn" in u else None,
                across + shifted(rear, n_P),
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
            s_f = np.concatenate((s_lo, s_up[::-1][1:]))

        out = []
        for name, B in zip(LABELS, xy):
            R_i, R_k = refine[name]
            if name == "P":
                # Along the upper side of the blade at k = 0 and the lower at
                # k = -1, each as the O block it meets is refined, and
                # blended linearly across.
                w = np.linspace(0.0, 1.0, B.shape[2], dtype=B.dtype)
                B = (R_i @ B) * (1.0 - w) + (refine["P_lo"] @ B) * w
            elif R_i is not None:
                B = R_i @ B
            if R_k is not None:
                B = B @ R_k.T
            out.append(B)

        # The O block to the blade and clustered off it, its last cells as
        # long as the first ones beyond them.
        O = out[0]
        O[:, :, 0] = _at_arc(wall, np.mod(s_f, S))
        normal = _wall_normals(wall, s_f)
        # Round the leading edge, on the section's own outline: `wall` is even
        # in arc length, coarser there than the refined cells round a sharp
        # nose, and nodes on its straight pieces would facet it, the normal
        # jumping at each of its points. At the same fraction of the way round.
        i0, i1 = self._front_nodes(lay)
        section = plane["wall_section"]
        s_sec = np.mod(s_f[i0 : i1 + 1], S) * (util.arc_length(section) / S)
        O[:, i0 : i1 + 1, 0] = _at_arc(section, s_sec)
        normal[:, i0 : i1 + 1] = _wall_normals(section, s_sec)
        dtype = O.dtype
        # Each line a cubic spline through its coarse nodes, leaving the
        # blade along its normal, sampled finely, so the nodes respaced along
        # it leave the blade square and do not cut the corners where the
        # lines between coarse nodes bend.
        O = O.astype(float)
        speed = np.linalg.norm(O[:, :, 1] - O[:, :, 0], axis=0)
        start = normal * speed
        O = np.concatenate((O, start[:, :, None]), axis=2) @ _spline_O(O.shape[2]).T
        nk = lay.nk_O
        d_wall = lay.d_wall / plane["cmap"].xr_from_mp(O[0, :, 0])[1]
        d = np.diff(O, axis=2)
        d *= d
        s = np.zeros(O.shape[1:])
        np.cumsum(np.sqrt(d[0] + d[1]), axis=1, out=s[:, 1:])
        L = s[:, -1]
        # Double-sided stretching along each line, from the wall spacing, no
        # coarser than even, to the first cell of the block beyond, its
        # expansion ratio varying linearly between them.
        even = 1.0 / (nk - 1)
        f0 = np.clip(d_wall / L, 1e-12, even)
        # The outer end at the first cell beyond, coarser than even spacing
        # if it is, so the line meets it rather than growing past it and
        # shrinking back; only a line too short to hold both ends is held to
        # half its length there.
        if self.geometric_O:
            T = _geometric_many(f0, nk)
            if self.geometric_O_match > 1.0:
                # The outer cell towards the first beyond, by a factor
                # capped and smoothed round the blade, its last line the
                # first again.
                g1 = T[:, -1] - T[:, -2]
                c = np.log(self.geometric_O_match)
                q = np.clip(np.log(_beyond_O(out, lay) / L / g1), -c, c)[:-1]
                w = np.exp(
                    -0.5
                    * (
                        np.arange(-3 * O_MATCH_SMOOTH, 3 * O_MATCH_SMOOTH + 1)
                        / O_MATCH_SMOOTH
                    )
                    ** 2
                )
                q = np.convolve(
                    np.concatenate((q[-len(w) :], q, q[: len(w)])), w / w.sum(), "same"
                )[len(w) : -len(w)]
                f1 = g1 * np.exp(np.append(q, q[0]))
                T = clusterfunc.double.smooth_many(f0, f1, nk)
            T = L[:, None] * T
        else:
            f1 = np.clip(_beyond_O(out, lay) / L, 1e-12, 0.5)
            T = L[:, None] * clusterfunc.double.smooth_many(f0, f1, nk)
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
            out[7] = _dn_lines(out[7], out[4], out[5], self.ER_mix).astype(dtype)
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
            bad = _cell_angles(B) < self.polish_below
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
            power=POLISH_POWER,
            hold=hold,
        )
        return polished - X

    #
    # TIP GAP
    #

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
        from each inner node on side `tip_match`, which meets the O block
        node for node, as does the rear, round the tail through the cut,
        whose nodes set the H block's count across; the front, of as many
        nodes, and the other side meet it by non-matching interfaces
        (:func:`_gap_H`).

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
        corners = u_rear = k_cut = None
        gap = []
        for W, plane in zip(walls, planes):
            cmap = plane["cmap"]
            cam = _Camber(*_surfaces(blade, cmap, cmap.spf))
            inner = _gap_offset(W, plane["wall"], self.t_tip_O)
            if corners is None:
                corners, u_rear = _gap_corners(inner, i_LE, cam, self)
            if self.tip_rear_search:
                # Through the corners' own nodes, which so stay where they
                # are, once leant level along the camber
                if self.tip_rear_lean:
                    inner = _gap_rear_bisect(
                        inner, W, corners, self, self.tip_rear_lean
                    )
                rear = (inner[:, corners[0]], inner[:, corners[3]])
            else:
                rear = _gap_rear(inner, i_LE, cam, u_rear)
            if self.tip_front_lean:
                inner = _gap_lean(inner, W, corners, cam, self.tip_front_lean)
            # Both sides, each rear corner landing where the rear line meets
            # it, so the O block's lines lean progressively towards it on
            # either side rather than cross by the corner it is moved to
            # The far side's slide faded out over `tip_blend` of the camber
            # from its corner, so the rest of that side stays square to the
            # blade
            fade = self.tip_blend * cam.s[-1] if self.tip_blend > 0.0 else None
            for side in (1, 2):
                inner = _gap_spread(
                    inner,
                    corners,
                    rear,
                    i_LE,
                    side,
                    fade=None if side & self.tip_match else fade,
                )
            inner, front = _gap_square(inner, corners, rear, self)
            H, k_cut = _gap_H(
                inner,
                i_LE,
                corners,
                front,
                cam,
                self.tip_match,
                self.tip_blend,
                both=self.tip_match == 3,
            )
            O = _gap_O_lines(inner, W, self.tip_O_square)
            d_wall = layout.d_wall / cmap.xr_from_mp(W[0])[1]
            O = _respace_gap_O(O, d_wall, self.nk_tip_O)
            if self.n_sweep_tip:
                O = _respace_gap_O(_winslow(O, self.n_sweep_tip, periodic=True), d_wall)
                H = _winslow(H, self.n_sweep_tip)
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

    #
    # BLOCKS
    #

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
        casing = self._inviscid_ducts(machine, i_row, up, dn)
        if jtip is None:
            return blocks

        # Over the gap the blade's outline is the face the O block of the gap
        # shares with the O block round the blade, node for node. Inside it,
        # the gap's O block meets its H block node for node along the matched
        # side and the rear, either side of the cut, and by non-matching
        # interfaces along the front and the other side, or with
        # `tip_match` 3 node for node all round.
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
        Lower, Upper = (Per, NonMatch) if self.tip_match == 1 else (NonMatch, Per)
        if self.tip_match == 3:
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
        # Not over an inviscid stretch of the casing, which a rotating patch
        # would overlap.
        blocks += [tO, tH]
        for blk in blocks:
            i_rot = casing.get(id(blk), (0, -1))
            blk.patches.append(ember.patch.RotatingPatch(i=i_rot, j=-1, label="casing"))
        return blocks


EVEN_O_CELLS = 16
"""Cells either side of a singular point over which the spacing of the O
block's outer face is evened, rather than along the whole of a run, so the
O block's lines lean only near its corners; each window keeps its length,
so the more cells, the less each is changed to make room for its corner's."""


def _even_corners(xy, topo):
    """Return blocks `xy` (2, ni, nk) respaced so that the first cell out of
    each singular point of `topo` along every grid line leaving it is within
    `EVEN_RATIO` of their geometric mean, so a few long spokes do not pull the
    short ones up to them, and those already within it stay as they are.

    Along every block edge, split at the singular points and the block
    corners on it, the end cells of each run at a singular point are scaled to
    that band, 1 at an end that is not one, the factor easing to 1 over
    `EVEN_O_CELLS` from the end, each such window keeping its length. The spoke into the
    O block from each corner, a line inside it rather than an edge, is respaced
    likewise, and blended in `i` over the same number of cells either side. The
    nodes slide along the edge to
    their new places, which as fractions of its index are blended across
    each block from its edges, transfinitely, in the O block from one node
    off the blade rather than from the blade. Each node then moves by arc
    length along the block's grid lines to its new fractions: along its
    `i` line first, then along the `k` line through where that left it, so
    it stays on the grid lines rather than cutting across the cells.
    """
    irregular = {int(c) for c in topo.irregular}
    blocks = [np.asarray(B, dtype=float) for B in xy]

    def edges(B, g):
        """The four edges of a block: along i at k = 0 and -1, along k at
        i = 0 and -1; each its nodes (2, n) and their node numbers."""
        return (
            (B[:, :, 0], g[:, 0]),
            (B[:, :, -1], g[:, -1]),
            (B[:, 0, :], g[0, :]),
            (B[:, -1, :], g[-1, :]),
        )

    # The first cell of every spoke out of each singular point
    first = {c: [] for c in irregular}
    for B, g in zip(blocks, topo.gid):
        for P, ids in edges(B, g):
            L = np.linalg.norm(np.diff(P, axis=1), axis=0)
            for q, c in enumerate(ids):
                if int(c) in irregular:
                    if q > 0:
                        first[int(c)].append(L[q - 1])
                    if q < len(L):
                        first[int(c)].append(L[q])
    # The spoke into the O block from each corner on its outer face, a line
    # of it rather than one of its edges
    O, gO = blocks[0], topo.gid[0]
    O_spokes = [
        (i, int(gO[i, -1])) for i in range(gO.shape[0]) if int(gO[i, -1]) in irregular
    ]
    for i, c in O_spokes:
        first[c].append(float(np.linalg.norm(O[:, i, -1] - O[:, i, -2])))
    mean = {c: float(np.exp(np.mean(np.log(v)))) for c, v in first.items() if v}

    def to_band(L, c):
        """Return first cell `L` of a spoke of `c` moved to within `EVEN_RATIO`
        of their geometric mean."""
        return float(np.clip(L, mean[c] / EVEN_RATIO, mean[c] * EVEN_RATIO))

    # Runs end at every block's corners as well as at the singular points,
    # so that an edge one block shares with two neighbours is respaced in
    # the same runs as theirs, and the nodes they share agree
    stops = set(mean) | {
        int(g[a, b]) for g in topo.gid for a in (0, -1) for b in (0, -1)
    }

    def held(run, g, ends):
        """Return the cells of `run` scaled by `g`, those at `ends` held at
        that and the rest rescaled to keep the run's length."""
        new = run * g
        rest = np.ones(len(run), dtype=bool)
        rest[list(ends)] = False
        spare = run.sum() - new[~rest].sum()
        if rest.any() and spare > 0.0:
            new[rest] *= spare / new[rest].sum()
        else:
            new *= run.sum() / new.sum()
        return new

    def ramped(run, r0, r1):
        """Return the run's cells scaled by `r0` at its start and `r1` at its
        end, over `EVEN_O_CELLS` from each end only, each window keeping its
        length; the end cells are those scaled exactly."""
        m = len(run)
        new = run.copy()
        n_w = max(min(EVEN_O_CELLS, m // 2), 1)
        # The first cell of each window its corner's whole factor, easing to
        # one at the far side of the window
        fall = 1.0 - np.arange(n_w) / n_w
        for r, window, f, end in (
            (r0, slice(0, n_w), fall, 0),
            (r1, slice(m - n_w, m), fall[::-1], n_w - 1),
        ):
            new[window] = held(new[window], 1.0 + (r - 1.0) * f, (end,))
        return new

    def respaced(P, ids):
        """Return the new fractional index of each node along an edge."""
        L = np.linalg.norm(np.diff(P, axis=1), axis=0)
        n = len(L)
        cuts = [0] + [q for q in range(1, n) if int(ids[q]) in stops] + [n]
        f = np.arange(n + 1, dtype=float)
        for a, b in zip(cuts[:-1], cuts[1:]):
            run = L[a:b]
            m = len(run)
            r0 = to_band(run[0], int(ids[a])) / run[0] if int(ids[a]) in mean else 1.0
            r1 = to_band(run[-1], int(ids[b])) / run[-1] if int(ids[b]) in mean else 1.0
            new = ramped(run, r0, r1)
            s_old = np.concatenate(([0.0], np.cumsum(run)))
            s_new = np.concatenate(([0.0], np.cumsum(new)))
            f[a : b + 1] = a + np.interp(s_new, s_old, np.arange(m + 1))
        return f

    out = []
    for b, (B, g) in enumerate(zip(blocks, topo.gid)):
        ni, nk = B.shape[1:]
        (xi0, xi1, eta0, eta1) = (respaced(P, ids) for P, ids in edges(B, g))
        a = np.linspace(0.0, 1.0, nk)[None, :]
        if b == 0:
            # The O block blended from one node off the blade, so the cells
            # against it, and the lines leaving it, stay as the sweeps left them
            a = np.clip((np.arange(nk) - 1.0) / (nk - 2.0), 0.0, 1.0)[None, :]
        b_i = np.linspace(0.0, 1.0, ni)[:, None]
        xi = (1.0 - a) * xi0[:, None] + a * xi1[:, None]
        eta = (1.0 - b_i) * eta0[None, :] + b_i * eta1[None, :]
        if b == 0:
            # Each corner's spoke into the O block from the blade, held, to
            # the corner, eased back to as it was over `EVEN_O_CELLS` in i
            for i_c, c in O_spokes:
                L = np.linalg.norm(np.diff(B[:, i_c, :], axis=1), axis=0)
                new_L = ramped(L, 1.0, to_band(L[-1], c) / L[-1])
                s_old = np.concatenate(([0.0], np.cumsum(L)))
                s_new = np.concatenate(([0.0], np.cumsum(new_L)))
                eta_c = np.interp(s_new, s_old, np.arange(nk))
                w = np.clip(1.0 - np.abs(np.arange(ni) - i_c) / EVEN_O_CELLS, 0.0, 1.0)
                eta += w[:, None] * (eta_c - np.arange(nk))[None, :]
        # Along each i line to its new fraction of i, then along each k line
        # of the result to its new fraction of k: each a move by arc length
        # along a polyline, linear in the fraction within a cell
        along_i = np.empty_like(B)
        for k in range(nk):
            for c in range(2):
                along_i[c, :, k] = np.interp(xi[:, k], np.arange(ni), B[c, :, k])
        new = np.empty_like(B)
        for i in range(ni):
            for c in range(2):
                new[c, i, :] = np.interp(eta[i, :], np.arange(nk), along_i[c, i, :])
        out.append(new)
    return out


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


def _cluster_O(xy):
    """Return blocks `xy` (2, ni, nk) with the O block respaced along `k`.

    Each line from the blade out to the O block's outer face by double-sided
    clustering between its own first and last cells, the expansion ratio
    varying linearly between, its nodes moved by arc length along it, so
    both its ends stay.
    """
    out = [np.asarray(B, dtype=float) for B in xy]
    O = out[0]
    nk = O.shape[2]
    d = np.linalg.norm(np.diff(O, axis=2), axis=0)
    s = np.concatenate((np.zeros((d.shape[0], 1)), np.cumsum(d, axis=1)), axis=1)
    L = s[:, -1]
    f = clusterfunc.double.smooth_many(d[:, 0] / L, d[:, -1] / L, nk)
    new = O.copy()
    for i in range(O.shape[1]):
        for c in range(2):
            new[c, i] = np.interp(f[i] * L[i], s[i], O[c, i])
    out[0] = new
    return out


def _cluster_P(xy):
    """Return blocks `xy` (2, ni, nk) with the passage block respaced along
    `i`.

    Its mid-pitch line by double-sided clustering between its own first and
    last cells, the expansion ratio varying linearly between, so the cells
    along it grade smoothly from one end to the other; the change in where
    each node is along its line, as a fraction of `i`, falling linearly in
    `k` to none at the pitchwise boundaries, which do not move. Each node
    moves by arc length along its `i` line, so the ends of every line stay.
    """
    out = [np.asarray(B, dtype=float) for B in xy]
    P = out[1]
    ni, nk = P.shape[1:]
    k_mid = nk // 2
    line = P[:, :, k_mid]
    s = np.concatenate(
        ([0.0], np.cumsum(np.linalg.norm(np.diff(line, axis=1), axis=0)))
    )
    L = s[-1]
    f = clusterfunc.double.smooth_many([(s[1] - s[0]) / L], [(s[-1] - s[-2]) / L], ni)[
        0
    ]
    xi_mid = np.interp(f * L, s, np.arange(ni))
    w = 1.0 - np.abs(np.arange(nk) - 0.5 * (nk - 1)) / (0.5 * (nk - 1))
    xi = np.arange(ni)[:, None] + w[None, :] * (xi_mid - np.arange(ni))[:, None]
    new = P.copy()
    for k in range(nk):
        for c in range(2):
            new[c, :, k] = np.interp(xi[:, k], np.arange(ni), P[c, :, k])
    out[1] = new
    return out


def _match_interfaces(xy):
    """Return blocks `xy` (2, ni, nk) respaced so the cells either side of
    each interface between LE and Pin, Pin and P, P and Pout, and Pout and
    TE are within `MATCH_RATIO` of each other.

    The interfaces are :data:`INTERFACES`. Along every grid line crossing one,
    cells within the ratio are left be, and others moved to it with their sum
    kept; along
    each line of a block in that direction, the cells are scaled by a
    factor varying linearly by index between those at its two ends, one at
    an end on no interface, its end cells held there and the rest rescaled
    to keep its length, and its nodes moved by arc length along it. A block
    with interfaces both ways is respaced along `i` first, then along `k`.
    The lines at the ends of a face are held, so the edges the blocks share
    with others stay where they are, whatever is left at the singular points.
    """
    out = [np.asarray(B, dtype=float) for B in xy]

    def first(B, f):
        """The cell next to face `f` on every line crossing it."""
        if f == "i0":
            return np.linalg.norm(B[:, 1, :] - B[:, 0, :], axis=0)
        if f == "i1":
            return np.linalg.norm(B[:, -1, :] - B[:, -2, :], axis=0)
        if f == "k0":
            return np.linalg.norm(B[:, :, 1] - B[:, :, 0], axis=0)
        return np.linalg.norm(B[:, :, -1] - B[:, :, -2], axis=0)

    def eased(r):
        """Return factors `r` along a face, one on the lines at its ends, which
        are the edges of the block and shared with others, and easing to `r`
        over `MATCH_EASE` lines."""
        n = len(r)
        w = np.minimum(np.minimum(np.arange(n), np.arange(n)[::-1]) / MATCH_EASE, 1.0)
        return 1.0 + (r - 1.0) * w

    ratio = {}
    for a, fa, b, fb in INTERFACES:
        la, lb = first(out[a], fa), first(out[b], fb)
        least = (la + lb) / (1.0 + MATCH_RATIO)
        most = MATCH_RATIO * least
        ratio[a, fa] = eased(
            np.where(
                la > MATCH_RATIO * lb, most, np.where(lb > MATCH_RATIO * la, least, la)
            )
            / la
        )
        ratio[b, fb] = eased(
            np.where(
                lb > MATCH_RATIO * la, most, np.where(la > MATCH_RATIO * lb, least, lb)
            )
            / lb
        )

    def respace(line, r0, r1):
        """A line (2, n) respaced between its end ratios, its length kept."""
        L = np.linalg.norm(np.diff(line, axis=1), axis=0)
        m = len(L)
        w = np.arange(m) / (m - 1) if m > 1 else np.full(1, 0.5)
        new = L * (r0 + (r1 - r0) * w)
        if m > 2:
            spare = L.sum() - new[0] - new[-1]
            if spare > 0.0:
                new[1:-1] *= spare / new[1:-1].sum()
            else:
                new *= L.sum() / new.sum()
        else:
            new *= L.sum() / new.sum()
        s = np.concatenate(([0.0], np.cumsum(L)))
        s_new = np.concatenate(([0.0], np.cumsum(new)))
        return np.stack([np.interp(s_new, s, c) for c in line])

    for b, B in enumerate(out):
        ni, nk = B.shape[1:]
        if (b, "i0") in ratio or (b, "i1") in ratio:
            r0 = ratio.get((b, "i0"), np.ones(nk))
            r1 = ratio.get((b, "i1"), np.ones(nk))
            B = B.copy()
            for k in range(nk):
                B[:, :, k] = respace(B[:, :, k], r0[k], r1[k])
        if (b, "k0") in ratio or (b, "k1") in ratio:
            r0 = ratio.get((b, "k0"), np.ones(ni))
            r1 = ratio.get((b, "k1"), np.ones(ni))
            B = B.copy()
            for i in range(ni):
                B[:, i, :] = respace(B[:, i, :], r0[i], r1[i])
        out[b] = B
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
        return np.stack([np.interp(s_new, s, c) for c in line])

    new = band.copy()
    full = band.copy()
    w = np.zeros(ni)
    cap = np.ones(ni) if cap is None else np.asarray(cap, dtype=float)
    # Each station's line, and the floor of the cells between it and the one
    # after, nearer the blade
    stations, floors = [None] * ni, [None] * ni

    def holds(k, a, b):
        """Whether the cells between lines `a` at station k and `b` at k + 1
        keep to their floor."""
        return np.all(_strip_angles(b)(a) >= floors[k] - 1e-9)

    for i in range(ni - 2, -1, -1):
        angles = _strip_angles(new[:, i + 1])
        floor = np.minimum(_strip_angles(band[:, i + 1])(band[:, i]), skew_min)
        floors[i] = floor
        hi = min(1.0, w[i + 1] + dw_max, cap[i])
        lo = min(w[i + 1], hi)
        line, s, s_even = stations[i] = station(i)
        full[:, i] = blended(line, s, s_even, 1.0)

        def ok(wi):
            return np.all(angles(blended(line, s, s_even, wi)) >= floor - 1e-9)

        if ok(hi):
            lo = hi
        else:
            if not ok(lo):
                # Holding the weight of the station before would break the
                # floor too: the most below it that does not, or none
                lo, hi = 0.0, lo
            for _ in range(20):
                mid = 0.5 * (lo + hi)
                lo, hi = (mid, hi) if ok(mid) else (lo, mid)
        w[i] = lo
        new[:, i] = blended(line, s, s_even, lo)
        # Where none at this station keeps the floor, the stations nearer
        # the blade took too much: each lowered in turn, as little as keeps
        # the cells either side of it, until the floor holds. All of them at
        # none is the band as it was, which does.
        k = i
        while not holds(k, new[:, k], new[:, k + 1]) and k + 1 < ni - 1:
            m = k + 1

            def both(wm):
                line_m = blended(*stations[m], wm)
                return holds(k, new[:, k], line_m) and holds(m, line_m, new[:, m + 1])

            lo, hi = 0.0, w[m]
            for _ in range(20):
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


def _gap_offset(W, wall, frac):
    """Return outline `W` (2, N + 1) offset inwards along the blade's normal.

    Along the normal of the blade, closed line `wall`, at each node, by
    `frac` of the blade's greatest thickness, the diameter of the largest
    circle inside it, capped at `GAP_CAP` of the radius of the largest circle
    inside it that touches it at that node, round the edges where the blade
    is thinner.
    """
    n = _wall_normals(wall, _wall_arc({"wall": wall}, W))[:, :-1]
    r = _inscribed_radius(W[:, :-1], -n, wall)
    d = np.minimum(frac * 2.0 * r.max(), GAP_CAP * r)
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
        t = np.array([np.interp(s, self.s, c) for c in self.t])
        return t / np.linalg.norm(t)

    def normal(self, s):
        """Return the point (2,) at arc length `s` and the unit normal there."""
        P = np.array([np.interp(s, self.s, c) for c in self.xy])
        t = self.tangent(s)
        return P, np.array([-t[1], t[0]])

    def across(self, s, sides):
        """Return where the normal at `s` meets each of `sides`, or None."""
        P, n = self.normal(s)
        return [_intersect(Q, P, n) for Q in sides]


class _GapScan:
    """Lines normal to the camber line of a tip gap's inner face `inner`
    (2, N + 1), at `N_GAP_SCAN` points from its nose to its tail: where each
    meets the two sides, the width across there, and the H block's corners
    on them."""

    def __init__(self, inner, i_LE, cam, mesher):
        self.inner, self.i_LE, self.mesher = inner, i_LE, mesher
        self.sides = (inner[:, : i_LE + 1], inner[:, i_LE:])
        s_nose, s_tail = cam.at(inner[:, [i_LE, 0]])
        self.s = np.linspace(s_nose, s_tail, N_GAP_SCAN)
        self.u = np.linspace(0.0, 1.0, N_GAP_SCAN)
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
        self.to_tail = (s_tail - self.s) / (0.5 * self.width)
        front = self.ok & (self.from_nose >= mesher.tip_end)
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

    def corners(self, j):
        """Return the corners (iA, iB, iC, iD), A and D the nodes nearest
        where line `j` meets each side, the far one on side `tip_match` the
        node a multiple of eight cells along that side from B or C, the H
        block's length, nearest its crossing; or None if out of order. That
        side's nodes are then spread to put it on the crossing: see
        :func:`_gap_spread`."""
        uA = _on_polyline(self.sides[0], self.across[j][0])
        uD = self.i_LE + _on_polyline(self.sides[1], self.across[j][1])
        iA, iD = int(round(uA)), int(round(uD))
        iB, iC = self.iB, self.iC
        if self.mesher.tip_match == 1:
            iA = iB - max(8, 8 * round((iB - uA) / 8))
        else:
            iD = iC + max(8, 8 * round((uD - iC) / 8))
        N = self.inner.shape[1] - 1
        return (iA, iB, iC, iD) if 0 < iA < iB < iC < iD < N else None

    def rear(self, to_tail):
        """Return the line furthest aft at least `to_tail` half widths from
        the tail, or None if there is none behind the front."""
        js = np.flatnonzero(self.ok & (self.to_tail >= to_tail))
        js = js[js > self.j_F]
        return int(js[-1]) if len(js) else None


def _gap_rear_search(
    inner, i_LE, cam, scan, to_tail, multiple=8, most=0, symmetric=False
):
    """Return the rear corners (iA, iD) of a tip gap's H block, nodes of
    inner face `inner` (2, N + 1), whose line is nearest normal to the
    camber, and how far along the camber that line's middle is, from the
    nose to the tail.

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
    s_nose, s_tail = cam.at(inner[:, [i_LE, 0]])
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
    (tilt, _), iA, iD, s_mid = best
    logger.debug(f"Tip gap rear corners A={iA}, D={iD}, {tilt:.2f} deg off normal")
    return iA, iD, float((s_mid - s_nose) / (s_tail - s_nose))


def _gap_rear(inner, i_LE, cam, u):
    """Return where the line normal to the camber of inner face `inner`
    (2, N + 1), `u` of the way along it from the nose to the tail, meets
    each side."""
    s_nose, s_tail = cam.at(inner[:, [i_LE, 0]])
    sides = (inner[:, : i_LE + 1], inner[:, i_LE:])
    pA, pD = cam.across(s_nose + u * (s_tail - s_nose), sides)
    if pA is None or pD is None:
        raise ValueError("The tip gap's H block's rear line misses the inner face.")
    return pA, pD


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
        x = np.clip(np.abs(s[idx] - s[i0]) / fade, 0.0, 1.0)
        s_new = s[idx] + (1.0 - x * x * (3.0 - 2.0 * x)) * (s_new - s[idx])
    out = inner.copy()
    out[:, idx] = [np.interp(s_new, s, c) for c in inner]
    return out


def _gap_lean(inner, W, corners, cam, lean, rear=False):
    """Return inner face `inner` (2, N + 1) with the H block's front corners
    B and C, or with `rear` its rear corners A and D, slid along it towards
    the place along camber `cam` midway between them, each no further than
    leans the O block's line from it to outline `W` `lean` degrees off the
    normal; the nodes on its side towards the other end of the block
    following it less and less, to none half way to the corner there.
    Round the nose or the tail, the nodes between are spread again when it
    is squared off (:func:`_gap_square`)."""
    iA, iB, iC, iD = corners
    N = inner.shape[1] - 1
    s = util.cum_arc_length(inner)
    pairs = ((iA, iB), (iD, iC)) if rear else ((iB, iA), (iC, iD))
    u = cam.at(inner[:, [pairs[0][0], pairs[1][0]]])
    target = 0.5 * (u[0] + u[1])
    out = inner.copy()
    tan = np.tan(np.radians(lean))
    for i, i_rear in pairs:
        reach = tan * np.linalg.norm(W[:, i] - inner[:, i])
        # Where along the face, within reach of the node, is nearest the
        # target along the camber
        a = s[i] + np.linspace(-reach, reach, 201)
        pts = np.array([np.interp(a, s, c) for c in inner])
        slide = a[int(np.argmin(np.abs(cam.at(pts) - target)))] - s[i]
        # Its side behind it eased from the whole slide to none
        fade = 0.5 * abs(s[i_rear] - s[i])
        step = 1 if i_rear > i else -1
        idx = np.arange(i, i_rear + step, step)
        x = np.clip(np.abs(s[idx] - s[i]) / fade, 0.0, 1.0)
        s_new = s[idx] + (1.0 - x * x * (3.0 - 2.0 * x)) * slide
        out[:, idx] = [np.interp(s_new, s, c) for c in inner]
    out[:, N] = out[:, 0]
    return out


def _gap_slide(inner, s, i, i_far, slide):
    """Return inner face `inner` (2, N + 1), arc lengths `s` along it, with
    node `i` slid `slide` along it, the nodes towards `i_far` following less
    and less, to none half way there."""
    fade = 0.5 * abs(s[i_far] - s[i])
    step = 1 if i_far > i else -1
    idx = np.arange(i, i_far + step, step)
    x = np.clip(np.abs(s[idx] - s[i]) / fade, 0.0, 1.0)
    s_new = s[idx] + (1.0 - x * x * (3.0 - 2.0 * x)) * slide
    out = inner.copy()
    out[:, idx] = [np.interp(s_new, s, c) for c in inner]
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
            reach = tan * np.linalg.norm(W[:, i] - inner[:, i])
            a = s[i] + np.linspace(-reach, reach, 201)
            pts = np.array([np.interp(a, s, c) for c in inner])
            o = W[:, i, None] - pts
            cos = (b @ o) / np.linalg.norm(b) / np.linalg.norm(o, axis=0)
            slides[i] = a[int(np.argmax(cos))] - s[i]
    return slid()


def _gap_corners(inner, i_LE, cam, mesher):
    """Return the corners (iA, iB, iC, iD) of a tip gap's H block, nodes of
    inner face `inner` (2, N + 1), and how far along the camber its rear line
    is, from the nose to the tail.

    B and C the nodes nearest where a line normal to the camber, back from
    the inner face's nose along it by `tip_end` of half the width across
    there, meets each side (:class:`_GapScan`). The rear corner on side
    `tip_match` the node a multiple of eight cells along it from B or C, the
    H block's length, nearest where another, `tip_end_rear` from its tail,
    meets it; the rear line then moved to the normal to the camber through
    that node, so that it sits at its own node of the outline and its O
    line leaves the side square, and the other rear corner the node nearest
    where that meets the other side.
    The other rear corner is then moved, for a multiple of eight cells round
    the tail from A through the cut to D, the H block's count across, which
    its rear meets node for node: to the nearest such node at or behind
    where the rear line crosses its side, so that :func:`_gap_spread` only
    ever slides it forward (:func:`_gap_rear_corners`). With `tip_match` 1,
    if then no front has as many cells round the nose, the rear corner on
    side 1 is moved forward eight cells at a time, up to three times, until
    one does.
    """
    scan = _GapScan(inner, i_LE, cam, mesher)
    if mesher.tip_match == 3:
        # The halves of the inner face alike, node h - k across from h + k:
        # the rear corners as far round from the cut either way, and the front
        # as far round from the leading edge, so the rear and front have as
        # many cells, and the sides too
        iA, iD, u = _gap_rear_search(
            inner,
            i_LE,
            cam,
            scan,
            mesher.tip_end_rear,
            mesher.tip_rear_multiple,
            mesher.tip_rear_max,
            symmetric=True,
        )
        h = (inner.shape[1] - 1) // 2
        corners = (iA, h - iA, h + iA, iD)
        if not iA < h - iA:
            raise ValueError(
                f"The tip gap's rear corners {(iA, iD)} are past its front ones "
                f"{(h - iA, h + iA)}, the halves of its inner face {h} cells."
            )
        return corners, u
    if mesher.tip_rear_search:
        iA, iD, u = _gap_rear_search(
            inner,
            i_LE,
            cam,
            scan,
            mesher.tip_end_rear,
            mesher.tip_rear_multiple,
            mesher.tip_rear_max,
        )
        # As many cells round the nose as the tail, centred where the scan
        # put the front corners, so the H block's front meets the O block
        # node for node and only its unmatched side does not
        n = iA + inner.shape[1] - 1 - iD
        iB = int(round(0.5 * (scan.iB + scan.iC - n)))
        corners = (iA, iB, iB + n, iD)
        if (
            not 0
            < corners[0]
            < corners[1]
            < corners[2]
            < corners[3]
            < inner.shape[1] - 1
        ):
            raise ValueError(
                f"The tip gap's front corners, {n} cells round the nose to match "
                f"the tail, overlap its rear corners {(iA, iD)}."
            )
        return corners, u
    j = scan.rear(mesher.tip_end_rear)
    corners = None if j is None else scan.corners(j)
    u = None
    if corners is not None:
        iA, iB, iC, iD = corners
        if mesher.tip_match == 1:
            # The matched corner where the scan put it, or failing a front
            # that matches, a multiple of eight cells further forward: the
            # rear line then crosses the other side further forward too,
            # leaving more cells round the tail for the nose to match
            first = None
            for iA_try in range(iA, iB, 8)[:4]:
                tried = _gap_rear_corners(inner, i_LE, cam, scan, iA_try, iB, iD)
                first = first or tried
                if tried[-1]:
                    break
            *corners, u, _ = tried if tried[-1] else first
        else:
            *corners, u, _ = _gap_rear_corners(
                inner, i_LE, cam, scan, iA, iB, iD, side=2
            )
        corners = tuple(corners)
        if (
            not 0
            < corners[0]
            < corners[1]
            < corners[2]
            < corners[3]
            < inner.shape[1] - 1
        ):
            corners = None
    if corners is None:
        raise ValueError(
            f"The tip gap's H block has no rear corners in order behind "
            f"{(scan.iB, scan.iC)} round {inner.shape[1] - 1} cells: reduce "
            "tip_end or tip_end_rear, or give the blade more nodes round it."
        )
    return corners, u


def _gap_rear_corners(inner, i_LE, cam, scan, iA, iB, iD, side=1):
    """Return the corners (iA, iB, iC, iD), how far along the camber the rear
    line is, and whether the front matches, for the rear corner on side
    `side`, the matched one, at node `iA` (side 1) or `iD` (side 2).

    The rear line is the normal to the camber through the matched corner.
    The cells round the tail, the H block's across, a multiple of eight, by
    the other rear corner: at or behind where the rear line crosses its
    side, never ahead of it. :func:`_gap_spread` then slides it forward onto
    the crossing, closing up the nodes between it and the front corner.
    Ahead of the crossing it would slide back over the nodes behind it,
    which stay where they are until :func:`_gap_square` puts them on the
    tail, and near a thin trailing edge the O block's lines from them cross
    its own.

    The front as many cells round the nose, so it too meets the O block node
    for node, on side 1: B any node, the H block's length then not a
    multiple of eight, and C that count on, nearest where the normal to the
    camber through B meets the upper side, so the front line is square to
    the camber; of those, nearest the scan's B. Otherwise B and C stay the
    scan's and the front does not match.
    """
    N = inner.shape[1] - 1
    i_far = iA if side == 1 else iD
    s_far = cam.at(inner[:, [i_far]])[0]
    u = float((s_far - scan.s[0]) / (scan.s[-1] - scan.s[0]))
    pA, pD = _gap_rear(inner, i_LE, cam, u)
    if side == 1:
        uD = i_LE + _on_polyline(inner[:, i_LE:], pD)
        n = 8 * max(1, int(np.floor((iA + N - uD) / 8)))
        iD = N - (n - iA)
    else:
        uA = _on_polyline(inner[:, : i_LE + 1], pA)
        n = 8 * max(1, int(np.floor((uA + N - iD) / 8)))
        iA = n - (N - iD)
    iC = scan.iC
    matched = False
    if side == 1:
        best = None
        for jB in range(iA + 1, iD - n):
            jC = jB + n
            if not jB < i_LE < jC:
                continue
            P, nrm = cam.normal(cam.at(inner[:, [jB]])[0])
            q = _intersect(inner[:, i_LE:], P, nrm)
            if q is None:
                continue
            key = (abs(jC - i_LE - _on_polyline(inner[:, i_LE:], q)), abs(jB - iB))
            if best is None or key < best[0]:
                best = (key, jB, jC)
        if best is not None:
            _, iB, iC = best
            matched = True
    return iA, iB, iC, iD, u, matched


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
    tail = 2.0 * (M_rear + mesher.tip_bow * (inner[:, 0] - M_rear)) - M_rear
    M = 0.5 * (pB + pC)
    n = np.array([pB[1] - pC[1], pC[0] - pB[0]]) / np.linalg.norm(pC - pB)
    n *= np.sign(n @ (M - M_rear))
    hit = _intersect(inner[:, iB : iC + 1], M, n)
    nose = 2.0 * (M + mesher.tip_bow_front * (hit - M)) - M

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
        w = mesher.tip_nose_outline if front is None else mesher.tip_tail_outline
        if w:
            # Part way to as the nodes before squaring lie along it
            a = util.cum_arc_length(inner[:, idx])
            f = (1.0 - w) * f + w * a / a[-1]
        out[:, idx] = [np.interp(f, g / g[-1], c) for c in curve]
        if front is None:
            u = np.linspace(0.0, 1.0, iA + N - iD + 1)
            fH = np.interp(u, np.linspace(0.0, 1.0, len(idx)), f)
            front = np.array([np.interp(fH, g / g[-1], c) for c in curve])
    out[:, N] = out[:, 0]
    if iC - iB == iA + N - iD:
        # As many round the nose as the tail: the front meets them node for node
        front = out[:, iB : iC + 1]
    return out, front


def _gap_H(inner, i_LE, corners, front, cam, match, blend=0.0, both=False):
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
    face's bow eases out along the block over `GAP_BOW_DECAY` of its widths.
    With `both`, the far side's nodes, as many, are the lines' ends, so it
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
    ends = np.array([np.interp(f_own / f_own[-1] * s_far[-1], s_far, c) for c in far])
    if both:
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
            x = np.clip(d / (blend * cam.s[-1]), 0.0, 1.0)
            return 1.0 - x * x * (3.0 - 2.0 * x)

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
        x = np.clip(dist / (GAP_BOW_DECAY * np.linalg.norm(P1 - P0)), 0.0, 1.0)
        H += (1.0 - x * x * (3.0 - 2.0 * x))[None, :, None] * bow[:, None, :]
    H[:, 0], H[:, -1] = front, rear
    return H, k_cut


def _winslow(B, n_sweep, periodic=False, relax=0.5):
    """Return block `B` (2, ni, nk) after `n_sweep` under-relaxed point-Jacobi
    sweeps of Winslow's equations on its inside, its faces fixed; `periodic`
    in `i`, its first and last lines one."""
    X = np.array(B, dtype=float)
    for _ in range(n_sweep):
        P = np.concatenate((X[:, -2:-1], X, X[:, 1:2]), axis=1) if periodic else X
        c = P[:, 1:-1, 1:-1]
        xi = 0.5 * (P[:, 2:, 1:-1] - P[:, :-2, 1:-1])
        eta = 0.5 * (P[:, 1:-1, 2:] - P[:, 1:-1, :-2])
        a = np.sum(eta**2, axis=0)
        b = np.sum(xi * eta, axis=0)
        g = np.sum(xi**2, axis=0)
        cross = P[:, 2:, 2:] - P[:, 2:, :-2] - P[:, :-2, 2:] + P[:, :-2, :-2]
        new = (
            a * (P[:, 2:, 1:-1] + P[:, :-2, 1:-1])
            + g * (P[:, 1:-1, 2:] + P[:, 1:-1, :-2])
            - 0.5 * b * cross
        ) / (2.0 * (a + g))
        new = c + relax * (new - c)
        if periodic:
            X[:, :, 1:-1] = new
            X[:, -1] = X[:, 0]
        else:
            X[:, 1:-1, 1:-1] = new
    return X


def _gap_O_lines(inner, W, square, n=129):
    """Return the lines (2, ni, n) of a tip gap's O block from its inner face
    `inner` to the blade's outline `W` (2, ni), sampled finely: cubics
    leaving the inner face straight towards the outline and arriving at it
    `square` of the way from straight to along its outward normal, but no
    further than `TIP_O_TURN` over the line's length times the outline's
    curvature there."""
    chord = W - inner
    L = np.linalg.norm(chord, axis=0)
    nrm = _outward_normals(W)
    # How fast the outline's normal turns along it, per unit length
    turn = np.arccos(np.clip(np.sum(nrm[:, 1:] * nrm[:, :-1], axis=0), -1.0, 1.0))
    ds = np.linalg.norm(np.diff(W, axis=1), axis=0)
    kappa = turn / np.maximum(ds, 1e-300)
    kappa = np.maximum(np.append(kappa, kappa[0]), np.insert(kappa, 0, kappa[-1]))
    w = np.minimum(square, TIP_O_TURN / np.maximum(L * kappa, 1e-300))
    # Eased round the blade, so neighbouring lines bend alike
    g = np.exp(
        -0.5 * (np.arange(-3 * TIP_O_EASE, 3 * TIP_O_EASE + 1) / TIP_O_EASE) ** 2
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


def _symmetric_halves(blade_wall, density):
    """Return the halves of the blade of `blade_wall`, as :meth:`_wall`
    returns it, from the middle of the trailing edge to the leading edge, at
    one density, `density` sampled at its `N_WALL` points.

    Returns `t`, fractions of the way along each half; `c`, cells from the
    trailing edge at `t`, by the greater of the two sides' densities there;
    and `s_lo`, `s_up`, where `t` lies on each side, as arc lengths round the
    blade from the middle of the trailing edge.
    """
    _, marks, _, s_LE = blade_wall
    S = marks[-1]
    s = np.linspace(0.0, S, N_WALL)
    t = np.linspace(0.0, 1.0, 4001)
    s_lo = t * s_LE
    s_up = S - t * (S - s_LE)
    d = np.maximum(
        np.interp(s_lo, s, density) * s_LE, np.interp(s_up, s, density) * (S - s_LE)
    )
    c = np.concatenate(([0.0], np.cumsum(0.5 * (d[1:] + d[:-1]) * np.diff(t))))
    return t, c, s_lo, s_up


def _symmetric_counts(blade_walls, densities, i_wall):
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
        raise ValueError(f"tip_match=3 needs the corners in order, not {i_wall}.")
    halves, ks = [], []
    for blade_wall, density in zip(blade_walls, densities):
        t, c, _, _ = _symmetric_halves(blade_wall, density)
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
            f"tip_match=3 puts a corner at the leading edge: {n_half} cells "
            f"each half, corners {k3}, {k4} from the trailing edge on one side "
            f"and {k2}, {k1} on the other."
        )
    return [k3, side, 2 * n_half - k4 - k1, side, k2]


def _symmetric_wall(blade_wall, density, counts):
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
        raise ValueError(f"tip_match=3 needs an even count round the blade, not {N}.")
    n_half = N // 2
    t, c, s_lo, s_up = _symmetric_halves(blade_wall, density)
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


def _coarse_density(wall, h_free, turn, ER):
    """Return nodes per unit length along a closed line (2, n), evenly spaced,
    for cells of `h_free` or less, across which it turns by no more than
    `turn` radians, growing from one to the next by `ER` at most."""
    d = np.gradient(wall, axis=1)
    phi = np.unwrap(np.arctan2(d[1], d[0]))
    ds = util.arc_length(wall) / (wall.shape[1] - 1)
    kappa = np.abs(np.gradient(phi)) / ds
    kappa = scipy.ndimage.uniform_filter1d(kappa, N_SMOOTH_TURN, mode="wrap")
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
def _spline_O(nc):
    """Return the samples (n, nc + 1), `N_SPLINE_O` per coarse cell, of the
    cubic spline through `nc` evenly spaced values with a given slope at the
    first and not-a-knot at the last, per unit of each value and the slope.

    The same at every plane, so each O block's lines are splined by one
    product with their coarse nodes and the slopes off the blade.
    """
    u = np.linspace(0.0, nc - 1.0, N_SPLINE_O * (nc - 1) + 1)
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
    bx, by = b[0], b[1]
    b1x, b1y = bx[1:] - bx[:-1], by[1:] - by[:-1]
    b2x, b2y = bx[:-1] - bx[1:], by[:-1] - by[1:]
    n_b1, n_b2 = norm(b1x, b1y), norm(b2x, b2y)

    def angles(a):
        ax, ay = a[0], a[1]
        u0x, u0y = bx[:-1] - ax[:-1], by[:-1] - ay[:-1]
        v0x, v0y = ax[1:] - ax[:-1], ay[1:] - ay[:-1]
        v1x, v1y = ax[:-1] - bx[:-1], ay[:-1] - by[:-1]
        u2x, u2y = ax[1:] - bx[1:], ay[1:] - by[1:]
        u3x, u3y = ax[:-1] - ax[1:], ay[:-1] - ay[1:]
        v3x, v3y = bx[1:] - ax[1:], by[1:] - ay[1:]
        out = corner(u0x, u0y, norm(u0x, u0y), v0x, v0y, norm(v0x, v0y))
        out = np.minimum(out, corner(b1x, b1y, n_b1, v1x, v1y, norm(v1x, v1y)))
        out = np.minimum(out, corner(u2x, u2y, norm(u2x, u2y), b2x, b2y, n_b2))
        out = np.minimum(
            out, corner(u3x, u3y, norm(u3x, u3y), v3x, v3y, norm(v3x, v3y))
        )
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


def _corners_at(corners, anchors, spf):
    """Return the corners at span fraction `spf`: piecewise linear between
    `anchors`, (spf, corners) pairs in rising span fraction, else `corners`."""
    if not anchors:
        return corners
    xp = [a for a, _ in anchors]
    fp = np.array([c for _, c in anchors])
    return tuple(float(np.interp(spf, xp, fp[:, k])) for k in range(4))


def _capped_corners(c0, c1, cap):
    """Return corners `c0` moved towards `c1`, both four fractions of the way
    round the blade, as far as leaves every one of the five pieces of the
    blade they and its ends bound within a factor `cap` of its length at
    `c0`."""

    def ends(c):
        return np.diff(np.concatenate(([0.0], c, [1.0])))

    L0, L1 = ends(np.asarray(c0, dtype=float)), ends(np.asarray(c1, dtype=float))
    dL = L1 - L0
    t = 1.0
    for l0, d in zip(L0, dL):
        # L0 + t dL within [l0 / cap, l0 * cap]
        if d > 0.0:
            t = min(t, (l0 * cap - l0) / d)
        elif d < 0.0:
            t = min(t, (l0 / cap - l0) / d)
    c = np.asarray(c0, dtype=float) + t * (
        np.asarray(c1, dtype=float) - np.asarray(c0, dtype=float)
    )
    return tuple(float(x) for x in c)


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
    s_curve = util.cum_arc_length(curve)
    return np.stack([np.interp(s, s_curve, c) for c in curve])


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


def _O_frame(plane, O):
    """Return the blade node, and the blade's unit tangent and outward normal
    there, (2, ni) each, at the blade end of each line of O block `O`."""
    wall = plane["wall"]
    n = _wall_normals(wall, _wall_arc(plane, O[:, :, 0]))
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


def _O_offsets(topo, X, plane):
    """Return the offsets (2, ni, nk) of the O block's nodes in plane `X`
    along and normal to the blade from their lines' blade nodes."""
    O = np.asarray(topo.scatter(X)[0], dtype=float)
    b, t, n = _O_frame(plane, O)
    d = O - b[:, :, None]
    return np.stack(
        (np.sum(d * t[:, :, None], axis=0), np.sum(d * n[:, :, None], axis=0))
    )


def _O_from_offsets(topo, X, plane, offsets):
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
    rebuilt shear past one another, cells under `O_CARRY_SOUND` or turned
    inside out, the nodes are moved as far from where `X` has them as their
    sound neighbours either side are, interpolated linearly round the blade,
    over the stretch and `O_CARRY_MARGIN` nodes either side of it. `X`, from
    the carried correction, does not fold there, and the offsets keep the
    rest of the layer as they made it.
    """
    O = np.asarray(topo.scatter(X)[0], dtype=float)
    b, t, n = _O_frame(plane, O)
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
    unsound = (_cell_angles(new) < O_CARRY_SOUND) | (
        area * np.sign(np.median(area)) <= 0.0
    )
    if unsound.any():
        ni, nk = new.shape[1:]
        bad = np.zeros((ni, nk), dtype=bool)
        for di, dk in ((0, 0), (1, 0), (0, 1), (1, 1)):
            bad[di : ni - 1 + di, dk : nk - 1 + dk] |= unsound
        wide = np.ones(2 * O_CARRY_MARGIN + 1)
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


def _wall_normals(wall, s):
    """Return the outward unit normals (2, n) of the blade, closed line
    `wall`, at arc lengths `s` along it, as :func:`_outward_normals` makes
    them."""
    S = util.arc_length(wall)
    eps = S / N_WALL
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


def _outward_normals(loop):
    """Return smoothed unit normals out of a closed clockwise loop (2, n)."""
    pts = loop[:, :-1]
    t = np.roll(pts, -1, axis=1) - np.roll(pts, 1, axis=1)
    t /= np.linalg.norm(t, axis=0)
    n = np.stack((-t[1], t[0]))
    for _ in range(N_SMOOTH_NORMAL):
        n = 0.25 * np.roll(n, 1, axis=1) + 0.5 * n + 0.25 * np.roll(n, -1, axis=1)
        n /= np.linalg.norm(n, axis=0)
    return np.concatenate((n, n[:, :1]), axis=1)


def _periodic_line(blade, cmap, pitch):
    """Return `theta(m')` of the initial periodic line above the blade.

    Half a pitch above the camber line, extended straight beyond it at the
    end slopes of the camber line, limited to `SLOPE_MAX_LE` ahead and
    `SLOPE_MAX` behind.
    """
    # The camber line along the chord and over each end evaluated together:
    # evaluated alone, an end would be stretched over the whole chord.
    m_line = util.cluster_cosine(400)
    m_ends = (
        np.linspace(0.0, END_FIT, N_END_FIT),
        np.linspace(1.0 - END_FIT, 1.0, N_END_FIT),
    )
    m_all, inverse = np.unique(np.concatenate((m_line, *m_ends)), return_inverse=True)
    cam = blade.evaluate_camber(cmap.spf, m=m_all)
    mp_all = cmap.mp_from_xr(cam[:2])
    i_line, i_LE, i_TE = np.split(inverse, np.cumsum([len(m_line), N_END_FIT]))
    mp, theta = mp_all[i_line], cam[2][i_line] + 0.5 * pitch
    i_TE = i_TE[::-1]
    slope = (
        np.clip(_end_slope(mp_all[i_LE], cam[2][i_LE]), -SLOPE_MAX_LE, SLOPE_MAX_LE),
        np.clip(_end_slope(mp_all[i_TE], cam[2][i_TE]), -SLOPE_MAX, SLOPE_MAX),
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


def _stitch_mixing_planes(rows, labels=LABELS, straight=False):
    """Force xr coordinates to match exactly at mixing planes, and with
    `straight`, the lines crossing each run straight through it, by
    :func:`_straighten_mixing`."""
    for i_row in range(len(rows) - 1):
        dn = rows[i_row][labels.index("dn")]
        up = rows[i_row + 1][labels.index("up")]
        x = 0.5 * (dn.x[-1, :, 0] + up.x[0, :, 0])[:, None]
        r = 0.5 * (dn.r[-1, :, 0] + up.r[0, :, 0])[:, None]
        dn[-1].set_x(x)
        up[0].set_x(x)
        dn[-1].set_r(r)
        up[0].set_r(r)
        if straight:
            _straighten_mixing(dn, up)


def _straighten_mixing(dn, up, n_blend=MIX_BLEND):
    """Make the lines along `i` of blocks `dn` and `up`, which meet at a
    mixing plane at the last `i` of `dn` and the first of `up`, run straight
    through it in (x, r) and square to it in theta, in place.

    Every line's cell at the plane is set to one length in (x, r) for each
    `j`, the mean of the cells there across the pitch, averaged over the two
    sides. Along the line, the cells are scaled by a factor easing as a
    half cosine from that at the plane to one over `n_blend` cells, those
    after the first rescaled to keep the length the `n_blend` cells span, and
    the nodes moved along the line, which lies on its plane's surface of
    revolution, by arc length in (x, r). The nodes one in from the plane then
    sit the same distance from it along that surface on both sides and at
    every `k`, but for the sag of the lines they move along, which setting
    them to their mean across the pitch takes out; each node on the plane is
    then put midway between them in
    (x, r), endwalls included, which takes the plane off the surface by the
    sag of the chord between them. Each node one in from the plane is also
    put at the theta of the node on it, so the cell there has no slope in
    theta, the shift eased away as a half cosine over the `n_blend` nodes in
    and the plane's nodes held.
    """
    sides = []
    for block, flip in ((dn, True), (up, False)):
        X = np.stack([np.asarray(getattr(block, c), dtype=float) for c in "xrt"])
        if flip:
            X = X[:, ::-1]
        n = min(n_blend, X.shape[1] - 2)
        c = np.linalg.norm(np.diff(X[:2, : n + 1], axis=1), axis=0)
        sides.append((block, flip, X, n, c))
    d = 0.5 * sum(c[0].mean(axis=-1) for *_, c in sides)[:, None]

    for block, flip, X, n, c in sides:
        w = 0.5 * (1.0 + np.cos(np.pi * np.arange(n) / n))
        new = c * (1.0 + (d / c[0] - 1.0) * w[:, None, None])
        spare = c.sum(axis=0) - new[0]
        if np.any(spare <= 0.0):
            raise ValueError(
                f"Block {block.label} is too short over {n} cells from its mixing "
                f"plane for an end cell of {d.max():.3g} m."
            )
        new[1:] *= spare / new[1:].sum(axis=0)
        s = np.zeros((n + 1,) + c.shape[1:])
        s_new = np.zeros_like(s)
        np.cumsum(c, axis=0, out=s[1:])
        np.cumsum(new, axis=0, out=s_new[1:])
        # Linear interpolation along every line at once, by the cell each new
        # node falls in
        at = np.clip((s[None] <= s_new[:, None]).sum(axis=1) - 1, 0, n - 1)
        s0 = np.take_along_axis(s, at, axis=0)
        s1 = np.take_along_axis(s, at + 1, axis=0)
        f = (s_new - s0) / (s1 - s0)
        X0 = np.take_along_axis(X[:, : n + 1], at[None].repeat(3, axis=0), axis=1)
        X1 = np.take_along_axis(X[:, : n + 1], at[None].repeat(3, axis=0) + 1, axis=1)
        Y = X.copy()
        Y[:, : n + 1] = X0 + f * (X1 - X0)
        # Level in theta with the plane over the cell at it, the shift that
        # takes eased away as a half cosine to none `n` nodes in, the plane
        # itself held
        w = np.zeros(n + 1)
        w[1:] = 0.5 * (1.0 + np.cos(np.pi * np.arange(n) / max(n - 1, 1)))
        Y[2, : n + 1] += (Y[2, 0] - Y[2, 1]) * w[:, None, None]
        if flip:
            Y = Y[:, ::-1]
        block.set_x(Y[0])
        block.set_r(Y[1])
        block.set_t(Y[2])

    # The nodes one in from the plane differ across the pitch by the sag of
    # the lines they were moved along, so set each side's to its pitch mean,
    # which puts every line through the plane exactly straight. Each side's
    # pitch counted apart, as the two need not have as many nodes.
    xr = {}
    for block, i in ((dn, -2), (up, 1)):
        xr[i] = (block.x[i].mean(axis=-1)[:, None], block.r[i].mean(axis=-1)[:, None])
        block[i].set_x(xr[i][0])
        block[i].set_r(xr[i][1])
    x = 0.5 * (xr[-2][0] + xr[1][0])
    r = 0.5 * (xr[-2][1] + xr[1][1])
    dn[-1].set_x(x)
    up[0].set_x(x)
    dn[-1].set_r(r)
    up[0].set_r(r)
