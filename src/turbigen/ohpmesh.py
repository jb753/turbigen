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
block, whose outer face is the O block's inner one over the gap, renoded and
joined to it by a non-matching interface, and an H block filling it from the
leading edge to the trailing edge. See :meth:`OHPassage._tip_gap`.
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
from turbigen.hmesh import _by_theta
from turbigen.mesh import Mesher

logger = logging.getLogger("turbigen")

LABELS = ("O", "P", "LE", "Pin", "TE", "Pout", "up", "dn")
"""Block names, in the order blocks are held."""

N_MERID = 4001
"""Samples along a span line from which its conformal coordinate is integrated."""

N_CHORD = 4000
"""Chordwise samples of a blade surface, which nodes are interpolated along."""

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

POLISH_COARSE_RINGS = 2
"""How many cells out from each poor cell of a swept plane are polished."""

HOLD_COARSE = 1
"""Layers of cells off the blade the polish of a swept plane leaves be."""

POLISH_FINE_RINGS = 3
"""How many cells out from each poor cell of the refined plane are polished."""

HOLD_FINE = 3
"""Layers of cells off the blade the polish of the refined plane leaves be."""

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

GAP_PAIR_MIX = 0.1
"""Share of the fraction of each side's length mixed into the chordwise
parameter the nodes either side of a tip gap are paired by, so that it keeps
rising where one side's parameter stands still."""

GAP_PAIR_ENDS = 0.03
"""Fraction of each side's length from either edge of a tip gap over which
its nodes are paired at the same fractions of the two sides' lengths,
blending to the pairing by chordwise parameter."""

GAP_FADE = 3.0
"""Distance back from the leading edge, in bulges of the front of a tip gap's
H block ahead of its first straight line, over which that bulge fades out."""

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


@dataclasses.dataclass(frozen=True)
class _Layout:
    """Counts of one row on the coarse mesh, and how it is refined, fixed at
    mid-span."""

    pitch: float
    m_in: float
    m_Zin: float
    m_Zout: float
    m_out: float
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

    dspf_mid: float = 0.03
    """Spanwise grid spacing at midspan, as a fraction of span."""

    ER_span: float = 1.2
    """Expansion ratio of spanwise grid away from hub and casing."""

    resolution_factor: float = 1.0
    """Multiply the number of points in each direction, keeping relative spacings."""

    n_span_smooth: int = 5
    """Number of spanwise planes to smooth, hub and casing included. The rest
    are interpolated between them. Zero smooths every plane."""

    ds: float = 0.02
    """Cell size along the blade between its edges, next to the O block and
    across the passage there, as a fraction of meridional chord."""

    ds_LE: float = 0.01
    """Cell size along the blade at the leading edge, as a fraction of
    meridional chord."""

    ds_TE: float = 0.002
    """Cell size along the blade at the middle of the trailing edge, as a
    fraction of meridional chord: a round trailing edge is much smaller than
    the leading edge, and too few nodes round it fan the O block's lines
    sharply off it."""

    ds_free: float = 0.05
    """Cell size in the free stream, as a fraction of meridional chord; the
    coarse mesh that is smoothed is `COARSE_FACTOR` times finer."""

    turn_max: float = 15.0
    """Most the blade turns across one cell of the coarse mesh, degrees."""

    ER_surf: float = 1.2
    """Expansion ratio along the blade away from its edges."""

    ER_stream: float = 1.2
    """Expansion ratio away from the O block, across the passage and along
    the blocks ahead of and behind the blade."""

    ER_O: float = 1.2
    """Expansion ratio off the blade in the O block."""

    n_sweep: int = 500
    """Smoothing sweeps on the coarse mesh."""

    t_O: float = 0.1
    """Thickness of the O block, as a fraction of pitch, before smoothing."""

    frac_Z: float = 0.5
    """Where the blocks ahead of and behind the blade end, as a fraction of
    the way from the blade to the mixing plane between rows."""

    cut: float = 0.5
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

    stagger_max: float = 30.0
    """Most the bounding box that places the corners is turned, degrees."""

    cor_fac: float = 0.75
    """How hard the nodes near each corner are drawn in towards it each
    sweep: lower draws them in harder."""

    n_corner: int = 16
    """Nodes along each side of the patch drawn in towards a corner."""

    polish_below: float = 32.0
    """Once swept, and again once refined, the nodes of each control plane
    near any cell whose worst corner angle, folded onto 0 to 90 degrees, is
    under this are polished: within a few cells of it, short of the O
    block's first layers. None if zero."""

    njtip_min: int = 9
    """Least number of spanwise nodes across a tip gap, one more than a
    multiple of eight."""

    nk_tip: int = 9
    """Least number of nodes across the H block of a tip gap, from one side
    of the blade to the other, one more than a multiple of eight. More if the
    blade has more nodes on one side than the other: see :meth:`_tip_gap`."""

    nk_tip_O: int = 9
    """Nodes across the O block of a tip gap, from the H block to the outline
    of the blade, one more than a multiple of eight."""

    t_tip_O: float = 0.25
    """Thickness of the O block of a tip gap before it is polished, as a
    fraction of the local thickness of the blade."""

    n_size_smooth_tip: int = 10
    """Passes of smoothing of the spacing the tip gap's polish aims for,
    along each grid line, which draws out the steps in spacing the
    algebraic plane is built with. See :func:`turbigen.elliptic.polish`."""

    n_polish_tip: int = 250
    """Iterations of the target-matrix optimisation that squares up the
    plane of a tip gap that is smoothed, from the algebraic plane."""

    def __post_init__(self):
        if min(self.ds, self.ds_LE, self.ds_TE, self.ds_free, self.t_O) <= 0.0:
            raise ValueError("ds, ds_LE, ds_TE, ds_free and t_O must be positive.")
        if not max(self.ds_LE, self.ds_TE) <= self.ds <= self.ds_free:
            raise ValueError(
                f"Need ds_LE and ds_TE <= ds <= ds_free, got {self.ds_LE}, "
                f"{self.ds_TE}, {self.ds}, {self.ds_free}."
            )
        if not 0.0 < self.frac_Z < 1.0:
            raise ValueError(f"frac_Z={self.frac_Z} must be between zero and one.")
        if not 0.0 <= self.spf_corners <= 1.0:
            raise ValueError(f"spf_corners={self.spf_corners} must be from 0 to 1.")
        if not 0.0 <= self.polish_below < 90.0:
            raise ValueError(
                f"polish_below={self.polish_below} must be from zero to 90 degrees."
            )
        for name in ("njtip_min", "nk_tip", "nk_tip_O"):
            n = getattr(self, name)
            if n < 9 or (n - 1) % 8:
                raise ValueError(
                    f"{name}={n} is not a count a multigrid level can halve; "
                    f"give it nine or more, one above a multiple of eight."
                )
        if not 0.0 < self.t_tip_O < 0.5:
            raise ValueError(f"t_tip_O={self.t_tip_O} must be between zero and 0.5.")

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

        tip = self._tip_fraction(machine)
        self._check_inviscid_ducts(machine.annulus)
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

        blocks = []
        for i_row, row in enumerate(machine.rows):
            j = jtip if row.tip_gap else None
            blocks.append(self.mesh_row(machine, i_row, spacing, spf, j))

        _stitch_mixing_planes([row[: len(self.labels)] for row in blocks], self.labels)
        self._relax_outlet([b for row in blocks for b in row], machine.annulus, spf)
        grid = ember.grid.Grid([b for row in blocks for b in row])
        grid.connectivity.periodic.pair()
        return grid

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
        dspf_tip = min(dspf_casing, tip / (self.njtip_min - 1))
        try:
            spf_tip = clusterfunc.double.free(
                dspf_tip, dspf_tip, 4.0 * dspf_tip, self.ER_span, L_main, 1.0
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

        The planes nearest even steps in span fraction, rather than even
        steps in index, which would follow the spanwise clustering and crowd
        them at the endwalls; always including the hub and the casing. Every
        plane if :attr:`n_span_smooth` is zero.
        """
        nj = len(spf)
        if not self.n_span_smooth or self.n_span_smooth >= nj:
            return np.arange(nj)
        target = np.linspace(spf[0], spf[-1], self.n_span_smooth)
        return np.unique(np.argmin(np.abs(spf[:, None] - target), axis=0))

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
        S = util.arc_length(wall)
        s_lo, s_hi = half, S - half
        # The leading edge where the two surfaces meet, at the start of the
        # camber line they hang off; not the furthest point upstream, which
        # on a staggered blade lies round the nose on one side.
        s_LE = (s_te[-1] - half) + util.arc_length(lower)
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

    def layout(self, machine, i_row, spacing):
        """Fix the coarse counts of one row, and how it is refined, at mid-span."""
        row = machine.rows[i_row]
        blade = row.blade
        n_row = len(machine.rows)
        pitch = 2.0 * np.pi / row.n_blade

        L_up = 1.0 if i_row == 0 else 0.5
        L_dn = 1.0 if i_row == n_row - 1 else 0.5
        m_LE, m_TE = 2.0 * i_row + 1.0, 2.0 * i_row + 2.0
        m_in, m_out = m_LE - L_up, m_TE + L_dn
        m_Zin = m_LE - self.frac_Z * L_up
        m_Zout = m_TE + self.frac_Z * L_dn
        extrude_in, extrude_out = i_row == 0, i_row == n_row - 1
        chord = blade.evaluate_chord(0.5)
        if extrude_in:
            m_Zin = _m_at_distance(machine.annulus, m_LE, m_in, self.cut * chord)
        if extrude_out:
            m_Zout = _m_at_distance(machine.annulus, m_TE, m_out, self.cut * chord)

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
        cmap_C = ConformalMap(machine.annulus, self.spf_corners, m_in, m_out)
        _, marks_C, _, _ = self._wall(blade, cmap_C, t_O, pitch=pitch)
        corners = tuple(float(x) for x in marks_C[1:5] / marks_C[-1])
        blade_wall = self._wall(blade, cmap, t_O, corners)
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

        dw = float(spacing.surface[i_row]) / r_ref
        f_O = clusterfunc.single.free(dw / t_O, max(h, dw) / t_O, self.ER_O)

        layout = _Layout(
            pitch=pitch,
            m_in=m_in,
            m_Zin=m_Zin,
            m_Zout=m_Zout,
            m_out=m_out,
            d_wall=float(spacing.surface[i_row]),
            t_O=t_O,
            h=h,
            n_wall=tuple(int(x) for x in n),
            f_wall=f_wall,
            corners=corners,
            i_wall=i_C,
            n_O=max(2, round(2.0 * t_O / Hc)),
            n_P=n_P,
            n_in=n_in,
            n_out=n_out,
            n_up=1 if extrude_in else _up8(round((mp_Zin - mp_in) * stretch_in / Hc)),
            n_dn=1
            if extrude_out
            else _up8(round((mp_out - mp_Zout) * stretch_out / Hc)),
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
            f"{len(layout.u['out']) - 1}, O {layout.nk_O} nodes"
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

    def plane(self, machine, i_row, layout, cmap, blade_wall, density, section):
        """Return the coarse algebraic mesh of the constant-span plane of
        `cmap`, its blade as :meth:`_wall` returns it and the density of
        coarse cells along it."""
        blade = machine.rows[i_row].blade
        spf = cmap.spf
        pitch = layout.pitch
        pJ = pitch * _J
        mp_in, mp_Zin, mp_Zout, mp_out = cmap.mp_from_m(
            [layout.m_in, layout.m_Zin, layout.m_Zout, layout.m_out]
        )
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
            "s_loop": g["s_loop"],
            "sigma_LE": g["sigma_LE"],
            "chord_frac": _chord(blade, spf, section) / util.arc_length(g["wall"]),
        }

    #
    # SMOOTHING
    #

    def _sweep_operators(self, topo):
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
        topo.sweep_corner = self._corner_patches(topo)
        return topo

    def _corner_patches(self, topo):
        """Return the nodes drawn in towards each corner, and how far.

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
                        f = np.tanh(np.arange(1, m + 1) * self.cor_fac)
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

    def _sweeps(self, topo, X, n_sweep, plane):
        """Return `X` after `n_sweep` point-Jacobi sweeps of Winslow's equations.

        By :func:`ember.elliptic.winslow_sweeps`: each sweep the regular and
        singular nodes from their stencils, the inlet and outlet level with
        the nodes inside, the nodes near the corners drawn in, the blade
        nodes of `plane` slid along the blade towards the foot of the
        perpendicular from the node outside them, by :meth:`_slide_factors`,
        and the nodes outside them drawn square to the blade by `ORTH`.
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
            self._slide_factors(plane, line, s_line, tree, w0),
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
        lay = self.layout(machine, i_row, spacing)
        # Each plane's blade evaluated once, for its coarse counts and nodes
        blade = machine.rows[i_row].blade
        cmaps = [ConformalMap(machine.annulus, s, lay.m_in, lay.m_out) for s in spf]
        sections = [_section(blade, s) for s in spf]
        walls = [
            self._wall(blade, c, lay.t_O, lay.corners, section=sec)
            for c, sec in zip(cmaps, sections)
        ]
        dens = [self._wall_coarse_density(w, lay) for w in walls]
        lay = self._coarse_counts(i_row, lay, walls, dens)
        planes = [
            self.plane(machine, i_row, lay, *cwds)
            for cwds in zip(cmaps, walls, dens, sections)
        ]
        j_mid = int(np.argmin(np.abs(spf - 0.5)))

        topo = self._sweep_operators(_plane_topology(planes[j_mid]["xy"], lay.pitch))
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
        for j in j_ctrl:
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
            own, _ = self._match_blade(topo, X, planes[j], lay, None)
            fracs[j] = np.concatenate(own)
            correction[j] = X - X_alg[j]
            logger.debug(f"Row {i_row} spf={spf[j]:.3f} swept")

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

        correction = between(correction)
        X_smooth = [X_alg[j] + correction[j] for j in range(len(spf))]
        # A correction carried between planes leaves their blade nodes off
        # the blade, and round a thin edge out of order along it: put back
        # on it at fractions of each piece between those of the control
        # planes either side.
        fracs = between(fracs)
        cuts = np.cumsum([n + 1 for n in lay.n_wall])[:-1]
        X_smooth = [
            X
            if j in j_ctrl
            else self._match_blade(topo, X, planes[j], lay, np.split(fracs[j], cuts))[1]
            for j, X in enumerate(X_smooth)
        ]
        # The O block of each plane between rebuilt from its lines' offsets
        # along and normal to the blade from their blade nodes, interpolated
        # between the control planes: displacements interpolated in the
        # plane would skew lines that leave each control plane's blade square
        # where the blade between turns, as it does where its stagger changes.
        offsets = [None] * len(spf)
        for j in j_ctrl:
            offsets[j] = _O_offsets(topo, X_alg[j] + correction[j], planes[j])
        offsets = between(offsets)
        X_smooth = [
            X if j in j_ctrl else _O_from_offsets(topo, X, planes[j], offsets[j])
            for j, X in enumerate(X_smooth)
        ]

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
        per_plane = [
            self._refined(topo.scatter(X), plane, lay, refine)
            for X, plane in zip(X_smooth, planes)
        ]

        # The control planes are polished once refined, and how far that
        # moves each node carried to the planes between, as the smoothing is.
        if self.polish_below:
            fine = _plane_topology(per_plane[j_mid], lay.pitch, stencils=False)
            hold = fine.gid[0][:, : HOLD_FINE + 1].ravel()
            shift = [None] * len(spf)
            for j in j_ctrl:
                shift[j] = self._polish_shift(
                    fine, hold, per_plane[j], POLISH_FINE_RINGS
                )
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
        """
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
        the other.
        """
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
            "dn": (None, across + shifted(rear, n_P)),
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

    def _refined(self, xy, plane, lay, refine):
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
        dtype = O.dtype
        # Each line a cubic spline through its coarse nodes, leaving the
        # blade along its normal, sampled finely, so the nodes respaced along
        # it leave the blade square and do not cut the corners where the
        # lines between coarse nodes bend.
        O = O.astype(float)
        speed = np.linalg.norm(O[:, :, 1] - O[:, :, 0], axis=0)
        start = _wall_normals(wall, s_f) * speed
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
        whose outer face is the O block round the blade's inner one, and an H
        block filling it, running `i` from the leading edge to the trailing
        edge and `k` from the lower side of the blade to the upper. The
        corners of the H block are where its edges meet the O block's inner
        face, at `i` of it `iA` and `iB` along the lower side, `iC` and `iD`
        along the upper.

        The O block round the blade has unequal counts either side of the
        blade, not level chordwise, so the gap's outline is that face
        renoded, half its nodes along each side, those of one index either
        side across the blade from each other, and joined to it by a
        non-matching interface; the H block then runs straight across. See
        :func:`_gap_fractions`. The plane at the tip of the blade
        is built algebraically, which folds round a strongly curved blade,
        its O block's lines respaced to the wall spacing at the outline, and
        the whole plane polished, which unfolds it: polishing only its poor
        cells, even again and again, leaves it worse in its worst cell. It is
        not swept by Winslow's equations first, as a main plane is: at the
        gap's final resolution, its cells long and thin, the sweeps set the
        lines zigzagging from node to node round the H block's corners and
        fold the O block's cells there, and the polish then needs four times
        the iterations to undo that and still ends worse in its worst cell.
        That plane is carried to each above it by
        moving its outline to theirs and the nodes inside by a discrete
        Laplacian, then respacing. Carried instead as a correction to each
        plane's algebraic one, it folds near the casing where the blade
        twists, the algebraic planes' folds not being alike.

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

        """
        walls = [np.asarray(xy[0][:, :, 0], dtype=float) for xy in per_plane]
        cmap = planes[0]["cmap"]
        upper, lower = _surfaces(blade, cmap, cmap.spf)
        # The leading edge node, nearest where the two surfaces meet.
        i_LE = int(np.argmin(np.linalg.norm(walls[0] - upper[:, :1], axis=0)))
        frac = _gap_fractions(walls[0], lower, upper, i_LE)
        walls = [_gap_outline(W, frac, i_LE) for W in walls]
        corners = _gap_corners(walls[0], self.nk_tip)
        alg = [_gap_plane(W, corners, self.t_tip_O, self.nk_tip_O) for W in walls]

        fixed = [np.zeros(B.shape[1:], dtype=bool) for B in alg[0]]
        fixed[0][:, -1] = True
        topo = elliptic.PlaneTopology(alg[0], fixed=fixed, stencils=False)
        d_wall = [
            layout.d_wall / plane["cmap"].xr_from_mp(W[0])[1]
            for W, plane in zip(walls, planes)
        ]

        def respaced(X, d):
            tO, tH = topo.scatter(X)
            return topo.gather([_respace_gap_O(tO, d), tH])

        # Respaced before the polish, which keeps the sizes it finds:
        # respaced only after, each line on its own, the lines either side of
        # the H block's corners, whose lengths differ sharply, skew the cells
        # between them. Respaced again after, as the polish moves the cells
        # at the outline a little.
        X = respaced(topo.gather(alg[0]), d_wall[0])
        X = elliptic.polish(
            topo,
            X,
            np.arange(topo.n_node),
            rings=0,
            max_iter=self.n_polish_tip,
            power=POLISH_POWER,
            smooth=self.n_size_smooth_tip,
        )
        X = respaced(X, d_wall[0])

        # The nodes inside the outline moved by a discrete Laplacian of the
        # outline's move, over the edges of the cells.
        n = topo.n_node
        edges = np.concatenate(
            [
                np.stack((a.ravel(), b.ravel()), axis=1)
                for g in topo.gid
                for a, b in ((g[:-1], g[1:]), (g[:, :-1], g[:, 1:]))
            ]
        )
        edges = np.unique(np.sort(edges, axis=1), axis=0)
        edges = edges[edges[:, 0] != edges[:, 1]]
        adj = scipy.sparse.coo_matrix(
            (np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n)
        )
        adj = (adj + adj.T).tocsr()
        lap = scipy.sparse.diags(np.asarray(adj.sum(axis=1)).ravel()) - adj
        edge = np.unique(topo.gid[0][:, -1])
        inside = np.setdiff1d(np.arange(n), edge)
        solve = scipy.sparse.linalg.factorized(lap[inside][:, inside].tocsc())
        lap_edge = lap[inside][:, edge]

        gap = []
        for j, (xy, W) in enumerate(zip(alg, walls)):
            move = np.zeros_like(X)
            move[edge] = topo.gather(xy)[edge] - X[edge]
            for c in range(2):
                move[inside, c] = solve(-(lap_edge @ move[edge, c]))
            tO, tH = topo.scatter(X + move)
            tO = _respace_gap_O(tO, d_wall[j])
            tO = tO.astype(DTYPE)
            tO[:, :, -1] = W
            gap.append([tO, tH.astype(DTYPE)])
            for B in gap[-1]:
                assert np.isfinite(B).all(), "non-finite coordinates in tip gap"

        q_tip = min(_cell_angles(B).min() for xy in gap for B in xy)
        logger.debug(
            f"Tip gap: O {gap[0][0].shape[1:]}, H {gap[0][1].shape[1:]}, "
            f"corners {corners}, worst cell corner {q_tip:.1f} deg"
        )
        return gap, corners

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
        # shares with the O block round the blade, noded differently.
        NonMatch = ember.patch.NonMatchPatch
        O.patches.append(NonMatch(k=0, j=(jtip, -1), label="tip_gap"))
        gap, (iA, iB, iC, iD) = self._tip_gap(
            machine.rows[i_row].blade, layout, planes[jtip:], per_plane[jtip:]
        )
        tO, tH = self.blocks(machine, i_row, planes[jtip:], gap, ("tipO", "tipH"))
        N = tO.shape[0] - 1
        tO.patches.extend(
            [
                Per(i=0, label="tip_cut0"),
                Per(i=-1, label="tip_cut1"),
                NonMatch(k=-1, label="tip_outline"),
                Per(i=(iA, iB), k=0, label="tip_lower"),
                Per(i=(iB, iC), k=0, label="tip_front"),
                Per(i=(iC, iD), k=0, label="tip_upper"),
            ]
        )
        # The rear of the H block runs round the trailing edge, across the cut.
        if iD < N:
            tO.patches.append(Per(i=(iD, -1), k=0, label="tip_rear_upper"))
        if iA > 0:
            tO.patches.append(Per(i=(0, iA), k=0, label="tip_rear_lower"))
        tH.patches.extend([Per(k=0), Per(k=-1), Per(i=0)])
        if 0 < iA < tH.shape[2] - 1:
            tH.patches.extend([Per(i=-1, k=(0, iA)), Per(i=-1, k=(iA, -1))])
        else:
            tH.patches.append(Per(i=-1))

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


def _plane_topology(xy, pitch, stencils=True):
    """Return the topology of a plane of a row's main blocks `xy`, coarse or
    refined: the O block's blade nodes fixed, the inlet and outlet sliding;
    with the sweeps' `stencils`, or only what the polish needs."""
    shapes = [B.shape[1:] for B in xy]
    fixed = [np.zeros(sh, dtype=bool) for sh in shapes]
    slide = [np.zeros(sh, dtype=bool) for sh in shapes]
    fixed[0][:, 0] = True
    slide[6][0, :] = True
    slide[7][-1, :] = True
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


def _gap_corners(W, nk_tip):
    """Return where the H block of a tip gap has its corners round outline `W`.

    `W` (2, N + 1) is the outline of the blade, clockwise from the middle of
    the trailing edge, and `N` a multiple of sixteen. The H block takes
    `a` cells along each side and `b` across each end, `2 (a + b) = N`. Its
    front runs round the leading edge, `f` cells of it on the lower side,
    and its rear round the trailing edge, `r` of it on the lower side, so
    that `f + r` is `b` plus half the difference between the lower and upper
    sides' counts; the end is lengthened until both fit within it.
    """
    N = W.shape[1] - 1
    i_LE = N // 2
    D = i_LE - N // 2
    b = max(nk_tip - 1, _up8(abs(D)))
    a = N // 2 - b
    if a < 8:
        raise ValueError(
            f"The blade has too few nodes round it, {N}, for a tip gap's H block "
            f"with {b} cells across: give it more, or lower nk_tip."
        )
    f = (b + D) // 2
    r = b + D - f
    iB = i_LE - f
    return (r, iB, iB + b, N - b + r)


def _gap_fractions(W, lower, upper, i_LE):
    """Return where the outline of a tip gap has its nodes along each side.

    `W` (2, N + 1) is the outline of the blade, clockwise from the middle of
    the trailing edge, and `N` a multiple of sixteen; `lower` and `upper`
    (2, N_CHORD) the blade's surfaces at the chordwise parameters `m` the
    blade hangs them off its camber line at. Fractions (2, N / 2 + 1) of the
    lower and upper sides' lengths from the leading edge, at which the nodes
    of one index either side lie at the same chordwise parameter, each side's
    rescaled to run from zero to one along it and mixed with `GAP_PAIR_MIX`
    of the fraction of its length, so the line between them crosses the
    blade about square to its camber; at the leading and trailing edges, at
    the same fractions of the two sides' lengths. The parameters are spaced as the mean
    of the two sides' own spacings, so the outline keeps the clustering it
    had. At the same fraction of each side's length instead, nodes on a
    highly cambered blade lie well apart chordwise, its suction side the
    longer, and the H block's lines across the gap lean by up to sixty
    degrees from one to the other. Paired across the camber line by
    projection onto it instead, the projections fold back on the concave
    side near a thick leading edge.
    """
    N = W.shape[1] - 1
    sides = (W[:, i_LE::-1], W[:, i_LE:])
    m = util.cluster_cosine(N_CHORD)
    # Spaced along each side in proportion to the blade's local thickness,
    # so that the H block's ends, a fixed count of cells, span the leading
    # and trailing edges alike, whatever the main O block's spacing round
    # them: kept as it is, a leading edge spaced five times as coarsely as
    # the trailing edge wrapped the front far round the nose and folded it.
    t = np.log(2.0 * _inscribed_radius(W))
    for _ in range(N_SMOOTH_NORMAL):
        t = 0.25 * np.roll(t, 1) + 0.5 * t + 0.25 * np.roll(t, -1)
    t = np.exp(np.append(t, t[0]))
    f0 = []
    for side, ts in zip(sides, (t[i_LE::-1], t[i_LE:])):
        sa = util.cum_arc_length(side)
        c = util.cumtrapz0(1.0 / ts, sa)
        f0.append(np.interp(np.linspace(0.0, c[-1], N // 2 + 1), c, sa / sa[-1]))
    f0 = np.mean(f0, axis=0)
    curves = []
    for side, surf in zip(sides, (lower, upper)):
        f = np.linspace(0.0, 1.0, N_WALL)
        pts = _at_arc(side, f * util.arc_length(side))
        t = np.maximum.accumulate(_project(surf, m, cKDTree(surf.T), pts.T))
        t = (t - t[0]) / (t[-1] - t[0])
        curves.append((f, (1.0 - GAP_PAIR_MIX) * t + GAP_PAIR_MIX * f))
    t_pair = np.mean([np.interp(f0, f, t) for f, t in curves], axis=0)
    paired = np.stack([np.interp(t_pair, t, f) for f, t in curves])
    # Round the nose one side's parameter stands still, and round the
    # trailing edge both, and the nodes there would lie far apart; at the
    # same fractions of the two sides instead, blending to the pairs over
    # `GAP_PAIR_ENDS` of each side's length.
    x = np.clip(np.minimum(f0, 1.0 - f0) / GAP_PAIR_ENDS, 0.0, 1.0)
    return f0 + x * x * (3.0 - 2.0 * x) * (paired - f0)


def _gap_outline(W, frac, i_LE):
    """Return the outline `W` (2, N + 1) of a tip gap renoded at `frac`.

    The lower and upper sides at fractions `frac` (2, N / 2 + 1) of their
    lengths from the leading edge, the leading edge at node `N / 2`, so that
    nodes of one index along either side lie across the blade from each
    other.
    """
    lower, upper = (
        _at_arc(side, f * util.arc_length(side))
        for side, f in zip((W[:, i_LE::-1], W[:, i_LE:]), frac)
    )
    out = np.concatenate((lower[:, ::-1], upper[:, 1:]), axis=1)
    out[:, 0], out[:, -1] = W[:, 0], W[:, -1]
    return out


def _inscribed_radius(W):
    """Return the radius of the largest circle inside closed outline `W`
    (2, N + 1) that touches it at each of its first `N` nodes."""
    n_in = -_outward_normals(W)
    P, d = W[:, :-1], n_in[:, :-1]
    # The circle tangent at node a through node b has radius
    # |b - a|^2 / (2 (b - a) . n_a); the largest inside is the least of
    # these over the nodes on its inward side.
    rel = P[:, None, :] - P[:, :, None]
    along = np.einsum("cab,ca->ab", rel, d)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(along > 0.0, np.sum(rel**2, axis=0) / (2.0 * along), np.inf)
    return r.min(axis=1)


def _gap_inset(W, frac):
    """Return the outline `W` (2, N + 1) moved inwards by `frac` of its thickness.

    Along the inward normals, by `frac` of the diameter of the largest circle
    inside the outline that touches it at each node: the local thickness of
    the blade, across it on the sides, round the edges. Measured along the
    normal instead, the distance to the far side runs long wherever the
    normal leaves the blade aslant, as it does behind a cambered leading
    edge, and the two sides of the inset outline nearly meet there. With
    `frac` under a half, no inset node reaches its circle's far side, nor
    further than its radius, no more than the radius of curvature, so the
    normals do not cross before it either. Rounded down where it rises
    sharply between neighbours.
    """
    n_in = -_outward_normals(W)
    r = _inscribed_radius(W)
    depth = frac * 2.0 * r
    for _ in range(N_SMOOTH_NORMAL):
        depth = np.minimum(
            depth,
            0.25 * np.roll(depth, 1) + 0.5 * depth + 0.25 * np.roll(depth, -1),
        )
    depth = np.append(depth, depth[0])
    inner = W + depth * n_in
    inner[:, -1] = inner[:, 0]
    return inner


def _gap_plane(W, corners, frac, nk_O):
    """Return the algebraic O and H blocks (2, ni, nk) of a tip gap in `W`.

    The O block from the H block at `k = 0` to the outline at `k = -1`,
    evenly across; the H block of straight lines across the blade, from each
    node of the lower side of the face the corners cut its inner face into
    to the node of the upper side as far along, between its ends round the
    leading and trailing edges. Not by transfinite interpolation, which
    blends an end's bulge round the leading edge in by the fraction of the
    way along the sides, so the many lines clustered towards the leading
    edge all take nearly the whole of it, and cross the sides there.
    """
    iA, iB, iC, iD = corners
    N = W.shape[1] - 1
    inner = _gap_inset(W, frac)
    lower = inner[:, iA : iB + 1][:, ::-1]
    upper = inner[:, iC : iD + 1]
    front = inner[:, iB : iC + 1]
    rear = np.concatenate((inner[:, iA::-1], inner[:, N - 1 : iD - 1 : -1]), axis=1)
    # Spaced evenly across. Spaced as the ends are, the leading edge's
    # spacing round the nose, uneven where the corners sit apart round it,
    # carries down the whole block.
    v = np.linspace(0.0, 1.0, front.shape[1])[None, None]
    H = (1.0 - v) * lower[:, :, None] + v * upper[:, :, None]
    # The front bulges ahead of the first straight line round the leading
    # edge; its bulge carried into the lines behind, fading by the square of
    # the distance back over `GAP_FADE` of the bulge, so the first cells do
    # not step from long to short. Faded by distance rather than by
    # fraction along the sides, which the lines crowd towards the leading
    # edge, so that many would take nearly the whole bulge and cross.
    bulge = front - H[:, 0, :]
    d = 0.5 * (util.cum_arc_length(lower) + util.cum_arc_length(upper))
    reach = GAP_FADE * np.linalg.norm(bulge, axis=0).max()
    if reach > 0.0:
        w = (1.0 - np.minimum(d / reach, 1.0)) ** 2
        H += w[None, :, None] * bulge[:, None, :]
    H[:, 0, :], H[:, -1, :] = front, rear
    f = np.linspace(0.0, 1.0, nk_O)[None, None]
    O = inner[..., None] + f * (W - inner)[..., None]
    return [O, H]


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


def _respace_gap_O(O, d_wall):
    """Return the O block (2, ni, nk) of a tip gap, each line respaced.

    Along itself from the H block, where its cells are as long as even
    spacing would make them, to the outline, where they are `d_wall` (ni,),
    or as long as even spacing would make them if that is less.
    """
    nk = O.shape[2]
    seg = np.linalg.norm(np.diff(O, axis=2), axis=0)
    s = np.concatenate((np.zeros((len(seg), 1)), np.cumsum(seg, axis=1)), axis=1)
    L = s[:, -1]
    even = 1.0 / (nk - 1)
    T = L[:, None] * clusterfunc.double.fixed_many(
        np.full(len(L), even), np.minimum(d_wall / L, even), nk
    )
    out = np.empty_like(O)
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
    upper, lower = blade._evaluate_section_m(spf, util.cluster_cosine(N_CHORD))
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
    m = util.cluster_cosine(N_CHORD)
    m_upper, m_lower = caps if blade.suction_is_upper else caps[::-1]
    i_u, i_l = (int(np.argmin(np.abs(m - m_cap))) for m_cap in (m_upper, m_lower))
    te = np.concatenate((upper[:, i_u:], lower[:, i_l:][:, ::-1][:, 1:]), axis=1)
    return upper[:, : i_u + 1], lower[:, : i_l + 1], te


def _at_arc(curve, s):
    """Return points on a polyline (2, n) at arc lengths `s` from its start."""
    s_curve = util.cum_arc_length(curve)
    return np.stack([np.interp(s, s_curve, c) for c in curve])


def _m_at_distance(annulus, m0, m1, dist):
    """Return the annulus meridional coordinate `dist` along mid-span from
    `m0` towards `m1`."""
    m = np.linspace(m0, m1, 2001)
    s = util.cum_arc_length(annulus.evaluate_xr(m, 0.5))
    if dist >= s[-1]:
        raise ValueError(
            f"The cut, {dist:.3g} m from the blade, lies beyond the inlet or "
            f"outlet, {s[-1]:.3g} m from it; make the cut smaller."
        )
    return float(np.interp(dist, s, m))


def _O_frame(plane, O):
    """Return the blade node, and the blade's unit tangent and outward normal
    there, (2, ni) each, at the blade end of each line of O block `O`."""
    wall = plane["wall"]
    n = _wall_normals(wall, _wall_arc(plane, O[:, :, 0]))
    return O[:, :, 0], np.stack((n[1], -n[0])), n


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
    their lines' blade nodes."""
    O = np.asarray(topo.scatter(X)[0], dtype=float)
    b, t, n = _O_frame(plane, O)
    new = (
        b[:, :, None]
        + offsets[0][None] * t[:, :, None]
        + offsets[1][None] * n[:, :, None]
    )
    g = topo.gid[0][:, 1:-1]
    pts = new[:, :, 1:-1]
    if topo.pitch:
        pts = pts - np.stack(
            (np.zeros_like(pts[0]), topo.shift[0][:, 1:-1] * topo.pitch)
        )
    X = np.array(X)
    X[g.ravel()] = np.moveaxis(pts, 0, -1).reshape(-1, 2).astype(X.dtype)
    return X


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
