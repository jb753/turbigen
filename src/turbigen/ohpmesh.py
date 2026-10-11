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
"""Largest `dtheta/dm'` the initial periodic line is extended at beyond the
blade, being the tangent of 45 degrees. Only a starting guess: the line floats."""

POLISH_POWER = 4.0
"""Power the polish raises each cell corner's shape measure to, so that the
worst cells dominate rather than being traded for small gains elsewhere."""

N_MATCH_SPAN = 3
"""Corrections of the main span's end spacing to meet a tip gap's."""

N_MATCH_REAR = 4
"""Corrections of the blade's spacing at the corners behind it, to match
the cells across the passage there."""

N_WALL = 20001
"""Samples along the blade, which nodes on it are placed on."""

CORNER_COS, CORNER_SIN = np.cos(np.pi / 3.0), np.sin(np.pi / 3.0)
"""Direction of an edge out of a corner of the O block, against the O block's
face and its outward normal there."""

POLISH_FINE_RINGS = 3
"""How many cells out from each poor cell of the refined plane are polished."""

HOLD_FINE = 3
"""Layers of cells off the blade the polish of the refined plane leaves be."""

DTYPE = np.float32
"""Precision the planes are held, swept, refined and polished in: the
finished grid is stored in single precision, and the sweeps run faster in
it. Only their geometry is worked out in double before."""

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
        self.m = np.linspace(m_lo, m_hi, N_MERID)
        self.xr = annulus.evaluate_xr(self.m, spf)
        ds = np.linalg.norm(np.diff(self.xr, axis=1), axis=0)
        r_mid = 0.5 * (self.xr[1, 1:] + self.xr[1, :-1])
        self.mp = np.concatenate(([0.0], np.cumsum(ds / r_mid)))
        self._tree = cKDTree(self.xr.T)

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

    def mp_from_xr(self, xr):
        """Return the conformal coordinate of points (2, n) on the surface.

        Each point is projected onto the nearer of the two sampled segments
        either side of its nearest sample.
        """
        pts = np.reshape(xr, (2, -1)).T
        _, near = self._tree.query(pts)
        n = len(self.m)
        best_mp = np.zeros(len(pts))
        best_d = np.full(len(pts), np.inf)
        for seg in (near - 1, near):
            seg = np.clip(seg, 0, n - 2)
            a = self.xr[:, seg].T
            ab = self.xr[:, seg + 1].T - a
            t = np.clip(np.sum((pts - a) * ab, 1) / np.sum(ab**2, 1), 0.0, 1.0)
            d = np.linalg.norm(a + t[:, None] * ab - pts, axis=1)
            better = d < best_d
            best_d[better] = d[better]
            mp = self.mp[seg] + t * (self.mp[seg + 1] - self.mp[seg])
            best_mp[better] = mp[better]
        return best_mp.reshape(np.shape(xr)[1:])


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
    """Coarse cells across the O block."""
    n_P: int
    """Coarse cells across the passage block."""
    n_in: int
    n_out: int
    n_up: int
    n_dn: int
    """Coarse cells along the blocks ahead of and behind the blade, and the
    inlet and outlet blocks."""
    nk_O: int
    """Nodes across the O block once clustered to the blade."""
    wall_density: tuple
    """Refined cells per unit length round the blade, as fractions of its
    length from the middle of the trailing edge, and the density there,
    for the O block's nodes on the blade."""
    u: dict
    """Where each refined node lies, in fractional coarse indices from the
    start of its stretch: along the rear of the lower and upper sides of the
    blade from the cut, `rl` and `ru`; along the lower and upper sides
    between corners, upstream to downstream, `side_lo` and `side_up`; round the leading edge, `front`; across
    the passage, `P`; and along the blocks ahead of and behind the blade,
    `in` and `out`."""

    def i_corners_coarse(self):
        """O block `i` of the corners C3, C4, C1, C2 on the coarse mesh."""
        ends = np.cumsum((0, *self.n_wall))
        return tuple(int(ends[i]) for i in self.i_wall)

    def i_corners(self):
        """O block `i` of the corners C3, C4, C1, C2 once refined."""
        n = [len(self.u[k]) - 1 for k in ("rl", "side_lo", "front", "side_up")]
        i3 = n[0]
        return i3, i3 + n[1], i3 + n[1] + n[2], i3 + n[1] + n[2] + n[3]


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

    n_span_smooth: int = 9
    """Number of spanwise planes to smooth, hub and casing included. The rest
    are interpolated between them. Zero smooths every plane."""

    ds: float = 0.01
    """Cell size along the blade between its edges, next to the O block and
    across the passage there, as a fraction of meridional chord."""

    ds_LE: float = 0.002
    """Cell size along the blade at the leading and trailing edges, as a
    fraction of meridional chord."""

    ds_free: float = 0.03
    """Cell size in the free stream, and of the coarse mesh that is smoothed,
    as a fraction of meridional chord."""

    turn_max: float = 15.0
    """Most the blade turns across one cell of the coarse mesh, degrees."""

    ER_surf: float = 1.15
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
    the way from the blade to the inlet, outlet or mixing plane."""

    stagger_max: float = 30.0
    """Most the bounding box that places the corners is turned, degrees."""

    cor_fac: float = 0.75
    """How hard the nodes near each corner are drawn in towards it each
    sweep: lower draws them in harder."""

    n_corner: int = 16
    """Nodes along each side of the patch drawn in towards a corner."""

    polish_rings: int = 0
    """How many coarse cells out from each corner of the O block are polished
    once the sweeps are done, by local target-matrix optimisation against the
    worst cells there, short of the inner half of the O block. None if zero,
    the default: the polish of the refined plane does better, and sooner."""

    polish_below: float = 32.0
    """Once refined, the nodes near any cell whose worst corner angle, folded
    onto 0 to 90 degrees, is under this are polished: within three cells of
    it, short of the O block's first layers. None if zero."""

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

    n_polish_tip: int = 1000
    """Iterations of the target-matrix optimisation that squares up the
    plane of a tip gap that is smoothed."""

    def __post_init__(self):
        if min(self.ds, self.ds_LE, self.ds_free, self.t_O) <= 0.0:
            raise ValueError("ds, ds_LE, ds_free and t_O must be positive.")
        if not self.ds_LE <= self.ds <= self.ds_free:
            raise ValueError(
                f"Need ds_LE <= ds <= ds_free, got {self.ds_LE}, {self.ds}, "
                f"{self.ds_free}."
            )
        if not 0.0 < self.frac_Z < 1.0:
            raise ValueError(f"frac_Z={self.frac_Z} must be between zero and one.")
        if self.polish_rings < 0:
            raise ValueError(f"polish_rings={self.polish_rings} must not be negative.")
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
        or finer, the main span ending as finely as the gap begins.
        """
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

        Evenly spaced in index, so following the spanwise clustering, and
        always including the hub and the casing. Every plane if
        :attr:`n_span_smooth` is zero.
        """
        nj = len(spf)
        if not self.n_span_smooth or self.n_span_smooth >= nj:
            return np.arange(nj)
        return np.unique(
            np.round(np.linspace(0, nj - 1, self.n_span_smooth)).astype(int)
        )

    #
    # GEOMETRY
    #

    def _wall(self, blade, cmap, t_O, corners=None, pitch=None):
        """Return the blade and where the corners of the O block sit on it.

        The blade as one closed line (2, N_WALL), evenly spaced in arc length,
        in the O block's order, from the middle of the trailing edge round the
        lower side; the arc lengths along it of the ends of its pieces, from
        the middle of the trailing edge through each corner of the O block;
        and which of those ends are C3, C4, C1, C2. The corners at `corners`,
        fractions of the way round, if given; else C3 and C1 from the bounding
        box, C2 at the throat, the point of the upper side's outline nearest
        C3 on the next blade `pitch` above, and C4 where the normal out of C1
        on the blade below lands on the lower side's.
        """
        upper, lower, te = _blade_pieces(blade, cmap, cmap.spf)
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
        s = np.linspace(0.0, S, N_WALL)
        wall = _at_arc(wall, s)
        if corners is not None:
            return wall, np.array([0.0, *(np.asarray(corners) * S), S]), (1, 2, 3, 4)

        # The outline of the O block, turned by the stagger about the leading
        # edge; its corners are where it comes nearest those of its box.
        outline = wall + t_O * _outward_normals(wall)
        i_LE = int(np.argmin(wall[0]))
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
        # Clear of the trailing edge, and in order round the blade.
        gap = 0.02 * S
        s_LE = s[i_LE]
        s_C["C3"] = np.clip(s_C["C3"], s_lo + gap, s_LE - 3 * gap)
        s_C["C4"] = np.clip(s_C["C4"], s_C["C3"] + gap, s_LE - gap)
        # The passage block's rear edge, from C2 across to C3 on the next
        # blade, leaves the O block square: on a staggered blade the box's
        # corner can sit far up the upper side, its edge to C3 running along
        # the blade, and the blocks round C2 starting as slivers.
        upper = (s > s_LE + 3 * gap) & (s < s_hi - gap)
        i_C3 = int(np.argmin(np.abs(s - s_C["C3"])))
        to_C3 = outline - (outline[:, i_C3] + pitch * _J)[:, None]
        s_C["C2"] = s[upper][np.argmin(np.hypot(*to_C3[:, upper]))]
        s_C["C1"] = np.clip(s_C["C1"], s_LE + gap, s_C["C2"] - gap)
        # Likewise its front edge, from C1 on the blade below across to C4:
        # C4 where the normal out of C1 lands on the lower side's outline, if
        # it does, so the edge leaves C1 square too. The nearest point to C1
        # instead can lie far back along a staggered blade's pressure side.
        lower = (s > s_C["C3"] + gap) & (s < s_LE - gap)
        i_C1 = int(np.argmin(np.abs(s - s_C["C1"])))
        A = outline[:, i_C1] - pitch * _J
        n = _outward_normals(wall)[:, i_C1]
        rel = outline[:, lower] - A[:, None]
        ahead = rel[0] * n[0] + rel[1] * n[1] > 0.0
        off = np.abs(rel[0] * n[1] - rel[1] * n[0])
        if ahead.any():
            s_C["C4"] = s[lower][ahead][np.argmin(off[ahead])]

        corners = [s_C[k] for k in ("C3", "C4", "C1", "C2")]
        return wall, np.array([0.0, *corners, S]), (1, 2, 3, 4)

    def _geometry(self, blade, cmap, t_O, corners, f_wall):
        """Return the blade, its coarse nodes, and the O block outline.

        The corners of the O block at `corners`, fractions of the way round
        the blade, and the nodes at fractions `f_wall` of each piece of it
        between them, the same at every span.
        """
        wall, marks, i_C = self._wall(blade, cmap, t_O, corners)
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

        cmap = ConformalMap(machine.annulus, 0.5, m_in, m_out)
        r_ref = float(np.mean(_by_theta(blade, 0.5)[0][1]))
        to_c = blade.evaluate_chord(0.5) / r_ref / self.resolution_factor
        h, h_LE, H = self.ds * to_c, self.ds_LE * to_c, self.ds_free * to_c
        t_O = self.t_O * pitch

        # Coarse along the blade: the free stream size, or less where it
        # turns sharply; the two sides of the passage block alike.
        # The corners found at mid-span, and kept as fractions of the way
        # round at every span: found at each, the nearest point to a corner of
        # the box can jump from one part of the blade to another between
        # spans, and the planes either side no longer fit together.
        wall, marks, i_C = self._wall(blade, cmap, t_O, pitch=pitch)
        S = marks[-1]
        corners = tuple(float(x) for x in marks[1:5] / S)
        s = np.linspace(0.0, S, N_WALL)
        density = _coarse_density(wall, H, np.radians(self.turn_max))
        i3, i4, i1, _ = i_C
        n = [max(1, round(_integral(s, density, a, b))) for a, b in _pairs(marks)]
        n[i3] = n[i1] = max(n[i3], n[i1])
        # Where the coarse nodes lie along each piece, by the curvature at
        # mid-span, kept at every span so a correction carried between spans
        # moves the same nodes.
        f_wall = tuple(
            (_by_density(s, density, a, b, k) - a) / (b - a)
            for (a, b), k in zip(_pairs(marks), n)
        )
        geo = self._geometry(blade, cmap, t_O, corners, f_wall)
        s_nodes = geo["s_loop"] * S
        ends = np.cumsum((0, *n))

        # Refined along the blade: down to h_LE at the leading edge and the
        # middle of the trailing edge, growing at ER_surf to h between them.
        s_LE = s[int(np.argmin(wall[0]))]
        grow = self.ER_surf - 1.0
        fine = np.minimum.reduce(
            [
                np.full_like(s, h),
                h_LE + grow * np.abs(s - s_LE),
                h_LE + grow * np.minimum(s, S - s),
            ]
        )
        nf = [round(_integral(s, 1.0 / fine, a, b)) for a, b in _pairs(marks)]
        nf_side = _up8(max(nf[i3], nf[i1]))
        nf_front = _up8(nf[i4])
        nf_rl = max(nf[0], 2)
        nf_ru = max(nf[4], 2)
        nf_ru += -(nf_rl + nf_ru) % 8
        # The H block of a tip gap takes half the cells round the blade along
        # each side, and its counts must halve under multigrid too.
        if row.tip_gap:
            nf_ru += -(nf_rl + nf_ru + 2 * nf_side + nf_front) % 16

        def along(j, m):
            """Fractional coarse indices of `m` refined cells along piece `j`."""
            s_f = _by_density(s, 1.0 / fine, marks[j], marks[j + 1], m)
            u = np.interp(s_f, s_nodes[ends[j] : ends[j + 1] + 1], np.arange(n[j] + 1))
            u[0], u[-1] = 0.0, n[j]
            return u

        u_side = 0.5 * (along(i1, nf_side) + _reverse(along(i3, nf_side), n[i3]))

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
        n_P, n_in, n_out = (max(2, round(L / H)) for L in (L_P, L_in, L_out))
        f_P = _mult(_double(h / L_P, h / L_P, H / L_P, self.ER_stream), 8)
        f_in = 1.0 - np.flip(_mult(_single(h / L_in, H / L_in, self.ER_stream), 8))
        f_out = _mult(_single(h / L_out, H / L_out, self.ER_stream), 8)

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
            n_O=max(2, round(t_O / H)),
            n_P=n_P,
            n_in=n_in,
            n_out=n_out,
            n_up=_up8(round((mp_Zin - mp_in) * stretch_in / H)),
            n_dn=_up8(round((mp_out - mp_Zout) * stretch_out / H)),
            nk_O=len(_mult(f_O, 8)),
            wall_density=(s / S, S / fine),
            u={
                "rl": along(0, nf_rl),
                "side_lo": u_side,
                "side_up": u_side,
                "front": along(i4, nf_front),
                "ru": along(4, nf_ru),
                "P": f_P * n_P,
                "in": f_in * n_in,
                "out": f_out * n_out,
            },
        )
        logger.debug(
            f"Row {i_row} OH passage layout, coarse cells: blade {layout.n_wall}, "
            f"O {layout.n_O}, passage {n_P}, in {n_in}, out {n_out}, up "
            f"{layout.n_up}, dn {layout.n_dn}; refined: blade "
            f"{[len(v) - 1 for v in layout.u.values()]}, O {layout.nk_O} nodes"
        )
        return layout

    #
    # ONE PLANE
    #

    def plane(self, machine, i_row, layout, spf):
        """Return the coarse algebraic mesh of one constant-span plane."""
        blade = machine.rows[i_row].blade
        pitch = layout.pitch
        pJ = pitch * _J
        cmap = ConformalMap(machine.annulus, spf, layout.m_in, layout.m_out)
        mp_in, mp_Zin, mp_Zout, mp_out = cmap.mp_from_m(
            [layout.m_in, layout.m_Zin, layout.m_Zout, layout.m_out]
        )
        g = self._geometry(blade, cmap, layout.t_O, layout.corners, layout.f_wall)
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
            "control": None,
            "layout": layout,
        }

    #
    # SMOOTHING
    #

    def _topology(self, xy, pitch):
        """Return the topology of a plane, with what the sweeps need."""
        shapes = [b.shape[1:] for b in xy]
        fixed = [np.zeros(s, dtype=bool) for s in shapes]
        slide = [np.zeros(s, dtype=bool) for s in shapes]
        fixed[0][:, 0] = True
        slide[6][0, :] = True
        slide[7][-1, :] = True
        topo = elliptic.PlaneTopology(xy, pitch=pitch, fixed=fixed, slide=slide)
        return self._sweep_operators(topo)

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
                # so the nodes there stay free to square up to it.
                most = self.n_corner if b else (nk - 1) // 2
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

    def _gap_sweeps(self, topo, X, n_sweep):
        """Return `X` of a tip gap's plane after `n_sweep` sweeps of Winslow's
        equations: as :meth:`_sweeps`, but with the outline fixed rather than
        a wall to slide along, and no period."""
        W_irr, c_irr = topo.sweep_irr
        W_sl, c_sl = topo.sweep_slide
        cp = topo.sweep_corner
        none = np.zeros(0, dtype=np.int32)
        X, _, _ = ember.elliptic.winslow_sweeps(
            X.astype(DTYPE),
            topo.stencil_g,
            topo.stencil_s,
            0.0,
            (topo.irregular, W_irr.indptr, W_irr.indices, W_irr.data, c_irr),
            (topo.slide, W_sl.indptr, W_sl.indices, W_sl.data, c_sl),
            (cp["node"], cp["corner"], 0.0 * cp["shift"], cp["f"]),
            (
                none,
                none,
                np.zeros(0, dtype=DTYPE),
                np.zeros(0, dtype=DTYPE),
                np.zeros((2, 2), dtype=DTYPE),
                np.zeros(2, dtype=DTYPE),
                np.zeros(0, dtype=DTYPE),
                none,
                none,
            ),
            n_sweep,
        )
        return X.astype(float)

    def _sweeps(self, topo, X, plane, n_sweep):
        """Return `X` after `n_sweep` point-Jacobi sweeps of Winslow's equations.

        By :func:`ember.elliptic.winslow_sweeps`: each sweep the regular and
        singular nodes from their stencils, the inlet and outlet level with
        the nodes inside, the nodes near the corners drawn in, and the blade
        nodes slid along the blade to the foot of the perpendicular from the
        node outside them.
        """
        pitch = topo.pitch
        W_irr, c_irr = topo.sweep_irr
        W_sl, c_sl = topo.sweep_slide
        cp = topo.sweep_corner
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
            X,
            topo.stencil_g,
            topo.stencil_s,
            pitch,
            (topo.irregular, W_irr.indptr, W_irr.indices, W_irr.data, c_irr),
            (topo.slide, W_sl.indptr, W_sl.indices, W_sl.data, c_sl),
            (cp["node"], cp["corner"], cp["shift"] * pitch, cp["f"]),
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
        )
        return X

    def mesh_row(self, machine, i_row, spacing, spf, jtip=None):
        """Return the ember blocks of one row, with a tip gap from `jtip`."""
        lay = self.layout(machine, i_row, spacing)
        planes = [self.plane(machine, i_row, lay, s) for s in spf]
        j_mid = int(np.argmin(np.abs(spf - 0.5)))

        topo = self._topology(planes[j_mid]["xy"], lay.pitch)
        # The polish leaves the inner half of the O block alone, whose lines
        # off the blade the clustering afterwards puts the first cells on.
        nk = topo.gid[0].shape[1]
        inner = topo.gid[0][:, : (nk + 1) // 2].ravel()
        X_alg = [topo.gather(p["xy"]) for p in planes]

        # Swept on the control planes, and the correction that made carried
        # to the planes between.
        j_ctrl = np.union1d(self.control_planes(spf), [j_mid])
        correction = [None] * len(spf)
        for j in j_ctrl:
            # Started from the algebraic plane as it is: the sweeps unfold
            # what folds in it without untangling first.
            X = self._sweeps(topo, X_alg[j], planes[j], self.n_sweep)
            if self.polish_rings:
                X = elliptic.polish(
                    topo,
                    X,
                    topo.irregular,
                    rings=self.polish_rings,
                    X_target=X_alg[j],
                    power=POLISH_POWER,
                    hold=inner,
                )
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

        # The refinement along the blade taken from where the sweeps left the
        # coarse nodes at mid-span, not where the layout first put them.
        lay = self._wall_u(lay, planes[j_mid], topo.scatter(X_smooth[j_mid]))
        for p in planes:
            p["layout"] = lay

        # The refined planes share one topology, built when first needed. The
        # control planes are polished once refined, and how far that moves
        # each node carried to the planes between, as the smoothing is.
        fine = {}
        for p in planes:
            p["fine"] = fine
        if self.polish_below:
            shift = [None] * len(spf)
            for j in j_ctrl:
                xy = self._refined(topo.scatter(X_smooth[j]), planes[j])
                shift[j] = self._polish_shift(xy, planes[j])
            for p, d in zip(planes, between(shift)):
                p["polish_shift"] = d
        return self.assemble(machine, i_row, lay, planes, topo, X_smooth, jtip)

    def _wall_u(self, lay, plane, xy):
        """Return `lay` refined along the blade to its density on plane `xy`.

        The layout places refined nodes at fractional coarse indices worked
        out from where the coarse nodes were before the sweeps, which slide
        them along the blade, and by interpolating linearly between them,
        where :meth:`_refined` uses a spline: so the spacing along the blade
        steps wherever the coarse spacing changes sharply, as it does where a
        round trailing edge meets the surfaces. Here the indices are the
        inverse of that spline through the swept coarse nodes, at the nodes
        the layout's density puts along each piece, so the spacing along the
        blade at this plane is the density's, scaled to fit each piece's
        count of cells. The scale runs linearly along each piece and
        continuously from one to the next, so the spacing does not step where
        they meet, as it would scaling each piece alone. At the corners
        behind the blade, C3 and C2, the scale is instead what makes the
        rear pieces' first cells as long as the passage block's last cells
        across there, at the downstream end of the trailing edge block, which
        the blocks behind the blade continue along `k`. The two sides are refined apart, and the passage block between
        them blends from one to the other.
        """
        wall = plane["wall"]
        S = util.arc_length(wall)
        s_c = _wall_arc(wall, xy[0][:, :, 0])
        e = np.cumsum((0, *lay.n_wall))
        m = {k: len(lay.u[k]) - 1 for k in ("rl", "side_lo", "front", "side_up", "ru")}
        counts = np.array(list(m.values()))

        # The density round the whole blade, from the first corner to where
        # it closes on itself.
        sigma, dens = lay.wall_density
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
            s_q = scipy.interpolate.PchipInterpolator(
                np.arange(b - a + 1), s_c[a : b + 1]
            )(q)
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

    def finish(self, xy, plane):
        """Return the blocks of a smoothed coarse plane, refined, and moved by
        the polish carried to it from the control planes."""
        out = self._refined(xy, plane)
        if "polish_shift" in plane:
            topo = plane["fine"]["topo"]
            out = topo.scatter(topo.gather(out) + plane["polish_shift"])
        return out

    def _refined(self, xy, plane):
        """Return the blocks of a smoothed coarse plane, refined.

        Each block refined along each of its directions on cubic splines
        through the coarse nodes, one between each pair of block junctions;
        the O block's blade nodes put back on the blade, and each line off it
        respaced along itself from the wall spacing at the blade to `h` at
        the outside.
        """
        lay = plane["layout"]
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

        e = np.cumsum((0, *lay.n_wall))
        O_i = [
            (e[0], e[1], u["rl"]),
            (e[1], e[2], _reverse(u["side_lo"], n_sd)),
            (e[2], e[3], u["front"]),
            (e[3], e[4], u["side_up"]),
            (e[4], e[5], u["ru"]),
        ]
        plan = {
            "O": (O_i, None),
            "P": (side, across),
            "LE": ([(0, lay.n_in, u["in"])], front),
            "Pin": ([(0, lay.n_in, u["in"])], across),
            "TE": ([(0, lay.n_out, u["out"])], rear),
            "Pout": ([(0, lay.n_out, u["out"])], across),
            "up": (None, across + shifted(front, n_P)),
            "dn": (None, across + shifted(rear, n_P)),
        }
        # The O block's blade nodes refined through where they lie along the
        # blade, monotonically, so they stay on it and in order round it.
        wall = plane["wall"]
        S = util.arc_length(wall)
        s_c = _wall_arc(wall, xy[0][:, :, 0])
        s_f = np.concatenate(
            [
                scipy.interpolate.PchipInterpolator(
                    np.arange(a, b + 1), s_c[a : b + 1]
                )(a + v)[(0 if j == 0 else 1) :]
                for j, (a, b, v) in enumerate(O_i)
            ]
        )

        # Each block refined along each direction by a matrix, the splines'
        # values for each coarse node alone, built once for the row: the
        # refinement is linear in the coarse nodes, and its pieces fixed.
        fine = plane["fine"]
        if "refine" not in fine:
            fine["refine"] = {
                name: tuple(
                    None if pieces is None else _refine_matrix(n, pieces, B.dtype)
                    for n, pieces in zip(B.shape[1:], plan[name])
                )
                for name, B in zip(LABELS, xy)
            }
            fine["refine_P_lo"] = _refine_matrix(n_sd + 1, side_lo, xy[1].dtype)
        out = []
        for name, B in zip(LABELS, xy):
            R_i, R_k = fine["refine"][name]
            if name == "P":
                # Along the upper side of the blade at k = 0 and the lower at
                # k = -1, each as the O block it meets is refined, and
                # blended linearly across.
                w = np.linspace(0.0, 1.0, B.shape[2], dtype=B.dtype)
                B = (R_i @ B) * (1.0 - w) + (fine["refine_P_lo"] @ B) * w
            elif R_i is not None:
                B = R_i @ B
            if R_k is not None:
                B = B @ R_k.T
            out.append(B)

        # The O block to the blade and clustered off it.
        O = out[0]
        O[:, :, 0] = _at_arc(wall, np.mod(s_f, S))
        nk = lay.nk_O
        d_wall = lay.d_wall / plane["cmap"].xr_from_mp(O[0, :, 0])[1]
        seg = np.linalg.norm(np.diff(O, axis=2), axis=0)
        s = np.concatenate((np.zeros((len(seg), 1)), np.cumsum(seg, axis=1)), axis=1)
        L = s[:, -1]
        T = L[:, None] * clusterfunc.double.fixed_many(
            d_wall / L, _outer_spacing(d_wall, lay.h, L, nk) / L, nk
        )
        # Every line at once: which coarse segment each new node falls in,
        # and how far along it.
        q = np.clip((T[:, :, None] >= s[:, None, 1:-1]).sum(axis=2), 0, len(s[0]) - 2)
        s_lo = np.take_along_axis(s, q, axis=1)
        s_hi = np.take_along_axis(s, q + 1, axis=1)
        w = ((T - s_lo) / np.maximum(s_hi - s_lo, 1e-30)).astype(O.dtype)
        lo = np.take_along_axis(O, q[None], axis=2)
        hi = np.take_along_axis(O, q[None] + 1, axis=2)
        out[0] = lo + w[None] * (hi - lo)
        return out

    def _polish_shift(self, xy, plane):
        """Return how far polishing moves each node of refined blocks.

        The nodes within `POLISH_FINE_RINGS` cells of every cell whose worst
        corner is under :attr:`polish_below`, against cells the size they
        are; the O block's first `HOLD_FINE` layers of cells off the blade
        held, so its spacing and squareness there stay as clustered.
        """
        fine = plane["fine"]
        pitch = plane["layout"].pitch
        if "topo" not in fine:
            shapes = [B.shape[1:] for B in xy]
            fixed = [np.zeros(sh, dtype=bool) for sh in shapes]
            slide = [np.zeros(sh, dtype=bool) for sh in shapes]
            fixed[0][:, 0] = True
            slide[6][0, :] = True
            slide[7][-1, :] = True
            topo = elliptic.PlaneTopology(xy, pitch=pitch, fixed=fixed, slide=slide)
            fine["topo"] = topo
            fine["hold"] = topo.gid[0][:, : HOLD_FINE + 1].ravel()
        topo = fine["topo"]
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
            rings=POLISH_FINE_RINGS,
            X_target=X,
            power=POLISH_POWER,
            hold=fine["hold"],
        )
        return polished - X

    #
    # TIP GAP
    #

    def _tip_gap(self, layout, planes, per_plane):
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
        renoded, half its nodes along each side at the same fractions of each
        side's length, and joined to it by a non-matching interface; the H
        block then runs straight across. The plane at the tip of the blade
        is built algebraically, which folds round a strongly curved blade,
        then smoothed as a main plane is: swept by Winslow's equations, which
        unfolds all but the O block's cells at the H block's corners, its O
        block's lines respaced to the wall spacing at the outline, and the
        whole plane polished: polishing only its poor cells, even again and
        again, leaves it worse in its worst cell, and fewer iterations of the
        whole can stop with cells the sweeps folded only just unfolded. That plane is carried to each above it by
        moving its outline to theirs and the nodes inside by a discrete
        Laplacian, then respacing. Carried instead as a correction to each
        plane's algebraic one, it folds near the casing where the blade
        twists, the algebraic planes' folds not being alike.

        Parameters
        ----------
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
        frac = _gap_fractions(walls[0])
        walls = [_gap_outline(W, frac) for W in walls]
        corners = _gap_corners(walls[0], self.nk_tip)
        alg = [_gap_plane(W, corners, self.t_tip_O, self.nk_tip_O) for W in walls]

        fixed = [np.zeros(B.shape[1:], dtype=bool) for B in alg[0]]
        fixed[0][:, -1] = True
        topo = self._sweep_operators(elliptic.PlaneTopology(alg[0], fixed=fixed))
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
        X = self._gap_sweeps(topo, topo.gather(alg[0]), self.n_sweep)
        X = respaced(X, d_wall[0])
        X = elliptic.polish(
            topo,
            X,
            np.arange(topo.n_node),
            rings=0,
            max_iter=self.n_polish_tip,
            power=POLISH_POWER,
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

    def assemble(self, machine, i_row, layout, planes, topo, X_smooth, jtip=None):
        """Return ember blocks from the smoothed planes of a row, with the two
        blocks of a tip gap from `jtip` after the rest."""
        n_row = len(machine.rows)
        per_plane = [
            self.finish(topo.scatter(X), plane) for X, plane in zip(X_smooth, planes)
        ]
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
        if jtip is None:
            return blocks

        # Over the gap the blade's outline is the face the O block of the gap
        # shares with the O block round the blade, noded differently.
        NonMatch = ember.patch.NonMatchPatch
        O.patches.append(NonMatch(k=0, j=(jtip, -1), label="tip_gap"))
        gap, (iA, iB, iC, iD) = self._tip_gap(layout, planes[jtip:], per_plane[jtip:])
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
        blocks += [tO, tH]
        for blk in blocks:
            blk.patches.append(ember.patch.RotatingPatch(j=-1, label="casing"))
        return blocks


def _project(wall, s_wall, tree, pts):
    """Return the arc lengths along `wall` (2, n) nearest each of `pts` (m, 2).

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


def _wall_arc(wall, nodes):
    """Return where `nodes` (2, n) lie along closed line `wall` (2, N),
    unwrapped to increase from the first, which the last closes on."""
    s_wall = util.cum_arc_length(wall)
    S = s_wall[-1]
    s_c = _unwrap(_project(wall, s_wall, cKDTree(wall.T), nodes.T), S)
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
    i_LE = int(np.argmin(W[0]))
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


def _gap_fractions(W):
    """Return where the outline of a tip gap has its nodes along each side.

    `W` (2, N + 1) is the outline of the blade, clockwise from the middle of
    the trailing edge, and `N` a multiple of sixteen. Fractions (N / 2 + 1,)
    of each side's length from the leading edge, the mean of the two sides'
    own spacings, so the outline keeps the clustering it had.
    """
    N = W.shape[1] - 1
    i_LE = int(np.argmin(W[0]))
    sides = (W[:, i_LE::-1], W[:, i_LE:])
    return np.mean(
        [util.resample_to(_norm_arc(side), N // 2 + 1) for side in sides], axis=0
    )


def _gap_outline(W, frac):
    """Return the outline `W` (2, N + 1) of a tip gap renoded at `frac`.

    Each side at the same fractions `frac` of its length from the leading
    edge, the leading edge at node `N / 2`, so that nodes of one index along
    either side lie level with each other.
    """
    i_LE = int(np.argmin(W[0]))
    lower, upper = (
        _at_arc(side, frac * util.arc_length(side))
        for side in (W[:, i_LE::-1], W[:, i_LE:])
    )
    out = np.concatenate((lower[:, ::-1], upper[:, 1:]), axis=1)
    out[:, 0], out[:, -1] = W[:, 0], W[:, -1]
    return out


def _gap_inset(W, frac):
    """Return the outline `W` (2, N + 1) moved inwards by `frac` of its thickness.

    Along the inward normals, by `frac` of the distance to the far side of
    the blade along them, but no more than `frac` of the diameter of
    curvature where the outline is convex: round a leading edge the normals
    meet, and the distance along them to the far side runs to the length of
    the blade. Rounded down where it rises sharply between neighbours.
    """
    n_in = -_outward_normals(W)
    P, d = W[:, :-1], n_in[:, :-1]
    A, E = W[:, :-1], np.diff(W, axis=1)
    # Where the ray from each node meets each segment of the outline.
    den = d[0][:, None] * E[1][None] - d[1][:, None] * E[0][None]
    AP = A[:, None, :] - P[:, :, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (AP[0] * E[1][None] - AP[1] * E[0][None]) / den
        u = (AP[0] * d[1][:, None] - AP[1] * d[0][:, None]) / den
    tol = 1e-6 * util.arc_length(W)
    t = np.where((t > tol) & (u >= 0.0) & (u <= 1.0), t, np.inf).min(axis=1)

    # Radius of curvature where the outline is convex, turning clockwise.
    d1 = 0.5 * (np.roll(P, -1, axis=1) - np.roll(P, 1, axis=1))
    d2 = np.roll(P, -1, axis=1) - 2.0 * P + np.roll(P, 1, axis=1)
    kappa = (d1[0] * d2[1] - d1[1] * d2[0]) / np.linalg.norm(d1, axis=0) ** 3
    rho = np.where(kappa < 0.0, -1.0 / np.minimum(kappa, -1e-300), np.inf)

    depth = frac * np.minimum(t, 2.0 * rho)
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
    evenly across; the H block by transfinite interpolation of the edges the
    corners cut its inner face into.
    """
    iA, iB, iC, iD = corners
    N = W.shape[1] - 1
    inner = _gap_inset(W, frac)
    H = _tfi(
        inner[:, iA : iB + 1][:, ::-1],
        inner[:, iC : iD + 1],
        inner[:, iB : iC + 1],
        np.concatenate((inner[:, iA::-1], inner[:, N - 1 : iD - 1 : -1]), axis=1),
    )
    f = np.linspace(0.0, 1.0, nk_O)[None, None]
    O = inner[..., None] + f * (W - inner)[..., None]
    return [O, H]


def _outer_spacing(d_wall, h, L, n):
    """Return the spacing at the outside of lines of length `L` (m,) and
    `n` nodes, `d_wall` (m,) at their other ends, for cells `h` beyond them.

    `h` where a line is long enough to grow to it smoothly; on a shorter
    line, the geometric mean of `h` and the spacing a geometric
    progression from `d_wall` reaches over the line. Holding `h` there
    squeezes the cells before it below the wall spacing and steps up by
    several times into the last; growing only as the progression does
    moves the whole step to the next block. The mean shares it between
    the two.
    """
    N = n - 1
    lo = np.ones_like(L)
    hi = np.full_like(L, 2.0)
    # Bisect for the ratio q of the progression, d_wall (q^N - 1)/(q - 1) = L.
    for _ in range(60):
        q = 0.5 * (lo + hi)
        total = d_wall * np.where(q > 1.0, (q**N - 1.0) / np.maximum(q - 1.0, 1e-30), N)
        long = total < L
        lo = np.where(long, q, lo)
        hi = np.where(long, hi, q)
    d_geo = np.where(L > N * d_wall, d_wall * lo ** (N - 1), L / N)
    return np.minimum(np.minimum(h, np.sqrt(h * d_geo)), L / 2)


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


def _coarse_density(wall, h_free, turn):
    """Return nodes per unit length along a closed line (2, n), evenly spaced,
    for cells of `h_free` or less, across which it turns by no more than
    `turn` radians."""
    d = np.gradient(wall, axis=1)
    phi = np.unwrap(np.arctan2(d[1], d[0]))
    ds = util.arc_length(wall) / (wall.shape[1] - 1)
    kappa = np.abs(np.gradient(phi)) / ds
    window = max(1, int(0.25 * h_free / ds))
    kappa = scipy.ndimage.uniform_filter1d(kappa, window, mode="wrap")
    return np.maximum(kappa / turn, 1.0 / h_free)


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


def _surfaces(blade, cmap, spf):
    """Return the two surfaces in conformal coordinates, higher-theta first.

    Each (2, n), from the leading edge to its trailing edge corner.
    """
    upper, lower = _by_theta(blade, spf, m=util.cluster_cosine(N_CHORD))
    return tuple(np.stack((cmap.mp_from_xr(s[:2]), s[2])) for s in (upper, lower))


def _blade_pieces(blade, cmap, spf):
    """Return the two surfaces and the trailing edge in conformal coordinates.

    Each surface (2, n) from the leading edge to its trailing edge corner,
    higher-theta first, and the trailing edge (2, n) from the upper corner to
    the lower. On a round trailing edge, the corners are where each surface's
    arc begins, and the trailing edge is the arc.
    """
    upper, lower = _surfaces(blade, cmap, spf)
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


def _norm_arc(curve):
    """Return normalised arc length along a polyline (2, n)."""
    s = util.cum_arc_length(curve)
    return s / s[-1]


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
    end slopes of the camber line, limited to 45 degrees.
    """
    cam = blade.evaluate_camber(cmap.spf, m=util.cluster_cosine(400))
    mp = cmap.mp_from_xr(cam[:2])
    theta = cam[2] + 0.5 * pitch
    slope = np.clip(
        [
            (theta[1] - theta[0]) / (mp[1] - mp[0]),
            (theta[-1] - theta[-2]) / (mp[-1] - mp[-2]),
        ],
        -SLOPE_MAX,
        SLOPE_MAX,
    )

    def periodic(mq):
        mq = np.asarray(mq, dtype=float)
        out = np.interp(mq, mp, theta)
        out = np.where(mq < mp[0], theta[0] + slope[0] * (mq - mp[0]), out)
        out = np.where(mq > mp[-1], theta[-1] + slope[1] * (mq - mp[-1]), out)
        return out

    return periodic


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
    for a, b in (
        (bottom[:, 0], left[:, 0]),
        (bottom[:, -1], right[:, 0]),
        (top[:, 0], left[:, -1]),
        (top[:, -1], right[:, -1]),
    ):
        assert np.allclose(a, b, rtol=0.0, atol=1e-12), f"TFI corners disagree {a} {b}"

    sb = _norm_arc(bottom)[:, None]
    st = _norm_arc(top)[:, None]
    tl = _norm_arc(left)[None, :]
    tr = _norm_arc(right)[None, :]
    u = (sb + (st - sb) * tl) / (1.0 - (st - sb) * (tr - tl))
    v = tl + (tr - tl) * u

    B, T = bottom[:, :, None], top[:, :, None]
    L, R = left[:, None, :], right[:, None, :]
    c00, c10 = bottom[:, 0, None, None], bottom[:, -1, None, None]
    c01, c11 = top[:, 0, None, None], top[:, -1, None, None]
    xy = (1 - v) * B + v * T + (1 - u) * L + u * R
    xy -= (1 - u) * (1 - v) * c00 + u * (1 - v) * c10 + (1 - u) * v * c01 + u * v * c11

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
        return clusterfunc.single.free(dmin, dmax, ER, mult=1)


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
