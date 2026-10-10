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

Each constant-`j` plane is a surface of constant span fraction meshed in its
conformal coordinates `(m', theta)`, as in :mod:`turbigen.ohmesh`. It is built
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
"""

import dataclasses
import itertools
import logging
from typing import ClassVar

import ember.patch
import numpy as np
import scipy.interpolate
import scipy.ndimage
import scipy.sparse
from scipy.spatial import cKDTree

from turbigen import clusterfunc, elliptic, util
from turbigen.hmesh import _by_theta
from turbigen.ohmesh import (
    _J,
    ConformalMap,
    PlaneMesher,
    _at_arc,
    _blade_pieces,
    _double,
    _line,
    _mult,
    _norm_arc,
    _outward_normals,
    _periodic_line,
    _single,
    _tfi,
)
from turbigen.omesh import POLISH_POWER

logger = logging.getLogger("turbigen")

LABELS = ("O", "P", "LE", "Pin", "TE", "Pout", "up", "dn")
"""Block names, in the order blocks are held."""

N_WALL = 20001
"""Samples along the blade, which nodes on it are placed on."""

CORNER_COS, CORNER_SIN = np.cos(np.pi / 3.0), np.sin(np.pi / 3.0)
"""Direction of an edge out of a corner of the O block, against the O block's
face and its outward normal there."""

POLISH_FINE_RINGS = 3
"""How many cells out from each poor cell of the refined plane are polished."""

HOLD_FINE = 3
"""Layers of cells off the blade the polish of the refined plane leaves be."""

MIN_GAP = 0.1
"""Least fraction of its length as built a gap between blade nodes keeps as
they slide: only a guard against a run of them collapsing, the crowding the
sliding wants at the trailing edge being well above it."""


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
    u: dict
    """Where each refined node lies, in fractional coarse indices from the
    start of its stretch: along the rear of the lower and upper sides of the
    blade from the cut, `rl` and `ru`; along a side between corners,
    upstream to downstream, `side`; round the leading edge, `front`; across
    the passage, `P`; and along the blocks ahead of and behind the blade,
    `in` and `out`."""

    def i_corners_coarse(self):
        """O block `i` of the corners C3, C4, C1, C2 on the coarse mesh."""
        ends = np.cumsum((0, *self.n_wall))
        return tuple(int(ends[i]) for i in self.i_wall)

    def i_corners(self):
        """O block `i` of the corners C3, C4, C1, C2 once refined."""
        n = [len(self.u[k]) - 1 for k in ("rl", "side", "front", "side")]
        i3 = n[0]
        return i3, i3 + n[1], i3 + n[1] + n[2], i3 + n[1] + n[2] + n[3]


class OHPassage(PlaneMesher):
    """Generate a mesh with an O block round each blade and a passage block
    between neighbouring O blocks, smoothed coarse and refined where needed."""

    type: ClassVar[str] = "ohp"

    labels: ClassVar[tuple] = LABELS

    grids_round_TE: ClassVar[bool] = True

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
        return super().forward(machine, spacing)

    def __post_init__(self):
        self._check_end_shear()
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
            u={
                "rl": along(0, nf_rl),
                "side": u_side,
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
            "xy": xy,
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
        n = topo.n_node

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

    def _sweeps(self, topo, X, plane, n_sweep):
        """Return `X` after `n_sweep` point-Jacobi sweeps of Winslow's equations."""
        pitch = topo.pitch
        g, s = topo.stencil_g, topo.stencil_s * pitch
        W_irr, s_irr = topo.sweep_irr
        W_sl, s_sl = topo.sweep_slide
        cp = topo.sweep_corner
        wall = plane["wall"]
        tree = cKDTree(wall.T)
        s_wall = util.cum_arc_length(wall)
        gid, shift = topo.gid[0], topo.shift[0] * pitch
        S = s_wall[-1]

        # The least gaps between blade nodes, a fraction of those the plane
        # was built with, so a run of nodes cannot gather onto one point.
        s_ref = plane["s_loop"]
        n = gid.shape[0]
        s_ref = np.interp(
            np.linspace(0, len(s_ref) - 1, n), np.arange(len(s_ref)), s_ref
        )
        least = MIN_GAP * np.diff(s_ref) * s_wall[-1]
        floor = np.concatenate(([0.0], np.cumsum(least)))

        X = X.copy()
        for _ in range(n_sweep):
            # Stencil: node, +i, (+i, +k), +k, (-i, +k), -i, (-i, -k), -k, (+i, -k).
            P = X[g]
            P[..., 1] += s
            di = 0.5 * (P[:, 1] - P[:, 5])
            dk = 0.5 * (P[:, 3] - P[:, 7])
            alpha = np.sum(dk * dk, axis=1)[:, None]
            gamma = np.sum(di * di, axis=1)[:, None]
            beta = np.sum(di * dk, axis=1)[:, None]
            cross = P[:, 2] - P[:, 8] + P[:, 6] - P[:, 4]
            new = (
                alpha * (P[:, 1] + P[:, 5])
                + gamma * (P[:, 3] + P[:, 7])
                - 0.5 * beta * cross
            ) / (2.0 * (alpha + gamma))
            irr = W_irr @ X
            irr[:, 1] += s_irr
            X[topo.regular] = new
            X[topo.irregular] = irr
            X[topo.slide, 1] = W_sl @ X[:, 1] + s_sl

            # Nodes near the corners drawn in towards them.
            if len(cp["node"]):
                Xc = X[cp["corner"]]
                Xc[:, 1] += cp["shift"] * pitch
                X[cp["node"]] = Xc + (X[cp["node"]] - Xc) * cp["f"][:, None]

            # Blade nodes to the foot of the perpendicular from the next node
            # out, no further than half way to either neighbour. The nodes
            # close in a ring, the cut included, its two instances one turn
            # of the blade apart.
            w0 = X[gid[:, 0]]
            w0[:, 1] += shift[:, 0]
            w1 = X[gid[:, 1]]
            w1[:, 1] += shift[:, 1]
            s0 = _unwrap(_project(wall, s_wall, tree, w0), S)
            s1 = _unwrap(_project(wall, s_wall, tree, w1), S)
            s0[-1] = s0[0] + S
            s1[-1] = s1[0] + S
            before = np.concatenate(([s0[-2] - S], s0[:-1]))
            after = np.concatenate((s0[1:], [s0[1] + S]))
            s_new = np.clip(s1, 0.5 * (before + s0), 0.5 * (s0 + after))
            # No gap below its least: pushed along from the cut, then back to
            # it a turn on.
            s_new = np.maximum.accumulate(s_new - floor) + floor
            s_new[-1] = s_new[0] + S
            s_new = np.minimum.accumulate((s_new - floor)[::-1])[::-1] + floor
            foot = _at_arc(wall, np.mod(s_new, S)).T
            foot[:, 1] -= shift[:, 0]
            X[gid[:, 0]] = foot
        return X

    def mesh_row(self, machine, i_row, spacing, spf):
        """Return the ember blocks of one row."""
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
                    w = (spf[j] - spf[a]) / (spf[b] - spf[a])
                    values[j] = (1.0 - w) * V[k - 1] + w * V[k]
            return values

        correction = between(correction)
        X_smooth = [X_alg[j] + correction[j] for j in range(len(spf))]

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
        return self.assemble(machine, i_row, lay, planes, topo, X_smooth)

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
        side = [(0, n_sd, u["side"])]
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
            (e[1], e[2], _reverse(u["side"], n_sd)),
            (e[2], e[3], u["front"]),
            (e[3], e[4], u["side"]),
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
        s_wall = util.cum_arc_length(wall)
        S = s_wall[-1]
        s_c = _unwrap(_project(wall, s_wall, cKDTree(wall.T), xy[0][:, :, 0].T), S)
        s_c[-1] = s_c[0] + S
        s_f = np.concatenate(
            [
                scipy.interpolate.PchipInterpolator(
                    np.arange(a, b + 1), s_c[a : b + 1]
                )(a + v)[(0 if j == 0 else 1) :]
                for j, (a, b, v) in enumerate(O_i)
            ]
        )

        out = []
        for name, B in zip(LABELS, xy):
            along_i, along_k = plan[name]
            if along_i is not None:
                B = _refine_axis(B, 1, along_i)
            if along_k is not None:
                B = _refine_axis(B, 2, along_k)
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
            d_wall / L, np.minimum(lay.h, L / 2) / L, nk
        )
        clustered = np.empty((2, O.shape[1], nk))
        for i in range(O.shape[1]):
            clustered[0, i] = np.interp(T[i], s[i], O[0, i])
            clustered[1, i] = np.interp(T[i], s[i], O[1, i])
        out[0] = clustered
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
    # BLOCKS
    #

    def assemble(self, machine, i_row, layout, planes, topo, X_smooth):
        """Return ember blocks from the smoothed planes of a row."""
        n_row = len(machine.rows)
        blocks = self.blocks(machine, i_row, planes, topo, X_smooth)
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


def _refine_axis(B, axis, pieces):
    """Return block `B` (2, ni, nk) refined along `axis`, 1 or 2.

    `pieces` are stretches `(a, b, u)` from coarse node `a` to `b`, and where
    the refined nodes lie in fractional indices from `a`; each on a cubic
    spline through its own coarse nodes only, so a stretch of edge comes out
    the same in every block that shares it.
    """
    V = np.moveaxis(B, axis, -1)
    parts = []
    for j, (a, b, u) in enumerate(pieces):
        x = np.arange(a, b + 1, dtype=float)
        seg = V[..., a : b + 1]
        kind = "cubic" if b - a >= 3 else "linear"
        if kind == "cubic":
            vals = scipy.interpolate.CubicSpline(x, seg, axis=-1)(a + u)
        else:
            vals = scipy.interpolate.interp1d(x, seg, axis=-1)(a + u)
        parts.append(vals if j == 0 else vals[..., 1:])
    return np.moveaxis(np.concatenate(parts, axis=-1), -1, axis)


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
