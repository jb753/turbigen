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
conformal coordinates `(m', theta)`, as in :mod:`turbigen.ohmesh`, with cells
of about the same size everywhere. A plane is built algebraically on a coarse
mesh, smoothed by point-Jacobi sweeps of Winslow's equations, refined, and
swept again. Each sweep, the corners take the mean of where each cell round
them would put them, and the nodes near them are drawn in towards them; the
nodes on the blade slide along it to the foot of the perpendicular from the
node outside them, which squares the cells there; and those on the inlet and
outlet sit level with the node inside, which squares the ends. The O block's
lines off the blade are only clustered to the wall spacing once the smoothing
is done, respaced along themselves from the wall spacing at the blade to the
size of their last cell at the outside.
"""

import dataclasses
import itertools
import logging
from typing import ClassVar

import ember.patch
import numpy as np
import scipy.interpolate
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

MIN_GAP = 0.1
"""Least fraction of its length as built a gap between blade nodes keeps as
they slide: only a guard against a run of them collapsing, the crowding the
sliding wants at the trailing edge being well above it."""


@dataclasses.dataclass(frozen=True)
class _Layout:
    """Counts of one row on the coarse mesh, fixed at mid-span."""

    pitch: float
    m_in: float
    m_Zin: float
    m_Zout: float
    m_out: float
    d_wall: float
    """Wall spacing [m]."""
    t_O: float
    """Thickness of the O block, in conformal units."""
    n_wall: tuple
    """Cells along each piece of the blade, from the middle of the trailing
    edge round the lower side, the leading edge and the upper side."""
    i_wall: tuple
    """Which ends of those pieces are the corners C3, C4, C1, C2."""
    n_O: int
    """Cells across the O block."""
    n_P: int
    """Cells across the passage block."""
    n_in: int
    n_out: int
    n_up: int
    n_dn: int
    """Cells along the blocks ahead of and behind the blade, and the inlet and
    outlet blocks."""
    nk_O: int
    """Nodes across the O block once clustered to the blade."""
    f_stream: dict
    """Normalised distributions along the inlet and outlet blocks, and the
    blocks ahead of and behind the blade, they are respaced to after
    smoothing, by block name."""

    def i_corners(self, refine=1):
        """O block `i` of the corners C3, C4, C1, C2, on the coarse mesh or
        refined by `refine`."""
        ends = np.cumsum((0, *self.n_wall)) * refine
        return tuple(int(ends[i]) for i in self.i_wall)


class OHPassage(PlaneMesher):
    """Generate a mesh with an O block round each blade and a passage block
    between neighbouring O blocks, of near uniform cell size."""

    type: ClassVar[str] = "ohp"

    labels: ClassVar[tuple] = LABELS

    grids_round_TE: ClassVar[bool] = True

    ds: float = 0.01
    """Size of the cells on the fine mesh, everywhere, as a fraction of
    meridional chord."""

    refine: int = 2
    """Fine cells to each coarse one: one, two, four or eight."""

    n_sweep_coarse: int = 500
    """Smoothing sweeps on the coarse mesh."""

    n_sweep_fine: int = 200
    """Smoothing sweeps on the fine mesh."""

    t_O: float = 0.1
    """Thickness of the O block, as a fraction of pitch, before smoothing."""

    ER_O: float = 1.2
    """Expansion ratio off the blade in the O block, once clustered."""

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

    ER_stream: float = 1.2
    """Expansion ratio of the cells along the inlet and outlet blocks, and
    those ahead of and behind the blade, once respaced after smoothing."""

    AR_stream: float = 4.0
    """Largest cell length along those blocks, as a multiple of `ds`."""

    polish_rings: int = 12
    """How many cells out from each corner of the O block are polished once
    the sweeps are done, by local target-matrix optimisation against the
    worst cells there, short of the inner half of the O block. None if
    zero."""

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
        if self.refine not in (1, 2, 4, 8):
            raise ValueError(f"refine={self.refine} must be one, two, four or eight.")
        if self.ds <= 0.0 or self.t_O <= 0.0:
            raise ValueError("ds and t_O must be positive.")
        if not 0.0 < self.frac_Z < 1.0:
            raise ValueError(f"frac_Z={self.frac_Z} must be between zero and one.")
        if self.polish_rings < 0:
            raise ValueError(f"polish_rings={self.polish_rings} must not be negative.")

    #
    # GEOMETRY
    #

    def _wall(self, blade, cmap, t_O):
        """Return the blade and where the corners of the O block sit on it.

        The blade as one closed line (2, N_WALL), evenly spaced in arc length,
        in the O block's order, from the middle of the trailing edge round the
        lower side; the arc lengths along it of the ends of its pieces, from
        the middle of the trailing edge through each corner of the O block;
        and which of those ends are C3, C4, C1, C2.
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
        s_C["C2"] = np.clip(s_C["C2"], s_LE + 3 * gap, s_hi - gap)
        s_C["C1"] = np.clip(s_C["C1"], s_LE + gap, s_C["C2"] - gap)

        corners = [s_C[k] for k in ("C3", "C4", "C1", "C2")]
        return wall, np.array([0.0, *corners, S]), (1, 2, 3, 4)

    def _geometry(self, blade, cmap, t_O, n_wall):
        """Return the blade, its nodes, and the O block outline on a plane."""
        wall, marks, i_C = self._wall(blade, cmap, t_O)
        s = np.linspace(0.0, marks[-1], N_WALL)
        s_loop = np.concatenate(
            [
                np.linspace(a, b, k + 1)[(0 if j == 0 else 1) :]
                for j, (a, b, k) in enumerate(zip(marks[:-1], marks[1:], n_wall))
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
        """Fix the counts of one row on the coarse mesh, at mid-span."""
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
        chord = blade.evaluate_chord(0.5) / r_ref
        h = self.ds * chord / self.resolution_factor
        h_c = h * self.refine
        q = 8 // self.refine
        t_O = self.t_O * pitch

        def count(L):
            """Coarse cells of about `h_c` along `L`, a multiple of `q`."""
            return max(q, q * round(L / h_c / q))

        _, marks, i_C = self._wall(blade, cmap, t_O)
        L = np.diff(marks)
        i3, i4, i1, i2 = i_C
        n = [max(1, round(x / h_c)) for x in L]
        # The two sides of the passage block alike, the leading edge piece
        # and the rear pieces together whole multigrid counts.
        n[i3] = n[i1] = count(max(L[i3], L[i1]))
        n[i4] = count(L[i4])
        rear = [k for k in range(len(n)) if k < i3 or k >= i2]
        n[rear[-2]] += -sum(n[k] for k in rear) % q

        geo = self._geometry(blade, cmap, t_O, n)
        C1, C2, C3, C4 = (geo[k] for k in ("C1", "C2", "C3", "C4"))
        mp_in, mp_Zin, mp_Zout, mp_out = cmap.mp_from_m([m_in, m_Zin, m_Zout, m_out])
        across = max(
            np.linalg.norm(C4 - C1 + pitch * _J), np.linalg.norm(C3 - C2 + pitch * _J)
        )

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
            n_wall=tuple(int(x) for x in n),
            i_wall=i_C,
            n_O=count(t_O),
            n_P=count(across),
            n_in=count(min(C1[0], C4[0]) - mp_Zin),
            n_out=count(mp_Zout - max(C2[0], C3[0])),
            n_up=count(mp_Zin - mp_in),
            n_dn=count(mp_out - mp_Zout),
            nk_O=len(_mult(f_O, 8)),
            f_stream=self._stream(
                blade, cmap, pitch, h, (mp_in, mp_Zin, mp_Zout, mp_out), C1, C2, C3, C4
            ),
        )
        logger.debug(
            f"Row {i_row} OH passage layout, coarse cells: blade {layout.n_wall}, "
            f"O {layout.n_O}, passage {layout.n_P}, in {layout.n_in}, out "
            f"{layout.n_out}, up {layout.n_up}, dn {layout.n_dn}; refined by "
            f"{self.refine}, O clustered to {layout.nk_O} nodes"
        )
        return layout

    def _stream(self, blade, cmap, pitch, h, mp, C1, C2, C3, C4):
        """Return the distributions the streamwise blocks are respaced to.

        Cells of `h` at the split lines and the O block, growing at
        `ER_stream` away from them up to `AR_stream` times `h`; along the
        lines from the corners at the slope of the periodic line.
        """
        mp_in, mp_Zin, mp_Zout, mp_out = mp
        periodic = _periodic_line(blade, cmap, pitch)
        d = 1e-6 * (mp_out - mp_in)
        stretch_in = np.hypot(1.0, (periodic(mp_Zin) - periodic(mp_Zin - d)) / d)
        stretch_out = np.hypot(1.0, (periodic(mp_Zout + d) - periodic(mp_Zout)) / d)
        h_max = self.AR_stream * h

        def one(L):
            return _mult(_single(h / L, h_max / L, self.ER_stream), 8)

        def two(L):
            return _mult(_double(h / L, h / L, h_max / L, self.ER_stream), 8)

        return {
            "up": 1.0 - np.flip(one((mp_Zin - mp_in) * stretch_in)),
            "LE": two((min(C1[0], C4[0]) - mp_Zin) * stretch_in),
            "TE": two((mp_Zout - max(C2[0], C3[0])) * stretch_out),
            "dn": one((mp_out - mp_Zout) * stretch_out),
        }

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
        g = self._geometry(blade, cmap, layout.t_O, layout.n_wall)
        loop, outer = g["loop"], g["outer"]
        C1, C2, C3, C4 = (g[k] for k in ("C1", "C2", "C3", "C4"))
        i3, i4, i1, i2 = layout.i_corners()
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
        for name, i in zip(("C4", "C1"), (i4, i1)):
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
        top_out = _line(C2, R_top, np.linspace(0.0, 1.0, layout.n_out + 1))
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
            "nk_O": layout.nk_O,
            "d_wall": layout.d_wall,
            "f_stream": layout.f_stream,
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

    @staticmethod
    def _refine(xy, r):
        """Return blocks (2, ni, nk) refined `r` times, bilinearly in index.

        Along an edge that is straight between its nodes, so it matches the
        edge of a neighbouring block whatever stretch of it they share; the
        sweeps on the fine mesh smooth it after.
        """
        out = []
        for B in xy:
            ni, nk = B.shape[1:]
            i, k = np.arange(ni), np.arange(nk)
            fi = np.linspace(0, ni - 1, (ni - 1) * r + 1)
            fk = np.linspace(0, nk - 1, (nk - 1) * r + 1)
            out.append(
                np.stack(
                    [
                        scipy.interpolate.RectBivariateSpline(i, k, c, kx=1, ky=1)(
                            fi, fk
                        )
                        for c in B
                    ]
                )
            )
        return out

    @staticmethod
    def _on_wall(xy, plane):
        """Return blocks with the O block's blade nodes put back on the blade."""
        O = xy[0].copy()
        wall = plane["wall"]
        s_wall = util.cum_arc_length(wall)
        S = s_wall[-1]
        s = _unwrap(_project(wall, s_wall, cKDTree(wall.T), O[:, :, 0].T), S)
        s[-1] = s[0] + S
        O[:, :, 0] = _at_arc(wall, np.mod(s, S))
        return [O, *xy[1:]]

    def mesh_row(self, machine, i_row, spacing, spf):
        """Return the ember blocks of one row."""
        lay = self.layout(machine, i_row, spacing)
        planes = [self.plane(machine, i_row, lay, s) for s in spf]
        j_mid = int(np.argmin(np.abs(spf - 0.5)))
        r = self.refine

        topo_c = self._topology(planes[j_mid]["xy"], lay.pitch)
        xy_alg = [self._on_wall(self._refine(p["xy"], r), p) for p in planes]
        topo_f = self._topology(xy_alg[j_mid], lay.pitch)
        # The polish leaves the inner half of the O block alone, whose lines
        # off the blade the clustering afterwards puts the first cells on.
        nk_f = topo_f.gid[0].shape[1]
        inner = topo_f.gid[0][:, : (nk_f + 1) // 2].ravel()
        X_alg = [topo_f.gather(xy) for xy in xy_alg]

        # Coarse then fine on the control planes, and the correction that
        # made carried to the planes between.
        j_ctrl = np.union1d(self.control_planes(spf), [j_mid])
        correction = [None] * len(spf)
        for j in j_ctrl:
            X = elliptic.untangle(topo_c, topo_c.gather(planes[j]["xy"]))
            X = self._sweeps(topo_c, X, planes[j], self.n_sweep_coarse)
            xy = self._on_wall(self._refine(topo_c.scatter(X), r), planes[j])
            X = elliptic.untangle(topo_f, topo_f.gather(xy))
            X = self._sweeps(topo_f, X, planes[j], self.n_sweep_fine)
            if self.polish_rings:
                X = elliptic.polish(
                    topo_f,
                    X,
                    topo_f.irregular,
                    rings=self.polish_rings,
                    X_target=X_alg[j],
                    power=POLISH_POWER,
                    hold=inner,
                )
            correction[j] = X - X_alg[j]
            logger.debug(f"Row {i_row} spf={spf[j]:.3f} swept")

        C = np.stack([correction[j] for j in j_ctrl])
        X_smooth = []
        for j in range(len(spf)):
            if correction[j] is None:
                k = np.searchsorted(j_ctrl, j)
                a, b = j_ctrl[k - 1], j_ctrl[k]
                w = (spf[j] - spf[a]) / (spf[b] - spf[a])
                correction[j] = (1.0 - w) * C[k - 1] + w * C[k]
            X_smooth.append(X_alg[j] + correction[j])

        return self.assemble(machine, i_row, lay, planes, topo_f, X_smooth)

    def finish(self, xy, plane):
        """Return the blocks of a smoothed plane, clustered where it matters.

        The blade nodes put back on the blade, and each line off it respaced
        along itself to the O block's full count of nodes, from the wall
        spacing at the blade to the length of its last cell at the outside.
        The blocks ahead of and behind the blade, and at the inlet and outlet,
        respaced along their streamwise lines to fewer nodes, growing away
        from the blade and the split lines.
        """
        O = self._on_wall(xy, plane)[0]
        nk = plane["nk_O"]
        d_wall = plane["d_wall"] / plane["cmap"].xr_from_mp(O[0, :, 0])[1]
        seg = np.linalg.norm(np.diff(O, axis=2), axis=0)
        s = np.concatenate((np.zeros((len(seg), 1)), np.cumsum(seg, axis=1)), axis=1)
        out = np.empty((2, O.shape[1], nk))
        for i in range(O.shape[1]):
            L = s[i, -1]
            t = L * clusterfunc.double.fixed(d_wall[i] / L, seg[i, -1] / L, nk)
            out[0, i] = np.interp(t, s[i], O[0, i])
            out[1, i] = np.interp(t, s[i], O[1, i])
        xy = [out, *xy[1:]]

        # The inlet and outlet blocks, and those ahead of and behind the
        # blade, respaced along their streamwise lines; a block and the one
        # beside it across the pitch alike, so their shared edges match.
        f = plane["f_stream"]
        for name, key in (
            ("up", "up"),
            ("LE", "LE"),
            ("Pin", "LE"),
            ("TE", "TE"),
            ("Pout", "TE"),
            ("dn", "dn"),
        ):
            b = LABELS.index(name)
            xy[b] = _respace_i(xy[b], f[key])
        return xy

    #
    # BLOCKS
    #

    def assemble(self, machine, i_row, layout, planes, topo, X_smooth):
        """Return ember blocks from the smoothed planes of a row."""
        n_row = len(machine.rows)
        blocks = self.blocks(machine, i_row, planes, topo, X_smooth)
        O, P, LE, Pin, TE, Pout, up, dn = blocks
        i3, i4, i1, i2 = layout.i_corners(self.refine)
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


def _respace_i(B, f):
    """Return block `B` (2, ni, nk) respaced along `i` to normalised `f`.

    In index rather than arc length: every line along `i` takes its new nodes
    at the same fractional indices, `f` of the way along, between the nodes
    it has. The block, smoothed to near even cells, is then spaced about as
    `f` is, and each new line across it is a blend of two smoothed ones, so
    lines of different length do not shear the lines across them.
    """
    ni = B.shape[1]
    u = f * (ni - 1)
    i0 = np.minimum(np.floor(u).astype(int), ni - 2)
    w = (u - i0)[None, :, None]
    return (1.0 - w) * B[:, i0, :] + w * B[:, i0 + 1, :]
