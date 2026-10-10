"""O-topology meshing with a shifted periodic boundary.

Three blocks per blade row: an O block wrapping the blade, whose outer face is
the periodic boundary itself, and an H block from each end of it to the inlet,
outlet or mixing plane::

      inlet/MP        Z_in                                  Z_out     outlet/MP
        |              D------------- upper periodic ---------F           |
        |      up      |                                      |    dn     |
        |              |          O (blade at k = 0)          |           |
        |              C------------- lower periodic ---------A           |

After Li (1994), shifted periodic grids. The outer nodes of the O block are
where the blade normals land. Those on the lower periodic line are paired node
for node with those on the upper, `A + j` with `F - j`, so the periodic
boundary is matched without the two lines having the same index range, and
each node of a pair sits midway, in `m'`, between where the rays of the two
landed. The corners are chosen so the rays of each pair land as close together
as they can, which is what keeps the grid lines off the blade near-normal
right out to the periodic boundary. Between A and C, and between D and F, the
outer face is a periodic line; from C to D it is the line `Z_in` a fixed
fraction of chord ahead of the leading edge, and from F round the trailing
edge to A the line `Z_out` behind it. The pitchwise H blocks of an O4H mesh
--- and the five block corners they bring --- are not needed.

As in :mod:`turbigen.ohmesh`, each constant-`j` plane is a surface of constant
span fraction meshed in its conformal coordinates `(m', theta)`, built
algebraically and then smoothed as one mesh by :mod:`turbigen.elliptic` with
the spacing and angle held on the blade by Sorenson-type control functions.
The O block's outer face is not a boundary there: the periodic lines and the
lines `Z_in` and `Z_out` all float, only the blade, inlet and outlet being
held.

Node counts and normalised distributions are fixed once per row at mid-span,
including the corner indices, and reused at every span.
"""

import dataclasses
import itertools
import logging
from typing import ClassVar

import ember.patch
import numpy as np

from turbigen import elliptic, util
from turbigen.ohmesh import (
    CORNER,
    ConformalMap,
    PlaneMesher,
    _outward_normals,
    _periodic_line,
    _surfaces,
    _tfi,
)

logger = logging.getLogger("turbigen")

LABELS = ("O", "up", "dn")
"""Names of the blocks of a row, in the order they are built."""

N_FILTER = 20
"""Passes of the three-point filter that smooths where the pairs of periodic
nodes sit, as Li's boundary filter."""

N_RAY = 2001
"""Samples along each blade normal when finding where it crosses a periodic
line."""

N_DENSITY = 4001
"""Samples along each piece of the blade loop on which the spacing of its
nodes is built."""

Z_MAX = 0.5
"""Furthest `Z_in` and `Z_out` go towards the inlet and outlet, as a fraction
of the distance from the blade."""


@dataclasses.dataclass(frozen=True)
class _Layout:
    """Counts and normalised distributions of one row, fixed at mid-span."""

    pitch: float
    m_in: float
    m_out: float
    d_wall: float
    """Wall spacing [m]."""
    h: float
    """Spacing of nodes along the blade away from the leading and trailing
    edges, as a fraction of the conformal chord at mid-span."""
    f_loop: tuple
    """Normalised arc length of the nodes along each of the four pieces of the
    blade loop: mid trailing edge to lower corner, lower corner to leading edge,
    leading edge to upper corner, upper corner to mid trailing edge."""
    nk: int
    """Nodes from the blade to the outer face of the O block."""
    i_A: int
    i_C: int
    i_D: int
    i_F: int
    """O block `i` of the corners of its outer face."""
    f_up: np.ndarray
    f_dn: np.ndarray
    """Streamwise distributions of the inlet and outlet H blocks."""

    @property
    def ni(self):
        return sum(len(f) - 1 for f in self.f_loop) + 1

    @property
    def i_te_corners(self):
        """O block `i` of the lower and upper trailing edge corners."""
        return len(self.f_loop[0]) - 1, self.ni - len(self.f_loop[3])

    @property
    def n_inlet(self):
        """Cells across the pitch at `Z_in`."""
        return self.i_D - self.i_C

    @property
    def n_exit(self):
        """Cells across the pitch at `Z_out`."""
        return self.ni - 1 - self.i_F + self.i_A


class OPeriodic(PlaneMesher):
    """Generate a mesh with one O block round each blade, periodic with itself."""

    type: ClassVar[str] = "o"

    labels: ClassVar[tuple] = LABELS

    ds_mid: float = 0.0125
    """Spacing of nodes along the blade surface, as a fraction of chord. Also
    the size of cells in mid-passage and on the lines `Z_in` and `Z_out`."""

    ratio_LE: float = 0.5
    """Spacing at the leading edge, as a fraction of :attr:`ds_mid`."""

    ER_surf: float = 1.15
    """Expansion ratio of the spacing along the blade, away from the leading
    edge and from the trailing edge corners. Those are approached at a few
    wall spacings, so the O block turns each over several cells."""

    extra_SS: float = 0.2
    """Fraction more nodes per unit length on the suction side. Its normals
    spread over the convex surface where those of the concave pressure side
    bunch, and the pairing wants as many landing on each periodic line."""

    ni_TE: int = 13
    """Fewest points across the trailing edge. Odd, so there is one in the
    middle for the O block to be cut at. The O block turns through half a
    circle round the trailing edge, at two square corners; with too few
    points there the turn is so sharp that the smoothing folds it."""

    blend_SS: float = 0.04
    """Distance over which the suction side's extra nodes blend in, at the
    leading edge and at mid trailing edge, as a fraction of the perimeter."""

    L_Z: float = 0.25
    """Distance of `Z_in` ahead of the leading edge, and `Z_out` behind the
    trailing edge, as a fraction of chord. No more than half way to the
    inlet, outlet or mixing plane."""

    dZ_max: float = 4.0
    """Largest pitchwise cell on `Z_in` and `Z_out`, as a multiple of
    :attr:`ds_mid`, which sets the fewest cells across them."""

    ER_O: float = 1.2
    """Expansion ratio away from the blade in the O block."""

    ER_stream: float = 1.2
    """Expansion ratio of the streamwise grid towards the inlet and outlet."""

    AR_stream: float = 2.0
    """Aspect ratio of cells at the inlet and outlet boundaries."""

    AR_mix: float = 2.0
    """Aspect ratio of cells at a mixing plane."""

    max_iter: int = 400
    """Most smoothing iterations on each plane."""

    polish_rings: int = 12
    """After smoothing, how many cells out from each block corner the mesh is
    polished by :func:`turbigen.elliptic.polish`. Zero not to. Four O and H
    cells meet at each corner across the periodic line, two and one either
    side, and the smoothing rounds the O block's outer face off there, which
    leaves slivers in the H block beside it."""

    omega: float = 0.5
    """Under-relaxation of each smoothing iteration. Lower than for an OH
    mesh: with the wall control acting across the whole O block, out to a
    boundary that floats, 0.8 can run away."""

    def __post_init__(self):
        if not 0.0 < self.ratio_LE <= 1.0:
            raise ValueError(f"ratio_LE={self.ratio_LE} must be in (0, 1].")
        if self.ni_TE < 3 or not self.ni_TE % 2:
            raise ValueError(f"ni_TE={self.ni_TE} must be odd and at least three.")
        if self.ER_surf <= 1.0:
            raise ValueError(f"ER_surf={self.ER_surf} must be more than one.")
        if self.polish_rings < 0:
            raise ValueError(f"polish_rings={self.polish_rings} must not be negative.")
        if self.L_Z <= 0.0:
            raise ValueError(f"L_Z={self.L_Z} must be positive.")

    #
    # ONE ROW
    #

    def layout(self, machine, i_row, spacing):
        """Fix the counts, distributions and corners of one row, at mid-span."""
        row = machine.rows[i_row]
        blade = row.blade
        n_row = len(machine.rows)
        pitch = 2.0 * np.pi / row.n_blade

        # Inlet and outlet of the row: the domain boundary, or half way across
        # the gap to a mixing plane.
        L_up = 1.0 if i_row == 0 else 0.5
        L_dn = 1.0 if i_row == n_row - 1 else 0.5
        m_in, m_out = 2.0 * i_row + 1.0 - L_up, 2.0 * i_row + 2.0 + L_dn

        cmap = ConformalMap(machine.annulus, 0.5, m_in, m_out)
        pieces = _loop_pieces(blade, cmap)
        chord = _chord(pieces)
        h = self.ds_mid * chord / self.resolution_factor

        # Wall spacing, in conformal units at mid-span.
        dense = np.concatenate([p[:, 1:] if n else p for n, p in enumerate(pieces)], 1)
        d_wall = float(spacing.surface[i_row])
        dw = float(np.median(d_wall / cmap.xr_from_mp(dense[0])[1]))

        # Nodes round the blade from a spacing that grows at ER_surf away from
        # the leading edge and from the trailing edge corners, which are
        # approached at a few wall spacings so the O block turns them over
        # several cells, up to ds_mid, or less on the suction side. The step
        # between the sides is blended in over blend_SS.
        S = util.arc_length(dense)
        ends = np.cumsum([0.0] + [util.arc_length(p) for p in pieces])
        # Sampled finer towards the ends of each piece, where the spacing is.
        s = np.unique(
            np.concatenate(
                [
                    a + (b - a) * util.cluster_cosine(N_DENSITY)
                    for a, b in itertools.pairwise(ends)
                ]
            )
        )
        w = self.blend_SS * S
        ind = 0.5 * (1.0 - np.tanh(s / w)) + 0.5 * (
            np.tanh((s - ends[2]) / w) - np.tanh((s - S) / w)
        )
        t_TE = 2.0 * ends[1]
        ds_TE = min(CORNER * dw, t_TE / 4.0)
        d_corner = np.minimum(np.abs(s - ends[1]), np.abs(s - ends[3]))
        grow = self.ER_surf - 1.0
        ds = np.minimum.reduce(
            [
                np.full_like(s, h) / (1.0 + self.extra_SS * ind),
                self.ratio_LE * h + grow * np.abs(s - ends[2]),
                ds_TE + grow * d_corner,
            ]
        )
        density = 1.0 / ds
        count = np.concatenate(
            ([0.0], np.cumsum(0.5 * (density[1:] + density[:-1]) * np.diff(s)))
        )

        # Whole cells on each piece, at least one, the total a multigrid count
        # made up on the suction side.
        c_ends = np.interp(ends, s, count)
        n_cell = np.maximum(np.round(np.diff(c_ends)).astype(int), 1)
        n_te_half = (self.ni_TE - 1) // 2
        n_cell[[0, 3]] = np.maximum(n_cell[[0, 3]], n_te_half)
        n_cell[2] += (-n_cell.sum()) % 8
        f_loop = []
        for a in range(4):
            c = np.linspace(c_ends[a], c_ends[a + 1], n_cell[a] + 1)
            sn = np.interp(c, count, s)
            f = (sn - ends[a]) / (ends[a + 1] - ends[a])
            f[0], f[-1] = 0.0, 1.0
            f_loop.append(f)

        # The blade loop and where its normals land, at mid-span.
        loop = _loop(pieces, f_loop)
        ni = loop.shape[1]
        i_LE = len(f_loop[0]) + len(f_loop[1]) - 2
        periodic = _periodic_line(blade, cmap, pitch)
        mp_in, mp_out = cmap.mp_from_m([m_in, m_out])
        mp_Zin, mp_Zout = _Z_lines(loop, i_LE, self.L_Z, mp_in, mp_out)
        hit = _landings(loop, i_LE, periodic, pitch, mp_Zin, mp_Zout)

        # Corners. A = a and F = ni-1-b bound the exit, of a + b cells through
        # mid trailing edge; C and D the inlet. Lower node A+j pairs with upper
        # node F-j. Chosen for the closest pairs, with the inlet nodes' normals
        # landing near Z_in and the exit nodes' near Z_out.
        n_min = max(8 * int(np.ceil(pitch / (self.dZ_max * h) / 8)), 8)
        best = None
        # The exit takes the whole trailing edge face, so each periodic line
        # starts from a node on a surface proper.
        a_min, b_min = len(f_loop[0]) - 1, len(f_loop[3]) - 1
        for NE in range(n_min, n_min + 65, 8):
            for a in range(a_min, NE - b_min + 1):
                b = NE - a
                iF = ni - 1 - b
                for NI in range(n_min, n_min + 65, 8):
                    n_per, odd = divmod(iF - a - NI, 2)
                    if n_per < 2 or odd:
                        continue
                    iC, iD = a + n_per, a + n_per + NI
                    if not iC < i_LE < iD:
                        continue
                    j = np.arange(n_per + 1)
                    ex = np.concatenate((np.arange(iF, ni), np.arange(0, a + 1)))
                    cost = (
                        np.mean((hit[a + j] - hit[iF - j]) ** 2)
                        + np.mean((hit[iC : iD + 1] - mp_Zin) ** 2)
                        + np.mean((hit[ex] - mp_Zout) ** 2)
                    )
                    if best is None or cost < best[0]:
                        best = (cost, a, iC, iD, iF)
        if best is None:
            raise ValueError(
                f"Row {i_row}: no corners fit {ni} nodes round the blade; "
                f"refine ds_mid."
            )
        _, iA, iC, iD, iF = best

        # Off the wall, enough nodes for the growth from the wall spacing to
        # the surface spacing and then to fill the median distance out.
        outer = _outer(loop, hit, periodic, pitch, mp_Zin, mp_Zout, iA, iC, iD, iF)
        D = np.linalg.norm(outer - loop, axis=0)
        n_grow = int(np.ceil(np.log(max(h / dw, 1.0)) / np.log(self.ER_O))) + 1
        L_grow = dw * (self.ER_O**n_grow - 1.0) / (self.ER_O - 1.0)
        nk = _up8(n_grow + max(np.median(D) - L_grow, 0.0) / h + 1)

        # Streamwise in the H blocks, square cells at Z growing to the boundary.
        AR_up = self.AR_stream if i_row == 0 else self.AR_mix
        AR_dn = self.AR_stream if i_row == n_row - 1 else self.AR_mix
        d_in, d_out = pitch / (iD - iC), pitch / (ni - 1 - iF + iA)
        L_in, L_out = mp_Zin - mp_in, mp_out - mp_Zout
        f_up = 1.0 - np.flip(_stream(d_in, AR_up * d_in, self.ER_stream, L_in))
        f_dn = _stream(d_out, AR_dn * d_out, self.ER_stream, L_out)

        layout = _Layout(
            pitch=pitch,
            m_in=m_in,
            m_out=m_out,
            d_wall=d_wall,
            h=h / chord,
            f_loop=tuple(f_loop),
            nk=nk,
            i_A=iA,
            i_C=iC,
            i_D=iD,
            i_F=iF,
            f_up=f_up,
            f_dn=f_dn,
        )
        j = np.arange(iC - iA + 1)
        logger.debug(
            f"Row {i_row} O layout: O {ni}x{nk}, inlet {layout.n_inlet} and exit "
            f"{layout.n_exit} cells, up {len(f_up)}, dn {len(f_dn)}, rms pair "
            f"mismatch {np.sqrt(np.mean((hit[iA + j] - hit[iF - j]) ** 2)) / h:.1f} "
            f"surface spacings"
        )
        return layout

    def plane(self, machine, i_row, layout, spf):
        """Return the algebraic mesh of one constant-span plane of a row."""
        blade = machine.rows[i_row].blade
        pitch = layout.pitch
        cmap = ConformalMap(machine.annulus, spf, layout.m_in, layout.m_out)
        pieces = _loop_pieces(blade, cmap)
        loop = _loop(pieces, layout.f_loop)
        ni, nk = layout.ni, layout.nk
        i_LE = len(layout.f_loop[0]) + len(layout.f_loop[1]) - 2
        iA, iC, iD, iF = layout.i_A, layout.i_C, layout.i_D, layout.i_F

        periodic = _periodic_line(blade, cmap, pitch)
        mp_in, mp_out = cmap.mp_from_m([layout.m_in, layout.m_out])
        mp_Zin, mp_Zout = _Z_lines(loop, i_LE, self.L_Z, mp_in, mp_out)
        hit = _landings(loop, i_LE, periodic, pitch, mp_Zin, mp_Zout)
        outer = _outer(loop, hit, periodic, pitch, mp_Zin, mp_Zout, iA, iC, iD, iF)

        # O block: straight rays, first cell the wall spacing growing at ER_O
        # to whatever fills each ray.
        r_wall = cmap.xr_from_mp(loop[0])[1]
        dw = layout.d_wall / r_wall
        D = np.linalg.norm(outer - loop, axis=0)
        frac = _ray_fractions(D, dw, self.ER_O, nk - 1)
        O = loop[:, :, None] + frac[None] * (outer - loop)[:, :, None]

        def h_block(left, right, mp0, mp1, f):
            mp = mp0 + f * (mp1 - mp0)
            bot = np.stack((mp, periodic(mp) - pitch))
            top = np.stack((mp, periodic(mp)))
            bot[:, 0], bot[:, -1] = left[:, 0], right[:, 0]
            top[:, 0], top[:, -1] = left[:, -1], right[:, -1]
            return _tfi(bot, top, left, right)

        def across(mp, n):
            th = periodic(mp)
            return np.stack((np.full(n + 1, mp), np.linspace(th - pitch, th, n + 1)))

        up = h_block(
            across(mp_in, layout.n_inlet),
            outer[:, iC : iD + 1],
            mp_in,
            mp_Zin,
            layout.f_up,
        )
        # The exit from A down through mid trailing edge, then from the top
        # of the loop down to F, reversed to run up the pitch.
        exit_ = np.concatenate(
            (outer[:, iA::-1], outer[:, ni - 2 : iF - 1 : -1]), axis=1
        )
        dn = h_block(exit_, across(mp_out, layout.n_exit), mp_Zout, mp_out, layout.f_dn)
        xy = [O, up, dn]
        for name, blk in zip(LABELS, xy):
            assert np.isfinite(blk).all(), f"non-finite coordinates in {name}"

        # The blade is fixed, the inlet and outlet slide along the pitch, and
        # everything else floats, the outer face of the O block included.
        fixed = [np.zeros(O.shape[1:], dtype=bool), None, None]
        fixed[0][:, 0] = True
        slide = [
            None,
            np.zeros(up.shape[1:], dtype=bool),
            np.zeros(dn.shape[1:], dtype=bool),
        ]
        slide[1][0, :] = True
        slide[2][-1, :] = True

        # Wall spacing in conformal units, against the local radius, left to
        # itself at the trailing edge corners, which have no normal.
        weight = np.ones(ni)
        for i_c in layout.i_te_corners:
            weight[i_c] = 0.0
            weight[i_c - 1] = weight[i_c + 1] = 0.5
        weight[-1] = weight[0]
        control = elliptic.WallControl(block=0, spacing=dw, closed=True, weight=weight)

        return {
            "xy": xy,
            "fixed": fixed,
            "slide": slide,
            "control": control,
            "cmap": cmap,
        }

    def polish(self, topo, X, X_alg):
        """Return `X` polished round the block corners.

        Against target cell sizes from the algebraic plane rather than the
        smoothed one, whose slivers would otherwise be taken as the target.
        """
        if not self.polish_rings:
            return X
        return elliptic.polish(
            topo, X, topo.irregular, rings=self.polish_rings, X_target=X_alg
        )

    def assemble(self, machine, i_row, layout, planes, topo, X_smooth):
        """Return ember blocks from the smoothed planes of a row."""
        n_row = len(machine.rows)
        O, up, dn = self.blocks(machine, i_row, planes, topo, X_smooth)
        iA, iC, iD, iF = layout.i_A, layout.i_C, layout.i_D, layout.i_F
        Per = ember.patch.PeriodicPatch

        O.patches.extend(
            [
                Per(i=0, label="cut0"),
                Per(i=-1, label="cut1"),
                Per(i=(0, iA), k=-1, label="exit_lower"),
                Per(i=(iA, iC), k=-1, label="lower"),
                Per(i=(iC, iD), k=-1, label="inlet"),
                Per(i=(iD, iF), k=-1, label="upper"),
                Per(i=(iF, -1), k=-1, label="exit_upper"),
            ]
        )
        up.patches.extend(
            [
                ember.patch.InletPatch(i=0)
                if i_row == 0
                else ember.patch.MixingPatch(i=0),
                Per(i=-1),
                Per(k=0),
                Per(k=-1),
            ]
        )
        dn.patches.extend(
            [
                ember.patch.OutletPatch(i=-1)
                if i_row == n_row - 1
                else ember.patch.MixingPatch(i=-1),
                Per(i=0, k=(0, iA)),
                Per(i=0, k=(iA, -1)),
                Per(k=0),
                Per(k=-1),
            ]
        )
        return [O, up, dn]


#
# GEOMETRY HELPERS
#


def _up8(n):
    """Round a node count up to 8 m + 1, m >= 1."""
    return 8 * max(int(np.ceil((n - 1) / 8)), 1) + 1


def _loop_pieces(blade, cmap):
    """Return the four pieces of the blade loop (2, n), clockwise.

    Mid trailing edge to its lower corner, lower surface to the leading edge,
    upper surface to its trailing edge corner, and back across the trailing
    edge to its middle.
    """
    upper, lower = _surfaces(blade, cmap, cmap.spf)
    mid = 0.5 * (upper[:, -1] + lower[:, -1])
    t = np.linspace(0.0, 1.0, 101)[None]
    return (
        mid[:, None] + t * (lower[:, -1] - mid)[:, None],
        lower[:, ::-1],
        upper,
        upper[:, -1:] + t * (mid - upper[:, -1])[:, None],
    )


def _chord(pieces):
    """Conformal chord: leading edge to mid trailing edge."""
    return float(np.linalg.norm(pieces[2][:, 0] - pieces[0][:, 0]))


def _loop(pieces, f_loop):
    """Return the blade loop (2, ni) at normalised arc lengths along each piece."""
    nodes = []
    for n, (p, f) in enumerate(zip(pieces, f_loop)):
        s = util.cum_arc_length(p)
        pts = np.stack([np.interp(f * s[-1], s, c) for c in p])
        nodes.append(pts if n == 0 else pts[:, 1:])
    loop = np.concatenate(nodes, axis=1)
    loop[:, -1] = loop[:, 0]
    return loop


def _Z_lines(loop, i_LE, L_Z, mp_in, mp_out):
    """Return `m'` of the lines ahead of and behind the blade.

    `L_Z` chords from the edges, or half way to the inlet or outlet if that
    is nearer, so the H block beyond is never squeezed flat against a mixing
    plane in a tight gap.
    """
    mp_LE, mp_TE = loop[0, i_LE], loop[0, 0]
    chord = mp_TE - mp_LE
    return (
        max(mp_LE - L_Z * chord, mp_LE - Z_MAX * (mp_LE - mp_in)),
        min(mp_TE + L_Z * chord, mp_TE + Z_MAX * (mp_out - mp_TE)),
    )


def _landings(loop, i_LE, periodic, pitch, mp_Zin, mp_Zout):
    """Return `m'` where each blade normal meets its periodic line.

    The pressure side's (before the leading edge) the lower periodic line, the
    suction side's the upper. A normal that misses, or lands past `Z_out`,
    counts as landing on `Z_out`; each side made monotone from the trailing
    edge towards the leading edge, and the lot clipped to between the lines.
    """
    normal = _outward_normals(loop)
    ni = loop.shape[1]
    shift = np.where(np.arange(ni) < i_LE, -pitch, 0.0)
    s = np.linspace(0.0, 2.0 * pitch, N_RAY)
    pts = loop[:, :, None] + s[None, None, :] * normal[:, :, None]
    gap = pts[1] - (periodic(pts[0]) + shift[:, None])
    cross = np.sign(gap[:, 1:]) != np.sign(gap[:, :-1])
    found = cross.any(axis=1)
    k = np.argmax(cross, axis=1)
    i = np.arange(ni)
    w = gap[i, k] / (gap[i, k] - gap[i, k + 1])
    hit = pts[0, i, k] + w * (pts[0, i, k + 1] - pts[0, i, k])
    hit = np.where(found, hit, mp_Zout)
    hit[: i_LE + 1] = np.minimum.accumulate(hit[: i_LE + 1])
    hit[i_LE:] = np.minimum.accumulate(hit[i_LE:][::-1])[::-1]
    return np.clip(hit, mp_Zin, mp_Zout)


def _outer(loop, hit, periodic, pitch, mp_Zin, mp_Zout, iA, iC, iD, iF):
    """Return the outer face of the O block (2, ni).

    Pairs of periodic nodes midway between where their two normals landed
    (Li's equation 1), filtered and kept in order; `Z_in` and `Z_out` uniform
    across the pitch.
    """
    ni = loop.shape[1]
    n_per = iC - iA
    j = np.arange(n_per + 1)
    m_pair = 0.5 * (hit[iA + j] + hit[iF - j])
    m_pair[0], m_pair[-1] = mp_Zout, mp_Zin
    for _ in range(N_FILTER):
        m_pair[1:-1] = 0.25 * m_pair[:-2] + 0.5 * m_pair[1:-1] + 0.25 * m_pair[2:]
    m_pair = np.maximum.accumulate(m_pair[::-1])[::-1]

    outer = np.empty((2, ni))
    outer[:, iA + j] = np.stack((m_pair, periodic(m_pair) - pitch))
    outer[:, iF - j] = np.stack((m_pair, periodic(m_pair)))
    th_C = periodic(mp_Zin) - pitch
    n_in = iD - iC
    outer[:, iC : iD + 1] = np.stack(
        (np.full(n_in + 1, mp_Zin), np.linspace(th_C, th_C + pitch, n_in + 1))
    )
    th_F = periodic(mp_Zout)
    n_ex = ni - 1 - iF + iA
    th = np.linspace(th_F, th_F - pitch, n_ex + 1)
    ex = np.concatenate((np.arange(iF, ni - 1), np.arange(0, iA + 1)))
    outer[:, ex] = np.stack((np.full(n_ex + 1, mp_Zout), th))
    outer[:, ni - 1] = outer[:, 0]
    return outer


def _stream(d0, d1, ER, L):
    """Return a multigrid count of normalised points over a length `L`.

    Cells grow from `d0` at `ER` to no more than `d1`, as few as fill it.
    """
    n_max = (
        int(np.ceil(np.log(max(d1, d0) / d0) / np.log(ER))) + int(np.ceil(L / d0)) + 2
    )
    cells = np.minimum(d0 * ER ** np.minimum(np.arange(n_max), 2000), max(d1, d0))
    n = int(np.searchsorted(np.cumsum(cells), L * (1.0 - 1e-9))) + 1
    n = _up8(n + 1) - 1
    return _ray_fractions(np.array([L]), min(d0, L / n), ER, n)[0]


def _ray_fractions(D, dw, ER, n_cell):
    """Return normalised node positions (n_ray, n_cell + 1) along each ray.

    Cells grow from `dw` at `ER` up to a cap, the cap found so the cells fill
    the ray. A ray too long for that keeps `dw` and grows faster; one too
    short to grow at all is uniform.
    """
    D = np.asarray(D, dtype=float)
    dw = np.broadcast_to(np.asarray(dw, dtype=float), D.shape)
    n = np.arange(n_cell)[None, :]

    # The expansion ratio, at least ER, that fills each ray with no cap.
    lo, hi = np.full(D.shape, float(ER)), np.full(D.shape, 10.0)
    for _ in range(60):
        r = 0.5 * (lo + hi)
        short = (dw[:, None] * r[:, None] ** n).sum(axis=1) < D
        lo, hi = np.where(short, r, lo), np.where(short, hi, r)
    g = dw[:, None] * np.maximum(lo, ER)[:, None] ** n

    # The cap that fills it, where growing at ER overfills.
    lo, hi = dw.copy(), D.copy()
    for _ in range(60):
        cap = 0.5 * (lo + hi)
        short = np.minimum(g, cap[:, None]).sum(axis=1) < D
        lo, hi = np.where(short, cap, lo), np.where(short, hi, cap)
    cells = np.minimum(g, hi[:, None])
    cells[dw * n_cell > D] = 1.0
    f = np.concatenate((np.zeros((len(D), 1)), np.cumsum(cells, axis=1)), axis=1)
    return f / f[:, -1:]
