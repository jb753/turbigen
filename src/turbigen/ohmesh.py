"""OH-topology meshing.

Seven blocks per blade row, in the layout AutoGrid calls O4H: an O block
wrapping the blade, four H blocks round it --- ahead of it, behind it, and
either side between it and the periodic line --- and an H block from each end
of those to the inlet, outlet or mixing plane::

     inlet/MP   Z_in                                     Z_out   outlet/MP
       |  up    |  H_in |         H_up           | H_out |  dn    |  periodic
       |        |       +C1--------------------C2+       |        |
       |        |       |   O (blade at k = 0)   |       |        |
       |        |       +C4--------------------C3+       |        |
       |        |       |         H_down         |       |        |  periodic

Every block runs from hub to casing in `j`, and every constant-`j` plane is a
surface of constant span fraction of the annulus --- the surface the blade
section at that span fraction lies on. Each is meshed in its conformal
coordinates `(m', theta)`, `dm' = dm / r`, in which the surface of revolution
is a plane and angles are angles on the surface: a mesh orthogonal there is
orthogonal on the stream surface, and the blade and its metal angles appear
undistorted, `dtheta/dm' = tan(chi)`, at any radius. A plane is first built
algebraically, then smoothed as one mesh by :mod:`turbigen.elliptic`, with
the blade surface fixed, the inlet and outlet sliding along the pitch, wall
spacing and orthogonality held on the blade, and every block interface and the
periodic line free to move.

Node counts and the normalised distributions along every block edge are fixed
once per row, at mid-span, and reused at every span, so the planes differ only
in the geometry they are fitted to. Planes are smoothed from mid-span outwards,
each starting from its own algebraic mesh plus the correction smoothing made to
its neighbour.

Indices follow the conventions the rest of the package reads: `j` runs from hub
to casing, the O block wraps in `i` with `i = 0` and `i = -1` the same nodes at
the middle of the trailing edge, and the blade is its `k = 0` face. H blocks
run `i` downstream and `k` in the direction of increasing `theta`.
"""

import dataclasses
import itertools
import logging
from typing import ClassVar

import ember.block
import ember.grid
import ember.patch
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from scipy.spatial import cKDTree

from turbigen import clusterfunc, elliptic, util
from turbigen.hmesh import _by_theta
from turbigen.mesh import Mesher

logger = logging.getLogger("turbigen")

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

CORNER = 2.0
"""Grid spacing into the trailing edge corners, as a multiple of wall spacing."""

N_RAY = 2001
"""Samples along a ray from a corner of the O block, to find where it meets
the periodic line."""

LABELS = ("O", "Hup", "Hdown", "Hin", "Hout", "up", "dn")
"""Block names, in the order blocks are held."""


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
    """Counts and normalised distributions of one row, fixed at mid-span."""

    pitch: float
    m_in: float
    m_Zin: float
    m_Zout: float
    m_out: float
    d_wall: float
    """Wall spacing [m]."""
    f_front: np.ndarray
    """Leading edge to corner of the O block, on each surface."""
    f_side_u: np.ndarray
    """Corner to corner of the O block, along the upper side."""
    f_side_l: np.ndarray
    """Corner to corner of the O block, along the lower side, which may have
    a different count from the upper."""
    f_rear: np.ndarray
    """Rear corner of the O block to the trailing edge corner."""
    f_te: np.ndarray
    """Across the trailing edge, upper corner to lower."""
    f_O: np.ndarray
    """Blade to the outside of the O block."""
    f_Hup: np.ndarray
    """O block to periodic line, in the upper H block."""
    f_Hdown: np.ndarray
    """O block to periodic line, in the lower H block."""
    f_up: np.ndarray
    f_Hin: np.ndarray
    f_Hout: np.ndarray
    f_dn: np.ndarray
    """Streamwise in each of the four blocks spanning the pitch."""
    split: tuple
    """Fractions of each surface's length at the leading and trailing corners
    of the O block."""
    t_O: float
    """Thickness of the O block, in conformal units."""
    s_f: int
    """Cells along the periodic line by which the lower side's H block starts
    downstream of the upper side's; negative if upstream."""

    @property
    def n_te_half(self):
        return (len(self.f_te) - 1) // 2

    @property
    def n_rear(self):
        """Nodes from the trailing edge midpoint to a rear corner."""
        return len(self.f_rear) + self.n_te_half

    @property
    def n_front(self):
        return 2 * len(self.f_front) - 1

    @property
    def ni_O(self):
        return (
            2 * self.n_rear + len(self.f_side_u) + len(self.f_side_l) + self.n_front - 4
        )

    @property
    def i_corners(self):
        """O block `i` of the corners C3, C4, C1, C2, in increasing order."""
        i3 = self.n_rear - 1
        i4 = i3 + len(self.f_side_l) - 1
        i1 = i4 + self.n_front - 1
        i2 = i1 + len(self.f_side_u) - 1
        return i3, i4, i1, i2

    @property
    def i_per(self):
        """Where the H blocks either side of the blade meet the periodic line.

        Counted in cells along the stretch of it between the inlet and outlet
        H blocks: the start and end of the upper side's H block on the upper
        line, the same for the lower side's on the lower line, and the length
        of the stretch.
        """
        n_u, n_l = len(self.f_side_u) - 1, len(self.f_side_l) - 1
        iu0, il0 = max(-self.s_f, 0), max(self.s_f, 0)
        return iu0, iu0 + n_u, il0, il0 + n_l, max(iu0 + n_u, il0 + n_l)

    @property
    def i_te_corners(self):
        """O block `i` of the lower and upper trailing edge corners."""
        return self.n_te_half, self.ni_O - 1 - self.n_te_half


class PlaneMesher(Mesher):
    """Base for meshers that grid each row on planes of constant span fraction.

    A subclass lays out a row once, at mid-span, with :meth:`layout`; builds
    the algebraic blocks of one plane with :meth:`plane`; and turns the
    smoothed planes into ember blocks with :meth:`assemble`. The rest --- the
    spanwise grid, which planes are smoothed, and carrying the smoothing from
    those to every other plane --- is shared here.
    """

    labels: ClassVar[tuple] = ()
    """Names of the blocks of a row, in the order :meth:`plane` returns them.
    Must include ``up`` and ``dn``, the blocks at a mixing plane."""

    grids_round_TE: ClassVar[bool] = False
    """Whether this mesher can grid a blade whose trailing edge is rounded
    off, see :attr:`turbigen.thickness.ThicknessDesign.round_TE`, rather than
    blunt."""

    dspf_mid: float = 0.03
    """Spanwise grid spacing at midspan, as a fraction of span."""

    ER_span: float = 1.2
    """Expansion ratio of spanwise grid away from hub and casing."""

    resolution_factor: float = 1.0
    """Multiply the number of points in each direction, keeping relative spacings."""

    n_span_smooth: int = 9
    """Number of spanwise planes to smooth, hub and casing included. The rest
    are interpolated between them. Zero smooths every plane."""

    max_iter: int = 150
    """Most smoothing iterations on each plane."""

    tol: float = 1e-3
    """Smoothing converges when no node moves more than this fraction of the
    wall spacing in an iteration."""

    omega: float = 0.8
    """Under-relaxation of each smoothing iteration."""

    end_shear: float = 1.0
    """Fraction of their shear along the periodic line that the blocks at the
    inlet and outlet keep, from one to zero.

    At one their streamwise grid lines follow the periodic line, as the flow
    does, and cross the pitchwise inlet or outlet at the angle it makes. At
    zero they leave the inlet and outlet square, and are as far off the flow.
    A cross-pitch line must start and end at the same `m'` to match its
    periodic image, so it runs pitchwise on average, and the skew and the
    turn off the flow add up to the periodic line's angle whatever is chosen
    between them.
    """

    #
    # THE STAGE
    #

    def forward(self, machine, spacing):
        """Generate a Grid for a designed machine."""
        for i_row, row in enumerate(machine.rows):
            if row.blade.round_TE and not self.grids_round_TE:
                raise ValueError(
                    f"Row {i_row} has a round trailing edge, which the "
                    f"{type(self).__name__} mesher does not grid. Mesh it with "
                    f"type: o, or leave it blunt."
                )
            if row.tip_gap:
                raise ValueError(
                    f"Row {i_row} has a tip gap of {row.tip_gap:.3g} m, and the "
                    f"{type(self).__name__} mesher does not yet grid clearances. Mesh it with "
                    f"type: h, or set the clearance to zero."
                )

        span_ref = machine.annulus.evaluate_span(1)
        spf = self.spanwise_grid(spacing.hub / span_ref, spacing.casing / span_ref)
        logger.debug(f"{type(self).__name__} spanwise points: {len(spf)}")

        n_row = len(machine.rows)
        blocks = []
        for i_row in range(n_row):
            blocks.append(self.mesh_row(machine, i_row, spacing, spf))

        _stitch_mixing_planes(blocks, self.labels)
        grid = ember.grid.Grid([b for row in blocks for b in row])
        grid.connectivity.periodic.pair()
        return grid

    def _check_end_shear(self):
        """Raise unless :attr:`end_shear` is a fraction."""
        if not 0.0 <= self.end_shear <= 1.0:
            raise ValueError(
                f"end_shear={self.end_shear} must be between zero and one."
            )

    def spanwise_grid(self, dspf_hub, dspf_casing):
        """Return the spanwise grid vector given hub and casing spacings."""
        return util.resample(
            clusterfunc.double.free(dspf_hub, dspf_casing, self.dspf_mid, self.ER_span),
            self.resolution_factor,
            mult=8,
        )

    #
    # ONE ROW
    #

    def mesh_row(self, machine, i_row, spacing, spf):
        """Return the ember blocks of one row."""
        layout = self.layout(machine, i_row, spacing)
        planes = [self.plane(machine, i_row, layout, s) for s in spf]
        j_mid = int(np.argmin(np.abs(spf - 0.5)))

        topo = elliptic.PlaneTopology(
            planes[j_mid]["xy"],
            pitch=layout.pitch,
            fixed=planes[j_mid]["fixed"],
            slide=planes[j_mid]["slide"],
            control=planes[j_mid]["control"],
        )

        X_tfi = [topo.gather(p["xy"]) for p in planes]
        correction = [None] * len(spf)

        def smooth_plane(j, pq):
            topo.control = planes[j]["control"]
            # Unfolded first, so the smoothing starts from a valid mesh and
            # reads its spacing off one. The correction is from the algebraic
            # mesh as it was, so the planes between take the unfolding too.
            X_alg = elliptic.untangle(topo, X_tfi[j])
            X, pq, n_iter = self.smooth(topo, X_alg, planes[j], pq)
            X = self.polish(topo, X, X_alg)
            correction[j] = X - X_tfi[j]
            logger.debug(f"Row {i_row} spf={spf[j]:.3f}: {n_iter} iterations")
            return pq

        # Smoothed from mid-span outwards, each control plane starting from its
        # own algebraic mesh and the wall control the last one finished with.
        # The control is what takes the iterations to settle, and it varies
        # little between neighbouring planes. Starting the mesh itself from the
        # neighbour's is no faster and ends worse: the sources that keep the
        # spacing are best read off the algebraic mesh they were built into.
        j_ctrl = self.control_planes(spf)
        above = j_ctrl[j_ctrl > j_mid]
        below = j_ctrl[j_ctrl < j_mid][::-1]
        pq_mid = smooth_plane(j_mid, None)
        for chain in (above, below):
            pq = pq_mid
            for j in chain:
                pq = smooth_plane(j, pq)

        # Every other plane takes its own algebraic mesh plus the smoothing
        # correction interpolated from the control planes either side. The
        # correction is zero on the blade, and has no streamwise part on the
        # inlet and outlet, so the interpolated one leaves the blade where it
        # is and moves the inlet and outlet only along the pitch.
        j_all = np.sort(np.append(j_ctrl, j_mid))
        C = np.stack([correction[j] for j in j_all])
        X_smooth = []
        for j in range(len(spf)):
            if correction[j] is None:
                k = np.searchsorted(j_all, j)
                w = (spf[j] - spf[j_all[k - 1]]) / (spf[j_all[k]] - spf[j_all[k - 1]])
                correction[j] = (1.0 - w) * C[k - 1] + w * C[k]
            X_smooth.append(X_tfi[j] + correction[j])

        return self.assemble(machine, i_row, layout, planes, topo, X_smooth)

    def smooth(self, topo, X, plane, pq):
        """Return a plane smoothed from its algebraic mesh `X`.

        With the wall control functions it finished with, to start the next
        plane from given `pq` from the last, and the iterations taken. Here,
        elliptic smoothing with the wall held by control functions and the
        algebraic spacing kept by inherent sources.
        """
        return elliptic.smooth(
            topo,
            X,
            pq=pq,
            scale=plane["control"].spacing.min(),
            omega=self.omega,
            tol=self.tol,
            max_iter=self.max_iter,
            keep_spacing=True,
            keep_slide_angle=self.end_shear,
        )

    def finish(self, xy, plane):
        """Return the blocks (2, ni, nk) of a smoothed plane as they are kept.

        As smoothed here; a subclass may change them, given the `plane`.
        """
        return xy

    def polish(self, topo, X, X_alg):
        """Return a smoothed plane `X` after any final local improvement.

        None here; a subclass may override, given the algebraic plane `X_alg`
        it was smoothed from.
        """
        return X

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

    def blocks(self, machine, i_row, planes, topo, X_smooth):
        """Return bare ember blocks, without patches, from the smoothed planes."""
        nj = len(planes)
        per_plane = [
            self.finish(topo.scatter(X), plane) for X, plane in zip(X_smooth, planes)
        ]
        blocks = []
        for b, name in enumerate(self.labels):
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


class OH(PlaneMesher):
    """Generate a mesh with an O block round each blade, in H blocks."""

    type: ClassVar[str] = "oh"

    labels: ClassVar[tuple] = LABELS

    grids_round_TE: ClassVar[bool] = True

    ds_LE: float = 0.002
    """Surface grid spacing at the leading edge, as a fraction of meridional chord."""

    ds_mid: float = 0.03
    """Largest surface grid spacing, as a fraction of meridional chord. Also the
    size of cells at the outside of the O block and in mid-passage."""

    ER_surf: float = 1.15
    """Expansion ratio of grid spacing along the blade surface."""

    ni_TE: int = 25
    """Number of points across the trailing edge. Odd, so there is one in the
    middle for the O block to be cut at.

    More than an H mesh needs, because the O block turns through half a circle
    round the trailing edge, and does it at two square corners. Clustered into
    those corners at a few wall spacings, with this many points the turn is
    spread over several cells; with nine, the smoothed mesh folds at the
    corners of a blade of two per cent thick trailing edge."""

    LE_split: float = 0.05
    """Fraction of each surface's length from the leading edge to the front
    corners of the O block."""

    TE_split: float = 0.95
    """Fraction of each surface's length from the leading edge to the rear
    corners of the O block."""

    t_O: float = 0.1
    """Thickness of the O block, as a fraction of pitch."""

    ER_O: float = 1.2
    """Expansion ratio away from the blade, in the O block and the H blocks
    either side."""

    ER_stream: float = 1.2
    """Expansion ratio of the streamwise grid towards the inlet and outlet."""

    AR_stream: float = 2.0
    """Aspect ratio of cells at the inlet and outlet boundaries."""

    AR_mix: float = 2.0
    """Aspect ratio of cells at a mixing plane."""

    frac_Z: float = 0.5
    """Where the blocks ahead of and behind the O block are split, as a
    fraction of the way from the blade to the inlet, outlet or mixing plane."""

    shift: float = 0.0
    """How far the H blocks either side of the blade meet the periodic line
    from level with each other, from zero to one.

    At zero, the two sides have the same count of nodes, and the blocks on
    each meet the inlet and outlet H blocks at the same `m'`, so the grid
    lines across the passage run pitchwise on average whatever the stagger.
    At one, each meets the periodic line where the O block's normal at its
    corner lands, the two sides' counts differing by the cells between, and
    those cells taken up along the periodic line by the faces of the inlet
    and outlet H blocks.
    """

    def __post_init__(self):
        self._check_end_shear()
        if self.ni_TE < 3 or not self.ni_TE % 2:
            raise ValueError(f"ni_TE={self.ni_TE} must be odd and at least three.")
        if not 0.0 < self.LE_split < self.TE_split < 1.0:
            raise ValueError(
                f"Need 0 < LE_split < TE_split < 1, got {self.LE_split}, "
                f"{self.TE_split}."
            )
        if not 0.0 < self.frac_Z < 1.0:
            raise ValueError(f"frac_Z={self.frac_Z} must be between zero and one.")
        if not 0.0 <= self.shift <= 1.0:
            raise ValueError(f"shift={self.shift} must be between zero and one.")

    def layout(self, machine, i_row, spacing):
        """Fix the counts and distributions of one row, at mid-span."""
        annulus = machine.annulus
        row = machine.rows[i_row]
        blade = row.blade
        n_row = len(machine.rows)
        pitch = 2.0 * np.pi / row.n_blade

        # Inlet and outlet of the row: the domain boundary, or half way across
        # the gap to a mixing plane.
        L_up = 1.0 if i_row == 0 else 0.5
        L_dn = 1.0 if i_row == n_row - 1 else 0.5
        m_LE, m_TE = 2.0 * i_row + 1.0, 2.0 * i_row + 2.0
        m_in, m_out = m_LE - L_up, m_TE + L_dn
        m_Zin = m_LE - self.frac_Z * L_up
        m_Zout = m_TE + self.frac_Z * L_dn

        cmap = ConformalMap(annulus, 0.5, m_in, m_out)
        upper, lower, te = _blade_pieces(blade, cmap, 0.5)
        S = np.array([util.arc_length(upper), util.arc_length(lower)])

        # Spacings in conformal units, scaled from physical at the mean radius.
        r_ref = float(np.mean(_by_theta(blade, 0.5)[0][1]))
        chord = blade.evaluate_chord(0.5)
        to_c = chord / r_ref
        ds_LE = self.ds_LE * to_c
        ds_mid = self.ds_mid * to_c
        t_TE = util.arc_length(te)
        n_te = int(self.ni_TE)
        d_wall = float(spacing.surface[i_row])
        # Into the trailing edge corners, a few wall spacings, so the O block
        # turns the corner over several cells rather than one node. A round
        # trailing edge has no corners, and is evenly spaced round its arc.
        if blade.round_TE:
            ds_TE = t_TE / (n_te - 1)
        else:
            ds_TE = min(CORNER * d_wall / r_ref, t_TE / (n_te - 1))

        # Along the blade, one distribution from leading edge to trailing edge
        # corner over the mean of the two surfaces, cut at the nodes nearest the
        # corners of the O block, so the spacing runs on smoothly across them.
        S_mean = S.mean()
        f_surf = clusterfunc.double.free(
            ds_LE / S_mean,
            min(ds_TE, ds_mid) / S_mean,
            ds_mid / S_mean,
            self.ER_surf,
            mult=1,
        )
        f_surf = util.resample(f_surf, self.resolution_factor)
        a = int(np.argmin(np.abs(f_surf - self.LE_split)))
        b = int(np.argmin(np.abs(f_surf - self.TE_split)))
        a = max(a, 1)
        b = min(max(b, a + 1), len(f_surf) - 2)
        split = (float(f_surf[a]), float(f_surf[b]))
        f_front = _mult(f_surf[: a + 1] / f_surf[a], 4)
        f_side = _mult((f_surf[a : b + 1] - f_surf[a]) / (f_surf[b] - f_surf[a]), 8)
        f_rear = (f_surf[b:] - f_surf[b]) / (1.0 - f_surf[b])

        # Rear piece plus half the trailing edge must be a multigrid count.
        n_te_half = (n_te - 1) // 2
        n_cell = len(f_rear) - 1 + n_te_half
        f_rear = util.resample_to(f_rear, len(f_rear) + (-n_cell) % 8)
        if blade.round_TE:
            f_te = np.linspace(0.0, 1.0, n_te)
        else:
            f_te = clusterfunc.symmetric.fixed(ds_TE / t_TE, n_te)

        # Away from the wall, to cells along-surface sized at the outside.
        t_O = self.t_O * pitch
        d_wall_c = d_wall / r_ref
        f_O = clusterfunc.single.free(
            d_wall_c / t_O, max(ds_mid, d_wall_c) / t_O, self.ER_O
        )
        f_O = _mult(util.resample(f_O, self.resolution_factor), 8)
        dk_O = (f_O[-1] - f_O[-2]) * t_O

        # Where the H blocks either side meet the periodic line, in whole
        # multigrid steps of the mean spacing along it, which fixes how many
        # more nodes the upper side has than the lower.
        mp_Z = cmap.mp_from_m([m_Zin, m_Zout])
        geom = self._plane_geometry(
            blade, cmap, pitch, split, f_front, f_side, f_side, f_rear, f_te, t_O, mp_Z
        )
        P1u, P1d, P2u, P2d = (geom[k][0] for k in ("P1u", "P1d", "P2u", "P2d"))
        ds_per = 0.5 * (P2u - P1u + P2d - P1d) / (len(f_side) - 1)
        s_f = 8 * round((P1d - P1u) / ds_per / 8)
        s_r = 8 * round((P2u - P2d) / ds_per / 8)
        n_u = len(f_side) - 1 + 8 * round((s_f + s_r) / 16)
        n_l = n_u - s_f - s_r
        if min(n_u, n_l) < 8:
            raise ValueError(
                f"Row {i_row}: the H blocks either side of the blade meet the "
                f"periodic line {s_f} and {s_r} cells apart at the front and "
                f"rear, more than the {len(f_side) - 1} along a side; reduce "
                f"shift={self.shift}."
            )
        f_side_u = util.resample_to(f_side, n_u + 1)
        f_side_l = util.resample_to(f_side, n_l + 1)

        # The H blocks either side, from the O block out to the periodic line.
        h_up = max(
            np.linalg.norm(geom["P1u"] - geom["C1"]),
            np.linalg.norm(geom["P2u"] - geom["C2"]),
        )
        h_dn = max(
            np.linalg.norm(geom["P1d"] - geom["C4"]),
            np.linalg.norm(geom["P2d"] - geom["C3"]),
        )
        f_Hup = _single(dk_O / h_up, ds_mid / h_up, self.ER_O)
        f_Hdown = _single(dk_O / h_dn, ds_mid / h_dn, self.ER_O)
        f_Hup = _mult(util.resample(f_Hup, self.resolution_factor), 8)
        f_Hdown = _mult(util.resample(f_Hdown, self.resolution_factor), 8)

        # Streamwise, from the largest cells in the passage at the split lines,
        # rather than the mean pitchwise spacing, which the clustering round
        # the leading edge makes far finer than the flow there needs.
        d_pitch = ds_mid
        mp_in, mp_Zin, mp_Zout, mp_out = cmap.mp_from_m([m_in, m_Zin, m_Zout, m_out])
        L_Hin = min(P1u, P1d) - mp_Zin
        L_Hout = mp_Zout - max(P2u, P2d)
        if L_Hin <= 0.0 or L_Hout <= 0.0:
            raise ValueError(
                f"Row {i_row}: the O block reaches past the split at "
                f"frac_Z={self.frac_Z}; move the split further from the blade "
                f"or thin the O block with t_O."
            )
        AR_up = self.AR_stream if i_row == 0 else self.AR_mix
        AR_dn = self.AR_stream if i_row == n_row - 1 else self.AR_mix

        f_Hin = _double(d_pitch / L_Hin, dk_O / L_Hin, d_pitch / L_Hin, self.ER_stream)
        f_Hout = _double(
            dk_O / L_Hout, d_pitch / L_Hout, d_pitch / L_Hout, self.ER_stream
        )
        L_up = mp_Zin - mp_in
        L_dn = mp_out - mp_Zout
        f_up = 1.0 - np.flip(
            _single(d_pitch / L_up, AR_up * d_pitch / L_up, self.ER_stream)
        )
        f_dn = _single(d_pitch / L_dn, AR_dn * d_pitch / L_dn, self.ER_stream)
        f_Hin, f_Hout, f_up, f_dn = (
            _mult(util.resample(f, self.resolution_factor), 8)
            for f in (f_Hin, f_Hout, f_up, f_dn)
        )

        layout = _Layout(
            pitch=pitch,
            m_in=m_in,
            m_Zin=m_Zin,
            m_Zout=m_Zout,
            m_out=m_out,
            d_wall=d_wall,
            f_front=f_front,
            f_side_u=f_side_u,
            f_side_l=f_side_l,
            f_rear=f_rear,
            f_te=f_te,
            f_O=f_O,
            f_Hup=f_Hup,
            f_Hdown=f_Hdown,
            f_up=f_up,
            f_Hin=f_Hin,
            f_Hout=f_Hout,
            f_dn=f_dn,
            split=split,
            t_O=t_O,
            s_f=s_f,
        )
        logger.debug(
            f"Row {i_row} OH layout: O {layout.ni_O}x{len(f_O)}, "
            f"H_up {n_u + 1}x{len(f_Hup)}, H_down {n_l + 1}x{len(f_Hdown)}, "
            f"junctions {s_f} and {s_r} cells apart at the front and rear, "
            f"H_in {len(f_Hin)}, H_out {len(f_Hout)}, up {len(f_up)}, dn {len(f_dn)}"
        )
        return layout

    def _plane_geometry(
        self,
        blade,
        cmap,
        pitch,
        split,
        f_front,
        f_side_u,
        f_side_l,
        f_rear,
        f_te,
        t_O,
        mp_Z,
    ):
        """Return the blade loop, O block outline and corner points of a plane.

        With where the H blocks either side would meet the periodic line, at
        :attr:`shift` of the way from level with each other to where the O
        block's normals at its corners land: `P1u` and `P2u` on the upper
        line, `P1d` and `P2d` on the lower. Never more than half way from
        level to the split lines at `mp_Z`, so the inlet and outlet H blocks
        keep some length.
        """
        upper, lower, te = _blade_pieces(blade, cmap, cmap.spf)
        loop = _blade_loop(
            upper, lower, te, *split, f_front, f_side_u, f_side_l, f_rear, f_te
        )
        normal = _outward_normals(loop)
        outer = loop + t_O * normal

        n_rear = len(f_rear) + (len(f_te) - 1) // 2
        i3 = n_rear - 1
        i4 = i3 + len(f_side_l) - 1
        i1 = i4 + 2 * len(f_front) - 2
        i2 = i1 + len(f_side_u) - 1

        periodic = _periodic_line(blade, cmap, pitch)
        mp_P1 = 0.5 * (outer[0, i1] + outer[0, i4])
        mp_P2 = 0.5 * (outer[0, i2] + outer[0, i3])

        lo = mp_P1 - 0.5 * (mp_P1 - mp_Z[0])
        hi = mp_P2 + 0.5 * (mp_Z[1] - mp_P2)

        def meet(i, offset, mp_level):
            mp = _landing(outer[:, i], normal[:, i], periodic, offset, pitch)
            mp = np.clip(mp_level + self.shift * (mp - mp_level), lo, hi)
            return np.array([mp, periodic(mp) + offset])

        return {
            "loop": loop,
            "outer": outer,
            "periodic": periodic,
            "C1": outer[:, i1],
            "C2": outer[:, i2],
            "C3": outer[:, i3],
            "C4": outer[:, i4],
            "P1u": meet(i1, 0.0, mp_P1),
            "P1d": meet(i4, -pitch, mp_P1),
            "P2u": meet(i2, 0.0, mp_P2),
            "P2d": meet(i3, -pitch, mp_P2),
        }

    def plane(self, machine, i_row, layout, spf):
        """Return the algebraic mesh of one constant-span plane of a row."""
        row = machine.rows[i_row]
        blade = row.blade
        pitch = layout.pitch
        cmap = ConformalMap(machine.annulus, spf, layout.m_in, layout.m_out)

        g = self._plane_geometry(
            blade,
            cmap,
            pitch,
            layout.split,
            layout.f_front,
            layout.f_side_u,
            layout.f_side_l,
            layout.f_rear,
            layout.f_te,
            layout.t_O,
            cmap.mp_from_m([layout.m_Zin, layout.m_Zout]),
        )
        loop, outer, periodic = g["loop"], g["outer"], g["periodic"]
        C1, C2, C3, C4 = (g[k] for k in ("C1", "C2", "C3", "C4"))
        i3, i4, i1, i2 = layout.i_corners
        iu0, iu1, il0, il1, n_per = layout.i_per

        mp_in, mp_Zin, mp_Zout, mp_out = cmap.mp_from_m(
            [layout.m_in, layout.m_Zin, layout.m_Zout, layout.m_out]
        )

        def upper_line(mp):
            return np.stack((mp, periodic(mp)))

        def lower_line(mp):
            return np.stack((mp, periodic(mp) - pitch))

        def along_periodic(mp0, mp1, f):
            return upper_line(mp0 + f * (mp1 - mp0))

        # O block: straight lines from the blade out along the normals.
        fO = layout.f_O.reshape(1, 1, -1)
        O = loop[..., None] + fO * (outer - loop)[..., None]

        # The stretch of periodic line between the inlet and outlet H blocks.
        # Each side's H block spaces its part as the O block face it faces,
        # and where both lie on the same cells, the two are averaged.
        mp_u = g["P1u"][0] + _norm_arc(outer[:, i1 : i2 + 1]) * (
            g["P2u"][0] - g["P1u"][0]
        )
        mp_l = g["P1d"][0] + _norm_arc(outer[:, i3 : i4 + 1][:, ::-1]) * (
            g["P2d"][0] - g["P1d"][0]
        )
        mp_per = np.zeros(n_per + 1)
        count = np.zeros(n_per + 1)
        mp_per[iu0 : iu1 + 1] += mp_u
        count[iu0 : iu1 + 1] += 1.0
        mp_per[il0 : il1 + 1] += mp_l
        count[il0 : il1 + 1] += 1.0
        mp_per = np.maximum.accumulate(mp_per / count)
        P1u, P2u = upper_line(mp_per[[iu0, iu1]]).T
        P1d, P2d = lower_line(mp_per[[il0, il1]]).T

        # Either side, from the O block to the periodic line.
        Hup = _tfi(
            outer[:, i1 : i2 + 1],
            upper_line(mp_per[iu0 : iu1 + 1]),
            _line(C1, P1u, layout.f_Hup),
            _line(C2, P2u, layout.f_Hup),
        )
        g_dn = 1.0 - np.flip(layout.f_Hdown)
        Hdown = _tfi(
            lower_line(mp_per[il0 : il1 + 1]),
            outer[:, i3 : i4 + 1][:, ::-1],
            _line(P1d, C4, g_dn),
            _line(P2d, C3, g_dn),
        )

        # Ahead of the blade: the face it shares with the blocks round the O
        # block, running up the pitch, and a split line with the same spacing.
        # Whichever side's H block starts further downstream, the face runs
        # along the periodic line to it first.
        right_in = np.concatenate(
            (
                lower_line(mp_per[: il0 + 1])[:, :-1],
                Hdown[:, 0, :],
                O[:, i4 : i1 + 1, -1][:, 1:],
                Hup[:, 0, 1:],
                upper_line(mp_per[iu0::-1])[:, 1:],
            ),
            axis=1,
        )
        f_k_in = _norm_arc(right_in)
        Q1 = np.array([mp_Zin, periodic(mp_Zin)])
        Zin = np.stack((np.full_like(f_k_in, mp_Zin), Q1[1] - pitch + pitch * f_k_in))
        top_in = along_periodic(mp_Zin, mp_per[0], layout.f_Hin)
        top_in[:, -1] = right_in[:, -1]
        Hin = _tfi(top_in - pitch * _J[:, None], top_in, Zin, right_in)
        Hin[:, -1, 0] = right_in[:, 0]
        Hin[:, -1, -1] = right_in[:, -1]
        if iu0 or il0:
            Hin = _laplace(Hin)

        # Behind the blade, the same.
        rear = np.concatenate((O[:, i3::-1, -1], O[:, -2 : i2 - 1 : -1, -1]), axis=1)
        left_out = np.concatenate(
            (
                lower_line(mp_per[il1:][::-1])[:, :-1],
                Hdown[:, -1, :],
                rear[:, 1:],
                Hup[:, -1, 1:],
                upper_line(mp_per[iu1:])[:, 1:],
            ),
            axis=1,
        )
        f_k_out = _norm_arc(left_out)
        Q2 = np.array([mp_Zout, periodic(mp_Zout)])
        Zout = np.stack(
            (np.full_like(f_k_out, mp_Zout), Q2[1] - pitch + pitch * f_k_out)
        )
        top_out = along_periodic(mp_per[-1], mp_Zout, layout.f_Hout)
        top_out[:, 0] = left_out[:, -1]
        Hout = _tfi(top_out - pitch * _J[:, None], top_out, left_out, Zout)
        if iu1 < n_per or il1 < n_per:
            Hout = _laplace(Hout)

        # Out to the boundaries.
        Q0 = np.array([mp_in, periodic(mp_in)])
        inlet = np.stack((np.full_like(f_k_in, mp_in), Q0[1] - pitch + pitch * f_k_in))
        top_up_blk = along_periodic(mp_in, mp_Zin, layout.f_up)
        top_up_blk[:, -1] = Q1
        up = _tfi(top_up_blk - pitch * _J[:, None], top_up_blk, inlet, Zin)

        Q3 = np.array([mp_out, periodic(mp_out)])
        outlet = np.stack(
            (np.full_like(f_k_out, mp_out), Q3[1] - pitch + pitch * f_k_out)
        )
        top_dn_blk = along_periodic(mp_Zout, mp_out, layout.f_dn)
        top_dn_blk[:, 0] = Q2
        dn = _tfi(top_dn_blk - pitch * _J[:, None], top_dn_blk, Zout, outlet)

        xy = [O, Hup, Hdown, Hin, Hout, up, dn]
        for name, blk in zip(LABELS, xy):
            assert np.isfinite(blk).all(), f"non-finite coordinates in {name}"

        # The blade is fixed. The inlet and outlet slide along the pitch, so
        # their pitchwise distribution is whatever the mesh inside settles to
        # at each span. Each node keeps its algebraic offset from the node
        # inside it, so the first grid line keeps the angle of the periodic
        # line there instead of turning square to the boundary, and the end
        # blocks stay sheared along the flow.
        fixed = [None] * len(xy)
        fixed[0] = np.zeros(O.shape[1:], dtype=bool)
        fixed[0][:, 0] = True
        slide = [None] * len(xy)
        slide[5] = np.zeros(up.shape[1:], dtype=bool)
        slide[5][0, :] = True
        slide[6] = np.zeros(dn.shape[1:], dtype=bool)
        slide[6][-1, :] = True

        # Wall spacing in conformal units, against the local radius.
        r_wall = cmap.xr_from_mp(loop[0])[1]
        # Left to itself at the trailing edge corners, which have no normal; a
        # round trailing edge has a normal everywhere.
        weight = np.ones(layout.ni_O)
        if not blade.round_TE:
            for i_c in layout.i_te_corners:
                weight[i_c] = 0.0
                weight[i_c - 1] = weight[i_c + 1] = 0.5
        control = elliptic.WallControl(
            block=0, spacing=layout.d_wall / r_wall, closed=True, weight=weight
        )

        return {
            "xy": xy,
            "fixed": fixed,
            "slide": slide,
            "control": control,
            "cmap": cmap,
        }

    def assemble(self, machine, i_row, layout, planes, topo, X_smooth):
        """Return ember blocks from the smoothed planes of a row."""
        n_row = len(machine.rows)
        blocks = self.blocks(machine, i_row, planes, topo, X_smooth)
        O, Hup, Hdown, Hin, Hout, up, dn = blocks
        i3, i4, i1, i2 = layout.i_corners
        iu0, iu1, il0, il1, n_per = layout.i_per
        nkd = Hdown.shape[2] - 1
        nku = Hup.shape[2] - 1
        nk_in = Hin.shape[2] - 1
        nk_out = Hout.shape[2] - 1
        n_front = layout.n_front - 1
        n_rear = layout.n_rear - 1
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
        Hup.patches.extend([Per(k=0), Per(i=0), Per(i=-1)])
        Hdown.patches.extend([Per(k=-1), Per(i=0), Per(i=-1)])
        k = np.cumsum([il0, nkd, n_front, nku])
        assert k[-1] + iu0 == nk_in
        Hin.patches.extend(
            [Per(i=0), Per(k=0), Per(k=-1)]
            + [Per(i=-1, k=(a, b)) for a, b in itertools.pairwise(k)]
        )
        k = np.cumsum([n_per - il1, nkd, n_rear, n_rear, nku])
        assert k[-1] + n_per - iu1 == nk_out
        Hout.patches.extend(
            [Per(i=-1), Per(k=0), Per(k=-1)]
            + [Per(i=0, k=(a, b)) for a, b in itertools.pairwise(k)]
        )

        # Along the periodic line between the inlet and outlet H blocks, each
        # face on one line split where a face on the other begins or ends, so
        # every patch has one partner. Each face as its cells along the line,
        # and the patch over cells `a` to `b` of it.
        upper = [
            (0, iu0, lambda a, b: Per(i=-1, k=(nk_in - b, nk_in - a)), Hin),
            (iu0, iu1, lambda a, b: Per(i=(a - iu0, b - iu0), k=-1), Hup),
            (
                iu1,
                n_per,
                lambda a, b: Per(i=0, k=(nk_out - n_per + a, nk_out - n_per + b)),
                Hout,
            ),
        ]
        lower = [
            (0, il0, lambda a, b: Per(i=-1, k=(a, b)), Hin),
            (il0, il1, lambda a, b: Per(i=(a - il0, b - il0), k=0), Hdown),
            (il1, n_per, lambda a, b: Per(i=0, k=(n_per - b, n_per - a)), Hout),
        ]
        breaks = sorted({0, iu0, iu1, il0, il1, n_per})
        for a, b in itertools.pairwise(breaks):
            for faces in (upper, lower):
                for lo, hi, patch, block in faces:
                    if lo <= a and b <= hi:
                        block.patches.append(patch(a, b))

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
                Per(i=0),
                Per(k=0),
                Per(k=-1),
            ]
        )
        return blocks


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


def _blade_loop(
    upper, lower, te, LE_split, TE_split, f_front, f_side_u, f_side_l, f_rear, f_te
):
    """Return the nodes round the blade, clockwise from mid trailing edge.

    Clockwise in `(m', theta)`, so that with `k` outward the O block has the
    same handedness as H blocks running `i` downstream and `k` up the pitch.
    """
    S_u = util.arc_length(upper)
    S_l = util.arc_length(lower)

    def piece(curve, S, a, b, f):
        return _at_arc(curve, S * (a + f * (b - a)))

    # Each surface from its leading edge to its trailing edge corner.
    u_front = piece(upper, S_u, 0.0, LE_split, f_front)
    u_side = piece(upper, S_u, LE_split, TE_split, f_side_u)
    u_rear = piece(upper, S_u, TE_split, 1.0, f_rear)
    l_front = piece(lower, S_l, 0.0, LE_split, f_front)
    l_side = piece(lower, S_l, LE_split, TE_split, f_side_l)
    l_rear = piece(lower, S_l, TE_split, 1.0, f_rear)

    # Across the trailing edge, upper corner to lower.
    te = _at_arc(te, util.arc_length(te) * f_te)
    n_half = (len(f_te) - 1) // 2

    return np.concatenate(
        (
            te[:, n_half:],
            l_rear[:, ::-1][:, 1:],
            l_side[:, ::-1][:, 1:],
            l_front[:, ::-1][:, 1:],
            u_front[:, 1:],
            u_side[:, 1:],
            u_rear[:, 1:],
            te[:, 1 : n_half + 1],
        ),
        axis=1,
    )


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


def _landing(point, normal, periodic, offset, pitch):
    """Return `m'` where a ray from `point` along `normal` meets a periodic line.

    The line `theta = periodic(m') + offset`. Where the ray misses it within
    two pitches, the `m'` of the point.
    """
    s = np.linspace(0.0, 2.0 * pitch, N_RAY)
    pts = point[:, None] + s[None] * normal[:, None]
    gap = pts[1] - periodic(pts[0]) - offset
    k = np.flatnonzero(np.sign(gap[1:]) != np.sign(gap[:-1]))
    if not len(k):
        return float(point[0])
    k = k[0]
    w = gap[k] / (gap[k] - gap[k + 1])
    return float(pts[0, k] + w * (pts[0, k + 1] - pts[0, k]))


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


def _laplace(blk):
    """Return a block (2, ni, nk) with its inside nodes each the mean of their
    four neighbours, its edges as they are.

    Where a face doubles back along the edge next to it, the interpolation
    between opposite faces folds; this does not, the boundary being near
    enough convex.
    """
    ni, nk = blk.shape[1:]
    idx = np.arange((ni - 2) * (nk - 2)).reshape(ni - 2, nk - 2)
    rows, cols = [idx.ravel()], [idx.ravel()]
    vals = [np.full(idx.size, 4.0)]
    rhs = np.zeros((2, idx.size))
    for di, dk in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        i = np.arange(1, ni - 1)[:, None] + di
        k = np.arange(1, nk - 1)[None, :] + dk
        i, k = np.broadcast_arrays(i, k)
        inside = (i > 0) & (i < ni - 1) & (k > 0) & (k < nk - 1)
        rows.append(idx[inside])
        cols.append(idx[i[inside] - 1, k[inside] - 1])
        vals.append(-np.ones(inside.sum()))
        np.add.at(rhs, (slice(None), idx[~inside]), blk[:, i[~inside], k[~inside]])
    A = scipy.sparse.csc_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(idx.size, idx.size),
    )
    solve = scipy.sparse.linalg.factorized(A)
    out = blk.copy()
    for c in range(2):
        out[c, 1:-1, 1:-1] = solve(rhs[c]).reshape(ni - 2, nk - 2)
    return out


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
