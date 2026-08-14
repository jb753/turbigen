"""H-topology meshing.

One block per blade row, each an H mesh: the streamwise index runs from the
upstream boundary to the downstream one, the pitchwise index between the two
sides of the passage, and the spanwise index from hub to casing.

The mesh mathematics is carried over from the package this replaces. What is
new is the boundary of the class: it reads geometry from a
:class:`~turbigen2.machine.Machine` and wall spacings from a
:class:`~turbigen2.mesh.WallSpacing` handed to it by
:meth:`~turbigen2.mesh.Mesher.mesh`, rather than from an adapter object built
by the config, and it no longer plots, logs memory use, or carries the three
options that no configuration ever set.
"""

import dataclasses
import logging
from typing import ClassVar

import numpy as np

import ember.block
import ember.grid
import ember.patch
from turbigen import clusterfunc, util
from turbigen2.mesh import Mesher

logger = logging.getLogger("turbigen")


@dataclasses.dataclass(frozen=True, slots=True)
class _RowGeometry:
    pitch_theta: float
    chord_mid: np.ndarray
    pitch_chord_ref: np.ndarray
    pitch_chord_max: float
    pitch_rtheta_max: float
    drt_norm: float
    AR_row: float
    ist: int


@dataclasses.dataclass(frozen=True, slots=True)
class _RowGrids:
    pitch_frac_nom: np.ndarray
    nk_not_resampled: int
    span_frac: np.ndarray
    stream_frac: np.ndarray
    ile: int
    ite: int
    ni: int
    nj: int
    nk: int
    L: tuple
    tte: float


@dataclasses.dataclass(frozen=True, slots=True)
class _RowCoords:
    xr: np.ndarray
    theta_lim: np.ndarray
    pitch_frac_relax: np.ndarray
    njtip: int
    ite: int


class H(Mesher):
    """Generate a mesh using H topology for each row."""

    type: ClassVar[str] = "h"

    ER_stream: float = 1.2
    """Expansion ratio of streamwise grid from first LE to inlet boundary."""

    AR_stream: float = 2.0
    """Aspect ratio in blade-to-blade plane of cells at outlet boundary."""

    AR_passage: float = 1.0
    """Nominal aspect ratio in blade-to-blade plane of mid-passage cells."""

    AR_merid: float = 1.0
    """Aspect ratio of mid-chord cells in meridional plane."""

    ER_span: float = 1.2
    """Expansion ratio of spanwise grid away from hub and casing."""

    dm_LE: float = 0.001
    """Streamwise grid spacing at LE, normalised by meridional chord."""

    ni_TE: int = 9
    """Number of streamwise points across trailing edge."""

    dm_TE: float = 0.05
    """Normalised meridional length over which to cluster the TE points, 0. for
    the true actual TE."""

    dspf_mid: float = 0.03
    """Spanwise grid spacing at midspan, as a fraction of span."""

    ER_pitch: float = 1.2
    """Expansion ratio away from aerofoil surfaces."""

    nk_min: int = 37
    """Minimum number of pitchwise grid points per row."""

    nchord_relax: float = 1.0
    """Number of meridional chords over which pitchwise clustering is relaxed."""

    resolution_factor: float = 1.0
    """Multiply the number of points in each direction, keeping relative spacings."""

    skew_max: float = 45.0
    """Largest angle the mesh is skewed to, upstream and downstream [deg]."""

    deswirl: bool = False
    """If True, blend skew to axial over the outer 50% of the downstream gap
    of the last row (fully skewed at TE, fully axial at the domain exit).
    Only affects the downstream side of the final row."""

    AR_cusp: float = 0.0
    """Length of the trailing edge cusp, as a multiple of TE thickness."""

    ni_cusp: int = 0
    """Number of streamwise points along the trailing edge cusp."""

    gap_contraction: float = 0.6
    """Fraction of the tip gap over which the blade is pinched to zero
    thickness."""

    def __post_init__(self):
        if self.ni_cusp and self.dm_TE != 0.0:
            raise ValueError("ni_cusp requires dm_TE = 0.0")

    def forward(self, machine, spacing):
        """Generate a Grid for a designed machine."""
        n_row = len(machine.blades)
        dspf_hub, dspf_casing, tip_ref = self._normalise_spacings(machine, spacing)

        blocks = []
        for i_row in range(n_row):
            geom = self._row_geometry(machine, i_row, spacing)
            grids = self._row_1d_grids(
                machine, i_row, geom, dspf_hub, dspf_casing, tip_ref, n_row
            )
            coords = self._row_coords(machine, i_row, geom, grids, tip_ref, n_row)
            blocks.append(self._row_block(machine, i_row, geom, grids, coords, n_row))
            del coords

        grid = ember.grid.Grid(blocks)
        self._stitch_mixing_planes(grid)
        grid.connectivity.periodic.pair()
        return grid

    def _normalise_spacings(self, machine, spacing):
        """Compute normalised hub/casing spacings and tip_ref."""
        annulus = machine.annulus
        span_all = np.array(
            [
                np.mean(annulus.span(np.arange(i_row * 2, i_row * 2 + 2)))
                for i_row in range(len(machine.blades))
            ]
        )
        tip_all = np.array([blade.tip_gap for blade in machine.blades])
        tip_ref = np.max(tip_all / span_all)
        span_ref = annulus.span(1)
        return spacing.hub / span_ref, spacing.casing / span_ref, tip_ref

    def _row_geometry(self, machine, i_row, spacing):
        """Derive scalar geometry quantities for a single row."""
        annulus = machine.annulus
        pitch_theta = 2.0 * np.pi / float(machine.blades[i_row].n_blade)

        mrow = np.linspace(2.0 * i_row + 1.0, 2.0 * i_row + 2)
        xr_hub = annulus.evaluate_xr(mrow, 0.0)
        xr_cas = annulus.evaluate_xr(mrow, 1.0)
        xr_mid = annulus.evaluate_xr(mrow, 0.5)

        ist = 2 * i_row
        chord_hub = annulus.chords(0.0)[ist : ist + 3]
        chord_mid = annulus.chords(0.5)[ist : ist + 3]
        chord_cas = annulus.chords(1.0)[ist : ist + 3]

        pitch_rtheta_hub = pitch_theta * xr_hub[1]
        pitch_rtheta_cas = pitch_theta * xr_cas[1]

        pitch_chord_hub = pitch_rtheta_hub / chord_hub[1]
        pitch_chord_cas = pitch_rtheta_cas / chord_cas[1]

        pitch_chord_ref = pitch_theta * xr_mid[1].mean() / chord_mid
        pitch_chord_max = np.maximum(pitch_chord_hub.max(), pitch_chord_cas.max())
        pitch_rtheta_max = np.maximum(pitch_rtheta_hub.max(), pitch_rtheta_cas.max())

        drt_norm = spacing.surface[i_row] / pitch_rtheta_max

        span_row = np.mean(annulus.span(np.arange(i_row * 2, i_row * 2 + 2)))
        AR_row = span_row / chord_mid[1]

        return _RowGeometry(
            pitch_theta=pitch_theta,
            chord_mid=chord_mid,
            pitch_chord_ref=pitch_chord_ref,
            pitch_chord_max=pitch_chord_max,
            pitch_rtheta_max=pitch_rtheta_max,
            drt_norm=drt_norm,
            AR_row=AR_row,
            ist=ist,
        )

    def _row_1d_grids(
        self, machine, i_row, geom, dspf_hub, dspf_casing, tip_ref, n_row
    ):
        """Generate the three 1-D grid vectors for a row."""
        # Pitchwise
        safety_fac = 1.01
        pitch_frac_nom = self.pitchwise_grid(
            geom.drt_norm, geom.pitch_chord_max * safety_fac, geom.AR_row
        )
        logger.debug(
            f"Nominal pitchwise grid: {geom.drt_norm}, "
            f"{geom.pitch_chord_max}, {geom.AR_row}"
        )
        pitch_frac_not_resampled = self.pitchwise_grid(
            geom.drt_norm,
            geom.pitch_chord_max * safety_fac,
            geom.AR_row,
            resample=False,
        )
        self.pitchwise_grid_fixed_npts(
            geom.drt_norm,
            geom.pitch_chord_max,
            geom.AR_row,
            len(pitch_frac_not_resampled),
        )
        nk_not_resampled = len(pitch_frac_not_resampled)
        nk = len(pitch_frac_nom)
        assert not (nk - 1) % 8, f"nk-1={nk - 1} not divisible by 8"
        logger.debug(f"nk={nk}, nk_not_resampled={nk_not_resampled}")

        # Spanwise
        span_frac = self.spanwise_grid(
            dspf_hub, dspf_casing, tip_ref * self.gap_contraction
        )
        nj = len(span_frac)

        # Streamwise: choose inlet/exit lengths
        if n_row == 1:
            L = (1.0, 1.0)
        elif i_row == 0:
            L = (1.0, 0.5)
        elif i_row == (n_row - 1):
            L = (0.5, 1.0)
        else:
            L = (0.5, 0.5)

        if self.dm_TE:
            tte = 1.0 - self.dm_TE
        else:
            xrt_u, xrt_l = machine.blades[i_row].evaluate_section(0.5)
            tq = np.linspace(0.8, 1.0, 500)
            _, _, tte = _theta_limits(tq, xrt_u, xrt_l, np.array((0, 1)))

        stream_frac, ile, ite = self.streamwise_grid(
            geom.pitch_chord_ref,
            nk_not_resampled,
            L,
            geom.AR_row,
            tte,
            ni_cusp=self.ni_cusp + 8,
        )

        ni = len(stream_frac)
        assert not ((ni - 1) % 8), f"ni-1={ni - 1} not divisible by 8"
        assert len(np.unique(stream_frac)) == ni
        assert (np.diff(stream_frac) > 0.0).all()

        return _RowGrids(
            pitch_frac_nom=pitch_frac_nom,
            nk_not_resampled=nk_not_resampled,
            span_frac=span_frac,
            stream_frac=stream_frac,
            ile=ile,
            ite=ite,
            ni=ni,
            nj=nj,
            nk=nk,
            L=L,
            tte=tte,
        )

    def _row_coords(self, machine, i_row, geom, grids, tip_ref, n_row):
        """Build 2-D xr, theta_lim and 3-D pitch_frac_relax for a row."""
        annulus = machine.annulus
        blade = machine.blades[i_row]
        ni, nj, nk = grids.ni, grids.nj, grids.nk
        span_frac = grids.span_frac
        stream_frac = grids.stream_frac
        ite = grids.ite

        spfr = span_frac.reshape(1, -1)
        tte = grids.tte  # midspan TE-corner fraction the streamwise grid was built on

        # Per-span TE corner alignment. The grid's TE breakpoint t = tte is fixed
        # at the midspan corner, but the true squared-TE corner fraction drifts
        # with span, so off-midspan the corner falls between streamwise nodes.
        # Find the corner fraction at each span and warp the chord mapping so the
        # grid knot t = tte lands on that span's corner -- keeping ite coincident
        # with the TE corner at every span. Without this the corner is only
        # resolved at midspan, forcing add_cusp to detect and re-square it, which
        # is fragile and sensitive to the spanwise node count.
        tte_span = np.full(nj, tte)
        tq_te = np.linspace(0.8, 1.0, 500)
        for j in range(nj):
            xrt_u_j, xrt_l_j = blade.evaluate_section(span_frac[j])
            _, _, tte_j = _theta_limits(tq_te, xrt_u_j, xrt_l_j, np.array((0, 1)))
            if tte_j is not None:
                tte_span[j] = tte_j

        # 1. Meridional xr on a (ni, nj) grid
        stream_frac_span = np.broadcast_to(stream_frac.reshape(-1, 1), (ni, nj)).copy()
        for j in range(nj):
            # Warp so grid breakpoints [-1, 0, tte, 1, 2] map to section
            # [-1, 0, tte_span[j], 1, 2]; with mlim = (0, 1) the only moving
            # interior knot is the corner at grid t = tte -> tte_span[j].
            stream_frac_span[:, j] = np.interp(
                stream_frac_span[:, j],
                [-1, 0, tte, 1, 2],
                [-1, 0, tte_span[j], 1, 2],
            )
        xr = annulus.evaluate_xr(stream_frac_span + geom.ist + 1.0, spfr)

        # 2. Pitchwise clustering
        relax = self.pitchwise_relaxation(stream_frac, geom.pitch_chord_ref).reshape(
            -1, 1, 1
        )
        uniform = np.linspace(0.0, 1.0, nk).reshape(1, 1, -1)
        assert np.all(relax >= 0.0) and np.all(relax <= 1.0)

        Theta = blade.chi(0.5)
        pitch_frac_clust = np.tile(grids.pitch_frac_nom.reshape(1, 1, -1), (ni, nj, 1))

        # 3. Theta limits from blade sections
        theta_lim = np.zeros((2, ni, nj))
        m = util.cluster_cosine(20000)
        for j in range(nj):
            xrt_u, xrt_l = blade.evaluate_section(span_frac[j], m=m)
            assert np.all(xrt_u[2] >= xrt_l[2])

            stream_frac_now = stream_frac_span[:, j]
            xr[..., j] = annulus.evaluate_xr(
                stream_frac_now + geom.ist + 1.0, span_frac[j]
            )
            theta_lim[..., j] = _theta_limits(
                stream_frac_now,
                xrt_u,
                xrt_l,
                (0, 1),
                Theta,
                geom.chord_mid[
                    (0, -1),
                ],
                Theta_max=self.skew_max,
                deswirl_dn=self.deswirl and (i_row == n_row - 1),
            )[:2]

        # 4. Cusp insertion and tip pinching
        xrt_ul = np.stack(
            np.broadcast_arrays(
                xr[0, ..., None],
                xr[1, ..., None],
                np.moveaxis(theta_lim, 0, -1),
            )
        )
        if self.AR_cusp:
            xrt_cusped = add_cusp(xrt_ul, ite, self.AR_cusp, self.ni_cusp, self.ni_TE)
            xr = xrt_cusped[:2, ...].mean(axis=-1)
            theta_lim = np.moveaxis(xrt_cusped[2, ...], -1, 0)

        assert np.isfinite(xr).all()
        assert np.isfinite(pitch_frac_clust).all()
        assert np.isfinite(theta_lim).all()
        assert np.isfinite(relax).all()
        assert np.isfinite(uniform).all()

        np.multiply(pitch_frac_clust, 1.0 - relax, out=pitch_frac_clust)
        np.add(pitch_frac_clust, relax * uniform, out=pitch_frac_clust)
        pitch_frac_relax = pitch_frac_clust
        assert np.isfinite(pitch_frac_relax).all()
        assert (pitch_frac_relax >= 0.0).all() and (pitch_frac_relax <= 1.0).all()

        if blade.tip_gap:
            theta_mid = np.mean(theta_lim, axis=0, keepdims=True)
            spf_pinch = [
                1.0 - tip_ref * 2.0,
                1.0 - tip_ref * self.gap_contraction,
                1.0,
            ]
            pinch_pinch = [0.0, 1.0, 1.0]
            pinch_frac = np.interp(span_frac, spf_pinch, pinch_pinch).reshape(1, 1, -1)
            theta_lim = pinch_frac * theta_mid + (1.0 - pinch_frac) * theta_lim
            njtip = int(np.sum(pinch_frac == 1.0))
        else:
            njtip = 0

        return _RowCoords(
            xr=xr,
            theta_lim=theta_lim,
            pitch_frac_relax=pitch_frac_relax,
            njtip=njtip,
            ite=ite,
        )

    def _row_block(self, machine, i_row, geom, grids, coords, n_row):
        """Assemble xrt, build patches, and return a Block."""
        ni, nj, nk = grids.ni, grids.nj, grids.nk
        ile = grids.ile
        ite = coords.ite
        njtip = coords.njtip

        # 3-D coordinate array, a single allocation filled in-place
        xrt_now = np.empty((3, ni, nj, nk))

        # x and r: broadcast (2, ni, nj) -> (2, ni, nj, nk)
        xrt_now[:2] = coords.xr[..., np.newaxis]

        # theta: interpolate between upper/lower angular limits then flip
        # pitchwise. Compute unflipped into xrt_now[2], then copy reversed.
        pfr3 = np.expand_dims(coords.pitch_frac_relax, 0)
        theta_lim3 = np.expand_dims(coords.theta_lim, 3)
        assert (np.diff(theta_lim3, axis=0) <= 0.0).all()
        np.add(
            pfr3
            * theta_lim3[
                (0,),
            ],
            (1.0 - pfr3)
            * (
                theta_lim3[
                    (1,),
                ]
                + geom.pitch_theta
            ),
            out=xrt_now[2:3],
        )
        xrt_now[2] = xrt_now[2, :, :, ::-1]
        assert np.isfinite(geom.pitch_theta)
        assert np.isfinite(xrt_now[2]).all()

        # Periodic patches and cusp indices
        icusp = ite + self.ni_cusp - 1 if self.ni_cusp else ite
        assert ile % 8 == 0, f"upstream periodic span ile={ile} not a multiple of 8"
        patches = [
            ember.patch.PeriodicPatch(i=(0, ile), k=0),
            ember.patch.PeriodicPatch(i=(0, ile), k=-1),
            ember.patch.PeriodicPatch(i=(icusp, -1), k=0),
            ember.patch.PeriodicPatch(i=(icusp, -1), k=-1),
        ]
        if self.AR_cusp:
            logger.info(f"Adding cusps {ite, icusp}")
            patches.extend(
                [
                    ember.patch.CuspPatch(i=(ite, icusp), k=0, label="cusp_k0"),
                    ember.patch.CuspPatch(i=(ite, icusp), k=-1, label="cusp_nk"),
                ]
            )

        # Inlet / mixing / outlet
        if i_row == 0:
            patches.append(ember.patch.InletPatch(i=0))
        else:
            patches.append(ember.patch.MixingPatch(i=0))
        if i_row == (n_row - 1):
            patches.append(ember.patch.OutletPatch(i=-1))
        else:
            patches.append(ember.patch.MixingPatch(i=-1))

        # Tip gap
        if njtip:
            ilim_tip = (ile, icusp)
            jlim_tip = (-njtip, -1)
            logger.info(f"Adding tip patches i={ilim_tip}, j={jlim_tip}")
            patches.extend(
                [
                    ember.patch.PeriodicPatch(i=ilim_tip, j=jlim_tip, k=0),
                    ember.patch.PeriodicPatch(i=ilim_tip, j=jlim_tip, k=-1),
                ]
            )

        block = ember.block.Block(shape=(ni, nj, nk))
        block.set_label(f"row{i_row}")
        block.set_Nb(machine.blades[i_row].n_blade)
        block.set_x(xrt_now[0])
        block.set_r(xrt_now[1])
        block.set_t(xrt_now[2])
        block.patches.extend(patches)

        nic, njc, nkc = block.shape
        assert not (nic - 1) % 8, f"nic-1={nic - 1} not divisible by 8"
        assert not (njc - 1) % 8, f"njc-1={njc - 1} not divisible by 8"
        assert not (nkc - 1) % 8, f"nkc-1={nkc - 1} not divisible by 8"

        return block

    def _stitch_mixing_planes(self, grid):
        """Force xr coordinates to match exactly at mixing planes."""
        for i_row in range(0, len(grid) - 1):
            xr0 = grid[i_row].xrt[-1, :, 0, :2]
            xr1 = grid[i_row + 1].xrt[0, :, 0, :2]
            xrav = 0.5 * (xr0 + xr1)
            xav = xrav[..., 0, None]
            rav = xrav[..., 1, None]
            grid[i_row][-1].set_x(xav)
            grid[i_row + 1][0].set_x(xav)
            grid[i_row][-1].set_r(rav)
            grid[i_row + 1][0].set_r(rav)

    #
    # ONE-DIMENSIONAL GRID VECTORS
    #

    def spanwise_grid(self, dspf_hub, dspf_casing, tip):
        """Evaluate a spanwise grid vector given hub and casing spacings."""
        if tip:
            Lmain = 1.0 - tip

            # We want at least 9 nodes across the tip gap
            # So the minimum grid spacing should be the smallest of:
            #   - 9 pts uniform
            #   - target shroud spacing
            njtip_min = 9
            dspf_tip = np.minimum(dspf_casing, tip / njtip_min)

            spf_main = clusterfunc.double.free(
                dspf_hub, dspf_tip, self.dspf_mid, self.ER_span, 0.0, Lmain
            )

            try:
                spf_tip = clusterfunc.double.free(
                    dspf_tip, dspf_tip, 4.0 * dspf_tip, self.ER_span, Lmain, 1.0
                )
            except clusterfunc.exceptions.ClusteringException:
                spf_tip = clusterfunc.double.fixed(
                    dspf_tip, dspf_tip, njtip_min, Lmain, 1.0
                )

            spf_main = util.resample(spf_main, self.resolution_factor, mult=8)
            spf_tip = util.resample(spf_tip, self.resolution_factor, mult=8)
            spf = np.concatenate((spf_main[:-1], spf_tip))

            assert spf[0] == 0.0
            assert np.isclose(spf[-1], 1.0)
            assert (np.diff(spf) > 0.0).all()
            assert (spf >= tip).sum() >= njtip_min

            return spf

        else:
            return util.resample(
                clusterfunc.double.free(
                    dspf_hub, dspf_casing, self.dspf_mid, self.ER_span
                ),
                self.resolution_factor,
                mult=8,
            )

    def pitchwise_grid(self, drt_row, pitch_chord, AR_row, resample=True):
        """Evaluate a pitchwise grid vector given surface spacing."""
        dm_mid = self.dspf_mid * AR_row / self.AR_merid
        drt_mid = dm_mid / pitch_chord * self.AR_passage
        logger.debug(
            f"Free npts: drt_row={drt_row}, drt_mid={drt_mid}, ER={self.ER_pitch}"
        )

        x = clusterfunc.symmetric.free(drt_row, drt_mid, self.ER_pitch)

        dx = np.diff(x)
        assert np.isclose(x[0], 0.0)
        assert np.isclose(x[-1], 1.0)
        assert np.all(dx > 0.0)
        ERout = dx[1:] / dx[:-1]
        ERout[ERout < 1.0] = 1.0 / ERout[ERout < 1.0]
        assert np.all(ERout <= self.ER_pitch)
        assert np.isfinite(x).all()

        if resample:
            x = util.resample(x, self.resolution_factor, mult=8)
            if len(x) < self.nk_min:
                npts = int(8 * np.ceil((self.nk_min - 1) / 8)) + 1
                x = clusterfunc.symmetric.fixed(drt_row, npts)

        return x

    def pitchwise_grid_fixed_npts(self, drt_row, pitch_chord, AR_row, npts):
        """Evaluate a pitchwise grid vector with a prescribed point count."""
        x = clusterfunc.symmetric.fixed(drt_row, npts)

        dx = np.diff(x)
        assert np.isclose(x[0], 0.0)
        assert np.isclose(x[-1], 1.0)
        assert np.all(dx > 0.0)
        assert len(x) == npts
        assert np.isfinite(x).all()

        return util.resample(x, self.resolution_factor)

    def streamwise_grid(
        self,
        pitch_chord,
        nrt,
        L,
        AR_row,
        tte=None,
        chord_factor=1.0,
        ni_chord=None,
        ni_cusp=0,
    ):
        """Evaluate streamwise grid vector for a blade row."""

        assert len(pitch_chord) == 3
        assert (pitch_chord > 0.0).all()
        assert nrt > 1

        # Normalised grid spacings at endpoints (normalised by their gap chord)
        dm_boundary = self.AR_stream * pitch_chord / nrt

        dm_mid = self.dspf_mid * AR_row / self.AR_merid

        # Convert the LE/TE grid spacings from chord-normalised to gap-normalised
        dm_upstream_LE = self.dm_LE * pitch_chord[0] / pitch_chord[1]
        dm_TE = (1.0 - tte) / self.ni_TE
        dm_downstream_TE = dm_TE * pitch_chord[-1] / pitch_chord[1]

        t_upstream = 1.0 - np.flip(
            clusterfunc.single.free(
                dm_upstream_LE, dm_boundary[0] * L[0], self.ER_stream, 0.0, L[0]
            )
        )

        # Apply chord length adjustment factor
        dm_LE_adj = self.dm_LE / chord_factor
        dm_mid_adj = dm_mid / chord_factor
        dm_TE_adj = dm_TE / chord_factor

        t_chord = clusterfunc.double.free(
            dm_LE_adj, dm_TE_adj, dm_mid_adj, self.ER_stream, 0.0, tte
        )

        t_te = np.linspace(tte, 1.0, self.ni_TE)

        try:
            t_downstream = clusterfunc.single.free(
                dm_downstream_TE, dm_boundary[-1] * L[1], self.ER_stream, 0.0, L[1]
            )
        except clusterfunc.exceptions.ClusteringException:
            t_downstream = clusterfunc.single.free(
                dm_downstream_TE,
                dm_boundary[-1] * L[1],
                self.ER_stream,
                0.0,
                L[1],
                mult=1,
            )

        t_upstream = util.resample(t_upstream, self.resolution_factor, mult=8)
        t_downstream = util.resample(t_downstream, self.resolution_factor, mult=8)
        t_te = util.resample(t_te, self.resolution_factor)
        t_chord = util.resample(t_chord, self.resolution_factor, mult=8)

        # t_te has ni_TE points which may not be 8m+1. Adjust t_downstream
        # to satisfy the cusp alignment requirement (len(t_downstream) -
        # ni_cusp) % 8 == 0, then adjust t_chord to restore total ni-1
        # divisibility by 8.
        if ni_cusp:
            # Less 8 further points to coarsen grids downstream of cusp a bit
            dn_deficit = (ni_cusp - len(t_downstream)) % 8 - 8
            if dn_deficit:
                t_downstream = np.interp(
                    np.linspace(0.0, 1.0, len(t_downstream) + dn_deficit),
                    np.linspace(0.0, 1.0, len(t_downstream)),
                    t_downstream,
                )
        ni_total = len(t_upstream) + len(t_chord) + len(t_te) + len(t_downstream) - 3
        chord_deficit = (8 - (ni_total - 1) % 8) % 8
        if chord_deficit:
            t_chord = np.interp(
                np.linspace(0.0, 1.0, len(t_chord) + chord_deficit),
                np.linspace(0.0, 1.0, len(t_chord)),
                t_chord,
            )

        logger.debug(f"ni_TE={self.ni_TE}, tte={tte}")
        logger.debug(f"t_te ({len(t_te)} pts): {t_te}")

        t = np.concatenate(
            [t_upstream - 1.0, t_chord[1:], t_te[1:], t_downstream[1:] + 1.0]
        )

        dt = np.diff(t)
        assert (dt > 0.0).all()

        ile = len(t_upstream) - 1
        ite = ile + len(t_chord) + len(t_te) - 2

        return t, ile, ite

    def pitchwise_relaxation(self, stream_frac, pitch_chord):
        """Relax clustering towards a uniform distribution at inlet and exit."""
        dt_relax = (
            np.ones((2,))
            * self.nchord_relax
            * pitch_chord[
                (0, -1),
            ]
            / pitch_chord[1]
        )
        relax_ref = np.array([1.0, 0.0, 0.0, 1.0])
        t_ref = np.array([-dt_relax[0], 0.0, 1.0, 1.0 + dt_relax[1]])
        return np.interp(stream_frac, t_ref, relax_ref)


def _te_corner_curvature(tq, theta, m_lo):
    """Locate the TE corner as the peak curvature of theta(m), sub-sample.

    The corner is the sharpest turning point of the surface theta distribution
    near the TE. On the uniform tq grid the discrete second difference is
    proportional to curvature; the peak is refined to sub-sample resolution by
    a parabolic fit through the three points about the discrete maximum. This
    is a single continuous criterion (no zero-crossing set membership, no grid
    quantisation), so the returned location varies smoothly with span unless the
    underlying geometry genuinely has a spanwise step.

    Returns (tte, strength) or (None, 0.0) if no interior peak is found in the
    window tq > m_lo. strength is the peak |curvature|, for picking a surface.
    """
    # Second difference ~ curvature (uniform tq spacing)
    d2 = theta[:-2] - 2.0 * theta[1:-1] + theta[2:]
    m_c = tq[1:-1]  # centres of the second-difference stencil
    win = m_c > m_lo
    if not win.any():
        return None, 0.0
    idx_win = np.flatnonzero(win)
    a2 = np.abs(d2)
    ipk = idx_win[np.argmax(a2[idx_win])]
    # Need interior neighbours for the parabolic refinement
    if ipk == 0 or ipk == len(d2) - 1:
        return tq[ipk + 1], a2[ipk]
    # Parabolic vertex offset in [-0.5, 0.5] from the three |curvature| samples
    yl, y0, yr = a2[ipk - 1], a2[ipk], a2[ipk + 1]
    denom = yl - 2.0 * y0 + yr
    delta = 0.5 * (yl - yr) / denom if denom != 0.0 else 0.0
    delta = float(np.clip(delta, -0.5, 0.5))
    dtq = tq[1] - tq[0]
    tte = tq[ipk + 1] + delta * dtq
    return tte, a2[ipk]


def _theta_limits(
    tq,
    xrt_u,
    xrt_l,
    mlim,
    Theta=(0.0, 0.0),
    c=(1.0, 1.0),
    Theta_max=30.0,
    deswirl_dn=False,
):
    """Evaluate pitchwise limits given upper/lower surface section coordinates."""

    # Put geometric leading edge where it should be
    # Must handle axial and radial inlets differently

    # If x varies more than r near LE, is axial, split on min x
    if np.ptp(xrt_u[0][:10]) > np.ptp(xrt_u[1][:10]):
        ind_split = 0
        iule = np.argmin(xrt_u[ind_split])
        ille = np.argmin(xrt_l[ind_split])
    # Otherwise, is radial, split on max r
    elif xrt_u[1][0] < xrt_u[1][-1]:
        ind_split = 1
        iule = np.argmin(xrt_u[ind_split])
        ille = np.argmin(xrt_l[ind_split])
    else:
        ind_split = 1
        iule = np.argmax(xrt_u[ind_split])
        ille = np.argmax(xrt_l[ind_split])

    # If the geometric leading edge is on upper surface
    # we need to move some points from upper to lower
    if xrt_u[ind_split][iule] < xrt_l[ind_split][ille]:
        xrt_l = np.concatenate(
            (np.flip(xrt_u[:, 1 : iule + 1], axis=-1), xrt_l), axis=-1
        )
        xrt_u = xrt_u[:, iule:]
    # If the geometric leading edge is on lower surface
    # we need to move some points from lower to upper
    elif xrt_u[ind_split][iule] > xrt_l[ind_split][ille]:
        xrt_u = np.concatenate(
            (np.flip(xrt_l[:, 1 : ille + 1], axis=-1), xrt_u), axis=-1
        )
        xrt_l = xrt_l[:, ille:]

    # Join the curves together at trailing edge
    if xrt_u[ind_split].max() > xrt_l[ind_split].max():
        xrt_l = np.concatenate((xrt_l, xrt_u[:, (-1,)]), axis=-1)
    else:
        xrt_u = np.concatenate((xrt_u, xrt_l[:, (-1,)]), axis=-1)

    # Evaluate normalised meridional distances for each surface
    m_u = util.cum_arc_length(xrt_u[:2])
    m_l = util.cum_arc_length(xrt_l[:2])
    m_u /= m_u[-1]
    m_l /= m_l[-1]

    m_u = mlim[0] + np.ptp(mlim) * m_u
    m_l = mlim[0] + np.ptp(mlim) * m_l

    # Interpolate the pitchwise limits
    # Values outside unit interval constant at boundary values
    theta_u = np.interp(tq, m_u, xrt_u[2])
    theta_l = np.interp(tq, m_l, xrt_l[2])

    # TE corner = sharpest turning point (peak curvature) of theta(m) in the
    # last ~20% chord, found on each surface and refined to sub-sample. Pick the
    # surface with the stronger corner. Single continuous criterion, so tte
    # varies smoothly with span unless the geometry has a real spanwise step.
    m_lo = mlim[1] - 0.2
    tte_l, str_l = _te_corner_curvature(tq, theta_l, m_lo)
    tte_u, str_u = _te_corner_curvature(tq, theta_u, m_lo)
    if tte_l is None and tte_u is None:
        # Fallback: cluster grid over last 1.0% chord
        tte = mlim[1] - 0.01
    elif tte_u is None or (tte_l is not None and str_l >= str_u):
        tte = tte_l
    else:
        tte = tte_u

    if np.any(theta_u < theta_l):
        raise ValueError("Blade is thicker than the calculated pitch.")

    r_u = np.interp(tq, m_u, xrt_u[1])
    r_l = np.interp(tq, m_l, xrt_l[1])
    rref = 0.5 * (r_u + r_l)

    # Skew the mesh upstream of LE and downstream of TE
    dtheta_skew = np.zeros_like(theta_u)
    ind_up = tq < mlim[0]
    ind_dn = tq >= mlim[1]
    if Theta_max >= 0.0:
        Theta_now = np.clip(Theta, -Theta_max, Theta_max)
    else:
        thresh = -Theta_max
        Theta_now = np.where(
            np.abs(Theta) <= thresh,
            Theta,
            np.asarray(Theta) - np.sign(Theta) * 90.0,
        )
    tanTheta = np.tan(np.radians(Theta_now))
    if ind_up.any():
        tq_up = tq[ind_up]
        blend = np.ones_like(tq_up)
        dtheta_skew[ind_up] = (
            tanTheta[0] * c[0] * util.cumtrapz0(blend / rref[ind_up], tq_up)
        )
        dtheta_skew[ind_up] -= dtheta_skew[ind_up][-1]
    if ind_dn.any():
        tq_dn = tq[ind_dn]
        if deswirl_dn:
            # Blend skew->axial over the outer 50% of the downstream gap.
            # Fully skewed (blend=1) from TE up to halfway across the gap,
            # then linearly ramp to 0 (axial) at the domain exit.
            m_exit = tq_dn[-1]
            gap = m_exit - mlim[1]
            dist_norm = (tq_dn - mlim[1]) / (0.5 * gap)
            blend = np.clip(1.0 - (dist_norm - 1.0), 0.0, 1.0)
        else:
            blend = np.ones_like(tq_dn)
        dtheta_skew[ind_dn] = (
            tanTheta[1] * c[1] * util.cumtrapz0(blend / rref[ind_dn], tq_dn)
        )
    theta_u += dtheta_skew
    theta_l += dtheta_skew

    return theta_u, theta_l, tte


def _blend_te_corner(xrrt, iTE, ni_TE, k):
    """Blend the corner-side surface towards a straight line approaching the cusp.

    On the cusp corner side, the window [iTE-2*ni_TE, iTE] holds the original
    blade points (upstream half) and the new loaded points (downstream half).
    Draw one straight line between the two window endpoints and blend the
    interior points towards it, with weight ramping linearly from 0 at the
    upstream end (iTE-2*ni_TE) to 1 at the cusp start (iTE). Endpoints are
    pinned; this removes the kink between the blade and the loaded cusp points.
    """
    i0 = iTE - 2 * ni_TE
    if i0 < 0:
        return
    n = iTE - i0  # number of intervals across the window
    # Quadratic weight u**2: 0 -> 1 with zero slope at the upstream blade edge,
    # so the blend leaves the original surface curvature-continuously.
    w = ((np.arange(n + 1) / n) ** 2).reshape(-1, 1)  # 0 -> 1, indexed [i, j]
    # Blend only theta (rt); leave x and r untouched so points keep their
    # streamwise/spanwise positions and only the tangential coordinate relaxes
    # towards the straight line. The target is theta linear in physical x
    # between the two window endpoints.
    x = xrrt[0, i0 : iTE + 1, :, k]
    t = xrrt[2, i0 : iTE + 1, :, k]
    x0, x1 = x[0], x[-1]
    t0, t1 = t[0], t[-1]
    frac = (x - x0[None, :]) / (x1 - x0)[None, :]
    line = t0[None, :] + frac * (t1 - t0)[None, :]
    xrrt[2, i0 : iTE + 1, :, k] = (1.0 - w) * t + w * line


def add_cusp(xrt, iTE, AR_cusp, ni_cusp, ni_TE):
    """Change block coordinates from square TE to cusped TE.

    This assumes that the trailing edge is located exactly
    at ni_TE points upstream of the zero-thickness point of the blade
    (achieve this by setting dm_TE to 0.0 and ni_TE > 0 in config.)

    """

    assert AR_cusp > 0.0

    nj = xrt.shape[2]
    jmid = nj // 2

    # Convert to xrrt
    rref = xrt[1, iTE, jmid, 0]
    xrrt = util.to_xrrt_ref(xrt, rref)

    assert np.allclose(xrt[2, iTE, :, 0], xrt[2, iTE, :, 1])

    # Determine if the exit angle is +ve or -ve
    plus_exit = np.diff(xrrt[2, iTE : iTE + 2, jmid, 0]).item() > 0.0
    logger.debug(f"{plus_exit=}")

    is_axial = bool(
        np.ptp(xrrt[0, iTE : iTE + 2, jmid, 0])
        > np.ptp(xrrt[1, iTE : iTE + 2, jmid, 0])
    )
    logger.debug(f"{is_axial=}")

    # Find the trailing-edge corner and extrapolate the corner-side surface
    # straight to the TE. The corner index is detected independently at each
    # spanwise station from that station's own theta profile, rather than once
    # at midspan and reused span-wide: the corner shifts in the streamwise
    # index with span, so a single midspan choice ties the re-squaring to the
    # spanwise node count -- at some resolutions the reused index overshoots
    # and collapses the hub TE to zero thickness (breaking the cusp below).
    istlook = iTE - 12
    if plus_exit:
        # Turning point on the lower surface (side 0), per station.
        ilower = istlook + np.argmax(xrrt[2, istlook:iTE, :, 0], axis=0)
        # Extrapolate lower surface to TE
        if is_axial:
            for j in range(nj):
                il = int(ilower[j])
                grad = (xrrt[2, il, j, 0] - xrrt[2, il - 1, j, 0]) / (
                    xrrt[0, il, j, 0] - xrrt[0, il - 1, j, 0]
                )
                xrrt[2, il + 1 : iTE + 1, j, 0] = xrrt[2, il, j, 0] + grad * (
                    xrrt[0, il + 1 : iTE + 1, j, 0] - xrrt[0, il, j, 0]
                )
            _blend_te_corner(xrrt, iTE, ni_TE, 0)

        else:
            raise NotImplementedError()
        logger.debug(f"ilower={ilower}, iTE={iTE}")
        xrrt_TE = np.moveaxis(xrrt[:, iTE - 10 : iTE + 1, :, :], -1, 0)
    else:
        # Turning point on the upper surface (side -1), per station.
        ilower = istlook + np.argmin(xrrt[2, istlook:iTE, :, -1], axis=0)
        # Extrapolate upper surface to TE
        if is_axial:
            for j in range(nj):
                il = int(ilower[j])
                grad = (xrrt[2, il, j, -1] - xrrt[2, il - 1, j, -1]) / (
                    xrrt[0, il, j, -1] - xrrt[0, il - 1, j, -1]
                )
                xrrt[2, il : iTE + 1, j, -1] = xrrt[2, il, j, -1] + grad * (
                    xrrt[0, il : iTE + 1, j, -1] - xrrt[0, il, j, -1]
                )
            _blend_te_corner(xrrt, iTE, ni_TE, -1)

        else:
            raise NotImplementedError()
        xrrt_TE = np.stack(
            (
                xrrt[:, iTE - 10 : iTE + 1, :, 0],
                xrrt[:, iTE - 10 : iTE + 1, :, -1],
            )
        )

    # Centre of trailing edge
    xrrt_cent = np.mean(xrrt_TE[:, :, -1], axis=0)

    # Vectors across TE and along each surface
    # xrrt_TE is indexed[side, coord, i, j]
    vec_TE = xrrt_TE[0, :, -1] - xrrt_TE[1, :, -1]
    W_TE = util.vecnorm(vec_TE)
    vec_TE /= W_TE

    # Slope of each surface over its last TE segment, indexed [side, coord, j].
    # vecnorm contracts the leading axis, so norm each side's [coord, j] vector.
    vec_side = np.diff(xrrt_TE[:, :, -2:, :], axis=2).squeeze(axis=2)
    vec_side = np.stack([vs / util.vecnorm(vs) for vs in vec_side])

    # Mean camber direction (used below to pick the in-plane interpolation axis)
    vec_cam = vec_side.mean(axis=0)
    vec_cam /= util.vecnorm(vec_cam)

    # The cusp tip is the intersection of the two surface tangent lines, so that
    # each side continues its own TE slope (a true wedge with no kink). Solve in
    # the two in-plane coordinates: (x, rt) if axial, else (r, rt).
    ip = (0, 2) if is_axial else (1, 2)
    L_cusp = AR_cusp * W_TE
    logger.debug(f"L_cusp={L_cusp.mean():.3g}")

    # Per-span intersection of P0 + s*v0 == P1 + t*v1 in the in-plane coords.
    P0 = xrrt_TE[0, :, -1]  # [coord, j]
    P1 = xrrt_TE[1, :, -1]
    v0 = vec_side[0]
    v1 = vec_side[1]
    a0, a1 = ip
    # det of [v0  -v1]
    det = v0[a0] * (-v1[a1]) - v0[a1] * (-v1[a0])
    rhs0 = P1[a0] - P0[a0]
    rhs1 = P1[a1] - P0[a1]
    s = (rhs0 * (-v1[a1]) - rhs1 * (-v1[a0])) / det

    xrrt_point = P0 + s[None, :] * v0
    # Fall back to the camber-line placement where the tangents are near-parallel
    # (det -> 0) or the intersection falls upstream of the TE (s <= 0).
    bad = ~np.isfinite(s) | (np.abs(det) < 1e-12) | (s <= 0.0)
    if bad.any():
        cam_point = xrrt_cent + L_cusp * vec_cam
        xrrt_point[:, bad] = cam_point[:, bad]

    # Now get the coordinates to be added. Distribute the ni_cusp points along
    # the cusp so the first cell matches the blade's last cell at the TE, then
    # expand smoothly towards the tip, avoiding a spacing jump at the join.
    xrrt_cusp = np.zeros((2, 3, ni_cusp, nj))
    for j in range(nj):
        # Blade's last cell into the TE on this side (arc length)
        dm_blade = util.cum_arc_length(xrrt_TE[0, :, -2:, j])[-1]
        # Cusp length (straight tangent line from TE point to tip) on this side
        for side in range(2):
            L_side = util.vecnorm(xrrt_point[:, j] - xrrt_TE[side, :, -1, j])
            try:
                f = clusterfunc.single.fixed(
                    dm_blade, L_side, 1.4, ni_cusp, 0.0, L_side
                )
            except clusterfunc.exceptions.ClusteringException:
                f = np.linspace(0.0, L_side, ni_cusp)
            f = (f / L_side).reshape(-1, 1)
            xrrt_cusp[side, :, :, j] = (
                f.T * xrrt_point[:, j, None]
                + (1.0 - f.T) * xrrt_TE[side, :, -1, j, None]
            )

    # Now make the grid spacing at TE match
    for j in range(nj):
        m_TE = util.cum_arc_length(xrrt_TE[0, :, :, j])
        dm_TE = np.diff(m_TE, axis=0)
        dm_end = dm_TE[-2].mean()
        dm_start = dm_TE[0].mean()
        m_TE_new = clusterfunc.double.fixed(
            dm_start,
            dm_end,
            len(m_TE),
            0.0,
            m_TE[-1],
        )
        for k in range(2):
            for c in range(3):
                xrrt[c, iTE - len(m_TE_new) + 1 : iTE + 1, j, k] = np.interp(
                    m_TE_new, m_TE, xrrt_TE[k, c, :, j]
                )

    xrrt_new = np.concatenate(
        (xrrt[:, : iTE + 1, :, :], np.moveaxis(xrrt_cusp, 0, -1)[:, 1:]), axis=1
    )

    # Shift the downstream coordinates to new theta TE
    dtheta_TE = xrrt_new[2, -1, :, 0] - xrrt[2, iTE, :, 0]
    xrrt_extra = xrrt[:, iTE:, :, :].copy()
    xrrt_extra[2] += dtheta_TE.reshape(1, -1, 1)

    # Define a curvilinear meridional coordinate along xrrt_extra
    # m_extra[i, j]
    m_extra = util.cum_arc_length(xrrt_extra[:2, :, :, 0])

    # Find the value of m that corresponds to cusp point
    nidown = xrrt_extra.shape[1] - ni_cusp + 1
    xrrt_new_down = np.zeros((3, nidown, nj, 2))
    if np.abs(vec_cam[0]).mean() > np.abs(vec_cam[1]).mean():
        # Mostly axial, interpolate using x
        for j in range(nj):
            m_point = np.interp(xrrt_point[0, j], xrrt_extra[0, :, j, 0], m_extra[:, j])
            # Theta offset required to exactly align TE

            dt_point = xrrt_point[2, j] - np.interp(
                m_point, m_extra[:, j], xrrt_extra[2, :, j, 0]
            )
            dt = np.interp(m_extra[:, j], [m_point, m_extra[-1, j]], [dt_point, 0.0])

            xrrt_extra[2, :, j, :] += dt.reshape(-1, 1)

            L = m_extra[-1, j] - m_point

            # Cusp's last cell in the meridional (x, r) metric, matching the
            # m_extra axis the downstream block is clustered on, so the first
            # downstream cell equals the cusp's last cell in the same metric.
            dm_start = util.cum_arc_length(xrrt_new[:2, -2:, j, 0])[-1]
            dm_end = m_extra[-1, j] - m_extra[-2, j]
            clu = clusterfunc.double.fixed(
                dm_start, dm_end, nidown, m_point, m_point + L
            )

            for ii in range(3):
                xrrt_new_down[ii, :, j, :] = np.interp(
                    clu, m_extra[:, j], xrrt_extra[ii, :, j, 0]
                )[:, None]

    else:
        raise NotImplementedError()

    xrrt_new = np.concatenate((xrrt_new[:, :-1, :, :], xrrt_new_down), axis=1)

    # Convert back to xrt
    return util.from_xrrt_ref(xrrt_new, rref)
