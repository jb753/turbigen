"""H-topology meshing.

One block per blade row, each an H mesh: the streamwise index runs from the
upstream boundary to the downstream one, the pitchwise index between the two
sides of the passage, and the spanwise index from hub to casing.

The mesh mathematics is carried over from the package this replaces. What is
new is the boundary of the class: it reads geometry from a
:class:`~turbigen.machine.Machine` and wall spacings from a
:class:`~turbigen.mesh.WallSpacing` handed to it by
:meth:`~turbigen.mesh.Mesher.mesh`, rather than from an adapter object built
by the config, and it no longer plots, logs memory use, or carries the three
options that no configuration ever set.
"""

import dataclasses
import logging
from typing import ClassVar

import ember.block
import ember.grid
import ember.patch
import numpy as np

from turbigen import clusterfunc, util
from turbigen.mesh import Mesher

logger = logging.getLogger("turbigen")

TE_MIN = 1e-4
"""Thinnest trailing edge a cusp can be built on, as a fraction of chord [--].

Two orders of magnitude under the one per cent of chord a blunt trailing edge
is drawn at, so that this catches a section that closes to a point rather than
one a designer deliberately drew thin.
"""


def _by_theta(blade, spf):
    """Return a blade section's two surfaces, the higher-angle one first.

    `evaluate_section` orders its pair by which surface the flow will call
    suction, which is what everything measuring a blade wants and what a
    thickness distribution is written in. A mesh wants neither: the pitchwise
    direction is an angle, the upper and lower halves of an H-block are the two
    sides of a pitch, and which of them the flow accelerates over has nothing
    to do with where the nodes go. So this is the one place the angular
    ordering is asked for, and it is read off `theta` rather than promised by
    the blade.
    """
    upper, lower = (surf[:3] for surf in blade.evaluate_section(spf))
    if upper[2].mean() < lower[2].mean():
        upper, lower = lower, upper
    return upper, lower


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
    dm_LE: float
    """Streamwise grid spacing at LE, normalised by meridional chord."""


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
    jtip: int
    """Spanwise index of the node at the top of the blade, `nj - 1` without a
    gridded tip gap."""
    L: tuple
    tte: float
    ni_cusp: int
    """Streamwise points along the cusp, trailing edge and tip included, or 0."""
    tte_span: np.ndarray
    """Where the trailing edge closure starts at each span, as a fraction of
    the chord between the edges."""
    m_edge: np.ndarray
    """Row `m` of the leading and trailing edge at each span, (nj, 2)."""
    cusps: list
    """The :class:`_CuspSection` at each span, or None without a cusp."""


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

    meshes_sweep: ClassVar[bool] = True

    ER_stream: float = 1.2
    """Expansion ratio of streamwise grid from first LE to inlet boundary."""

    AR_stream: float = 2.0
    """Aspect ratio in blade-to-blade plane of cells at outlet boundary."""

    AR_mix: float = 2.0
    """Aspect ratio in blade-to-blade plane of cells at a mixing plane.

    The row-to-row interface, as against the true machine inlet and outlet
    that :attr:`AR_stream` governs. Same default, so a mesh is unchanged until
    this is lowered --- which tightens the streamwise cells feeding a mixing
    plane without refining the real boundaries. A coarse mixing-plane cell is
    where a restart from a neighbouring design tends to come apart: a small
    design change can drop a multigrid block out of the wake there, and the
    field resampled onto the coarser grid then blows up.
    """

    AR_passage: float = 1.0
    """Nominal aspect ratio in blade-to-blade plane of mid-passage cells."""

    AR_merid: float = 1.0
    """Aspect ratio of mid-chord cells in meridional plane."""

    ER_span: float = 1.2
    """Expansion ratio of spanwise grid away from hub and casing."""

    half_span: bool = False
    """Cluster the spanwise grid at the hub only, not at both ends.

    For a row that is a wall at one end of the span and a symmetry plane at the
    other --- half of a linear cascade, say --- there is nothing at the casing
    to resolve, and the default two-sided distribution spends points reaching a
    spacing it was never asked for. Setting this uses a single-sided
    distribution instead: geometric growth from :attr:`WallSpacing.hub` at
    spacing :attr:`ER_span`, capped at :attr:`dspf_mid`, and flat from there to
    the casing, where the casing spacing is ignored.

    Passing the mid-span spacing as the casing spacing gets close to the same
    grid, since a two-sided distribution whose two targets coincide has nothing
    left to cluster towards. It is not the same grid: the expansion ratio still
    comes out well under :attr:`ER_span` because the two-sided solve is
    matching a spacing at each end rather than growing freely from one. On a
    half-span cascade that was 97 spanwise points at an expansion ratio of 1.03
    where this gives 49 at the 1.2 that was asked for.

    Incompatible with a tip gap, which needs the casing clustering it asks for.
    """

    dm_LE: float = 0.0
    """Streamwise grid spacing at LE, normalised by meridional chord, 0. to set
    it from the wall spacing using :attr:`AR_LE`."""

    AR_LE: float = 1.5
    """Aspect ratio of the cell at the LE, streamwise over pitchwise, when
    :attr:`dm_LE` is 0.

    The pitchwise spacing is the surface wall spacing that `yplus` asks for,
    so the LE resolution follows each row's Reynolds number rather than being
    one fraction of chord for every row. The axial turbine example at
    `yplus` 30 comes out at 1.35 and 1.67 with a fixed `dm_LE` of 0.001.
    """

    frac_TE: float = 0.1
    """Streamwise cell at the trailing edge, as a fraction of the mid-chord cap.

    The cap is the largest streamwise cell allowed on the blade,
    ``dspf_mid * AR_row / AR_merid``: the midspan spanwise cell in chord
    units, stretched by :attr:`AR_merid`. As a fraction of it, the count of
    cells the blade grid takes to grow from the trailing edge to its cap is
    the same whatever the spanwise resolution or row aspect ratio. The
    trailing edge closure is filled with cells of this size, at least one,
    and the downstream gap and any cusp grow from it at :attr:`ER_stream`.
    """

    dspf_mid: float = 0.03
    """Spanwise grid spacing at midspan, as a fraction of span."""

    ER_pitch: float = 1.2
    """Expansion ratio away from aerofoil surfaces."""

    nk_min: int = 37
    """Minimum number of pitchwise grid points per row."""

    njtip_min: int = 5
    """Minimum number of spanwise grid points across the open tip gap.

    Five rather than nine, because nine is not always affordable: the count
    sets a spacing, and a wide passage stretches the pitchwise direction
    around it while the radial spacing stays at a tenth of a millimetre. On a
    stage of pitch-to-chord 1.14 that gave cells up to 37:1 across the
    clearance, and a march that failed in the tip flow at step 30 -- where the
    same design at five ran four design iterations without trouble.

    A minimum, not a target: where the wall spacing is already finer than the
    clearance divided by this, the clustering puts in more. The total spanwise
    count stays a multigrid one either way; see
    :meth:`spanwise_grid`, where the main passage makes up whatever the gap
    leaves.
    """

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

    cusp: bool = True
    """Extend a blunt trailing edge into a cusp, False to mesh it square.

    The tip is where the two surfaces, run straight on along their own
    trailing edge slopes, would meet, so its length is the section's. Its
    points are the downstream gap's own, which grows from the trailing edge
    cell at :attr:`ER_stream`, scaled together to put one on the tip at every
    span, so they grow as fast as the grid does. Each side is a cubic matching the blade's slope and curvature where it
    leaves it, and the tip is fixed as above; see :class:`_CuspSection`. It needs a blunt trailing edge whose surfaces
    converge: a blade closing to a point is refused by
    :meth:`_check_trailing_edges`, and surfaces that never meet, or whose
    cubics would cross before the tip, by :func:`_cusp_section`.
    """

    ni_down_extra: int = 0
    """Streamwise points added to each row's downstream gap, past any cusp.

    Must be a multiple of eight, so the multigrid alignment still holds. The
    points are spread by re-interpolating that part of the gap, so it keeps
    its clustering.
    """

    nk_tip: int = 0
    """Number of points across the blade thickness in a gridded tip block.

    Zero pinches the blade to nothing over the clearance, which is what this
    mesher has always done: the two sides of the section are brought together
    and the collapsed faces made periodic, so the flow crosses where the blade
    used to be. Non-zero gives the clearance a block of its own, filling the
    blade-shaped prism between the tip and the casing, and the blade keeps the
    thickness it was designed with all the way up.

    The prism closes to a line at the leading edge and at the trailing edge or
    cusp tip, so the block is degenerate at both ends of `i`. Those faces carry
    no flux, having no area, and are marked as such rather than left to look
    like walls; see :meth:`_tip_block`.

    A count, not a flag, because the thickness has to be resolved: nine is the
    least a multigrid level can halve twice, and `gap_contraction` and
    `pinch_ramp` mean nothing once this is set.
    """

    gap_contraction: float = 0.6
    """Fraction of the tip gap over which the blade is pinched to zero
    thickness.

    Ignored when :attr:`nk_tip` grids the clearance instead of pinching it.
    """

    pinch_ramp: float = 2.0
    """Span over which the blade thins into the tip gap, as a multiple of the
    gap.

    Ignored when :attr:`nk_tip` grids the clearance instead of pinching it.

    The blade is at full thickness `pinch_ramp` gap-heights below the casing
    and reaches zero at the top of the pinch, so this sets how abruptly it
    ends. Too steep and the blade stops in what is nearly a step: the flow over
    the last solid layer cannot follow it, and the tip can drive the solver
    non-physical there. What matters is the spanwise distance the collapse
    takes, not the thickness of the last solid layer, so a wider ramp is the
    remedy rather than a blunter tip.
    """

    def __post_init__(self):
        if not 0.0 < self.frac_TE <= 1.0:
            raise ValueError(
                f"frac_TE={self.frac_TE} is a fraction of the mid-chord cell, "
                f"so must be over zero and at most one."
            )
        if self.ni_down_extra % 8:
            raise ValueError(
                f"ni_down_extra={self.ni_down_extra} must be a "
                f"multiple of eight to keep the multigrid alignment."
            )
        if self.nk_tip and (self.nk_tip < 9 or (self.nk_tip - 1) % 8):
            raise ValueError(
                f"nk_tip={self.nk_tip} is not a count a multigrid level can "
                f"halve; give it nine or more, one above a multiple of eight."
            )
        if self.inviscid_endwalls and self.half_span:
            raise ValueError(
                "half_span clusters at a viscous hub, but inviscid_endwalls "
                "makes it a slip wall; set one or the other."
            )

    def forward(self, machine, spacing):
        """Generate a Grid for a designed machine."""
        n_row = len(machine.rows)
        for i_row, row in enumerate(machine.rows):
            if row.blade.round_TE:
                raise ValueError(
                    f"Row {i_row} has a round trailing edge, which the H mesher "
                    f"does not grid. Mesh it with type: ohp, or leave it blunt."
                )
        if self.cusp:
            self._check_trailing_edges(machine)
        self._check_inviscid_ducts(machine.annulus)
        dspf_hub, dspf_casing, tip_ref = self._normalise_spacings(machine, spacing)

        blocks = []
        passages = []
        for i_row in range(n_row):
            geom = self._row_geometry(machine, i_row, spacing)
            grids = self._row_1d_grids(
                machine, i_row, geom, dspf_hub, dspf_casing, tip_ref, n_row
            )
            coords = self._row_coords(machine, i_row, geom, grids, tip_ref, n_row)
            passage = self._row_block(machine, i_row, geom, grids, coords, n_row)
            blocks.append(passage)
            passages.append(passage)
            if self.nk_tip and machine.rows[i_row].tip_gap:
                blocks.append(self._tip_block(machine, i_row, geom, grids, coords))
            del coords

        # The last row's spanwise grid, which reaches the outlet
        self._relax_outlet(blocks, machine.annulus, grids.span_frac)
        grid = ember.grid.Grid(blocks)
        # The passage blocks alone, in row order: a mixing plane is between two
        # rows, and a tip block is inside one. With a gridded clearance the
        # block index is no longer the row index, and it is the row that this
        # stitches.
        self._stitch_mixing_planes(passages)
        grid.connectivity.periodic.pair()
        return grid

    def _check_trailing_edges(self, machine):
        """Refuse to cusp a blade whose trailing edge has no thickness.

        :func:`add_cusp` builds the cusp by extrapolating each surface's own
        slope to the tip where the two meet downstream. On a knife edge there
        is no gap between the two surfaces to extrapolate into: the sides
        cross, and what fails is an assertion on the pitchwise
        ordering of the block coordinates several steps later, naming nothing a
        designer wrote. Said here instead, before any of the mesh is built,
        against the trailing edge the sections actually draw rather than
        against a thickness parameter --- not every distribution has one, and
        :class:`~turbigen.thickness.ClarkThickness` defaults its to zero.
        """
        for i_row, row in enumerate(machine.rows):
            upper, lower = (surf[:, -1] for surf in row.blade.evaluate_section(0.5))
            # The angular coordinate turned into a length at the mean radius of
            # the two points, which is how `evaluate_section` offsets a surface
            # from the camber line in the first place.
            gap = np.array(
                [
                    upper[0] - lower[0],
                    upper[1] - lower[1],
                    0.5 * (upper[1] + lower[1]) * (upper[2] - lower[2]),
                ]
            )
            t_TE = float(util.vecnorm(gap)) / row.blade.evaluate_chord(0.5)
            if t_TE < TE_MIN:
                raise ValueError(
                    f"A trailing edge cusp is built on the width of the "
                    f"trailing edge, so cusp: true needs one to be there, "
                    f"but row {i_row} closes to t_TE/chord={t_TE:.2e} at "
                    f"midspan, under the {TE_MIN:g} a cusp can be built on. "
                    f"Give the blade a blunt trailing edge, or mesh it "
                    f"square with cusp: false."
                )

    def _normalise_spacings(self, machine, spacing):
        """Compute normalised hub/casing spacings and tip_ref."""
        annulus = machine.annulus
        span_all = np.array(
            [
                np.mean(annulus.evaluate_span(np.arange(i_row * 2, i_row * 2 + 2)))
                for i_row in range(len(machine.rows))
            ]
        )
        tip_all = np.array([row.tip_gap for row in machine.rows])
        tip_frac = tip_all / span_all
        tip_ref = np.max(tip_frac)

        # Every row is meshed on one spanwise vector, because a mixing plane
        # matches its two sides span-for-span and will not join rows of
        # different spanwise count. A pinch does not mind: it only asks for
        # clustering near the casing, and where the pinch ends is a meshing
        # choice. A gridded tip does mind, its blocks meeting at the node that
        # sits at the top of the blade, so one level has to serve every row
        # that has a gap.
        if self.nk_tip:
            gaps = tip_frac[tip_frac > 0.0]
            if gaps.size and not np.allclose(gaps, gaps[0], rtol=1e-6):
                raise ValueError(
                    f"A gridded tip puts a grid line at the top of the blade, "
                    f"and every row is meshed on one spanwise grid, so the "
                    f"rows that have a clearance must have the same one as a "
                    f"fraction of span. This machine has {tip_frac} -- pinch "
                    f"it with nk_tip = 0, or give the rows equal clearance."
                )

        span_ref = annulus.evaluate_span(1)
        return spacing.hub / span_ref, spacing.casing / span_ref, tip_ref

    def _row_geometry(self, machine, i_row, spacing):
        """Derive scalar geometry quantities for a single row."""
        annulus = machine.annulus
        pitch_theta = 2.0 * np.pi / float(machine.rows[i_row].n_blade)

        mrow = np.linspace(2.0 * i_row + 1.0, 2.0 * i_row + 2)
        xr_hub = annulus.evaluate_xr(mrow, 0.0)
        xr_cas = annulus.evaluate_xr(mrow, 1.0)
        xr_mid = annulus.evaluate_xr(mrow, 0.5)

        ist = 2 * i_row
        chord_hub = annulus.evaluate_chords(0.0)[ist : ist + 3]
        chord_mid = annulus.evaluate_chords(0.5)[ist : ist + 3]
        chord_cas = annulus.evaluate_chords(1.0)[ist : ist + 3]

        pitch_rtheta_hub = pitch_theta * xr_hub[1]
        pitch_rtheta_cas = pitch_theta * xr_cas[1]

        pitch_chord_hub = pitch_rtheta_hub / chord_hub[1]
        pitch_chord_cas = pitch_rtheta_cas / chord_cas[1]

        pitch_chord_ref = pitch_theta * xr_mid[1].mean() / chord_mid
        pitch_chord_max = np.maximum(pitch_chord_hub.max(), pitch_chord_cas.max())
        pitch_rtheta_max = np.maximum(pitch_rtheta_hub.max(), pitch_rtheta_cas.max())

        drt_norm = spacing.surface[i_row] / pitch_rtheta_max

        span_row = np.mean(annulus.evaluate_span(np.arange(i_row * 2, i_row * 2 + 2)))
        AR_row = span_row / chord_mid[1]

        if self.dm_LE:
            dm_LE = self.dm_LE
        else:
            dm_LE = self.AR_LE * spacing.surface[i_row] / chord_mid[1]
        logger.debug(f"Row {i_row} dm_LE={dm_LE:.4g}")

        return _RowGeometry(
            pitch_theta=pitch_theta,
            chord_mid=chord_mid,
            pitch_chord_ref=pitch_chord_ref,
            pitch_chord_max=pitch_chord_max,
            pitch_rtheta_max=pitch_rtheta_max,
            drt_norm=drt_norm,
            AR_row=AR_row,
            ist=ist,
            dm_LE=dm_LE,
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

        # Spanwise. A pinch is a shape imposed on the top of the span and can
        # end wherever `gap_contraction` puts it; a gridded tip is a block
        # boundary and has to sit at the top of the blade exactly.
        span_frac = self.spanwise_grid(
            dspf_hub,
            dspf_casing,
            tip_ref if self.nk_tip else tip_ref * self.gap_contraction,
        )
        nj = len(span_frac)
        assert not (nj - 1) % 8, f"nj-1={nj - 1} not divisible by 8"

        # Index of the node at the top of the blade, which is the one the two
        # blocks of a gridded row meet on. Found by value rather than counted
        # out of `spanwise_grid`, the two pieces being concatenated about it so
        # that it is exact rather than nearly so.
        if self.nk_tip and tip_ref:
            jtip = int(np.argmin(np.abs(span_frac - (1.0 - tip_ref))))
            assert np.isclose(span_frac[jtip], 1.0 - tip_ref), (
                f"no spanwise node at the top of the blade: "
                f"spf={span_frac[jtip]} against 1-tip={1.0 - tip_ref}"
            )
            assert not (nj - 1 - jtip) % 8, (
                f"tip gap spans {nj - jtip} nodes, which a multigrid level cannot halve"
            )
        else:
            jtip = nj - 1

        # Streamwise: choose inlet/exit lengths
        if n_row == 1:
            L = (1.0, 1.0)
        elif i_row == 0:
            L = (1.0, 0.5)
        elif i_row == (n_row - 1):
            L = (0.5, 1.0)
        else:
            L = (0.5, 0.5)

        # The trailing edge closure starts at the section's corner. The grid is
        # laid on the midspan one, and `_row_coords` moves it to each span's.
        blade = machine.rows[i_row].blade
        tq = np.linspace(0.8, 1.0, 500)
        xrt_u, xrt_l = _by_theta(blade, 0.5)
        _, _, tte = _theta_limits(tq, xrt_u, xrt_l, np.array((0, 1)))
        tte_span = np.empty(nj)
        for j in range(nj):
            xrt_u_j, xrt_l_j = _by_theta(blade, span_frac[j])
            _, _, tte_span[j] = _theta_limits(tq, xrt_u_j, xrt_l_j, np.array((0, 1)))

        # Where the blade's edges are at each span, in row `m`: on the
        # stations, unless the blade is swept. The corner was found as a
        # fraction of the chord between them.
        m_edge = np.tile([0.0, 1.0], (nj, 1))
        ends = (-L[0], 1.0 + L[1])
        if blade.swept:
            m_edge = np.array(
                [blade.evaluate_swept_m([0.0, 1.0], s) for s in span_frac]
            )
            if np.any(m_edge[:, 0] <= ends[0]) or np.any(m_edge[:, 1] >= ends[1]):
                raise ValueError(
                    f"Row {i_row} has edges swept to m={m_edge[:, 0].min():.4g} "
                    f"and {m_edge[:, 1].max():.4g} of the row, past the ends of "
                    f"its mesh at {ends[0]:.4g} and {ends[1]:.4g}."
                )
        m_corner = m_edge[:, 0] + tte_span * np.ptp(m_edge, axis=1)

        # The cusp of every section, its side with the corner the midspan's.
        # One node is the tip at every span, so the grid's cusp reaches the
        # geometric mean of the shortest and longest, which scales the grid
        # least at the worst span to put it on each one's tip.
        cusps, L_cusp = None, 0.0
        if self.cusp:
            m_edge_mid = np.array((0.0, 1.0))
            if blade.swept:
                m_edge_mid = np.asarray(blade.evaluate_swept_m([0.0, 1.0], 0.5))
            args = (blade, machine.annulus, geom.ist, L[1])
            mid = _cusp_section(
                *args, 0.5, m_edge_mid, m_edge_mid[0] + tte * np.ptp(m_edge_mid)
            )
            cusps = [
                _cusp_section(*args, span_frac[j], m_edge[j], m_corner[j], mid.k)
                for j in range(nj)
            ]
            L_span = [c.L_cusp for c in cusps]
            L_cusp = float(np.sqrt(min(L_span) * max(L_span)))
            logger.debug(f"Row {i_row} cusp length {L_cusp:.4g} of the gap chord")

        stream_frac, ile, ite, ni_cusp = self.streamwise_grid(
            geom.pitch_chord_ref,
            nk_not_resampled,
            L,
            geom.AR_row,
            tte,
            L_cusp=L_cusp,
            dm_LE=geom.dm_LE,
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
            jtip=jtip,
            L=L,
            tte=tte,
            ni_cusp=ni_cusp,
            tte_span=tte_span,
            m_edge=m_edge,
            cusps=cusps,
        )

    def _row_coords(self, machine, i_row, geom, grids, tip_ref, n_row):
        """Build 2-D xr, theta_lim and 3-D pitch_frac_relax for a row."""
        annulus = machine.annulus
        row = machine.rows[i_row]
        blade = row.blade
        ni, nj, nk = grids.ni, grids.nj, grids.nk
        span_frac = grids.span_frac
        stream_frac = grids.stream_frac
        ite = grids.ite

        spfr = span_frac.reshape(1, -1)
        tte = grids.tte  # midspan TE-corner fraction the streamwise grid was built on

        # Per-span TE corner alignment. The grid's TE breakpoint t = tte is fixed
        # at the midspan corner, but the true squared-TE corner fraction drifts
        # with span, so off-midspan the corner falls between streamwise nodes.
        # The chord mapping is warped so the grid knot t = tte lands on each
        # span's own corner, keeping one node on the corner at every span: that
        # is where the cusp leaves the blade on that side.
        tte_span = grids.tte_span
        m_edge = grids.m_edge
        ends = (-grids.L[0], 1.0 + grids.L[1])
        tte_row = m_edge[:, 0] + tte_span * (m_edge[:, 1] - m_edge[:, 0])

        # 1. Meridional xr on a (ni, nj) grid
        stream_frac_span = np.broadcast_to(stream_frac.reshape(-1, 1), (ni, nj)).copy()
        for j in range(nj):
            # Warp so grid breakpoints [-1, 0, tte, 1, 2] map to section
            # [-1, 0, tte_span[j], 1, 2]; with mlim = (0, 1) the only moving
            # interior knot is the corner at grid t = tte -> tte_span[j]. A
            # swept blade's edges move too, and the ends of the mesh, where
            # the mixing planes are, stay put.
            if blade.swept:
                stream_frac_span[:, j] = np.interp(
                    stream_frac_span[:, j],
                    [ends[0], 0, tte, 1, ends[1]],
                    [ends[0], *m_edge[j, :1], tte_row[j], *m_edge[j, 1:], ends[1]],
                )
                continue
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

        Theta = blade.evaluate_chi(0.5)
        pitch_frac_clust = np.tile(grids.pitch_frac_nom.reshape(1, 1, -1), (ni, nj, 1))

        # 3. Theta limits from blade sections
        theta_lim = np.zeros((2, ni, nj))
        for j in range(nj):
            xrt_u, xrt_l = _by_theta(blade, span_frac[j])
            assert np.all(xrt_u[2] >= xrt_l[2])

            stream_frac_now = stream_frac_span[:, j]
            xr[..., j] = annulus.evaluate_xr(
                stream_frac_now + geom.ist + 1.0, span_frac[j]
            )
            theta_lim[..., j] = _theta_limits(
                stream_frac_now,
                xrt_u,
                xrt_l,
                m_edge[j],
                Theta,
                geom.chord_mid[(0, -1),],
                Theta_max=self.skew_max,
                deswirl_dn=self.deswirl and (i_row == n_row - 1),
            )[:2]

        # 4. Cusp insertion
        if grids.ni_cusp:
            i_corner = int(np.argmin(np.abs(stream_frac - tte)))
            xr, theta_lim = add_cusp(
                xr,
                theta_lim,
                stream_frac_span + geom.ist + 1.0,
                lambda u: annulus.evaluate_xr(u, spfr),
                i_corner,
                ite,
                grids.ni_cusp,
                grids.cusps,
            )

        # 5. A tip gap the same height in metres all along the chord
        if row.tip_gap and tip_ref:
            xr, theta_lim = self._warp_casing(
                grids, row.tip_gap, tip_ref, xr, theta_lim
            )

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

        if row.tip_gap and self.nk_tip:
            # The blade keeps its thickness; the clearance gets a block of its
            # own. Nothing to do to the section here, the top of the span
            # already carrying it -- only the count of nodes the two blocks
            # share, which is what the patches are written against.
            njtip = grids.nj - grids.jtip
        elif row.tip_gap:
            theta_mid = np.mean(theta_lim, axis=0, keepdims=True)
            spf_pinch = [
                1.0 - tip_ref * self.pinch_ramp,
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

    def _warp_casing(self, grids, tip_gap, tip_ref, xr, theta_lim):
        """Move the spanwise nodes so the tip gap is uniform in metres.

        The spanwise grid is one vector of span fractions, which puts the top
        of the blade at ``1 - tip_ref`` of the local span everywhere: a gap
        that grows and shrinks with the annulus. Here each streamwise station
        has the distance of its nodes from the casing, as a fraction of span,
        multiplied by ``r = tip_gap / span / tip_ref`` near the casing, so the
        node at the top of the blade lands :attr:`~BladeDesign.tip` metres
        from it. Below the gap and the pinch the stretch eases off, to a
        constant that holds the hub where it was; see :func:`_casing_warp`.

        Over the blade ``r`` follows the local span; up- and downstream it
        eases back to one at the ends of the row, so a mixing plane or the
        outlet still sees the grid's own span fractions.

        The pinch, applied after this to node indices, then sits at fixed
        multiples of the gap in metres too. Blade angles and coordinates are
        interpolated in span fraction onto the moved nodes: the annulus is
        linear in span fraction, so for `xr` that is exact but for the slight
        difference in streamwise position between neighbouring nodes.
        """
        span_frac = grids.span_frac
        stream_frac = grids.stream_frac
        # Each station's span is the length of its own grid line, hub node to
        # casing node, which the trailing edge corner warp and the cusp have
        # moved off the nominal streamwise position
        span = np.hypot(*(xr[:, :, -1] - xr[:, :, 0]))
        r_blade = tip_gap / span / tip_ref

        # Weight of the local ratio: one over the blade and any cusp, easing
        # to zero at the ends of the row
        ends = (-grids.L[0], 1.0 + grids.L[1])
        m_LE = grids.m_edge[:, 0].min()
        m_TE = grids.m_edge[:, 1].max()
        if grids.ni_cusp:
            m_TE = max(m_TE, stream_frac[grids.ite + grids.ni_cusp - 1])
        x = np.where(
            stream_frac < m_LE,
            (stream_frac - ends[0]) / (m_LE - ends[0]),
            np.where(
                stream_frac > m_TE, (ends[1] - stream_frac) / (ends[1] - m_TE), 1.0
            ),
        )
        x = np.clip(x, 0.0, 1.0)
        w = x * x * (3.0 - 2.0 * x)
        r = 1.0 + w * (r_blade - 1.0)

        band = tip_ref * max(self.pinch_ramp, 1.0)
        d = 1.0 - span_frac
        xr_new = np.empty_like(xr)
        theta_new = np.empty_like(theta_lim)
        for i in range(len(stream_frac)):
            spf_i = 1.0 - _casing_warp(d, r[i], band)
            for c in range(2):
                xr_new[c, i] = np.interp(spf_i, span_frac, xr[c, i])
                theta_new[c, i] = np.interp(spf_i, span_frac, theta_lim[c, i])
        # The endwalls are where they were, to the last bit
        xr_new[:, :, (0, -1)] = xr[:, :, (0, -1)]
        logger.debug(
            f"Tip gap {tip_gap:.4g} m: casing band stretched by "
            f"{r.min():.3g} to {r.max():.3g}"
        )
        return xr_new, theta_new

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
            pfr3 * theta_lim3[(0,),],
            (1.0 - pfr3) * (theta_lim3[(1,),] + geom.pitch_theta),
            out=xrt_now[2:3],
        )
        xrt_now[2] = xrt_now[2, :, :, ::-1]
        assert np.isfinite(geom.pitch_theta)
        assert np.isfinite(xrt_now[2]).all()

        # Periodic patches and cusp indices
        icusp = ite + grids.ni_cusp - 1 if grids.ni_cusp else ite
        assert ile % 8 == 0, f"upstream periodic span ile={ile} not a multiple of 8"
        patches = [
            ember.patch.PeriodicPatch(i=(0, ile), k=0),
            ember.patch.PeriodicPatch(i=(0, ile), k=-1),
            ember.patch.PeriodicPatch(i=(icusp, -1), k=0),
            ember.patch.PeriodicPatch(i=(icusp, -1), k=-1),
        ]
        if grids.ni_cusp:
            logger.debug(f"Adding cusps {ite, icusp}")
            # A cusp is the wake of a blade that has ended, so it stops where
            # the blade does. Over a gridded clearance the same two faces are
            # the sides of the tip block and the flow crosses them, which the
            # tip patch below says; a cusp there would say the opposite.
            jlim_cusp = (0, grids.jtip) if njtip and self.nk_tip else (0, -1)
            patches.extend(
                [
                    ember.patch.CuspPatch(
                        i=(ite, icusp), j=jlim_cusp, k=0, label="cusp_k0"
                    ),
                    ember.patch.CuspPatch(
                        i=(ite, icusp), j=jlim_cusp, k=-1, label="cusp_nk"
                    ),
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
            logger.debug(f"Adding tip patches i={ilim_tip}, j={jlim_tip}")
            patches.extend(
                [
                    ember.patch.PeriodicPatch(i=ilim_tip, j=jlim_tip, k=0),
                    ember.patch.PeriodicPatch(i=ilim_tip, j=jlim_tip, k=-1),
                ]
            )

        # Slip walls along the whole row, hub and casing
        patches.extend(self._endwall_patches())

        block = ember.block.Block(shape=(ni, nj, nk))
        block.set_label(f"row{i_row}")
        block.set_Nb(machine.rows[i_row].n_blade)
        block.set_x(xrt_now[0])
        block.set_r(xrt_now[1])
        block.set_t(xrt_now[2])
        block.patches.extend(patches)

        # Slip walls in the ducts ahead of the leading edge and behind the
        # trailing edge, from the annulus
        casing = self._inviscid_ducts(machine, i_row, block, block, ile, ite)
        if njtip:
            # Walls turn with their block unless a `RotatingPatch` overrides
            # them, so a casing over a tip gap needs one. The mesher only places
            # it, at `j=-1` by this mesher's convention; `bconds` sets its speed,
            # so changing speed needs no re-mesh. On a stator it has no effect.
            # Not over an inviscid stretch of the casing, which it would overlap.
            i_rot = casing.get(id(block), (0, -1))
            block.patches.append(
                ember.patch.RotatingPatch(i=i_rot, j=-1, label="casing")
            )

        nic, njc, nkc = block.shape
        assert not (nic - 1) % 8, f"nic-1={nic - 1} not divisible by 8"
        assert not (njc - 1) % 8, f"njc-1={njc - 1} not divisible by 8"
        assert not (nkc - 1) % 8, f"nkc-1={nkc - 1} not divisible by 8"

        return block

    def _tip_block(self, machine, i_row, geom, grids, coords):
        """Assemble the block filling a row's tip clearance.

        The prism between the top of the blade and the casing, which the main
        block leaves empty: its two sides are the two surfaces of the section,
        its floor is the tip of the blade and its roof the casing. Built from
        the coordinates the main block was built from, sliced rather than
        recomputed, so that the faces the two share agree exactly rather than
        to the tolerance of a second evaluation of the same geometry.
        """
        ile, jtip = grids.ile, grids.jtip
        icusp = coords.ite + grids.ni_cusp - 1 if grids.ni_cusp else coords.ite
        ni = icusp - ile + 1
        nj = grids.nj - jtip
        nk = self.nk_tip

        assert not (ni - 1) % 8, f"tip block ni-1={ni - 1} not divisible by 8"
        assert not (nj - 1) % 8, f"tip block nj-1={nj - 1} not divisible by 8"

        # The section's two angular limits over the window, upper first. These
        # meet at both ends of the window: at the leading edge because both
        # surfaces start there, and at the trailing edge or cusp tip because
        # that is where they are joined. So the block is degenerate at i=0 and
        # i=-1 by construction rather than by a tolerance.
        theta_lim = coords.theta_lim[:, ile : icusp + 1, jtip:]
        xr = coords.xr[:, ile : icusp + 1, jtip:]

        # Across the thickness: one distribution per streamwise station, sized
        # so the cell against the surface has the wall spacing all along the
        # chord and matches the passage block across the shared face. Where the
        # section is too thin to hold that spacing, near the ends, the cells are
        # even instead; Vinokur stretching makes the changeover smooth.
        thick = (theta_lim[0] - theta_lim[1]) * xr[1]
        d_surf = geom.drt_norm * geom.pitch_rtheta_max
        uniform = np.linspace(0.0, 1.0, nk)

        frac = np.empty((ni, nk))
        n_even = 0
        for i in range(ni):
            # One thickness per station, averaged over the height of the gap:
            # the section changes little over a clearance, and a distribution
            # per station keeps the k lines of a station together.
            t_i = float(np.mean(thick[i]))
            dmin = d_surf / t_i if t_i > 0.0 else np.inf
            if dmin >= 1.0 / (nk - 1):
                frac[i] = uniform
                n_even += 1
                continue
            try:
                frac[i] = clusterfunc.symmetric.fixed(dmin, nk)
            except clusterfunc.exceptions.ClusteringException:
                frac[i] = uniform
                n_even += 1
        logger.debug(
            f"Tip block: {ni - n_even} of {ni} stations clustered to "
            f"dsurf={d_surf:.3g}, the rest too thin and spaced evenly."
        )
        frac = frac.reshape(ni, 1, nk)

        xrt_now = np.empty((3, ni, nj, nk))
        xrt_now[:2] = xr[..., np.newaxis]

        # k=-1 is the upper surface, k=0 the lower, which is the sense the main
        # block ends up in after its own pitchwise flip: its k=0 face is the
        # upper limit and its k=-1 face the lower one a pitch away. Meeting
        # face to face, the two blocks run their k in opposite directions.
        np.add(
            frac * theta_lim[0][..., np.newaxis],
            (1.0 - frac) * theta_lim[1][..., np.newaxis],
            out=xrt_now[2],
        )
        assert np.isfinite(xrt_now).all()

        patches = [
            # The two sides of the section, which the main block has as the two
            # sides of its passage. The upper is coincident with the main
            # block's k=0 face; the lower is its k=-1 face, one pitch away.
            ember.patch.PeriodicPatch(k=0, label="tip_lower"),
            ember.patch.PeriodicPatch(k=-1, label="tip_upper"),
            # Collapsed to a line at each end of the chord. No area, so no
            # flux and no boundary condition -- but they have to be told apart
            # from walls, which every unpatched face otherwise is.
            ember.patch.NotWallPatch(i=0, label="tip_LE"),
            ember.patch.NotWallPatch(i=-1, label="tip_TE"),
            # The casing stands still over a turning row, as it does over the
            # main block; `bconds` values it.
            ember.patch.RotatingPatch(j=-1, label="casing"),
        ]
        # j=0 is left bare: it is the tip of the blade, a wall turning with the
        # row, which is what an unpatched face already means.

        block = ember.block.Block(shape=(ni, nj, nk))
        block.set_label(f"row{i_row}_tip")
        block.set_Nb(machine.rows[i_row].n_blade)
        block.set_x(xrt_now[0])
        block.set_r(xrt_now[1])
        block.set_t(xrt_now[2])
        block.patches.extend(patches)

        return block

    def _stitch_mixing_planes(self, passages):
        """Force xr coordinates to match exactly at mixing planes.

        Takes the passage blocks in row order rather than the grid: a tip
        block is inside a row and has no mixing plane of its own, so once one
        exists the block index and the row index are different numbers.
        """

        # `x` and `r` sliced apart rather than `xrt[..., :2]`, which is a
        # derived array: reading it materialises and copies the whole block --
        # tens of megabytes of three-component nodes -- to be thrown away
        # bar one face. Slicing first costs a hundredth of that and reads the
        # same numbers.
        def face(block, i):
            return np.stack((block.x[i, :, 0], block.r[i, :, 0]), axis=-1)

        for i_row in range(len(passages) - 1):
            xr0 = face(passages[i_row], -1)
            xr1 = face(passages[i_row + 1], 0)
            xrav = 0.5 * (xr0 + xr1)
            xav = xrav[..., 0, None]
            rav = xrav[..., 1, None]
            passages[i_row][-1].set_x(xav)
            passages[i_row + 1][0].set_x(xav)
            passages[i_row][-1].set_r(rav)
            passages[i_row + 1][0].set_r(rav)

    #
    # ONE-DIMENSIONAL GRID VECTORS
    #

    def spanwise_grid(self, dspf_hub, dspf_casing, tip):
        """Evaluate a spanwise grid vector given hub and casing spacings.

        ``dspf_casing`` is ignored when :attr:`half_span` is set, and both
        with :attr:`inviscid_endwalls`, which spaces the span evenly.
        """
        if self.inviscid_endwalls:
            return self._uniform_span(self.dspf_mid / self.resolution_factor, tip)
        if tip and self.half_span:
            raise ValueError(
                "half_span clusters at the hub only, but a tip gap needs "
                "the casing clustering; set one or the other."
            )

        if tip:
            Lmain = 1.0 - tip

            # We want at least njtip_min nodes across the tip gap
            # So the minimum grid spacing should be the smallest of:
            #   - njtip_min pts uniform, which is njtip_min - 1 cells
            #   - target shroud spacing
            # Dividing by the node count instead leaves the fallback five
            # spacings to fill with four cells, pinned at both ends, so the
            # middle two come out half as large again as their neighbours.
            njtip_min = self.njtip_min
            if self.nk_tip:
                # Gridded, the gap is a block and its spanwise count is the
                # block's, so it has to be one a multigrid level can halve.
                # Pinched it is a slice of one, free to be the five a tip flow
                # has been found to run at, and the fallback below is where
                # that five is spent exactly.
                njtip_min = int(8 * np.ceil((njtip_min - 1) / 8)) + 1
            dspf_tip = np.minimum(dspf_casing, tip / (njtip_min - 1))

            spf_main = clusterfunc.double.plateau_free(
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

            # Only the main passage takes the resolution factor. Resampling the
            # tip gap with mult=8 rounded its four cells up to eight at any
            # factor but one, halving the casing cell with it; left alone, the
            # gap and the casing wall spacing hold across a resolution study.
            spf_main = util.resample(spf_main, self.resolution_factor, mult=8)

            # The two pieces share a node, and the total must be one more than a
            # multiple of eight for multigrid. The fallback above can give the
            # tip any count, so pad the main passage, where an extra node barely
            # changes the spacing, rather than the tip gap.
            deficit = -(len(spf_main) + len(spf_tip) - 2) % 8
            if deficit:
                logger.debug(
                    f"Adding {deficit} spanwise points to the main passage so "
                    f"the span is a multigrid count."
                )
                spf_main = util.resample_to(spf_main, len(spf_main) + deficit)

            spf = np.concatenate((spf_main[:-1], spf_tip))

            assert spf[0] == 0.0
            assert np.isclose(spf[-1], 1.0)
            assert (np.diff(spf) > 0.0).all()
            assert (spf >= tip).sum() >= njtip_min

            return spf

        elif self.half_span:
            return util.resample(
                clusterfunc.single.plateau_free(dspf_hub, self.dspf_mid, self.ER_span),
                self.resolution_factor,
                mult=8,
            )

        else:
            return util.resample(
                clusterfunc.double.plateau_free(
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

        x = clusterfunc.symmetric.plateau_free(drt_row, drt_mid, self.ER_pitch)

        dx = np.diff(x)
        assert np.isclose(x[0], 0.0)
        assert np.isclose(x[-1], 1.0)
        assert np.all(dx > 0.0)
        ERout = dx[1:] / dx[:-1]
        ERout[ERout < 1.0] = 1.0 / ERout[ERout < 1.0]
        assert np.all(ERout <= self.ER_pitch)
        assert np.isfinite(x).all()

        if resample:
            x_free = x
            x = util.resample(x_free, self.resolution_factor, mult=8)
            if len(x) < self.nk_min:
                npts = int(8 * np.ceil((self.nk_min - 1) / 8)) + 1
                # More points than the plateau needs at full resolution can
                # always be fitted, by ramping more gently. Fewer cannot, so a
                # coarsened grid is resampled to the minimum count instead,
                # coarsening the wall cell with it as the resolution factor
                # does everywhere else.
                try:
                    x = clusterfunc.symmetric.plateau_fixed(
                        drt_row, drt_mid, self.ER_pitch, npts
                    )
                except clusterfunc.exceptions.ClusteringException:
                    x = util.resample_to(x_free, npts)

        return x

    def pitchwise_grid_fixed_npts(self, drt_row, pitch_chord, AR_row, npts):
        """Evaluate a pitchwise grid vector with a prescribed point count."""
        dm_mid = self.dspf_mid * AR_row / self.AR_merid
        drt_mid = dm_mid / pitch_chord * self.AR_passage
        x = clusterfunc.symmetric.plateau_fixed(drt_row, drt_mid, self.ER_pitch, npts)

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
        tte,
        chord_factor=1.0,
        L_cusp=0.0,
        dm_LE=None,
    ):
        """Evaluate streamwise grid vector for a blade row.

        `tte` is where the trailing edge closure starts, as a fraction of
        chord. The closure is filled with cells of :attr:`frac_TE` times the
        mid-chord cap, at least one, and the blade and downstream grids meet it
        at that size. `L_cusp` is the length of a cusp past the trailing edge,
        normalised by the downstream gap chord, or zero for none: the tip is
        put on the downstream node nearest it in ratio, and the rest of the gap is
        given a multiple of eight cells so the periodic patches from the tip
        line up with a multigrid level.

        `dm_LE` is the LE spacing normalised by meridional chord, defaulting
        to :attr:`dm_LE`.

        Returns the grid, the indices of the leading and trailing edges, and
        the number of points along the cusp, trailing edge and tip included,
        or zero without one.
        """
        if dm_LE is None:
            dm_LE = self.dm_LE
        if not dm_LE > 0.0:
            raise ValueError(f"LE grid spacing must be positive, got dm_LE={dm_LE}")

        assert len(pitch_chord) == 3
        assert (pitch_chord > 0.0).all()
        assert nrt > 1
        assert 0.0 < tte < 1.0

        # Normalised grid spacings at endpoints (normalised by their gap chord).
        # An end is a true machine boundary when its inlet/exit length is the
        # full 1.0 gap-chord, and a mixing plane when it is the 0.5 half-gap ---
        # so `L` tells the two ends apart without threading the row index down.
        ar_up = self.AR_stream if L[0] == 1.0 else self.AR_mix
        ar_dn = self.AR_stream if L[-1] == 1.0 else self.AR_mix
        dm_boundary = np.array([ar_up, self.AR_stream, ar_dn]) * pitch_chord / nrt

        dm_mid = self.dspf_mid * AR_row / self.AR_merid

        # The closure gets whole cells of no more than the target size, and
        # that is the size the blade grid ends on and the downstream one starts.
        n_te = max(1, int(np.ceil((1.0 - tte) / (self.frac_TE * dm_mid))))
        dm_TE = (1.0 - tte) / n_te

        # Convert the LE/TE grid spacings from chord-normalised to gap-normalised
        dm_upstream_LE = dm_LE * pitch_chord[0] / pitch_chord[1]
        dm_downstream_TE = dm_TE * pitch_chord[-1] / pitch_chord[1]

        logger.debug(
            "streamwise_grid drivers: pitch_chord=%s L=%s AR_row=%.4f tte=%.4f "
            "chord_factor=%.4f | dm_boundary=%s dm_mid=%.4g dm_upstream_LE=%.4g "
            "dm_TE=%.4g dm_downstream_TE=%.4g L_cusp=%.4g",
            np.array2string(np.asarray(pitch_chord), precision=4),
            np.array2string(np.asarray(L), precision=4),
            AR_row,
            tte,
            chord_factor,
            np.array2string(np.asarray(dm_boundary), precision=4),
            dm_mid,
            dm_upstream_LE,
            dm_TE,
            dm_downstream_TE,
            L_cusp,
        )

        t_upstream = 1.0 - np.flip(
            clusterfunc.single.free(
                dm_upstream_LE, dm_boundary[0] * L[0], self.ER_stream, 0.0, L[0]
            )
        )

        # Apply chord length adjustment factor
        dm_LE_adj = dm_LE / chord_factor
        dm_mid_adj = dm_mid / chord_factor
        dm_TE_adj = dm_TE / chord_factor

        t_chord = clusterfunc.double.free(
            dm_LE_adj, dm_TE_adj, dm_mid_adj, self.ER_stream, 0.0, tte
        )

        t_te = np.linspace(tte, 1.0, n_te + 1)

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

        # The cusp is the start of the downstream gap, up to the node at its
        # tip; the rest of the gap is what lies beyond.
        if L_cusp:
            if L_cusp >= L[1]:
                raise ValueError(
                    f"The trailing edge cusp reaches {L_cusp:.3g} of the "
                    f"downstream gap, which is only {L[1]:g} long."
                )
            # The node that scales the grid least to put it on the tip; the
            # scaling leaves how fast the cusp grows alone, and is the step
            # on to it from the blade.
            ic = 1 + int(np.argmin(np.abs(np.log(t_downstream[1:-1] / L_cusp))))
            t_cusp = util.resample(t_downstream[: ic + 1], self.resolution_factor)
            t_rest = t_downstream[ic:]
        else:
            t_cusp = t_downstream[:1]
            t_rest = t_downstream
        t_rest = util.resample(t_rest, self.resolution_factor, mult=8)
        n_rest = 8 * int(np.ceil((len(t_rest) - 1) / 8)) + self.ni_down_extra
        t_rest = util.resample_to(t_rest, n_rest + 1)
        t_downstream = np.concatenate((t_cusp, t_rest[1:]))
        ni_cusp = len(t_cusp) if L_cusp else 0

        t_upstream = util.resample(t_upstream, self.resolution_factor, mult=8)
        t_te = util.resample(t_te, self.resolution_factor)
        t_chord = util.resample(t_chord, self.resolution_factor, mult=8)

        # The upstream gap and the rest of the downstream one are already in
        # multiples of eight cells, so the blade takes up whatever the closure
        # and the cusp leave for the total to be one too.
        ni_total = len(t_upstream) + len(t_chord) + len(t_te) + len(t_downstream) - 3
        chord_deficit = (8 - (ni_total - 1) % 8) % 8
        if chord_deficit:
            t_chord = np.interp(
                np.linspace(0.0, 1.0, len(t_chord) + chord_deficit),
                np.linspace(0.0, 1.0, len(t_chord)),
                t_chord,
            )

        logger.debug(
            "streamwise segments (upstream, chord, te, cusp, downstream) = %s, "
            "chord_deficit=%d -> ni=%d",
            (len(t_upstream), len(t_chord), len(t_te), ni_cusp, len(t_downstream)),
            chord_deficit,
            len(t_upstream) + len(t_chord) + len(t_te) + len(t_downstream) - 3,
        )

        t = np.concatenate(
            [t_upstream - 1.0, t_chord[1:], t_te[1:], t_downstream[1:] + 1.0]
        )

        dt = np.diff(t)
        assert (dt > 0.0).all()

        ile = len(t_upstream) - 1
        ite = ile + len(t_chord) + len(t_te) - 2

        return t, ile, ite, ni_cusp

    def pitchwise_relaxation(self, stream_frac, pitch_chord):
        """Relax clustering towards a uniform distribution at inlet and exit."""
        dt_relax = (
            np.ones((2,)) * self.nchord_relax * pitch_chord[(0, -1),] / pitch_chord[1]
        )
        relax_ref = np.array([1.0, 0.0, 0.0, 1.0])
        t_ref = np.array([-dt_relax[0], 0.0, 1.0, 1.0 + dt_relax[1]])
        return np.interp(stream_frac, t_ref, relax_ref)


def _te_corner_curvature(tq, theta, m_lo, m_hi):
    """Locate the TE corner as the peak curvature of theta(m), sub-sample.

    The corner is the sharpest turning point of the surface theta distribution
    near the TE. On the uniform tq grid the discrete second difference is
    proportional to curvature; the peak is refined to sub-sample resolution by
    a parabolic fit through the three points about the discrete maximum. This
    is a single continuous criterion (no zero-crossing set membership, no grid
    quantisation), so the returned location varies smoothly with span unless the
    underlying geometry genuinely has a spanwise step.

    The window is closed at `m_hi`, the trailing edge, at both ends. The blade
    stops there and theta is held constant beyond it, so the first stencil
    centred downstream of the trailing edge straddles the whole of that kink
    while the one before it straddles only part: left open, the search would
    take the corner to sit a node past the blade whenever the tq grid has no
    node on the trailing edge itself. Only a tq that stops at the trailing edge
    is safe without this, which is what the callers below happen to pass.

    Returns (tte, strength) or (None, 0.0) if no interior peak is found in the
    window m_lo < tq <= m_hi. strength is the peak |curvature|, for picking a
    surface.
    """
    # Second difference ~ curvature (uniform tq spacing)
    d2 = theta[:-2] - 2.0 * theta[1:-1] + theta[2:]
    m_c = tq[1:-1]  # centres of the second-difference stencil
    win = (m_c > m_lo) & (m_c <= m_hi)
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
    # The refinement shifts by up to half a cell, so cap it at the trailing
    # edge as well: the corner is on the blade or it is nowhere.
    tte = min(tq[ipk + 1] + delta * dtq, m_hi)
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
    m_lo = mlim[1] - 0.2 * np.ptp(mlim)
    tte_l, str_l = _te_corner_curvature(tq, theta_l, m_lo, mlim[1])
    tte_u, str_u = _te_corner_curvature(tq, theta_u, m_lo, mlim[1])
    if tte_l is None and tte_u is None:
        # Fallback: cluster grid over last 1.0% chord
        tte = mlim[1] - 0.01 * np.ptp(mlim)
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


# Meridional-chord fraction the TE slope is taken over, from two of these to
# one of these upstream of the section's TE corner. Short enough to be the
# tangent at any grid resolution, and stopping short of the corner, which is
# located only to sub-sample accuracy on its own 500-point search.
_DM_TE_SLOPE = 2e-3


def _casing_warp(d, r, band):
    """Return distances from the casing `d`, as fractions of span, stretched.

    Within `band` of the casing every distance is multiplied by `r`. Over the
    next four `band` of span the factor eases, on a smoothstep, to a constant
    `c` that carries on to the hub and is chosen so that the hub stays at one.
    The mapping is the integral of that factor, so its slope -- the ratio of
    new to old spanwise spacing -- is continuous, and it is monotonic for
    any positive `c`. Easing over a single `band` raised the worst spanwise
    expansion ratio of a radial impeller from 1.19 to 1.35, the gap there
    being stretched threefold at the trailing edge; over four it stays at
    1.19.
    """
    D1, D2 = band, 5.0 * band
    W = D2 - D1
    c = (1.0 - r * D1 - 0.5 * r * W) / (0.5 * W + 1.0 - D2)
    if c <= 0.0:
        raise ValueError(
            f"A tip gap stretched by {r:.3g} near the casing leaves no span "
            f"for the rest of the blade; the gap is too large for its annulus."
        )
    d = np.asarray(d, dtype=float)
    x = np.clip((d - D1) / W, 0.0, 1.0)
    S_int = W * (x**3 - 0.5 * x**4)  # Integral of the smoothstep from D1
    inner = r * np.minimum(d, D1)
    blend = r * (np.minimum(d, D2) - D1).clip(0.0) + (c - r) * S_int
    outer = c * (d - D2).clip(0.0)
    return inner + blend + outer


def _no_cusp_tip(where):
    """Return the error for blade surfaces that never meet past the TE."""
    return ValueError(
        f"A trailing edge cusp runs each blade surface on along its own "
        f"slope to where the two meet, but at {where} the surfaces are "
        f"parallel or spreading apart at the trailing edge, so never meet. "
        f"Thin the blade towards its trailing edge, or mesh it square with "
        f"cusp: false."
    )


@dataclasses.dataclass(frozen=True, slots=True)
class _CuspSection:
    """The cusp of one blade section, in the conformal plane.

    Positions `x` are in `m'` from the trailing edge, negative upstream. Side
    `k` has the corner and leaves the blade there; the other side leaves at
    the trailing edge. Each side is the cubic
    ``theta0 + s*X + kappa/2*X**2 + c*X**3`` in the distance `X` from where it
    leaves, matching the blade's slope and curvature there and reaching the
    tip, which is where the two sides' tangent lines meet.
    """

    k: int
    x0: np.ndarray
    theta0: np.ndarray
    s: np.ndarray
    kappa: np.ndarray
    c: np.ndarray
    x_tip: float
    theta_tip: float
    L_cusp: float
    """How far the tip is past the TE, as a fraction of the downstream gap
    chord in the grid's streamwise coordinate."""

    def theta(self, side, x):
        """Return the angle of `side` at `x`, on the cubic."""
        X = np.asarray(x) - self.x0[side]
        return (
            self.theta0[side]
            + self.s[side] * X
            + 0.5 * self.kappa[side] * X**2
            + self.c[side] * X**3
        )


def _cusp_section(blade, annulus, ist, L_dn, spf, m_edge, m_corner, k=None):
    """Return the :class:`_CuspSection` of `blade` at span fraction `spf`.

    `m_edge` and `m_corner` are row `m` of the edges and of the TE corner, and
    `L_dn` the length of the downstream gap. The corner side `k` is the one
    whose slope turns most between the corner and the TE, unless given.

    Each side's value, slope and second derivative are those of a quadratic
    through three points of the section upstream of where it leaves the
    blade, `_DM_TE_SLOPE` apart, so that none lands past a corner located only
    to sub-sample accuracy.
    """
    h = _DM_TE_SLOPE * np.arange(3.0, 0.0, -1.0)
    u = np.linspace(m_corner - 4.0 * _DM_TE_SLOPE, 1.0 + L_dn, 4001)
    x, r = annulus.evaluate_xr(u + ist + 1.0, spf)
    dmp = np.hypot(np.diff(x), np.diff(r)) / (0.5 * (r[1:] + r[:-1]))
    mp = np.concatenate(((0.0,), np.cumsum(dmp)))
    mp -= np.interp(m_edge[1], u, mp)
    xrt_u, xrt_l = _by_theta(blade, spf)

    def fit(u_end):
        """Value, slope and second derivative of each side at `u_end`."""
        u_fit = u_end - h
        th = np.stack(_theta_limits(u_fit, xrt_u, xrt_l, m_edge)[:2])
        x0 = np.interp(u_end, u, mp)
        X = np.interp(u_fit, u, mp) - x0
        c2, c1, c0 = np.polyfit(X, th.T, 2)
        return x0, c0, c1, 2.0 * c2

    x_c, th_c, s_c, kap_c = fit(m_corner)
    x_t, th_t, s_t, kap_t = fit(m_edge[1])
    if k is None:
        k = int(np.argmax(np.abs(s_t - s_c)))

    x0 = np.array((x_t, x_t))
    th0, s, kap = th_t.copy(), s_t.copy(), kap_t.copy()
    x0[k], th0[k], s[k], kap[k] = x_c, th_c[k], s_c[k], kap_c[k]

    # The tip is where the two tangent lines meet.
    with np.errstate(divide="ignore", invalid="ignore"):
        x_tip = (th0[1] - th0[0] + s[0] * x0[0] - s[1] * x0[1]) / (s[0] - s[1])
    if not np.isfinite(x_tip) or x_tip <= 0.0:
        raise _no_cusp_tip(f"spf={spf:g}")
    th_tip = th0[0] + s[0] * (x_tip - x0[0])
    X = x_tip - x0
    c = (th_tip - th0 - s * X - 0.5 * kap * X**2) / X**3

    u_tip = np.interp(x_tip, mp, u)
    frac = (u_tip - m_edge[1]) / (1.0 + L_dn - m_edge[1])
    cusp = _CuspSection(k, x0, th0, s, kap, c, x_tip, th_tip, frac * L_dn)

    # The sides must stay apart until the tip: from the corner to the TE the
    # other side is the blade's own surface, past it its cubic.
    xq = np.linspace(x0[k], x_tip, 401)[:-1]
    other = 1 - k
    th_other = cusp.theta(other, xq)
    upstream = xq < 0.0
    th_other[upstream] = _theta_limits(
        np.interp(xq[upstream], mp, u), xrt_u, xrt_l, m_edge
    )[other]
    th_k = cusp.theta(k, xq)
    gap = th_k - th_other if k == 0 else th_other - th_k
    if np.any(gap <= 0.0):
        x_bad = float(xq[np.argmax(gap <= 0.0)])
        where = (
            "over the trailing edge closure"
            if x_bad < 0.0
            else f"{x_bad / x_tip:.2f} of the way from the trailing edge to the tip"
        )
        raise ValueError(
            f"The trailing edge cusp at spf={spf:g} carries each blade surface "
            f"on with its own slope and curvature to the tip where their "
            f"tangents meet, but the two sides cross {where}. The surfaces "
            f"curve too sharply towards each other at the trailing edge; mesh "
            f"it square with cusp: false."
        )
    return cusp


def _meridional(xr):
    """Return arc length `m` and conformal coordinate `m'` along grid lines.

    `xr` is shaped (2, ni, nj). Both results are (ni, nj), zero at i = 0, and
    summed over the chords of the grid, with `dm' = dm / r` at each chord's
    mid-radius.
    """
    dm = np.hypot(*np.diff(xr, axis=1))
    dmp = dm / (0.5 * (xr[1, 1:] + xr[1, :-1]))
    zero = np.zeros((1, xr.shape[2]))
    return tuple(np.concatenate((zero, np.cumsum(d, axis=0))) for d in (dm, dmp))


def add_cusp(xr, theta, u, xr_of_u, i_corner, iTE, ni_cusp, cusps):
    """Extend a square trailing edge into a cusp, at any exit pitch angle.

    `xr` (2, ni, nj) is the meridional grid, shared by both surfaces, and
    `theta` (2, ni, nj) the angle of each surface, or of the periodic line off
    the blade. The TE corner must be exactly at `i_corner` and the trailing
    edge at `iTE`. The new `xr` and `theta` are returned with the same number
    of streamwise points: the cusp is the first `ni_cusp - 1` cells of the
    downstream gap, up to the node chosen as the tip, scaled to end on it.
    Its cells are the grid's own, scaled together to fit, so they grow as
    fast as the grid did, and the step from the blade's last cell to the first
    of them is the scaling. The grid beyond starts from its last cell.

    `cusps` is the :class:`_CuspSection` at each station `j`. The side with
    the corner follows its cubic from `i_corner`, over the closure, and the
    other from `iTE`; both to the tip.

    Each station `j` is a line on the annulus, worked on in three coordinates.
    Slopes are taken in the conformal coordinate `m'`, from
    :func:`_meridional`, in which a straight line keeps its blade angle at any
    pitch angle of the annulus. Spacings are taken in arc length `m`. Every
    point is put back on the annulus through its annulus coordinate `u`
    (ni, nj), interpolated in `m` and evaluated by `xr_of_u`.
    """
    assert ni_cusp >= 2
    ni, nj = u.shape
    m, mp = _meridional(xr)

    ic = iTE + ni_cusp - 1  # Index of the tip
    m_new = m.copy()
    th_new = theta.copy()
    for j in range(nj):
        mj, mpj = m[:, j], mp[:, j]
        cusp = cusps[j]
        k = cusp.k

        # Over the closure, the side with the corner is carried on from it.
        x = mpj[i_corner : iTE + 1] - mpj[iTE]
        th_new[k, i_corner : iTE + 1, j] = cusp.theta(k, x)

        # The cusp, on the grid's own nodes to the one chosen as the tip,
        # scaled in `m` to land on it.
        m_tip = np.interp(mpj[iTE] + cusp.x_tip, mpj, mj)
        f = (mj[iTE : ic + 1] - mj[iTE]) / (mj[ic] - mj[iTE])
        m_new[iTE : ic + 1, j] = mj[iTE] + f * (m_tip - mj[iTE])
        x = np.interp(m_new[iTE : ic + 1, j], mj, mpj) - mpj[iTE]
        for side in range(2):
            th_new[side, iTE : ic + 1, j] = cusp.theta(side, x)
        th_new[:, ic, j] = cusp.theta_tip

        # Downstream, the periodic line is moved to start at the tip and eased
        # back by the outlet, and respaced in `m` from the cusp's last cell to
        # the grid's.
        md = mj[iTE:]
        line = theta[0, iTE:, j] + cusp.theta_tip - theta[0, iTE, j]
        dt_tip = cusp.theta_tip - np.interp(m_tip, md, line)
        line += np.interp(md, [m_tip, md[-1]], [dt_tip, 0.0])
        m_new[ic:, j] = clusterfunc.double.fixed(
            m_new[ic, j] - m_new[ic - 1, j], md[-1] - md[-2], ni - ic, m_tip, md[-1]
        )
        th_new[:, ic:, j] = np.interp(m_new[ic:, j], md, line)

    # Back on to the annulus, through its own coordinate.
    u_new = np.stack([np.interp(m_new[:, j], m[:, j], u[:, j]) for j in range(nj)], 1)
    return xr_of_u(u_new), th_new
