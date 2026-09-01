"""Annulus geometry.

An :class:`AnnulusDesign` is a config node describing the hub and casing lines;
designing one produces an :class:`Annulus`, which holds the fitted curves and
the geometry read off them. The split matters: the package this replaces stored
the fitted splines on the designer itself, so the config and the result were one
object and an un-designed annulus had no defined state.

Two designs are provided, :class:`FixedAxialChord` and :class:`AspectRatio`,
and they differ in one number per segment. See ARCHITECTURE.md for why: the
four classes they replace are a 2x2 of chord specification against
merged-or-not, merging is a continuous parameter rather than a type, and what
remains is a single choice of how a segment's length is stated.
"""

import dataclasses
import functools
import logging
from typing import ClassVar

import numpy as np
from scipy.interpolate import PchipInterpolator

import turbigen.util
from turbigen.node import Node

logger = logging.getLogger("turbigen")


def _segment_average(values):
    """Average a (2*n_row,) per-station array to (2*n_row+1,) segment values."""
    inner = 0.5 * (values[:-1] + values[1:])
    return np.concatenate([[values[0]], inner, [values[-1]]])


def _interleave(row, gap, n_segment, name_row, name_gap):
    """Lay per-row and per-gap values out over the segments.

    Segments alternate gap, row, gap, ..., gap, so the gaps take the even
    positions and the rows the odd ones. The two lengths are checked here
    rather than in each design, because getting them wrong is the same mistake
    whatever the values mean.
    """
    n_row = (n_segment - 1) // 2
    row = np.asarray(row, dtype=float)
    gap = np.asarray(gap, dtype=float)

    if row.shape != (n_row,):
        raise ValueError(
            f"{name_row} must have one value per row, expected {(n_row,)} "
            f"but got {row.shape}."
        )
    if gap.shape != (n_row + 1,):
        raise ValueError(
            f"{name_gap} must have one value per gap, expected {(n_row + 1,)} "
            f"but got {gap.shape}."
        )

    values = np.empty(n_segment)
    values[::2] = gap
    values[1::2] = row
    return values


def _fit_pchips(s_init, xhub, rhub, xcas, rcas, Ds_target, rtol=1e-6, max_iter=20):
    """Fit PCHIP curves through control points, iterating on arc length.

    The control points are known in (x, r) but their spacing in arc length is
    not, so the parameterisation is refined until each segment's arc length
    matches its target.
    """
    s = np.asarray(s_init, dtype=float).copy()
    n_segment = len(s) - 1

    err = np.inf
    curves = None
    for _ in range(max_iter):
        curves = tuple(
            PchipInterpolator(s, values) for values in (xhub, rhub, xcas, rcas)
        )
        pchip_xhub, pchip_rhub, pchip_xcas, pchip_rcas = curves

        Ds_actual = np.empty(n_segment)
        for k in range(n_segment):
            sq = np.linspace(s[k], s[k + 1], 50)
            xm = 0.5 * (pchip_xhub(sq) + pchip_xcas(sq))
            rm = 0.5 * (pchip_rhub(sq) + pchip_rcas(sq))
            Ds_actual[k] = turbigen.util.arc_length(np.stack([xm, rm]))

        Ds_norm = Ds_actual / Ds_actual.sum() * np.asarray(Ds_target).sum()
        s_new = turbigen.util.cumsum0(Ds_norm)
        err = np.max(np.abs(s_new - s)) / s[-1]
        s = s_new
        if err < rtol:
            break

    if err >= rtol:
        raise RuntimeError(
            f"Annulus arc-length iteration did not converge: err={err:.2e} "
            f"after {max_iter} iterations."
        )

    return s, curves


@dataclasses.dataclass(frozen=True, eq=False)
class StreamSurface:
    """The annulus within one blade row.

    Addressed by a normalised meridional coordinate running 0 at the leading
    edge to 1 at the trailing edge, and a span fraction. A blade design is
    handed one of these rather than the whole annulus and a row index, so that
    the convention mapping a row onto the annulus coordinate stays inside
    :class:`Annulus`, which is the only thing that defines it.
    """

    evaluate_xr: object = dataclasses.field(repr=False)
    """The annulus coordinate map this is a restriction of."""

    m_LE: float
    """Annulus meridional coordinate of this row's leading edge."""

    chord: float
    """Meridional chord of the row at mid-span [m]."""

    def xr(self, m, spf):
        """Return meridional coordinates within the row.

        Takes its arguments in the same order as
        :meth:`Annulus.evaluate_xr`, which it is a restriction of. Both take
        two broadcastable array-likes, so a transposed call would be silent.

        Parameters
        ----------
        m : array_like
            Normalised meridional distance, 0 at the leading edge and 1 at the
            trailing edge. Broadcast against `spf`.
        spf : array_like
            Span fraction, 0 at the hub and 1 at the casing.

        Returns
        -------
        xr : ndarray, shape (2, ...)
            Axial and radial coordinates, stacked on the first axis.

        """
        return self.evaluate_xr(self.m_LE + np.asarray(m, dtype=float), spf)


@dataclasses.dataclass(frozen=True, eq=False)
class Annulus:
    """Hub and casing lines of a designed annulus.

    Coordinates are addressed by a normalised meridional coordinate ``m``,
    where 0 is the inlet, 1 the first row leading edge, 2 its trailing edge and
    so on, and a span fraction ``spf`` running 0 at the hub to 1 at the casing.

    Frozen, like every other result: an annulus is what the design produced and
    nothing downstream has any business changing it. `eq=False` because the
    fields hold arrays and spline objects, so a generated `__eq__` would raise
    on the first comparison -- and unlike a config Node, whose round trip is
    checked by value, nothing ever compares two annuli.
    """

    s: np.ndarray = dataclasses.field(repr=False)
    """Arc-length parameter of each control point."""

    curves: tuple = dataclasses.field(repr=False)
    """Fitted hub and casing curves, (x_hub, r_hub, x_cas, r_cas)."""

    curves_merged: tuple = dataclasses.field(repr=False)
    """The same, fitted through the end segments only, for the merge blend."""

    merge_weight: float
    """Blend between `curves` at 0 and `curves_merged` at 1 [--]."""

    n_row: int
    """Number of blade rows."""

    #
    # STRUCTURE
    #

    @property
    def n_segment(self):
        """Number of segments, being the rows and the gaps between them."""
        return 2 * self.n_row + 1

    @property
    def mmax(self):
        """Largest valid value of the normalised meridional coordinate."""
        return float(self.n_segment)

    #
    # COORDINATES
    #

    def evaluate_xr(self, m, spf):
        """Return meridional coordinates within the annulus.

        Parameters
        ----------
        m : array_like
            Normalised meridional distance. Broadcast against `spf`.
        spf : array_like
            Span fraction, 0 at the hub and 1 at the casing.

        Returns
        -------
        xr : ndarray, shape (2, ...)
            Axial and radial coordinates, stacked on the first axis.

        """
        mb, spfb = np.broadcast_arrays(
            np.asarray(m, dtype=float), np.asarray(spf, dtype=float)
        )
        sq = np.interp(mb, np.arange(len(self.s)), self.s)

        weight = self.merge_weight
        if weight == 0.0:
            xhub, rhub, xcas, rcas = (curve(sq) for curve in self.curves)
        elif weight == 1.0:
            xhub, rhub, xcas, rcas = (curve(sq) for curve in self.curves_merged)
        else:
            xhub, rhub, xcas, rcas = (
                (1.0 - weight) * plain(sq) + weight * merged(sq)
                for plain, merged in zip(self.curves, self.curves_merged)
            )

        x = (1.0 - spfb) * xhub + spfb * xcas
        r = (1.0 - spfb) * rhub + spfb * rcas
        return np.stack([x, r])

    @functools.cached_property
    def _xr_stations(self):
        """Hub and casing coordinates at every row inlet and outlet.

        Cached: every station property below reads it, so printing the annulus
        table used to evaluate the splines ten times over for five rows of
        numbers. Caching is safe because the annulus is frozen, and possible
        because `cached_property` writes the instance dict directly rather than
        going through the `__setattr__` that freezing blocks.
        """
        m = np.arange(1, 2 * self.n_row + 1, dtype=float)
        return self.evaluate_xr(m, spf=0.0), self.evaluate_xr(m, spf=1.0)

    #
    # GEOMETRY AT THE STATIONS
    #

    @property
    def r_hub(self):
        """Hub radii at all row inlet and outlet stations [m]."""
        return self._xr_stations[0][1]

    @property
    def r_tip(self):
        """Casing radii at all row inlet and outlet stations [m]."""
        return self._xr_stations[1][1]

    @property
    def r_mid(self):
        """Mid-span radii at all row inlet and outlet stations [m]."""
        return 0.5 * (self.r_hub + self.r_tip)

    @property
    def r_rms(self):
        """Root-mean-square radii at all row inlet and outlet stations [m]."""
        return np.sqrt(0.5 * (self.r_hub**2 + self.r_tip**2))

    @property
    def htr(self):
        """Hub-to-tip ratio at all row inlet and outlet stations [--]."""
        return self.r_hub / self.r_tip

    def span(self, m):
        """Return the hub-to-casing distance at meridional position(s) `m` [m].

        Parameters
        ----------
        m : array_like
            Normalised meridional distance.

        Returns
        -------
        span : ndarray
            Distance from hub to casing at each position [m].

        """
        xr_hub = self.evaluate_xr(m, 0.0)
        xr_cas = self.evaluate_xr(m, 1.0)
        return np.sqrt(np.sum((xr_cas - xr_hub) ** 2.0, axis=0))

    def chords(self, spf):
        """Return the meridional chord of every segment at span fraction `spf`.

        Segments alternate gap, row, gap, row, ..., gap, so there are
        ``2 * n_row + 1`` of them.
        """
        chords = np.zeros(self.n_segment)
        for i in range(self.n_segment):
            mq = np.linspace(i, i + 1, 100)
            chords[i] = turbigen.util.arc_length(self.evaluate_xr(mq, spf))
        return chords

    def row(self, i_row):
        """Return the stream surfaces within blade row `i_row`.

        Rows occupy the odd segments, so this is where the mapping from a row
        index onto the annulus meridional coordinate lives, and the only place
        it does.
        """
        if not 0 <= i_row < self.n_row:
            raise IndexError(
                f"Blade row {i_row} is out of range for an annulus with "
                f"{self.n_row} rows."
            )
        return StreamSurface(
            self.evaluate_xr,
            m_LE=2 * i_row + 1,
            chord=self.chords(0.5)[2 * i_row + 1],
        )

    def to_string(self):
        """Tabular string representation of the annulus at row stations."""
        m = np.arange(1, 2 * self.n_row + 1, dtype=float)
        span = self.span(m)
        cx_row = self.chords(0.5)[1::2]
        span_row = 0.5 * (span[::2] + span[1::2])
        properties = [
            ("r_rms/m", self.r_rms, ".4f"),
            ("r_hub/m", self.r_hub, ".4f"),
            ("r_tip/m", self.r_tip, ".4f"),
            # Both of these are the true distance and the true area, normal to
            # the meridional flow, so that they agree with MeanLine.span and
            # MeanLine.Am. A radial difference r_tip - r_hub is the span
            # projected onto the axis, shorter by cos(Beta), and
            # pi * (r_tip**2 - r_hub**2) is the area projected the same way.
            ("Am/m2", 2.0 * np.pi * self.r_mid * span, ".4f"),
            ("htr", self.htr, ".4f"),
            ("cx/m", cx_row, ".4f"),
            ("AR", span_row / cx_row, ".4f"),
        ]
        return turbigen.util.format_table("Annulus:", self.n_row, properties)


class AnnulusDesign(Node):
    """Base for annulus designs.

    Unlike a mean-line design, an annulus declares no ``n_row``: it is generic
    over row count, which comes from the mean line handed to :meth:`forward`.
    """

    def forward(self, mean_line) -> Annulus:
        """Return the annulus this design describes for `mean_line`."""
        raise NotImplementedError(
            f"{type(self).__name__} must implement forward(self, mean_line)"
        )

    def design(self, mean_line) -> Annulus:
        """Return the annulus this design describes for `mean_line`."""
        return self.forward(mean_line)


class PchipAnnulus(AnnulusDesign):
    """Hub and casing lines fitted as PCHIP curves in arc-length space.

    Everything about the fit is here: the control points placed from the mean
    line, the arc-length iteration, the duct extensions, the nozzle scaling and
    the merge blend. A member supplies one thing, :meth:`segment_lengths`, and
    that is the whole difference between the designs below.

    Deliberately not selectable --- it declares no ``type``, so it is not in
    the registry and cannot appear in a config file. It is the shared body of
    two designs, not a third one.
    """

    nozzle_ratio: float = 1.0
    """Scaling applied to the exit span, for a nozzle area ratio [--]."""

    merge_weight: float = 0.0
    """Blend towards a curve fitted through the endpoints only [--].

    At 0 the hub and casing pass through every station. At 1 they follow a
    curve fitted through the first and last segments alone, which smooths
    curvature across the rows at the cost of missing the intermediate
    stations. Values between blend the two.
    """

    def segment_lengths(self, span_avg, cos_Beta_avg):
        """Return the meridional arc length of each segment [m].

        Arc length rather than axial length, because it is what both the
        parameterisation and the duct extensions are measured in; the axial
        length follows from the pitch angle. A design that states axial chords
        divides by `cos_Beta_avg` to get here, and one that states aspect
        ratios never needs it at all.

        Parameters
        ----------
        span_avg : ndarray, shape (n_segment,)
            Annulus span averaged over each segment [m].
        cos_Beta_avg : ndarray, shape (n_segment,)
            Cosine of the pitch angle averaged over each segment [--].

        Returns
        -------
        Ds : ndarray, shape (n_segment,)
            Meridional arc length of each segment [m].

        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement "
            f"segment_lengths(self, span_avg, cos_Beta_avg)"
        )

    def forward(self, mean_line):
        # The annulus is addressed by station in streamwise order, whereas a
        # mean line is stored (2, n_row) by station and row.
        flat = mean_line.flat
        r_mid = np.asarray(flat.r_mid, dtype=float)
        span = np.asarray(flat.span, dtype=float)
        Beta = np.asarray(flat.Beta, dtype=float)

        n_station = r_mid.size
        n_row = n_station // 2
        n_segment = 2 * n_row + 1

        if not 0.0 <= self.merge_weight <= 1.0:
            raise ValueError(f"merge_weight={self.merge_weight} must lie in [0, 1].")

        # Ask the design how long each segment is, and take the axial length
        # from the pitch angle rather than the other way round.
        span_avg = _segment_average(span)
        cos_Beta = np.cos(np.radians(_segment_average(Beta)))
        Ds = np.asarray(self.segment_lengths(span_avg, cos_Beta), dtype=float)
        if Ds.shape != (n_segment,):
            raise ValueError(
                f"{type(self).__name__}.segment_lengths must return one length "
                f"per segment, expected {(n_segment,)} but got {Ds.shape}."
            )
        Dx = Ds * cos_Beta

        # Integrate for the mid-span axial coordinates, origin at the first
        # row leading edge
        x_mid = turbigen.util.cumsum0(Dx)
        x_mid -= x_mid[1]

        # Pad the station arrays out to the n_segment + 1 control points
        span_ext = np.pad(span, 1, "edge")
        Beta_ext = np.pad(Beta, 1, "edge")
        r_mid_ext = np.pad(r_mid, 1, "edge")

        # Replace the padded ends with duct extensions at constant pitch angle
        sin_Beta_in, sin_Beta_out = np.sin(np.radians(Beta[[0, -1]]))
        r_mid_ext[0] = r_mid_ext[1] - Ds[0] * sin_Beta_in
        r_mid_ext[-1] = r_mid_ext[-2] + Ds[-1] * sin_Beta_out

        # Scale the exit span for the nozzle area ratio
        span_ext[-1] *= self.nozzle_ratio * r_mid_ext[-2] / r_mid_ext[-1]

        # Hub and casing control points
        sin_B = np.sin(np.radians(Beta_ext))
        cos_B = np.cos(np.radians(Beta_ext))
        xhub = x_mid + 0.5 * span_ext * sin_B
        rhub = r_mid_ext - 0.5 * span_ext * cos_B
        xcas = x_mid - 0.5 * span_ext * sin_B
        rcas = r_mid_ext + 0.5 * span_ext * cos_B

        s_init = turbigen.util.cumsum0(Ds)
        s, curves = _fit_pchips(s_init, xhub, rhub, xcas, rcas, Ds)

        # A second fit through the endpoints only, for the merged blend. When
        # the weight is zero it is never evaluated, so do not pay for it.
        if self.merge_weight > 0.0:
            ends = np.array([0, 1, n_segment - 1, n_segment])
            s_ends = s_init[ends]
            _, curves_merged = _fit_pchips(
                s_ends,
                xhub[ends],
                rhub[ends],
                xcas[ends],
                rcas[ends],
                np.diff(s_ends),
            )
        else:
            curves_merged = curves

        return Annulus(s, curves, curves_merged, self.merge_weight, n_row)


class FixedAxialChord(PchipAnnulus):
    """Annulus with a prescribed axial chord for each row and gap.

    Note that an axial chord cannot describe a segment at 90 degrees pitch
    angle: the arc length it implies is the chord divided by ``cos(Beta)``, so
    a radial segment asks for an infinite one. :class:`AspectRatio` states the
    arc length directly and has no such limit.
    """

    type: ClassVar[str] = "fixed_axial_chord"

    cx_row: tuple[float, ...]
    """Axial chord of each blade row [m], length n_row."""

    cx_gap: tuple[float, ...]
    """Axial chord of each gap, including the inlet and exit ducts [m],
    length n_row + 1."""

    def segment_lengths(self, span_avg, cos_Beta_avg):
        Dx = _interleave(self.cx_row, self.cx_gap, len(span_avg), "cx_row", "cx_gap")
        return Dx / cos_Beta_avg


class AspectRatio(PchipAnnulus):
    """Annulus with a prescribed span-to-chord ratio for each row and gap.

    The chord is meridional, and the span it is measured against is the
    average over the segment, so a row's aspect ratio is set by the mean line
    on both sides of it. This is the specification the design correlations are
    written in --- an aspect ratio is a number a designer carries between
    machines, where an axial chord in metres is not.
    """

    type: ClassVar[str] = "aspect_ratio"

    AR_row: tuple[float, ...]
    """Span-to-meridional-chord ratio of each blade row [--], length n_row."""

    AR_gap: tuple[float, ...]
    """Span-to-meridional-chord ratio of each gap, including the inlet and
    exit ducts [--], length n_row + 1."""

    def __post_init__(self):
        # Checked on the way in rather than at design time, because it needs
        # no mean line: a non-positive aspect ratio is wrong on sight. The old
        # package gave a negative value a second meaning, a segment whose
        # length is chosen to smooth the curvature instead; that is not ported,
        # so it must fail rather than be quietly reinterpreted.
        for name in ("AR_row", "AR_gap"):
            values = np.asarray(getattr(self, name), dtype=float)
            if np.any(values <= 0.0):
                raise ValueError(f"{name} must be positive, got {list(values)}.")

    def segment_lengths(self, span_avg, cos_Beta_avg):
        AR = _interleave(self.AR_row, self.AR_gap, len(span_avg), "AR_row", "AR_gap")
        return span_avg / AR
