"""Annulus geometry.

An :class:`AnnulusDesign` is a config node describing the hub and casing lines;
designing one produces an :class:`Annulus`, which holds the fitted curves and
the geometry read off them. The split matters: the package this replaces stored
the fitted splines on the designer itself, so the config and the result were one
object and an un-designed annulus had no defined state.

Only one design is provided, :class:`FixedAxialChord`. See ARCHITECTURE.md for
why: the four classes it replaces are a 2x2 of chord specification against
merged-or-not, and merging is a continuous parameter rather than a type.
"""

import logging
from typing import ClassVar

import numpy as np
from scipy.interpolate import PchipInterpolator

import turbigen.util
from turbigen2.node import Node

logger = logging.getLogger("turbigen")


def _segment_average(values):
    """Average a (2*n_row,) per-station array to (2*n_row+1,) segment values."""
    inner = 0.5 * (values[:-1] + values[1:])
    return np.concatenate([[values[0]], inner, [values[-1]]])


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


class Annulus:
    """Hub and casing lines of a designed annulus.

    Coordinates are addressed by a normalised meridional coordinate ``m``,
    where 0 is the inlet, 1 the first row leading edge, 2 its trailing edge and
    so on, and a span fraction ``spf`` running 0 at the hub to 1 at the casing.
    """

    def __init__(self, s, curves, curves_merged, merge_weight, n_row):
        self._s = s
        self._curves = curves
        self._curves_merged = curves_merged
        self._merge_weight = float(merge_weight)
        self._n_row = int(n_row)

    def __repr__(self):
        return f"Annulus(n_row={self.n_row}, merge_weight={self._merge_weight:.3g})"

    #
    # STRUCTURE
    #

    @property
    def n_row(self):
        """Number of blade rows."""
        return self._n_row

    @property
    def n_segment(self):
        """Number of segments, being the rows and the gaps between them."""
        return 2 * self._n_row + 1

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
        sq = np.interp(mb, np.arange(len(self._s)), self._s)

        weight = self._merge_weight
        if weight == 0.0:
            xhub, rhub, xcas, rcas = (curve(sq) for curve in self._curves)
        elif weight == 1.0:
            xhub, rhub, xcas, rcas = (curve(sq) for curve in self._curves_merged)
        else:
            xhub, rhub, xcas, rcas = (
                (1.0 - weight) * plain(sq) + weight * merged(sq)
                for plain, merged in zip(self._curves, self._curves_merged)
            )

        x = (1.0 - spfb) * xhub + spfb * xcas
        r = (1.0 - spfb) * rhub + spfb * rcas
        return np.stack([x, r])

    def _xr_stations(self):
        """Hub and casing coordinates at every row inlet and outlet."""
        m = np.arange(1, 2 * self.n_row + 1, dtype=float)
        return self.evaluate_xr(m, spf=0.0), self.evaluate_xr(m, spf=1.0)

    #
    # GEOMETRY AT THE STATIONS
    #

    @property
    def r_hub(self):
        """Hub radii at all row inlet and outlet stations [m]."""
        return self._xr_stations()[0][1]

    @property
    def r_tip(self):
        """Casing radii at all row inlet and outlet stations [m]."""
        return self._xr_stations()[1][1]

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

    @property
    def Am(self):
        """Annular flow area at all row inlet and outlet stations [m^2]."""
        return np.pi * (self.r_tip**2 - self.r_hub**2)

    @property
    def x_rms(self):
        """Axial coordinates at the RMS radius, at all stations [m]."""
        xr_hub, xr_cas = self._xr_stations()
        spf_rms = (self.r_rms - xr_hub[1]) / (xr_cas[1] - xr_hub[1])
        return xr_hub[0] + (xr_cas[0] - xr_hub[0]) * spf_rms

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

    def to_string(self):
        """Tabular string representation of the annulus at row stations."""
        cx_row = self.chords(0.5)[1::2]
        span_stations = self.r_tip - self.r_hub
        span_row = 0.5 * (span_stations[::2] + span_stations[1::2])
        properties = [
            ("r_rms/m", self.r_rms, ".4f"),
            ("r_hub/m", self.r_hub, ".4f"),
            ("r_tip/m", self.r_tip, ".4f"),
            ("Am/m2", self.Am, ".4f"),
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


class FixedAxialChord(AnnulusDesign):
    """Smooth annulus with a prescribed axial chord for each row and gap."""

    type: ClassVar[str] = "fixed_axial_chord"

    cx_row: tuple
    """Axial chord of each blade row [m], length n_row."""

    cx_gap: tuple
    """Axial chord of each gap, including the inlet and exit ducts [m],
    length n_row + 1."""

    nozzle_ratio: float = 1.0
    """Scaling applied to the exit span, for a nozzle area ratio [--]."""

    merge_weight: float = 0.0
    """Blend towards a curve fitted through the endpoints only [--].

    At 0 the hub and casing pass through every station. At 1 they follow a
    curve fitted through the first and last segments alone, which smooths
    curvature across the rows at the cost of missing the intermediate
    stations. Values between blend the two.
    """

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

        cx_row = np.asarray(self.cx_row, dtype=float)
        cx_gap = np.asarray(self.cx_gap, dtype=float)
        if cx_row.shape != (n_row,):
            raise ValueError(
                f"cx_row must have one value per row, expected {(n_row,)} "
                f"but got {cx_row.shape}."
            )
        if cx_gap.shape != (n_row + 1,):
            raise ValueError(
                f"cx_gap must have one value per gap, expected {(n_row + 1,)} "
                f"but got {cx_gap.shape}."
            )
        if not 0.0 <= self.merge_weight <= 1.0:
            raise ValueError(f"merge_weight={self.merge_weight} must lie in [0, 1].")

        # Interleave the axial chords across the segments: gaps are even,
        # rows are odd.
        Dx = np.empty(n_segment)
        Dx[::2] = cx_gap
        Dx[1::2] = cx_row

        # Convert to meridional arc length using the average pitch angle
        cos_Beta = np.cos(np.radians(_segment_average(Beta)))
        Ds = Dx / cos_Beta

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
