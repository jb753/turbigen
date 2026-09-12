"""Classes to represent machine geometry."""

import dataclasses
import math

import numpy as np
from scipy.interpolate import PchipInterpolator

from turbigen_ref import util


def _seg_avg(arr):
    """Average a (2*nrow,) per-station array to (2*nrow+1,) segment values."""
    inner = 0.5 * (arr[:-1] + arr[1:])
    return np.concatenate([[arr[0]], inner, [arr[-1]]])


@dataclasses.dataclass
class Camber:
    """Bernstein polynomial camber line of variable degree.

    The camber line sets the variation in metal angle tangent from leading edge
    to trailing edge as a function of meriondal distance. The angles are
    normalised by their edge values, so linear variation from (0,0) to (1,1)
    corresponds to a quadratic camber line (constant curvature).

    The Bernstein coefficents describe local changes about a linear variation;
    setting all coefficients to zero gives a quadratic camber line as a
    sensible default. The first and last coefficients are assumed to be zero to
    fix the camber line at the endpoints, and only non-zero coefficients need be provided for the design variables.


    """

    order: int
    """Order of the Bernstein polynomial; number of coefficients minus one."""

    coeff: np.ndarray
    """Bernstein polynomial coefficients."""

    @classmethod
    def from_design_vector(cls, q):
        """Create camber line from a vector of design variables.

        Parameters
        ----------
        q: (N,) array-like
            Interior Bernstein coefficients (excludes fixed zero endpoints),
            where N is the order of the polynomial minus one.
        """
        q = np.asarray(q, dtype=float)
        assert q.ndim == 1
        order = len(q) + 1
        return cls(order, q)

    def to_design_vector(self):
        """Convert camber line to a vector of design variables.

        Returns
        -------
        q: (N,) array
            Interior Bernstein coefficients (excludes fixed zero endpoints),
            where N is the order of the polynomial minus one.
        """
        return self.coeff

    def evaluate(self, m):
        """Evaluate camber line at meridional locations.

        Parameters
        ----------
        m: (N,) array
            Normalised meridional distance to evaluate camber at.
        Returns
        -------
        z: (N,) array
            Camber line height at the requested points.
        """
        m = np.asarray(m, dtype=float)
        n = self.order
        i = np.arange(1, n)
        binom = np.array([math.comb(n, k) for k in i], dtype=float)
        # basis: (n-1, len(m))
        basis = (
            binom[:, None]
            * m[None, :] ** i[:, None]
            * (1.0 - m)[None, :] ** (n - i)[:, None]
        )
        return m + self.coeff @ basis


@dataclasses.dataclass
class Thickness:
    """Cubic thickness distribution in Kulfan shape space.

    The trailing edge thickness is input as the total thickness, with half
    contributed by each side.

    All lengths are normalised by the meridional chord.

    """

    R_LE: float
    """Leading-edge radius."""

    t_max: float
    """Maximum thickness."""

    m_tmax: float
    """Meridional location of maximum thickness."""

    t_TE: float
    """Trailing-edge thickness, total for both sides."""

    wedge: float
    """Trailing-edge wedge angle in degrees."""

    @classmethod
    def from_design_vector(cls, q):
        """Create distribution from a vector of design variables.

        Parameters
        ----------
        q: (5,) array-like
            Design vector containing [R_LE, t_max, m_tmax, t_TE, wedge].
        """
        assert q.ndim == 1 and len(q) == 5
        return cls(*q)

    def to_design_vector(self):
        """Convert distribution to a vector of design variables.

        Returns
        -------
        q: (5,) array
            Design vector containing [R_LE, t_max, m_tmax, t_TE, wedge].
        """
        return np.array(
            [
                self.R_LE,
                self.t_max,
                self.m_tmax,
                self.t_TE,
                self.wedge,
            ]
        )

    def __post_init__(self):
        """Calculate polynomial coefficients from geometric parameters."""

        # For brevity
        t_TE = self.t_TE
        R_LE = self.R_LE
        t_max = self.t_max
        m_tmax = self.m_tmax
        wedge = self.wedge

        # We define TE thickness as total due to both sides,
        # so each side contributes half of this value.
        t_TE2 = t_TE / 2.0

        # Evaluate max thickness point in shape space
        tau_max = (t_max - m_tmax * t_TE2) / np.sqrt(m_tmax) / (1.0 - m_tmax)
        num = np.sqrt(m_tmax) - 0.5 * (1.0 - m_tmax) / np.sqrt(m_tmax)
        dtau_max = (tau_max * num - t_TE2) / np.sqrt(m_tmax) / (1.0 - m_tmax)

        # LE and TE values in shape space
        tau0 = 2.0 * np.sqrt(R_LE)
        tau1 = t_TE2 + np.tan(np.radians(wedge))

        # Four constraints fix a cubic polynomial in shape space:
        # tau(0) = tau0
        # tau(m_tmax) = tau_max
        # tau'(m_tmax) = dtau_max
        # tau(1) = tau1
        A = np.array(
            [
                [0.0, 0.0, 0.0, 1.0],
                [m_tmax**3, m_tmax**2, m_tmax, 1.0],
                [3 * m_tmax**2, 2 * m_tmax, 1.0, 0.0],
                [1.0, 1.0, 1.0, 1.0],
            ]
        )
        b = np.array([tau0, tau_max, dtau_max, tau1])
        self._coeff = np.linalg.solve(A, b)

    def evaluate(self, m):
        """Evaluate thickness distribution at meridional locations.

        Parameters
        ----------
        m: (N,) array
            Normalised meridional distance to evaluate thickness at.
        Returns
        -------
        tau: (N,) array
            Thickness at the requested points.
        """
        return np.polyval(self._coeff, m) * np.sqrt(m) * (1.0 - m) + m * self.t_TE / 2.0


@dataclasses.dataclass(frozen=True)
class Annulus:
    """Axisymmetric turbomachine annulus defined by aspect ratio targets.

    Construct with `Annulus(r_mid, span, Beta, AR_chord, AR_gap)`. The primary
    method is `evaluate(m, spf)` which returns x/r coordinates as a function of
    normalised meridional distance and span fraction.

    Control points are placed analytically from AR targets and connected by
    PCHIP curves fit in true arc-length space.
    """

    r_mid: np.ndarray
    """(nrow*2,) mid-span radii at row inlet/exit stations [m]."""

    span: np.ndarray
    """(nrow*2,) annulus span perpendicular to pitch angle [m]."""

    Beta: np.ndarray
    """(nrow*2,) pitch angles at all stations [deg]."""

    AR_chord: np.ndarray
    """(nrow,) span-to-meridional-chord aspect ratio for each blade row."""

    AR_gap: np.ndarray
    """(nrow+1,) span-to-meridional-chord AR for inlet, inter-row gaps, and exit."""

    nozzle_ratio: float = 1.0
    """Exit area ratio (default 1.0 = no contraction)."""

    merge_weight: float = 0.0
    """Blend toward reduced-knot PCHIP that smooths curvature across blade rows.
    0 = curvature confined to gaps (duplicate row-LE/TE knots);
    1 = each row-LE/TE pair collapses to a single knot, curvature spreads freely."""

    @property
    def nrow(self) -> int:
        """Number of blade rows."""
        return len(self.r_mid) // 2

    @property
    def nseg(self) -> int:
        """Number of segments (gaps + rows): 2*nrow + 1."""
        return 2 * self.nrow + 1

    def __post_init__(self):
        nrow = self.nrow
        nseg = self.nseg
        util.check_vector((nrow * 2,), r_mid=self.r_mid, span=self.span, Beta=self.Beta)
        util.check_vector((nrow,), AR_chord=self.AR_chord)
        util.check_vector((nrow + 1,), AR_gap=self.AR_gap)
        util.check_scalar(
            nozzle_ratio=self.nozzle_ratio, merge_weight=self.merge_weight
        )
        if not 0.0 <= self.merge_weight <= 1.0:
            raise ValueError(f"merge_weight={self.merge_weight} must lie in [0, 1].")

        # Interleave AR across all segments (even=gaps, odd=rows)
        AR = np.empty(nseg)
        AR[::2] = self.AR_gap
        AR[1::2] = self.AR_chord

        # Segment-averaged span and pitch angle
        span_avg = _seg_avg(self.span)
        Beta_avg = _seg_avg(self.Beta)
        cosBeta_avg = np.cos(np.radians(Beta_avg))

        # Meridional chord and axial length per segment
        Ds = span_avg / AR
        Dx = Ds * cosBeta_avg

        # Integrate to get mid-span x-coordinates of control points
        xmid = util.cumsum0(Dx)
        xmid -= xmid[1]  # origin at first row LE

        # Control-point arrays (nseg+1,): edge-pad station arrays
        span_ext = np.pad(self.span, 1, "edge")
        Beta_ext = np.pad(self.Beta, 1, "edge")
        rmid_ext = np.pad(self.r_mid, 1, "edge")

        # Overwrite inlet/exit with duct extensions
        sinBeta0, sinBeta1 = np.sin(np.radians(self.Beta[[0, -1]]))
        rmid_ext[0] = rmid_ext[1] - Ds[0] * sinBeta0
        rmid_ext[-1] = rmid_ext[-2] + Ds[-1] * sinBeta1

        # Scale exit span for nozzle area ratio
        radius_ratio = rmid_ext[-2] / rmid_ext[-1]
        span_ext[-1] *= self.nozzle_ratio * radius_ratio

        # Hub and casing control point coordinates
        sinB = np.sin(np.radians(Beta_ext))
        cosB = np.cos(np.radians(Beta_ext))
        xhub = xmid + 0.5 * span_ext * sinB
        rhub = rmid_ext - 0.5 * span_ext * cosB
        xcas = xmid - 0.5 * span_ext * sinB
        rcas = rmid_ext + 0.5 * span_ext * cosB

        # Full PCHIP through all nseg+1 control points.
        s_full, pchips_full = self._fit_pchips(
            util.cumsum0(Ds), xhub, rhub, xcas, rcas, Ds
        )

        # Reduced PCHIP: four knots at [inlet, first row LE, last row TE, outlet].
        # Curvature is free to spread continuously across the rows because there
        # are no equal-radius interior knots pinning slopes to zero.
        if self.merge_weight > 0.0:
            idx_r = np.array([0, 1, nseg - 1, nseg])
            xhub_r = xhub[idx_r]
            rhub_r = rhub[idx_r]
            xcas_r = xcas[idx_r]
            rcas_r = rcas[idx_r]
            s_full_init = util.cumsum0(Ds)
            s_init_r = s_full_init[idx_r]
            Ds_r = np.diff(s_init_r)

            _, pchips_reduced = self._fit_pchips(
                s_init_r, xhub_r, rhub_r, xcas_r, rcas_r, Ds_r
            )
        else:
            pchips_reduced = pchips_full

        # Assign to frozen fields
        object.__setattr__(self, "_s", s_full)
        object.__setattr__(self, "_pchip_xhub", pchips_full[0])
        object.__setattr__(self, "_pchip_rhub", pchips_full[1])
        object.__setattr__(self, "_pchip_xcas", pchips_full[2])
        object.__setattr__(self, "_pchip_rcas", pchips_full[3])
        object.__setattr__(self, "_pchip_xhub_r", pchips_reduced[0])
        object.__setattr__(self, "_pchip_rhub_r", pchips_reduced[1])
        object.__setattr__(self, "_pchip_xcas_r", pchips_reduced[2])
        object.__setattr__(self, "_pchip_rcas_r", pchips_reduced[3])

    @staticmethod
    def _fit_pchips(s_init, xhub, rhub, xcas, rcas, Ds_target):
        """Arc-length-iterate PCHIPs through the given control points.

        Returns (s, (pchip_xhub, pchip_rhub, pchip_xcas, pchip_rcas)).
        Raises RuntimeError if iteration fails to converge.
        """
        s = np.asarray(s_init, dtype=float).copy()
        nseg = len(s) - 1
        err = np.inf
        pchip_xhub = pchip_rhub = pchip_xcas = pchip_rcas = None
        for _ in range(20):
            pchip_xhub = PchipInterpolator(s, xhub)
            pchip_rhub = PchipInterpolator(s, rhub)
            pchip_xcas = PchipInterpolator(s, xcas)
            pchip_rcas = PchipInterpolator(s, rcas)

            Ds_actual = np.empty(nseg)
            for k in range(nseg):
                sq = np.linspace(s[k], s[k + 1], 50)
                xm = 0.5 * (pchip_xhub(sq) + pchip_xcas(sq))
                rm = 0.5 * (pchip_rhub(sq) + pchip_rcas(sq))
                Ds_actual[k] = util.arc_length(np.stack([xm, rm]))

            Ds_norm = Ds_actual / Ds_actual.sum() * np.asarray(Ds_target).sum()
            s_new = util.cumsum0(Ds_norm)
            err = np.max(np.abs(s_new - s)) / s[-1]
            s = s_new
            if err < 1e-6:
                break

        if err >= 1e-6:
            raise RuntimeError(
                f"Annulus arc-length iteration failed to converge: err={err:.2e}"
            )

        return s, (pchip_xhub, pchip_rhub, pchip_xcas, pchip_rcas)

    def evaluate(self, m, spf) -> np.ndarray:
        """Evaluate annulus coordinates.

        Parameters
        ----------
        m : array_like
            Normalised meridional coordinate: 0 = inlet, 1 = first row LE,
            2 = first row TE, ..., nseg = exit.
        spf : array_like
            Span fraction, broadcastable with m. 0 = hub, 1 = casing.

        Returns
        -------
        xr : np.ndarray, shape (2, ...)
            x and r coordinates at the queried points.
        """
        mb, spfb = np.broadcast_arrays(
            np.asarray(m, dtype=float), np.asarray(spf, dtype=float)
        )
        sq = np.interp(mb, np.arange(len(self._s)), self._s)

        w = self.merge_weight
        if w == 0.0:
            xhub = self._pchip_xhub(sq)
            rhub = self._pchip_rhub(sq)
            xcas = self._pchip_xcas(sq)
            rcas = self._pchip_rcas(sq)
        elif w == 1.0:
            xhub = self._pchip_xhub_r(sq)
            rhub = self._pchip_rhub_r(sq)
            xcas = self._pchip_xcas_r(sq)
            rcas = self._pchip_rcas_r(sq)
        else:
            xhub = (1.0 - w) * self._pchip_xhub(sq) + w * self._pchip_xhub_r(sq)
            rhub = (1.0 - w) * self._pchip_rhub(sq) + w * self._pchip_rhub_r(sq)
            xcas = (1.0 - w) * self._pchip_xcas(sq) + w * self._pchip_xcas_r(sq)
            rcas = (1.0 - w) * self._pchip_rcas(sq) + w * self._pchip_rcas_r(sq)

        x = (1.0 - spfb) * xhub + spfb * xcas
        r = (1.0 - spfb) * rhub + spfb * rcas
        return np.stack([x, r])
