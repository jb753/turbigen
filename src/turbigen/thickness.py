"""Classes to represent thickness distributions.

Each class needs to accept a 1D vector of parameters and have a `t(m)` method
that returns thickness as a function of meridional distance.

All lengths are normalised by the meridional chord.

The trailing edge thickness is input as the total thickness, with half
contributed by each side.

"""

import numpy as np

from abc import ABC, abstractmethod


class BaseThickness(ABC):
    """Define the interface for a thickness distribution."""

    def __init__(self, q_thick):
        """Initialise thickness distribution with parameter vector.

        Parameters
        ----------
        q_thick: array
            Parameter vector for thickness distribution.

        """
        self.q_thick = np.reshape(q_thick, -1)

    @abstractmethod
    def thick(self, m):
        """Evaluate thickness distribution at meridional locations.

        Parameters
        ----------
        m: (N,) array
            Normalised meridional distance to evaluate thickness at.

        Returns
        -------
        t: (N,) array
            Thickness at the requested points.

        """
        raise NotImplementedError


class Taylor(BaseThickness):
    """After Taylor (2016), two cubic splines in shape space."""

    @property
    def R_LE(self):
        return self.q_thick[0]

    @property
    def t_max(self):
        return self.q_thick[1]

    @property
    def s_tmax(self):
        return self.q_thick[2]

    @property
    def kappa_max(self):
        return self.q_thick[3]

    @property
    def t_te(self):
        return self.q_thick[4]

    @property
    def tanwedge(self):
        return self.q_thick[5]

    def _to_shape(self, x, t, eps=1e-5):
        """Transform real thickness to shape space."""
        # Ignore singularities at leading and trailing edges
        ii = np.abs(x - 0.5) < (0.5 - eps)
        s = np.ones(x.shape) * np.nan
        if self.t_te < 0.0:
            s[ii] = t[ii] / np.sqrt(x[ii]) / np.sqrt(1.0 - x[ii])
        else:
            s[ii] = (t[ii] - x[ii] * self.t_te / 2.0) / np.sqrt(x[ii]) / (1.0 - x[ii])
        return s

    def _from_shape(self, x, s):
        """Transform shape space to real coordinates."""
        if self.t_te < 0.0:
            return np.sqrt(x) * np.sqrt(1.0 - x) * s
        else:
            return np.sqrt(x) * (1.0 - x) * s + x * self.t_te / 2.0

    @property
    def _coeff(self):
        """Coefficients for piecewise polynomials in shape space."""

        # Evaluate control points
        sle = np.sqrt(2.0 * self.R_LE)
        t_te = self.t_te
        if t_te < 0.0:
            t_te = 0.0
            smax = self.t_max / np.sqrt(self.s_tmax) / np.sqrt(1.0 - self.s_tmax)
            dsmax = (
                smax
                / 2.0
                * (2.0 * self.s_tmax - 1.0)
                / self.s_tmax
                / (1.0 - self.s_tmax)
            )
        else:
            smax = (
                (self.t_max - self.s_tmax * t_te / 2.0)
                / np.sqrt(self.s_tmax)
                / (1.0 - self.s_tmax)
            )
            dsmax = (
                (
                    smax
                    * (
                        np.sqrt(self.s_tmax)
                        - (1.0 - self.s_tmax) / 2.0 / np.sqrt(self.s_tmax)
                    )
                    - t_te / 2.0
                )
                / np.sqrt(self.s_tmax)
                / (1.0 - self.s_tmax)
            )

        ste = t_te + self.tanwedge

        # For brevity
        x3 = self.s_tmax**3.0
        x2 = self.s_tmax**2.0
        x1 = self.s_tmax

        # Fit front cubic
        A = np.zeros((4, 4))
        b = np.zeros((4, 1))

        # LE radius
        A[0] = [0.0, 0.0, 0.0, 1.0]
        b[0] = sle

        # Value of max thickness
        A[1] = [x3, x2, x1, 1.0]
        b[1] = smax

        # Slope at max thickness
        A[2] = [3.0 * x2, 2.0 * x1, 1.0, 0.0]
        b[2] = dsmax

        # Curvature at max thickness
        A[3] = [6.0 * x1, 2.0, 0.0, 0.0]
        b[3] = self.kappa_max

        coeff_front = np.linalg.solve(A, b).reshape(-1)

        # Fit rear cubic
        # TE thick/wedge (other points are the same)
        A[0] = [1.0, 1.0, 1.0, 1.0]
        b[0] = ste

        coeff_rear = np.linalg.solve(A, b).reshape(-1)

        coeff = np.stack((coeff_front, coeff_rear))

        return coeff

    def tau(self, s):
        r"""Thickness in shape space as function of normalised meridional distance.

        Parameters
        ----------
        s: array
            Fractions of normalised meridional distance to evaluate at.

        Returns
        -------
        t: array
            Samples of thickness distribution at the requested points.
        """

        s = np.array(s)

        coeff_front, coeff_rear = self._coeff
        tau = np.zeros_like(s)
        tau[s <= self.s_tmax] = np.polyval(coeff_front, s[s <= self.s_tmax])
        tau[s > self.s_tmax] = np.polyval(coeff_rear, s[s > self.s_tmax])
        return tau

    def thick(self, m):
        r"""Thickness as function of normalised meridional distance.

        Parameters
        ----------
        m: (N) array
            Fractions of normalised meridional distance to evaluate at.

        Returns
        -------
        t: (N) array
            Samples of thickness distribution at the requested points :math:`t(m)`.

        """
        return self._from_shape(m, self.tau(m))


# class Impeller:
#     """For radial impellers."""
#
#     def __init__(self, q_thick):
#         r"""A constant thickness with curvature-continuous elliptical leading edge
#
#         Parameters
#         ----------
#         q_thick: (1,) array
#             Geometry parameter vector with elements:
#                 * Maximum thickness, :math:`t_\max\,`;
#                 * Ellipse ratio, :math:`a\,`;
#                 * Meridional location of maximum thickness, :math:`m_\max\,`.
#                 * Trailing edge thickness, :math:`t_\TE\,`;
#
#         """
#
#         # Record inputs
#         self.q_thick = np.reshape(q_thick, 4)
#
#         # Make a circular LE curvature-continuous by adding a blending polynomial
#         # t = sqrt(x(2R-x)) + x^3/2/R^2 - x^2/R + x/2
#         # Then scale x-coord to make ellipse
#
#         # Calculate coefficients for blending polynomial
#         # Scaling factor on x coord
#         # 1.25 empirically chosen to fit a real curvature-discontinous circle
#         # Then divide by ellipse ratio
#         fm = 1.0 / 1.25 / self.a
#         R = self.tmax
#         A = 0.5 / R**2.0 * fm**3.0
#         B = -1.0 / R * fm**2.0
#         C = 0.5 * fm
#         self._coeffs = np.array([A, B, C])
#         self._fm = fm
#
#         # Trailing edge blend
#
#         b = np.zeros((4, 1))
#         A = np.zeros((4, 4))
#
#         # t(1) = tTE
#         A[0] = [1.0, 1.0, 1.0, 1.0]
#         b[0] = self.tTE / 2.0
#
#         # t(mm) = tmax
#         mm = self.mmax
#         A[1] = [mm**3, mm**2, mm, 1.0]
#         b[1] = self.tmax
#
#         # t'(mm) = 0
#         A[2] = [3.0 * mm**2.0, 2.0 * mm, 1.0, 0.0]
#         b[2] = 0.0
#
#         # t''(mm) = 0
#         A[3] = [6.0 * mm, 2.0, 0.0, 0.0]
#         b[3] = 0.0
#
#         self._coeffTE = np.linalg.solve(A, b).reshape(-1)
#
#     @property
#     def a(self):
#         return self.q_thick[1]
#
#     @property
#     def tmax(self):
#         return self.q_thick[0]
#
#     @property
#     def tTE(self):
#         return self.q_thick[3]
#
#     @property
#     def R_LE(self):
#         return self.tmax / self.a * 5.0 / 2.0
#
#     @property
#     def mmax(self):
#         return self.q_thick[2]
#
#     def t(self, m):
#         r"""Thickness as function of normalised meridional distance.
#
#         Parameters
#         ----------
#         m: (N) array
#             Fractions of normalised meridional distance to evaluate at.
#
#         Returns
#         -------
#         t: (N) array
#             Samples of thickness distribution at the requested points :math:`t(m)`.
#
#         """
#
#         t = np.ones_like(m) * self.tmax
#         fm = self._fm
#         R = self.tmax
#         iLE = m <= (R / fm)
#         iTE = m > (self.mmax)
#         A, B, C = self._coeffs
#
#         t[iLE] = (
#             np.sqrt(m[iLE] * fm * (2.0 * R - m[iLE] * fm))
#             + A * m[iLE] ** 3
#             + B * m[iLE] ** 2
#             + C * m[iLE]
#         )
#         t[iTE] = np.polyval(self._coeffTE, m[iTE])
#
#         return t
#
#
# class CircularLE:
#     def __init__(self, q_thick):
#         r"""
#
#         Parameters
#         ----------
#         q_thick: (6,) array
#             Geometry parameter vector with elements:
#                 * Leading-edge radius, :math:`R_\LE\,`;
#                 * Maximum thickness, :math:`t_\max\,`;
#                 * Location of maximum thickness, :math:`m_\max\,`;
#                 * Curvature at maximum thickness, :math:`\kappa_\max\,`;
#                 * Trailing edge thickness, :math:`2t_\TE\,`;
#                 * Leading edge wedge angle tangent, :math:`\tan\zeta\,`.
#                 * Trailing edge wedge angle tangent, :math:`\tan\zeta\,`.
#                 * Ellipse ratio
#
#         """
#
#         # Record inputs
#         self.q_thick = np.reshape(q_thick, 8)
#
#         # Cache for polynomial coefficients
#         self._coeff_cache = {}
#         self._use_cache = True
#
#     eps = 1e-3
#     qbound = (
#         (eps, 0.05),
#         (eps, 0.5),
#         (0.05, 0.95),
#         (-10.0, 10.0),
#         (0.005, 0.2),
#         (eps, 1.0),
#         (eps, 2.0),
#         (0.2, 10.0),
#     )
#
#     @property
#     def R_LE(self):
#         return self.q_thick[0]
#
#     @property
#     def t_max(self):
#         return self.q_thick[1]
#
#     @property
#     def m_max(self):
#         return self.q_thick[2]
#
#     @property
#     def kappa(self):
#         return self.q_thick[3]
#
#     @property
#     def t_TE(self):
#         return self.q_thick[4]
#
#     @property
#     def wedge_LE(self):
#         return self.q_thick[5]
#
#     @property
#     def wedge_TE(self):
#         return self.q_thick[6]
#
#     @property
#     def a(self):
#         return self.q_thick[7]
#
#     @property
#     def xb(self):
#         # R = self.R_LE
#         # Rsq = R**2
#         # ysq = self.wedge_LE**2
#         # return (-np.sqrt(Rsq*ysq*(ysq+1.)) + R*(ysq + 1.))/(ysq+1.)
#         dy = self.wedge_LE
#         R = self.R_LE
#         a = self.a
#         k = np.sqrt(1.0 + (a * dy) ** 2)
#         return a * R * (1.0 - a * dy / k)
#
#     @property
#     def tb(self):
#         R = self.R_LE
#         Rsq = R**2
#         return np.sqrt(Rsq - (self.xb / self.a - R) ** 2)
#
#     def __hash__(self):
#         """Hash based on tuple of the geometry parameters."""
#         return hash(tuple(self.q_thick))
#
#     @property
#     def _coeff(self):
#         """Coefficients for piecewise polynomials in shape space."""
#
#         # Skip if we have already fitted polynomials for current params
#         if self._use_cache and (hash(self) in self._coeff_cache):
#             return self._coeff_cache[hash(self)]
#
#         # Rear coefficients are easiest
#         # Curvature continuous from max thickness down to trailing edge
#         # Needs a quartic because we specify five boundary conditions
#
#         # For brevity
#         # Powers of meridional location of max thickness
#         x4 = self.m_max**4
#         x3 = self.m_max**3
#         x2 = self.m_max**2
#         x1 = self.m_max
#
#         # Fit the quartic
#         A = np.zeros((5, 5))
#         b = np.zeros((5, 1))
#
#         # Value of max thickness
#         # f(xm) = tm
#         A[0] = [x4, x3, x2, x1, 1.0]
#         b[0] = self.t_max
#
#         # Slope at max thickness
#         # A stationary point
#         # f'(xm) = 0
#         A[1] = [4.0 * x3, 3.0 * x2, 2.0 * x1, 1.0, 0.0]
#         b[1] = 0.0
#
#         # Curvature at max thickness
#         # f''(xm) = kappa
#         A[2] = [12.0 * x2, 6.0 * x1, 2.0, 0.0, 0.0]
#         b[2] = self.kappa
#
#         # Value at trailing edge
#         # f(1) = tte
#         A[3] = [1.0, 1.0, 1.0, 1.0, 1.0]
#         b[3] = self.t_TE
#
#         # Slope at trailing edge
#         # f(1) = -tan wedge
#         A[4] = [4.0, 3.0, 2.0, 1.0, 0.0]
#         b[4] = -self.wedge_TE
#
#         coeff_rear = np.linalg.solve(A, b).reshape(-1)
#
#         # Determine the slope and value at LE blend point
#         # from the wedge angle
#         xb = self.xb
#         dtb = self.wedge_LE
#         tb = self.tb
#
#         xb4 = xb**4
#         xb3 = xb**3
#         xb2 = xb**2
#
#         # Fit front cubic, overwrite TE bconds
#         # (other bcond at max thickness are the same)
#         # Value at blend f(xb) = tb
#         A[3] = [xb4, xb3, xb2, xb, 1.0]
#         b[3] = tb
#
#         # Slope at blend f'(xb) = wedge_TE
#         A[4] = [4.0 * xb3, 3.0 * xb2, 2.0 * xb, 1.0, 0.0]
#         b[4] = dtb
#
#         coeff_front = np.linalg.solve(A, b).reshape(-1)
#
#         # Store
#         coeff = np.stack((coeff_front, coeff_rear))
#         if self._use_cache:
#             self._coeff_cache[hash(self)] = coeff
#
#         return coeff
#
#     def t(self, m):
#         r"""Thickness as function of normalised meridional distance.
#
#         Parameters
#         ----------
#         m: (N,) array
#             Fractions of normalised meridional distance to evaluate at.
#
#         Returns
#         -------
#         t: (N,) array
#             Samples of thickness distribution at the requested points :math:`t(m)`.
#
#         """
#         coeff_front, coeff_rear = self._coeff
#
#         t = np.full_like(m, np.nan)
#
#         # Circular LE
#         R = self.R_LE
#         Rsq = R**2
#         ind_circ = m < self.xb
#         t[ind_circ] = np.sqrt(Rsq - (m[ind_circ] / self.a - R) ** 2)
#
#         # Quartics for front and rear
#         coeff_front, coeff_rear = self._coeff
#         ind_front = np.logical_and(m >= self.xb, m <= self.m_max)
#         t[ind_front] = np.polyval(coeff_front, m[ind_front])
#         ind_rear = m > self.m_max
#         t[ind_rear] = np.polyval(coeff_rear, m[ind_rear])
#
#         return t
