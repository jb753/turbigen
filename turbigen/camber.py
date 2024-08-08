"""Classes to parameterise camber lines.

The purpose of these objects is to evaluate camber lines in streamsurface
coordinates a function of chordwise position.
"""

import numpy as np
from scipy.optimize import minimize


class Brind:
    """Use a quartic polynomial for camber angle tangent."""

    DEFAULT_PVEC = np.array([0.0, 0.0, 1.0, 1.0, 0.0])

    eps = 1e-3
    qbound = ((-5.0, 5.0), (-5.0, 5.0), (0.0, 3.0), (0.0, 3.0), (-0.2, 0.2))

    def __init__(self, q_camber):
        # Store input parameter vector
        self.q_camber = np.reshape(q_camber, 5)

        # Cache for polynomial coefficients
        self._coeff_cache = {}
        self._use_cache = True

    def __hash__(self):
        """Hash based on tuple of the geometry parameters."""
        return hash(tuple(self.q_camber))

    @classmethod
    def from_fit(cls, m, y):
        r"""Initialise by fitting camber line to some coordinates.

        Parameters
        ----------
        m: (N,) array
            Normalised meridional distance, :math:`m`.

        y: (N,) array
            Normalised tangential coordinate, :math:`y`.

        """

        # Put input origin at LE
        m -= m[0]
        y -= y[0]

        # Scale to unit meridional distance
        y /= np.ptp(m)
        m /= np.ptp(m)

        # Fit quadratics to robustly estimate end gradients
        delta = 0.15
        coeff_le = np.polyfit(m[m < delta], y[m < delta], 2)
        coeff_te = np.polyfit(m[m > 1.0 - delta], y[m > 1.0 - delta], 2)
        tanchi = np.array(
            [
                np.polyval(np.polyder(coeff_le), 0.0),
                np.polyval(np.polyder(coeff_te), 1.0),
            ]
        )

        # Initial guess for parameter vector
        p0 = cls.DEFAULT_PVEC + 0.0
        p0[:2] = tanchi

        def _iter(p):
            cam_now = cls(p)
            y_fit = cam_now.y(m)
            return np.sum((y_fit - y) ** 2.0)

        popt = minimize(_iter, p0).x
        return cls(popt)

    def chi(self, m):
        r"""Camber angle as function of normalised meridional distance.

        Note that camber angle is the arctangent of camber line slope.

        Parameters
        ----------
        m: array
            Meridional chord fractions to evaluate camber line at, :math:`m\,`.

        Returns
        -------
        chi: array
            Camber angle at requested meridional positions, :math:`\chi/^\circ\,`.
        """
        return np.degrees(np.arctan(self.dydm(m)))

    def dydm(self, m):
        r"""Camber line slope as function of normalised meridional distance.

        Note that camber line slope is the tangent of camber angle.

        Parameters
        ----------
        m: array
            Meridional chord fractions to evaluate camber line at, :math:`m\,`.

        Returns
        -------
        dydm: array
            Camber line slope at requested meridional positions, :math:`y'\,`.

        """
        return self.q_camber[0] + np.polyval(self._coeff, m) * self._Dtanchi

    def d2ydm2(self, m):
        r"""Camber line curvature as function of normalised meridional distance.

        Parameters
        ----------
        m: array
            Meridional chord fractions to evaluate camber line at, :math:`m\,`.

        Returns
        -------
        d2ydm2: array
            Camber line curvature at requested meridional positions, :math:`y''\,`.

        """
        return np.polyval(np.polyder(self._coeff), m) * self._Dtanchi

    def y(self, m):
        r"""Circumferential camber coordinate at meridional locations.

        The leading edge is assumed to be located at the origin.

        Parameters
        ----------
        m: array
            Meridional chord fractions to evaluate camber line at, :math:`m\,`.

        Returns
        -------
        y: array
            Tangential camber line coordinate at requested positions, :math:`y\,`.

        """
        return (
            self.q_camber[0] * m
            + np.polyval(np.polyint(self._coeff, 1), m) * self._Dtanchi
        )

    def chi_hat(self, m):
        """Non-dimensional camber line as function of normalised meridional distance."""
        return np.polyval(self._coeff, m)

    @property
    def _Dtanchi(self):
        r"""Camber angle tangent change."""
        return np.diff(self.q_camber[:2]).item()

    @property
    def _coeff(self):
        r"""Coefficients for quartic thickness distribution."""

        # Skip if we have already fitted polynomials for current params
        if self._use_cache and (hash(self) in self._coeff_cache):
            return self._coeff_cache[hash(self)]

        # TODO - could probably get an analytical solution instead of numerical
        # matrix inversion
        A = np.empty((5, 5))
        b = np.empty((5, 1))

        # y-intercept at 1
        A[0] = [0.0, 0.0, 0.0, 0.0, 1.0]
        b[0] = 0.0

        # x-intercept at 0
        A[1] = [1.0, 1.0, 1.0, 1.0, 1.0]
        b[1] = 1.0

        # Unpack parameters
        slope_le, slope_te, kappa = self.q_camber[2:]

        # Curvature parameter
        A[2] = [3.0, 3.0, 2.0, 0.0, 0.0]
        b[2] = kappa

        # LE gradient
        A[3] = [0.0, 0.0, 0.0, 1.0, 0.0]
        b[3] = slope_le

        # TE gradient
        A[4] = [4.0, 3.0, 2.0, 1.0, 0.0]
        b[4] = slope_te

        # Solve for polynomial coeffs
        coeff = np.linalg.solve(A, b).reshape(-1)

        # Store in cache if using
        if self._use_cache:
            self._coeff_cache[hash(self)] = coeff

        return coeff


class Taylor:
    """Use a quartic polynomial for camber angle."""

    DEFAULT_PVEC = np.array([0.0, 0.0, 1.0, 1.0, 0.0])

    def __init__(self, q_camber):
        # Store input parameter vector
        self.q_camber = np.reshape(q_camber, 5)

        # Fit the quartic coefficients
        self._coeff = self._fit_coeff()

    def chi(self, m):
        return self.chi_in + self.chi_hat(m) * self.Dchi

    def chi_hat(self, m):
        return np.polyval(self._coeff, m)

    def dydm(self, m):
        return np.tan(np.radians(self.chi(m)))

    @property
    def chi_in(self):
        return np.degrees(np.arctan(self.q_camber[0]))

    @property
    def chi_out(self):
        return np.degrees(np.arctan(self.q_camber[1]))

    @property
    def Dchi(self):
        r"""Camber angle tangent change."""
        return np.diff(np.degrees(np.arctan(self.q_camber[:2]))).item()

    def _fit_coeff(self):
        r"""Coefficients for quartic thickness distribution."""

        # TODO - could probably get an analytical solution instead of numerical
        # matrix inversion
        A = np.empty((5, 5))
        b = np.empty((5, 1))

        # y-intercept at 1
        A[0] = [0.0, 0.0, 0.0, 0.0, 1.0]
        b[0] = 0.0

        # x-intercept at 0
        A[1] = [1.0, 1.0, 1.0, 1.0, 1.0]
        b[1] = 1.0

        # Unpack parameters
        slope_le, slope_te, kappa = self.q_camber[2:]

        # Curvature parameter
        A[2] = [3.0, 3.0, 1.0, 0.0, 0.0]
        b[2] = kappa

        # LE gradient
        A[3] = [0.0, 0.0, 0.0, 1.0, 0.0]
        b[3] = slope_le

        # TE gradient
        A[4] = [4.0, 3.0, 2.0, 1.0, 0.0]
        b[4] = slope_te

        # Solve for polynomial coeffs
        coeff = np.linalg.solve(A, b).reshape(-1)

        return coeff
