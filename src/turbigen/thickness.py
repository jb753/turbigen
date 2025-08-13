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
    def scale(self, fac):
        """Scale the thickness distribution by a factor.

        This method should modify things like the LE radius, maximum thickness,
        and TE thickness, but not the position of the maximum thickness or
        wedge angle.
        """
        raise NotImplementedError

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

    def scale(self, fac):
        self.q_thick[0] *= fac  # Scale LE radius
        self.q_thick[1] *= fac  # Scale maximum thickness
        self.q_thick[4] *= fac  # Scale TE thickness

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
        t = self._from_shape(m, self.tau(m))
        assert t.max() <= self.t_max, "Thickness exceeds maximum thickness."
        return t
