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

    def _validate_domain(self, m):
        """Validate that m is within the valid domain [0, 1]."""
        m_array = np.asarray(m)
        if np.any(m_array < 0) or np.any(m_array > 1):
            raise ValueError("Meridional distance m must be in the range [0, 1]")

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

    @property
    @abstractmethod
    def t_max(self):
        """Maximum thickness of the airfoil."""
        raise NotImplementedError

    @property
    @abstractmethod
    def R_LE(self):
        """Leading edge radius of the airfoil."""
        raise NotImplementedError

    @property
    @abstractmethod
    def t_te(self):
        """Trailing edge thickness of the airfoil.

        Note that this is the total thickness that would arise from
        both sides, so self.thick(m) will return half of this value.

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
    def t_te(self):
        return self.q_thick[4]

    def scale(self, fac):
        self.q_thick[0] *= fac  # Scale LE radius
        self.q_thick[1] *= fac  # Scale maximum thickness
        self.q_thick[4] *= fac  # Scale TE thickness

    def _to_shape(self, x, t, eps=1e-5):
        """Transform real thickness to shape space."""
        # Ignore singularities at leading and trailing edges
        ii = np.abs(x - 0.5) < (0.5 - eps)
        s = np.ones(x.shape) * np.nan
        s[ii] = (t[ii] - x[ii] * self.t_te / 2.0) / np.sqrt(x[ii]) / (1.0 - x[ii])
        return s

    def _from_shape(self, x, s):
        """Transform shape space to real coordinates."""
        return np.sqrt(x) * (1.0 - x) * s + x * self.t_te / 2.0

    @property
    def _coeff(self):
        """Coefficients for piecewise polynomials in shape space."""

        m_tmax = self.q_thick[2]
        # Evaluate control points
        sle = np.sqrt(2.0 * self.R_LE)
        t_te = self.t_te
        smax = (self.t_max - m_tmax * t_te / 2.0) / np.sqrt(m_tmax) / (1.0 - m_tmax)
        dsmax = (
            (
                smax * (np.sqrt(m_tmax) - (1.0 - m_tmax) / 2.0 / np.sqrt(m_tmax))
                - t_te / 2.0
            )
            / np.sqrt(m_tmax)
            / (1.0 - m_tmax)
        )

        tanwedge = self.q_thick[5]
        ste = t_te + tanwedge

        # For brevity
        x3 = m_tmax**3.0
        x2 = m_tmax**2.0
        x1 = m_tmax

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
        kappa_max = self.q_thick[3]
        b[3] = kappa_max

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
        m_tmax = self.q_thick[2]
        tau[s <= m_tmax] = np.polyval(coeff_front, s[s <= m_tmax])
        tau[s > m_tmax] = np.polyval(coeff_rear, s[s > m_tmax])
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
        # Validate domain
        self._validate_domain(m)

        # Convert to array to ensure consistent behavior
        m_array = np.asarray(m)
        t = self._from_shape(m_array, self.tau(m_array))

        # Check maximum thickness constraint
        if np.any(t > self.t_max + 1e-10):  # Small tolerance for numerical errors
            raise ValueError("Thickness exceeds maximum thickness.")

        # Return scalar for scalar input, array otherwise
        if np.isscalar(m):
            return float(t.item())
        else:
            return t
