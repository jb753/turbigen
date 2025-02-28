"""Classes to parameterise camber lines.

The purpose of these objects is to evaluate camber lines in streamsurface
coordinates a function of chordwise position.

To make a new camber line, subclass the BaseCamber and implement:
    - chi_hat(m)

"""

import numpy as np
from abc import ABC, abstractmethod


def _fit_quartic_camber(slope_le, slope_te, kappa):
    """Get the polynomial coefficients for a quartic camber line.

    We set the slope at the leading edge, trailing edge, and curvature at the
    mid-chord position.

    """

    A = np.zeros((5, 5))
    b = np.zeros((5, 1))

    # y-intercept at 1
    A[0] = [0.0, 0.0, 0.0, 0.0, 1.0]
    b[0] = 0.0

    # x-intercept at 0
    A[1] = [1.0, 1.0, 1.0, 1.0, 1.0]
    b[1] = 1.0

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
    return np.linalg.solve(A, b).reshape(-1)


class BaseCamber(ABC):
    """Define the interface for a camber line."""

    def __init__(self, q_camber):
        """Initialise camber line with parameter vector.

        Parameters
        ----------
        q_camber: array
          Parameter vector for camber line.
          q_camber[0] = tan chi_LE
          q_camber[1] = tan chi_TE
          q_camber[2:] = choice of camber line parameters

        """
        self.q_camber = np.reshape(q_camber, -1)

    def __hash__(self):
        return hash(tuple(self.q_camber))

    @property
    def tanchi_LE(self):
        """Camber angle tangent at leading edge."""
        return self.q_camber[0]

    @property
    def tanchi_TE(self):
        """Camber angle tangent at trailing edge."""
        return self.q_camber[1]

    @property
    def chi_LE(self):
        """Camber angle at leading edge."""
        return np.degrees(self.tanchi_LE)

    @property
    def chi_TE(self):
        """Camber angle at trailing edge."""
        return np.degrees(self.tanchi_TE)

    @property
    def Dchi(self):
        """Camber angle change."""
        return self.chi_TE - self.chi_LE

    @property
    def Dtanchi(self):
        """Camber angle tangend change."""
        return self.tanchi_TE - self.tanchi_LE

    @abstractmethod
    def chi_hat(self, m) -> np.ndarray:
        """Normalised camber as function of meridional distance.

        Parameters
        ----------
        m: (n,) array
            Meridional chord fractions to evaluate camber line at.

        Returns
        -------
        chi_hat: (n,) array
            Normalised camber at requested meridional positions.
            chi_hat(m=0) = 0, chi_hat(m=1) = 1.

        """
        raise NotImplementedError

    def chi(self, m):
        r"""Camber angle as function of normalised meridional distance.

        Note that camber angle is the arctangent of camber line slope.

        Parameters
        ----------
        m: (n,) array
            Meridional chord fractions to evaluate camber line at.

        Returns
        -------
        chi: (n,) array
            Camber angle at requested meridional positions [deg].
        """
        return np.degrees(np.arctan(self.tanchi_LE + self.chi_hat(m) * self.Dtanchi))

    def dydm(self, m):
        """Camber line slope as function of normalised meridional distance.

        Note that camber line slope is the tangent of camber angle.

        Parameters
        ----------
        m: (n,) array
            Meridional chord fractions to evaluate camber line at.

        Returns
        -------
        dydm: (n,) array
            Camber line slope at requested meridional positions.
        """
        return np.tan(np.radians(self.chi(m)))


class Quartic(BaseCamber):
    """Use a quartic polynomial to set normalised camber."""

    def chi_hat(self, m):
        return np.polyval(_fit_quartic_camber(*self.q_camber[2:]), m)


class Taylor(BaseCamber):
    """Use a quartic polynomial to set camber angle."""

    def chi_hat(self, m):
        chi = np.polyval(_fit_quartic_camber(*self.q_camber[2:]), m)
        tanchi = np.tan(np.radians(chi))
        return (tanchi - self.tanchi_LE) / self.Dtanchi


class Quadratic(BaseCamber):
    """Use a quadratic polynomial to set camber angle."""

    def chi_hat(self, m):
        a = self.q_camber[2]  # Aft-loading factor
        m = np.array(m)
        return m * (a * m + (1 - a))


def load_camber(camber_type):
    """Get camber class by string, including any custom classes."""
    available_types = {a.__name__: a for a in BaseCamber.__subclasses__()}
    if camber_type not in available_types:
        raise ValueError(
            f"Unknown camber type: {camber_type}, should be one of {available_types.keys()}"
        )
    else:
        return available_types[camber_type]
