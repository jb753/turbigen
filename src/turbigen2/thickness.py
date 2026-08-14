"""Thickness distributions.

A :class:`ThicknessDesign` is a config node that evaluates thickness as a
function of normalised meridional distance. Thickness is always normalised by
the meridional chord, so unlike the package this replaces there is no reference
length to choose and no ``scale`` method: a design is evaluable as soon as it
exists, rather than only after a stream surface has been pushed onto it.

The parameters are named fields rather than positions in a vector. What used to
be ``q_thick[3]`` is :attr:`Taylor.kappa_max`.
"""

import logging
from typing import ClassVar

import numpy as np

from turbigen2.node import Node

logger = logging.getLogger("turbigen")


def _validate_domain(m):
    """Check that a normalised meridional coordinate lies in [0, 1]."""
    if np.any(np.asarray(m) < 0.0) or np.any(np.asarray(m) > 1.0):
        raise ValueError("Meridional distance m must be in the range [0, 1].")


class ThicknessDesign(Node):
    """Base for thickness distributions, normalised by meridional chord."""

    def thick(self, m):
        """Return half-thickness at normalised meridional distance `m`."""
        raise NotImplementedError(f"{type(self).__name__} must implement thick(m)")


class Taylor(ThicknessDesign):
    """After Taylor (2016), two cubic splines in shape space.

    The splines meet at the point of maximum thickness, where the value, slope
    and curvature are all continuous.
    """

    type: ClassVar[str] = "taylor"

    R_LE: float
    """Leading edge radius, normalised by meridional chord [--]."""

    t_max: float
    """Maximum thickness, normalised by meridional chord [--]."""

    m_tmax: float
    """Normalised meridional position of maximum thickness [--]."""

    kappa_max: float = 0.0
    """Curvature in shape space at maximum thickness [--]."""

    t_TE: float = 0.0
    """Trailing edge thickness, the total due to both sides [--]."""

    tanwedge: float = 0.0
    """Tangent of the trailing edge wedge angle [--]."""

    @property
    def _coeff(self):
        """Coefficients of the front and rear cubics in shape space."""
        m_tmax = self.m_tmax
        t_TE = self.t_TE

        # Control points in shape space
        s_LE = np.sqrt(2.0 * self.R_LE)
        s_max = (self.t_max - m_tmax * t_TE / 2.0) / np.sqrt(m_tmax) / (1.0 - m_tmax)
        ds_max = (
            (
                s_max * (np.sqrt(m_tmax) - (1.0 - m_tmax) / 2.0 / np.sqrt(m_tmax))
                - t_TE / 2.0
            )
            / np.sqrt(m_tmax)
            / (1.0 - m_tmax)
        )
        s_TE = t_TE + self.tanwedge

        x1 = m_tmax
        x2 = m_tmax**2.0
        x3 = m_tmax**3.0

        # The front cubic is set by the leading edge radius, and by the value,
        # slope and curvature at maximum thickness
        A = np.array(
            [
                [0.0, 0.0, 0.0, 1.0],
                [x3, x2, x1, 1.0],
                [3.0 * x2, 2.0 * x1, 1.0, 0.0],
                [6.0 * x1, 2.0, 0.0, 0.0],
            ]
        )
        b = np.array([s_LE, s_max, ds_max, self.kappa_max])
        coeff_front = np.linalg.solve(A, b.reshape(-1, 1)).reshape(-1)

        # The rear cubic shares the maximum thickness conditions, with the
        # leading edge radius replaced by the trailing edge thickness and wedge
        A[0] = [1.0, 1.0, 1.0, 1.0]
        b[0] = s_TE
        coeff_rear = np.linalg.solve(A, b.reshape(-1, 1)).reshape(-1)

        return coeff_front, coeff_rear

    def tau(self, m):
        """Return thickness in shape space at normalised meridional distance `m`."""
        m = np.asarray(m, dtype=float)
        coeff_front, coeff_rear = self._coeff
        front = m <= self.m_tmax
        tau = np.zeros_like(m)
        tau[front] = np.polyval(coeff_front, m[front])
        tau[~front] = np.polyval(coeff_rear, m[~front])
        return tau

    def thick(self, m):
        """Return half-thickness at normalised meridional distance `m`.

        The trailing edge thickness is specified as the total due to both
        sides, so half of it is returned at the trailing edge.
        """
        _validate_domain(m)

        m_array = np.asarray(m, dtype=float)
        t = np.sqrt(m_array) * (1.0 - m_array) * self.tau(m_array) + (
            m_array * self.t_TE / 2.0
        )

        if np.any(t > self.t_max + 1e-10):
            raise ValueError(
                f"Thickness exceeds the maximum thickness t_max={self.t_max}."
            )

        return float(t.item()) if np.isscalar(m) else t
