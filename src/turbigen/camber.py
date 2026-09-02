"""
.. _camber:

Camber lines
^^^^^^^^^^^^

A :class:`CamberDesign` describes the *shape* of a camber line between its end
angles; pairing it with those angles gives a :class:`CamberLine`, which can be
evaluated. The end angles are not design variables but a result --- the local
flow angle plus the recamber asked for by a
:class:`~turbigen.blade.SectionDesign` --- so the split here is the same one made
everywhere else, one level down.

:class:`Quadratic` (``quadratic``) is the built-in shape: the camber line slope
varies quadratically along the chord, with :attr:`~Quadratic.aft_loading`
shifting the turning towards the trailing edge.
"""

import dataclasses
import logging
from typing import ClassVar

import numpy as np

from turbigen.node import Node

logger = logging.getLogger("turbigen")


def _validate_domain(m):
    """Check that a normalised meridional coordinate lies in [0, 1]."""
    if np.any(np.asarray(m) < 0.0) or np.any(np.asarray(m) > 1.0):
        raise ValueError("Meridional distance m must be in the range [0, 1].")


class CamberDesign(Node):
    """Base for camber line shapes.

    A shape knows nothing of the blade angles: it interpolates between them.
    The :doc:`/blade` page covers where the angles come from.
    """

    def chi_hat(self, m):
        """Return normalised camber at normalised meridional distance `m`.

        Zero at the leading edge and one at the trailing edge, so that the
        camber angle is recovered by interpolating the end angles with it.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement chi_hat(m)")


class Quadratic(CamberDesign):
    """Quadratic variation of camber line slope."""

    type: ClassVar[str] = "quadratic"

    aft_loading: float = 0.0
    """Shift of the camber towards the trailing edge [--].

    Zero gives a linear slope distribution, positive values move the turning
    aft, negative values forward.
    """

    def chi_hat(self, m):
        _validate_domain(m)
        a = self.aft_loading
        m = np.asarray(m, dtype=float)
        return m * (a * m + (1.0 - a))


@dataclasses.dataclass(frozen=True, eq=False)
class CamberLine:
    """A camber shape placed between known metal angles.

    Not a :class:`~turbigen.node.Node`: the end angles come from the mean line
    handed to a blade design, so a camber line is a result and never appears in
    a config file. Frozen for the same reason every result is.
    """

    shape: CamberDesign
    """Normalised camber distribution between the end angles."""

    tanchi_LE: float
    """Tangent of the metal angle at the leading edge."""

    tanchi_TE: float
    """Tangent of the metal angle at the trailing edge."""

    @property
    def Dtanchi(self):
        """Change in the tangent of the metal angle, leading to trailing [--]."""
        return self.tanchi_TE - self.tanchi_LE

    def dydm(self, m):
        """Return camber line slope at normalised meridional distance `m`."""
        _validate_domain(m)
        return self.tanchi_LE + self.shape.chi_hat(m) * self.Dtanchi

    def chi(self, m):
        """Return camber angle at normalised meridional distance `m` [deg]."""
        return np.degrees(np.arctan(self.dydm(m)))
