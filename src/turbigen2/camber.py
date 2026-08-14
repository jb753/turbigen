"""Camber lines.

A :class:`CamberDesign` is a config node describing the *shape* of a camber
line between its end angles; pairing it with those angles gives a
:class:`CamberLine`, which can be evaluated. The split is the same one the rest
of the package makes, applied one level down: the end angles are not design
variables at all but a result, being the local flow angle plus the recamber
asked for by a :class:`~turbigen2.blade.Section`.

That is what retires ``apply_recamber``/``undo_recamber``. The package this
replaces stores the recamber angles in the first two slots of a parameter
vector, overwrites them in place with the metal angles once a mean line is
known, and guards the whole thing with an ``is_recambered`` flag that plots
toggle on and off. Here the two live in different objects and a metal angle is
computed once.
"""

import dataclasses
import logging
from typing import ClassVar

import numpy as np

from turbigen2.node import Node

logger = logging.getLogger("turbigen")


def _validate_domain(m):
    """Check that a normalised meridional coordinate lies in [0, 1]."""
    if np.any(np.asarray(m) < 0.0) or np.any(np.asarray(m) > 1.0):
        raise ValueError("Meridional distance m must be in the range [0, 1].")


class CamberDesign(Node):
    """Base for camber line shapes.

    A shape knows nothing of the blade angles: it interpolates between them.
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

    Not a :class:`~turbigen2.node.Node`: the end angles come from the mean line
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
    def chi_LE(self):
        """Metal angle at the leading edge [deg]."""
        return np.degrees(np.arctan(self.tanchi_LE))

    @property
    def chi_TE(self):
        """Metal angle at the trailing edge [deg]."""
        return np.degrees(np.arctan(self.tanchi_TE))

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
