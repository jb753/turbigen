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

:class:`Bernstein` (``bernstein``) is the flexible alternative: a linear
angle-tangent ramp plus :attr:`~Bernstein.order` ``- 1`` interior Bernstein
coefficients that perturb it. The endpoint coefficients are pinned at zero so
the ends stay put, and all-zero coefficients recover :class:`Quadratic` with
zero aft loading.
"""

import dataclasses
import logging
import math
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


def _elevate_control_points(b):
    """Degree-elevate a Bezier control-point array by one degree.

    Exact identity, not an approximation: the elevated curve traces the same
    shape as `b`, just written in the next degree's basis.
    """
    n = len(b) - 1
    i = np.arange(n + 2)
    left = np.concatenate(([0.0], b))
    right = np.concatenate((b, [0.0]))
    t = i / (n + 1)
    return t * left + (1.0 - t) * right


class Bernstein(CamberDesign):
    """Bernstein-polynomial perturbation of a quadratic camber line."""

    type: ClassVar[str] = "bernstein"

    order: int = 3
    """Degree of the Bernstein polynomial; one more than the coefficient count."""

    coeff: tuple[float, ...] = ()
    """Interior Bernstein coefficients, local changes about a linear ramp [--].

    The two endpoint coefficients are fixed at zero so the camber line ends
    stay put; supply all :attr:`order` ``- 1`` interior values here, or none
    at all to recover a quadratic camber line. A shorter, non-empty `coeff`
    is only accepted with :attr:`upsample` set, since padding it with zeros
    would silently change the shape rather than leave it alone.
    """

    upsample: bool = False
    """Reinterpret a short :attr:`coeff` as a lower-order curve, exactly.

    A `coeff` shorter than :attr:`order` ``- 1`` is read, once at
    construction, as the interior coefficients of a *lower-order* Bernstein
    curve, which is then raised to :attr:`order` by Bezier degree elevation
    --- an exact identity, not a fit --- and :attr:`coeff` is replaced by the
    elevated, full-length array. The camber line is unchanged by this and the
    extra coefficients are then free to perturb it further. Sampling the
    low-order curve and solving a matrix problem to match points would only
    approximate this, and get worse-conditioned as the order grows; degree
    elevation gets the same result exactly and in closed form, so use that
    instead.
    """

    def __post_init__(self):
        n = self.order
        if n < 2:
            raise ValueError("Bernstein camber order must be at least 2.")
        if len(self.coeff) > n - 1:
            raise ValueError(
                f"Bernstein camber of order {n} takes at most {n - 1} "
                f"coefficients, got {len(self.coeff)}."
            )
        if self.coeff and len(self.coeff) < n - 1:
            if not self.upsample:
                raise ValueError(
                    f"Bernstein camber of order {n} needs all {n - 1} "
                    f"interior coefficients, got {len(self.coeff)}. Either "
                    f"supply them all, or set upsample=True to raise this "
                    f"shorter curve to order {n} exactly."
                )
            # Full control points of the low-order curve, ramp plus
            # perturbation, endpoints pinned at 0 and 1.
            n_low = len(self.coeff) + 1
            b = np.concatenate(
                ([0.0], np.arange(1, n_low) / n_low + np.asarray(self.coeff), [1.0])
            )
            for _ in range(n - n_low):
                b = _elevate_control_points(b)
            coeff = tuple(float(v) for v in b[1:-1] - np.arange(1, n) / n)
            object.__setattr__(self, "coeff", coeff)

    def chi_hat(self, m):
        _validate_domain(m)
        n = self.order
        m = np.asarray(m, dtype=float)
        scalar = m.ndim == 0
        m = np.atleast_1d(m)
        c = np.zeros(n - 1) if not self.coeff else np.asarray(self.coeff)
        i = np.arange(1, n)
        binom = np.array([math.comb(n, k) for k in i], dtype=float)
        basis = (
            binom[:, None]
            * m[None, :] ** i[:, None]
            * (1.0 - m)[None, :] ** (n - i)[:, None]
        )
        chi_hat = m + c @ basis
        return chi_hat[0] if scalar else chi_hat


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
