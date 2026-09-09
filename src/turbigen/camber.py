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

A shape states one thing: the camber line slope, between the end tangents it is
handed. Everything else about a shape is a statement about *which function of
the camber angle* runs smoothly along the chord, and the built-in shapes do not
agree on that, which is why the slope itself is the protocol:

:class:`Quadratic` (``quadratic``) is the built-in shape: the camber line slope
varies quadratically along the chord, with :attr:`~Quadratic.aft_loading`
shifting the turning towards the trailing edge.

:class:`Bernstein` (``bernstein``) is the flexible alternative: a linear
angle-tangent ramp plus :attr:`~Bernstein.order` ``- 1`` interior Bernstein
coefficients that perturb it. The endpoint coefficients are pinned at zero so
the ends stay put, and all-zero coefficients recover :class:`Quadratic` with
zero aft loading.

:class:`ClarkCamber` (``clark``) is the shape after :cite:`Clark2019`, the
companion of the thickness distribution of the same name: the end tangents
faded along the chord by a common power, whose :attr:`~ClarkCamber.exponent`
is the one parameter. An exponent of two is :class:`Quadratic` with no aft
loading; above that the chord flattens and the turning moves towards the more
turned end. The stagger it lands on is a *result*, which
:meth:`~ClarkCamber.stagger` reports.

:class:`CircularArc` (``circular_arc``) is the shape with no parameters at all:
constant curvature, which the end angles alone fix. Both shapes above
distribute ``tan chi`` along the chord, and so normalise into a curve that is
the same whatever the end angles are; an arc distributes ``sin chi``, and does
not. That is the whole reason a shape is asked for a slope between two known
tangents rather than for a normalised camber it could state on its own.
"""

import dataclasses
import logging
from typing import ClassVar

import numpy as np

from turbigen import shapespace
from turbigen.node import Node

logger = logging.getLogger("turbigen")


class CamberDesign(Node):
    """Base for camber line shapes.

    The :doc:`/blade` page covers where the end angles come from.
    """

    def dydm(self, m, tanchi_LE, tanchi_TE):
        """Return camber line slope at normalised meridional distance `m`.

        Running between `tanchi_LE` at the leading edge and `tanchi_TE` at the
        trailing edge, which a shape is handed rather than choosing: they are
        the mean line's business, and only the path between them is the
        shape's.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement dydm(...)")


class Quadratic(CamberDesign):
    """Quadratic variation of camber line slope."""

    type: ClassVar[str] = "quadratic"

    aft_loading: float = 0.0
    """Shift of the camber towards the trailing edge [--].

    Zero gives a linear slope distribution, positive values move the turning
    aft, negative values forward.
    """

    def dydm(self, m, tanchi_LE, tanchi_TE):
        shapespace.validate_domain(m)
        a = self.aft_loading
        m = np.asarray(m, dtype=float)
        return tanchi_LE + m * (a * m + (1.0 - a)) * (tanchi_TE - tanchi_LE)


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
            # The perturbation is elevated on its own, ramp and all left
            # out of it: elevation is linear and carries a ramp to a ramp, so
            # elevating ramp plus perturbation and subtracting the new ramp
            # would put the ramp in and take the same ramp back out again.
            # Its pinned ends survive, elevation leaving end coefficients be.
            b = shapespace.elevate_bernstein((0.0, *self.coeff, 0.0), n)
            object.__setattr__(self, "coeff", tuple(float(v) for v in b[1:-1]))

    def dydm(self, m, tanchi_LE, tanchi_TE):
        shapespace.validate_domain(m)
        n = self.order
        m = np.asarray(m, dtype=float)

        # The ramp is added to the evaluated curve rather than to the
        # coefficients, which leaves the coefficients purely the perturbation
        # and its ends pinned at zero.
        c = np.zeros(n - 1) if not self.coeff else np.asarray(self.coeff)
        chi_hat = m + shapespace.evaluate_bernstein((0.0, *c, 0.0), m)
        return tanchi_LE + chi_hat * (tanchi_TE - tanchi_LE)


class CircularArc(CamberDesign):
    """Camber line of constant curvature between the end angles.

    The shape with nothing to state: the end angles fix the arc, so there is
    no parameter here and nothing for an iterator to move. Constant curvature
    means the camber angle turns uniformly with arc length, or
    ``sin chi = (1 - m) sin chi_LE + m sin chi_TE`` against meridional
    distance --- closed form, so the arc costs no more to evaluate than the
    polynomial shapes do.

    The curvature is constant in the unwrapped meridional-tangential plane the
    other shapes are written in, before
    :meth:`~turbigen.blade.Blade.evaluate_section` projects onto the annulus.
    On a constant-radius annulus that is a true circle in ``(m, r theta)``;
    where the radius changes it is not, which is the same approximation every
    shape here already makes.
    """

    type: ClassVar[str] = "circular_arc"

    def dydm(self, m, tanchi_LE, tanchi_TE):
        shapespace.validate_domain(m)
        # No guard on the square root: the interior sine is a convex
        # combination of the two end sines, and a metal angle at or beyond 90
        # degrees is refused by `Blade.evaluate_chi` long before it reaches a
        # shape, so the magnitude here is strictly below one.
        s_LE, s_TE = (t / np.hypot(1.0, t) for t in (tanchi_LE, tanchi_TE))
        s = s_LE + np.asarray(m, dtype=float) * (s_TE - s_LE)
        return s / np.sqrt(1.0 - s * s)


class ClarkCamber(CamberDesign):
    """Power-law camber line after :cite:`Clark2019`.

    The camber line of the paper whose thickness distribution is
    :class:`~turbigen.thickness.ClarkThickness`, written in the slope the
    protocol here asks for. Clark states the camber itself,

    ``zeta = a m^n + b (1 - m)^n``, with ``a = tanchi_TE / n`` and
    ``b = -tanchi_LE / n``,

    whose slope is ``tanchi_TE m^(n-1) + tanchi_LE (1 - m)^(n-1)`` --- the two
    end tangents, each faded out along the chord by the same power. The ends
    are then exact for any :attr:`exponent` above one, and the rise of the
    line is ``(tanchi_LE + tanchi_TE) / n``, so that the stagger angle falls
    out of the exponent rather than the other way about.

    **Clark asks for the stagger and solves for the exponent; this asks for
    the exponent.** Both say the same curve, and stagger is the more physical
    of the two --- a designer has a view on it, and it is what the paper's
    optimiser and its network actually output. But the staggers a camber line
    can reach are an open interval its end angles fix, and taking the stagger
    as the parameter would put those angles into the meaning of the parameter:
    the same configured number would be a different curve at hub and at
    casing, a shape could refuse angles it is handed, a
    :class:`~turbigen.iterate.Deviation` moving ``dchi_TE`` could walk it into
    that refusal, and there would be no value to default to. The exponent has
    none of that, and it is the parameter the other shapes here are written
    like: a normalised curve, the same whatever the ends are. Ask
    :meth:`stagger` for the angle it lands on.

    An exponent of two gives ``tanchi_LE + m (tanchi_TE - tanchi_LE)``, exactly
    :class:`Quadratic` with zero aft loading, at the stagger of the mean of the
    end tangents that :class:`~turbigen.blade.DiffusionFactor` already assumes
    when it resolves a true chord. The converged designs of the paper sit
    between about 1.9 and 3.2, read off its Fig. 11.

    **The exponent flattens the middle rather than shifting the loading aft.**
    Both faded tangents carry the same factor ``0.5^(n-1)`` at mid-chord, so
    raising the exponent drives the camber angle there towards axial and packs
    the turning into the two ends. On a turbine that reads as aft loading, but
    only because the leading edge is the end already near axial: between metal
    angles of 20 and -70 degrees, 78% of the turning is done by mid-chord at an
    exponent of two and 41% of it at four. Turn a section the other way about,
    so the leading edge is the more turned end, and the same change front-loads
    it; on a section symmetric about axial it changes no fore-and-aft balance
    at all, only how tightly the turning bunches in the middle.

    **Above two, end tangents of the same sign give an inflected camber line.**
    The slope has an interior stationary point for any ``n > 2``, and where the
    ends turn the same way that point lies outside both of them --- the camber
    angle turns past its exit angle and comes back, an S in the annulus. A
    section that turns through axial has a monotonic slope at any exponent,
    which is every section in the paper; compressors it claims only as an
    extension.
    """

    type: ClassVar[str] = "clark"

    exponent: float = 2.0
    """Power the end tangents are faded along the chord by [--].

    Strictly greater than one, or the camber line does not reach the tangents
    it is placed between: the fading factors ``m^(n-1)`` and ``(1 - m)^(n-1)``
    only vanish at the far end for a positive power. Two is the quadratic
    camber line.
    """

    def __post_init__(self):
        if not self.exponent > 1.0:
            raise ValueError(
                f"A clark camber line needs an exponent greater than one to "
                f"reach the metal angles it is placed between, got "
                f"{self.exponent}. Two is the quadratic camber line."
            )

    def stagger(self, tanchi_LE, tanchi_TE):
        """Return the stagger angle between these end tangents [deg].

        The rise of the camber line over its chord, which is what
        :cite:`Clark2019` states and solves the exponent from. Here it is the
        other way round and this is a result, so it is a method taking the end
        angles rather than a field: a shape is not handed them until it is
        evaluated.
        """
        return np.degrees(np.arctan((tanchi_LE + tanchi_TE) / self.exponent))

    def dydm(self, m, tanchi_LE, tanchi_TE):
        shapespace.validate_domain(m)
        m = np.asarray(m, dtype=float)
        power = self.exponent - 1.0
        return tanchi_TE * m**power + tanchi_LE * (1.0 - m) ** power


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

    def dydm(self, m):
        """Return camber line slope at normalised meridional distance `m`."""
        return self.shape.dydm(m, self.tanchi_LE, self.tanchi_TE)

    def chi(self, m):
        """Return camber angle at normalised meridional distance `m` [deg]."""
        return np.degrees(np.arctan(self.dydm(m)))
