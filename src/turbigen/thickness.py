"""
.. _thickness:

Thickness distributions
^^^^^^^^^^^^^^^^^^^^^^^^

A :class:`ThicknessDesign` evaluates thickness as a function of normalised
meridional distance. Thickness is always normalised by the meridional chord, so
there is no reference length to choose: a design is evaluable as soon as it
exists, rather than only once a chord is known.

:class:`Taylor` (``taylor``) is the built-in distribution: two cubic splines in
shape space meeting at the point of maximum thickness, after
:cite:`Taylor2016`.

:class:`Clark` (``clark``) is the two-sided alternative, after Clark (2019): a
Bernstein polynomial in shape space per surface, so the aerofoil need not be
symmetric about its camber line. A distribution answers with a half-thickness
for each surface, suction first, and one that says nothing about sides gives
the same number twice.
"""

import dataclasses
import logging
from typing import ClassVar

import numpy as np

from turbigen import shapespace
from turbigen.node import Node

logger = logging.getLogger("turbigen")

_PEAK_TOL = 1e-10
"""Slack on the peak thickness check, for round-off in the cubic fit."""

_ROOT_TOL = 1e-9
"""Largest imaginary part for a root to count as real."""


class ThicknessDesign(Node):
    """Base for thickness distributions, normalised by meridional chord.

    The :doc:`/blade` page covers how a distribution is placed on a blade.
    """

    def thick(self, m):
        """Return half-thickness at normalised meridional distance `m`.

        For a distribution that is the same both sides of the camber line. A
        distribution that is not implements :meth:`thick_both` instead, and
        has no one number to answer with here.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement thick(m)")

    def thick_both(self, m):
        """Return the half-thickness of each surface, suction first.

        The order :meth:`~turbigen.blade.Blade.evaluate_section` returns the
        surfaces in, so a two-sided distribution and the section it produces
        index the same way with no permutation between them.

        **A distribution cannot check this about itself.** Which side of a
        camber line is the suction one is a property of the camber, and a
        thickness has none --- so "suction first" is a promise the *blade*
        keeps when it hangs these two numbers off its camber line (see
        :attr:`~turbigen.blade.Blade._suction_is_upper`), and a claim about
        intent when it is written in a config file. Evaluated on its own, a
        distribution can only say that its first answer goes on whichever
        surface a blade would call suction.

        The same number twice unless a distribution says otherwise, so a
        symmetric one need only write :meth:`thick`.
        """
        t = self.thick(m)
        return t, t


class Taylor(ThicknessDesign):
    """Two cubic splines in shape space, after :cite:`Taylor2016`.

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
        s_LE = shapespace.tau_LE(self.R_LE)
        s_max = (self.t_max - m_tmax * t_TE / 2.0) / np.sqrt(m_tmax) / (1.0 - m_tmax)
        ds_max = (
            (
                s_max * (np.sqrt(m_tmax) - (1.0 - m_tmax) / 2.0 / np.sqrt(m_tmax))
                - t_TE / 2.0
            )
            / np.sqrt(m_tmax)
            / (1.0 - m_tmax)
        )
        s_TE = shapespace.tau_TE(t_TE, self.tanwedge)

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

    def __post_init__(self):
        peak = self.peak()
        if peak > self.t_max + _PEAK_TOL:
            raise ValueError(
                f"This thickness distribution peaks at {peak:.4g}, above the "
                f"t_max={self.t_max} it declares. Its cubic bulges between the "
                f"leading edge and the point of maximum thickness; try a "
                f"smaller R_LE, a larger m_tmax, or a different kappa_max."
            )

    def peak(self):
        """Return the largest half-thickness anywhere in ``0 <= m <= 1`` [--].

        Exact rather than sampled. The square root in :meth:`thick` is what
        makes it awkward, and substituting ``m = u**2`` removes it: for a cubic
        ``tau = a m^3 + b m^2 + c m + d`` the half-thickness becomes a
        polynomial in ``u``,

            t(u) = -a u^9 + (a-b) u^7 + (b-c) u^5 + (c-d) u^3
                   + (t_TE/2) u^2 + d u

        so the stationary points are the real roots of a degree-8 derivative
        and the maximum is the largest of ``t`` at those and at the ends of the
        interval. Sampling instead would need a point count chosen by taste: a
        coarse grid misses shallow excursions, and there is no density that is
        obviously enough.

        The ends are always included, so the peak location can never be missed
        outright; and because ``m_tmax`` is an end of both pieces, ``t_max``
        itself is always a candidate. The comparison is therefore always
        "does it exceed its declared peak", never "did we happen to find it".
        """
        coeff_front, coeff_rear = self._coeff

        best = -np.inf
        for coeff, m_lim in (
            (coeff_front, (0.0, self.m_tmax)),
            (coeff_rear, (self.m_tmax, 1.0)),
        ):
            a, b, c, d = coeff
            poly = np.array(
                [-a, 0.0, a - b, 0.0, b - c, 0.0, c - d, self.t_TE / 2.0, d, 0.0]
            )

            u_lo, u_hi = np.sqrt(m_lim[0]), np.sqrt(m_lim[1])
            roots = np.roots(np.polyder(poly))
            real = roots[np.abs(roots.imag) < _ROOT_TOL].real
            candidates = np.concatenate(
                [real[(real >= u_lo) & (real <= u_hi)], [u_lo, u_hi]]
            )
            best = max(best, np.polyval(poly, candidates).max())

        return float(best)

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
        shapespace.validate_domain(m)

        m_array = np.asarray(m, dtype=float)
        t = shapespace.thickness_from_tau(m_array, self.tau(m_array), self.t_TE)

        # No bound check here. Whether the distribution stays under its own
        # t_max is a property of the parameters, not of the points asked for,
        # so `__post_init__` settles it once -- and settles it for the whole
        # domain, where checking here could only ever cover the samples given.
        return float(t.item()) if np.isscalar(m) else t


class ClarkThickness(ThicknessDesign):
    """A Bernstein polynomial in shape space per surface, after Clark (2019).

    Where :class:`Taylor` describes one thickness reflected in the camber
    line, this describes each surface separately, which is what lets a
    section be shaped to a loading distribution rather than to a thickness
    parameter. The camber line stays where it was put: only the two surfaces
    move.

    Two things are shared rather than written per surface. The leading edge
    radius, so that the nose is one curvature rather than two meeting at a
    point --- Clark's own reason, and the one sharing that cannot be undone.
    And the trailing edge thickness, split evenly, so a blunt trailing edge
    stays centred on the camber line however the surfaces arrive at it.

    The wedge angle is per surface, always: each leaves the trailing edge at
    its own angle, which is a degree of freedom in the rear of the blade and a
    claim on the exit direction --- see :attr:`tanwedge`. Writing the same
    number twice is the symmetric section this class started as, and is the
    only way to ask for it.

    What is left is the interior of each surface, and only the interior: each
    is the straight line in shape space between its own two endpoints plus a
    Bernstein perturbation pinned at zero at both ends. So the leading edge
    radius and the wedge angles are exactly what they say however the
    coefficients move, and all-zero coefficients with one wedge angle give a
    section symmetric about the camber line.
    """

    type: ClassVar[str] = "clark"

    R_LE: float
    """Leading edge radius, normalised by meridional chord [--]. Shared by
    both surfaces, so that the nose is a single curvature."""

    coeff: tuple[tuple[float, ...], ...] = ((), ())
    """Interior Bernstein coefficients in shape space, one row per surface
    [--].

    The suction surface first, which is the order
    :meth:`~turbigen.blade.Blade.evaluate_section` returns the surfaces in and
    the order :meth:`thick_both` answers in --- so a row here and the surface
    it shapes line up by index, with nothing in between to get backwards.

    Which side that is comes from the camber line, not from here: see
    :meth:`ThicknessDesign.thick_both` for why a thickness distribution cannot
    check its own row ordering, and
    :attr:`~turbigen.blade.Blade._suction_is_upper` for where it is decided.

    The two rows are the same length, and that length sets the order of the
    curve --- `order - 1` interior coefficients, the two endpoint ones being
    the leading edge radius and the wedge angle rather than free. Empty rows
    are the straight line in shape space between them, and a section
    symmetric about its camber line.
    """

    tanwedge: tuple[float, float] = (0.0, 0.0)
    """Tangent of each surface's trailing edge wedge angle [--].

    One per surface, suction first as :attr:`coeff` is, and always a pair:
    equal entries are how a section asks to be symmetric about its camber line
    at the trailing edge, rather than a different kind of section. Nothing
    downstream has to ask which form was written, and the loop that shapes
    these has the same number of knobs whatever they hold.

    **What a pair costs, which is not thickness.** The trailing edge point
    stays where it was: half-thickness at ``m = 1`` is ``t_TE / 2`` on each
    surface whatever the wedge does, that being the linear ramp in
    :func:`~turbigen.shapespace.thickness_from_tau` rather than anything the
    shape space curve reaches. What moves is the direction the aerofoil
    leaves in. With one wedge the two surfaces depart symmetrically about the
    camber line, so the camber's
    :meth:`~turbigen.blade.Blade.evaluate_chi` *is* the section's exit
    direction; with two the bisector of the surfaces rotates away from it, by
    of order twenty-five degrees per unit of difference between them.

    **So a pair is a claim on the exit angle**, which the camber line
    otherwise owns alone.
    :class:`~turbigen.iterate.Deviation` will see that rotation as an exit
    flow angle error and pull ``dchi_TE`` back against it, which is the right
    answer and within the authority it has; but an asymmetry small enough to
    move the exit angle by less than that iterator's tolerance is one it will
    not chase, and the section carries it. Nothing here bounds the asymmetry:
    it is a design decision, not a range.

    **What a pair buys.** A shape-space endpoint moves the thickness as
    ``m^1.5 (1 - m)``, peaking at ``m = 0.6``, so splitting this is authority
    over the rear of the surface rather than a detail at its very end. It is
    the one place :class:`~turbigen.iterate.ClarkProfile` had none: with one
    wedge, the two surfaces' residuals at that end enter only as their mean,
    and a pair that is equally wrong in opposite directions reads as
    converged.
    """

    t_TE: float = 0.0
    """Trailing edge thickness, the total due to both sides [--]."""

    def __post_init__(self):
        if len(self.coeff) != 2:
            raise ValueError(
                f"A two-sided thickness takes one row of coefficients per "
                f"surface, so two rows, got {len(self.coeff)}."
            )

        widths = [len(row) for row in self.coeff]
        if widths[0] != widths[1]:
            raise ValueError(
                f"Both surfaces must carry the same number of coefficients, "
                f"which is what sets the order of the curve, got {widths}."
            )

        if self.R_LE <= 0.0:
            raise ValueError(
                f"A leading edge radius must be positive, got R_LE={self.R_LE}."
            )

        if len(self.tanwedge) != 2:
            raise ValueError(
                f"A wedge angle is written per surface, so two of them, got "
                f"{len(self.tanwedge)}."
            )

    @property
    def order(self):
        """Degree of the shape-space curve on each surface [--].

        One more than the number of interior coefficients, the two endpoint
        ones being the leading edge radius and the wedge angle. Read off
        :attr:`coeff` rather than declared, so there is no second place for it
        to be written down and disagree.
        """
        return len(self.coeff[0]) + 1

    @property
    def m_ctl(self):
        """Return where each control point moves this section most, shape (order+1,).

        One array, not one per surface: it is a property of the basis, and both
        rows carry the same number of coefficients by construction. See
        :func:`turbigen.shapespace.control_m` for why these sit strictly inside
        the ends.

        Public because an iterator shaping this distribution against a loading
        distribution has to sample the achieved curve at the points its knobs
        actually act on, and a sampler guessing at its own positions would be
        reading somewhere no coefficient answers for.
        """
        return shapespace.control_m(self.order)

    @property
    def tau_coeff(self):
        """Return the full shape-space coefficients of each surface, shape (2, order+1).

        :attr:`coeff` holds only the interior *perturbation*, on a straight line
        between two endpoints the physics fixes. A straight line is exactly
        representable in the Bernstein basis, though, so the whole curve is one
        Bernstein polynomial whose coefficients are

        .. math::
            c_k = \\tau_{LE} + (\\tau_{TE} - \\tau_{LE}) k / n + p_k

        with `p_0` and `p_n` zero. That makes ``c[0]`` exactly
        :func:`~turbigen.shapespace.tau_LE` of the leading edge radius and
        ``c[-1]`` exactly :func:`~turbigen.shapespace.tau_TE` of the trailing
        edge --- the two ends are not special cases of the curve, they *are* its
        first and last coefficients.

        Why anything wants this: in these coordinates every coefficient does the
        same kind of thing, which is to raise `tau` and thicken the surface
        locally. A knob per coefficient therefore has one sensitivity sign
        rather than one for the interior and something else for the ends, and a
        positive `R_LE` comes for free since it is `c[0]**2 / 2`.

        **The two rows share their first entry and need not share their last.**
        The first is the one leading edge radius, returned duplicated rather
        than split out so that a row lines up index for index with
        :attr:`m_ctl`, and :meth:`with_tau_coeff` checks that a caller writing
        it back has kept the two equal. The last is that surface's own wedge
        angle, equal between the rows only when :attr:`tanwedge` was written as
        one number.
        """
        n = self.order
        tau_LE = shapespace.tau_LE(self.R_LE)
        k = np.arange(n + 1) / n
        lines = np.array(
            [
                tau_LE + (shapespace.tau_TE(self.t_TE, tanwedge) - tau_LE) * k
                for tanwedge in self.tanwedge
            ]
        )

        return np.array([[0.0, *row, 0.0] for row in self.coeff]) + lines

    def with_tau_coeff(self, c):
        """Return this thickness rebuilt from full shape-space coefficients.

        The inverse of :attr:`tau_coeff`, and the only supported way to write
        through it. **Not because the arithmetic is hard, but because it does
        not decompose**: moving `c[0]` moves the straight line under the whole
        curve, so every interior perturbation has to be recomputed to leave its
        own coefficient where the caller put it. Setting `R_LE` and then the
        interior coefficients one at a time --- the obvious thing for a caller
        holding the fields --- silently drags the interior along behind the
        nose.

        Parameters
        ----------
        c : array_like, shape (2, order + 1)
            Full shape-space coefficients of each surface, suction first. The
            two rows must agree at the leading edge, that being the one radius
            the nose has; their trailing edge entries are each surface's own
            wedge angle and are free to differ.

        """
        c = np.asarray(c, dtype=float)
        if c.shape != (2, self.order + 1):
            raise ValueError(
                f"An order {self.order} two-sided thickness takes coefficients "
                f"of shape {(2, self.order + 1)}, got {c.shape}."
            )

        # Not defaulted to row zero: the surfaces sharing one nose is the
        # whole of what "one leading edge radius" means here, and a caller who
        # has broken it is asking for a section this class cannot represent
        # rather than one it should quietly round off. The trailing edge is not
        # checked, the two surfaces being free to leave at their own angles.
        if c[0][0] != c[1][0]:
            raise ValueError(
                f"Both surfaces share one leading edge, so their coefficients "
                f"there must be equal, got {c[0][0]} and {c[1][0]}."
            )

        tau_LE = c[0][0]
        k = np.arange(self.order + 1) / self.order
        lines = np.array([tau_LE + (row[-1] - tau_LE) * k for row in c])

        # Back to plain floats in plain tuples, which is what a Node holds and
        # what a config file has to be able to carry: `dataclasses.replace`
        # writes whatever it is handed, and numpy scalars survive as far as
        # `to_dict` and the YAML it is written to.
        wedges = tuple(
            float(shapespace.tanwedge_from_tau(row[-1], self.t_TE)) for row in c
        )

        return dataclasses.replace(
            self,
            R_LE=float(shapespace.R_LE_from_tau(tau_LE)),
            tanwedge=wedges,
            coeff=tuple(tuple(float(v) for v in row[1:-1]) for row in (c - lines)),
        )

    def tau(self, m):
        """Return shape space of each surface at meridional distance `m`.

        Suction first, as :meth:`thick_both` returns them.
        """
        m = np.asarray(m, dtype=float)

        # The straight line between the two endpoints the physics fixes. It is
        # added to the evaluated perturbation rather than to its coefficients,
        # which is what keeps the coefficients purely the perturbation --- see
        # `turbigen.shapespace`.
        tau_LE = shapespace.tau_LE(self.R_LE)

        # One line per surface: they leave the same nose and arrive at their
        # own wedge, which is the whole of what a split wedge angle means.
        return tuple(
            tau_LE
            + (shapespace.tau_TE(self.t_TE, tanwedge) - tau_LE) * m
            + shapespace.evaluate_bernstein((0.0, *row, 0.0), m)
            for row, tanwedge in zip(self.coeff, self.tanwedge)
        )

    def thick_both(self, m):
        """Return the half-thickness of each surface, suction first.

        The trailing edge thickness is the total due to both sides, so half of
        it is left on each surface at the trailing edge.
        """
        shapespace.validate_domain(m)

        m_array = np.asarray(m, dtype=float)
        t = tuple(
            shapespace.thickness_from_tau(m_array, tau, self.t_TE)
            for tau in self.tau(m_array)
        )

        if np.isscalar(m):
            return tuple(float(side.item()) for side in t)
        return t
