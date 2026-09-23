"""
.. _recentre:

Re-centring a Clark section
^^^^^^^^^^^^^^^^^^^^^^^^^^^

A :class:`~turbigen.thickness.ClarkThickness` can differ side to side, so a
loading iterator free to thicken one surface and thin the other can leave the
camber line a long way from the middle of the aerofoil it carries. That costs
more than looks. On the concave side a surface offset from a curved camber line
has its curvature amplified by ``1 / (1 - t kappa)``, so a pressure surface
twice as thick as the suction surface can approach the fold at ``t kappa = 1``
and every small move of its coefficients then comes back as a large swing in
surface curvature.

This refits the section so that the camber line runs through the middle again,
without changing the aerofoil. A :class:`~turbigen.camber.ClarkCamber` has three
numbers: the two end tangents and the exponent. The exit tangent stays, because
the wedge is shared about the camber line and the exit angle is the camber's
alone. The rise over the chord, ``(tanchi_LE + tanchi_TE) / n``, stays too,
because it places the trailing edge relative to the leading edge. That leaves
one free parameter: the exponent, with the inlet tangent following it. The
round nose takes up the change in inlet metal angle.

For each trial exponent, both thickness curves are refitted to the existing
surfaces. The trailing edge coefficient is held and the nose one is shared, so
the refit keeps the same knobs :class:`~turbigen.iterate.ClarkProfile` moves.
The exponent kept is the one that best equalises the two thicknesses.

The fit is done in the normalised plane the camber line is drawn in, before the
section is rescaled onto the row and wrapped onto the annulus. Both of those
steps are nearly the same map for sections this close, so a match here is a
match on the blade.
"""

import dataclasses
import logging

import numpy as np
from scipy.optimize import least_squares, minimize_scalar
from scipy.spatial import cKDTree

import turbigen.util
from turbigen.blade import offset_surfaces
from turbigen.camber import CamberLine, ClarkCamber

logger = logging.getLogger("turbigen")

EXPONENT_LIM = (1.2, 6.0)
"""Range of Clark exponents searched [--].

Above one so the camber reaches its end tangents, and wide enough to hold the
1.9 to 3.2 :cite:`Clark2019` reports with room either side."""

N_REFERENCE = 20001
"""Points the original surfaces are sampled at, as the curve refitted to."""

N_FIT = 400
"""Points each trial surface is compared at."""


@dataclasses.dataclass(frozen=True)
class Recentred:
    """A refitted section: the same aerofoil, hung off a different camber line."""

    camber: CamberLine
    """Camber line with the refitted exponent and inlet tangent."""

    thickness: object
    """Two-sided thickness refitted to the original surfaces."""

    error: float
    """Largest distance from a refitted surface to the original one,
    normalised by meridional chord [--]."""


def _layout(camber, thickness, fac_tangential, suction_is_upper, m):
    """Return both surfaces in the normalised plane, shape (2, 2, n)."""
    y = turbigen.util.cumtrapz0(camber.dydm(m), m)
    Dm, Dy = offset_surfaces(camber, thickness, fac_tangential, suction_is_upper, m)
    return np.stack((m + Dm, y + Dy), axis=1)


def imbalance(thickness):
    """Return the mean square difference between the two thicknesses [--].

    Zero for a section whose camber line runs midway between its surfaces.
    """
    m = np.linspace(0.0, 1.0, 201)
    t_s, t_p = thickness.thick_both(m)
    return float(np.trapezoid((t_s - t_p) ** 2, m))


def recentre(camber, thickness, fac_tangential, suction_is_upper):
    """Return the section refitted onto the Clark exponent that equalises thickness.

    Parameters
    ----------
    camber : CamberLine
        Current camber line. Its shape must be a
        :class:`~turbigen.camber.ClarkCamber`.
    thickness : ClarkThickness
        Current two-sided thickness.
    fac_tangential : float
        See :attr:`~turbigen.blade.SectionDesign.fac_tangential`.
    suction_is_upper : bool
        See :attr:`~turbigen.blade.Blade._suction_is_upper`.

    Returns
    -------
    Recentred
        The refitted camber line and thickness, and how closely they trace the
        original surfaces. The original section, with zero error, if no
        exponent in :data:`EXPONENT_LIM` improves on its balance.

    """
    if not isinstance(camber.shape, ClarkCamber):
        raise ValueError(
            f"Re-centring moves a Clark camber exponent, and this section has a "
            f"{type(camber.shape).__name__} camber line."
        )

    tanchi_TE = camber.tanchi_TE
    rise = (camber.tanchi_LE + tanchi_TE) / camber.shape.exponent

    original = _layout(
        camber,
        thickness,
        fac_tangential,
        suction_is_upper,
        turbigen.util.cluster_cosine(N_REFERENCE),
    )
    trees = [cKDTree(surface.T) for surface in original]
    m_fit = turbigen.util.cluster_cosine(N_FIT)

    c0 = thickness.tau_coeff
    n_interior = thickness.order - 1

    def placed(exponent):
        return CamberLine(
            ClarkCamber(exponent=exponent), exponent * rise - tanchi_TE, tanchi_TE
        )

    def with_knobs(p):
        # The trailing edge column stays as it was: the wedge is the exit
        # angle's, and the exit angle is not being moved.
        c = c0.copy()
        c[:, 0] = p[0]
        c[:, 1:-1] = np.reshape(p[1:], (2, n_interior))
        return thickness.with_tau_coeff(c)

    def residual(p, trial):
        surfaces = _layout(
            trial, with_knobs(p), fac_tangential, suction_is_upper, m_fit
        )
        return np.concatenate(
            [tree.query(surface.T)[0] for tree, surface in zip(trees, surfaces)]
        )

    p0 = np.concatenate(([c0[0, 0]], c0[:, 1:-1].ravel()))
    fits = {}

    def fit(exponent):
        if exponent not in fits:
            sol = least_squares(residual, p0, args=(placed(exponent),), x_scale=0.1)
            fits[exponent] = Recentred(
                placed(exponent), with_knobs(sol.x), float(np.max(sol.fun))
            )
        return fits[exponent]

    best = minimize_scalar(
        lambda exponent: imbalance(fit(exponent).thickness),
        bounds=EXPONENT_LIM,
        method="bounded",
        options={"xatol": 0.01},
    )
    refitted = fit(float(best.x))

    if imbalance(refitted.thickness) >= imbalance(thickness):
        return Recentred(camber, thickness, 0.0)
    return refitted
