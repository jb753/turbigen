"""Geometry functions for manipulating aerofoil sections and annulus lines.

This module """
import numpy as np
from scipy.special import binom
from numpy.linalg import lstsq
from scipy.optimize import newton

import matplotlib.pyplot as plt

def fillet(x, r, dx):
    """Fillet over a join at |x|<dx."""

    # Get indices for the points at boundary of fillet
    ind = np.array(np.where(np.abs(x) <= dx)[0])
    ind1 = ind[ (0, -1), ]

    dr = np.diff(r) / np.diff(x)

    # Assemble matrix problem
    rpts = r[ ind1 ]
    xpts = x[ ind1 ]
    drpts = dr[ ind1 ]

    b = np.atleast_2d(np.concatenate((rpts, drpts))).T
    A = np.array(
        [
            [xpts[0] ** 3.0, xpts[0] ** 2.0, xpts[0], 1.0],
            [xpts[1] ** 3.0, xpts[1] ** 2.0, xpts[1], 1.0],
            [3.0 * xpts[0] ** 2.0, 2.0 * xpts[0], 1.0, 0.0],
            [3.0 * xpts[1] ** 2.0, 2.0 * xpts[1], 1.0, 0.0],
        ]
    )
    poly = np.matmul(np.linalg.inv(A), b).squeeze()

    r[ind] = np.polyval(poly, x[ind])

def blade_section(chi, a=0.0, tte=0.04, xttmax=(0.4, 0.2)):
    r"""Make a simple blade geometry with specified metal angles.

    Assume a cubic camber line :math:`\chi(\hat{x})` between inlet and exit
    metal angles, with a coefficient :math:`\mathcal{A}` controlling the
    chordwise loading distribution. Positive values of :math:`\mathcal{A}`
    correspond to aft-loading, and negative values front-loading. Default to a
    quadratic camber line with :math:`\mathcal{A}=0`.

    .. math ::

        \tan \chi(\hat{x}) = \left(\tan \chi_2 - \tan \chi_1
        \right)\left[\mathcal{A}\hat{x}^2 + (1-\mathcal{A})\hat{x}\right] +
        \tan \chi_1\,, \quad 0\le \hat{x} \le 1 \,.

    Use a quadratic thickness distribution parameterised by the maximum
    thickness location, with an aditional linear component to force finite
    leading and trailing edge thicknesses.

    This is a variation of the geometry parameterisation used by,

    Denton, J. D. (2017). "Multall---An Open Source, Computational Fluid
    Dynamics Based, Turbomachinery Design System."
    *J. Turbomach* 139(12): 121001.
    https://doi.org/10.1115/1.4037819

    Parameters
    ----------
    chi : array
        Metal angles at inlet and exit :math:`(\chi_1,\chi_2)` [deg].
    a : float, default=0
        Aft-loading factor :math:`\mathcal{A}` [--].

    Returns
    -------
    xy : array
        Non-dimensional blade coordinates [--], one row each for chordwise
        coordinate, tangential upper, and tangential lower coordinates.
    """

    xtmax, tmax = xttmax

    tle = tte/2.

    # Camber line
    xc = np.linspace(0.,1.,1000)

    tlin =  tle + xc * (tte - tle)

    # Additional component to make up maximum thickness
    tmod = tmax - xtmax * (tte - tle) - tle

    # Total thickness
    thick = tlin + tmod * (1.0 - 4.* np.abs(xc**(np.log(0.5)/np.log(xtmax)) - 0.5) **2.)

    # Assemble preliminary upper and lower coordiates
    xy_prelim = [thickness_to_coord(xc, sgn*thick, chi) for sgn in [1,-1]]

    # Fit Bernstein polynomials to the prelim coords in shape space
    A, _ = fit_aerofoil(xy_prelim, chi, tte, 4)

    # Upper and lower surfaces
    xy_up, xy_dn = [evaluate_surface(Ai,xc,chi, tte) for Ai in A]

    # Assemble coordinates and return
    return np.array(np.stack((xy_up, xy_dn)))


def bernstein(x, n, i):
    """Evaluate ith Bernstein polynomial of degree n at some x-coordinates."""
    return binom(n, i) * x ** i * (1.0 - x) ** (n - i)


def to_shape_space(x, z, zte):
	"""Transform real thickness to shape space."""
	# Ignore singularities at leading and trailing edges
	eps = 1e-6
	ii = np.abs(x - 0.5) < (0.5 - eps)
	s = np.ones(x.shape) * np.nan
	# s[ii] = (z[ii] - x[ii] * zte) / (np.sqrt(x[ii]) * (1.0 - x[ii]))
	s[ii] = z[ii] / (np.sqrt(x[ii]) * np.sqrt(1.0 - x[ii]))
	return s


def from_shape_space(x, s, zte):
    """Transform shape space to real coordinates."""
    # return np.sqrt(x) * (1.0 - x) * s + x * zte
    return np.sqrt(x) * np.sqrt(1.0 - x) * s


def evaluate_coefficients(x, A):
    """Evaluate a set of Bernstein polynomial coefficients at some x-coords."""
    n = len(A) - 1
    return np.sum(
        np.stack([A[i] * bernstein(x, n, i) for i in range(0, n + 1)]), axis=0
    )


def fit_coefficients(x, s, order, dx=0.05):
    """Fit shape-space distribution with Bernstein polynomial coefficients.

    Return both a vector of coefficients, length `order`, and sum residual."""
    n = order - 1
    # When converting from real coordinates to shape space, we end up with
    # singularities and numerical instability at leading and trailing edges.
    # So in these cases, ignore within dx at LE and TE
    itrim = np.abs(x - 0.5) < (0.5 - dx)
    xtrim = x[itrim]
    strim = s[itrim]
    # Evaluate all polynomials
    X = np.stack([bernstein(xtrim, n, i) for i in range(0, n + 1)]).T
    # All coefficients free - normal fit
    return lstsq(X, strim, rcond=None)[:2]


def fit_aerofoil(xy, chi, zte, order):
    """Fit Bernstein polynomial coefficients for both aerofoils surfaces."""
    n = order - 1
    dx = 0.02
    # When converting from real coordinates to shape space, we end up with
    # singularities and numerical instability at leading and trailing edges.
    # So in these cases, ignore within dx at LE and TE
    xtrim_all = []
    strim_all = []
    X_all = []
    X_le_all = []
    for xyi in xy:
        xc, yc, t = coord_to_thickness(xyi, chi)
        s = to_shape_space(xc, t, zte)
        itrim = np.abs(xc - 0.5) < (0.5 - dx)
        xtrim_all.append(xc[itrim])
        strim_all.append(s[itrim])
        X_all.append(np.stack([bernstein(xc[itrim], n, i) for i in range(1, n + 1)]).T)
        X_le_all.append(bernstein(xc[itrim], n, 0))

    strim = np.concatenate(strim_all)
    X_le = np.concatenate(X_le_all)
    X = np.block(
        [[X_all[0], np.zeros(X_all[1].shape)], [np.zeros(X_all[0].shape), X_all[1]]]
    )
    X = np.insert(X, 0, X_le, 1)
    A_all, resid = lstsq(X, strim, rcond=None)[:2]
    Au = A_all[:order]
    Al = -np.insert(A_all[order:], 0, A_all[0])
    return (Au, Al), resid


def resample_coefficients(A, order):
    """Up- or down-sample a set of coefficients to a new order."""
    x = np.linspace(0.0, 1.0)
    s = evaluate_coefficients(x, A)
    return fit_coefficients(x, s, order)


def evaluate_camber(x, chi):
    """Camber line as a function of x, given inlet and exit angles."""
    tanchi = np.tan(np.radians(chi))
    return tanchi[0] * x + 0.5 * (tanchi[1] - tanchi[0]) * x ** 2.0


def evaluate_camber_slope(x, chi):
    """Camber line slope as a function of x, given inlet and exit angles."""
    tanchi = np.tan(np.radians(chi))
    return tanchi[0] + (tanchi[1] - tanchi[0]) * x


def coord_to_thickness(xy, chi):
    """Perpendicular thickness distribution given camber line angles."""
    xu, yu = xy
    # Find intersections of xu, yu with camber line perpendicular
    def iterate(x):
        return yu - evaluate_camber(x, chi) + (xu - x) / evaluate_camber_slope(x, chi)

    with np.errstate(invalid="ignore",divide="ignore"):
        xc = newton(iterate, xu)
    yc = evaluate_camber(xc, chi)
    # Now evaluate thickness
    t = np.sqrt(np.sum(np.stack((xu - xc, yu - yc)) ** 2.0, 0))
    return xc, yc, t


def thickness_to_coord(xc, t, chi):
    theta = np.arctan(evaluate_camber_slope(xc, chi))
    yc = evaluate_camber(xc, chi)
    xu = xc - t * np.sin(theta)
    yu = yc + t * np.cos(theta)
    return xu, yu


def evaluate_surface(A, x, chi, zte):
    """Given a set of coefficients, return coordinates."""
    s = evaluate_coefficients(x, A)
    t = from_shape_space(x, s, zte*np.sign(s[0]))
    return thickness_to_coord(x, t, chi)


if __name__ == "__main__":

    pass
