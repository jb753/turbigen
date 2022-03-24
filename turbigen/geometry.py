"""Geometry functions for manipulating aerofoil sections and annulus lines.

This module """
import numpy as np
from scipy.special import binom
from numpy.linalg import lstsq
from scipy.optimize import newton
from scipy.interpolate import interp1d

nx = 201


def fillet(x, r, dx):
    """Fillet over a join at |x|<dx."""

    # Get indices for the points at boundary of fillet
    ind = np.array(np.where(np.abs(x) <= dx)[0])
    ind1 = ind[
        (0, -1),
    ]

    dr = np.diff(r) / np.diff(x)

    # Assemble matrix problem
    rpts = r[ind1]
    xpts = x[ind1]
    drpts = dr[ind1]

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


def cluster(npts):
    """Return a cosinusoidal clustering function with a set number of points."""
    # Define a non-dimensional clustering function
    return 0.5 * (1.0 - np.cos(np.pi * np.linspace(0.0, 1.0, npts)))


def evaluate_prelim_thickness(x, tte=0.04, xtmax=0.4, tmax=0.2):
    """A rough cubic thickness distribution."""
    tle = tte / 2.0
    tlin = tle + x * (tte - tle)
    # Additional component to make up maximum thickness
    tmod = tmax - xtmax * (tte - tle) - tle
    # Total thickness
    thick = tlin + tmod * (
        1.0 - 4.0 * np.abs(x ** (np.log(0.5) / np.log(xtmax)) - 0.5) ** 2.0
    )
    return thick


def prelim_A(chi):
    """Get values of A corresponding to preliminary thickness distribution."""

    xc = cluster(nx)

    # Camber line
    thick = evaluate_prelim_thickness(xc)

    # Assemble preliminary upper and lower coordiates
    xy_prelim = [thickness_to_coord(xc, sgn * thick, chi) for sgn in [1, -1]]

    # Fit Bernstein polynomials to the prelim coords in shape space
    A, _ = fit_aerofoil(xy_prelim, chi, order=4)

    return A


def evaluate_section_xy(chi, A, x=None):
    r"""Coordinates for blade section with specified camber and thickness.

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
    A : array, 2-by-(order-1)
        Thickness coefficients :math:`\mathcal{A}` [--].

    Returns
    -------
    xy : array
        Non-dimensional blade coordinates [--], one row each for chordwise
        coordinate, tangential upper, and tangential lower coordinates.
    """

    if x is None:
        # Choose some x coordinates
        x = cluster(nx)

    # Convert from shape space to thickness 
    s = evaluate_coefficients(x, A)
    t = from_shape_space(x, s, zte=None)

    # Flip the lower thickness
    t[1] = -t[1]

    # Apply thickness to camber line and return
    xy = np.stack([thickness_to_coord(x, ti, chi) for ti in t])
    return xy

def bernstein(x, n, i):
    """Evaluate ith Bernstein polynomial of degree n at some x-coordinates."""
    return binom(n, i) * x ** i * (1.0 - x) ** (n - i)


def to_shape_space(x, z, zte):
    """Transform real thickness to shape space."""
    # Ignore singularities at leading and trailing edges
    eps = 1e-6
    with np.errstate(invalid="ignore", divide="ignore"):
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
    A = np.atleast_2d(A)
    nsurf, order = A.shape
    n = order - 1
    t = np.empty((order,nsurf,) + x.shape)
    # Loop over surfaces and polynomials
    for i in range(order):
        for j in range(nsurf):
            t[i,j,...] = A[j,i] * bernstein(x, n, i)
    # Sum the polynomials
    return np.sum(t,axis=0)


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


def fit_aerofoil(xy, chi, order):
    """Fit Bernstein polynomial coefficients for both aerofoil surfaces."""
    n = order - 1
    # When converting from real coordinates to shape space, we end up with
    # singularities and numerical instability at leading and trailing edges.
    # So in these cases, ignore within dx at LE and TE
    dx = 0.02
    xtrim_all = []
    strim_all = []
    X_all = []
    X_le_all = []
    for xyi in xy:
        xc, yc, t = coord_to_thickness(xyi, chi)
        s = to_shape_space(xc, t, None)
        with np.errstate(invalid="ignore", divide="ignore"):
            itrim = np.abs(xc - 0.5) < (0.5 - dx)
        xtrim_all.append(xc[itrim])
        strim_all.append(s[itrim])
        X_all.append(
            np.stack([bernstein(xc[itrim], n, i) for i in range(1, n + 1)]).T
        )
        X_le_all.append(bernstein(xc[itrim], n, 0))

    strim = np.concatenate(strim_all)
    X_le = np.concatenate(X_le_all)
    X = np.block(
        [
            [X_all[0], np.zeros(X_all[1].shape)],
            [np.zeros(X_all[0].shape), X_all[1]],
        ]
    )
    X = np.insert(X, 0, X_le, 1)
    A_all, resid = lstsq(X, strim, rcond=None)[:2]
    Au = A_all[:order]
    Al = np.insert(A_all[order:], 0, A_all[0])
    return np.vstack((Au, Al)), resid


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
    """Perpendicular thickness distribution given camber line angles.

    Parameters
    ----------

    xy : array, 2-by-...
        Cartesian coordinates of a blade surface.
    chi : array, len 2
        Camber angles for inlet and outlet."""

    # Split into x and y
    x, y = xy
    # Find intersections of xu, yu with camber line perpendicular
    def iterate(xi):
        return (
            y
            - evaluate_camber(xi, chi)
            + (x - xi) / evaluate_camber_slope(xi, chi)
        )

    with np.errstate(invalid="ignore", divide="ignore"):
        xc = newton(iterate, x)
    yc = evaluate_camber(xc, chi)
    # Now evaluate thickness
    t = np.sqrt(np.sum(np.stack((x - xc, y - yc), axis=0) ** 2.0, axis=0))

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
    t = from_shape_space(x, s, zte)
    return thickness_to_coord(x, t, chi)


def radially_interpolate_section(spf, chi, spf_q, A=None, spf_A=None):
    """Given radial metal angle distributions, get blade section at a span."""

    # Check input shape
    spf = spf.reshape(-1)
    nr = len(spf)
    if not chi.shape == (2, nr):
        raise ValueError("Input metal angle data wrong shape.")

    func_chi = interp1d(spf, chi)

    # Fit A at midspan using preliminary thickness distribution if not specified
    if A is None:
        chi_mid = func_chi(0.5)
        A = prelim_A(chi_mid)

    if np.shape(spf_q) == ():
        nq = 1
        spf_q = (spf_q,)
    else:
        nq = len(spf_q)
    chi_q = func_chi(spf_q).T

    if np.ndim(A) == 2:
        # If we only have one set of thickness coefficients, just repeat them
        A_q = np.tile(np.expand_dims(A,0),(nq,1,1))
    else:
        # Otherwise Interpolate thicknesses to desired spans
        A_q = interp1d(spf_A,A,axis=0)(spf_q)

    sec_xrt = np.stack([evaluate_section_xy(chii , Ai) for chii, Ai in zip(chi_q, A_q) ])

    return sec_xrt
