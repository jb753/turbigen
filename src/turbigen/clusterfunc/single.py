"""Distribute points with single-sided clustering."""

import numpy as np
from scipy.optimize import root_scalar

import turbigen.clusterfunc.check
import turbigen.clusterfunc.double
import turbigen.clusterfunc.plateau
import turbigen.clusterfunc.util
from turbigen.clusterfunc.exceptions import ClusteringException


def fixed(dmin, dmax, ERmax, N, x0=0.0, x1=1.0, check=True):
    """Single-sided clustering with fixed number of points.

    Generate a grid vector x of length N, by default over the unit interval.
    Use geometric stretching from a minimum start spacing, with spacings capped
    to a maximum value.

    Parameters
    ----------
    dmin: float
        Minimum spacing at start.
    dmax: float
        Maximum spacing.
    ERmax: float
        Expansion ratio > 1.
    x0: float
        Start value.
    x1: float
        End value.
    N: int
        Number of points in the grid vector.

    Returns
    -------
    x: array
        Grid vector of clustered points.

    """

    assert isinstance(N, int)

    if np.isclose(x0, x1):
        raise ClusteringException(
            "Cannot distribute points without distinct start and end points, "
            f"got x0={x0} and x1={x1}"
        )

    # Scale the unit distribution between the given values
    Dx = x1 - x0
    Dxa = np.abs(Dx)
    x = x0 + Dx * _unit_fixed(dmin / Dxa, dmax / Dxa, ERmax, N, check)

    return x


def free(dmin, dmax, ERmax, x0=0.0, x1=1.0, mult=8):
    """Single-sided clustering with free number of points.

    Generate a grid vector x, by default over the unit interval. Use geometric
    stretching from a minimum start spacing, with spacings capped to a maximum
    value. Use the minimum number of points required to satsify the
    constraints.

    Parameters
    ----------
    dmin: float
        Minimum spacing at zero.
    dmax: float
        Maximum spacing.
    ERmax: float
        Expansion ratio > 1.
    x0: float
        Start value.
    x1: float
        End value.
    mult: int
        Choose a number of cells divisible by this factor.

    Returns
    -------
    x: array
        Grid vector of clustered points.

    """

    assert isinstance(mult, int)

    if np.isclose(x0, x1):
        raise ClusteringException(
            "Cannot distribute points without distinct start and end points,  "
            f"got x0={x0} and x1={x1}"
        )

    # Scale the unit distribution between the given values
    Dx = x1 - x0
    Dxa = np.abs(Dx)
    x = x0 + Dx * _unit_free(dmin / Dxa, dmax / Dxa, ERmax, mult)

    return x


def plateau_fixed(dmin, dmax, ERmax, N, x0=0.0, x1=1.0, width=None):
    """Single-sided plateau clustering with fixed number of points.

    Generate a grid vector x of length N, by default over the unit interval.
    From the specified spacing at x0, grow geometrically and turn over onto a
    uniform plateau at the maximum spacing, which holds to x1. The expansion
    ratio is lowered from its limit to fit the interval exactly, so with more
    points than the plateau needs the ramp ends short of it. See
    :mod:`turbigen.clusterfunc.plateau`.

    Parameters
    ----------
    dmin: float
        Boundary spacing at x0.
    dmax: float
        Plateau spacing, which no cell exceeds.
    ERmax: float
        Expansion ratio limit > 1.
    N: int
        Number of points in the grid vector.
    x0: float
        Start value.
    x1: float
        End value.
    width: float
        Turnover half-width as a fraction of the ramp's length, in (0, 1].

    Returns
    -------
    x: array
        Grid vector of clustered points.

    """
    Dx = _interval(x0, x1)
    Dxa = np.abs(Dx)
    kwargs = {} if width is None else {"width": width}
    return x0 + Dx * turbigen.clusterfunc.plateau.unit_fixed(
        dmin / Dxa, None, dmax / Dxa, ERmax, N, **kwargs
    )


def plateau_free(dmin, dmax, ERmax, x0=0.0, x1=1.0, mult=8, width=None):
    """Single-sided plateau clustering with free number of points.

    Generate a grid vector x from x0 to x1, with the fewest points, in a
    multiple of `mult` cells, that meet the start spacing, the maximum spacing
    and the expansion ratio limit. From x0 the spacing grows geometrically and
    turns over onto a uniform plateau at the maximum spacing, which holds to
    x1. See :mod:`turbigen.clusterfunc.plateau`.

    Parameters
    ----------
    dmin: float
        Boundary spacing at x0.
    dmax: float
        Plateau spacing, which no cell exceeds.
    ERmax: float
        Expansion ratio limit > 1.
    x0: float
        Start value.
    x1: float
        End value.
    mult: int
        Choose a number of cells divisible by this factor.
    width: float
        Turnover half-width as a fraction of the ramp's length, in (0, 1].

    Returns
    -------
    x: array
        Grid vector of clustered points.

    """
    Dx = _interval(x0, x1)
    Dxa = np.abs(Dx)
    kwargs = {} if width is None else {"width": width}
    return x0 + Dx * turbigen.clusterfunc.plateau.unit_free(
        dmin / Dxa, None, dmax / Dxa, ERmax, mult, **kwargs
    )


def vinokur(dmin, N, x0=0.0, x1=1.0):
    """Single-sided clustering with fixed number of points.

    Generate a grid vector x of length N, by default over the unit interval.
    Use Vinokur stretching from the specified spacing at x0, leaving the
    spacing at x1 free. Expansion ratio and maximum spacing are not controlled,
    so unlike :func:`fixed` this cannot fail for want of points: a start
    spacing finer than uniform grows towards x1, and a coarser one shrinks.

    Parameters
    ----------
    dmin: float
        Boundary spacing at x0.
    N: int
        Number of points in the grid vector.
    x0: float
        Start value.
    x1: float
        End value.

    Returns
    -------
    x: array
        Grid vector of clustered points.

    """
    assert isinstance(N, int)
    Dx = _interval(x0, x1)
    return x0 + Dx * _unit_vinokur(dmin / np.abs(Dx), N)


def _vinokur(ds, N):
    """One sided analytic clustering function after Vinokur.

    The one-sided case of the two-sided function in
    :func:`turbigen.clusterfunc.double._vinokur`: the start slope of `u` is
    `1/B`, which fixes the start spacing, and nothing constrains the end.
    """

    B = 1.0 / (N - 1) / ds

    xi = np.linspace(0.0, 1.0, N)

    if np.abs(B - 1.0) < 0.001:
        u = xi
    elif B < 1.0:
        # sin(D)/D = B
        D = turbigen.clusterfunc.double._invert_sinx_x(B)
        assert np.isclose(np.sin(D) / D, B, rtol=1e-1)
        u = 1.0 + np.tan(0.5 * D * (xi - 1.0)) / np.tan(0.5 * D)
    else:
        # sinh(D)/D = B
        D = turbigen.clusterfunc.double._invert_sinhx_x(B)
        assert np.isclose(np.sinh(D) / D, B, rtol=1e-1)
        u = 1.0 + np.tanh(0.5 * D * (xi - 1.0)) / np.tanh(0.5 * D)

    # Force to unit interval
    u -= u[0]
    u /= u[-1]

    return u


def _unit_vinokur(dmin, N, rtol=1e-2):
    """Single-sided Vinokur clustering on the unit interval, start spacing met.

    The inverse of sinh(x)/x is approximate and the discrete first spacing is
    not the continuous slope, so the target is corrected until the first
    spacing comes out within `rtol`, as :func:`turbigen.clusterfunc.double.
    _unit_fixed` does for both ends.
    """
    if N < 2:
        raise ClusteringException(f"Need at least two points to cluster, N={N}")
    if not 0.0 < dmin < 1.0:
        raise ClusteringException(f"dmin={dmin} should be in (0, 1)")
    maxiter = 100
    ds_in = dmin
    for _ in range(maxiter):
        x = _vinokur(ds_in, N)
        fac = (x[1] - x[0]) / dmin
        if np.abs(fac - 1.0) < rtol:
            break
        ds_in /= fac
    return x


def _interval(x0, x1):
    """Return the signed length of the interval, refusing an empty one."""
    if np.isclose(x0, x1):
        raise ClusteringException(
            "Cannot distribute points without distinct start and end points, "
            f"got x0={x0} and x1={x1}"
        )
    return x1 - x0


def _unit_fixed(dmin, dmax, ERmax, N, check=True):
    """Single-sided clustering on the unit interval with specified number of points."""

    if N < 2:
        raise ClusteringException(f"Need at least two points to cluster, N={N}")

    if dmax <= dmin:
        raise ClusteringException(f"dmax={dmax} should be > dmin={dmin}")

    if ERmax <= 1:
        raise ClusteringException(f"ERmax={ERmax} should be > 1")

    # Rule out having too many points for a uniform spacing at dmin
    Nunif = np.ceil(1.0 / dmin).astype(int) + 1
    if N > Nunif:
        raise ClusteringException(
            f"Too many points N={N} for a uniform spacing of dmin={dmin}."
        )

    # Rule out too few points to reach unit length, regardless of capping
    Nfull = np.ceil(np.log(dmax / dmin) / np.log(ERmax)).astype(int) + 1
    Lfull = dmin * (1.0 - ERmax ** (Nfull - 1)) / (1.0 - ERmax)
    if N < Nfull and Lfull < 1.0:
        raise ClusteringException(
            f"Not enough points N={N} to reach unit length for "
            f"ERmax={ERmax}, dmin={dmin}, regardless of dmax., "
            f"L={Lfull}, Nfull={Nfull}"
        )

    # Assume we do not need to cap the spacing
    # What expansion ratio is needed, and what is the max spacing?
    Nm1, Nm2 = N - 1, N - 2
    Np1 = N + 1

    def _iter_ER(ERi):
        y = dmin * (1.0 - ERi**Nm1) / (1.0 - ERi) - 1.0
        dy = dmin * (Nm2 * ERi**Np1 - Nm1 * ERi**N + ERi**2) / (1.0 - ERi) ** 2 / ERi**2
        return y, dy

    # If we cannot reach irregardless of capping, give up
    if _iter_ER(ERmax)[0] < 0.0:
        raise ClusteringException(
            f"Insufficient points N={N} to reach unit length with dmin={dmin}, "
            f"ERmax={ERmax}, no capping to dmax needed."
        )

    # Check for a uniform solution first
    err_unif = 1.0 / (N - 1) - dmin
    rtol = 1e-9
    if np.abs(err_unif) < rtol:
        ERnocap = 1.0
    # Otherwise, find ER root
    else:
        soln = root_scalar(_iter_ER, x0=ERmax, fprime=True, maxiter=1000)
        assert soln.converged
        ERnocap = soln.root

    Dnocap = dmin * ERnocap**Nm2

    # If capping constraint is satisfied, but ER is too big, we don't have enough points
    if ERnocap > ERmax and Dnocap <= dmax:
        raise ClusteringException(
            f"Insufficient points to reach unit length with dmin={dmin}, "
            f"ERmax={ERmax}, no capping to dmax needed."
        )
    # If ERnocap and Dnocap satisfy our constraints, we have the solution
    elif ERnocap <= ERmax and Dnocap <= dmax:
        dx = dmin * ERnocap ** np.arange(Nm1)
    # We need to have a capped section
    else:
        # Length as a function of spacing cap, with ERmax constant
        def _iter_dmax(ERi, dmaxi):
            Nstretch_i = np.floor(np.log(dmaxi / dmin) / np.log(ERi)).astype(int) + 1
            Lstretch_i = dmin * (1.0 - ERi ** (Nstretch_i - 1)) / (1.0 - ERi)
            dx1max_i = dmin * ERi ** (Nstretch_i - 2)
            Ncap_i = N - Nstretch_i
            Lcap_i = Ncap_i * 0.5 * (dx1max_i + dmaxi)
            return Lstretch_i + Lcap_i - 1.0

        # If we use all points at max ER and don't have enough length, will not work
        LL = _iter_dmax(ERmax, dmax)
        if LL < 0.0:
            raise ClusteringException(
                "Not enough points to cluster with "
                f"dmin={dmin}, ERmax={ERmax}, dmax={dmax} capping, "
                f"total length only {LL + 1}."
            )

        # Binary search for cap distance that gives closes to unit length
        dmax_high = dmax
        dmax_low = dmin
        while dmax_high - dmax_low > 0.01 * dmax:
            dmax_new = 0.5 * (dmax_high + dmax_low)
            err_new = _iter_dmax(ERmax, dmax_new)
            if err_new > 0:
                dmax_high = dmax_new
            else:
                dmax_low = dmax_new

        # Fine-tune the expansion ratio for exactly unit length
        soln = root_scalar(_iter_dmax, bracket=(1.0 + 1e-9, ERmax), args=(dmax_high,))
        assert soln.converged
        ER = soln.root

        # Evaluate the clustering
        Nstretch = np.floor(np.log(dmax_high / dmin) / np.log(ER)).astype(int) + 1
        Ncap = N - Nstretch
        dx1 = dmin * ER ** np.arange(0, Nstretch - 1)
        dx2 = np.linspace(dx1[-1], dmax_high, Ncap)
        dx = np.concatenate((dx1, dx2))

    x = turbigen.clusterfunc.util.cumsum0(dx)

    x /= x[-1]

    if not len(x) == N:
        raise ClusteringException(
            f"Incorrect number of points len(x)={len(x)}, expected {N}."
        )

    if check:
        turbigen.clusterfunc.check.unit_single(x, dmin, dmax, ERmax)

    return x


def _unit_free(dmin, dmax, ERmax, mult=8):
    """Single-sided clustering on the unit interval with free number of points."""

    # Find high and low guesses for n, N = mult*n + 1
    nlow = np.round(1.0 / dmax / mult).astype(int)
    nhigh = np.minimum(
        np.floor(1.0 / dmin / mult).astype(int), 1024 // mult
    )  # Limit to protect against overflow
    if nlow > nhigh:
        nlow = 1
    if nhigh == nlow:
        raise ClusteringException(
            f"Unable to find distinct guesses for numbers of points with mult={mult}, "
            "try decreasing mult"
        )

    Nhigh = nhigh * mult + 1
    try:
        x = _unit_fixed(dmin, dmax, ERmax, Nhigh, check=True)
    except ClusteringException:
        raise ClusteringException(
            f"Failed to cluster with high number of points guess Nhigh={Nhigh}"
        )

    # Loop until converged
    while nhigh - nlow > 1:
        # Bisect high and low guesses
        nnew = (nhigh + nlow) // 2
        Nnew = nnew * mult + 1

        # Attempt to cluster
        try:
            x = _unit_fixed(dmin, dmax, ERmax, Nnew, check=True)
            # New n is valid, use as high limiit
            nhigh = nnew
        except ClusteringException:
            # New n is not valid, use as low limiit
            nlow = nnew

    return x
