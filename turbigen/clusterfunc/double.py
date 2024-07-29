"""Distribute points with symmetric clustering."""

import numpy as np
from turbigen.clusterfunc.exceptions import ClusteringException
import turbigen.clusterfunc.check
import turbigen.clusterfunc.util
import turbigen.clusterfunc.single

import warnings


def fixed(dx0, dx1, N, x0=0.0, x1=1.0):
    """Double-sided clustering with fixed number of points.

    Generate a grid vector x of length N, by default over the unit interval.
    Use Vinokur stretching from specified spacings at x0 and x1. Expansion
    ratio and maximum spacing are not controlled.

    Parameters
    ----------
    dx0 float
        Boundary spacing at x0.
    dx1 float
        Boundary spacing at x1.
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

    if np.isclose(x0, x1):
        raise ClusteringException(
            "Cannot distribute points without distinct start and end points, "
            f"got x0={x0} and x1={x1}"
        )

    # Scale the unit distribution between the given values
    Dx = x1 - x0
    Dxa = np.abs(Dx)
    x = x0 + Dx * _unit_fixed(dx0 / Dxa, dx1 / Dxa, N)

    return x


def free(dx0, dx1, dmax, ERmax, x0=0.0, x1=1.0, mult=8):
    """Double-sided clustering between two values with with fixed number of points.

    Generate a grid vector x from x0 to x1. Use Vinokur stretching from
    specified spacings at x0 and x1. Increase the number of points until
    maximum spacing and expansion ratio criteria are satisfied.

    Parameters
    ----------
    dx0 float
        Boundary spacing at x0.
    dx1 float
        Boundary spacing at x1.
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
            "Cannot distribute points without distinct start and end points, "
            f"got x0={x0} and x1={x1}"
        )

    # Scale the unit distribution between the given values
    Dx = x1 - x0
    Dxa = np.abs(Dx)
    x = x0 + Dx * _unit_free(dx0 / Dxa, dx1 / Dxa, dmax / Dxa, ERmax, mult)

    return x


def _vinokur(ds, N):
    """Two sided analytic clustering function after Vinokur."""

    s0 = 1.0 / N / ds[0]
    s1 = N * ds[1]

    A = np.sqrt(s0 * s1)
    B = np.sqrt(s0 / s1)

    xi = np.linspace(0.0, 1.0, N)

    if np.abs(B - 1.0) < 0.001:
        # Eqn. (52)
        u = xi * (1.0 + 2.0 * (B - 1.0) * (xi - 0.5) * (1.0 - xi))
    elif B < 1.0:
        # Solve Eqn. (49)
        Dx = _invert_sinx_x(B)
        assert np.isclose(np.sin(Dx) / Dx, B, rtol=1e-1)
        # Eqn. (50)
        u = 0.5 * (1.0 + np.tan(Dx * (xi - 0.5)) / np.tan(Dx / 2.0))
    elif B >= 1.0:
        # Solve Eqn. (46)
        Dy = _invert_sinhx_x(B)
        assert np.isclose(np.sinh(Dy) / Dy, B, rtol=1e-1)
        # Eqn. (47)
        u = 0.5 * (1.0 + np.tanh(Dy * (xi - 0.5)) / np.tanh(Dy / 2.0))
    else:
        raise Exception(f"Unexpected B={B}, s0={s0}, s1={s1}")

    t = u / (A + (1.0 - A) * u)

    # Force to unit interval
    t -= t[0]
    t /= t[-1]

    return t


def _invert_sinhx_x(y):
    """Return solution x for y = sinh(x)/x in Eqns. (62-67)."""

    if y < 2.7829681:
        y1 = y - 1.0
        x2 = np.sqrt(6.0 * y1) * (
            1.0
            - 0.15 * y1
            + 0.057321429 * y1**2.0
            - 0.024907295 * y1**3.0
            + 0.0077424461 * y1**4.0
            - 0.0010794123 * y1**5.0
        )
    else:
        v = np.log(y)
        w = 1.0 / y - 0.028527431
        x2 = (
            v
            + (1.0 + 1.0 / v) * np.log(2.0 * v)
            - 0.02041793
            + 0.24902722 * w
            + 1.9496443 * w**2.0
            - 2.6294547 * w**3.0
            + 8.56795911 * w**4.0
        )

    # if not np.isclose(x, x2):
    #     print(x, x2, y)
    #     raise Exception('beans')

    return x2


def _invert_sinx_x(y):
    """Return solution x for y = sin(x)/x from Eqns. (68-71)."""
    if y < 0.26938972:
        x = np.pi * (
            1.0
            - y
            + y**2.0
            - (1.0 + np.pi**2.0 / 6.0) * y**3.0
            + 6.794732 * y**4.0
            - 13.205501 * y**5.0
            + 11.726095 * y**6.0
        )
    else:
        y1 = 1.0 - y
        x = np.sqrt(6.0 * y1) * (
            1.0
            + 0.15 * y1
            + 0.057321429 * y1**2.0
            + 0.048774238 * y1**3.0
            - 0.053337753 * y1**4.0
            + 0.075845134 * y1**5.0
        )

    return x


def _unit_fixed(dx0, dx1, N, rtol=1e-2):
    """General double-side clustering with fixed number of points"""
    maxiter = 100
    dx = np.array([dx0, dx1])
    dx_in = np.copy(dx)
    for _ in range(maxiter):
        x = _vinokur(dx_in, N)
        dx_out = np.diff(x)[
            (0, -1),
        ]
        fac_dx = dx_out / dx
        if (np.abs(fac_dx - 1.0) < rtol).all():
            break
        else:
            dx_in /= fac_dx
    return x


def _unit_free(dx0, dx1, dmax, ERmax, mult=8, rtol=1e-2):
    """General double-side clustering with free number of points"""
    n = 1
    maxiter = 1000
    flag = False
    with warnings.catch_warnings():
        warnings.filterwarnings("error")
        for _ in range(maxiter):
            N = mult * n + 1
            if N < 4:
                n += 1
                continue
            try:
                x = _unit_fixed(dx0, dx1, N, rtol)
            except RuntimeWarning:
                n += 1
                continue
            dx = np.diff(x)
            ER = turbigen.clusterfunc.util.ER(x)
            if (ER < ERmax).all() and (dx <= dmax).all():
                flag = True
                break
            else:
                n += 1

    if not flag:
        raise ClusteringException(
            f"Could not double cluster with dx0={dx0}, dx1={dx1}, dmax={dmax},"
            f" ERmax={ERmax}"
        )

    return x
