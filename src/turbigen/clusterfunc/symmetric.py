"""Distribute points with symmetric clustering."""

import numpy as np

import turbigen.clusterfunc.check
import turbigen.clusterfunc.double
import turbigen.clusterfunc.plateau
import turbigen.clusterfunc.util


def fixed(dmin, N, x0=0.0, x1=1.0):
    """Double-sided clustering between two values with with fixed number of points.

    Generate a grid vector x of length N, defaulting to the unit interval. Use
    Vinokur stretching from specified minimum end spacing. Expansion ratio and
    maximum spacing are not controlled.

    Parameters
    ----------
    dmin float
        Boundary spacing.
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
    return turbigen.clusterfunc.double.fixed(dmin, dmin, N, x0, x1)


def free(dmin, dmax, ERmax, x0=0.0, x1=1.0, mult=8):
    """Symmetric clustering between two values with with fixed number of points.

    Generate a grid vector x, by default over the unit interval. Use Vinokur
    stretching from specified minimum spacing at both ends. Increase the number
    of points until maximum spacing and expansion ratio criteria are satisfied.

    Parameters
    ----------
    dmin float
        Boundary spacing at both ends.
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
    return turbigen.clusterfunc.double.free(dmin, dmin, dmax, ERmax, x0, x1, mult)


def plateau_fixed(dmin, dmax, ERmax, N, x0=0.0, x1=1.0, width=None):
    """Symmetric plateau clustering with fixed number of points.

    As :func:`turbigen.clusterfunc.double.plateau_fixed` with the same spacing
    at both ends, and exactly symmetric about the midpoint.

    Parameters
    ----------
    dmin float
        Boundary spacing at both ends.
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
        Turnover half-width as a fraction of each ramp's length, in (0, 1].

    Returns
    -------
    x: array
        Grid vector of clustered points.

    """
    Dxa = np.abs(turbigen.clusterfunc.double._interval(x0, x1))
    xu = turbigen.clusterfunc.double.plateau_fixed(
        dmin / Dxa, dmin / Dxa, dmax / Dxa, ERmax, N, 0.0, 1.0, width
    )
    return _scale(turbigen.clusterfunc.plateau.symmetrise(xu), x0, x1)


def plateau_free(dmin, dmax, ERmax, x0=0.0, x1=1.0, mult=8, width=None):
    """Symmetric plateau clustering with free number of points.

    As :func:`turbigen.clusterfunc.double.plateau_free` with the same spacing
    at both ends, and exactly symmetric about the midpoint.

    Parameters
    ----------
    dmin float
        Boundary spacing at both ends.
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
        Turnover half-width as a fraction of each ramp's length, in (0, 1].

    Returns
    -------
    x: array
        Grid vector of clustered points.

    """
    Dxa = np.abs(turbigen.clusterfunc.double._interval(x0, x1))
    xu = turbigen.clusterfunc.double.plateau_free(
        dmin / Dxa, dmin / Dxa, dmax / Dxa, ERmax, 0.0, 1.0, mult, width
    )
    return _scale(turbigen.clusterfunc.plateau.symmetrise(xu), x0, x1)


def _scale(xu, x0, x1):
    """Map a unit distribution onto the interval from x0 to x1."""
    return x0 + (x1 - x0) * xu
