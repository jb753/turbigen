"""Distribute points with symmetric clustering."""

import turbigen.clusterfunc.check
import turbigen.clusterfunc.util
import turbigen.clusterfunc.double


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
