"""Distribute points on a geometric ramp from each end onto a uniform plateau.

Vinokur stretching gives a spacing with a single smooth peak, so only a cell
or two in the middle of the interval ever reach the maximum spacing, and a
limit on that maximum is met by adding points everywhere. Here the cell
spacings are prescribed directly instead. From each end the spacing grows at a
constant expansion ratio, turns over smoothly, and then holds constant at the
maximum spacing over as many cells as the interval needs.

In the log of the spacing, the ramp from an end with spacing `dx` is

    log d(k) = log dx + b * f(k * log(ER) / b),   b = log(dmax / dx),

where `k` counts cells from that end, and the turnover `f` is linear with unit
slope up to `1 - width`, a parabola from there to `1 + width`, and exactly one
beyond. Because `0 <= f' <= 1`, no ratio between neighbouring cells exceeds
`ER`, and because `f` reaches one at a finite argument the plateau is exactly
`dmax` rather than approaching it. The spacing along the whole interval is the
smaller of the ramps from the two ends. Where the ramps cross, the step
between neighbours is bounded by a step along one ramp or the other, so the
expansion ratio limit still holds.

The expansion ratio is what absorbs the slack when the number of cells is
rounded up to a multiple: the plateau stays at `dmax` and the ramps grow more
gently. For a fixed number of cells, the total length increases monotonically
with the expansion ratio, so the ratio that gives unit length is bracketed
between one and the limit and found with a bracketing root finder.

A requested end spacing coarser than a uniform distribution of the same
number of cells cannot be met without shrinking cells towards the middle. A
uniform distribution is returned instead, which is finer than asked for at
both ends and satisfies the other limits.

A far-end spacing of None makes the clustering one-sided: there is no ramp
from that end, so the spacing grows from the start onto the plateau and stays
there, or, given more cells than that needs, ramps more gently and ends short
of the plateau. Nothing is asked of the last cell beyond the other limits.
"""

import numpy as np
from scipy.optimize import brentq

import turbigen.clusterfunc.util
from turbigen.clusterfunc.exceptions import ClusteringException

WIDTH = 0.1
"""Default half-width of the turnover, as a fraction of each ramp's length."""

NCELL_MAX = 4096
"""Largest number of cells a free distribution may use before giving up."""

NSTEP_ENDS = 8
"""Multiples to step past the first count at which the ends are reachable.

Reachable at the ratio limit is necessary but not sufficient, because the
ratio is then lowered to fit. More cells lower it further, so once the ends
are missed they are usually missed from then on, and a long scan only delays
the exception.
"""

# Relative tolerance for treating an end spacing as equal to the plateau
RTOL_EQUAL = 1e-9

# Keep the solved expansion ratio this far inside the limit, in the log, so
# that rounding in the cell spacings cannot carry a realised ratio past it
MARGIN_LOG_ER = 1e-9


def _turnover(t, width):
    """Linear, then a parabola, then one; zero at zero, with slope in [0, 1]."""
    t0 = 1.0 - width
    f = np.where(t <= t0, t, t - (t - t0) ** 2 / (4.0 * width))
    return np.where(t >= 1.0 + width, 1.0, np.minimum(f, 1.0))


def _log_ramp(dx, dmax, log_ER, k, width):
    """Log of the ramp spacing `k` cells from an end with spacing `dx`."""
    b = np.log(dmax / dx)
    if b <= 0.0:
        # End spacing equal to the plateau, nothing to grow
        return np.full(k.shape, np.log(dx))
    if log_ER == 0.0:
        return np.full(k.shape, np.log(dx))
    return np.log(dx) + b * _turnover(k * log_ER / b, width)


def _log_spacings(dx0, dx1, dmax, log_ER, M, width):
    """Log cell spacings for `M` cells: the smaller ramp from either end."""
    k = np.arange(M)
    log_d = _log_ramp(dx0, dmax, log_ER, k, width)
    if dx1 is None:
        return log_d
    return np.minimum(log_d, _log_ramp(dx1, dmax, log_ER, k[::-1], width))


def _dx_end_min(dx0, dx1):
    """The finer of the two end spacings, or the start one if one-sided."""
    return dx0 if dx1 is None else min(dx0, dx1)


def _length(dx0, dx1, dmax, log_ER, M, width):
    """Total length of `M` cells ramping at the given expansion ratio."""
    return np.exp(_log_spacings(dx0, dx1, dmax, log_ER, M, width)).sum()


def _ends_met(dx0, dx1, dmax, log_ER, M, width):
    """Do the end cells take the requested spacings, not the far ramp's?

    Where one end is much coarser than the other, the ramp from the fine end
    may not have grown to the coarse spacing by the time it gets there, and
    the coarse end would be left with a finer cell than it asked for.
    """
    log_d = _log_spacings(dx0, dx1, dmax, log_ER, M, width)
    start = np.isclose(log_d[0], np.log(dx0), rtol=0.0, atol=RTOL_EQUAL)
    if dx1 is None:
        return start
    return start and np.isclose(log_d[-1], np.log(dx1), rtol=0.0, atol=RTOL_EQUAL)


def _validate(dx0, dx1, dmax, ERmax, width):
    """Check the arguments, and return the end spacings clipped to the plateau.

    A `dx1` of None, for one-sided clustering, is passed through as None.
    """
    ends = (("dx0", dx0),) if dx1 is None else (("dx0", dx0), ("dx1", dx1))
    for name, val in ends + (("dmax", dmax),):
        if not (np.isfinite(val) and val > 0.0):
            raise ClusteringException(f"{name}={val} should be finite and > 0.")

    if not (np.isfinite(ERmax) and ERmax > 1.0):
        raise ClusteringException(f"ERmax={ERmax} should be finite and > 1.")

    if not (0.0 < width <= 1.0):
        raise ClusteringException(f"width={width} should be in (0, 1].")

    # An end spacing equal to the plateau to within rounding is taken as equal,
    # so that a caller passing the same number twice is not refused
    out = []
    for name, val in ends:
        if val > dmax * (1.0 + RTOL_EQUAL):
            raise ClusteringException(
                f"End spacing {name}={val} exceeds the maximum spacing dmax={dmax}."
            )
        out.append(min(val, dmax))

    return out if dx1 is not None else [out[0], None]


def _uniform(M):
    return np.linspace(0.0, 1.0, M + 1)


def _solve(dx0, dx1, dmax, log_ER_max, M, width):
    """Return the unit distribution of `M` cells, or None if the ends are not met.

    Assumes the length at the maximum expansion ratio reaches one and the
    length at unit expansion ratio does not.
    """
    log_ER = brentq(
        lambda g: _length(dx0, dx1, dmax, g, M, width) - 1.0,
        0.0,
        log_ER_max,
        xtol=1e-15,
        rtol=4.0 * np.finfo(float).eps,
        maxiter=200,
    )

    if not _ends_met(dx0, dx1, dmax, log_ER, M, width):
        return None

    dx = np.exp(_log_spacings(dx0, dx1, dmax, log_ER, M, width))
    x = turbigen.clusterfunc.util.cumsum0(dx)
    x /= x[-1]
    return x


def unit_fixed(dx0, dx1, dmax, ERmax, N, width=WIDTH):
    """Plateau clustering on the unit interval with `N` points.

    Parameters
    ----------
    dx0, dx1 : float
        Spacings at zero and one; `dx1` None for one-sided clustering.
    dmax : float
        Plateau spacing, which no cell exceeds.
    ERmax : float
        Expansion ratio limit, > 1.
    N : int
        Number of points.
    width : float
        Turnover half-width as a fraction of each ramp's length, in (0, 1].

    Returns
    -------
    x : array
        Grid vector of `N` points from zero to one.

    Raises
    ------
    ClusteringException
        If `N` points cannot reach unit length within the limits, or if the
        ramp from the finer end cannot grow to the coarser end's spacing.
    """
    if not isinstance(N, (int, np.integer)) or N < 2:
        raise ClusteringException(f"Need an integer N >= 2 points, got N={N}.")
    dx0, dx1 = _validate(dx0, dx1, dmax, ERmax, width)

    M = int(N) - 1

    # Ends coarser than uniform: uniform is finer at both ends, within limits
    if M * _dx_end_min(dx0, dx1) >= 1.0:
        return _uniform(M)

    log_ER_max = np.log(ERmax) * (1.0 - MARGIN_LOG_ER)
    if _length(dx0, dx1, dmax, log_ER_max, M, width) < 1.0:
        raise ClusteringException(
            f"Not enough points N={N} to reach unit length with dx0={dx0}, "
            f"dx1={dx1}, dmax={dmax}, ERmax={ERmax}."
        )

    x = _solve(dx0, dx1, dmax, log_ER_max, M, width)
    if x is None:
        raise ClusteringException(
            f"With N={N} points the ramp from the finer end cannot grow to the "
            f"coarser end spacing, dx0={dx0}, dx1={dx1}, ERmax={ERmax}."
        )

    return x


def unit_free(dx0, dx1, dmax, ERmax, mult=8, width=WIDTH, ncell_max=NCELL_MAX):
    """Plateau clustering on the unit interval with the fewest points.

    The number of cells is the smallest multiple of `mult` for which a
    distribution meets both end spacings, the maximum spacing and the
    expansion ratio limit.

    Parameters
    ----------
    dx0, dx1 : float
        Spacings at zero and one; `dx1` None for one-sided clustering.
    dmax : float
        Plateau spacing, which no cell exceeds.
    ERmax : float
        Expansion ratio limit, > 1.
    mult : int
        Number of cells is a multiple of this.
    width : float
        Turnover half-width as a fraction of each ramp's length, in (0, 1].
    ncell_max : int
        Give up rather than use more cells than this.

    Returns
    -------
    x : array
        Grid vector from zero to one.

    Raises
    ------
    ClusteringException
        If no number of cells up to `ncell_max` meets the limits.
    """
    if not isinstance(mult, (int, np.integer)) or mult < 1:
        raise ClusteringException(f"Need an integer mult >= 1, got mult={mult}.")
    dx0, dx1 = _validate(dx0, dx1, dmax, ERmax, width)

    log_ER_max = np.log(ERmax) * (1.0 - MARGIN_LOG_ER)
    n_max = ncell_max // mult
    if n_max < 1:
        raise ClusteringException(f"ncell_max={ncell_max} is less than mult={mult}.")

    def _long_enough(n):
        return _length(dx0, dx1, dmax, log_ER_max, n * mult, width) >= 1.0

    def _ends_reachable(n):
        # Necessary for the ends at the solved ratio, which is no larger
        return _ends_met(dx0, dx1, dmax, log_ER_max, n * mult, width)

    def _first(pred, n_lo):
        """Smallest n in (n_lo, n_max] satisfying a predicate monotone in n."""
        n_hi = n_max
        while n_hi - n_lo > 1:
            n_mid = (n_lo + n_hi) // 2
            if pred(n_mid):
                n_hi = n_mid
            else:
                n_lo = n_mid
        return n_hi

    # Length and end reach at the ratio limit both grow with the number of
    # cells, so each can be bisected for
    if not _long_enough(n_max):
        raise ClusteringException(
            f"Could not reach unit length within ncell_max={ncell_max} cells "
            f"with dx0={dx0}, dx1={dx1}, dmax={dmax}, ERmax={ERmax}."
        )
    n_len = _first(_long_enough, 0)

    # Ends coarser than uniform at the fewest cells that reach unit length:
    # uniform is finer at both ends, and within the other limits
    if n_len * mult * _dx_end_min(dx0, dx1) >= 1.0:
        return _uniform(n_len * mult)

    if not _ends_reachable(n_max):
        raise ClusteringException(
            f"The ramp from the finer end cannot grow to the coarser end "
            f"spacing within ncell_max={ncell_max} cells, dx0={dx0}, "
            f"dx1={dx1}, dmax={dmax}, ERmax={ERmax}."
        )
    n_start = _first(lambda n: _long_enough(n) and _ends_reachable(n), n_len - 1)

    # At the solved ratio the ramps grow less than at the limit, so the ends
    # may still be missed; step up a little until they are met. Stop before the
    # counts where uniform would do, as a uniform grid many times finer than
    # asked for at the ends is not what a free distribution should return.
    for n in range(n_start, min(n_start + NSTEP_ENDS, n_max + 1)):
        M = n * mult
        if M * _dx_end_min(dx0, dx1) >= 1.0:
            break
        x = _solve(dx0, dx1, dmax, log_ER_max, M, width)
        if x is not None:
            return x

    raise ClusteringException(
        f"The ramp from the finer end cannot grow to the coarser end spacing "
        f"at any number of cells, dx0={dx0}, dx1={dx1}, dmax={dmax}, "
        f"ERmax={ERmax}."
    )


def symmetrise(x):
    """Make a unit distribution exactly symmetric about one half."""
    xs = 0.5 * (x + (1.0 - x[::-1]))
    xs[0] = 0.0
    xs[-1] = 1.0
    return xs
