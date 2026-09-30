"""Plateau clustering: geometric ramps from each end onto a uniform plateau."""

import itertools

import numpy as np
import pytest

from turbigen.clusterfunc import double, plateau, symmetric
from turbigen.clusterfunc.exceptions import ClusteringException
from turbigen.clusterfunc.util import ER

RTOL = 1e-9


def check_unit(x, dx0, dx1, dmax, ERmax, mult=None):
    """Assert a unit distribution obeys the limits, allowing the uniform fallback."""
    dx = np.diff(x)
    M = len(dx)

    assert x[0] == 0.0
    assert x[-1] == 1.0
    assert (dx > 0.0).all()
    if mult is not None:
        assert M % mult == 0

    # Strict, because the mesher asserts it strictly
    if M > 1:
        assert ER(x).max() <= ERmax
    assert dx.max() <= dmax * (1.0 + RTOL)

    if M * min(dx0, dx1) >= 1.0:
        # Uniform, and so finer than asked for at both ends
        assert np.allclose(dx, 1.0 / M, rtol=RTOL)
    else:
        assert dx[0] == pytest.approx(dx0, rel=RTOL)
        assert dx[-1] == pytest.approx(dx1, rel=RTOL)


#
# Symmetric, free number of points
#

SYM_GRID = [
    (dmin, dmax, ERmax, mult, width)
    for dmin, dmax, ERmax, mult, width in itertools.product(
        [1e-5, 1e-4, 1e-3, 5e-3, 2e-2],
        [0.02, 0.05, 0.2, 0.6, 2.0],
        [1.05, 1.2, 1.5],
        [1, 3, 8],
        [0.1, 1.0],
    )
    if dmin <= dmax
]


@pytest.mark.parametrize("dmin, dmax, ERmax, mult, width", SYM_GRID)
def test_symmetric_free_limits(dmin, dmax, ERmax, mult, width):
    x = symmetric.plateau_free(dmin, dmax, ERmax, mult=mult, width=width)
    check_unit(x, dmin, dmin, dmax, ERmax, mult)

    # Exactly symmetric
    assert np.max(np.abs(x + x[::-1] - 1.0)) <= 1e-15


@pytest.mark.parametrize("dmin, dmax, ERmax, mult, width", SYM_GRID)
def test_symmetric_free_minimal(dmin, dmax, ERmax, mult, width):
    """One multiple fewer cells cannot meet the limits."""
    x = symmetric.plateau_free(dmin, dmax, ERmax, mult=mult, width=width)
    N = len(x)
    if (N - 1) * dmin >= 1.0 or N - mult < 2:
        # Uniform fallback, or nothing fewer to try
        return
    with pytest.raises(ClusteringException):
        symmetric.plateau_fixed(dmin, dmax, ERmax, N - mult, width=width)


# Wall and mid-passage spacings from the pitchwise grids of the sweep11 datum,
# at its own Reynolds number and at a surface Reynolds number of 1e6, and some
# further typical ones
HMESH_CASES = [
    (5.056e-3, 0.0584),
    (5.948e-3, 0.0692),
    (2.190e-3, 0.0584),
    (2.579e-3, 0.0692),
    (2e-3, 0.03),
    (5e-4, 0.03),
    (1e-4, 0.05),
]


@pytest.mark.parametrize("dmin, dmax", HMESH_CASES)
def test_symmetric_free_no_more_points_than_vinokur(dmin, dmax):
    x = symmetric.plateau_free(dmin, dmax, 1.2)
    x_vinokur = symmetric.free(dmin, dmax, 1.2)
    assert len(x) <= len(x_vinokur)


def test_symmetric_free_has_plateau():
    """With room to spare, many cells in the middle sit exactly at dmax."""
    dmax = 0.03
    x = symmetric.plateau_free(5e-4, dmax, 1.2)
    dx = np.diff(x)
    n_flat = np.sum(np.isclose(dx, dmax, rtol=RTOL))
    assert n_flat >= 8

    # And those cells are contiguous, in the middle
    iflat = np.flatnonzero(np.isclose(dx, dmax, rtol=RTOL))
    assert np.all(np.diff(iflat) == 1)
    assert iflat[0] > 0 and iflat[-1] < len(dx) - 1


def test_symmetric_free_spacing_monotone_to_middle():
    x = symmetric.plateau_free(1e-3, 0.05, 1.2, mult=1)
    dx = np.diff(x)
    half = len(dx) // 2
    assert (np.diff(dx[: half + 1]) >= -RTOL * dx.max()).all()
    assert (np.diff(dx[-half - 1 :]) <= RTOL * dx.max()).all()


def test_symmetric_free_pure_ramp():
    """A plateau spacing too big to reach leaves two ramps meeting in the middle."""
    x = symmetric.plateau_free(1e-3, 2.0, 1.2)
    check_unit(x, 1e-3, 1e-3, 2.0, 1.2, 8)
    assert np.diff(x).max() < 0.5


def test_symmetric_free_end_equal_to_plateau():
    """Nothing to cluster: uniform at no more than the plateau spacing."""
    x = symmetric.plateau_free(0.03, 0.03, 1.2, mult=1)
    dx = np.diff(x)
    assert np.allclose(dx, dx[0], rtol=RTOL)
    assert dx.max() <= 0.03


def test_symmetric_free_end_rounding_above_plateau():
    """An end spacing past the plateau by rounding only is taken as equal."""
    x = symmetric.plateau_free(0.03 * (1.0 + 1e-12), 0.03, 1.2, mult=1)
    assert np.diff(x).max() <= 0.03 * (1.0 + RTOL)


def test_symmetric_free_coarse_end_is_uniform():
    """End spacing coarser than the fewest allowed cells: uniform, finer ends."""
    x = symmetric.plateau_free(0.3, 0.5, 1.2, mult=8)
    assert len(x) == 9
    assert np.allclose(np.diff(x), 1.0 / 8.0)


def test_symmetric_free_scaled_interval():
    x0, x1 = 2.0, 5.0
    L = x1 - x0
    x = symmetric.plateau_free(1e-3 * L, 0.05 * L, 1.2, x0, x1)
    xu = symmetric.plateau_free(1e-3, 0.05, 1.2)
    assert x[0] == x0
    assert x[-1] == pytest.approx(x1, rel=1e-15)
    assert np.allclose(x, x0 + L * xu, rtol=1e-12)


def test_symmetric_free_reversed_interval():
    x = symmetric.plateau_free(1e-3, 0.05, 1.2, 1.0, 0.0)
    assert x[0] == 1.0 and x[-1] == 0.0
    assert (np.diff(x) < 0.0).all()


def test_symmetric_free_too_many_cells():
    with pytest.raises(ClusteringException):
        symmetric.plateau_free(1e-7, 0.05, 1.002)


#
# Symmetric, fixed number of points
#


@pytest.mark.parametrize("N", [57, 65, 81, 129])
def test_symmetric_fixed_limits(N):
    dmin, dmax, ERmax = 2e-3, 0.03, 1.2
    x = symmetric.plateau_fixed(dmin, dmax, ERmax, N)
    assert len(x) == N
    check_unit(x, dmin, dmin, dmax, ERmax)
    assert np.max(np.abs(x + x[::-1] - 1.0)) <= 1e-15


def test_symmetric_fixed_matches_free():
    x = symmetric.plateau_free(2e-3, 0.03, 1.2)
    xf = symmetric.plateau_fixed(2e-3, 0.03, 1.2, len(x))
    assert np.allclose(x, xf, rtol=0.0, atol=1e-15)


def test_symmetric_fixed_more_points_lowers_ratio():
    x_few = symmetric.plateau_fixed(2e-3, 0.03, 1.2, 57)
    x_many = symmetric.plateau_fixed(2e-3, 0.03, 1.2, 97)
    assert ER(x_many).max() < ER(x_few).max()


def test_symmetric_fixed_too_many_points_is_uniform():
    """Uniform at the requested count is already finer than the ends."""
    x = symmetric.plateau_fixed(0.05, 0.1, 1.2, 41)
    assert len(x) == 41
    assert np.allclose(np.diff(x), 1.0 / 40.0)


def test_symmetric_fixed_too_few_points():
    with pytest.raises(ClusteringException):
        symmetric.plateau_fixed(1e-3, 0.05, 1.2, 9)


def test_symmetric_fixed_scaled_interval():
    x0, x1 = -1.0, 3.0
    L = x1 - x0
    x = symmetric.plateau_fixed(1e-3 * L, 0.05 * L, 1.2, 65, x0, x1)
    xu = symmetric.plateau_fixed(1e-3, 0.05, 1.2, 65)
    assert np.allclose(x, x0 + L * xu, rtol=1e-12)


#
# Double-sided
#

DOUBLE_GRID = list(
    itertools.product(
        [1e-4, 1e-3],
        [1e-3, 1e-2, 0.03],
        [0.05, 0.2],
        [1.1, 1.3],
        [1, 8],
    )
)


@pytest.mark.parametrize("dx0, dx1, dmax, ERmax, mult", DOUBLE_GRID)
def test_double_free_limits(dx0, dx1, dmax, ERmax, mult):
    x = double.plateau_free(dx0, dx1, dmax, ERmax, mult=mult)
    check_unit(x, dx0, dx1, dmax, ERmax, mult)

    # Swapping the ends mirrors the distribution
    xs = double.plateau_free(dx1, dx0, dmax, ERmax, mult=mult)
    assert np.allclose(xs, 1.0 - x[::-1], rtol=0.0, atol=1e-12)


@pytest.mark.parametrize("dx0, dx1, dmax, ERmax, mult", DOUBLE_GRID)
def test_double_free_minimal(dx0, dx1, dmax, ERmax, mult):
    x = double.plateau_free(dx0, dx1, dmax, ERmax, mult=mult)
    N = len(x)
    if (N - 1) * min(dx0, dx1) >= 1.0 or N - mult < 2:
        return
    with pytest.raises(ClusteringException):
        double.plateau_fixed(dx0, dx1, dmax, ERmax, N - mult)


def test_double_free_equal_ends_is_symmetric():
    x = double.plateau_free(1e-3, 1e-3, 0.05, 1.2)
    xs = symmetric.plateau_free(1e-3, 0.05, 1.2)
    assert np.allclose(x, xs, rtol=0.0, atol=1e-14)


def test_double_free_one_end_at_plateau():
    """An end already at the plateau spacing is met exactly, not approached."""
    dmax = 0.05
    x = double.plateau_free(1e-3, dmax, dmax, 1.2)
    check_unit(x, 1e-3, dmax, dmax, 1.2, 8)
    assert np.diff(x)[-1] == pytest.approx(dmax, rel=RTOL)


def test_double_free_ends_unreachable():
    """A fine end ramping slowly never grows to a coarse far end."""
    with pytest.raises(ClusteringException):
        double.plateau_free(1e-4, 0.2, 0.2, 1.05)


def test_double_fixed_ends_unreachable():
    with pytest.raises(ClusteringException):
        double.plateau_fixed(1e-4, 0.2, 0.2, 1.05, 131)


def test_double_fixed_limits():
    x = double.plateau_fixed(1e-4, 1e-2, 0.05, 1.2, 81)
    assert len(x) == 81
    check_unit(x, 1e-4, 1e-2, 0.05, 1.2)


def test_double_reversed_interval():
    x = double.plateau_free(1e-3, 1e-2, 0.05, 1.2, 1.0, 0.0)
    assert x[0] == 1.0 and x[-1] == 0.0
    dx = -np.diff(x)
    assert dx[0] == pytest.approx(1e-3, rel=RTOL)
    assert dx[-1] == pytest.approx(1e-2, rel=RTOL)


#
# Bad arguments
#


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(ERmax=1.0),
        dict(ERmax=0.9),
        dict(ERmax=np.nan),
        dict(dx0=0.0),
        dict(dx0=-1e-3),
        dict(dx1=np.inf),
        dict(dmax=0.0),
        dict(dx0=0.1, dmax=0.05),
        dict(width=0.0),
        dict(width=1.5),
        dict(mult=0),
        dict(x0=1.0, x1=1.0),
    ],
)
def test_double_free_bad_arguments(kwargs):
    args = dict(dx0=1e-3, dx1=1e-3, dmax=0.05, ERmax=1.2)
    args.update(kwargs)
    with pytest.raises(ClusteringException):
        double.plateau_free(**args)


@pytest.mark.parametrize("N", [1, 0, 8.0])
def test_double_fixed_bad_count(N):
    with pytest.raises(ClusteringException):
        double.plateau_fixed(1e-3, 1e-3, 0.05, 1.2, N)


#
# The turnover itself
#


@pytest.mark.parametrize("width", [0.05, 0.1, 0.5, 1.0])
def test_turnover_shape(width):
    t = np.linspace(0.0, 3.0, 30001)
    f = plateau._turnover(t, width)
    df = np.diff(f) / np.diff(t)

    assert f[0] == 0.0
    assert (f <= 1.0).all()
    assert (f[t >= 1.0 + width] == 1.0).all()
    assert (df >= -1e-12).all()
    assert (df <= 1.0 + 1e-12).all()

    # Continuous slope: no jumps bigger than one step's worth of curvature
    assert np.abs(np.diff(df)).max() <= np.diff(t)[0] / (2.0 * width) + 1e-9
