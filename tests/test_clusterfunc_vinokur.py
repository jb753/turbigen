"""Single-sided Vinokur clustering, ``turbigen.clusterfunc.single.vinokur``.

Only the start spacing is prescribed, so the properties pinned here are the
ones a caller relies on: the ends are met, the start spacing is met, the
points advance, and a spacing coarser than uniform is as good an input as a
finer one. That last is what makes it safe where :func:`single.fixed` raises
for want of points.
"""

import numpy as np
import pytest

from turbigen.clusterfunc import single
from turbigen.clusterfunc.exceptions import ClusteringException

CASES = [
    (N, dmin)
    for N in (5, 9, 17)
    for dmin in (1e-3, 0.0266, 0.08, 1.0 / (N - 1), 0.2, 0.3)
]


@pytest.mark.parametrize("N, dmin", CASES)
def test_meets_the_ends_and_the_start_spacing(N, dmin):
    x = single.vinokur(dmin, N)
    dx = np.diff(x)
    assert len(x) == N
    assert x[0] == pytest.approx(0.0, abs=1e-12)
    assert x[-1] == pytest.approx(1.0, abs=1e-12)
    assert (dx > 0.0).all()
    assert dx[0] == pytest.approx(dmin, rel=1e-2)


@pytest.mark.parametrize("N, dmin", CASES)
def test_spacing_varies_one_way(N, dmin):
    """Finer than uniform grows towards the end, coarser shrinks."""
    ER = np.diff(single.vinokur(dmin, N))
    ER = ER[1:] / ER[:-1]
    if dmin < 1.0 / (N - 1):
        assert (ER >= 1.0 - 1e-9).all()
    else:
        assert (ER <= 1.0 + 1e-9).all()


def test_uniform_start_spacing_is_uniform():
    np.testing.assert_allclose(single.vinokur(0.125, 9), np.linspace(0.0, 1.0, 9))


def test_scales_onto_a_reversed_interval():
    """Spacing is a length, so the start spacing holds running downwards."""
    x = single.vinokur(0.01, 9, 2.0, -1.0)
    assert x[0] == pytest.approx(2.0)
    assert x[-1] == pytest.approx(-1.0)
    assert x[0] - x[1] == pytest.approx(0.01, rel=1e-2)


def test_needs_no_cap_where_geometric_clustering_runs_out():
    """The case that sent `add_cusp` to its fallback on one side only.

    Nine points at an expansion ratio of 1.4 cover at most 34.4 start spacings,
    and this needs 37.5.
    """
    with pytest.raises(ClusteringException):
        single.fixed(1.0 / 37.5, 1.0, 1.4, 9)
    x = single.vinokur(1.0 / 37.5, 9)
    assert np.diff(x)[0] == pytest.approx(1.0 / 37.5, rel=1e-2)


@pytest.mark.parametrize("dmin", [0.0, 1.0, 1.5])
def test_refuses_a_start_spacing_outside_the_interval(dmin):
    with pytest.raises(ClusteringException):
        single.vinokur(dmin, 9)
