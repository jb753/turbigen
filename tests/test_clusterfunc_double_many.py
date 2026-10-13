"""Double-sided clustering of many intervals at once matches one at a time."""

import numpy as np
import pytest

from turbigen.clusterfunc import double


@pytest.mark.parametrize("N", [5, 17, 33])
def test_many_intervals_cluster_as_each_would_alone(N):
    rng = np.random.default_rng(0)
    # End spacings from tight at a wall to near uniform, either way round.
    dx0 = 10.0 ** rng.uniform(-4.0, np.log10(0.9 / (N - 1)), 40)
    dx1 = 10.0 ** rng.uniform(-3.0, np.log10(0.9 / (N - 1)), 40)
    many = double.fixed_many(dx0, dx1, N)
    one = np.stack([double.fixed(a, b, N) for a, b in zip(dx0, dx1)])
    np.testing.assert_allclose(many, one, rtol=1e-12, atol=1e-14)
