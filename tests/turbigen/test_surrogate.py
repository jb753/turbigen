"""Tests for fitting smooth trends through a batch of finished runs.

No CFD. The fit is tested on functions whose polynomial expansion is known, and
the scrape on fabricated runs laid out as in `test_database`.

Test cases:
- test_terms_count_total_order: the usual binomial count when unrestricted
- test_terms_cap_interactions: an additive set has no cross terms
- test_basis_is_orthonormal_over_the_box: what the sensitivity indices rest on
- test_a_polynomial_in_the_basis_is_recovered: exact data, exact fit
- test_leave_one_out_prefers_the_true_term_set: not the biggest that fits
- test_sensitivity_of_an_additive_function: shares of the variance, by hand
- test_total_index_counts_interactions: main and total differ by the cross term
- test_line_and_partial_residuals: the cut, and the runs moved onto it
- test_nan_runs_are_left_out_of_a_fit: a metric a run could not measure
- test_too_few_runs_is_refused: nothing to fit
- test_collect_reads_batch_runs: inputs from the box, outputs from the runs
- test_collect_needs_batch_bounds: no box, no fit
"""

import dataclasses
import math

import numpy as np
import pytest
from test_database import SPREAD, make, write

from turbigen import surrogate
from turbigen.batch import Batch

RNG = np.random.default_rng(0)


def table(x, lo, hi, **y):
    """Return a :class:`~turbigen.surrogate.Table` of synthetic data."""
    return surrogate.Table(
        paths=tuple(f"x{i}" for i in range(x.shape[1])),
        lo=np.asarray(lo, float),
        hi=np.asarray(hi, float),
        x=x,
        y={name: np.asarray(value, float) for name, value in y.items()},
        runs=(),
    )


def uniform(n, lo, hi):
    """Return `n` random points in the box."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    return lo + (hi - lo) * RNG.random((n, len(lo)))


def test_terms_count_total_order():
    for n_var, order in ((2, 2), (3, 3), (9, 2)):
        orders = surrogate.terms(n_var, order, n_var)

        assert len(orders) == math.comb(n_var + order, order)
        assert np.all(orders.sum(axis=1) <= order)
        assert not orders[0].any()


def test_terms_cap_interactions():
    orders = surrogate.terms(9, 2, 1)

    assert len(orders) == 1 + 9 * 2
    assert np.all(np.count_nonzero(orders, axis=1) <= 1)


def test_basis_is_orthonormal_over_the_box():
    """Gauss-Legendre quadrature of each product of terms over the square."""
    orders = surrogate.terms(2, 3, 2)
    nodes, weights = np.polynomial.legendre.leggauss(8)
    X0, X1 = np.meshgrid(nodes, nodes)
    W = np.outer(weights, weights).ravel() / 4.0
    A = surrogate.basis(np.column_stack((X0.ravel(), X1.ravel())), orders)

    gram = A.T @ (W[:, None] * A)

    np.testing.assert_allclose(gram, np.eye(len(orders)), atol=1e-12)


def test_a_polynomial_in_the_basis_is_recovered():
    lo, hi = [0.0, 10.0], [2.0, 20.0]
    x = uniform(30, lo, hi)
    f = 1.0 + 3.0 * x[:, 0] ** 2 - 0.5 * x[:, 0] * x[:, 1]

    fitted = surrogate.fit(table(x, lo, hi, f=f), "f")

    xq = uniform(10, lo, hi)
    np.testing.assert_allclose(
        fitted(xq), 1.0 + 3.0 * xq[:, 0] ** 2 - 0.5 * xq[:, 0] * xq[:, 1]
    )
    assert fitted.loo < 1e-9


def test_leave_one_out_prefers_the_true_term_set():
    """Noisy additive quadratic: the additive quadratic set wins."""
    lo, hi = [-1.0] * 4, [1.0] * 4
    x = uniform(60, lo, hi)
    f = x[:, 0] ** 2 + x[:, 1] + 0.01 * RNG.standard_normal(60)

    fitted = surrogate.fit(table(x, lo, hi, f=f), "f")

    assert fitted.orders.max() == 2
    assert np.count_nonzero(fitted.orders, axis=1).max() == 1


def test_sensitivity_of_an_additive_function():
    """f = a P1(x0) + b P2(x1), orthonormal, so the shares are a^2 : b^2."""
    lo, hi = [-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]
    x = uniform(40, lo, hi)
    a, b = 1.0, 2.0
    P1 = np.sqrt(3.0) * x[:, 0]
    P2 = np.sqrt(5.0) * 0.5 * (3.0 * x[:, 1] ** 2 - 1.0)

    fitted = surrogate.fit(table(x, lo, hi, f=a * P1 + b * P2), "f")

    np.testing.assert_allclose(fitted.variance, a**2 + b**2)
    np.testing.assert_allclose(fitted.main(), [0.2, 0.8, 0.0], atol=1e-12)
    np.testing.assert_allclose(fitted.total(), fitted.main(), atol=1e-12)


def test_total_index_counts_interactions():
    lo, hi = [-1.0, -1.0], [1.0, 1.0]
    x = uniform(40, lo, hi)
    f = 3.0 * x[:, 0] + x[:, 0] * x[:, 1]

    fitted = surrogate.fit(table(x, lo, hi, f=f), "f")

    # Orthonormal: 3 x0 = sqrt(3) P1, x0 x1 = P1 P1 / 3, so variances 3 and 1/9.
    V = 3.0 + 1.0 / 9.0
    np.testing.assert_allclose(fitted.main(), [3.0 / V, 0.0], atol=1e-10)
    np.testing.assert_allclose(fitted.total(), [1.0, (1.0 / 9.0) / V], atol=1e-10)


def test_line_and_partial_residuals():
    lo, hi = [0.0, 0.0], [1.0, 1.0]
    x = uniform(20, lo, hi)
    f = 2.0 * x[:, 0] + x[:, 1]
    noise = 0.1 * RNG.standard_normal(20)

    fitted = surrogate.fit(table(x, lo, hi, f=f + noise), "f")
    xi, y = fitted.line(0, n=5)
    moved = fitted.partial(x, f + noise, 0)

    np.testing.assert_allclose(xi, np.linspace(0.0, 1.0, 5))
    np.testing.assert_allclose(y, fitted(np.column_stack((xi, np.full(5, 0.5)))))
    # On the cut, each run sits off the line by exactly its own residual.
    on_line = fitted(np.column_stack((x[:, 0], np.full(20, 0.5))))
    np.testing.assert_allclose(moved - on_line, f + noise - fitted(x))


def test_nan_runs_are_left_out_of_a_fit():
    lo, hi = [0.0], [1.0]
    x = uniform(10, lo, hi)
    f = 1.0 + x[:, 0]
    f[3] = np.nan

    fitted = surrogate.fit(table(x, lo, hi, f=f), "f")

    assert fitted.n_sample == 9
    np.testing.assert_allclose(fitted(x[3]), 1.0 + x[3, 0])


def test_too_few_runs_is_refused():
    x = uniform(2, [0.0] * 3, [1.0] * 3)

    with pytest.raises(ValueError, match="Too few runs"):
        surrogate.fit(table(x, [0.0] * 3, [1.0] * 3, f=[1.0, 2.0]), "f")


#
# READING A BATCH
#


def datum(**kwargs):
    """Return the datum of a batch over `mean_line.psi`."""
    config = make(1.6, 0.0, **kwargs)
    return dataclasses.replace(
        config, batch=Batch(bounds={"mean_line.psi": [1.2, 2.0]})
    )


def test_collect_reads_batch_runs(tmp_path):
    for i_run, (psi, dchi_TE) in enumerate(SPREAD):
        write(tmp_path / "runs" / f"{i_run:03d}", make(psi, dchi_TE))

    got = surrogate.collect(datum(), "runs/*/config.yaml", tmp_path)

    assert got.paths == ("mean_line.psi",)
    np.testing.assert_allclose(got.lo, [1.2])
    np.testing.assert_allclose(got.hi, [2.0])
    np.testing.assert_allclose(got.x[:, 0], [psi for psi, _ in SPREAD])
    np.testing.assert_allclose(got.y["dchi_TE[1]"], [d for _, d in SPREAD])
    assert "error.dchi_TE[0]" in got.y
    assert got.names("dchi_TE*") == ("dchi_TE[0]", "dchi_TE[1]")


def test_collect_needs_batch_bounds(tmp_path):
    with pytest.raises(ValueError, match="batch: bounds:"):
        surrogate.collect(make(1.6, 0.0), "runs/*/config.yaml", tmp_path)
