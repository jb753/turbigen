"""Smooth trends through a batch of finished runs.

:mod:`~turbigen.database` blends the runs nearest a new design to start it
from, and deliberately fits nothing global: a first guess wants to reproduce
the samples, not to be smooth between them. This module is for the other
question a finished batch answers --- how does efficiency, a loss component or
a settled recamber vary over the space --- where a smooth global trend is the
point and reproducing each run exactly is not.

The trend is a sum of multivariate Legendre polynomials over the box
``batch: bounds:`` declares, picked by leave-one-out error from a short ladder
of term sets. Legendre rather than monomials because, normalised over the box a
Sobol' batch fills uniformly, the terms are orthonormal: the fit's variance over
the box is the sum of its squared coefficients, and splitting that sum by which
variables each term involves gives Sobol' sensitivity indices for nothing.

The box is the declared one and not the range the samples happen to cover. The
orthogonality holds over the box that was sampled, and a batch screened of the
points that would not design still meant to sample that box.
"""

import dataclasses
import fnmatch
import itertools
import logging
import math
from pathlib import Path

import numpy as np
from numpy.polynomial import legendre
from scipy.linalg import solve_triangular
from scipy.optimize import linprog, nnls

from turbigen import batch, iterate, node
from turbigen.database import Database, gather

logger = logging.getLogger("turbigen.surrogate")
"""Which runs were read, and which term set each output was fitted with."""

EFFICIENCIES = ("eta_tt", "eta_ts")
"""Efficiencies of the mixed-out mean line, inlet to outlet, added as outputs."""

LADDER = ((1, 1), (2, 1), (3, 1), (2, 2), (4, 1), (3, 2))
"""Term sets tried, as ``(order, interaction)``, in increasing size.

`order` caps the total polynomial order of a term and `interaction` the number
of variables one term may involve. Interactions are capped separately because
they are what a total-order basis spends its terms on: in nine variables a
total-order quadratic is 55 terms, of which 36 are pairwise products, against
19 for the same order with each variable alone. With a few tens of runs the
additive sets are usually the only ones leave-one-out can afford.
"""


@dataclasses.dataclass(frozen=True)
class Table:
    """Inputs and outputs of the finished runs of a batch, one row per run."""

    paths: tuple
    """Design variables, as `batch.paths` orders and `node.flatten` spells them."""

    lo: np.ndarray
    """Lower bound of each design variable, shape ``(n_var,)``."""

    hi: np.ndarray
    """Upper bound of each design variable, shape ``(n_var,)``."""

    x: np.ndarray
    """Design variables of each run, shape ``(n_run, n_var)``."""

    y: dict
    """Each output by name, shape ``(n_run,)``, NaN where a run did not have it."""

    runs: tuple
    """Case file each row was read from."""

    def names(self, pattern="*"):
        """Return the outputs whose names match a shell-style `pattern`, sorted."""
        return tuple(
            name for name in sorted(self.y) if fnmatch.fnmatchcase(name, pattern)
        )

    def swap(self, path, name, values):
        """Return a copy with design variable `path` replaced by `values`.

        For fitting against a quantity each run measured rather than one the
        batch set, such as reaction where the box varies an exit Mach number.
        The swap is only sound if `values` is a one-to-one function of `path`
        with the other design variables held: the fit cannot tell two runs
        apart that differ only in the variable it no longer sees.

        `values` is per run, shape ``(n_run,)``; the column takes `name` in
        :attr:`paths`, and its bounds become the range of `values`. Runs where
        it is NaN are dropped, with their outputs, as a run without a position
        cannot be fitted.

        **The basis is no longer orthonormal in that variable.** The box filled
        uniformly by the batch was over `path`; `values` are bunched however
        the runs map them, so the Sobol' indices of :meth:`Fit.main` and
        :meth:`Fit.total` become approximate. The fit itself, and its
        leave-one-out error, are unaffected.
        """
        values = np.asarray(values, float)
        if values.shape != (len(self.x),):
            raise ValueError(
                f"Swapping {path!r} needs one value per run, {len(self.x)}, "
                f"got shape {values.shape}."
            )
        if name != path and name in self.paths:
            raise ValueError(f"{name!r} is already a design variable.")

        i = self.paths.index(path)
        keep = np.isfinite(values)
        if not keep.any():
            raise ValueError(f"No run has a finite value to swap in for {path!r}.")
        x = self.x[keep].copy()
        x[:, i] = values[keep]
        lo, hi = self.lo.copy(), self.hi.copy()
        lo[i], hi[i] = values[keep].min(), values[keep].max()
        return dataclasses.replace(
            self,
            paths=self.paths[:i] + (name,) + self.paths[i + 1 :],
            lo=lo,
            hi=hi,
            x=x,
            y={key: value[keep] for key, value in self.y.items()},
            runs=tuple(r for r, k in zip(self.runs, keep) if k) if self.runs else (),
        )


def collect(config, path, anchor="."):
    """Return the finished runs matching `path` as a :class:`Table`.

    Parameters
    ----------
    config : Config
        The datum the batch was written from. Its ``batch: bounds:`` gives the
        design variables and their box, and its iterators decide which runs
        count, exactly as for a warm start.
    path : str
        Glob matching the case files to read, relative to `anchor`.
    anchor : Path or str
        Directory `path` is resolved against.

    Every run contributes:

    - the knobs its iterators settled on, named as `iterate.unknowns` names
      them: ``dchi_LE[1]``, ``mean_line.Ys[0]``;
    - every number under ``result: metrics:`` and ``result: error:``, prefixed
      ``metrics.`` and ``error.``: ``metrics.Ys_sec[1]``;
    - :data:`EFFICIENCIES`, from the mixed-out mean line.

    """
    if config.batch is None or not config.batch.bounds:
        raise ValueError(
            "A surrogate is fitted over the box `batch: bounds:` declares, and "
            "this config declares none."
        )

    rows = Database(path=path).load_samples(config, anchor)
    if not rows:
        raise ValueError(f"No finished runs match {path} under {anchor}.")

    paths = config.batch.paths()
    lo, hi = config.batch.limits()
    # A tied key names several leaves holding one value; read the first.
    leaves = [batch.members(path)[0] for path in paths]
    x = gather([sample for _, sample, _ in rows], leaves)

    outputs = [_outputs(sample, result) for _, sample, result in rows]
    names = sorted(set().union(*outputs))
    y = {
        name: np.array([out.get(name, np.nan) for out in outputs], float)
        for name in names
    }

    logger.info(
        f"Read {len(rows)} finished run(s) over {len(paths)} design variable(s), "
        f"with {len(names)} output(s)."
    )
    return Table(
        paths=paths,
        lo=lo,
        hi=hi,
        x=x,
        y=y,
        runs=tuple(match for match, _, _ in rows),
    )


def _outputs(config, result):
    """Return the numeric outputs of one run, by name."""
    out = dict(iterate.unknowns(config))

    leaves = {}
    node._leaves({"metrics": result.metrics, "error": result.error}, "", leaves)
    out.update(leaves)

    if result.actual is not None:
        for name in EFFICIENCIES:
            out[name] = getattr(result.actual, name)

    return {
        name: float(value)
        for name, value in out.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }


def terms(n_var, order, interaction):
    """Return the orders of each term, shape ``(n_term, n_var)``.

    Every term whose orders sum to at most `order` and which involves at most
    `interaction` variables, the constant first. ``interaction=n_var`` is the
    usual total-order basis.
    """
    rows = []
    for n_active in range(min(interaction, n_var) + 1):
        for dims in itertools.combinations(range(n_var), n_active):
            for k in itertools.product(range(1, order + 1), repeat=n_active):
                if sum(k) <= order:
                    row = np.zeros(n_var, int)
                    row[list(dims)] = k
                    rows.append(row)
    return np.array(rows).reshape(-1, n_var)


def basis(xn, orders):
    r"""Return orthonormal Legendre terms at `xn`, shape ``(n_point, n_term)``.

    Parameters
    ----------
    xn : array, shape ``(n_point, n_var)``
        Coordinates normalised onto ``[-1, 1]``.
    orders : array, shape ``(n_term, n_var)``
        Order of each term in each variable, as :func:`terms` returns.

    Each univariate polynomial is scaled by :math:`\sqrt{2k + 1}`, so a term
    has unit mean square over the box and distinct terms are orthogonal.
    """
    k_max = int(orders.max(initial=0))
    scale = np.sqrt(2.0 * np.arange(k_max + 1) + 1.0)
    V = legendre.legvander(xn, k_max) * scale
    n_var = xn.shape[1]
    return np.prod(V[:, np.arange(n_var)[None, :], orders], axis=-1)


@dataclasses.dataclass(frozen=True)
class Fit:
    """A Legendre polynomial fitted to one output over the batch box."""

    name: str
    """Output fitted."""

    paths: tuple
    """Design variables, in column order."""

    lo: np.ndarray
    """Lower bound of each design variable."""

    hi: np.ndarray
    """Upper bound of each design variable."""

    orders: np.ndarray
    """Order of each term in each variable, shape ``(n_term, n_var)``."""

    coeff: np.ndarray
    """Coefficient of each term, shape ``(n_term,)``."""

    rmse: float
    """Root-mean-square residual over the runs fitted."""

    loo: float
    """Root-mean-square leave-one-out error: each run predicted without it."""

    n_sample: int
    """Runs fitted."""

    samples: np.ndarray
    """Design variables of each run fitted, normalised, shape ``(n_sample, n_var)``.

    Kept so that :meth:`inside` can say where the fit is interpolating.
    """

    def normalise(self, x):
        """Return `x` mapped from the box onto ``[-1, 1]``."""
        return 2.0 * (np.asarray(x, float) - self.lo) / (self.hi - self.lo) - 1.0

    @property
    def centre(self):
        """Middle of the box."""
        return 0.5 * (self.lo + self.hi)

    def __call__(self, x, hull=False):
        """Return the fit at `x`, shape ``(..., n_var)`` to ``(...)``.

        A polynomial evaluates anywhere, so by default this does too, box or no
        box. With `hull` true, points outside the convex hull of the runs
        fitted are NaN instead: see :meth:`inside`.
        """
        x = np.asarray(x, float)
        shape = x.shape[:-1]
        xn = self.normalise(x.reshape(-1, len(self.paths)))
        y = basis(xn, self.orders) @ self.coeff
        if hull:
            y[~self.inside(x.reshape(-1, len(self.paths)))] = np.nan
        return y.reshape(shape)

    def inside(self, x, tol=1e-7):
        """Return whether each point lies in the convex hull of the runs fitted.

        Shape ``(..., n_var)`` to ``(...)``. Outside it the fit is
        extrapolating, however well it scores on the runs themselves.

        **The hull in every design variable at once**, not in the few a chart
        happens to vary: a point in the middle of the phi--psi plane is still
        outside if the runs near it all sit at one end of a third variable.

        Tested as a linear programme, the distance from the point to the
        nearest convex combination of the runs, rather than by building the
        hull: in eight variables, 347 runs make a hull of 320 thousand facets
        and a Delaunay triangulation of 2.8 million simplices, a minute to
        build and half as long again to query a chart's worth of points.

        **Each solve leaves a certificate that settles its neighbours**, so a
        grid of points costs far fewer solves than it has points. One outside
        the hull leaves the hyperplane separating it, read off the dual, with
        every run on the other side: any later point beyond that plane is
        outside too. One inside leaves the few runs it is a convex combination
        of: any later point that is a non-negative combination of those, a
        least-squares solve in a handful of unknowns, is inside too. Neither
        is a heuristic, so the answer is the one a solve per point would give,
        to within `tol` in normalised coordinates.
        """
        x = np.asarray(x, float)
        shape = x.shape[:-1]
        xn = self.normalise(x.reshape(-1, len(self.paths)))
        n_sample, n_var = self.samples.shape
        # Unknowns are the weights on the runs, then the slacks either way on
        # each coordinate, whose sum is the distance minimised.
        A_eq = np.vstack(
            (
                np.hstack((self.samples.T, np.eye(n_var), -np.eye(n_var))),
                np.append(np.ones(n_sample), np.zeros(2 * n_var)),
            )
        )
        c = np.append(np.zeros(n_sample), np.ones(2 * n_var))

        planes = np.empty((0, n_var + 1))
        simplex = None
        out = np.zeros(len(xn), bool)
        # The runs lie in the box, so a point outside it is outside the hull.
        in_box = np.all(np.abs(xn) <= 1.0, axis=1)
        for i in np.flatnonzero(in_box):
            b = np.append(xn[i], 1.0)
            if np.any(planes[:, :-1] @ xn[i] + planes[:, -1] > tol):
                continue
            if simplex is not None and nnls(simplex, b)[1] < tol:
                out[i] = True
                continue
            lp = linprog(c, A_eq=A_eq, b_eq=b, bounds=(0.0, None), method="highs")
            if lp.fun < tol:
                out[i] = True
                simplex = A_eq[:, np.flatnonzero(lp.x[:n_sample] > 0.0)]
            else:
                # The dual on the coordinates points away from the hull; the
                # offset is set by the runs themselves, not taken from the dual.
                normal = lp.eqlin.marginals[:n_var]
                plane = np.append(normal, -np.max(self.samples @ normal))
                if plane[:-1] @ xn[i] + plane[-1] > tol:
                    planes = np.vstack((planes, plane))
        return out.reshape(shape)

    @property
    def variance(self):
        """Variance of the fit over the box: every term but the constant."""
        return float(np.sum(self.coeff[1:] ** 2))

    def main(self):
        """Return each variable's main-effect Sobol' index, shape ``(n_var,)``.

        The share of the variance carried by terms in that variable alone.
        """
        alone = np.count_nonzero(self.orders, axis=1) == 1
        return np.array(
            [
                np.sum(self.coeff[alone & (self.orders[:, i] > 0)] ** 2)
                for i in range(len(self.paths))
            ]
        ) / max(self.variance, np.finfo(float).tiny)

    def total(self):
        """Return each variable's total Sobol' index, shape ``(n_var,)``.

        The share of the variance carried by every term involving it, so its
        interactions too. Equal to :meth:`main` for an additive fit.
        """
        return np.array(
            [
                np.sum(self.coeff[self.orders[:, i] > 0] ** 2)
                for i in range(len(self.paths))
            ]
        ) / max(self.variance, np.finfo(float).tiny)

    def line(self, axis, n=51, at=None):
        """Return a cut along one variable with the others held at `at`.

        Parameters
        ----------
        axis : int
            Variable to vary, as a column index.
        n : int
            Points along it, spanning the box.
        at : array, optional
            Where to hold the others; the centre of the box by default.

        Returns
        -------
        xi : array, shape ``(n,)``
        y : array, shape ``(n,)``

        """
        at = self.centre if at is None else np.asarray(at, float)
        xi = np.linspace(self.lo[axis], self.hi[axis], n)
        x = np.tile(at, (n, 1))
        x[:, axis] = xi
        return xi, self(x)

    def partial(self, x, y, axis, at=None):
        """Return the samples moved onto the cut of :meth:`line` along `axis`.

        Each run's residual added to the cut at its own value of `axis`, so a
        scatter of them about the line shows how well it is supported: the
        partial residuals of a main-effect plot.
        """
        at = self.centre if at is None else np.asarray(at, float)
        x = np.asarray(x, float)
        moved = np.tile(at, (len(x), 1))
        moved[:, axis] = x[:, axis]
        return y - self(x) + self(moved)


def fit(table, name, ladder=LADDER):
    """Return the fit to output `name` with the lowest leave-one-out error.

    Runs where the output is NaN are left out. Each term set on `ladder` with
    fewer terms than runs is fitted by least squares; its leave-one-out error
    comes from the diagonal of the hat matrix, so no run is refitted.
    """
    y = table.y[name]
    keep = np.isfinite(y)
    x, y = table.x[keep], y[keep]
    n_sample, n_var = x.shape

    xn = 2.0 * (x - table.lo) / (table.hi - table.lo) - 1.0

    best = None
    for order, interaction in ladder:
        orders = terms(n_var, order, interaction)
        if len(orders) >= n_sample:
            continue

        A = basis(xn, orders)
        Q, R = np.linalg.qr(A)
        if np.min(np.abs(np.diag(R))) < 1e-10 * np.max(np.abs(np.diag(R))):
            continue
        coeff = solve_triangular(R, Q.T @ y)

        residual = y - A @ coeff
        leverage = np.sum(Q**2, axis=1)
        loo = math.sqrt(np.mean((residual / (1.0 - leverage)) ** 2))

        logger.debug(
            f"{name}: order {order}, interaction {interaction}, "
            f"{len(orders)} terms, loo {loo:.4g}"
        )
        if best is None or loo < best.loo:
            best = Fit(
                name=name,
                paths=table.paths,
                lo=table.lo,
                hi=table.hi,
                orders=orders,
                coeff=coeff,
                rmse=math.sqrt(np.mean(residual**2)),
                loo=loo,
                n_sample=n_sample,
                samples=xn,
            )

    if best is None:
        raise ValueError(
            f"Too few runs ({n_sample}) with {name} to fit even a plane in "
            f"{n_var} variables."
        )

    logger.info(
        f"{name}: {len(best.orders)} terms over {n_sample} runs, "
        f"rmse {best.rmse:.4g}, loo {best.loo:.4g}"
    )
    return best


def read(datum, path):
    """Return the :class:`Table` of a batch, given its datum file.

    `path` is resolved against the datum's directory, which is where a batch
    writes its members, rather than against each member's own directory as the
    datum's ``database: path:`` is.
    """
    from turbigen import case

    datum = Path(datum)
    config, _ = case.read(datum, design=False)
    return collect(config, path, datum.parent)
