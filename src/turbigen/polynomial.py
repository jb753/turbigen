"""Functions for fitting polynomials to a design space."""

import os
import json
from multiprocessing import Pool
from itertools import repeat
import numpy as np
from scipy.spatial import Delaunay
import turbigen.util
import turbigen.flowfield
import turbigen.post_process

logger = turbigen.util.make_logger()


def legcoeff(n):
    r"""Coefficients of a univariate Legendre polynomial.

    By Rodrigues' formula, the Legendre polynomial of order :math:`n` is:

    .. math::

        P_n(x) = \frac{1}{2^n\,n!}\frac{\mathrm{d}^n}{\mathrm{d}x^n}
        \left[(x^2-1)^n\right]\,,\quad -1 \le x \le 1\,.

    The key property of the Legendre polynomials is *orthogonality*,

    .. math::
        \int^1_{-1} P_n(x) P_m(x) \mathrm{d} x =
        \begin{cases}
        \frac{1}{2n + 1} & \text{if}\ n=m\,; \\
        0 & \text{otherwise}\,.
        \end{cases}


    Parameters
    ----------
    n: int
        Order of the polynomial.

    Returns
    ------
    c: float array (n+1,)
        Polynomial coefficients in order of descending powers, compatible with
        `numpy.polyval`.

    """

    # Hard-code the coefficients up to a certain order
    if n == 0:
        c = np.array([1.0])
    elif n == 1:
        c = np.array([1.0, 0.0])
    elif n == 2:
        c = np.array([3.0, 0.0, -1.0]) / 2.0
    elif n == 3:
        c = np.array([5.0, 0.0, -3.0, 0.0]) / 2.0
    elif n == 4:
        c = np.array([35.0, 0.0, -30.0, 0.0, 3.0]) / 8.0
    elif n == 5:
        c = np.array([63.0, 0.0, -70.0, 0.0, 15.0, 0.0]) / 8.0
    elif n == 6:
        c = np.array([231.0, 0.0, -315.0, 0.0, 105.0, 0.0, -5.0]) / 16.0
    elif n == 7:
        c = np.array([429.0, 0.0, -693.0, 0.0, 315.0, 0.0, -35.0, 0.0]) / 16.0
    else:
        raise ValueError(f"Do not know Legendre coefficients for n={n}")
    return c


def legval(x, k, der=None):
    r"""Evaluate a multidimensional Legendre polynomial.

    Define a multidimensional polynomial :math:`\vec{x}` as a product of
    univariate polynomials in :math:`x_i`,

    .. math::

        P(\vec{x}\,; \vec{k}) = \prod_i P_{k_i}(x_i)\,,

    where :math:`k_i` is a vector of polynomial orders, one for each dimension.

    Parameters
    ----------
    x: array (npts, ndim)
        Coordinates at which to evaluate polynomials.
    k: array (ndim,)
        Polynomial orders for each dimension.
    der: int
        Index of dimension to take derivative with respect to.

    Returns
    -------
    y: array (npts,)
        Product of Legendre polynomials over the dimensions.

    """

    # Check input data
    npts, ndim = x.shape
    assert len(k) == ndim

    # Evaluate polynomials for each variable at given order
    y = np.zeros((npts, ndim))
    for i in range(ndim):
        if i == der:
            y[:, i] = np.polyval(np.polyder(legcoeff(k[i]), 1), x[:, i])
        else:
            y[:, i] = np.polyval(legcoeff(k[i]), x[:, i])

    # Take the product of univariate polynomials over all variables
    return np.prod(y, axis=1)


def _Rsq(y, y_fit):
    y_av = np.mean(y)
    SStot = np.sum((y - y_av) ** 2.0)
    SSres = np.sum((y - y_fit) ** 2.0)
    return 1.0 - SSres / SStot


def _rmse(y, y_fit):
    return np.sqrt(np.mean((y - y_fit) ** 2.0))


def legfit(x, y, N, basis="total-order"):
    r"""Fit a sum of multidimensional polynomials to a dataset.

    For a dependent variable :math:`y` and a vector independent variables
    :math:`\vec{x}`, approximate the function by summing multivariate Legendre
    polynomials,

    .. math::

        y(\vec{x}) \approx \sum_j^J c_j P_j(\vec{x}\,; K_j)\,,

    where :math:`c_j` are fitted coefficients for each multivariate polynomial.
    We also have to choose a matrix of orders :math:`K`, with rows for each
    polynomial and columns for each indepedent variable.

    The number of polynomials and order matrix are determined from the maximum
    polynomial order :math:`N`, number of independent variables and the choice
    of basis. This is best illustrated by examples, where we will assume two
    independent variables.

    A `tensor-grid` basis uses all possible combiations of polynomial orders up
    to :math:`N`. For :math:`N=1`, there are 4 polynomials, and the order matrix is,

    .. math::

        K =
        \begin{bmatrix}
            0, 0 \\
            1, 0 \\
            0, 1 \\
            1, 1
        \end{bmatrix}\, .

    For :math:`N=2`, there are 9 polynomials, and the order matrix is,

    .. math::

        K =
        \begin{bmatrix}
            0, 0 \\
            1, 0 \\
            2, 0 \\
            0, 1 \\
            1, 1 \\
            2, 1 \\
            0, 2 \\
            1, 2 \\
            2, 2
        \end{bmatrix}\, .

    A `total-order` basis uses only combiations of polynomial orders where the
    sum does not exceed :math:`N`. For :math:`N=2`, there are 6 polynomials,
    and the order matrix is,

    .. math::

        K =
        \begin{bmatrix}
            0, 0 \\
            1, 0 \\
            2, 0 \\
            0, 1 \\
            1, 1 \\
            0, 2 \\
        \end{bmatrix}\,.

    For :math:`N=3`, there are 10 polynomials, and the order matrix is,

    .. math::

        K =
        \begin{bmatrix}
            0, 0 \\
            1, 0 \\
            2, 0 \\
            3, 0 \\
            0, 1 \\
            1, 1 \\
            2, 1 \\
            0, 2 \\
            1, 2 \\
            0, 3 \\
        \end{bmatrix}\,.

    The fit coefficients :math:`c` are found by least-squares solution of a
    number of observations by number of polynomials linear system.

    Parameters
    ----------
    x: array (npts, ndim)
        Independent variables.
    y: array (npts)
        Dependent variable.
    N: int
        Maximum polynomial order.
    basis: str
        How to construct the matrix of orders: "tensor-grid" for all
        possible combinations, or "total-order" to restrict higher
        order terms.

    Returns
    -------
    func: callable
        A function that takes `x` as arguments and returns a fitted `y`, or
        `NaN` if `x` is outside the convex hull of the input data.

    """

    npts, ndim = x.shape

    # Prepare normalised independent variables
    xl = np.min(x, axis=0, keepdims=True)
    xu = np.max(x, axis=0, keepdims=True)
    xn = 2.0 * (x - xl) / (xu - xl) - 1.0
    hull = Delaunay(xn)

    # Generate all possible combinations up to n, then select a basis
    inds = np.meshgrid(*[np.arange(0, N + 1) for _ in range(ndim)])
    inds = np.column_stack([np.reshape(i, -1) for i in inds])
    if basis == "total-order":
        # Use a 'total-order' basis, i.e. reject combinations of polynomial orders
        # that correspond to high order interactions between dimensions over a
        # certain sum of orders
        inds = inds[np.sum(inds, axis=1) <= N]
    elif basis == "hyperbolic":
        # Use a 'hyperbolic' basis, with 0.2<q<1 a parameter that eliminates high
        # order interactions. q=1 is same as total order.
        q = 0.5
        inds = inds[np.sum(inds**q, axis=1) ** (1.0 / q) <= N]
    elif basis == "tensor-grid":
        # Keep all indices
        pass
    else:
        raise Exception(f'Unknown basis "{basis}"')

    # Assemble least squares problem
    # Rows correspond to observations
    # Columns are polynomials
    A = np.column_stack([legval(xn, i) for i in inds])
    c = np.linalg.lstsq(A, y, rcond=None)[0]

    def _func(*args, der=None):
        xi = np.broadcast_arrays(*args)
        s = xi[0].shape
        x = np.column_stack([xii.reshape(-1) for xii in xi])
        xn = 2.0 * (x - xl) / (xu - xl) - 1.0
        A = np.column_stack([legval(xn, i, der=der) for i in inds])
        y = np.matmul(A, c)
        y[hull.find_simplex(xn) < 0] = np.nan
        return y.reshape(s)

    return _func


class DesignSpace:
    """Fit a continuous polynomial representation to a set of mean-lines."""

    _done_init = False

    def __init__(
        self,
        meanline_type,
        train,
        test,
        independent,
        order=3,
        basis="total-order",
        extrapolate=True,
        fac_extrap=None,
        show_timing=False,
    ):
        # Split test and training data
        if isinstance(test, float):
            isplit = int(test * len(train))
            self._test = train[:isplit]
            self._train = train[isplit:]
        else:
            self._train = train
            self._test = test

        self._polys = {}
        self.basis = basis

        meanline_design = turbigen.util.load_mean_line(meanline_type)

        self._meanline_inverse = meanline_design.inverse

        self.set_independent(independent)

        self.set_order(order)

        self.set_extrapolate(extrapolate, fac_extrap)

        self.show_timing = show_timing

        self._done_init = True

    def set_extrapolate(self, extrapolate, fac_extrap):
        self._extrapolate = extrapolate
        if not extrapolate or fac_extrap:
            self._hull = Delaunay(self._xn_train * fac_extrap)

    def set_order(self, order):
        self.order = order

        # Indices for orders of each polynomial
        inds = np.meshgrid(*[np.arange(0, self.order + 1) for _ in range(self.ndim)])
        inds = np.column_stack([np.reshape(i, -1) for i in inds])
        if self.basis == "total-order":
            inds = inds[np.sum(inds, axis=1) <= self.order]
        elif self.basis == "hyperbolic":
            # 0.2<q<1 is a parameter that eliminates high order interactions.
            # q=1 is same as total order.
            q = 0.7
            inds = inds[np.sum(inds**q, axis=1) ** (1.0 / q) <= self.order]
        elif self.basis == "tensor-grid":
            pass
        else:
            raise Exception(f'Unknown basis "{self.basis}"')
        self._inds = inds
        self.ndof = len(inds)

        # Pre-compute Vandermode-like matrix for fitting
        self._A = np.column_stack([legval(self._xn_train, i) for i in inds])

    def set_independent(self, independent):
        self.ndim = len(independent)
        self._independent = independent

        # Independent variable matrices
        self._x_test = np.column_stack(
            [self._get_data_var(self._test, v) for v in self._independent]
        )
        self._x_train = np.column_stack(
            [self._get_data_var(self._train, v) for v in self._independent]
        )

        # Normalised independent variables
        self._xl = np.min(self._x_train, axis=0, keepdims=True)
        self._xu = np.max(self._x_train, axis=0, keepdims=True)
        self._xn_train = 2.0 * (self._x_train - self._xl) / (self._xu - self._xl) - 1.0
        self._xn_test = 2.0 * (self._x_test - self._xl) / (self._xu - self._xl) - 1.0

        self.ntrain = len(self._train)

    @classmethod
    def from_database(
        cls,
        mls,
        independent,
        order=3,
        basis="total-order",
        test_frac=0.0,
        extrapolate=True,
        fac_extrap=None,
        show_timing=False,
        shuffle=True,
    ):
        # Shuffle the designs
        if shuffle:
            ishuf = np.random.permutation(len(mls))
            mls = [mls[iishuf] for iishuf in ishuf]
        mean_line_type = mls[0]._metadata["mean_line_type"]

        return cls(
            mean_line_type,
            mls,
            test_frac,
            independent,
            order=order,
            basis=basis,
            extrapolate=extrapolate,
            fac_extrap=fac_extrap,
            show_timing=show_timing,
        )

    def _get_data_var(self, mls, v):
        """Retrieve a variable by name from the data."""
        try:
            y = np.array([self._meanline_inverse(ml)[v] for ml in mls])
        except KeyError:
            y = np.array([getattr(ml, v) for ml in mls])
        return y

    def _get_y_test(self, v):
        """Retrieve a dependent variable by name from training data."""
        return self._get_data_var(self._test, v)

    def _get_y_train(self, v):
        """Retrieve a dependent variable by name from training data."""
        return self._get_data_var(self._train, v)

    def _fit_coeff(self, v):
        """Get fit coefficients for predictor of a variable by name."""
        # Look in cache, otherwise make a new fit
        if v not in self._polys:
            y = self._get_y_train(v)
            ifit = np.logical_not(np.isnan(y))
            c = np.linalg.lstsq(self._A[ifit, :], y[ifit], rcond=None)[0]
            self._polys[v] = c
        return self._polys[v]

    def _fit_eval(self, xi, v, der=None):
        # Get normalised x coordinates at predictor locations
        xn = 2.0 * (xi - self._xl) / (self._xu - self._xl) - 1.0
        A = np.column_stack([legval(xn, i, der=der) for i in self._inds])
        y = np.matmul(A, self._fit_coeff(v))
        if not self._extrapolate:
            y[self._hull.find_simplex(xn) < 0] = np.nan
        return y

    def __setattr__(self, name, value):
        # Once init is finished, we set evaluation points in the design space
        # by assigning to instance attributes
        if (
            self._done_init
            and not name == "extrapolate"
            and name not in self._independent
        ):
            raise Exception(f"Cannot assign dependent variable {name}")
        else:
            object.__setattr__(self, name, value)

    def __getattr__(self, name):
        # Once init is finished, we retrieve independent vars directly, and
        # dependent vars by fitting
        if self._done_init and name not in self._independent:
            xi = np.broadcast_arrays(*[getattr(self, v) for v in self._independent])
            s = xi[0].shape
            x = np.column_stack([xii.reshape(-1) for xii in xi])
            return self._fit_eval(x, name).reshape(s)
        else:
            raise AttributeError(f"Unrecognised attribute {name}")

    def write_fit(self, varname, fname):
        """Write the fit for one variable to a JSON file."""

        fit_data = {
            "xl": self._xl.tolist(),
            "xu": self._xu.tolist(),
            "k": self._inds.tolist(),
            "c": self._fit_coeff(varname).tolist(),
            "vars": self._independent,
        }

        with open(fname, "w") as f:
            json.dump(fit_data, f)

    def get_calibration_train(self, v):
        xs = self._x_train
        ys = self._get_y_train(v)
        yfit = self._fit_eval(xs, v)
        yy = np.column_stack([ys, yfit]).T
        return yy

    def get_calibration_test(self, v, edge=0.0):
        xs = self._x_test
        ys = self._get_y_test(v)
        xsn = self._xn_test
        i1 = np.all(xsn >= (-1.0 + edge), axis=1)
        i2 = np.all(xsn <= (1.0 - edge), axis=1)
        i12 = np.logical_and(i1, i2)
        yfit = self._fit_eval(xs, v)
        yy = np.column_stack([ys, yfit]).T
        return yy[:, i12]

    def get_limits(self, v):
        val = self._get_y_train(v)
        return (val.min(), val.max())

    def MSE(self, v):
        yy = self.get_calibration_train(v)
        return np.nanmean((yy[0] - yy[1]) ** 2.0)

    def MSE_test(self, v, edge=0.0):
        yy = self.get_calibration_test(v, edge)
        if np.isnan(yy[1]).all():
            return np.nan
        else:
            return np.nanmean((yy[0] - yy[1]) ** 2.0)

    def Rsq(self, v):
        return _Rsq(*self.get_calibration_train(v))

    def Rsq_test(self, v):
        return _Rsq(*self.get_calibration_test(v))

    def RMSE(self, v):
        return np.sqrt(self.MSE(v))

    def RMSE_test(self, v, edge=0.0):
        return np.sqrt(self.MSE_test(v, edge))

    def get_pareto_database(self, v_names, negate):

        v = np.stack([self._get_data_var(self._train, vn) for vn in v_names])

        # Initialise front arbitrarily
        nv, npts = v.shape

        for iv in range(nv):
            if negate[iv]:
                v[iv] *= -1.0

        isort = np.argsort(v[0])

        ifront = [
            isort[0],
        ]

        for inow in isort[1:]:
            # If this point is better in one respect than any existing,
            # add it to the list
            vnow = v[:, inow].reshape(nv, -1)
            if (vnow > v[:, ifront]).any():
                # If this point dominates any others
                # Remove the old points
                idom = np.where((vnow > v[:, ifront]).all(axis=0))[0]
                for iidom in reversed(idom):
                    ifront.pop(iidom)

                # Add new point to list
                ifront.append(inow)

        ifront = np.array(ifront).astype(int)

        return [self._train[i] for i in ifront]


def _case(mean_line_type, mls, Nk, k, o, varname, independent):
    mtest, mtrain = turbigen.util.subsample_cases(mls, k, Nk)
    dspace = turbigen.polynomial.DesignSpace(
        mean_line_type,
        mtrain,
        mtest,
        independent,
        order=o,
    )
    return dspace.MSE(varname), dspace.MSE_test(varname)


def crossvalidate(mls, Nk, orders, varname, independent):
    ishuf = np.random.permutation(len(mls))
    mls = [mls[iishuf] for iishuf in ishuf]

    mean_line_type = mls[0]._metadata["mean_line_type"]
    kg, og = np.meshgrid(np.arange(0, Nk), orders, indexing="ij")

    # Send to workers and run each case in parallel
    Nworker = os.cpu_count()
    with Pool(Nworker) as p:
        err_out = np.stack(
            p.starmap(
                _case,
                zip(
                    repeat(mean_line_type),
                    repeat(mls),
                    repeat(Nk),
                    kg.reshape(-1),
                    og.reshape(-1),
                    repeat(varname),
                    repeat(independent),
                ),
            ),
        )
    err = np.sqrt(err_out.reshape(kg.shape + (2,)))

    err_err = np.std(err, axis=0) / np.sqrt(Nk)
    err = np.mean(err, axis=0)

    return err, err_err


class FittedDesignSpace:
    def __init__(self, fit_json):
        """Load a previously fitted design space.

        Parameters
        ----------
        fit_json: str
            Filename of the fit coefficients in JSON format

        """

        # Get the json data
        with open(fit_json, "r") as f:
            fit_data = json.load(f)

        # Store the data as arrays
        self._independent = np.array(fit_data["vars"])
        self._nvar = len(self._independent)
        self._xl = np.array(fit_data["xl"])
        self._xu = np.array(fit_data["xu"])
        self._k = np.array(fit_data["k"])
        self._c = np.array(fit_data["c"])

    def __call__(self, x):
        """Evaluate the fitted design space at query points.

        Parameters
        ----------
        x: array (nvar, npoint)
            Values of the independent variables, first dimension equal to
            number of independent variables in the fit, second the number of
            query points

        """

        # Check shape of input
        x = np.asarray(x)
        if not x.shape[0] == self._nvar:
            raise Exception(
                f"The input data has first axis length {x.shape[0]}, "
                f"expected the number of independent variables {self._nvar}"
            )

        # Transpose for compatibility with other functions
        # Get normalised x coordinates at predictor locations
        xn = 2.0 * (x.T - self._xl) / (self._xu - self._xl) - 1.0

        # Evaluate the fit
        A = np.column_stack([legval(xn, k) for k in self._k])
        y = np.matmul(A, self._c)

        return y
