"""Class to encapsulate a design space."""

import dataclasses
import numpy as np
import turbigen.yaml
import turbigen.config2
from scipy.stats.qmc import LatinHypercube
from pathlib import Path
from turbigen import util

logger = util.make_logger()


@dataclasses.dataclass
class IndependentConfig:
    """Define independent variables for a design space."""

    mean_line: dict = dataclasses.field(default_factory=lambda: ({}))
    """Keyed by design variable name, value a limits tuple of (min, max)."""

    nblade: list = dataclasses.field(default_factory=list)
    """dict keyed by row index of dict keyed by blade count parameter, value a limits tuple of (min, max)."""

    def __post_init__(self):
        # Check limits are valid
        xlim = self.get_limits()
        if (xlim[0] >= xlim[1]).any():
            raise ValueError("Invalid limits: min >= max")

    @property
    def nx(self):
        """Number of independent variables."""
        return len(self.mean_line) + len(self.nblade)

    def keys(self):
        """Get the keys of the independent variables."""
        keys = []
        for k in self.mean_line:
            keys.append(k)
        for k1 in self.nblade:
            for k2 in self.nblade[k1]:
                keys.append(f"nblade[{k1}][{k2}]")
        return keys

    def get_limits(self):
        """Get x vectors for upper and lower limits of the design space.

        Returns
        -------
        xlim : np.ndarray
            An array of shape (2, nx) containing the lower and upper limits of
            the design space. The first row is the lower limit, and the second
            row is the upper limit.

        """

        xlim = np.full((2, self.nx), np.nan)
        i = 0  # Keep track of index in x

        for v in self.mean_line.values():
            xlim[:, i] = v
            i += 1

        for k1 in self.nblade:
            for v in self.nblade[k1].values():
                xlim[:, i] = v
                i += 1

        return xlim

    def get_independent(self, config):
        """Extract a design variable vector from a full config object."""

        x = np.full((self.nx,), np.nan)
        i = 0  # Keep track of index in x

        for k in self.mean_line:
            x[i] = config.mean_line_actual[k]
            i += 1

        for k1 in self.nblade:
            for k2 in self.nblade[k1]:
                x[i] = getattr(config.nblade[k1], k2)
                i += 1

        return x

    def set_independent(self, config, x):
        """Insert a design variable vector into a full config object."""

        i = 0  # Keep track of index in x

        for k in self.mean_line:
            config.mean_line.design_vars[k] = x[i]
            i += 1

        for k1 in self.nblade:
            for k2 in self.nblade[k1]:
                setattr(config.nblade[k1], k2, x[i])
                i += 1


@dataclasses.dataclass
class DesignSpace:
    """Provide methods to sample and fit a design space."""

    independent: IndependentConfig
    """Independent variables for the design space."""

    nsample: int = 0
    """Target number of samples in the design space."""

    basedir: Path = None
    """Base directory for the design space."""

    basis: str = "total-order"
    """Basis for the polynomal orders of surrogate model."""

    order: int = 3
    """Maximum order of the polynomial surrogate model."""

    seed: int = 0
    """Seed for random number generator."""

    fast_load: bool = True
    """Skip loading 3D solution for speed."""

    frac_test: float = 0.2
    """Fraction of samples to use for fit error testing."""

    def __post_init__(self):
        # Conver independent to an object
        if isinstance(self.independent, dict):
            self.independent = IndependentConfig(**self.independent)

        # Initialise the sampler and fast-forward over the existing samples
        np.random.seed(self.seed)
        self._sampler = LatinHypercube(
            d=self.independent.nx, seed=self.seed, optimization="random-cd"
        )

    def to_dict(self):
        # Built-in dataclasses method gets us most of the way there
        data = dataclasses.asdict(self)
        data["basedir"] = str(data["basedir"])
        return data

    def load(self):
        # Search for all configs in subdirs under the base directory
        # get the YAML data and fast load them
        # Could parallelize this for big datasets
        # print(f"Loading configs from {self.datum.workdir}...")
        logger.iter(f"Loading design space from {self.basedir}")
        fnames = sorted(self.basedir.glob("*/config.yaml"))
        fnames = [f for f in fnames if f.parent.name.isnumeric()]
        confs = []
        for f in fnames:
            try:
                data = turbigen.yaml.read_yaml(f)
                data.pop("design_space", None)
                data["_fast_init"] = self.fast_load
                c = turbigen.config2.TurbigenConfig(**data)
                if not self.fast_load:
                    c.design_and_run(skip=True, skip_post=True)
                confs.append(c)
            except Exception as e:
                logger.iter(f"Error reading {f}")

        # Nothing else to do if no configs found
        if not confs:
            self.nall = 0
            self.samples = []
            return
        else:
            self.nall = len(fnames)

        # Check the ids are in order and consecutive
        ids = [int(f.parent.name) for f in fnames]
        if len(ids) != len(set(ids)):
            raise ValueError("IDs are not unique.")
        if not np.all(np.diff(ids) == 1):
            raise ValueError("IDs are not consecutive.")
        if len(ids) > 0 and ids[0] != 0:
            raise ValueError("IDs do not start at 0.")

        # Shuffle the samples from sorted order using the seed
        np.random.shuffle(confs)

        # Fast forward the sampler to the number of samples
        self._sampler.fast_forward(len(confs))

        # Now exclude any unconverged samples
        self.samples = [c for c in confs if c.converged]

        # If we have samples, pre-calculate for fitting
        if self.samples:
            self.setup()

    def normalise(self, x):
        nx = x.shape[0]
        assert nx == self.independent.nx

        # Shape the limits for broadcasting
        shape = (2, nx,) + (
            1,
        ) * (x.ndim - 1)
        xlim = self.xlim.reshape(shape)

        xn = (x - xlim[0]) / (xlim[1] - xlim[0])
        xn = 2.0 * xn - 1.0
        return xn

    def setup(self):
        # Extract and store all x vectors from the samples
        self.x = np.stack(
            [self.independent.get_independent(c) for c in self.samples], axis=-1
        )

        # Store the bounds of the design space
        # The most extreme of the prescribed limits and the actual
        # limits of the samples. Ensures we are in correct interval
        # for polynomial fitting after normalisation
        self.xlim = self.independent.get_limits()
        if self.samples:
            x = np.stack([self.independent.get_independent(c) for c in self.samples])
            xlim_samples = np.stack([np.min(x, axis=0), np.max(x, axis=0)])

            # Get the most extreme of the two
            self.xlim[0] = np.minimum(self.xlim[0], xlim_samples[0])
            self.xlim[1] = np.maximum(self.xlim[1], xlim_samples[1])

        self.xn = self.normalise(self.x)

        # Indices for orders of each polynomial
        nx = self.independent.nx
        inds = np.meshgrid(*[np.arange(0, self.order + 1) for _ in range(nx)])
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
        self._A = np.column_stack([legval(self.xn, i) for i in inds])

    def sample(self, datum):
        """Generate random configurations in the design space.

        Samples until we have the target number of samples. If we already have
        enough samples, return nothing.

        """

        n_current = self.nall
        logger.iter(f"Found {n_current} samples, target {self.nsample}.")
        n = self.nsample - n_current
        if n <= 0:
            return []

        # Sample n points in the design space
        xnorm = self._sampler.random(n)

        # Get the limits of the design space
        xlim = self.independent.get_limits()

        # De-normalize the samples
        x = xlim[0] * (1.0 - xnorm) + xlim[1] * xnorm

        # Create a list of configurations
        configs = []
        for i in range(n):
            c = datum.copy()
            self.independent.set_independent(c, x[i])
            # Set a numbered workdir under the datum workdir
            c.workdir = self.basedir / f"{i + n_current:03d}"
            configs.append(c)

        return configs

    def evaluate(self, func, xq, **kwargs):
        """Evaluate a fit to a function at query points.

        Parameters
        ----------
        func : callable
            Function to interpolate, takes a config object and returns a scalar.
        xq : (ndim, ...) array
            Query points to evaluate the fit at, first axis the independent vars,
            remaining axes any shape.

        Returns
        -------
        yq : (...) array
            Values of the fit at query points, same shape as xq[i].

        """

        # Perform the polynomial fit
        y = np.array([func(c, **kwargs) for c in self.samples])
        coeff = np.linalg.lstsq(self._A, y, rcond=None)[0]

        # Get the xn and A for query points
        xnq = self.normalise(xq)
        Aq = np.stack([legval(xnq, i) for i in self._inds], axis=-1)

        # Evaluate the polynomial at the query points
        yq = np.matmul(Aq, coeff)

        return yq

    def rmse(self, func):
        """Calculate train and test RMSE of a function over samples."""

        # Split the samples into train and test sets
        # (we shuffled the samples on initialization)
        n = len(self.samples)
        n_train = int(n * (1.0 - self.frac_test))

        # Perform the polynomial fit on train set
        y = np.array([func(c) for c in self.samples])
        coeff = np.linalg.lstsq(self._A[:n_train], y[:n_train], rcond=None)[0]

        # Evaluate the fit over all samples
        yfit = np.matmul(self._A, coeff)

        # Calculate the RMSE
        sqe = (y - yfit) ** 2
        rmse_train = np.sqrt(np.mean(sqe[:n_train])).item()
        rmse_test = np.sqrt(np.mean(sqe[n_train:])).item()

        return rmse_train, rmse_test

    def interpolate(self, func, confs, **kwargs):
        """Interpolate the value of a function for query configs.

        Parameters
        ----------
        func : callable
            Function to interpolate, takes a config object and returns a scalar.
        confs : (n,) list of turbigen.config2.TurbigenConfig
            Query configurations to perform interpolation at.
        kwargs : dict
            Additional keyword arguments to pass to the function.

        Returns
        -------
        yq : (n,) array
            Values of the fit at query points.

        """

        # Get independent variable vectors at query points
        xq = np.stack([self.independent.get_independent(c) for c in confs], axis=-1)

        # Now evaluate the function at the query points
        return self.evaluate(func, xq, **kwargs)

    def meshgrid(self, datum, N=11, **kwargs):
        # Get datum x
        xd = self.independent.get_independent(datum)

        # Assemble coordinate vectors
        xv = []
        for ik, k in enumerate(self.independent.keys()):
            if k in kwargs:
                # Get the limits from the keyword argument
                xv.append(np.linspace(*kwargs[k], N))
            else:
                xv.append(
                    np.array(
                        [
                            xd[ik],
                        ]
                    )
                )

        # Create a meshgrid of the coordinate vectors
        xg = np.stack(np.meshgrid(*xv, indexing="ij"))

        return xg


def legcoeff(n):
    r"""Coefficients of a univariate Legendre polynomial.

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
    elif n == 8:
        c = (
            np.array([6435.0, 0.0, -12012.0, 0.0, 6930.0, 0.0, -1260.0, 0.0, 35.0])
            / 128.0
        )
    else:
        raise ValueError(f"Do not know Legendre coefficients for n={n}")
    return c


def legval(x, k):
    r"""Evaluate a multidimensional Legendre polynomial.

    Define a multidimensional polynomial :math:`\vec{x}` as a product of
    univariate polynomials in :math:`x_i`,

    .. math::

        P(\vec{x}\,; \vec{k}) = \prod_i P_{k_i}(x_i)\,,

    where :math:`k_i` is a vector of polynomial orders, one for each dimension.

    Parameters
    ----------
    x: array (ndim, npts)
        Coordinates at which to evaluate polynomials.
    k: array (ndim,)
        Polynomial orders for each dimension.

    Returns
    -------
    y: array (npts,)
        Product of Legendre polynomials over the dimensions.

    """

    # Check input data
    ndim = x.shape[0]
    assert len(k) == ndim

    # Evaluate polynomials for each variable at given order
    y = np.stack([np.polyval(legcoeff(k[i]), x[i]) for i in range(ndim)])

    # Take the product of univariate polynomials over all variables
    return np.prod(y, axis=0)
