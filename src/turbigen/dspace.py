"""Class to encapsulate a design space."""

import dataclasses
import re
import numpy as np
import turbigen.yaml
import turbigen.config2
from scipy.stats.qmc import LatinHypercube
from pathlib import Path
from turbigen import util

logger = util.make_logger()


@dataclasses.dataclass
class IndependentConfig:
    """Select and extract independent variables for a design space.

    When mapping a design space, we might want to change some combination of:
    - the mean line design
    - the number of blades
    - and so on

    This class defines which independent variables to use, and provides methods
    to get/set a vector of those independent variables to/from a full
    configuration object. We do not need to worry about which element of the
    vector is which as the order is consistently fixed by this class, and
    no further code is needed in the configuration object itself.

    """

    mean_line: dict = dataclasses.field(default_factory=lambda: ({}))
    """Keyed by design variable name, value a limits tuple of (min, max)."""

    nblade: dict = dataclasses.field(default_factory=lambda: ({}))
    """Keyed by row index of dict keyed by blade count parameter, value a limits tuple of (min, max)."""

    tip: dict = dataclasses.field(default_factory=lambda: ({}))
    """Keyed by row index of tuples of (min, max) for each blade row."""

    def __post_init__(self):
        # Check limits are valid
        xlim = self.limits()
        if (xlim[0] >= xlim[1]).any():
            raise ValueError("Invalid limits: min >= max")

    def _split_nblade_key(self, key):
        assert key.startswith("nblade")
        match = re.match(r"^nblade\[(\d+)\]\[(.*)\]$", key)
        return int(match.group(1)), match.group(2)

    def _split_meanline_key(self, key):
        if "[" in key:
            k1 = key.split("[")[0]  # Vector variable name
            k2 = int(key.split("[")[1].split("]")[0])  # Index into the vector
        else:
            k1 = key
            k2 = None
        return k1, k2

    def _split_tip_key(self, key):
        assert key.startswith("tip")
        return int(key.split("[")[1].split("]")[0])

    @property
    def nvar(self):
        """Number of independent variables."""
        # Sum the lengths of all types of independent variables
        return len(self.mean_line) + len(self.nblade) + len(self.tip)

    def keys(self):
        """Get keys for the independent variables.

        This method returns string identifiers for each independent variable,
        that can be passed to the `get_by_key` and `set_by_key` methods of this
        class. The keys fix a consistent order of independent variables.

        """
        keys = []

        for k in self.mean_line:
            keys.append(k)

        for k1 in self.nblade:
            for k2 in self.nblade[k1]:
                keys.append(f"nblade[{k1}][{k2}]")

        for k in self.tip:
            keys.append(f"tip[{k}]")

        if len(keys) != len(set(keys)):
            raise ValueError("Independent variable keys are not unique.")

        return keys

    def limits(self):
        """Get x vectors for upper and lower limits of the design space.


        Returns
        -------
        xlim : np.ndarray
            An array of shape (2, nvar) containing the lower and upper limits of
            the design space. The first row is the lower limit, and the second
            row is the upper limit. The columns correspond to the independent
            variables in the order defined by the `keys()` method.

        """

        xlim = np.full((2, self.nvar), np.nan)
        i = 0  # Index in xlim

        for v in self.mean_line.values():
            xlim[:, i] = v
            i += 1

        for k1 in self.nblade:
            for v in self.nblade[k1].values():
                xlim[:, i] = v
                i += 1

        for v in self.tip.values():
            xlim[:, i] = v
            i += 1

        return xlim

    def get_by_key(self, config, key):
        """Given a string key, extract corresponding independent variable value."""

        # Mean-line default to actual value if set, otherwise nominal
        if key in self.mean_line:
            # Handle vector variables
            name, ind = self._split_meanline_key(key)

            # Get variable by name
            if name in config.mean_line_actual:
                var = config.mean_line_actual[name]
            else:
                var = config.mean_line.design_vars[name]

            # Index into the variable if needed
            if ind is not None:
                return var[ind]
            else:
                return var

        elif key.startswith("nblade"):
            irow, param = self._split_nblade_key(key)
            return getattr(config.nblade[irow], param)

        elif key.startswith("tip"):
            irow = self._split_tip_key(key)
            return config.blades[irow][0].tip

        else:
            raise ValueError(f"Unknown key: {key}")

    def set_by_key(self, config, key, value):
        """Given a string key and numeric value set it in config object."""

        # Always set the nominal mean-line value
        if key in self.mean_line:
            name, ind = self._split_meanline_key(key)
            if ind is not None:
                # If it's a vector variable, set the indexed value
                config.mean_line.design_vars[name][ind] = value
            else:
                config.mean_line.design_vars[name] = value

        elif key.startswith("nblade"):
            irow, param = self._split_nblade_key(key)
            setattr(config.nblade[irow], param, value)

        elif key.startswith("tip"):
            irow = self._split_tip_key(key)
            config.blades[irow][0].tip = value

        else:
            raise ValueError(f"Unknown key: {key}")

    def get_independent(self, config):
        """Extract a design variable vector from a full config object."""
        return np.array([self.get_by_key(config, k) for k in self.keys()])

    def set_independent(self, config, x):
        """Insert a design variable vector into a full config object."""
        for k, xi in zip(self.keys(), x):
            self.set_by_key(config, k, xi)


class Fit:
    def __init__(self, independent, samples, order_max, frac_dof, basis, frac_test):
        """Prepare a fit over a subset of the design space."""

        self.independent = independent
        self.samples = samples
        self.order_max = order_max
        self.frac_dof = frac_dof
        self.basis = basis
        self.frac_test = frac_test

        self.ns = len(samples)
        # Extract and store all x vectors from the samples
        self.x = np.stack(
            [self.independent.get_independent(c) for c in self.samples], axis=-1
        )

        # Store the limits of the design space
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

        # We can now normalise the sampled x vectors
        self.xn = self.normalise(self.x)

        # Select the degrees of freedom for the fit
        if not self.frac_dof:
            # Disable adaptive fitting
            self.set_basis_orders(self.order_max)
        else:
            # Adaptive fitting: start at order_max and reduce the number of
            # polynomial orders until we have few enough degrees of freedom
            dof_target = int(self.frac_dof * self.ns)
            logger.debug("Starting adaptive fitting loop.")
            logger.debug(f"Total number of samples: {self.ns}")
            logger.debug(f"Fraction of degrees of freedom: {self.frac_dof}")
            logger.debug(f"Target degrees of freedom: {dof_target}")
            for order in range(self.order_max, -1, -1):
                if order < 0:
                    raise ValueError("Unable to obtain a fit.")
                self.set_basis_orders(order)
                logger.debug(f"order={order}, dof={len(self._inds)}")
                if self.ndof <= dof_target:
                    logger.debug("Breaking adaptive fitting loop.")
                    break

        # Pre-compute Vandermode-like matrix for fitting
        self._A = np.column_stack([legval(self.xn, i) for i in self._inds])


@dataclasses.dataclass
class DesignSpace:
    """Provide methods to sample and fit a design space."""

    independent: IndependentConfig
    """Independent variables for the design space."""

    nsample_target: int = 0
    """Target number of samples in the design space."""

    basedir: Path = None
    """Base directory for the design space runs."""

    basis: str = "total-order"
    """Type of basis for the polynomal orders of surrogate model."""

    frac_dof: float = 0.7
    """Adaptive fitting max degrees of freedom as a fraction of number of samples.
    Set to 0.0 for no adaption and possible overfitting."""

    order_max: int = 3
    """Maximum order of the polynomial surrogate model."""

    seed: int = 0
    """Seed for random number generator."""

    fast_load: bool = True
    """Skip loading 3D solution for speed."""

    frac_test: float = 0.2
    """Fraction of samples to use for fit error testing."""

    def __post_init__(self):
        # Convert independent dict to an object
        if isinstance(self.independent, dict):
            self.independent = IndependentConfig(**self.independent)

        # Initialise the sampler
        np.random.seed(self.seed)
        self._sampler = LatinHypercube(
            d=self.independent.nvar, seed=self.seed, optimization="random-cd"
        )

    def to_dict(self):
        # Built-in dataclasses method gets us most of the way there
        data = dataclasses.asdict(self)
        data["basedir"] = str(data["basedir"])
        return data

    def load_configs(self):
        """Read in all configs under the design space base directory.
        Store them as a list in the `self.configs` attribute."""

        # Search for all configs in subdirs under the base directory
        # Could parallelize this for big datasets
        logger.iter(f"Loading design space from {self.basedir}")
        fnames = sorted(self.basedir.glob("**/config*.yaml"))

        # Exclude root config
        fnames = [f for f in fnames if not f.parent == self.basedir]

        # Exclude iterations, only keep finished runs
        fnames = [f for f in fnames if f.parent.parent == self.basedir]

        # Loop through the config files and read them
        confs = []
        for f in fnames:
            try:
                # Get raw yaml data first
                data = turbigen.yaml.read_yaml(f)

                # Don't load the design space info to avoid infinite recursion
                data.pop("design_space", None)

                # Add the fast_init flag to the data
                data["_fast_init"] = self.fast_load

                # Create a config object from the data
                c = turbigen.config2.TurbigenConfig(**data)

                # If we loaded the 3D solution, we repeat mean-line processing
                if not self.fast_load:
                    c.design_and_run(skip=True, skip_post=True)

                confs.append(c)

            except Exception as e:
                logger.iter(f"Error reading {f}")
                logger.iter(e)

        # Check the ids are in order and consecutive
        # This is so we know how many samples have been taken already
        fnames_done = [
            f
            for f in fnames
            if f.parent.name.isnumeric() and f.parent.parent == self.basedir
        ]
        ids = [int(f.parent.name) for f in fnames_done]
        if len(ids) != len(set(ids)):
            raise ValueError("IDs are not unique.")
        if not np.all(np.diff(ids) == 1):
            raise ValueError("IDs are not consecutive.")
        if len(ids) > 0 and ids[0] != 0:
            raise ValueError("IDs do not start at 0.")

        # Shuffle the samples from sorted order using the seed
        np.random.shuffle(confs)

        # Fast forward the sampler by the number of samples already taken
        self._nsampled = len(fnames_done)
        logger.iter(f"Fast forwarding sampler by {self._nsampled}")
        self._sampler.fast_forward(self._nsampled)

        # Now exclude any configs that have not ran yet or not converged
        nconf = len(confs)
        nconv = sum(c.converged for c in confs)
        self.configs = [c for c in confs if c.mean_line_actual and c.converged]
        logger.iter(f"Loaded {nconf} config files, {nconv} converged.")

    @property
    def nsample(self):
        """Number of converged, successfully run sample designs."""
        return len(self.configs)

    def normalise(self, x):
        """Given dimensional independent variable vectors, return normalised ones.

        Based on the limits for the current fit."""

        # Check we have the correct number of independent variables
        nx = x.shape[0]
        assert nx == self.independent.nvar

        # Shape our limits for broadcasting
        xlim = self.xlim.reshape((2, nx) + (1,) * (x.ndim - 1))

        # Normalise the x vector to [-1, 1] range
        xn = (x - xlim[0]) / (xlim[1] - xlim[0])
        xn = 2.0 * xn - 1.0

        return xn

    def set_basis_orders(self, order):
        """Get indices for polynomial orders in the basis.

        Parameters
        ----------
        order : int
            Order of the polynomial basis to use.

        Sets:
        self.inds : (n, nx) array
            Each row has one order for each independent variable.
            The number of combinations is determined by the order and the basis type.
        self.ndof = len(self.inds)
        """

        # Get the number of independent variables
        nvar = self.independent.nvar
        self.order = order

        # Initialise a tensor grid of all possible combinations
        # of orders for each independent variable
        inds = np.meshgrid(*[np.arange(0, order + 1) for _ in range(nvar)])
        inds = np.column_stack([np.reshape(i, -1) for i in inds])

        # Eliminate some orders that are not needed, depending on the chosen basis
        if self.basis == "tensor-grid":
            pass  # tensor grid is everything
        elif self.basis == "total-order":
            # Combinations of orders summing <= order_max
            inds = inds[np.sum(inds, axis=1) <= order]
        elif self.basis == "hyperbolic":
            # 0.2<q<1 is a parameter that eliminates high order interactions.
            # q=1 is same as total order.
            q = 0.7
            inds = inds[np.sum(inds**q, axis=1) ** (1.0 / q) <= order]
        else:
            raise Exception(f'Unknown basis "{self.basis}"')

        self._inds = inds
        self.ndof = len(self._inds)

    def setup(self):
        """Read in the design space and prepare the fits."""

        # Populate configs list
        self.load_configs()
        # No-op if we have no data
        if not self.configs:
            return

        # Extract and store all x vectors from the samples
        self.x = np.stack(
            [self.independent.get_independent(c) for c in self.configs], axis=-1
        )

        # Store the limits of the design space

        # Prescribed limits
        self.xlim = self.independent.limits()
        # Actual limits from samples
        xlim_samples = np.stack([np.min(self.x, axis=-1), np.max(self.x, axis=-1)])

        # Get the most extreme of the prescribed limits and the actual
        # limits of the samples. Ensures we are in correct interval
        # for polynomial fitting after normalisation
        self.xlim[0] = np.minimum(self.xlim[0], xlim_samples[0])
        self.xlim[1] = np.maximum(self.xlim[1], xlim_samples[1])

        # We can now normalise the sampled x vectors
        self.xnorm = self.normalise(self.x)

        # Select the degrees of freedom for the fit
        if not self.frac_dof:
            # Disable adaptive fitting
            self.set_basis_orders(self.order_max)
        else:
            # Adaptive fitting: start at order_max and reduce the number of
            # polynomial orders until we have few enough degrees of freedom
            dof_target = int(self.frac_dof * self.nsample)
            logger.debug("Starting adaptive fitting loop.")
            logger.debug(f"Total number of samples: {self.nsample}")
            logger.debug(f"Fraction of degrees of freedom: {self.frac_dof}")
            logger.debug(f"Target max degrees of freedom: {dof_target}")
            for order in range(self.order_max, -1, -1):
                if order < 0:
                    raise ValueError("Unable to obtain a fit.")
                self.set_basis_orders(order)
                logger.debug(f"order={order}, dof={len(self._inds)}")
                if self.ndof <= dof_target:
                    logger.debug("Breaking adaptive fitting loop.")
                    break

        # Pre-compute Vandermode-like matrix for fitting
        self._A = np.column_stack([legval(self.xnorm, i) for i in self._inds])

    def sample(self, datum):
        """Generate random configurations in the design space.

        Samples until we have the target number of samples. If we already have
        enough samples, return nothing.

        """

        n_current = self._nsampled
        logger.iter(f"Found {n_current} samples, target {self.nsample_target}.")
        n = self.nsample_target - n_current
        if n <= 0:
            return []

        # Sample n points in the normalised design space
        xnorm = self._sampler.random(n)

        # Get the limits of the design space
        xlim = self.independent.limits()
        logger.debug("Design space limits:")
        keys = self.independent.keys()
        for k in keys:
            ik = keys.index(k)
            logger.debug(f"  {k}: {xlim[0, ik]} to {xlim[1, ik]}")

        # De-normalize the samples
        x = xlim[0] * (1.0 - xnorm) + xlim[1] * xnorm

        # Create a list of new configurations
        configs = []
        for i in range(n):
            c = datum.copy()
            self.independent.set_independent(c, x[i])
            # Set a numbered workdir under the datum workdir
            c.workdir = self.basedir / f"{i + n_current:04d}"
            configs.append(c)

        return configs

    def meshgrid(self, datum, N=11, **kwargs):
        """Create a meshgrid of independent variable vectors.

        Parameters
        ----------
        datum : config2.TurbigenConfig
            A configuration object to use as a datum for the meshgrid.
            Design variables not specified as keyword arguments
            will be taken from this datum.
        N : int, optional
            Number of points in each dimension of the meshgrid.
        **kwargs : dict
            Keyword arguments specifying limits for each independent variable.
            Design variable names should be in `self.independent.keys()`.

        Returns
        -------
        xg : (ndim, N1, N2, ..., Nndim) array
            A meshgrid of independent variable vectors, where ndim is the number
            of design variables defined in kwargs.

        """

        # Get datum x
        xd = self.independent.get_independent(datum)

        # Assemble grid vectors
        xv = {}
        for ik, k in enumerate(self.independent.keys()):
            if k in kwargs:
                # Get the limits from the keyword argument
                xv[k] = np.linspace(*kwargs[k], N)
            else:
                xv[k](np.array([xd[ik]]))

        # Create a meshgrid of the coordinate vectors
        return {
            k: v for k, v in zip(np.meshgrid(*xv.values(), indexing="ij"), xv.keys())
        }

    def evaluate_samples(self, func):
        """Evaluate a function over all sampled points in the design space."""
        return np.array([func(c) for c in self.configs])

    def rmse(self, func):
        """Calculate train and test RMSE of a function over samples.

        Parameters
        ----------
        func : callable
            Dependent variable to evaluate, callable takes a config object and
            returns a scalar.

        """

        # Extract dependent variable vectors
        y = self.evaluate_samples(func)

        # Split the samples into train and test sets
        # (we shuffled the samples on initialization)
        n = self.nsample
        n_train = int(n * (1.0 - self.frac_test))

        # Perform the polynomial fit on train set only
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
            Function to interpolate, takes a config object and returns a scalar or 1D array.
        confs : config or (n,) list of turbigen.config2.TurbigenConfig
            Query configurations to perform interpolation at.
        kwargs : dict
            Additional keyword arguments to pass to the function.

        Returns
        -------
        yq : scalar or (n,) array
            Values of the fit at query points.

        """

        try:
            len(confs)
            flag_scalar = False
        except TypeError:
            # If confs is a single config, convert to a list
            confs = [confs]
            flag_scalar = True

        # Get independent variable vectors at query points
        xq = np.stack([self.independent.get_independent(c) for c in confs], axis=-1)

        # Check the callable gives the right shape
        yq = func(confs[0], **kwargs)
        assert np.shape(yq) == () or np.ndim(yq) == 1

        # Now evaluate the function at the query points
        yq = self.evaluate(func, xq, **kwargs)

        # If we had a single config, return a scalar
        if flag_scalar:
            yq = yq[0]

        return yq

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
        y = np.array([func(c, **kwargs) for c in self.configs])
        coeff = np.linalg.lstsq(self._A, y, rcond=None)[0]

        # Get the xn and A for query points
        xnq = self.normalise(xq)
        Aq = np.stack([legval(xnq, i) for i in self._inds], axis=-1)

        # Evaluate the polynomial at the query points
        yq = np.matmul(Aq, coeff)

        return yq


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
