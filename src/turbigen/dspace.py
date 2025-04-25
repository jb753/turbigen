"""Class to encapsulate a design space."""

import dataclasses
import numpy as np
import turbigen.yaml
import turbigen.config2
from scipy.stats.qmc import LatinHypercube


@dataclasses.dataclass
class IndependentConfig:
    """Define independent variables for a design space."""

    mean_line: dict = dataclasses.field(default_factory=lambda: ({}))
    """Keyed by design variable name, value a limits tuple of (min, max)."""

    nblade: list = dataclasses.field(default_factory=list)
    """dict keyed by row index of dict keyed by blade count parameter, value a limits tuple of (min, max)."""

    @property
    def nx(self):
        """Number of independent variables."""
        return len(self.mean_line) + len(self.nblade)

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
            x[i] = config.mean_line.design_vars[k]
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

    datum: turbigen.config2.TurbigenConfig
    """Datum configuration."""

    independent: IndependentConfig
    """Independent variables for the design space."""

    seed: int = 0
    """Seed for random number generator."""

    configs: list = dataclasses.field(default_factory=list)
    """Configurations for all simulated samples in the design space."""

    def __post_init__(self):
        # Conver independent to an object
        if isinstance(self.independent, dict):
            self.independent = IndependentConfig(**self.independent)

        # Consistency check for x getting and setting
        x0 = self.independent.get_independent(self.datum)
        c = self.datum.copy()
        self.independent.set_independent(c, x0)
        x1 = self.independent.get_independent(c)
        assert np.allclose(x0, x1), "Inconsistent x setting and getting"

        # Search for all configs under the datum workdir
        # get the YAML data and fast load them
        # Could parallelize this for big datasets
        # print(f"Loading configs from {self.datum.workdir}...")
        fnames = sorted(self.datum.workdir.glob("**/*.yaml"))
        confs = []
        for f in fnames:
            # print(f"  {f}")
            try:
                data = turbigen.yaml.read_yaml(f)
                # if not data.get("converged", False):
                #     print(f"  {f} not converged, skipping")
                #     continue
                data["_fast_init"] = True
                confs.append(turbigen.config2.TurbigenConfig(**data))
            except Exception as e:
                raise RuntimeError(f"Error reading {f}: {e}")
        self.samples = confs

        # Check the ids are in order and consecutive
        ids = [int(f.parent.name) for f in fnames]
        if len(ids) != len(set(ids)):
            raise ValueError("IDs are not unique.")
        if not np.all(np.diff(ids) == 1):
            raise ValueError("IDs are not consecutive.")
        if not ids[0] == 0:
            raise ValueError("IDs do not start at 0.")

        # Initialise the sampler and fast-forward over the existing samples
        self._sampler = LatinHypercube(
            d=self.independent.nx, seed=self.seed, optimization="random-cd"
        )
        self._sampler.fast_forward(len(self.samples))

    def sample(self, n):
        """Generate random configurations in the design space."""

        # Sample n points in the design space
        xnorm = self._sampler.random(n)

        # Get the limits of the design space
        xlim = self.independent.get_limits()

        # De-normalize the samples
        x = xlim[0] * (1.0 - xnorm) + xlim[1] * xnorm

        # Create a list of configurations
        configs = []
        for i in range(n):
            c = self.datum.copy()
            self.independent.set_independent(c, x[i])
            # Set a numbered workdir under the datum workdir
            c.workdir = self.datum.workdir / f"{i:03d}"
            configs.append(c)

        return configs

    def interpolate(self, func, x):
        """Interpolate something as a function of x through the design space.

        Parameters
        ----------
        func : callable
            Function to interpolate, takes a config object and returns a value.
        x : (nx,) array
            Design variable vector to interpolate at.

        """

        assert len(x) == self.independent.nx, "x must be of length nx"

    def fit(self, x, y):
        """Construct a surrogate model for y as a function of x."""

        # Use a cached fit if present
