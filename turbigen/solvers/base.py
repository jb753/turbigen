"""Define the basic interface that all solvers must conform to."""


class BaseSolver:

    skip = False
    """False to run the CFD as normal, True to write out initial guess and read
    back in, or use a previous solution if available."""

    workdir = ""
    """Working directory to run the simulation in."""

    soft_start = False
    """Run a robust initial guess solution first, then restart."""

    # Below non-user-facing attributes do not have docstrings, only comments

    ntask = 1  # Number of tasks for parallel executeion
    nnode = 1  # Number of nodes for parallel executeion

    _name = "base"

    def __setattr__(self, key, value):
        """Validate attribute assignment."""
        # Should not be able to define new attributes
        if key not in dir(self):
            raise TypeError(f"Invalid {self._name} configuration variable '{key}'")
        #
        # Type must match existing value
        elif not isinstance(value, type(getattr(self, key))):
            raise TypeError(
                f"Invalid type={type(value)} "
                f"for {self._name} configuration variable {key}={value}, "
                f"should be {type(getattr(self,key))}"
            )
        else:
            super().__setattr__(key, value)

    def __init__(self, **kwargs):
        """Override default parameters using keyword args."""
        for k, v in kwargs.items():
            setattr(self, k, v)
