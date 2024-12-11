"""Define the basic interface that all solvers must conform to."""
import dataclasses
import numpy as np


@dataclasses.dataclass
class BaseSolver:
    """Settings and methods common to all solvers."""

    skip: bool = False
    """False to run the CFD as normal, True to write out initial guess and read
    back in, or use a previous solution if available."""

    soft_start: bool = False
    """Run a robust initial guess solution first, then restart."""

    ntask: int = 1  # Number of tasks for parallel executeion
    nnode: int = 1  # Number of nodes for parallel executeion
    _name: str = "base"

    def _robust(self):
        """Create a copy of the config with more robust settings."""
        raise NotImplementedError()

    def __post_init__(self):
        """Validate the input data"""
        if self.ntask < 1:
            raise Exception(f"ntask={self._ntask} should be > 0")
        if self.nnode < 1:
            raise Exception(f"nnode={self._nnode} should be > 0")

    def replace(self, **kwargs):
        return dataclasses.replace(self, **kwargs)


class ConvergenceHistory:
    def __init__(self, istep, mdot, ho, Po, resid, state, istep_avg):
        """Store simulation convergence history.

        Parameters
        ----------
        istep: (nlog,) array
        mdot: (2, nlog) array
            Inlet and outlet mass flow rates.
        ho: (2, nlog) array
            Inlet and outlet stagnation enthalpies.
        Po: (2, nlog) array
            Inlet and outlet stagnation pressures.
        resid: (nlog,), array
            Iteration residuals for all time steps.
        state: object
            Working fluid object to calculate thermodynamic properties.

        """

        self.istep = istep
        self.istep_avg = istep_avg
        self.nlog = len(istep)
        self.mdot = mdot
        self.resid = resid
        self.state = state.empty((2, self.nlog))
        self.state.set_P_h(Po, ho)

    def raw_data(self):
        return np.column_stack(
            (self.istep, *self.mdot, self.resid * self.state.h * self.state.P)
        )
