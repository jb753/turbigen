"""Base class to define solver interface."""

from abc import ABC, abstractmethod
import dataclasses
import numpy as np


class ConvergenceHistory:
    def __init__(self, istep, istep_avg, resid, mdot, state):
        """Store simulation convergence history.

        Parameters
        ----------
        istep: (nlog,) array
            Indices of the logged time steps.
        resid: (nlog,), array
            Iteration residuals for logged time steps.
        mdot: (2, nlog) array
            Inlet and outlet mass flow rates for all time steps.
        state: Fluid size (nlog,)
            Working fluid object to logg thermodynamic properties.

        """
        self.istep = istep
        self.istep_avg = istep_avg
        self.nlog = len(istep)
        self.mdot = mdot
        self.resid = resid
        self.state = state

    def raw_data(self):
        return np.column_stack(
            (self.istep, *self.mdot, self.resid, self.state.rho, self.state.u)
        )

    @property
    def err_mdot(self):
        return self.mdot[1] / self.mdot[0] - 1.0


@dataclasses.dataclass
class BaseSolver(ABC):
    """Base class for flow solvers."""

    skip: bool = False
    """False to run the CFD as normal, True to write out initial guess and read
    back in, or use a previous solution if available."""

    soft_start: bool = False
    """Run a robust initial guess solution first, then restart."""

    convergence: ConvergenceHistory = None
    """Storage for convergence history."""

    def replace(self, **kwargs):
        return dataclasses.replace(self, **kwargs)

    @abstractmethod
    def robust(self):
        """Create a copy of the config with more robust settings."""
        raise NotImplementedError()

    @abstractmethod
    def run(self, grid, machine):
        """Run the solver on the given grid and machine geometry.

        Parameters
        ----------
        grid:
            Grid object.
        machine:
            Machine geometry object.

        Returns
        -------
        conv: ConvergenceHistory
            The time-marching convergence history of the flow solution.

        """
        raise NotImplementedError
