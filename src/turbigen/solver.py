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

    def to_dict(self):
        return {
            "istep": self.istep.tolist(),
            "istep_avg": self.istep_avg,
            "mdot": self.mdot.tolist(),
            "resid": self.resid.tolist(),
            "rho": self.state.rho.tolist(),
            "u": self.state.u.tolist(),
        }


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
    def restart(self):
        """Create a copy of the config with settings to restart from converged soln."""
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

    def setup_convergence(self, state):
        """Convert convergence history to an object, if needed."""
        if isinstance(self.convergence, dict):
            nstep = len(self.convergence["istep"])
            state = state.copy().empty(
                (
                    2,
                    nstep,
                )
            )
            rho = self.convergence.pop("rho")
            u = self.convergence.pop("u")
            state.set_rho_u(rho, u)
            self.convergence = ConvergenceHistory(**self.convergence, state=state)
            self.convergence.istep = np.array(self.convergence.istep)
            self.convergence.mdot = np.array(self.convergence.mdot)
            self.convergence.resid = np.array(self.convergence.resid)

    def to_dict(self):
        # Built in dataclasses.asdict() gets us most of way
        d = dataclasses.asdict(self)

        # Convert the convergence history to a dictionary
        if self.convergence is not None:
            d["convergence"] = self.convergence.to_dict()

        return d
