"""Classes for initialising an inlet state."""

import dataclasses
from abc import ABC, abstractmethod
import turbigen.fluid
import numpy as np


@dataclasses.dataclass
class InletConfig(ABC):
    """Base class for inlet boundary condition settings."""

    spf: list = None
    """Span fraction of some radial stations running 0 to 1."""

    profiles: np.ndarray = None
    """Po, To, Alpha, Beta at each span fraction, shape (4,nspf)"""

    @abstractmethod
    def get_inlet(self):
        """Return a State object for the inlet working fluid."""
        raise NotImplementedError


@dataclasses.dataclass
class PerfectInletConfig(InletConfig):
    """Settings for a perfect gas inlet boundary condition."""

    Po: float = np.nan
    """Stagnation pressure of the inlet [Pa]."""

    To: float = np.nan
    """Stagnation temperature of the inlet [K]."""

    cp: float = np.nan
    """Specific heat capacity of the inlet [J/kgK]."""

    gamma: float = np.nan
    """Specific heat ratio of the inlet [--]."""

    mu: float = np.nan
    """Dynamic viscosity of the inlet, or NaN if not specified [Pa.s]."""

    def get_inlet(self):
        """Return a State object for the inlet working fluid."""
        return turbigen.fluid.PerfectState.from_properties(
            self.cp,
            self.gamma,
            self.mu,
        ).set_P_T(self.Po, self.To)


@dataclasses.dataclass
class RealInletConfig(InletConfig):
    """Settings for a real gas inlet boundary condition."""

    Po: float = np.nan
    """Stagnation pressure of the inlet [Pa]."""

    To: float = np.nan
    """Stagnation temperature of the inlet [K]."""

    fluid_name: str = ""
    """Name of the working fluid in CoolProp database."""

    def get_inlet(self):
        """Return a State object for the inlet working fluid."""
        return turbigen.fluid.RealState.from_fluid_name(self.fluid_name).set_P_T(
            self.Po, self.To
        )


#
#
# # data = turbigen.yaml.read("examples/quad_camber.yaml")
# if __name__ == "__main__":
#     inlet_dat = {"Po": 101325, "To": 288.15, "cp": 1000, "gamma": 1.4}
#     inlet = util.init_subclass_by_signature(InletConfig, inlet_dat)
#     print(type(inlet))
#     print(inlet)
