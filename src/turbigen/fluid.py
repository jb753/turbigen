"""Configuration for the working fluid.

All equations of state are handled by `ember.fluid` module.

"""

import turbigen.util as util

import dataclasses
from abc import ABC, abstractmethod

import ember.fluid


@dataclasses.dataclass
class FluidConfig(ABC):
    """Base class for fluid configurations."""

    type: str
    """Which equation of state: {'perfect'}"""

    fluid: ember.fluid.BaseFluid = dataclasses.field(init=False, repr=False)
    """Instance of the fluid class."""

    def __post_init__(self):
        # Validate type and get the fluid class
        self._fluid_class = util.get_subclass_by_name(ember.fluid.BaseFluid, self.type)

        # Create the fluid instance
        self.fluid = self._create_fluid()

    @abstractmethod
    def _create_fluid(self) -> ember.fluid.BaseFluid:
        """Subclasses implement this to instantiate their fluid type."""
        raise NotImplementedError()


@dataclasses.dataclass
class PerfectFluidConfig(FluidConfig):
    """Configuration for perfect gas with constant specific heats."""

    cp: float
    """Specific heat at constant pressure [J/kg/K]"""

    gamma: float
    """Ratio of specific heats [--]"""

    mu: float
    """Dynamic viscosity [kg/m/s]"""

    Pr: float = 1.0
    """Prandtl number [--]"""

    Tu0: float = 300.0
    """Temperature datum for internal energy [K]"""

    def _create_fluid(self) -> ember.fluid.PerfectFluid:
        """Create the PerfectFluid instance."""
        return ember.fluid.PerfectFluid(
            cp=self.cp,
            gamma=self.gamma,
            mu=self.mu,
            Pr=self.Pr,
            Tu0=self.Tu0,
        )
