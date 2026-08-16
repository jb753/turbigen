"""Configuration for the working fluid.

All equations of state are handled by `ember.fluid` module.

"""

import dataclasses
from abc import ABC, abstractmethod

import ember.fluid


@dataclasses.dataclass(frozen=True)
class FluidConfig(ABC):
    """Base class for fluid configurations."""

    type: str
    """Which equation of state: {'perfect'}"""

    fluid: ember.fluid._Fluid = dataclasses.field(init=False, repr=False)
    """Instance of the fluid class."""

    def __post_init__(self):
        # Create the fluid instance (must use object.__setattr__ for frozen dataclass)
        object.__setattr__(self, "fluid", self._create_fluid())

    @abstractmethod
    def _create_fluid(self) -> ember.fluid._Fluid:
        """Subclasses implement this to instantiate their fluid type."""
        raise NotImplementedError()

    def to_dict(self) -> dict:
        """Convert the FluidConfig to a dictionary."""
        data = dataclasses.asdict(self)
        data.pop("fluid")  # Remove the fluid instance
        return data

    @staticmethod
    def from_dict(config_dict: dict) -> "FluidConfig":
        """Create a FluidConfig subclass from a dictionary.

        Parameters
        ----------
        config_dict : dict
            Dictionary containing 'type' field and other parameters.

        Returns
        -------
        FluidConfig
            Instance of the appropriate FluidConfig subclass.

        Examples
        --------
        >>> config = FluidConfig.from_dict({'type': 'perfect', 'cp': 1005.0, 'gamma': 1.4, 'mu': 1.8e-5})
        >>> isinstance(config, PerfectFluidConfig)
        True

        """
        config_dict = config_dict.copy()  # Don't modify the input
        fluid_type = config_dict.get("type")

        if fluid_type == "perfect":
            return PerfectFluidConfig(**config_dict)
        else:
            available_types = ["perfect"]
            raise ValueError(
                f"Unknown fluid type '{fluid_type}'. Available types: {available_types}"
            )


@dataclasses.dataclass(frozen=True)
class PerfectFluidConfig(FluidConfig):
    """Configuration for perfect gas with constant specific heats."""

    type: str
    """Fluid type identifier"""

    cp: float
    """Specific heat at constant pressure [J/kg/K]"""

    gamma: float
    """Ratio of specific heats [--]"""

    mu: float
    """Dynamic viscosity [kg/m/s]"""

    Pr: float = 0.7
    """Prandtl number [--]"""

    # Location of entropy and internal energy tables (for tabulated fluids)
    P_dtm: float = 1e5
    T_dtm: float = 300.0
    V_ref: float = 1.0
    rho_ref: float = 1.0
    rgas_ref: float = 1.0

    def __post_init__(self):
        # Validate that type is correct
        assert (
            self.type == "perfect"
        ), f"PerfectFluidConfig requires type='perfect', got '{self.type}'"
        # Call parent __post_init__ to create fluid
        super().__post_init__()

    def _create_fluid(self) -> ember.fluid.PerfectFluid:
        """Create the PerfectFluid instance."""
        return ember.fluid.PerfectFluid(
            cp=self.cp,
            gamma=self.gamma,
            mu=self.mu,
            Pr=self.Pr,
            P_dtm=self.P_dtm,
            T_dtm=self.T_dtm,
            V_ref=self.V_ref,
            rho_ref=self.rho_ref,
            Rgas_ref=self.rgas_ref,
        )
