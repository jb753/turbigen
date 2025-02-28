"""Configuration for setting blade number."""

import dataclasses
from abc import ABC, abstractmethod


class BladeNumberConfig(ABC):
    @abstractmethod
    def get_blade_number(self, mean_line, blade):
        """Calculate number of blades for a mean line flow field and blade geometry."""
        raise NotImplementedError

    def to_dict(self):
        return dataclasses.asdict(self)


@dataclasses.dataclass
class Nb(BladeNumberConfig):
    """Directly specify the number of blades."""

    Nb: int
    """Number of blades."""

    def get_blade_number(self, mean_line, blade):
        del mean_line, blade
        """Return the fixed number of blades."""
        return self.Nb


# @dataclasses.dataclass
# class DiffusionFactorConfig:
#     """Settings for calculating diffusion factor."""
#
#     spf: float = 0.5
#     """Span fraction at which to calculate diffusion factor."""
#
#     target: dict = dataclasses.field(default_factory=lambda: ({}))
#     """Mapping of row index to target diffusion factors."""
#
#     dNb_dDF: float = 0.5
#     """Factor to scale diffusion factor change to relative change in blade number."""
#
# def get_diffusion_factor(
#     grid,
#     machine,
#     meanline,
#     irow,
#     conf,
# ):
#     """Calculate diffusion factor for a blade in the machine."""
#
#     zeta_norm, Cp = turbigen.util_post.get_pressure_distribution(
#         grid, machine, meanline, irow, conf.spf
#     )
#
#     # Calculate diffusion factor
#     Cpmin = Cp.min()
#     Cpmax = Cp.max()
#     CpTE = 0.5 * (Cp[-1] + Cp[0]).item()
#     DCpmin = Cpmin - Cpmax
#     DCpTE = CpTE - Cpmax
#     DF = 1.0 - np.sqrt(DCpTE / DCpmin)
#     return DF
