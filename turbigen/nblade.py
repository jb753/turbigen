"""Configuration for setting blade number."""

import dataclasses
from abc import ABC, abstractmethod
import numpy as np


class BladeNumberConfig(ABC):
    @abstractmethod
    def get_blade_number(self, mean_line, blade):
        """Calculate number of blades for a mean line flow field and blade geometry."""
        raise NotImplementedError

    def to_dict(self):
        return dataclasses.asdict(self)

    @abstractmethod
    def adjust(self, dNb_rel):
        """Adjust the blade number by a relative amount."""
        raise NotImplementedError


@dataclasses.dataclass
class Nb(BladeNumberConfig):
    """Directly specify the number of blades."""

    Nb: int
    """Number of blades."""

    def get_blade_number(self, mean_line, blade):
        del mean_line, blade
        """Return the fixed number of blades."""
        return self.Nb

    def adjust(self, dNb_rel):
        """Adjust the blade number by a relative amount."""
        self.Nb += int(dNb_rel * self.Nb)


@dataclasses.dataclass
class Co(BladeNumberConfig):
    """Use non-dimensional circulation to set number of blades."""

    Co: float
    """Circulation coefficient [--]."""

    spf: float = 0.5
    """Span fraction to take surface length from."""

    def get_blade_number(self, mean_line, blade):
        # Surface length of blade from geometry
        ell = blade.surface_length(self.spf)

        # Pitch to surface length ratio from mean line
        s_ell = mean_line.s_ell(self.Co)
        s = s_ell * ell

        # Take reference radius to be mean of LE and TE rrms
        rref = np.mean(mean_line.rrms)

        # Number of blades
        Nb = np.round(2.0 * np.pi * rref / s)

        return Nb

    def adjust(self, dNb_rel):
        """Adjust the blade number by a relative amount."""
        self.Co /= 1.0 + dNb_rel


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
