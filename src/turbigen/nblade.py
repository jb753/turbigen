"""Configuration for setting blade number."""

import dataclasses
from abc import ABC, abstractmethod
import numpy as np
from turbigen import util

logger = util.make_logger()


class BladeNumberConfig(ABC):
    @abstractmethod
    def get_blade_number(self, mean_line, blade):
        """Calculate number of blades for a mean line flow field and blade geometry.

        Parameters
        ----------
        mean_line : (2,) MeanLine
            Mean line flow field for a single blade row.
        blade :
            Blade geometry for one row.

        """
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


@dataclasses.dataclass
class DFL(BladeNumberConfig):
    """Use Lieblein diffusion factor to set number of blades."""

    DFL: float
    """Lieblein diffusion factor [--]. Typical value 0.45, flow separation above 0.6."""

    spf: float = 0.5
    """Span fraction to take chord length from."""

    def adjust(self, dNb_rel):
        raise NotImplementedError(
            "DFL adjustment not implemented. Use Co or Nb instead."
        )

    def get_blade_number(self, mean_line, blade):
        # Get velocities from mean-line
        assert len(mean_line.V_rel) == 2, "Need a single blade row"
        V1, V2 = mean_line.V_rel
        DVt = np.ptp(mean_line.Vt_rel)

        # Pitch to true chord ratio
        # Eqn. (3.32) Dixon and Hall
        logger.debug(f"V1={V1}, V2={V2}, DVt={DVt}")
        if (self.DFL + V2 / V1 - 1.0) < 0.0:
            raise Exception(
                f"V2/V1={V2 / V1} is too low for this DFL={self.DFL}, need DFL + V2/V1 > 1"
            )
        s_c = 2.0 * V1 / DVt * (self.DFL + V2 / V1 - 1.0)
        logger.debug(f"s_c={s_c}")

        # Calculate stagger angle
        # This assumes a quadratic camber line
        Al1 = mean_line.Alpha_rel[::2]
        Al2 = mean_line.Alpha_rel[1::2]
        stag = util.atand(0.5 * (util.tand(Al1) + util.tand(Al2)))
        logger.debug(f"stag={stag}")

        # Resolve s_c into axial direction
        s_cx = s_c / util.cosd(stag)

        # Pitch
        s = blade.chord(self.spf) / s_cx

        # Number of blades
        rrms = np.mean(mean_line.rrms)
        Nb = np.round(2.0 * np.pi * rrms / s).astype(int)

        return Nb


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
