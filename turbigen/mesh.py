"""Classes to perform meshing."""

from abc import ABC, abstractmethod
import dataclasses
import numpy as np


@dataclasses.dataclass
class Mesher(ABC):
    """Base class for meshing."""

    yplus: float = 30.0

    meshdir: str = "mesh"

    @abstractmethod
    def make_grid(workdir, machine, dhub, dcas, dsurf):
        """Generate a mesh for the given configuration.

        Parameters
        ----------
        machine:
            Machine geometry object.
        dhub: float
            Wall cell size at hub [m].
        dcas: float
            Wall cell size at casing [m].
        dsurf: (nrow,) array
            Wall cell sizes on each blade surface [m].
        """
        raise NotImplementedError

    def get_dwall(self, mean_line, ell):
        """Estimate wall spacing for a single blade row."""

        # Use flat plate correlations to get viscous length scale
        Re_surf = ell / mean_line.L_visc
        Cf = (2.0 * np.log10(Re_surf) - 0.65) ** -2.3
        tauw = Cf * 0.5 * (mean_line.rho_ref * mean_line.V_ref**2)
        Vtau = np.sqrt(tauw / mean_line.rho_ref)
        Lvisc = (mean_line.mu_ref / mean_line.rho_ref) / Vtau

        return (self.yplus * Lvisc).squeeze()
