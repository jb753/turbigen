"""Classes to perform meshing."""

from abc import ABC, abstractmethod
import dataclasses


@dataclasses.dataclass
class Mesher(ABC):
    """Base class for meshing."""

    @abstractmethod
    def make_grid(machine, dhub, dcas, dsurf, unbladed):
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
        unbladed: (nrow,) bool array
            True where the row is unbladed.
        """
        raise NotImplementedError
