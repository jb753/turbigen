"""Classes to perform meshing."""

import dataclasses
from abc import ABC, abstractmethod


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
