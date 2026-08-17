"""Classes for initialising an inlet state."""

import dataclasses
import numpy as np


@dataclasses.dataclass
class InletConfig:
    """Inlet boundary condition settings."""

    spf: list = None
    """Span fraction of some radial stations running 0 to 1."""

    profiles: np.ndarray = None
    """Po, To, Alpha, Beta variation at each span fraction, shape (4,nspf)"""
