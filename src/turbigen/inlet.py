"""Classes for initialising an inlet state."""

import dataclasses
import numpy as np


@dataclasses.dataclass
class InletConfig:
    """Inlet boundary condition settings.

    Note that Po and To are specified here because they are usually
    somewhat arbitrary in a non-dimensional sense. However, the flow angles
    Alpha and Beta, and velocity are determined by the mean-line design and
    are not specified here.

    """

    spf: list = None
    """Span fraction of some radial stations running 0 to 1."""

    profiles: np.ndarray = None
    """Po, To, Alpha, Beta variation at each span fraction, shape (4,nspf)"""
