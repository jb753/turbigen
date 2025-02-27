"""Classes to iteratively update a mean-line design in response to a CFD solution."""

from abc import ABC, abstractmethod
import turbigen.util_post
import dataclasses


class BaseIterator(ABC):
    """Define the interface for an iterator."""

    _Config = None

    def __init__(self, mean_line, machine, iter_config, config):
        """Initialise with nominal mean line, machine geometry, and config dict."""
        self.mean_line_nominal = mean_line
        self.machine = machine
        self.config = config
        self.iter_config = self._Config(**iter_config)

    @abstractmethod
    def update(self, grid, mean_line_cfd):
        """Update the mean line in response to a CFD solution."""
        raise NotImplementedError


@dataclasses.dataclass
class PeakSuctionConfig:
    """Settings for peak suction iteration."""

    spf: float = 0.5
    """Span fraction at which to calculate peak suction location."""

    target: dict = dataclasses.field(default_factory=lambda: ({}))
    """Mapping of row index to target peak suction location."""

    K: float = 2.0
    """Factor to scale xpeak error to qcamber[2] change."""
