"""Config class using ember data structures."""

import dataclasses

import turbigen.fluid
import turbigen.inlet
import turbigen.dspace
import turbigen.job
import turbigen.iterators
import turbigen.yaml_utils

from typing import List

from pathlib import Path


@dataclasses.dataclass
class TurbigenConfig:
    """Top level configuration class for turbigen.

    A run is uniquely defined by an instance of this class.

    """

    workdir: Path
    """Directory in which to store run data."""

    fluid: turbigen.fluid.FluidConfig
    """Equation of state."""

    inlet: turbigen.inlet.InletConfig
    """Inflow boundary conditions."""

    # mean_line: turbigen.meanline_design.MeanLineDesigner
    # """Settings for the mean-line designer."""

    iterate: List[turbigen.iterators.IteratorConfig] = dataclasses.field(
        default_factory=list
    )
    """Iterators to modify the configuration after running."""

    design_space: turbigen.dspace.DesignSpace = None
    """Design space sampling and mapping."""

    job: turbigen.job.BaseJob = None
    """Queue job submission."""

    def __post_init__(self):
        """Ensure correct types after init."""

        self.workdir = Path(self.workdir).absolute()

    def to_dict(self):
        """Convert the config to a dictionary."""

        # Built-in dataclasses method gets us most of the way there
        data = dataclasses.asdict(self)

        return data

    def save(self, fname):
        """Write out to a file."""
        turbigen.yaml_utils.write_yaml(self.to_dict(), fname)
