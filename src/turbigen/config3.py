"""Config class using ember data structures."""

import dataclasses

import turbigen.fluid
import turbigen.inlet
import turbigen.dspace
import turbigen.job
import turbigen.iterators
import turbigen.yaml_utils
import turbigen.meanline_new

from typing import List

from pathlib import Path


@dataclasses.dataclass
class TurbigenConfig:
    """Top level configuration class for turbigen.

    A run is uniquely defined by an instance of this class.

    """

    work_dir: Path
    """Directory in which to store run data."""

    fluid: turbigen.fluid.FluidConfig
    """Equation of state."""

    inlet: turbigen.inlet.InletConfig
    """Inflow boundary conditions."""

    mean_line: turbigen.meanline_new.MeanLineConfig
    """Settings for the mean-line designer."""

    plug_dir: Path = None
    """Directory in which to store run data."""

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

        self.work_dir = Path(self.work_dir).absolute()

        if self.plug_dir:
            self.plug_dir = Path(self.plug_dir).absolute()

        self.fluid = turbigen.fluid.FluidConfig.from_dict(self.fluid)
        self.inlet = turbigen.inlet.InletConfig(**self.inlet)
        self.mean_line = turbigen.meanline_new.MeanLineConfig.from_dict(self.mean_line)

    def to_dict(self):
        """Convert the config to a dictionary."""

        # Built-in dataclasses method gets us most of the way there
        data = dataclasses.asdict(self)

        # Now convert any nested objects with to_dict methods
        data["mean_line"] = self.mean_line.to_dict()
        data["fluid"] = self.fluid.to_dict()

        return data

    def save(self, fname):
        """Write out to a file."""
        turbigen.yaml_utils.write_yaml(self.to_dict(), fname)
