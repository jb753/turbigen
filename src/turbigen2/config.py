"""The whole config file, as one object.

A config is a tree of :class:`~turbigen2.node.Node` values, so loading a file
is one recursive call and writing it back is its mirror image. It is frozen and
holds no results: designing returns a mean line rather than storing one, so a
config never accumulates state from a run.
"""

from pathlib import Path

import turbigen.yaml_utils

from turbigen2.design import MeanLineDesign
from turbigen2.fluid import Fluid
from turbigen2.node import Node


class Config(Node):
    """A complete turbigen case."""

    fluid: Fluid
    """Working fluid and its equation of state."""

    mean_line: MeanLineDesign
    """Mean-line design."""

    @classmethod
    def from_file(cls, path):
        """Read a config from a YAML file."""
        return cls.from_dict(turbigen.yaml_utils.read_yaml(Path(path)))

    def to_file(self, path):
        """Write this config to a YAML file, defaults included."""
        turbigen.yaml_utils.write_yaml(self.to_dict(), Path(path))

    def design(self):
        """Return the mean line this config describes."""
        return self.mean_line.design(self.fluid)
