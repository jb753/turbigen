"""The whole config file, as one object.

A config is a tree of :class:`~turbigen2.node.Node` values, so loading a file
is one recursive call and writing it back is its mirror image. It is frozen and
holds no results: designing returns a mean line rather than storing one, so a
config never accumulates state from a run.
"""

from pathlib import Path

import turbigen.yaml_utils

from turbigen2 import plugins
from turbigen2.annulus import AnnulusDesign
from turbigen2.design import MeanLineDesign
from turbigen2.fluid import Fluid
from turbigen2.machine import Machine
from turbigen2.node import Node
from turbigen2.post import Post


class Config(Node):
    """A complete turbigen case."""

    fluid: Fluid
    """Working fluid and its equation of state."""

    mean_line: MeanLineDesign
    """Mean-line design."""

    annulus: AnnulusDesign = None
    """Annulus design. Omit it to design the mean line alone."""

    post_process: tuple[Post, ...] = ()
    """Post-processors to run. Nothing is added implicitly: what the config
    asks for is what runs."""

    @classmethod
    def from_file(cls, path):
        """Read a config from a YAML file.

        User-defined designs are discovered relative to the file, since
        resolving the ``type:`` keys it contains is exactly what needs the
        registry populated. `from_dict` does no discovery: it has no path to
        anchor a search on, and a caller working from a dict is already in
        Python and can import whatever it needs.
        """
        path = Path(path)
        plugins.discover(path.parent)
        return cls.from_dict(turbigen.yaml_utils.read_yaml(path))

    def to_file(self, path):
        """Write this config to a YAML file, defaults included."""
        turbigen.yaml_utils.write_yaml(self.to_dict(), Path(path))

    def design(self) -> Machine:
        """Return the machine this config describes.

        Every configured stage runs. There is no stop point to pass: the depth
        of a design is set by what the config contains, so a config with no
        annulus designs only a mean line.
        """
        mean_line = self.mean_line.design(self.fluid)

        annulus = None
        if self.annulus is not None:
            annulus = self.annulus.design(mean_line)

        return Machine(mean_line=mean_line, annulus=annulus)
