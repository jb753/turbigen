"""The whole config file, as one object.

A config is a tree of :class:`~turbigen2.node.Node` values, so loading a file
is one recursive call and writing it back is its mirror image. It is frozen and
holds no results: designing returns a mean line rather than storing one, so a
config never accumulates state from a run.
"""

from pathlib import Path

import ember.yaml_util

from turbigen2 import plugins
from turbigen2.annulus import AnnulusDesign
from turbigen2.blade import BladeDesign
from turbigen2.design import MeanLineDesign
from turbigen2.fluid import Fluid
from turbigen2.machine import Machine
from turbigen2.mesh import Mesher
from turbigen2.node import Node
from turbigen2.post import Post
from turbigen2.solver import Solver


class Config(Node):
    """A complete turbigen case."""

    fluid: Fluid
    """Working fluid and its equation of state."""

    mean_line: MeanLineDesign
    """Mean-line design."""

    annulus: AnnulusDesign | None = None
    """Annulus design. Omit it to design the mean line alone."""

    blades: tuple[BladeDesign, ...] = ()
    """Blade designs, one per row. Omit them to design the annulus alone."""

    mesh: Mesher | None = None
    """Mesh generation. Only needed by the verbs that make a grid."""

    solver: Solver | None = None
    """Flow solver. Only needed by the verbs that solve."""

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

        data = ember.yaml_util.read_yaml(path)
        # A run writes its answer into the same file under `result:`. Ignore it
        # here: this returns a config. `turbigen2.case.read` returns both.
        data.pop("result", None)

        return cls.from_dict(data)

    def to_file(self, path):
        """Write this config to a YAML file, defaults included."""
        ember.yaml_util.write_yaml(self.to_dict(), Path(path))

    def design(self) -> Machine:
        """Return the machine this config describes.

        Every configured stage runs. There is no stop point to pass: the depth
        of a design is set by what the config contains, so a config with no
        annulus designs only a mean line.

        This is the only verb that lives here, because it is the only one that
        is a property of the config alone. Anything combining a config with a
        result -- meshing a machine, running a grid -- is a function of both,
        and `config.mesh.mesh(machine)` already says it without a second
        spelling on `Config`.
        """
        mean_line = self.mean_line.design(self.fluid)

        annulus = None
        if self.annulus is not None:
            annulus = self.annulus.design(mean_line)

        rows = ()
        if self.blades:
            if annulus is None:
                raise ValueError("Blades need an annulus to sit in.")
            if len(self.blades) != mean_line.n_row:
                raise ValueError(
                    f"Expected one blade per row, but got {len(self.blades)} "
                    f"blades for {mean_line.n_row} rows."
                )
            rows = tuple(
                blade.design(mean_line.row(i_row), annulus.row(i_row))
                for i_row, blade in enumerate(self.blades)
            )

        return Machine(mean_line=mean_line, annulus=annulus, rows=rows)
