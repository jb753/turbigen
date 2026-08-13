"""The result of designing a machine.

A :class:`Machine` collects what the design stages produce. It is deliberately
not a :class:`~turbigen2.node.Node`: it never appears in a config file, has no
``type``, and needs no serialisation. Keeping results out of the config objects
is the point of the architecture --- the package this replaces held its nominal
and actual mean lines on the config itself, and wrote an annulus designer's
fitted splines onto the designer, so a config *was* its result and could only
ever hold one.
"""

import dataclasses

from turbigen2.annulus import Annulus
from turbigen2.meanline import MeanLine


@dataclasses.dataclass(frozen=True)
class Machine:
    """Everything the design stages produced.

    Named after :class:`turbigen.geometry.Machine`, which collects the same
    sort of thing for the mesher, but widened: this one holds the mean line
    too, and will grow blade count, tip gaps and splitters as those stages are
    ported. The two are not interchangeable while both exist.
    """

    mean_line: MeanLine
    """Flow field and annulus areas along the mean line."""

    annulus: Annulus = None
    """Annulus geometry, if the config asked for one."""

    def to_string(self):
        """Tabular summary of every stage that was designed."""
        parts = [self.mean_line.to_string()]
        if self.annulus is not None:
            parts.append(self.annulus.to_string())
        return "\n".join(parts)
