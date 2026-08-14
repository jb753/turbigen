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

import numpy as np

import turbigen.util
from turbigen2.annulus import Annulus
from turbigen2.blade import Blade
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

    annulus: Annulus | None = None
    """Annulus geometry, if the config asked for one."""

    blades: tuple[Blade, ...] = ()
    """Blade geometry, one per row, if the config asked for any."""

    def blade_string(self):
        """Tabular string representation of the blades."""
        r_ref = 0.5 * (self.mean_line.r[0] + self.mean_line.r[1])
        n_blade = np.array([blade.n_blade for blade in self.blades])
        chord = np.array([blade.chord(0.5) for blade in self.blades])
        properties = [
            ("N_blade", n_blade, "d"),
            ("Gap/m", np.array([blade.tip_gap for blade in self.blades]), ".4f"),
            ("s/cm", 2.0 * np.pi * r_ref / n_blade / chord, ".3f"),
        ]
        return turbigen.util.format_table(
            "Blades:", len(self.blades), properties, paired=False
        )

    def to_string(self):
        """Tabular summary of every stage that was designed."""
        parts = [self.mean_line.to_string()]
        if self.annulus is not None:
            parts.append(self.annulus.to_string())
        if self.blades:
            parts.append(self.blade_string())
        return "\n".join(parts)
