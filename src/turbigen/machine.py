"""The result of designing a machine.

A :class:`Machine` collects what the design stages produce. It is deliberately
not a :class:`~turbigen.node.Node`: it never appears in a config file, has no
``type``, and needs no serialisation. Keeping results out of the config objects
is the point of the architecture --- the package this replaces held its nominal
and actual mean lines on the config itself, and wrote an annulus designer's
fitted splines onto the designer, so a config *was* its result and could only
ever hold one.
"""

import dataclasses

import numpy as np

import turbigen.util
from turbigen.annulus import Annulus
from turbigen.blade import Row
from turbigen.meanline import MeanLine


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

    rows: tuple[Row, ...] = ()
    """Blade rows, if the config asked for any: a shape, a count and a
    clearance each."""

    def Re_surf(self):
        """Return the surface Reynolds number of each blade row [--].

        A property of the design and nothing else: a surface length off the
        blade, a reference state off the mean line, no grid and no solution.
        It sat on the mesher until an iterator wanted it too, which is the
        usual sign that a quantity belongs to the thing it is measured from
        rather than to its first consumer.
        """
        ell = np.array([row.blade.surface_length(0.5) for row in self.rows])
        ref = [self.mean_line.ref(i) for i in range(len(self.rows))]
        L_visc = np.array([st.mu / st.rho / st.V_rel for st in ref])
        return ell / L_visc

    def blade_string(self):
        """Tabular string representation of the blade rows."""
        r_ref = 0.5 * (self.mean_line.r[0] + self.mean_line.r[1])
        n_blade = np.array([row.n_blade for row in self.rows])
        chord = np.array([row.blade.chord(0.5) for row in self.rows])
        properties = [
            ("N_blade", n_blade, "d"),
            ("Gap/m", np.array([row.tip_gap for row in self.rows]), ".4f"),
            ("s/cm", 2.0 * np.pi * r_ref / n_blade / chord, ".3f"),
        ]
        return turbigen.util.format_table(
            "Blades:", len(self.rows), properties, paired=False
        )

    def to_string(self):
        """Tabular summary of every stage that was designed."""
        parts = [self.mean_line.to_string()]
        if self.annulus is not None:
            parts.append(self.annulus.to_string())
        if self.rows:
            parts.append(self.blade_string())
        return "\n".join(parts)
