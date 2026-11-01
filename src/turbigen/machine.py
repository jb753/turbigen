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
import functools

import numpy as np

import turbigen.util
from turbigen.annulus import Annulus
from turbigen.blade import Row
from turbigen.meanline import MeanLine
from turbigen.spanwise import SpanwiseDesign, VortexExpon

CUT_OFFSET = 0.05
"""Cut planes sit this fraction of blade chord into the gap, clear of the row.

Cutting exactly at a leading or trailing edge would put the plane inside the
blade, where there is no single annulus-spanning surface to integrate over.
Clear of the row includes the H-mesh trailing edge cusp, which runs on
``AR_cusp`` trailing edge thicknesses past the blade: a plane through it cuts
less than a pitch of fluid. Measured from the furthest a swept edge reaches.
"""

N_EDGE_SPF = 41
"""Span fractions a swept row's edges are searched over for the furthest each
reaches, besides its sections'."""


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

    spanwise: SpanwiseDesign = VortexExpon()
    """How the flow the blades were designed against varies over the span."""

    @functools.cached_property
    def _edge_limits(self):
        """Furthest upstream leading edge and furthest downstream trailing edge
        of each row over the span, in row meridional coordinates; the stations,
        0 and 1, where a row is not swept."""
        limits = []
        for row in self.rows:
            blade = row.blade
            if not np.any(blade.placement[:, :2]):
                limits.append((0.0, 1.0))
                continue
            spf = np.union1d(np.linspace(0.0, 1.0, N_EDGE_SPF), blade.spf)
            edges = np.array([blade.evaluate_swept_m([0.0, 1.0], s) for s in spf])
            limits.append((float(edges[:, 0].min()), float(edges[:, 1].max())))
        return tuple(limits)

    def cut_planes(self, offset=None):
        """Return the meridional cut curve at each design station.

        A cut plane is the straight hub-to-casing line at one meridional
        position, `offset` blade chords upstream of a row's leading edge or
        downstream of its trailing edge. A swept edge reaches past its station
        at some spans, and the plane clears the furthest it reaches. The mean
        line is reduced between these planes, so a quantity integrated over the
        same bounds is directly comparable with the loss the mean line reports.

        Parameters
        ----------
        offset : float, optional
            Distance into the adjacent gap, as a fraction of blade chord at
            mid-span; :data:`CUT_OFFSET` if not given.

        Returns
        -------
        list of ndarray
            One ``(2, 2)`` array per station, in streamwise order, each holding
            two ``(x, r)`` points, hub then casing. This is the shape
            :func:`ember.cut.unstructured` takes.

        """
        annulus = self.annulus
        return [annulus.evaluate_xr(m, [0.0, 1.0]).T for m in self._m_cut(offset)]

    def _m_cut(self, offset=None):
        """Annulus meridional coordinate of each cut plane; see
        :meth:`cut_planes`."""
        if offset is None:
            offset = CUT_OFFSET
        annulus = self.annulus
        n_row = annulus.n_row
        limits = self._edge_limits or ((0.0, 1.0),) * n_row
        chords = annulus.evaluate_chords(0.5)

        # The offset is given in blade chords but applied to `m`, which is
        # normalised per segment -- so it has to be rescaled by the chord of
        # the gap each station opens into. Rows are the odd segments, gaps the
        # even.
        m_cut = np.empty(2 * n_row)
        for i_row, (m_LE, m_TE) in enumerate(limits):
            step = offset * chords[2 * i_row + 1]
            m_row = 2.0 * i_row + 1.0
            m_cut[2 * i_row] = m_row + m_LE - step / chords[2 * i_row]
            m_cut[2 * i_row + 1] = m_row + m_TE + step / chords[2 * i_row + 2]

        return m_cut

    def check_cuts(self):
        """Raise unless the default cut planes fit, inside the annulus and in
        streamwise order, clear of every row's furthest-reaching edges.

        A cut further out, as an exit profile is read at, may cross the next
        row's; that is the caller's to want. These are the cuts the mean line
        is reduced on, and swept edges can leave no room for them.
        """
        m_cut = self._m_cut()
        m_max = self.annulus.m_max
        if m_cut[0] <= 0.0 or m_cut[-1] >= m_max:
            raise ValueError(
                f"The cut planes {CUT_OFFSET} chords clear of the rows leave the "
                f"annulus, at m={m_cut[0]:.4g} and {m_cut[-1]:.4g} of 0 to "
                f"{m_max:.4g}."
            )
        for i_row in range(len(m_cut) // 2 - 1):
            if m_cut[2 * i_row + 1] >= m_cut[2 * i_row + 2]:
                raise ValueError(
                    f"Rows {i_row} and {i_row + 1} leave no room for cut planes "
                    f"{CUT_OFFSET} chords clear of both: their edges reach too "
                    "far into the gap between them."
                )

    def Re_surf(self):
        """Return the surface Reynolds number of each blade row [--].

        A property of the design and nothing else: a surface length off the
        blade, a characteristic state off the mean line, no grid and no
        solution. It sat on the mesher until an iterator wanted it too, which
        is the usual sign that a quantity belongs to the thing it is measured
        from rather than to its first consumer.
        """
        ell = np.array([row.blade.evaluate_surface_length(0.5)[0] for row in self.rows])
        stations = [
            self.mean_line.get_characteristic_station(i) for i in range(len(self.rows))
        ]
        L_visc = np.array([st.mu / st.rho / st.V_rel for st in stations])
        return ell / L_visc

    def blade_string(self):
        """Tabular string representation of the blade rows."""
        r_ref = 0.5 * (self.mean_line.r[0] + self.mean_line.r[1])
        n_blade = np.array([row.n_blade for row in self.rows])
        chord = np.array([row.blade.evaluate_chord(0.5) for row in self.rows])
        properties = [
            ("N_blade", n_blade, "d"),
            ("Gap/m", np.array([row.tip_gap for row in self.rows]), ".4f"),
            ("s/cm", 2.0 * np.pi * r_ref / n_blade / chord, ".3f"),
            ("Re_s/1e5", self.Re_surf() / 1e5, ".2f"),
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
