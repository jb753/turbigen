"""Post-processing.

A post-processor is a :class:`~turbigen.node.Node`, so it is configured like
any other part of the file and needs no registration or bespoke serialisation.
It *returns* figures rather than drawing into a shared document:

```python
class Post(Node):
    def report(self, config, result) -> list[Figure]
```

That is what lets one be run alone in a notebook, lets the caller decide
whether the figures become a PDF, and lets a test assert on a figure without
touching the filesystem. The package this replaces threads a `PdfPages` through
every processor, so none of those are possible.

It takes the config as well as the result, because the most useful plots
compare design intent against what was achieved, and intent lives in the
config. That is safe here only because a `Config` is frozen: the hazard in the
old interface is not reading the config but that its post-processors call
`config.apply_recamber()` and `config.undo_recamber()`, mutating the geometry
from inside a plot.
"""

import contextlib
import importlib.resources
import logging
from pathlib import Path
from typing import ClassVar

import numpy as np

import ember.average
import ember.block_util
import ember.cut
import ember.util
import turbigen.util
from turbigen.node import Node

logger = logging.getLogger("turbigen")

_STYLE = importlib.resources.files("turbigen") / "turbigen.mplstyle"
"""Shipped plotting style, applied by :func:`styled`."""

N_CHORD_PLOT = 501
"""Chordwise points to draw a blade section with.

Far fewer than the ten thousand `evaluate_section` defaults to, which is a
resolution for geometry and not for a line on a page.
"""

N_SPAN_CUT = 101
"""Meridional points defining the span curve a blade surface is cut along.

Only the placement of the cut surface depends on this: `structured_meridional`
walks the grid's own gridlines, so the resolution of what comes back is the
mesh's, not the curve's.
"""

N_SEGMENT_CUT = 50
"""Meridional points per annulus segment when cutting the whole machine."""

N_SPAN_ANNULUS = 10
"""Spanwise stations for the blade outline drawn on the annulus plot.

A meridional view flattens the section away, so this only has to be enough to
trace where a row sits and how it leans -- not to resolve a curve.
"""

LABELS = {
    "Ma": r"Mach Number, $\mathit{Ma}$",
    "Ma_rel": r"Relative Mach Number, $\mathit{Ma}^\mathrm{rel}$",
    "P": r"Static Pressure, $p$/Pa",
    "Po": r"Stagnation Pressure, $p_0$/Pa",
    "T": r"Static Temperature, $T$/K",
    "s": r"Specific Entropy, $s$/J kg$^{-1}$K$^{-1}$",
}
"""Axis labels for the block properties worth contouring, by attribute name."""

_SPANWISE_LABELS = {
    "Ys": r"Entropy Loss Coefficient, $Y_s$",
    "Cp": r"Static Pressure, $C_p$",
    "Cpo": r"Stagnation Pressure, $C_{p0}$",
    "Cho": r"Stagnation Enthalpy, $C_{h0}$",
    "Vm": r"Meridional Velocity, $V_m/{sym}$",
    "Vt": r"Circumferential Velocity, $V_\theta/{sym}$",
}
"""Axis labels for the quantities `SpanwisePlot` will average, by name.

Separate from `LABELS` because none of these are block properties: they are
coefficients built from a cut and a mean line, so nothing can look one up the
way `ContourPlot._values` looks up what it contours. ``{sym}`` is filled in
with the symbol of whatever reference velocity was available, by
`SpanwisePlot._profile`.
"""


@contextlib.contextmanager
def styled():
    """Draw with turbigen's defaults while the user's own rc still wins.

    Layers, lowest first: matplotlib's built-in defaults, turbigen's shipped
    style, then only the keys the user set explicitly in their own
    matplotlibrc. So an override behaves exactly as it would without turbigen,
    and everything left alone gets turbigen's look.

    Scoped rather than global: entering this does not disturb the matplotlib
    state a notebook caller is already working with. `write_report` wraps the
    whole report in it; a notebook running one processor by hand can do the
    same.
    """
    import matplotlib as mpl  # noqa: PLC0415
    import matplotlib.pyplot as plt  # noqa: PLC0415
    from matplotlib.style.core import STYLE_BLACKLIST  # noqa: PLC0415

    with importlib.resources.as_file(_STYLE) as style_path:
        layers = [str(style_path)]

        # matplotlib has already loaded whichever rc file it found into
        # rcParams; re-apply just the lines it actually contained, so they sit
        # back on top of our style. The install default is skipped -- layering
        # it would only overwrite the style with matplotlib's own defaults --
        # and the handful of keys a style may not carry (backend and friends)
        # are dropped, since they are already in effect and warn if restated.
        user_rc = mpl.matplotlib_fname()
        default_rc = Path(mpl.get_data_path()) / "matplotlibrc"
        if user_rc and Path(user_rc) != default_rc:
            explicit = mpl.rc_params_from_file(
                user_rc, fail_on_error=False, use_default_template=False
            )
            layers.append(
                {k: v for k, v in explicit.items() if k not in STYLE_BLACKLIST}
            )

        with plt.style.context(layers):
            yield


class Post(Node):
    """Base for post-processors."""

    def report(self, config, result):
        """Return figures describing `result`.

        Parameters
        ----------
        config : Config
            The configuration that was run, for design intent.
        result : Result
            What designing and running produced.

        Returns
        -------
        list of matplotlib.figure.Figure
            Empty if there was nothing to plot, which is not an error.

        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement report(self, config, result)"
        )


def _blade_annulus_lines(blade):
    """Return meridional ``(x, r)`` polylines outlining a blade on the annulus.

    Four lines per row -- leading edge, trailing edge, and the two diagonals of
    the passage -- each traced hub to casing along the blade's camber surface.
    Enough to show where a row sits and how it is staggered without drawing the
    sections themselves. Ported from the annulus plot of the package this
    replaces, which had to recamber the blade first; a `Blade` is already the
    shape it will be meshed as.
    """
    spf = np.linspace(0.0, 1.0, N_SPAN_ANNULUS)

    # Chordwise position of each line at every span station: leading edge at 0,
    # trailing edge at 1, and the two diagonals crossing between them.
    m = np.stack((np.zeros_like(spf), spf, 1.0 - spf, np.ones_like(spf)))

    # Camber-line xr is the mean of the two surfaces. evaluate_section takes a
    # scalar span fraction, so the stations are walked one at a time.
    xr = np.stack(
        [
            np.mean(blade.evaluate_section(s, m=m[:, j]), axis=0)[:2]
            for j, s in enumerate(spf)
        ]
    )
    # (station, coord, line) -> (line, coord, station)
    return xr.transpose(2, 1, 0)


class AnnulusPlot(Post):
    """Meridional view of the annulus."""

    type: ClassVar[str] = "annulus"

    m_cut: tuple[float, ...] = ()
    """Normalised meridional positions at which to draw a cut plane."""

    show_axis: bool = False
    """Draw the axis of rotation."""

    show_blades: bool = True
    """Outline each blade row with its leading and trailing edges and diagonals."""

    def report(self, config, result):
        annulus = result.machine.annulus
        if annulus is None:
            logger.info("No annulus was designed, skipping the annulus plot.")
            return []

        # Imported here so that turbigen can be used without a display, and
        # without paying for matplotlib when nothing is being plotted.
        import matplotlib.pyplot as plt  # noqa: PLC0415

        fig, ax = plt.subplots(layout="constrained")
        ax.axis("off")
        ax.axis("equal")

        # Sample the hub and casing straight from the annulus. Everything a
        # meridional view needs is a view on evaluate_xr, so nothing is added
        # to Annulus to support this: shapes wanted by one consumer belong in
        # that consumer.
        m = np.linspace(0.0, annulus.m_max, annulus.n_segment * 50 + 1)
        xr_hub = annulus.evaluate_xr(m, 0.0)
        xr_cas = annulus.evaluate_xr(m, 1.0)

        # A cut plane is the hub-to-casing line at one meridional position
        for m_cut in self.m_cut:
            ax.plot(*annulus.evaluate_xr(m_cut, [0.0, 1.0]), "-", color="C0")

        # Blade rows, drawn before the hub and casing so those stay on top.
        if self.show_blades:
            for row in result.machine.rows:
                for x, r in _blade_annulus_lines(row.blade):
                    ax.plot(x, r, "-", color="0.4")

        ax.plot(*xr_hub, "k-")
        ax.plot(*xr_cas, "k-")

        # Dotted hub-to-casing lines closing the annulus at inlet and exit.
        for m_end in (0.0, annulus.m_max):
            ax.plot(*annulus.evaluate_xr(m_end, [0.0, 1.0]), "k:")

        if self.show_axis:
            ax.plot(xr_hub[0, (0, -1)], np.zeros(2), "k-.")

        ax.set_title("Annulus")
        return [fig]


def _span_fractions(spf, blade):
    """Return span fractions to plot at, defaulting to the designed sections.

    Empty means "wherever this blade was defined", which is the only choice
    that needs no knowledge of the machine: a section the designer named is a
    section worth looking at.
    """
    return tuple(spf) if spf else tuple(float(s) for s in blade.spf)


class SectionsPlot(Post):
    """Blade-to-blade sections of each row."""

    type: ClassVar[str] = "sections"

    spf: tuple[float, ...] = ()
    """Span fractions to draw. Empty for the designed sections."""

    def report(self, config, result):
        rows = result.machine.rows if result.machine else ()
        annulus = result.machine.annulus if result.machine else None
        if not rows or annulus is None:
            logger.info("No blades were designed, skipping the sections plot.")
            return []

        import matplotlib.pyplot as plt  # noqa: PLC0415

        # One curve spanning the whole machine, unwrapped onto the conformal
        # (m', theta) plane exactly as the contour plot does it: angles and
        # aspect ratios are preserved, so a section keeps the shape it has in
        # the machine at any radius, and every row lands on one m' scale.
        m = np.linspace(0.0, annulus.m_max, annulus.n_segment * N_SEGMENT_CUT + 1)

        figures = []
        for i_row, row in enumerate(rows):
            fig, ax = plt.subplots(layout="constrained")
            ax.set_aspect("equal")
            ax.axis("off")
            ax.set_title(f"Row {i_row} Sections")

            # The geometry is drawn as it stands. The package this replaces has
            # to recamber the blade before it can plot one, and put it back
            # afterwards, because its sections are stored in an intermediate
            # form; here a Blade is already what it will be meshed as.
            mps, thetas = [], []
            for i_spf, spf in enumerate(_span_fractions(self.spf, row.blade)):
                # The datum curve follows the section's own span, so each one
                # sits in its true conformal plane.
                xr_curve = annulus.evaluate_xr(m, spf).T
                surfaces = row.blade.evaluate_section(spf, nchord=N_CHORD_PLOT)
                for i_surf, xrt in enumerate(surfaces):
                    mp = ember.util.unwrap_meridional(xr_curve, xrt[:2].T)
                    theta = xrt[2]
                    ax.plot(
                        mp,
                        theta,
                        color=f"C{i_spf}",
                        label=f"spf={spf:.2f}" if i_surf == 0 else None,
                    )
                    mps.append(mp)
                    thetas.append(theta)

            # Coordinate arrows below the blade, in place of the axes.
            mp_min = min(a.min() for a in mps)
            mp_max = max(a.max() for a in mps)
            th_min = min(a.min() for a in thetas)
            th_max = max(a.max() for a in thetas)
            span = mp_max - mp_min
            length = 0.32 * span
            gy = th_min - 0.1 * span - length
            _gnomon(ax, mp_min, gy, length, r"$m'$", r"$\theta$")
            ax.set_xlim(mp_min - 0.12 * span, mp_max + 0.05 * span)
            ax.set_ylim(gy - 0.1 * span, th_max + 0.05 * span)

            # Outside the axes on the right: the box is turned off and the
            # gnomon takes the corner, so there is nowhere clean for it inside.
            ax.legend(
                loc="upper left",
                bbox_to_anchor=(1.0, 1.0),
                borderaxespad=0.0,
                frameon=False,
            )
            figures.append(fig)

        return figures


def _arrow(ax, tail, tip, color, label=None, side=1):
    """Draw one to-scale vector in data coordinates, optionally labelled.

    annotate rather than quiver or arrow: it takes a tail and a tip straight
    from the mean line and does no scaling of its own, so the vector on the
    page is the velocity, measured against the coordinate arrows.

    A label sits at the midpoint, pushed clear of the shaft along its normal;
    `side` flips which way, for the vectors that would otherwise collide.
    """
    ax.annotate(
        "",
        xy=tip,
        xytext=tail,
        arrowprops={"arrowstyle": "-|>", "color": color, "linewidth": 1.5},
    )
    if label is None:
        return

    dx, dy = tip[0] - tail[0], tip[1] - tail[1]
    length = np.hypot(dx, dy) or 1.0
    ax.annotate(
        label,
        xy=(0.5 * (tail[0] + tip[0]), 0.5 * (tail[1] + tip[1])),
        xytext=(side * -dy / length * 9.0, side * dx / length * 9.0),
        textcoords="offset points",
        ha="center",
        va="center",
        color=color,
        fontsize="small",
    )


def _gnomon(ax, x0, y0, length, xlabel, ylabel):
    """Draw a pair of labelled coordinate arrows in place of the axes.

    Rooted at ``(x0, y0)``, one arrow to the right and one up, both `length`
    long in data units so they read at the same scale as whatever they sit
    beside.
    """
    _arrow(ax, (x0, y0), (x0 + length, y0), "0.3")
    _arrow(ax, (x0, y0), (x0, y0 + length), "0.3")
    pad = 0.15 * length
    ax.text(x0 + length, y0 - pad, xlabel, ha="center", va="top")
    ax.text(x0 - pad, y0 + length, ylabel, ha="right", va="center")


class VelocityTrianglePlot(Post):
    """Mean-line velocity triangles at inlet and exit of each row.

    Drawn from the mean line alone, so this is the one flow plot that has
    something to show at every pipeline depth -- a design that never meshed
    still has its triangles.

    The triangles are to scale against each other but carry no annulus or blade
    geometry: meridional velocity runs along x, swirl along y, and the stations
    are spread along x by a fixed pitch so neighbours do not overlap. The
    absolute velocity is C0 and the relative velocity C1; blade speed closes
    the two, tip to tip, and is only drawn where the row rotates.
    """

    type: ClassVar[str] = "triangle"

    def report(self, config, result):
        mean_line = result.machine.mean_line if result.machine else None
        if mean_line is None or not mean_line.n_row:
            logger.info("No mean line, skipping the velocity triangle plot.")
            return []

        import matplotlib.pyplot as plt  # noqa: PLC0415

        # Every station, in streamwise order: inlet then exit of each row.
        stations = []
        for i_row in range(mean_line.n_row):
            row = mean_line[:, i_row]
            for i_end, label in ((0, "inlet"), (1, "exit")):
                stations.append((f"Row {i_row} {label}", row[i_end]))

        # One pitch for every gap, wide enough that the longest triangle clears
        # its neighbour. Meridional velocity is the only thing that reaches
        # along x, and it is never negative on a mean line.
        Vm = np.array([float(st.Vm) for _, st in stations])
        pitch = 1.15 * Vm.max()

        # A row of triangles is a wide plot: widen the figure with the station
        # count rather than let the default squeeze the labels together.
        width, height = plt.rcParams["figure.figsize"]
        width = max(width, 1.2 * len(stations))
        fig, ax = plt.subplots(figsize=(width, height), layout="constrained")
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title("Velocity Triangles")

        for i_station, (label, st) in enumerate(stations):
            x0 = i_station * pitch
            vm = float(st.Vm)
            vt = float(st.Vt)
            vt_rel = float(st.Vt_rel)

            # Absolute velocity from the station origin.
            _arrow(ax, (x0, 0.0), (x0 + vm, vt), "C0", r"$V$")

            # Relative velocity and the blade speed that closes onto it, drawn
            # only where the frame actually rotates. Blade speed is purely
            # tangential, so it runs from the tip of the relative vector to the
            # tip of the absolute one; its label goes on the far side so it
            # clears the two arrowheads meeting there.
            if abs(float(st.Omega)) > 0.0:
                _arrow(ax, (x0, 0.0), (x0 + vm, vt_rel), "C1", r"$V^\mathrm{rel}$")
                _arrow(ax, (x0 + vm, vt_rel), (x0 + vm, vt), "k", r"$U$", side=-1)

        # annotate does not grow the data limits, so the arrows would fall
        # outside a default view. Framed from the vectors themselves.
        all_vt = np.concatenate(
            [
                [float(st.Vt) for _, st in stations],
                [float(st.Vt_rel) for _, st in stations],
                [0.0],
            ]
        )
        x_right = (len(stations) - 1) * pitch + Vm.max()
        y_bot = all_vt.min()

        # Station names on a single baseline below every triangle, standing in
        # for the axis that has been taken away.
        y_label = y_bot - 0.18 * pitch
        for i_station, (label, st) in enumerate(stations):
            ax.text(
                i_station * pitch + 0.5 * float(st.Vm),
                y_label,
                label,
                ha="center",
                va="top",
                fontsize="small",
            )

        # Coordinate arrows in place of the axes: meridional velocity to the
        # right, swirl up, at the same scale as everything else.
        gx = -0.7 * pitch
        _gnomon(ax, gx, y_label, 0.4 * pitch, r"$m$", r"$\theta$")

        ax.set_xlim(gx - 0.1 * pitch, x_right + 0.15 * pitch)
        ax.set_ylim(y_label - 0.15 * pitch, all_vt.max() + 0.15 * pitch)

        return [fig]


class ConvergencePlot(Post):
    """Residuals and integral errors over the course of a march."""

    type: ClassVar[str] = "convergence"

    def report(self, config, result):
        history = result.history
        if history is None:
            logger.info("No solution was marched, skipping the convergence plot.")
            return []
        if history.i_log < 0:
            logger.info("The march logged no records, skipping the convergence plot.")
            return []

        import matplotlib.pyplot as plt  # noqa: PLC0415
        import matplotlib.ticker as mticker  # noqa: PLC0415

        # A history arrives trimmed to the records actually written, so there
        # is no NaN tail to slice off here.
        steps = history.i_step

        fig_resid, ax = plt.subplots(layout="constrained")
        names = ("rho", "rhoVx", "rhoVr", "rhorVt", "rhoe")
        for i_var, name in enumerate(names):
            ax.semilogy(steps, history.residual[:, i_var], label=name)
        ax.set_xlabel("Step")
        ax.set_ylabel("Residual")
        ax.set_title("Convergence History")
        ax.legend()

        # All three panels as a percentage error, so they read on one scale
        # against one question: how far is each integral quantity from where
        # the march left it? The mass error is already a fraction of the mean
        # through-flow; work and loss are referred to their own final value,
        # (x / x[-1] - 1), so a settled march sits on zero. The residuals
        # carry the absolute story; these say how close to done it is.
        final_psi = history.psi[-1] or 1.0
        final_zeta = history.zeta[-1] or 1.0
        panels = (
            (100.0 * history.err_mdot, r"Mass, $\varepsilon$"),
            (100.0 * (history.psi / final_psi - 1.0), r"Work, $\psi$"),
            (100.0 * (history.zeta / final_zeta - 1.0), r"Loss, $\zeta$"),
        )

        # No axis labels: the title names the quantity and the ticks are plain
        # integer percentages, so a unit on the axis would only repeat them.
        fig_error, axs = plt.subplots(1, 3, layout="constrained")
        for axi, (y, title) in zip(axs, panels):
            axi.plot(steps, y)
            axi.set_title(title)

            # Symmetric about zero so the approach reads by eye, autoscaled to
            # the swing but never tighter than +-2%: a converged trace should
            # look flat, not fill the panel.
            finite = np.abs(y)[np.isfinite(y)]
            span = max(1.05 * float(finite.max()), 2.0) if finite.size else 2.0
            axi.set_ylim(-span, span)

            # Integer percentages, symmetric: -2 -1 0 1 2 at the floor, and a
            # coarser integer step once the swing outgrows it.
            axi.yaxis.set_major_locator(
                mticker.MaxNLocator(nbins=4, integer=True, symmetric=True)
            )

            # A march short enough to log once has nothing to span, and asking
            # for a zero-width axis is an error rather than an empty plot.
            if steps[-1] > steps[0]:
                axi.set_xlim(steps[0], steps[-1])

        return [fig_resid, fig_error]


def _isentropic_mach(cut, s_ref):
    """Return isentropic Mach number over `cut`, referred to entropy `s_ref`.

    Expanded isentropically from the row inlet entropy to the local static
    pressure, so the result reads as the Mach number the blade would see with
    no loss upstream of the point in question.
    """
    # Set in place on a copy, not chained off one: ember's setters return
    # nothing, whatever the idiom in the package this is ported from suggests.
    isen = cut.copy()
    isen.set_P_s(cut.P, s_ref)

    # Stagnation enthalpy and sound speed are taken as surface means so that
    # only local static pressure drives the distribution. Left local, radial
    # redistribution of ho_rel and variation in a split the two surfaces apart
    # at the trailing edge, where they must meet.
    ho = np.mean(cut.ho_rel)
    a_ref = np.mean(isen.a)

    # Shift so the lowest point sits exactly at rest rather than slightly
    # below it, which the discrete field can otherwise produce.
    hs = isen.h
    hs = hs + np.min(ho - hs)

    return np.sqrt(2.0 * np.maximum(ho - hs, 0.0)) / a_ref


def _normalise_surface_distance(cut, mas, xrt_nose):
    """Return surface distance in [-1, 1], zero at the stagnation point.

    Each surface is normalised by its own length, so both reach one at the
    trailing edge however asymmetric the blade is. The sign says which surface
    a point is on, following the direction the cut loops in; the plot folds it
    away, but normalising the two sides has to happen while they are still
    told apart.
    """
    zeta = turbigen.util.get_zeta(cut)[:, 0]

    # The geometric nose anchors the search window, which is more robust on
    # blades with a strongly asymmetric leading edge than the arc-length
    # midpoint the function falls back on. Whether it found a real maximum does
    # not matter here: the origin moves onto the lowest Mach number below in
    # any case, and this only has to land on the right side of the blade.
    i_stag = int(turbigen.util.get_i_stag(cut, xrt_LE=xrt_nose)[0][0])
    zeta = zeta - zeta[i_stag]

    # Then move the origin onto the lowest Mach number, which is the
    # stagnation point of the flow rather than of the grid.
    zeta = zeta - zeta[np.argmin(mas)]

    upper = zeta.max()
    lower = np.abs(zeta.min())
    return zeta / np.where(zeta > 0.0, upper or 1.0, lower or 1.0)


class SurfacePlot(Post):
    """Isentropic Mach number around the blade surfaces."""

    type: ClassVar[str] = "surface"

    spf: tuple[float, ...] = ()
    """Span fractions to plot at. Empty for the designed sections."""

    offset: int = 0
    """Cells away from the wall to take the distribution at."""

    def report(self, config, result):
        rows = result.machine.rows if result.machine else ()
        annulus = result.machine.annulus if result.machine else None
        if result.grid is None or not rows or annulus is None:
            logger.info("No solved grid, skipping the surface distribution plot.")
            return []

        # A march that blew up leaves a field full of NaN, which no isentropic
        # state can be evaluated from. Skipped on the solver's own verdict
        # rather than by inspecting the numbers: divergence is what the history
        # records. The convergence plot still draws -- a diverged run is
        # exactly the one whose residuals someone needs to look at.
        if result.history is not None and result.history.diverged:
            logger.info("The march diverged, skipping the surface distribution plot.")
            return []

        import matplotlib.pyplot as plt  # noqa: PLC0415

        # One cut of the whole blade per row, sliced at each span fraction
        # below: the cut is the expensive part and does not depend on span.
        surfaces = turbigen.util.cut_blade_surfs(result.grid, self.offset)

        # Whatever field the grid holds is what gets drawn: a solution, a
        # restored one, or the meridional guess a mesh starts with, which is a
        # legitimate way to look at a mesh.
        figures = []
        for i_row, row in enumerate(rows):
            if surfaces[i_row] is None:
                logger.info(
                    f"Could not cut the blade surface of row {i_row}, "
                    "skipping its surface distribution."
                )
                continue

            # `structured_meridional` walks the second axis of a three-axis
            # block, so the surface is padded to put its spanwise axis there
            # and the cut comes back one wide. Both halves are here rather
            # than split across the cut helper, because the shape is this
            # call's business and nothing else's.
            surface = surfaces[i_row][0][:, :, None]
            s_ref = result.machine.mean_line[:, i_row].s[0]

            fig, ax = plt.subplots(layout="constrained")
            ax.set_xlabel(r"Normalised Surface Distance, $|\zeta|$")
            ax.set_ylabel(r"Isentropic Mach Number, $\mathit{Ma}_s$")
            ax.set_title(f"Row {i_row} Surface Distribution")
            ax.set_xlim(0.0, 1.0)

            for spf in _span_fractions(self.spf, row.blade):
                # Rows occupy the odd meridional segments of the annulus, so
                # row i spans m from 2i+1 to 2i+2.
                m = np.linspace(2 * i_row + 1, 2 * i_row + 2, N_SPAN_CUT)
                xr = annulus.evaluate_xr(m, spf)
                cut = ember.cut.structured_meridional(surface, xr.T)

                # Above a clearance gap the blade has no surface to cut, the
                # span there being trimmed off as flow rather than wall. Asked
                # for a section that is not there, say so and draw the rest.
                if not len(cut):
                    logger.info(
                        f"Row {i_row} has no blade surface at spf={spf:.2f}, "
                        "skipping that section."
                    )
                    continue
                cut = cut[0]

                mas = _isentropic_mach(cut, s_ref)[:, 0]
                xrt_nose = row.blade.evaluate_section(spf, nchord=N_CHORD_PLOT)[0][:, 0]
                zeta = _normalise_surface_distance(cut, mas, xrt_nose)

                # Folded onto the positive axis, so both surfaces run from the
                # stagnation point at zero out to the trailing edge at one and
                # can be read against each other directly.
                ax.plot(np.abs(zeta), mas, label=f"spf={spf:.2f}")

            # Every section asked for was above the gap, so there is nothing on
            # the axes. An empty frame in the report is worse than no frame, and
            # a legend with no lines in it warns.
            if not ax.lines:
                plt.close(fig)
                continue

            ax.legend()
            figures.append(fig)

        return figures


class ContourPlot(Post):
    """Contours of a flow variable on a constant-span surface."""

    type: ClassVar[str] = "contour"

    spf: tuple[float, ...] = (0.5,)
    """Span fractions to cut at."""

    variable: str = "Ma_rel"
    """Block property to contour, e.g. ``Ma_rel``, ``P``, ``s``."""

    n_passage: int = 2
    """Passages to draw, repeated pitchwise."""

    n_level: int = 21
    """Upper bound on the number of filled bands. The band edges are rounded to
    a sensible step, so the actual count follows from that step and the range."""

    clip_percentile: float = 1.0
    """Percentile trimmed from each end of the field before the level range is
    rounded. A stagnation cell or a corner artefact should colour one pixel,
    not rescale the whole plot; 0 disables the trim."""

    cmap: str = "viridis"
    """Colour map to fill with."""

    def report(self, config, result):
        machine = result.machine
        annulus = machine.annulus if machine else None
        if result.grid is None or annulus is None:
            logger.info("No grid to cut, skipping the contour plot.")
            return []

        if result.history is not None and result.history.diverged:
            logger.info("The march diverged, skipping the contour plot.")
            return []

        import matplotlib.pyplot as plt  # noqa: PLC0415

        figures = []
        for spf in self.spf:
            # One curve spanning the whole machine, used twice: once to place
            # the cut surface, and once as the datum for the conformal
            # coordinate. Sharing it is what puts every row on a single
            # meridional scale, with no per-block offsets to reconcile.
            m = np.linspace(0.0, annulus.m_max, annulus.n_segment * N_SEGMENT_CUT + 1)
            xr_curve = annulus.evaluate_xr(m, spf).T

            cut = ember.cut.structured_meridional(result.grid, xr_curve)
            if not len(cut):
                logger.info(f"No block reaches spf={spf}, skipping its contour plot.")
                continue

            # Gathered before anything is drawn, so that every block and every
            # repeated passage shares one set of levels. Contoured as they came
            # they would each get their own, and the colours either side of a
            # mixing plane would mean different things.
            passages = []
            fields = []
            for block in cut:
                mp = ember.util.unwrap_meridional(xr_curve, block.xrt[..., :2])
                values = self._values(block)
                fields.append(values)
                # Only theta moves between passages, so the conformal
                # coordinate and the field are computed once and reused.
                for passage in ember.block_util.repeat_pitchwise(block, self.n_passage):
                    passages.append((mp, passage.t, values))

            levels, extend = self._levels(fields)

            figures.append(self._draw(plt, passages, levels, extend, f"spf={spf:.2f}"))

        return figures

    def _levels(self, fields):
        """Return rounded band edges for `fields`, and the ``extend`` for them.

        The range is taken from the ``clip_percentile``--``100 - clip_percentile``
        span of the pooled field, not its raw extremes, then rounded outward to
        a nice step. ``extend`` then reports whether any real value falls past
        the rounded edges -- which the percentile trim makes likely -- so the
        colourbar grows a triangle rather than the plot silently clipping.
        """
        import matplotlib.ticker as mticker  # noqa: PLC0415

        pooled = np.concatenate([field.ravel() for field in fields])
        lo, hi = np.percentile(
            pooled, [self.clip_percentile, 100 - self.clip_percentile]
        )
        levels = mticker.MaxNLocator(
            nbins=self.n_level, steps=[1, 2, 5, 10]
        ).tick_values(lo, hi)

        below = pooled.min() < levels[0]
        above = pooled.max() > levels[-1]
        extend = {(True, True): "both", (True, False): "min", (False, True): "max"}.get(
            (below, above), "neither"
        )
        return levels, extend

    def _draw(self, plt, passages, levels, extend, title):
        """Contour every passage of every row, on one set of axes.

        The whole machine in one frame, ducts included: rows drawn apart could
        not be read against one another, and where a wake leaves one row and
        arrives at the next is a thing the plot exists to show.
        """
        fig, ax = plt.subplots(layout="constrained")

        for mp, passage_theta, values in passages:
            filled = ax.contourf(
                mp, passage_theta, values, levels=levels, cmap=self.cmap, extend=extend
            )

            # Filled bands are drawn as separate polygons, so a vector backend
            # leaves a hairline of background between them. Giving each band an
            # edge in its own colour closes the seam.
            filled.set_edgecolor("face")
            filled.set_linewidth(0.05)

        # Equal aspect is not decoration: m' and theta are both dimensionless,
        # and scaling them alike is what makes the plane conformal, so a
        # section keeps the shape it has in the machine. Nothing is clipped, so
        # the limits are whatever was drawn and matplotlib finds them itself.
        ax.set_aspect("equal")

        # The plane carries its own scale -- m' and theta are dimensionless and
        # drawn conformal -- so the frame, ticks and axis labels only take room
        # from the field. The colourbar is the only quantitative key kept.
        ax.axis("off")
        ax.set_title(title)
        fig.colorbar(filled, label=LABELS.get(self.variable, self.variable), shrink=0.8)

        return fig

    def _values(self, block):
        """Return the variable to contour, as an array over `block`."""
        try:
            return np.asarray(getattr(block, self.variable))
        except AttributeError:
            raise ValueError(
                f"An ember block has no property {self.variable!r} to contour; "
                f"try one of {sorted(LABELS)}."
            ) from None


def _cut_row(m, n_row):
    """Return the row a cut at meridional position `m` is measured against.

    Rows are the odd segments of an annulus and gaps the even ones, so a cut
    inside a row belongs to it and there is nothing to decide. A cut in a gap
    is a choice, and it is attributed to the row *upstream* of it: a profile
    taken just past a trailing edge is read as what that row did, not as what
    the next one is about to be given. The inlet duct has no upstream row, so
    it falls to the first one.
    """
    segment = int(np.floor(m))
    if segment % 2:
        i_row = (segment - 1) // 2
    else:
        i_row = segment // 2 - 1
    return int(np.clip(i_row, 0, n_row - 1))


def _cut_spf(cut):
    """Return face-centred span fraction along a structured cut.

    A cut is not a patch, so `ember`'s own `spf` is out of reach and the span
    is measured from the cut's own meridional line: normalised arc length hub
    to casing. Face-centred, because everything `ember.average` returns is,
    and a profile plotted against nodal span would be off by half a cell.
    """
    # Index 0 of the cut runs meridionally and index 1 in theta, so any
    # constant-theta line carries the whole meridional extent.
    xr = np.asarray(cut.xrt[:, 0, :2])
    arc = np.concatenate(
        [[0.0], np.cumsum(np.sqrt((np.diff(xr, axis=0) ** 2).sum(axis=-1)))]
    )
    spf = arc / arc[-1]
    return 0.5 * (spf[:-1] + spf[1:])


class SpanwisePlot(Post):
    """Pitch-averaged flow quantity against span fraction, at a cut plane.

    Where a row loses, rather than how much: the mixed-out mean line reduces a
    station to one number and a blade-to-blade contour shows one span fraction
    at a time, so neither can say that the loss sits in a corner or at the tip.

    Everything is referred to `result.actual`, the mixed-out mean line the grid
    achieved, so the datum and the field being averaged are the same flow. A
    result that has not been mixed out yields no figures.
    """

    type: ClassVar[str] = "spanwise"

    m_cut: tuple[float, ...] = ()
    """Normalised meridional positions to cut and pitch-average at."""

    variable: str = "Ys"
    """Quantity to plot: ``Ys``, ``Cp``, ``Cpo``, ``Cho``, ``Vm`` or ``Vt``."""

    def report(self, config, result):
        machine = result.machine
        annulus = machine.annulus if machine else None
        if result.grid is None or annulus is None:
            logger.info("No grid to cut, skipping the spanwise plot.")
            return []

        if result.history is not None and result.history.diverged:
            logger.info("The march diverged, skipping the spanwise plot.")
            return []

        if result.actual is None:
            logger.info(
                "No mixed-out mean line to refer to, skipping the spanwise plot."
            )
            return []

        if self.variable not in _SPANWISE_LABELS:
            raise ValueError(
                f"There is no spanwise variable {self.variable!r} to plot; "
                f"try one of {sorted(_SPANWISE_LABELS)}."
            )

        import matplotlib.pyplot as plt  # noqa: PLC0415

        figures = []
        for m in self.m_cut:
            cut = ember.cut.unstructured(
                result.grid, annulus.evaluate_xr(m, [0.0, 1.0]).T
            )
            if cut is None:
                logger.info(f"No block reaches m={m}, skipping its spanwise plot.")
                continue

            i_row = _cut_row(m, annulus.n_row)

            # Resolved at the mesh's own resolution rather than an invented
            # one. A block is (streamwise, pitchwise, spanwise) and a cut is
            # (meridional, theta), so the spanwise count leads.
            _, nj, nk = result.grid.rows[i_row][0].shape
            try:
                structured = ember.cut.interpolate_to_structured(cut, (nk, nj))
            except ValueError as err:
                # A plane placed between two rows of different blade count has
                # no single pitch to wrap theta by. That is a bad cut, not a
                # broken report, so the other planes still get drawn.
                logger.info(f"Could not interpolate the cut at m={m}: {err}")
                continue

            values, label = self._profile(structured, result.actual, i_row)
            figures.append(
                self._draw(plt, values, _cut_spf(structured), label, m, i_row)
            )

        return figures

    def _profile(self, cut, mean_line, i_row):
        """Return the spanwise profile to plot, and the label for it."""
        row = mean_line[:, i_row]
        ref = mean_line.get_characteristic_station(i_row)

        def mass_avg(prop):
            return ember.average.mass_average(prop, cut, axes=(1,))

        def area_avg(prop):
            return ember.average.area_average(prop, cut, axes=(1,))

        # One reference velocity for the whole machine, so a stator profile can
        # be read against the rotor beside it. A machine that does not rotate
        # has no blade speed to scale by and falls back to the velocity of the
        # row's characteristic station, which the label then says.
        U_ref = float(np.max(np.abs(mean_line.U)))
        sym = "U"
        if not U_ref > 0.0:
            U_ref = float(ref.V)
            sym = "V"

        # Which end of the row sets the scale, on the mean line's own reading
        # of what the row does to the flow.
        is_compressor = float(row.P[1]) > float(row.P[0])
        dP = (
            float(row.Po_rel[0]) - float(row.P[0])
            if is_compressor
            else float(row.Po_rel[1]) - float(row.P[1])
        )

        if self.variable == "Ys":
            values = (
                float(row.T[1])
                * (mass_avg(cut.s) - float(row.s[0]))
                / float(ref.halfVsq_rel)
            )
        elif self.variable == "Cp":
            values = (area_avg(cut.P) - float(row.Po_rel[0])) / dP
        elif self.variable == "Cpo":
            values = (mass_avg(cut.Po) - float(row.Po_rel[0])) / dP
        elif self.variable == "Cho":
            values = (mass_avg(cut.ho) - float(row.ho[0])) / U_ref**2
        elif self.variable == "Vm":
            values = mass_avg(cut.Vm) / U_ref
        elif self.variable == "Vt":
            values = mass_avg(cut.Vt) / U_ref
        else:
            raise ValueError(f"Unhandled spanwise variable {self.variable!r}.")

        return values, _SPANWISE_LABELS[self.variable].replace("{sym}", sym)

    def _draw(self, plt, values, spf, label, m, i_row):
        fig, ax = plt.subplots(layout="constrained")
        ax.plot(values, spf)
        ax.set_xlabel(label)
        ax.set_ylabel("Span Fraction")
        ax.set_ylim((0.0, 1.0))
        ax.set_title(f"{self.variable} at m={m:.3g}, row {i_row}")
        return fig


STANDARD = (
    VelocityTrianglePlot(),
    AnnulusPlot(),
    SectionsPlot(),
    ConvergencePlot(),
    SurfacePlot(),
    ContourPlot(),
)
"""Post-processors that run whether or not a config asks for them.

Cheap next to a solve, and each one degrades to no figures when what it needs
is absent, so the set is safe to run for every verb. It is not part of a
config: `turbigen.cli.processors` combines it with whatever `post_process`
names, and a configured processor of the same type replaces its standard
counterpart rather than adding to it.
"""
