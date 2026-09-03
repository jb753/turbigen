"""Quantities measured from a solved field and kept in the result.

A :class:`Metric` maps a config and a result to a dict of named numbers ---
scalars, or nested lists of them. Unlike a :class:`~turbigen.post.Post` it
returns data rather than figures, and unlike an
:class:`~turbigen.iterate.Iterator` nothing acts on what it measures: a metric
is a passive observation of the flow, written to ``result: metrics:`` so that a
run archived today can be mined later.

Keys are the metric's own business, derived from its type and parameters the
way an iterator derives ``dchi_TE[0]`` --- there is no user-supplied label.
"""

import logging
from typing import ClassVar

import numpy as np

import ember.average
import ember.cut
import turbigen.util
from turbigen.node import Node
from turbigen.post import (
    N_CHORD_PLOT,
    _isentropic_mach,
    _normalise_surface_distance,
)

logger = logging.getLogger("turbigen")

N_SPAN_CUT = 101
"""Meridional points defining the span curve a blade surface is cut along."""


class Metric(Node):
    """Base for quantities measured from a solved field."""

    def evaluate(self, config, result):
        """Return ``{name: value}`` measured from `result`.

        Each `value` is a number or a nested list of numbers. Return an empty
        dict when the run gives nothing to measure --- no grid, or a diverged
        march --- exactly as a from-solution iterator's ``error`` does.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement evaluate(self, config, result)"
        )


def measure(config, result):
    """Return every configured metric's values, merged and made YAML-clean.

    Each metric is wrapped: it is an observation added after the CFD has already
    been paid for, so one that raises is logged and skipped rather than allowed
    to sink the run's output --- the same guard `solve` puts around mix-out and
    the design-comparison table.
    """
    merged = {}
    for m in config.metrics:
        try:
            values = m.evaluate(config, result)
        except Exception as err:
            logger.warning(f"Metric {m.type!r} could not be measured: {err}")
            continue

        for name, value in values.items():
            if name in merged:
                logger.warning(f"Two metrics both write {name!r}; keeping the last.")
            merged[name] = np.asarray(value, dtype=float).tolist()

    return merged


class SurfaceDissipation(Metric):
    r"""Denton's velocity-cubed estimate of boundary layer loss.

    Entropy generated in the boundary layers on the wetted walls of each row,
    after Denton (1993),

    .. math::
        \dot{S}_\mathrm{surf}
            = \int_\mathrm{surf} C_\mathrm{d}\,\frac{\rho V_s^3}{T}\,\mathrm{d}A

    where :math:`V_s` is the local velocity at the edge of the boundary layer.
    That is a real velocity of real fluid, not an idealised one: it is
    recovered from the surface static pressure by expanding isentropically from
    the entropy of the free stream entering *that row*, because reading it off
    the cell against the wall would return a velocity inside the layer rather
    than at its edge.

    Evaluated in the frame of each wall, which the cut carries --- so the
    casing over a tip gap is measured in the absolute frame while the blade
    below it is measured in the relative one, as the boundary layers on them
    are.

    The integral runs between the first and last mean-line cut planes, so the
    inlet and exit ducts are excluded and the surfaces bound the same control
    volume whose end states :attr:`~turbigen.result.Result.actual` reports.
    Loss in the gaps between rows is kept, attributed to the row whose blocks
    it sits in.
    """

    type: ClassVar[str] = "surface_dissipation"

    Cd: float = 0.002
    """Dissipation coefficient [--].

    Denton's classic value. The entropy rate is exactly linear in it, so a
    second instance at another value measures nothing a rescale would not
    give.
    """

    def evaluate(self, config, result):
        """Return the surface dissipation of each row.

        Returns
        -------
        dict
            ``Sdot_surf`` [W/K], ``A_surf`` [m^2] and ``Vcu_surf`` [m^5/s^3],
            each shaped like the mean line at ``(2, n_row)`` --- but with the
            leading axis running over *surface type*, 0 for the endwalls and 1
            for the blades, where a mean line's runs over station. `A_surf`
            and `Vcu_surf` are integrated over the same faces, so `Vcu/A` says
            whether a change in loss came from area or from the velocity over
            it.

        """
        grid, machine = result.grid, result.machine
        if grid is None or machine is None or result.actual is None:
            return {}
        if result.history is not None and getattr(result.history, "diverged", False):
            return {}

        planes = machine.annulus.cut_planes()
        endwalls = turbigen.util.cut_endwalls(grid)
        blades = turbigen.util.cut_blade_surfs(grid)

        n_row = len(grid.rows)
        Sdot = np.full((2, n_row), np.nan)
        A = np.full((2, n_row), np.nan)
        Vcu = np.full((2, n_row), np.nan)

        for i_row in range(n_row):
            # The free stream entering this row, not the machine: a downstream
            # row's boundary layers grow in fluid that already carries the loss
            # of everything upstream, and an edge velocity referred to the
            # machine inlet would be a velocity nothing in the domain has.
            s_ref = float(result.actual[:, i_row].s[0])

            for i_surf, cuts in enumerate((endwalls[i_row], blades[i_row] or [])):
                if not cuts:
                    # A row whose blade surface could not be cut is not a row
                    # whose blades did not dissipate, so it is unmeasured
                    # rather than zero.
                    continue

                totals = np.zeros(3)
                for cut in cuts:
                    totals += _dissipation(cut, s_ref, self.Cd, planes)

                Sdot[i_surf, i_row], A[i_surf, i_row], Vcu[i_surf, i_row] = totals

        return {"Sdot_surf": Sdot, "A_surf": A, "Vcu_surf": Vcu}


def isentropic_velocity(cut, s_ref):
    r"""Return boundary-layer edge velocity over `cut`, referred to `s_ref`.

    Expanded isentropically from `s_ref` to the local static pressure, in the
    frame of the wall the cut was taken from. Given the cut's own entropy this
    returns the wall-relative speed exactly, which is what says the frame and
    the expansion are both right.

    Parameters
    ----------
    cut : ember.block.Block
        A 2D wall cut carrying the angular velocity of its own wall.
    s_ref : float
        Specific entropy of the free stream entering the row [J/kg/K].

    Returns
    -------
    ndarray
        Nodal edge velocity [m/s].

    """
    # Set in place on a copy, not chained off one: ember's setters return
    # nothing, as `post._isentropic_mach` also has to work around.
    isen = cut.copy(keep_patches=False)
    isen.set_P_s(cut.P, s_ref)

    # `ho_rel` is in the frame the cut carries, which is the wall's own; the
    # isentropic static enthalpy has no velocity in it and so no frame.
    return np.sqrt(2.0 * np.maximum(cut.ho_rel - isen.h, 0.0))


def _dissipation(cut, s_ref, Cd, planes):
    """Return ``(Sdot, A, Vcubed)`` over one wall cut, for the whole annulus.

    All three are integrated over the same faces, so they can be read against
    one another: an area that grew and a velocity that grew are told apart
    only if the surface they were measured on is the same.
    """
    Vs = isentropic_velocity(cut, s_ref)

    face = ember.average._node_to_face_2d
    dA = np.linalg.norm(cut.dA_quad, axis=-1, ord=2)
    rho, T, Vs = face(cut.rho), face(cut.T), face(Vs)

    keep = _within_the_machine(cut, planes)

    # One passage is meshed; the annulus has Nb of them.
    Nb = float(cut.Nb)
    return np.array(
        [
            Cd * np.sum(keep * rho * Vs**3 / T * dA) * Nb,
            np.sum(keep * dA) * Nb,
            np.sum(keep * Vs**3 * dA) * Nb,
        ]
    )


def _within_the_machine(cut, planes):
    """Return a face mask, true between the first and last cut planes.

    The domain runs into an inlet duct upstream and an exit duct downstream,
    and neither is machine. Bounding the integral by the same planes the mean
    line is reduced between makes what this measures comparable with what
    `result.actual` reports across them.
    """
    face = ember.average._node_to_face_2d
    xr = np.stack([face(cut.x), face(cut.r)], axis=-1)

    # Signed distance runs negative downstream of a plane. Checked rather than
    # trusted, because the sign follows from ember's normal convention and from
    # the order `cut_planes` lists its two points in, and it would invert
    # silently if either changed. The exit station is downstream of the inlet
    # one on any machine, which is what makes this a statement about the
    # convention rather than about an axial layout.
    exit_midpoint = planes[-1].mean(axis=0)[None]
    assert ember.cut._signed_distance(planes[0], exit_midpoint) < 0.0, (
        "signed distance is not negative downstream; the clip would keep the "
        "ducts and drop the machine"
    )

    after_inlet = ember.cut._signed_distance(planes[0], xr) <= 0.0
    before_exit = ember.cut._signed_distance(planes[-1], xr) >= 0.0

    return after_inlet & before_exit


class DiffusionFactor(Metric):
    r"""Peak-to-exit diffusion over the blade surfaces of each row.

    The isentropic surface Mach number is a distribution, and the amount it
    falls from its peak back to the trailing edge is what the boundary layer on
    the late suction surface has to survive,

    .. math::
        \mathit{DF} = \frac{\mathit{Ma}_{s,\mathrm{max}}}{\mathit{Ma}_{s,\mathrm{TE}}} - 1

    where the trailing-edge value is the mean of the two sides of the cut, the
    surface being wrapped from one of them round to the other. Where the peak
    sits is reported with it: the same diffusion late on the surface is harder
    on the boundary layer than early, so the factor alone does not say what
    the blade is doing.

    The isentropic Mach number is referred to the entropy entering *that row*,
    the same reference :class:`~turbigen.post.SurfacePlot` draws it against, so
    what this measures is the number that plot shows.

    Measured, not iterated on. Blade count is what sets diffusion, and it is an
    integer: driving it from a target moves the design in a staircase whose
    tread is one blade, which is coarser than any tolerance worth asking for
    and slower to settle than it is worth. So the number is recorded, at every
    row and every span fraction, and the count is the designer's to choose.
    """

    type: ClassVar[str] = "diffusion_factor"

    spf: tuple[float, ...] = (0.5,)
    """Span fractions to measure the surface distribution at [--]."""

    offset: int = 0
    """Cells away from the wall to take the distribution at."""

    def evaluate(self, config, result):
        """Return the diffusion of each row, at each span fraction.

        Returns
        -------
        dict
            ``DF`` [--], ``Mas_max`` [--], ``Mas_TE`` [--] and ``zeta_max``
            [--], each shaped ``(n_spf, n_row)``. The two Mach numbers are the
            parts `DF` is made of, so a change in it says whether the peak grew
            or the exit fell. `zeta_max` is where the peak sits, in surface
            distance normalised as :class:`~turbigen.post.SurfacePlot` plots
            it: zero at the stagnation point and one at the trailing edge, on
            whichever surface the peak is on. NaN for a row and span with no
            blade surface to cut --- above a clearance gap, or a row with no
            blade at all.
        """
        grid, machine = result.grid, result.machine
        if grid is None or machine is None:
            return {}
        if result.history is not None and getattr(result.history, "diverged", False):
            return {}

        surfaces = turbigen.util.cut_blade_surfs(grid, self.offset)

        rows = machine.rows
        n_row = len(grid.rows)
        Mas_max = np.full((len(self.spf), n_row), np.nan)
        Mas_TE = np.full((len(self.spf), n_row), np.nan)
        zeta_max = np.full((len(self.spf), n_row), np.nan)

        for i_row in range(n_row):
            if i_row >= len(surfaces) or surfaces[i_row] is None:
                continue

            # `structured_meridional` walks the second axis of a three-axis
            # block, so the surface is padded to put its spanwise axis there
            # and the cut comes back one wide.
            surface = surfaces[i_row][0][:, :, None]
            s_ref = machine.mean_line[:, i_row].s[0]

            blade = rows[i_row].blade

            for i_spf, spf in enumerate(self.spf):
                measured = _surface_distribution(
                    machine.annulus, blade, surface, s_ref, i_row, spf
                )
                if measured is None:
                    logger.debug(
                        f"Row {i_row} has no blade surface at spf={spf:.2f}, "
                        "so its diffusion there is unmeasured."
                    )
                    continue
                mas, zeta = measured

                # The cut wraps the blade from one trailing edge round to the
                # other, so its endpoints are the two sides of the trailing
                # edge and their mean is the exit value.
                i_max = int(np.argmax(mas))
                Mas_max[i_spf, i_row] = mas[i_max]
                Mas_TE[i_spf, i_row] = 0.5 * (mas[0] + mas[-1])

                # Folded onto the positive axis, as the plot folds it: which
                # surface the peak is on is said by the peak being a peak.
                zeta_max[i_spf, i_row] = abs(zeta[i_max])

        # A trailing edge at rest divides no distribution: left as NaN, which
        # is what an unmeasured diffusion already is.
        with np.errstate(divide="ignore", invalid="ignore"):
            DF = np.where(Mas_TE > 0.0, Mas_max / Mas_TE - 1.0, np.nan)

        return {
            "DF": DF,
            "Mas_max": Mas_max,
            "Mas_TE": Mas_TE,
            "zeta_max": zeta_max,
        }


def _surface_distribution(annulus, blade, surface, s_ref, i_row, spf):
    """Return ``(mas, zeta)`` round the blade of row `i_row` at `spf`.

    The isentropic Mach number and the normalised surface distance it is
    plotted against, taken from the same cut so that a peak and where it sits
    are the same point.

    None where the section is not there --- above a clearance gap the blade has
    no surface to cut, the span there being trimmed off as flow rather than
    wall.
    """
    # Rows occupy the odd meridional segments, so row i spans m from 2i+1 to
    # 2i+2, exactly as the surface distribution plot cuts it.
    m = np.linspace(2 * i_row + 1, 2 * i_row + 2, N_SPAN_CUT)
    xr = annulus.evaluate_xr(m, spf)

    cut = ember.cut.structured_meridional(surface, xr.T)
    if not len(cut):
        return None
    cut = cut[0]

    mas = _isentropic_mach(cut, s_ref)[:, 0]

    # The geometric nose anchors the search for the stagnation point, exactly
    # as the surface distribution plot anchors it.
    xrt_nose = blade.evaluate_section(spf, nchord=N_CHORD_PLOT)[0][:, 0]
    return mas, _normalise_surface_distance(cut, mas, xrt_nose)
