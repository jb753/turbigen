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
import turbigen.loading
import turbigen.util
from turbigen.node import Node

logger = logging.getLogger("turbigen")

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

    Every number here comes from :func:`turbigen.loading.measure`, which is
    also what :class:`~turbigen.iterate.LoadingDistribution` and
    :class:`~turbigen.iterate.PeakMach` correct against. That sharing is the
    point: a metric and an iterator that disagreed about where the peak was
    would let a design be driven onto a target the report then contradicts.

    The peak is fitted rather than taken as a maximum of the data --- two
    straight lines meeting at a breakpoint, so it uses every point in the
    window and slides where an argmax steps between nodes. That makes `DF`
    a little different from a peak read straight off the curve, by around
    three per cent on the case it was checked against, and much steadier on the
    flat-topped distributions that are a design style rather than a pathology.
    """

    type: ClassVar[str] = "diffusion_factor"

    spf: tuple[float, ...] = (0.5,)
    """Span fractions to measure the surface distribution at [--]."""

    offset: int = 0
    """Cells away from the wall to take the distribution at."""

    zeta_front: float = 0.2
    """Front anchor of the window fitted [--].

    Matches the default the iterators carry, so a config that shapes a blade
    and a config that only measures one describe the same curve.
    """

    zeta_TE: float = 0.98
    """Far end of the window fitted [--]."""

    def evaluate(self, config, result):
        """Return the diffusion of each row, at each span fraction.

        Returns
        -------
        dict
            ``DF``, ``Mas_peak``, ``Mas_TE``, ``zeta_peak``, ``fac_front`` and
            ``fac_peak`` [--], each shaped ``(n_spf, n_row)``. The two Mach
            numbers are the parts `DF` is made of, so a change in it says
            whether the peak grew or the exit fell; `fac_peak` is the same
            ratio written as ``DF + 1``, which is the form the iterators take a
            target in. `zeta_peak` is where the peak sits and `fac_front` how
            hard the leading edge accelerates --- the same two numbers
            :class:`~turbigen.iterate.LoadingDistribution` shapes a blade to.
            NaN for a row and span with nothing to measure: above a clearance
            gap, a row with no blade, or a distribution with no peak in the
            window.
        """
        del config

        if result.grid is None or result.machine is None:
            return {}
        if result.history is not None and getattr(result.history, "diverged", False):
            return {}

        shape = (len(self.spf), len(result.grid.rows))
        out = {
            name: np.full(shape, np.nan)
            for name in ("Mas_max", "Mas_TE", "zeta_max",
                         "zeta_peak", "fac_front", "fac_peak")
        }

        for i_row in range(shape[1]):
            for i_spf, spf in enumerate(self.spf):
                measured = turbigen.loading.measure(
                    result, i_row, spf, self.zeta_front, self.zeta_TE
                )
                if measured is None:
                    logger.debug(
                        f"Row {i_row} has no loading to measure at "
                        f"spf={spf:.2f}, so its diffusion there is unmeasured."
                    )
                    continue

                out["Mas_max"][i_spf, i_row] = measured.ma_max
                out["Mas_TE"][i_spf, i_row] = measured.ma_TE
                out["zeta_max"][i_spf, i_row] = measured.zeta_max
                out["zeta_peak"][i_spf, i_row] = measured.zeta_peak
                out["fac_front"][i_spf, i_row] = measured.fac_front
                out["fac_peak"][i_spf, i_row] = measured.fac_peak

        # From the maximum rather than from the fit, so that every blade gets a
        # diffusion factor: one that accelerates to its trailing edge has no
        # interior peak to fit, and it is still diffusing nothing, which is a
        # measurement rather than a failure.
        with np.errstate(divide="ignore", invalid="ignore"):
            out["DF"] = np.where(
                out["Mas_TE"] > 0.0, out["Mas_max"] / out["Mas_TE"] - 1.0, np.nan
            )
        return out
