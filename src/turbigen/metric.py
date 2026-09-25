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

import ember.average
import ember.cut
import numpy as np

import turbigen.loading
import turbigen.util
from turbigen.node import Node

logger = logging.getLogger("turbigen")

I_TIP = 2
"""Where the blade tips sit on the surface-type axis of `SurfaceDissipation`.

The endwalls and the blades are at 0 and 1, and are read by position like
every other pair of them in this package. The tips are named because they are
the one type whose absence means zero rather than unmeasured, and the line
saying so has to point at them.
"""


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
            each shaped ``(3, n_row)`` --- like the mean line in its second
            axis, but with the leading axis running over *surface type* where
            a mean line's runs over station: 0 for the endwalls, 1 for the
            blades and 2 for the blade tips. `A_surf` and `Vcu_surf` are
            integrated over the same faces, so `Vcu/A` says whether a change
            in loss came from area or from the velocity over it.

            The tips are their own row rather than added to the blades,
            because they are there only when a clearance is gridded: summed
            into the blades they would make a mesh change look like a design
            one. A row whose tip is pinched, or which has no clearance, reads
            zero there rather than `nan` --- there is no such surface, so
            there is no area for a loss to be generated on, which is a
            measurement and not a gap in one.

        """
        grid, machine = result.grid, result.machine
        if grid is None or machine is None or result.actual is None:
            return {}
        if result.history is not None and getattr(result.history, "diverged", False):
            return {}

        planes = machine.annulus.cut_planes()
        endwalls = turbigen.util.cut_endwalls(grid)
        blades = turbigen.util.cut_blade_surfs(grid)
        tips = turbigen.util.cut_blade_tips(grid)

        n_row = len(grid.rows)
        Sdot = np.full((3, n_row), np.nan)
        A = np.full((3, n_row), np.nan)
        Vcu = np.full((3, n_row), np.nan)

        for i_row in range(n_row):
            # The free stream entering this row, not the machine: a downstream
            # row's boundary layers grow in fluid that already carries the loss
            # of everything upstream, and an edge velocity referred to the
            # machine inlet would be a velocity nothing in the domain has.
            s_ref = float(result.actual[:, i_row].s[0])

            # Each surface type is a list of cuts of no fixed length: one
            # endwall is two faces on a plain row and three where a gridded
            # clearance splits the casing, and a blade is one surface or none.
            surfaces = (endwalls[i_row], blades[i_row] or [], tips[i_row])
            for i_surf, cuts in enumerate(surfaces):
                if not cuts and i_surf != I_TIP:
                    # A row whose blade surface could not be cut is not a row
                    # whose blades did not dissipate, so it is unmeasured
                    # rather than zero.
                    continue

                # A tip with no cut is the other thing: the surface is absent
                # rather than unreadable, because the clearance is pinched or
                # there is none, and an absent surface dissipates nothing.
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
    assert ember.cut.signed_distance(planes[0], exit_midpoint) < 0.0, (
        "signed distance is not negative downstream; the clip would keep the "
        "ducts and drop the machine"
    )

    after_inlet = ember.cut.signed_distance(planes[0], xr) <= 0.0
    before_exit = ember.cut.signed_distance(planes[-1], xr) >= 0.0

    return after_inlet & before_exit


class DiffusionFactor(Metric):
    r"""Peak-to-exit diffusion and circulation of each row.

    The isentropic surface Mach number is a distribution, and the amount it
    falls from its peak back to the trailing edge is what the boundary layer on
    the late suction surface has to survive,

    .. math::
        \mathit{DF} = \frac{\mathit{Ma}_{s,\mathrm{max}}}{\mathit{Ma}_{s,\mathrm{TE}}} - 1

    where the peak is the largest value on the suction surface and the
    trailing-edge value is the mean of the two sides of the cut. Where the peak
    sits is reported with it, and the circulation coefficient the blade drew
    beside both.

    **Read directly, nothing fitted.** The peak is the maximum of the data and
    `zeta_peak` the surface fraction of that node, so they exist for every
    blade that has a surface --- including one that accelerates all the way to
    its trailing edge, where `DF` can sit slightly below zero because the
    suction side reads under the mean of the two --- and `Co` is the integral
    round the whole cut.

    **The numbers** :class:`~turbigen.iterate.ClarkProfile` **steers.** Every
    one comes from :func:`turbigen.loading.surfaces`, so `zeta_peak` is Clark's
    ``z = l / L_surf`` from the geometric leading edge, `DF + 1` is its
    `Ma_peak`, and `Co` is the circulation it drives the blade count with. A
    metric and an iterator that disagreed would let a design be driven onto a
    target the report then contradicts.

    The isentropic Mach number is referred to the entropy entering *that row*,
    the same reference :class:`~turbigen.post.SurfacePlot` draws it against.
    """

    type: ClassVar[str] = "diffusion_factor"

    spf: tuple[float, ...] = (0.5,)
    """Span fractions to measure the surface distribution at [--]."""

    def evaluate(self, config, result):
        """Return the diffusion of each row, at each span fraction.

        Returns
        -------
        dict
            ``DF``, ``zeta_peak`` and ``Co`` [--], each shaped
            ``(n_spf, n_row)``. NaN for a row and span with nothing to measure:
            above a clearance gap, or a row with no blade.
        """
        del config

        if result.grid is None or result.machine is None:
            return {}
        if result.history is not None and getattr(result.history, "diverged", False):
            return {}

        shape = (len(self.spf), len(result.grid.rows))
        out = {name: np.full(shape, np.nan) for name in ("DF", "zeta_peak", "Co")}

        for i_row in range(shape[1]):
            for i_spf, spf in enumerate(self.spf):
                measured = turbigen.loading.surfaces(result, i_row, spf)
                if measured is None:
                    logger.debug(
                        f"Row {i_row} has no loading to measure at "
                        f"spf={spf:.2f}, so its diffusion there is unmeasured."
                    )
                    continue

                i_peak = int(np.argmax(measured.ma[0]))
                out["DF"][i_spf, i_row] = measured.ma[0][i_peak] / measured.ma_TE - 1.0
                out["zeta_peak"][i_spf, i_row] = measured.z[0][i_peak]
                out["Co"][i_spf, i_row] = measured.Co

        return out


class LossBreakdown(Metric):
    r"""Profile and secondary entropy loss of each row.

    The profile loss is what the stream surface through the middle of the span
    generates, clear of the endwalls: the band carrying the central `band` of
    the mass flow, found by mass fraction at the row's inlet and exit planes and
    mixed out at each,

    .. math::
        \Delta s_\mathrm{mid} = s_\mathrm{mix}(\text{exit band})
            - s_\mathrm{mix}(\text{inlet band}) \,.

    The secondary loss is the rest of the row's loss, everything the endwalls,
    their secondary flows and any tip clearance add to it,

    .. math::
        \Delta s_\mathrm{sec} = \Delta s_\mathrm{total} - \Delta s_\mathrm{mid} \,,

    where :math:`\Delta s_\mathrm{total}` is the entropy rise across the row in
    :attr:`~turbigen.result.Result.actual`. That is mixed out at both ends on
    the same planes, which is why the bands are mixed out too: an average of
    one against a mix-out of the other would count mixing in one term only.

    **The band is matched by mass fraction, not traced.** The central tenth of
    the flow at the exit is taken to be the central tenth at the inlet. Where
    the flow migrates radially through a row it is not quite the same fluid,
    and the split shifts accordingly.

    **Two normalisations.** A loss coefficient is referred to the row itself,

    .. math::
        Y = \frac{T_\mathrm{out}\,\Delta s}{\tfrac{1}{2}V_\mathrm{rel}^2} \,,

    with the static temperature at the row's exit and the relative dynamic head
    at its characteristic station, as the spanwise plot draws :math:`Y_s`. A
    lost efficiency is referred to the machine, after Denton (1993),

    .. math::
        \Delta\eta = \frac{T_\mathrm{exit}\,\Delta s}{|\Delta h_0|} \,,

    with the static temperature at the machine exit and the machine's work.
    The denominator is the same for every row, so lost efficiencies compare
    across rows and add. Summed over the rows they fall a little short of the
    machine's own :math:`T_\mathrm{exit}\,\Delta s/|\Delta h_0|`, because the
    gaps between the rows belong to none of them.
    """

    type: ClassVar[str] = "loss_breakdown"

    band: float = 0.1
    """Fraction of the mass flow in the central stream surface [--]."""

    def evaluate(self, config, result):
        """Return the profile and secondary loss of each row.

        Returns
        -------
        dict
            ``Ys_mid``, ``Ys_sec``, ``Deta_mid`` and ``Deta_sec`` [--], each
            shaped ``(n_row,)``. NaN for a row whose planes cannot be cut or
            whose band cannot be mixed out, and the lost efficiencies are NaN
            throughout on a machine with no rotating row, which does no work.
        """
        del config

        grid, machine, actual = result.grid, result.machine, result.actual
        if grid is None or machine is None or actual is None:
            return {}
        if result.history is not None and getattr(result.history, "diverged", False):
            return {}

        planes = machine.annulus.cut_planes()
        lo, hi = 0.5 - 0.5 * self.band, 0.5 + 0.5 * self.band

        n_row = len(grid.rows)
        Ds_mid = np.full(n_row, np.nan)
        for i_row in range(n_row):
            try:
                s_in, s_out = (
                    _band_entropy(grid, planes[2 * i_row + i], lo, hi) for i in (0, 1)
                )
            except (ValueError, RuntimeError, AssertionError) as err:
                logger.debug(f"Row {i_row} has no band loss to measure: {err}")
                continue
            Ds_mid[i_row] = s_out - s_in

        Ds_total = np.asarray(actual.s[1] - actual.s[0], dtype=float)
        Ds_sec = Ds_total - Ds_mid

        # One dynamic head per row, for the row's own loss coefficient.
        Y_per_Ds = np.array(
            [
                float(actual[:, i_row].T[-1])
                / float(actual.get_characteristic_station(i_row).halfVsq_rel)
                for i_row in range(n_row)
            ]
        )

        # One work for the whole machine, so every row's lost efficiency is in
        # the same currency. A machine that does not rotate does no work, and
        # its enthalpy change is round-off that must not be divided by.
        if all(float(blocks[0].Omega) == 0.0 for blocks in grid.rows):
            eta_per_Ds = np.nan
        else:
            eta_per_Ds = float(actual.outlet.T) / abs(float(actual.Dho))

        return {
            "Ys_mid": Y_per_Ds * Ds_mid,
            "Ys_sec": Y_per_Ds * Ds_sec,
            "Deta_mid": eta_per_Ds * Ds_mid,
            "Deta_sec": eta_per_Ds * Ds_sec,
        }


def _band_entropy(grid, xr, lo, hi):
    """Return the mixed-out entropy of the band `lo..hi` of mass on a plane."""
    cut = turbigen.util.cut_structured(grid, xr)
    if cut is None:
        raise ValueError(f"The plane at {np.asarray(xr).tolist()} misses the grid.")
    band = ember.average.mass_band(cut, lo, hi, axis=0)
    return float(ember.average.mix_out(band).s)
