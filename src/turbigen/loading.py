"""What a blade does to the flow, read off both of its surfaces.

One measurement with two consumers, which is why it has a module of its own.
:class:`~turbigen.iterate.ClarkProfile` steers a blade onto a loading and
:class:`~turbigen.metric.DiffusionFactor` records one without correcting
anything --- and the two have to mean the same thing by "where the peak is"
and "what circulation the blade drew", or a design would be iterated onto a
target the report then contradicts. Both read :func:`surfaces`.

Neither an iterator nor a metric owns it. `iterate` importing `metric` would
reach `post`, which imports `iterate` back; that resolves today only because
the import in `post` is deferred, and a leaf module both can depend on has no
such trap in it.
"""

import dataclasses
import logging

import numpy as np

import turbigen.blade
import turbigen.util

logger = logging.getLogger("turbigen")

N_CHORD_NOSE = 501
"""Chordwise points used to place the geometric nose of a section."""


@dataclasses.dataclass(frozen=True)
class ClarkMeasurement:
    """What a solved row tells a :class:`~turbigen.iterate.ClarkProfile`.

    The samples its knobs are read at, and the circulation the blade actually
    drew --- which is not a reduction of those samples but a separate integral
    over the whole cut, and the reason this is a class rather than a pair.
    """

    z: np.ndarray
    """Surface fraction of each sample, shape (2, n), suction first."""

    fac: np.ndarray
    """Isentropic Mach at each sample over the trailing edge value, shape
    (2, n), suction first."""

    Co: float
    """Circulation coefficient the blade drew [--].

    Coull and Hodson's coefficient as :class:`~turbigen.blade.Circulation`
    writes it --- the blade circulation over an ideal one carrying exit
    velocity along the whole suction surface --- so it is directly comparable
    to the `Co` a design asked for, and not merely proportional to it.

    **Under a uniform acoustic speed.** The circulation is `∮V·dl` and what is
    integrated here is `∮Ma·dl`, because :mod:`turbigen.clark` is written in
    Mach number and a target expressed in `Ma / Ma_TE` can only be compared
    against a measurement in the same. Dividing by the trailing edge value is
    what makes the two stand in for `V / V_exit`.

    **Over the whole cut, not the samples.** Every node of both surfaces, nose
    and trailing edge included, weighted by its own arc length. A mean of the
    samples would be none of those things: they are unevenly spaced, they
    leave about a third of each surface unvisited at the two ends, and they
    move when the curve `order` changes --- which would make the measured
    circulation of an unchanged flow depend on how the thickness happens to be
    parameterised.
    """

    length_ratio: float
    """Pressure surface length over suction surface length [--].

    Carried because the caller needs it to convert a circulation error into
    the surface offset that would cause it, and recomputing it there would be
    a second opinion on a number this measurement already formed.
    """


def locate_arc_length(blade, spf, xrt, nchord=10000):
    """Return a measured point's arc length on the suction surface's scale.

    A point measured on the flow --- a stagnation point, say --- is not
    naturally expressed in `m` or in arc length; it is wherever the flow put
    it. This is how it gets placed back onto the curve
    :meth:`~turbigen.blade.Blade.evaluate_arc_length` returns: matched by
    nearest point in true `(x, r, r * theta)` distance, dense enough that the
    match lands within microns of the truth for any point actually on the
    surface.

    Here rather than on :class:`~turbigen.blade.Blade` because a blade answers
    from its design alone and this takes a measurement --- the line this module
    already draws in its own docstring. Being a free function is also what
    makes the sampling honest: it passes one `m` to both blade calls, where two
    methods each defaulting to their own `cluster_cosine(10000)` agreed only by
    coincidence, and their results are subtracted from one another.

    **Both surfaces are searched, and the answer is signed.** At any positive
    incidence the stagnation point sits on the pressure side of the nose, and a
    search restricted to the suction surface answers with approximately the
    leading edge --- silently, and wrongly by exactly the distance that matters
    most. A point at arc length `sigma` along the pressure surface is `sigma`
    from the leading edge the other way round the nose, so a suction-surface
    point at geometric `s` stands `s + sigma` from it along the blade. Negating
    such a point's arc length is what says so, and lets a caller subtract this
    from a suction-surface arc length without knowing which side the point
    landed on.

    Parameters
    ----------
    blade : Blade
        The blade to place the point on.
    spf : float
        Span fraction to evaluate at.
    xrt : array_like, shape (3,)
        A point close to the blade, as `(x, r, r * theta)`.
    nchord : int
        Chordwise points the surfaces are searched at.

    Returns
    -------
    float
        Arc length to the nearest point found, on the scale
        :meth:`~turbigen.blade.Blade.evaluate_arc_length` returns, negated
        where that point is on the pressure surface [m].

    """
    m = turbigen.util.cluster_cosine(nchord)
    surfaces = blade.evaluate_section(spf, m=m)
    _, s = blade.evaluate_arc_length(spf, m=m)

    # Nearest point on each surface, then whichever of the two is nearer. The
    # sign is the surface it landed on, suction first as everything is.
    nearest = [
        turbigen.util.vecnorm(
            turbigen.blade.to_xrrt(xrt_surf) - np.asarray(xrt)[:, None]
        )
        for xrt_surf in surfaces
    ]
    i_surf = int(np.argmin([distance.min() for distance in nearest]))
    j = int(np.argmin(nearest[i_surf]))

    return float(s[i_surf][j]) * (1.0 if i_surf == 0 else -1.0)


def _cut(result, i_row, spf):
    """Return `(blade, cut, mas, ma_TE)` for row `i_row` at `spf`, or None.

    Everything a surface distribution needs before it is split into its two
    surfaces by :func:`surfaces`.

    None where there is nothing to measure: no blade at this row, no section at
    this span, or a trailing edge so thin `ma_TE` reads as zero.
    """
    if result.grid is None or result.machine is None:
        return None

    surfaces = turbigen.util.cut_blade_surfs(result.grid)
    if i_row >= len(surfaces) or surfaces[i_row] is None:
        return None

    machine = result.machine
    blade = machine.rows[i_row].blade

    cut, _ = turbigen.util.cut_section(surfaces[i_row][0], machine.annulus, i_row, spf)
    if cut is None:
        return None

    mas = turbigen.util.isentropic_mach(cut, machine.mean_line[:, i_row].s[0])[:, 0]

    # The cut wraps the blade from one trailing edge round to the other, so its
    # two ends are the two sides of the trailing edge and their mean is the
    # exit value. Read here, while both sides are certainly still present.
    ma_TE = 0.5 * float(mas[0] + mas[-1])
    if not ma_TE:
        return None

    return blade, cut, mas, ma_TE


def mach_ratio(machine, i_row):
    """Return ``Ma_2 / Ma_1`` across row `i_row`, in the relative frame.

    Relative because a surface distribution is a relative-frame quantity ---
    `isentropic_mach` builds it from `ho_rel` --- so an absolute ratio would
    describe a rotor the blade does not see. The two are the same for a
    stationary row.

    Off the nominal mean line rather than the mixed-out one. It is a
    normalisation, and one that moved with the solution would make a target
    mean something slightly different every iteration; the design is also what
    the target was written against.
    """
    ml = machine.mean_line[:, i_row]
    return float(ml.Ma_rel[1] / ml.Ma_rel[0])


def mach_ratio_axial(machine, i_row):
    """Return ``Ma_2 / Max_1`` across row `i_row`, in the relative frame.

    As :func:`mach_ratio`, but referred to the inlet *axial* Mach number rather
    than the inlet velocity magnitude. The two differ by ``cos`` of the
    relative inlet flow angle, so they agree on a row the flow enters nearly
    axially and part company as the swirl rises --- by half as much again at
    fifty degrees.

    Which one a target should carry depends on what the target is a statement
    about. A pressure surface running at some fraction of the exit Mach number
    is a claim about how much the passage may diffuse before it accelerates,
    and what is available to diffuse is what passes through the throat, not
    what the blade sees arriving at an angle. See
    :attr:`turbigen.iterate.ClarkProfile.Ma_PS`.

    Off the nominal mean line, as :func:`mach_ratio` is, and for the same
    reason.
    """
    ml = machine.mean_line[:, i_row]
    return float(ml.Ma_rel[1] / ml.Max[0])


@dataclasses.dataclass(frozen=True)
class Surfaces:
    """Both surfaces of one section, as a solution loaded them.

    The raw distribution every two-sided consumer reduces:
    :func:`measure_clark_profile` samples it at a thickness's control points
    and :class:`~turbigen.metric.DiffusionFactor` reads its peak, so the two
    cannot disagree about where a surface starts or what its circulation is.
    """

    blade: object
    """The row's blade, for placing a point on its geometry."""

    z: tuple[np.ndarray, np.ndarray]
    """Surface fraction of every node from the geometric leading edge, suction
    first [--]."""

    ma: tuple[np.ndarray, np.ndarray]
    """Isentropic Mach number at every node, suction first [--]."""

    ma_TE: float
    """Isentropic Mach number at the trailing edge, the mean of its two sides
    [--]."""

    length_ratio: float
    """Pressure surface length over suction surface length [--]."""

    @property
    def Co(self):
        """Return the circulation coefficient the blade drew [--].

        See :attr:`ClarkMeasurement.Co` for what this integral is and is not.
        `z` is each surface's own arc length normalised by its own extent, so
        an integral over it is a mean along that surface and the length puts it
        back into a circulation: the two surfaces are not the same length, and
        a difference of per-surface means would be `Γ_ss/L_ss - Γ_ps/L_ps`
        rather than anything proportional to `Γ`. Divided through by the
        suction surface length, which is the ideal circulation Coull and Hodson
        normalise by, so what comes out is `Co` itself.
        """
        loop = [float(np.trapezoid(self.ma[i] / self.ma_TE, self.z[i])) for i in (0, 1)]
        return loop[0] - self.length_ratio * loop[1]


def surfaces(result, i_row, spf):
    """Return both surfaces of row `i_row` at span fraction `spf`, or None.

    **Measured from the geometric leading edge, not the stagnation point.**
    Clark's independent variable is ``z = l / L_surf``, the fraction of the
    way along a surface from where it starts. The stagnation point is not
    where a surface starts; it is where the flow happened to attach, and with
    no incidence iterator holding it, it wanders as the very thickness being
    driven changes. Anchoring on the geometry instead means the abscissa does
    not move with the knobs moving the blade.

    **Plain ``Ma / Ma_TE``, with no ``Ma_2 / Ma_1`` factor.** :mod:`turbigen.clark`
    works in that, and mixing two normalisations inside one curve --- whose
    pieces are built from differences between its own parameters --- would
    not evaluate to anything.

    None where there is nothing to measure: no blade at this row, no section at
    this span, a trailing edge so thin `ma_TE` reads as zero, or both halves of
    the cut landing on the same surface.
    """
    cut_all = _cut(result, i_row, spf)
    if cut_all is None:
        return None
    blade, cut, mas, ma_TE = cut_all

    xrrt = np.stack((cut.x[:, 0], cut.r[:, 0], cut.r[:, 0] * cut.t[:, 0]))

    # Split the loop at its own leading edge. The cut runs from one trailing
    # edge round to the other, so it crosses the nose exactly once and finding
    # that crossing is one nearest-node search rather than a match of every
    # node against both surfaces.
    xrt_nose = blade.evaluate_section(spf, nchord=N_CHORD_NOSE)[0][:, 0]
    i_nose = int(
        np.argmin(
            turbigen.util.vecnorm(xrrt - turbigen.blade.to_xrrt(xrt_nose)[:, None])
        )
    )

    # Each half measured from the nose outwards, and normalised by its own
    # extent -- which is what makes this the same `z` the geometry reports,
    # agreeing at both ends by construction and differing in between only by
    # what discretisation and the cut's own offset from the wall disagree
    # about.
    zeta = turbigen.util.get_zeta(cut)[:, 0]
    halves = (slice(i_nose, None, -1), slice(i_nose, None))

    # Which half is which surface, from the blade rather than from the cut:
    # `get_zeta` runs from index zero whichever way `cut_blade_surfs` happened
    # to wrap the blade, so the loop direction says nothing about which side of
    # the camber line a half is on. `locate_arc_length` already answers this,
    # signing a point by the surface it lands on.
    z, ma = [None, None], [None, None]
    for half in halves:
        distance = np.abs(zeta[half] - zeta[i_nose])
        extent = distance[-1] or 1.0

        middle = xrrt[:, half][:, len(distance) // 2]
        i_surf = 0 if locate_arc_length(blade, spf, middle) >= 0.0 else 1

        z[i_surf] = distance / extent
        ma[i_surf] = mas[half]

    if z[0] is None or z[1] is None:
        logger.info(
            f"Both halves of row {i_row}'s section at spf={spf:.2f} placed on "
            f"the same surface, so its loading cannot be split between them."
        )
        return None

    L = blade.evaluate_surface_length(spf)
    return Surfaces(
        blade=blade,
        z=tuple(z),
        ma=tuple(ma),
        ma_TE=ma_TE,
        length_ratio=float(L[1] / L[0]) if L[0] else 1.0,
    )


def measure_clark_profile(result, i_row, spf, m):
    """Return the loading of both surfaces at each `m`, suction first.

    For a :class:`~turbigen.iterate.ClarkProfile` shaping a
    :class:`~turbigen.thickness.ClarkThickness` against :mod:`turbigen.clark`:
    the :func:`surfaces` of the section, sampled where each coefficient acts.

    Parameters
    ----------
    result : Result
        A solved run.
    i_row : int
        Blade row to measure.
    spf : float
        Span fraction to measure at.
    m : array_like
        Normalised chordwise positions to sample at, as
        :attr:`~turbigen.thickness.ClarkThickness.m_ctl` gives them.

    Returns
    -------
    ClarkMeasurement
        The samples at `m`, suction surface first --- the order
        :meth:`~turbigen.blade.Blade.evaluate_section` returns surfaces in and
        :attr:`~turbigen.thickness.ClarkThickness.coeff` holds its rows in ---
        and the circulation the blade drew, which is measured over the whole
        cut rather than from those samples.

    None
        Where there was nothing to measure at all --- see :func:`surfaces`.

    """
    measured = surfaces(result, i_row, spf)
    if measured is None:
        return None

    # Where each sample sits, in the same surface fraction, off the geometry
    # alone -- the two surfaces have different lengths, so one `m` is not one
    # `z`.
    m_dense, s = measured.blade.evaluate_arc_length(spf)
    m = np.asarray(m, dtype=float)

    z = np.stack([np.interp(m, m_dense, s[i]) / (s[i][-1] or 1.0) for i in (0, 1)])
    fac = np.stack(
        [
            np.interp(z[i], measured.z[i], measured.ma[i]) / measured.ma_TE
            for i in (0, 1)
        ]
    )

    return ClarkMeasurement(
        z=z, fac=fac, Co=measured.Co, length_ratio=measured.length_ratio
    )
