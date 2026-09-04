"""What a blade does to the flow, read off its suction surface.

One measurement with three consumers, which is why it has a module of its own.
:class:`~turbigen.iterate.LoadingDistribution` corrects the shape of the
distribution, :class:`~turbigen.iterate.PeakMach` corrects its level, and
:class:`~turbigen.metric.DiffusionFactor` records both without correcting
anything --- and all three have to mean the same thing by "where the peak is",
or a design would be iterated onto a target the report then contradicts.

Neither an iterator nor a metric owns it. `iterate` importing `metric` would
reach `post`, which imports `iterate` back; that resolves today only because
the import in `post` is deferred, and a leaf module both can depend on has no
such trap in it.

The arithmetic on the curve itself lives in `turbigen.util` --- the fit, the
suction-side fold, the isentropic expansion --- because none of it needs to
know what a machine is. What is here is the part that does: cutting a row at a
span fraction, and referring the Mach numbers to a mean line.
"""

import dataclasses
import logging

import numpy as np

import turbigen.util

logger = logging.getLogger("turbigen")

N_CHORD_NOSE = 501
"""Chordwise points used to place the geometric nose of a section."""


@dataclasses.dataclass(frozen=True)
class Loading:
    """The loading distribution of one blade section, as numbers.

    Frozen, and not a :class:`~turbigen.node.Node`: this is measured from a
    solution rather than asked for in a config file.
    """

    zeta_peak: float
    """Surface fraction of the peak, from the fitted breakpoint [--]."""

    fac_front: float
    """``Ma(zeta_front) / Ma_TE * Ma_2 / Ma_1`` [--].

    Clark (2019) parameter 3: how hard the leading edge accelerates, referred
    to the trailing edge because that is a mean-line quantity fixed by the
    duty, and carrying the Mach ratio so the same number means the same style
    of leading edge across rows of differing duty.

    Read straight off the surface distribution at `zeta_front`, unlike
    :attr:`fac_peak`, which comes from a fit. A single interpolated point
    needs no peak to exist, so this is finite even on a blade that
    accelerates all the way to its trailing edge.
    """

    fac_peak: float
    """``Ma_peak / Ma_TE`` [--].

    The level of the loading, and one more than the diffusion factor.
    """

    ma_peak: float
    """Isentropic Mach number at the peak, from the fitted apex [--]."""

    ma_TE: float
    """Isentropic Mach number at the trailing edge [--]."""

    ma_max: float
    """Largest isentropic Mach number on the suction surface [--].

    A maximum of the data, where :attr:`ma_peak` is the apex of a fit. Noisier,
    and it steps between nodes rather than sliding, so it is the wrong thing to
    iterate on --- but it exists for every distribution, including one that
    accelerates all the way to its trailing edge and so has no interior peak to
    fit. A metric that has to describe every blade needs the one that always
    exists; a loop that has to steer needs the one that moves smoothly.
    """

    zeta_max: float
    """Surface fraction of :attr:`ma_max` [--]."""


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


def measure(result, i_row, spf, zeta_front=0.2, zeta_TE=0.98):
    """Return the loading of row `i_row` at span fraction `spf`.

    Cut the blade, expand isentropically from the row inlet entropy, keep the
    suction surface, fit two straight lines to it, and refer what they say to
    the mean line.

    The peak comes from the fit rather than from a maximum of the data, and so
    does the front value. A fitted peak uses every point in the window and
    moves smoothly as a design does; `np.argmax` returns whichever single node
    the noise happened to lift, and steps between nodes rather than sliding.
    That matters most on the flat-topped distributions that are a design style
    rather than a pathology.

    Parameters
    ----------
    result : Result
        A solved run.
    i_row : int
        Blade row to measure.
    spf : float
        Span fraction to measure at.
    zeta_front : float
        Front anchor, and the start of the window fitted.
    zeta_TE : float
        End of the window, short of the trailing edge.

    Returns
    -------
    Loading or None
        None where there was no distribution at all --- a row with no blade, or
        a section above a clearance gap. A distribution that *exists* but
        carries no interior peak comes back with :attr:`Loading.ma_max` and
        :attr:`Loading.zeta_max` measured and the four fitted fields NaN, which
        is the honest answer for a blade that accelerates to its trailing edge:
        there is something to describe and nothing to place a peak at. Callers
        that iterate check the field they use; a metric records both.

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
    # exit value. Taken before the suction side is folded out, which is the
    # only moment both sides are still there.
    ma_TE = 0.5 * float(mas[0] + mas[-1])
    if not ma_TE:
        return None

    # The geometric nose anchors the stagnation search, exactly as the surface
    # distribution plot anchors it.
    xrt_nose = blade.evaluate_section(spf, nchord=N_CHORD_NOSE)[0][:, 0]
    zeta = turbigen.util.normalise_surface_distance(cut, mas, xrt_nose)

    folded, suction = turbigen.util.suction_side(zeta, mas)
    i_max = int(np.argmax(suction))

    # Read straight off the data rather than off the two-line fit below: the
    # front value does not need a peak to be meaningful, only a point at
    # `zeta_front`, which interpolating the suction side always has.
    ma_front = float(np.interp(zeta_front, folded, suction))

    zeta_peak, ma_peak, _ = turbigen.util.loading_from_distribution(
        folded, suction, zeta_front, zeta_TE
    )
    fitted = np.isfinite(ma_peak)

    return Loading(
        zeta_peak=zeta_peak,
        fac_front=ma_front / ma_TE * mach_ratio(machine, i_row),
        fac_peak=ma_peak / ma_TE if fitted else np.nan,
        ma_peak=ma_peak,
        ma_TE=ma_TE,
        ma_max=float(suction[i_max]),
        zeta_max=float(folded[i_max]),
    )
