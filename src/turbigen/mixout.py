"""Reducing a solved grid back to a mean line.

Cuts the flow at each design station, mixes each cut out to a uniform state,
and assembles them into a :class:`~turbigen.meanline.MeanLine` --- the *actual*
mean line, against which the nominal one can be compared.

A free function of a grid and the machine it was meshed from, like
:mod:`turbigen.guess` and :mod:`turbigen.bconds`, so it can be run on a
solution without re-running anything.
"""

import logging

import numpy as np
import ember.average
import ember.cut

logger = logging.getLogger("turbigen")

CUT_OFFSET = 0.02
"""Cut planes sit this fraction of blade chord into the gap, clear of the row.

Cutting exactly at a leading or trailing edge would put the plane inside the
blade, where there is no single annulus-spanning surface to integrate over.
"""


def cut_planes(annulus, offset=CUT_OFFSET):
    """Return the meridional cut curve at each design station.

    A cut plane is the straight hub-to-casing line at one meridional position,
    so it is two points and needs nothing from the annulus beyond
    :meth:`~turbigen.annulus.Annulus.evaluate_xr`.

    Parameters
    ----------
    annulus : Annulus
        Geometry to cut.
    offset : float
        Distance into the adjacent gap, as a fraction of blade chord.

    Returns
    -------
    list of ndarray
        One ``(2, 2)`` array per station, in streamwise order, each holding
        two ``(x, r)`` points. This is the shape
        :func:`ember.cut.unstructured` takes.

    """
    n_row = annulus.n_row
    chords = annulus.chords(0.5)

    # The offset is given in blade chords but applied to `m`, which is
    # normalised per segment -- so it has to be rescaled by the chord of the
    # gap each station opens into. Rows are the odd segments, gaps the even.
    chord_blade = np.repeat(chords[1::2], 2)
    gaps = chords[::2]
    chord_gap = np.concatenate([[gaps[0]], np.repeat(gaps[1:-1], 2), [gaps[-1]]])

    # Leading edges step upstream, trailing edges downstream.
    signed = offset * np.ones(2 * n_row)
    signed[::2] *= -1.0
    signed *= chord_blade / chord_gap

    m_cut = np.arange(1.0, 2 * n_row + 1) + signed

    return [annulus.evaluate_xr(m, [0.0, 1.0]).T for m in m_cut]


def mean_line(grid, machine, offset=CUT_OFFSET):
    """Return the mean line `grid` actually achieved.

    Parameters
    ----------
    grid : ember.grid.Grid
        A solved grid. Not modified.
    machine : Machine
        The design it was meshed from.
    offset : float
        Cut plane offset, as a fraction of blade chord.

    Returns
    -------
    MeanLine
        Mixed-out state at each design station, in the same shape as the
        nominal mean line.

    """
    # A copy of the nominal, so the actual starts with the annulus areas it was
    # designed for. Those are what the contraction below reports against, and a
    # cut cannot measure them.
    actual = machine.mean_line.copy()
    flat = actual.flat
    nominal = machine.mean_line.flat

    # Shaft speed is the one thing here that comes from the grid rather than
    # from either the design or a cut. A cut genuinely cannot measure it, but
    # the blocks were told it, and what they were told is what the solver used
    # -- so this reports the speed that ran rather than the speed that was
    # asked for. The two are the same today and will not be as soon as an
    # operating point adjusts one, and every `_rel` quantity on this mean line
    # is derived from it: copied from the design, an off-design run would
    # report its relative Mach numbers in the wrong rotating frame, with
    # entirely plausible values.
    actual.set_Omega([float(blocks[0].Omega) for blocks in grid.rows])

    for i_station, xr in enumerate(cut_planes(machine.annulus, offset)):
        cut = ember.cut.unstructured(grid, xr)
        if cut is None:
            raise ValueError(
                f"The cut plane for station {i_station} at {xr.tolist()} does "
                f"not intersect the grid."
            )

        # Contract the mixed-out state from the area the cut actually spans to
        # the design annulus area, so every station is reported at the area the
        # mean line was designed for. The cut covers one blade passage, hence
        # the blade count; only the meridional components of the area vector
        # count, the tangential one being the periodic faces.
        A_cut = float(np.linalg.norm(ember.average.total_area(cut)[:2]) * cut.Nb)
        Am_nominal = float(nominal[i_station].Am)
        AR = Am_nominal / A_cut

        logger.debug(
            f"Station {i_station}: A_cut={A_cut:.6g} m2 (Nb={cut.Nb}), "
            f"Am={Am_nominal:.6g} m2, AR={AR:.6g}"
        )

        try:
            mixed = ember.average.mix_out(cut, AR=AR)
        except Exception as err:
            raise ValueError(f"Could not mix out station {i_station}: {err}") from err

        # Dimensional state, not `mixed.conserved`. The cut carries the grid's
        # fluid and this mean line carries the design's, and those have
        # different datums -- conserved energy is measured from one, so copying
        # it across would silently reinterpret it.
        station = flat[i_station]
        station.set_r(float(mixed.r))
        station.set_P_T(float(mixed.P), float(mixed.T))
        station.set_Vx(float(mixed.Vx))
        station.set_Vr(float(mixed.Vr))
        station.set_Vt(float(mixed.Vt))

    return actual
