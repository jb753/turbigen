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

import turbigen.annulus

logger = logging.getLogger("turbigen")


def mean_line(grid, machine, offset=turbigen.annulus.CUT_OFFSET):
    """Return the mean line `grid` actually achieved, and its mixing loss.

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
    ndarray
        Specific entropy rise from mixing each cut to uniformity [J/kg/K], one
        per station in the same shape as the mean line. This is the loss the
        reduction itself introduces --- the mixed-out entropy less the
        mass-averaged entropy of the cut --- and it is independent of the area
        contraction, which is reversible.

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

    # One mixing loss per cut, filled in streamwise order and reshaped to the
    # mean line's (2, n_row) layout at the end -- `flat` runs inlet-to-outlet,
    # so station i is row i // 2, end i % 2.
    Ds_mix_flat = np.empty(2 * machine.annulus.n_row)

    for i_station, xr in enumerate(machine.annulus.cut_planes(offset)):
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

        # The loss the reduction introduces: entropy of the uniform state less
        # the mass-averaged entropy of the cut it replaced. The AR contraction
        # is isentropic and does not move `mixed.s`, so this is measured at the
        # true cut area regardless of it. `mass_average` only takes a structured
        # block, so the cut is interpolated to the resolution of the row it
        # sits beside first.
        i_row = i_station // 2
        nj, nk = grid.rows[i_row][0].shape[1:]
        structured = ember.cut.interpolate_to_structured(cut, (nj, nk))
        s_cut = float(ember.average.mass_average(structured.s, structured))
        Ds_mix_flat[i_station] = float(mixed.s) - s_cut

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

    Ds_mix = Ds_mix_flat.reshape((machine.annulus.n_row, 2)).T

    return actual, Ds_mix
