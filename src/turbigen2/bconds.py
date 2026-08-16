"""Boundary conditions for a meshed grid.

The mesher creates the patches; this puts the design's flow onto them. Like
:mod:`turbigen2.guess` it is a free function of a grid and the machine it was
meshed from, so it can be run and inspected without solving anything.

Kept out of the mesher deliberately. Reference scales belong there because they
have to precede any flow being written at all, but boundary conditions are the
operating point --- the thing you vary while holding a mesh fixed. Baked into
the mesher, a speedline would cost a re-mesh per point.

Shaft speed is the clearest case of that, and it divides the same way: the
mesher decides *where* a wall is not attached to its row, and this decides *how
fast* everything turns. So a rotating patch is placed by geometry and valued
here, and changing the speed is a boundary condition rather than a redesign.
"""

import logging

import numpy as np

logger = logging.getLogger("turbigen")

OMEGA_CASING = 0.0
"""Angular velocity of a wall a `RotatingPatch` marks off [rad/s].

Zero, because the only such wall is a casing over a tip gap and a casing does
not turn. It is named rather than written in place because it is the value a
contra-rotating casing would change, and because zero here means "stationary in
the absolute frame" rather than "no rotation configured".
"""


def apply(grid, machine):
    """Impose the design's operating point on `grid`, in place.

    Stagnation pressure and temperature and both flow angles go onto every
    inlet patch, static pressure onto every outlet patch, and each row's shaft
    speed onto its blocks and their walls.

    Parameters
    ----------
    grid : ember.grid.Grid
        A meshed grid, which is modified.
    machine : Machine
        The design the grid was meshed from.

    """
    inlet = machine.mean_line.inlet
    outlet = machine.mean_line.outlet

    patches_in = grid.patches.inlet
    patches_out = grid.patches.outlet
    if not patches_in or not patches_out:
        raise ValueError(
            f"The grid has {len(patches_in)} inlet and {len(patches_out)} outlet "
            f"patches; boundary conditions need at least one of each."
        )

    for patch in patches_in:
        patch.set_Po_To(float(inlet.Po), float(inlet.To))
        patch.set_Alpha(float(inlet.Alpha))
        patch.set_Beta(float(inlet.Beta))

    for patch in patches_out:
        patch.set_P(float(outlet.P))

    apply_rotation(grid, machine)

    logger.debug(
        f"Inlet Po={float(inlet.Po):.4g} Pa, To={float(inlet.To):.4g} K, "
        f"Alpha={float(inlet.Alpha):.4g} deg; outlet P={float(outlet.P):.4g} Pa"
    )


def apply_rotation(grid, machine):
    """Set every row turning at the speed its mean line was designed for.

    A block's angular velocity is the frame its row is solved in, and every
    wall of that block turns with it unless a `RotatingPatch` says otherwise.
    So a shrouded row needs only the block speed, and the patches the mesher
    placed over tip gaps are set to :data:`OMEGA_CASING` --- the two halves of
    the same statement, made together so they cannot disagree.

    Parameters
    ----------
    grid : ember.grid.Grid
        A meshed grid, which is modified.
    machine : Machine
        The design the grid was meshed from.

    """
    Omega = np.asarray(machine.mean_line.Omega, dtype=float)
    # One number per row, taken from the row inlet: a mean line carries a value
    # per station and both stations of a row share a shaft.
    Omega_row = Omega[0] if Omega.ndim > 1 else np.atleast_1d(Omega)[::2]

    rows = grid.rows
    if len(rows) != len(Omega_row):
        raise ValueError(
            f"The grid has {len(rows)} row(s) but the mean line has "
            f"{len(Omega_row)}; boundary conditions need one speed per row."
        )

    for i_row, (blocks, Omega_now) in enumerate(zip(rows, Omega_row)):
        for block in blocks:
            block.set_Omega(float(Omega_now))
            for patch in block.patches.rotating:
                patch.set_Omega(OMEGA_CASING)

        logger.debug(f"Row {i_row}: Omega={float(Omega_now):.5g} rad/s")

    # No check that every rotating patch was reached, because there is no way
    # to miss one: `grid.rows` groups by periodic and mixing connectivity and
    # puts even a wholly disconnected block in a row of its own, so the loop
    # above visits every block on the grid. A grid whose rows and mean line
    # disagree is caught by count above, which is the only failure left.
    #
    # Worth knowing that an unvalued patch would be loud rather than quiet
    # anyway: `RotatingPatch` defaults to Omega=nan, not zero, so one that
    # never reached here would turn the solution to NaN rather than run a rotor
    # as though it were stationary. `test_bconds.py` pins that.
