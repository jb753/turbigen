"""Boundary conditions for a meshed grid.

The mesher creates the patches; this puts the design's flow onto them. Like
:mod:`turbigen2.guess` it is a free function of a grid and the machine it was
meshed from, so it can be run and inspected without solving anything.

Kept out of the mesher deliberately. Reference scales belong there because they
have to precede any flow being written at all, but boundary conditions are the
operating point --- the thing you vary while holding a mesh fixed. Baked into
the mesher, a speedline would cost a re-mesh per point.
"""

import logging

logger = logging.getLogger("turbigen")


def apply(grid, machine):
    """Impose the design's inlet and outlet conditions on `grid`, in place.

    Stagnation pressure and temperature and both flow angles go onto every
    inlet patch, and static pressure onto every outlet patch, taken from the
    ends of the mean line in streamwise order.

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

    logger.debug(
        f"Inlet Po={float(inlet.Po):.4g} Pa, To={float(inlet.To):.4g} K, "
        f"Alpha={float(inlet.Alpha):.4g} deg; outlet P={float(outlet.P):.4g} Pa"
    )
