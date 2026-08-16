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

from turbigen2.node import Node

logger = logging.getLogger("turbigen")

OMEGA_CASING = 0.0
"""Angular velocity of a wall a `RotatingPatch` marks off [rad/s].

Zero, because the only such wall is a casing over a tip gap and a casing does
not turn. It is named rather than written in place because it is the value a
contra-rotating casing would change, and because zero here means "stationary in
the absolute frame" rather than "no rotation configured".
"""


class OperatingPoint(Node):
    """Where a fixed machine is run, as a departure from its design point.

    A design states one condition; a machine has a whole characteristic. This
    is how to reach the rest of it without redesigning anything, which is why
    it is read here and not by any design stage --- and why it sits outside
    :data:`turbigen2.database.SUBTREE`, so that two runs of one machine at
    different back pressures are not read as two different designs.
    """

    DP_adjust: float = 0.0
    """Change in pressure change through the machine, as a fraction [--].

    The exit static pressure is moved so that

    .. math::

        \\Delta p = \\Delta p_\\mathrm{design} (1 + \\mathtt{DP\\_adjust}),
        \\qquad
        \\Delta p_\\mathrm{design} = p_{0,\\mathrm{in}} - p_\\mathrm{out}

    so zero reproduces the design exactly and positive always means *more*
    pressure change --- more throttled for a compressor, more expanded for a
    turbine. One formula covers both because the design's own
    :math:`\\Delta p` carries the sign, negative for a machine that raises
    pressure, and neither the sign convention nor a machine type has to appear
    in the file.

    **A pressure change and not a pressure ratio**, which is the whole reason
    this field exists. A ratio measures from one rather than from zero, so a
    fraction of it is not a fraction of anything physical, and the error grows
    without limit as a machine gets slower. Adjusting the same cascade by
    "5 per cent": through the pressure ratio it is 1.16x the design pressure
    change at Ma = 0.6 and 3.14x at Ma = 0.05, where through this field it is
    1.05x at both. The package this replaces offers only the ratio.

    The rule generalises --- adjust what vanishes when there is no machine,
    never what goes to one --- and is the same trap
    :meth:`turbigen2.iterate.MeanLine.tolerances` guards against for a design
    variable whose nominal is zero.
    """


def apply(grid, machine, operating_point=None):
    """Impose an operating point on `grid`, in place.

    Stagnation pressure and temperature and both flow angles go onto every
    inlet patch, static pressure onto every outlet patch, and each row's shaft
    speed onto its blocks and their walls.

    Parameters
    ----------
    grid : ember.grid.Grid
        A meshed grid, which is modified.
    machine : Machine
        The design the grid was meshed from.
    operating_point : OperatingPoint or None
        Where to run it, as a departure from the design point. ``None`` is the
        design point itself, which is what a config that says nothing means.

    """
    inlet = machine.mean_line.inlet

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

    P_out = exit_pressure(machine, operating_point)
    for patch in patches_out:
        patch.set_P(P_out)

    apply_rotation(grid, machine)

    logger.debug(
        f"Inlet Po={float(inlet.Po):.4g} Pa, To={float(inlet.To):.4g} K, "
        f"Alpha={float(inlet.Alpha):.4g} deg; outlet P={P_out:.4g} Pa"
    )


def exit_pressure(machine, operating_point=None):
    """Return the static pressure to hold at exit [Pa].

    The design's own, moved by :attr:`OperatingPoint.DP_adjust`. Split out so
    that where a machine is being run can be read, and tested, without a grid.

    Parameters
    ----------
    machine : Machine
        The design, which supplies the pressure change to scale.
    operating_point : OperatingPoint or None
        ``None`` is the design point.

    Returns
    -------
    float

    """
    Po_in = float(machine.mean_line.inlet.Po)
    P_out = float(machine.mean_line.outlet.P)

    if operating_point is None or not operating_point.DP_adjust:
        return P_out

    # Signed, so that one line covers a machine that raises pressure and one
    # that drops it: DP_design is negative for a compressor, and a positive
    # adjustment makes it more negative -- more rise -- exactly as it makes a
    # turbine's more positive.
    DP_design = Po_in - P_out
    DP = DP_design * (1.0 + operating_point.DP_adjust)

    P_out_now = Po_in - DP
    if P_out_now <= 0.0:
        raise ValueError(
            f"DP_adjust={operating_point.DP_adjust} asks for an exit pressure "
            f"of {P_out_now:.4g} Pa, which is not a pressure. The design's "
            f"pressure change is {DP_design:.4g} Pa."
        )

    logger.info(
        f"Operating point: DP={DP:.5g} Pa against a design {DP_design:.5g} Pa, "
        f"so exit P={P_out_now:.5g} Pa against a design {P_out:.5g} Pa."
    )
    return P_out_now


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
