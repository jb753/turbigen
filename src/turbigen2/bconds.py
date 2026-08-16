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
from typing import ClassVar

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


class InletProfile(Node):
    """A non-uniform inlet, as a spanwise perturbation from the mean line.

    Every field is a *departure* from what the design asked for, so zero is
    uniform and an absent section is exactly what this package did before there
    was one. Interpolated onto whatever span fractions the inlet patch has, so
    a profile does not have to know the mesh.

    Non-dimensionalised by inlet quantities that vanish with the flow rather
    than by the absolute level, for the reason
    :attr:`OperatingPoint.DP_adjust` is a pressure change and not a ratio: at
    low speed :math:`p_0` and :math:`p` converge, so a fraction of :math:`p_0`
    is not a fraction of anything physical, while :math:`p_0 - p` stays
    meaningful at every Mach number. A boundary layer therefore reads as
    ``DPo`` running from 0 in the free stream to -1 at the wall --- the
    fraction of dynamic head lost, which is a number that carries between
    machines.

    **The two scales are the same scale in disguise**, since
    :math:`(p_0-p)/p \\simeq \\gamma \\mathit{Ma}^2/2` and
    :math:`(T_0-T)/T = (\\gamma-1)\\mathit{Ma}^2/2`. So equal perturbations in
    ``DPo`` and ``DTo`` are *isentropic*, which makes each physical case a
    clean statement:

    ===========================  ===============================
    a clean velocity distortion  ``DPo`` and ``DTo`` equal
    a boundary layer or wake     ``DPo`` alone
    a hot streak                 ``DTo`` alone
    ===========================  ===============================

    Scaling ``DTo`` by the machine's temperature *rise* instead --- the closer
    analogue of ``DP_adjust`` --- was considered and rejected: it breaks that
    property, needs to know the machine's duty, and divides by zero for a
    cascade.

    Spanwise only. ember refuses a pitchwise-varying prescription at an inlet
    patch rather than averaging it, so there is nothing here to express one
    with.
    """

    spf: tuple[float, ...]
    """Span fractions the profile is given at, hub to casing [--].

    Must run from exactly 0 to exactly 1. Interpolation clamps outside the
    range it is given, so a profile stated over ``[0.1, 0.9]`` would quietly
    hold its end values across the rest of the span instead of saying it was
    incomplete.
    """

    DPo: tuple[float, ...] = ()
    """Stagnation pressure deficit, as a fraction of inlet dynamic head [--].

    :math:`(p_0 - p_{0,\\mathrm{nom}}) / (p_0 - p)_\\mathrm{nom}`. Empty for
    uniform stagnation pressure.
    """

    DTo: tuple[float, ...] = ()
    """Stagnation temperature excess, as a fraction of inlet dynamic
    temperature [--].

    :math:`(T_0 - T_{0,\\mathrm{nom}}) / (T_0 - T)_\\mathrm{nom}`. Empty for
    uniform stagnation temperature.
    """

    DAlpha: tuple[float, ...] = ()
    """Yaw angle added to the design value [deg]. Empty for uniform swirl."""

    DBeta: tuple[float, ...] = ()
    """Pitch angle added to the design value [deg]. Empty for uniform pitch."""

    def __post_init__(self):
        # Checked when the config is read: none of it needs a design, and a
        # profile that cannot be applied should not survive to the point where
        # a grid exists to apply it to.
        spf = np.asarray(self.spf, dtype=float)

        if spf.size < 2:
            raise ValueError(
                f"An inlet profile needs at least two span fractions, got "
                f"{list(spf)}."
            )
        if np.any(np.diff(spf) <= 0.0):
            raise ValueError(
                f"Inlet profile spf must increase from hub to casing, got "
                f"{list(spf)}."
            )
        if spf[0] != 0.0 or spf[-1] != 1.0:
            raise ValueError(
                f"An inlet profile must span the whole annulus, from spf 0 to "
                f"spf 1, but this one runs {spf[0]} to {spf[-1]}. Interpolation "
                f"clamps, so the ends would be held rather than reported "
                f"missing."
            )

        given = [name for name in self.COLUMNS if getattr(self, name)]
        if not given:
            raise ValueError(
                f"An inlet profile perturbs nothing: give at least one of "
                f"{list(self.COLUMNS)}, or leave the section out."
            )

        for name in given:
            column = np.asarray(getattr(self, name), dtype=float)
            if column.shape != spf.shape:
                raise ValueError(
                    f"Inlet profile {name} has {column.size} value(s) against "
                    f"{spf.size} span fraction(s)."
                )

    COLUMNS: ClassVar[tuple[str, ...]] = ("DPo", "DTo", "DAlpha", "DBeta")
    """The perturbations, in the order they are reported."""

    def column(self, name, spf):
        """Return one perturbation interpolated onto `spf`, or zero."""
        values = getattr(self, name)
        if not values:
            return np.zeros_like(np.asarray(spf, dtype=float))
        return np.interp(spf, np.asarray(self.spf, dtype=float), values)

    def state(self, spf, inlet):
        """Return the absolute inlet state at span fractions `spf`.

        The node that defines what its numbers mean is the node that turns them
        into a state, which is why this lives here rather than in `apply` ---
        the same reason `MeanLine.to_dict` owns the choice of quantities and
        the datum rule rather than whatever writes one out.

        Parameters
        ----------
        spf : array_like
            Span fractions to evaluate at, typically an inlet patch's own.
        inlet : MeanLine
            The design's inlet station, supplying both the values perturbed
            from and the scales perturbed by.

        Returns
        -------
        dict
            ``Po``, ``To``, ``Alpha`` and ``Beta``, each an array over `spf`.

        """
        spf = np.asarray(spf, dtype=float)

        Po = float(inlet.Po)
        To = float(inlet.To)
        # The scales: stagnation minus static, which is what vanishes when the
        # flow does. Both are positive for any moving fluid.
        q = Po - float(inlet.P)
        dT = To - float(inlet.T)

        return {
            "Po": Po + self.column("DPo", spf) * q,
            "To": To + self.column("DTo", spf) * dT,
            "Alpha": float(inlet.Alpha) + self.column("DAlpha", spf),
            "Beta": float(inlet.Beta) + self.column("DBeta", spf),
        }


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


def _spanwise(patch, values):
    """Return `values` laid out on `patch`'s own axes, spanwise.

    A bare ``(nspan,)`` array is documented as acceptable, and for three of the
    four setters it is. :meth:`ember.inlet.InletPatch.set_Beta` is the
    exception: it subtracts the face angle ``chi_node`` *before* the shape is
    checked, and that is already on the patch's axes, so a spanwise array
    broadcasts against it into a pitchwise-varying one and is then refused for
    varying pitchwise. Giving every setter the patch's own axes sidesteps that
    and keeps one shape for all four.
    """
    shape = [1] * len(patch.block_view.shape)
    shape[patch.span_dim] = -1
    return np.asarray(values, dtype=float).reshape(shape)


def apply(grid, machine, operating_point=None, inlet_profile=None):
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
    inlet_profile : InletProfile or None
        What feeds it, if not a uniform flow. A separate argument from the
        operating point, and not a field of it, because the same profile
        applies at every point of a characteristic: what feeds a machine does
        not change because you moved along its map.

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
        if inlet_profile is None:
            patch.set_Po_To(float(inlet.Po), float(inlet.To))
            patch.set_Alpha(float(inlet.Alpha))
            patch.set_Beta(float(inlet.Beta))
        else:
            # Evaluated per patch, on its own span fractions, so a profile is a
            # statement about the annulus rather than about a mesh.
            state = inlet_profile.state(patch.spf, inlet)
            patch.set_Po_To(
                _spanwise(patch, state["Po"]), _spanwise(patch, state["To"])
            )
            patch.set_Alpha(_spanwise(patch, state["Alpha"]))
            patch.set_Beta(_spanwise(patch, state["Beta"]))

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
