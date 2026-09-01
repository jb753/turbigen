"""Boundary conditions for a meshed grid.

The mesher creates the patches; this puts the design's flow onto them. Like
:mod:`turbigen.guess` it is a free function of a grid and the machine it was
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

import dataclasses
import logging
from typing import ClassVar

import numpy as np
from numpy.polynomial import legendre

from turbigen.node import Node

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

    Two members, differing only in how a column is written down. `Sampled` is
    values at span fractions, which is what a person writes from rig data.
    `Legendre` is the coefficients of a series, which is what anything
    *producing* a profile analytically should write --- storing such a profile
    as samples and interpolating it back is pure loss, and measurably so: a
    degree-3 profile kept at 21 span points comes back with a maximum error of
    2.7e-3, a quarter of the tolerance
    :class:`turbigen.iterate.Repeat` converges to, and at 11 points the error
    exceeds the tolerance outright.
    """

    COLUMNS: ClassVar[tuple[str, ...]] = ("DPo", "DTo", "DAlpha", "DBeta")
    """The perturbations, in the order they are reported.

    ``DPo`` and ``DTo`` are fractions of inlet dynamic head and dynamic
    temperature; ``DAlpha`` and ``DBeta`` are degrees added to the design
    angle. Empty means uniform in that quantity.
    """

    def column(self, name, spf):
        """Return one perturbation evaluated at `spf`, or zero if not given."""
        raise NotImplementedError(
            f"{type(self).__name__} must implement column(self, name, spf)"
        )

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


class Sampled(InletProfile):
    """A profile given as values at span fractions.

    What a person writes from rig data or a measured traverse. Interpolated
    linearly onto the patch, so the span fractions given are the resolution the
    profile has.
    """

    type: ClassVar[str] = "sampled"

    spf: tuple[float, ...]
    """Span fractions the profile is given at, hub to casing [--].

    Must run from exactly 0 to exactly 1. Interpolation clamps outside the
    range it is given, so a profile stated over ``[0.1, 0.9]`` would quietly
    hold its end values across the rest of the span instead of saying it was
    incomplete.
    """

    DPo: tuple[float, ...] = ()
    """Stagnation pressure deficit, as a fraction of inlet dynamic head [--]."""

    DTo: tuple[float, ...] = ()
    """Stagnation temperature excess, as a fraction of inlet dynamic
    temperature [--]."""

    DAlpha: tuple[float, ...] = ()
    """Yaw angle added to the design value [deg]."""

    DBeta: tuple[float, ...] = ()
    """Pitch angle added to the design value [deg]."""

    def __post_init__(self):
        # Checked when the config is read: none of it needs a design, and a
        # profile that cannot be applied should not survive to the point where
        # a grid exists to apply it to.
        spf = np.asarray(self.spf, dtype=float)

        if spf.size < 2:
            raise ValueError(
                f"An inlet profile needs at least two span fractions, got {list(spf)}."
            )
        if np.any(np.diff(spf) <= 0.0):
            raise ValueError(
                f"Inlet profile spf must increase from hub to casing, got {list(spf)}."
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

    def column(self, name, spf):
        values = getattr(self, name)
        if not values:
            return np.zeros_like(np.asarray(spf, dtype=float))
        return np.interp(spf, np.asarray(self.spf, dtype=float), values)


class Legendre(InletProfile):
    """A profile given as the coefficients of a Legendre series over the span.

    Evaluated at whatever span fractions the inlet patch has, so nothing is
    resampled and the mesh's own resolution is what the profile is applied at.
    That is the point of the member: anything producing a profile analytically
    --- :class:`turbigen.iterate.Repeat` above all --- would otherwise have to
    write it out as samples and lose accuracy doing so.

    Shifted to the span, so mode ``n`` is :math:`P_n(2\\,\\mathit{spf} - 1)`.
    Orthogonal, so the coefficients are independent: truncating drops a mode
    rather than redistributing the others, which is what makes a low order a
    *statement* rather than a fit artefact.

    **There is no constant term.** The lists start at mode 1, so a profile
    cannot carry a level. A level is the mean line's business, and one here
    would fight the design it is supposed to perturb --- the whole point of the
    node being that it redistributes and nothing else.

    There is no ``order`` field either: the order is the length of the lists,
    so nothing can contradict them.
    """

    type: ClassVar[str] = "legendre"

    DPo: tuple[float, ...] = ()
    """Coefficients of modes 1 upwards, in fractions of inlet dynamic head."""

    DTo: tuple[float, ...] = ()
    """Coefficients of modes 1 upwards, in fractions of dynamic temperature."""

    DAlpha: tuple[float, ...] = ()
    """Coefficients of modes 1 upwards, in degrees."""

    DBeta: tuple[float, ...] = ()
    """Coefficients of modes 1 upwards, in degrees."""

    def __post_init__(self):
        given = [name for name in self.COLUMNS if getattr(self, name)]
        if not given:
            raise ValueError(
                f"An inlet profile perturbs nothing: give at least one of "
                f"{list(self.COLUMNS)}, or leave the section out."
            )

        # One order for the whole profile, so a column cannot silently be
        # fitted to a different resolution from its neighbours.
        orders = {name: len(getattr(self, name)) for name in given}
        if len(set(orders.values())) > 1:
            raise ValueError(
                f"Every column of a Legendre inlet profile must have the same "
                f"number of coefficients, but got {orders}."
            )

    @property
    def order(self):
        """Highest Legendre mode carried, the lists starting at mode 1."""
        for name in self.COLUMNS:
            values = getattr(self, name)
            if values:
                return len(values)
        return 0

    def column(self, name, spf):
        spf = np.asarray(spf, dtype=float)
        values = getattr(self, name)
        if not values:
            return np.zeros_like(spf)

        # The leading zero is the absent constant term, which `legval` needs a
        # slot for and this node refuses to have a value in.
        return legendre.legval(2.0 * spf - 1.0, np.concatenate([[0.0], values]))


class OperatingPoint(Node):
    """Where a fixed machine is run, as a departure from its design point.

    A design states one condition; a machine has a whole characteristic. This
    is how to reach the rest of it without redesigning anything, which is why
    it is read here and not by any design stage --- and why it sits outside
    :data:`turbigen.database.SUBTREE`, so that two runs of one machine at
    different back pressures are not read as two different designs.
    """

    # The exit static pressure is moved so that
    #
    #   dp = dp_design * (1 + DP_adjust),  dp_design = Po_in - p_out
    #
    # so zero reproduces the design exactly and positive always means *more*
    # pressure change -- more throttled for a compressor, more expanded for a
    # turbine. One formula covers both because the design's own dp carries the
    # sign, negative for a machine that raises pressure, and neither the sign
    # convention nor a machine type has to appear in the file.
    #
    # A pressure change and not a pressure ratio, which is the whole reason
    # this field exists. A ratio measures from one rather than from zero, so a
    # fraction of it is not a fraction of anything physical, and the error
    # grows without limit as a machine gets slower. Adjusting the same cascade
    # by "5 per cent": through the pressure ratio it is 1.16x the design
    # pressure change at Ma = 0.6 and 3.14x at Ma = 0.05, where through this
    # field it is 1.05x at both. The package this replaces offers only the
    # ratio.
    #
    # The rule generalises -- adjust what vanishes when there is no machine,
    # never what goes to one -- and is the same trap MeanLine.tolerances
    # guards against for a design variable whose nominal is zero.
    DP_adjust: float = 0.0
    """Change in the design pressure change through the machine, as a fraction [--]."""

    # The other way to move along a characteristic. Where DP_adjust states an
    # exit pressure and lets the mass flow be whatever that draws, this states
    # a mass flow, mdot = mdot_design * (1 + mdot_adjust), and lets the
    # pressure be whatever holds it. The exit pressure computed from the design
    # is still imposed, but as a *starting point*: a proportional-integral
    # controller on the outlet patch (ember.outlet.OutletPatch.set_throttle)
    # then moves it each step until the measured mass flow reaches the target.
    # What the boundary imposes is still a pressure, so nothing about the
    # characteristic treatment changes -- the throttle only chooses which
    # pressure.
    #
    # None, the default, is no throttle at all: the exit pressure stands as
    # exit_pressure() set it, and the mass flow is an outcome. This is the
    # distinction the field exists to make, and the reason it is not simply 0.0
    # by default -- a design that asks for a mass flow and a design that
    # accepts one are different requests, and zero cannot say both.
    #
    # Which of the two to state is a property of the design, not a preference.
    # A design whose variables include mdot -- a fan parametrised on mass flow
    # and total pressure rise, say -- has no way to report whether it achieved
    # them if the mass flow is left to drift, because the nominal-vs-actual
    # table would be comparing the design against a different operating point.
    # A design parametrised on Mach number and exit angle does not care, and
    # the simpler prescribed pressure is right.
    #
    # The gains are ember's and are not exposed here. They are dimensionless
    # and scaled on the reference quantities, which MeanLine.get_referenced_fluid
    # takes from the design's own mean density and velocity -- representative
    # by construction, which is the condition ember states for its defaults
    # holding.
    #
    # Two restrictions, both from ember. Only one outlet patch may be
    # throttled, so a grid whose exit is spread over several blocks is refused
    # rather than over-throttled by the number of them; and the target must be
    # positive, so an adjustment of -1 or below is refused below.
    #
    # DP_adjust still applies alongside this, as the pressure the controller
    # *starts* from, and achieved() writes the pressure back once the mass flow
    # has been reached -- so a throttled run archives the operating point it
    # found, and the next one starts nearer to it.
    mdot_adjust: float | None = None
    """Change in the design mass flow, as a fraction [--]; null for no throttle."""

    def __post_init__(self):
        # Checked when the config is read, because it needs no design.
        if self.mdot_adjust is None:
            return

        if self.mdot_adjust <= -1.0:
            raise ValueError(
                f"mdot_adjust={self.mdot_adjust} asks for a mass flow of "
                f"{1.0 + self.mdot_adjust:.4g} times the design, which is not "
                f"a flow the exit can be throttled to."
            )


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
    speed onto its blocks and their walls. An operating point that states a
    mass flow additionally throttles the exit to it, the prescribed pressure
    becoming the controller's starting point rather than its answer.

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

    apply_throttle(patches_out, machine, operating_point)

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


def exit_mdot(machine, operating_point=None):
    """Return the annulus mass flow to throttle the exit to [kg/s], or None.

    ``None`` whenever nothing asked for a throttle, which is what an absent
    ``operating_point`` and an absent :attr:`OperatingPoint.mdot_adjust` both
    mean. Split out from :func:`apply` for the same reason
    :func:`exit_pressure` is: where a machine is being run should be readable,
    and testable, without a grid.

    Parameters
    ----------
    machine : Machine
        The design, which supplies the mass flow to scale.
    operating_point : OperatingPoint or None
        ``None`` is the design point, prescribing pressure.

    Returns
    -------
    float or None

    """
    if operating_point is None or operating_point.mdot_adjust is None:
        return None

    # The machine outlet, because that is the station the throttled patch sits
    # at. On a machine that neither bleeds nor is fed part-way this is the
    # inlet value as well, but reading it where the patch is means a machine
    # that does one of those still throttles to the flow passing the exit.
    mdot_design = float(machine.mean_line.outlet.mdot)
    mdot = mdot_design * (1.0 + operating_point.mdot_adjust)

    logger.info(
        f"Operating point: throttling to mdot={mdot:.5g} kg/s against a "
        f"design {mdot_design:.5g} kg/s."
    )
    return mdot


def apply_throttle(patches_out, machine, operating_point=None):
    """Throttle the exit to a mass flow, or leave its pressure prescribed.

    Always one or the other, never neither: a patch that is not being
    throttled has its throttle *cleared*, so that applying an operating point
    twice to one grid says the same thing as applying it once. `bconds` exists
    to be re-run on a meshed grid --- that is what makes a speedline cost no
    re-mesh --- and a controller left wound up from the previous point would
    be the one piece of state that did not come back.

    Parameters
    ----------
    patches_out : list of ember.outlet.OutletPatch
        Every outlet patch on the grid.
    machine : Machine
        The design the grid was meshed from.
    operating_point : OperatingPoint or None

    """
    mdot = exit_mdot(machine, operating_point)

    if mdot is None:
        for patch in patches_out:
            patch.set_throttle(None)
        return

    # Refused here rather than at the march, where `ember.solver
    # ._validate_throttle` would catch it several stages later: this is the
    # place that knows the exit is one patch or many, and the answer does not
    # depend on anything a solve produces. Splitting the target between them
    # is not an alternative -- the split is only known once the answer is.
    if len(patches_out) > 1:
        raise ValueError(
            f"The grid has {len(patches_out)} outlet patches, and a mass flow "
            f"can only be throttled through one. Either mesh the exit as a "
            f"single patch or state DP_adjust instead of mdot_adjust."
        )

    (patch,) = patches_out

    # Per passage, which is what the patch measures: ember integrates the mass
    # flux over the faces it actually has, and the mesh carries one passage per
    # row. `Nb` is the same number `mixout` multiplies by to get back to the
    # annulus, so the two agree by construction.
    Nb = int(patch.block.Nb)
    patch.set_throttle(mdot / Nb)

    logger.debug(
        f"Throttling {patch.label!r} to {mdot / Nb:.5g} kg/s per passage, Nb={Nb}."
    )


SETTLED_TOL = 1.0e-3
"""Mass-flow error a throttle must be inside for its pressure to be read back.

A fraction of the target. The controller reaches parts in ten thousand on a
converged march, so this is loose enough not to reject a run that arrived and
tight enough that a pressure recorded as an operating point is one.
"""


def achieved(grid, machine, operating_point=None):
    """Return `operating_point` with the pressure a throttle settled at, or None.

    The inverse of :func:`exit_pressure` composed with :func:`apply_throttle`:
    the throttle moved the exit pressure, and this says where to, in the units
    the config states a pressure in. ``None`` whenever there is nothing to say
    --- no throttle, or one that has not arrived --- which leaves the caller's
    operating point as the caller wrote it.

    **The boundary level, not the mixed-out mean line.** They are different
    numbers: the mean line is cut at the design stations, 2% of a chord behind
    the trailing edge, and is mixed out at constant area, while this is the
    pitchwise-mean pressure imposed at the exit plane a duct further
    downstream. Only the boundary one reproduces the run when it is prescribed
    back, which is the whole purpose of recording it --- and
    :attr:`DP_adjust` is defined against the boundary, so only the boundary one
    is even the right quantity.

    Measured against the *nominal* pressure change, like every other
    :attr:`DP_adjust`, so a sweep that follows reads on one scale set by the
    design rather than one that moves with the solution.

    Parameters
    ----------
    grid : ember.grid.Grid
        The solved grid, read but not modified.
    machine : Machine
        The design it was meshed from.
    operating_point : OperatingPoint or None
        What was asked for.

    Returns
    -------
    OperatingPoint or None

    """
    if operating_point is None or operating_point.mdot_adjust is None:
        return None

    patches_out = grid.patches.outlet
    if len(patches_out) != 1:
        return None

    (patch,) = patches_out

    # A controller still moving has not chosen a pressure yet, and recording
    # where it happened to be would archive an operating point the run was
    # never at. Said out loud, because the alternative is a config that
    # silently keeps the guess it started from.
    stats = patch.get_throttle_stats()
    if not stats["mdot_target"]:
        return None

    error = stats["mdot_throttle"] / stats["mdot_target"] - 1.0
    if abs(error) > SETTLED_TOL:
        logger.warning(
            f"The throttle finished {error:+.2%} from its target, outside "
            f"{SETTLED_TOL:.2%}, so the exit pressure it reached is not "
            f"recorded as an operating point."
        )
        return None

    Po_in = float(machine.mean_line.inlet.Po)
    DP_design = Po_in - float(machine.mean_line.outlet.P)
    DP_adjust = (Po_in - float(patch.P_throttle)) / DP_design - 1.0

    logger.info(
        f"Throttle settled at exit P={float(patch.P_throttle):.5g} Pa, which "
        f"is DP_adjust={DP_adjust:.5g}."
    )

    return dataclasses.replace(operating_point, DP_adjust=DP_adjust)


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
