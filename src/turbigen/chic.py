"""Sweeping a characteristic to the stability limit.

:attr:`~turbigen.bconds.OperatingPoint.DP_adjust` moves a fixed machine off
its design point one number at a time. The question it exists to answer is *how
far can it go*, and that cannot be asked one number at a time: you do not know
where the limit is until you have run past it.

So this marches the back pressure up by a coarse step until a point will not
converge, then comes back to the last good field and marches on from it by a
fine step until a point again will not converge. Every point is a point on the
characteristic, rather than half of them being spent narrowing a bracket from
the far side, and points are finest where the characteristic is near its limit.
A continuation, where :func:`turbigen.iterate.converge` is a root find --- the
two look alike and are not, which is why they are separate loops rather than
one loop with a mode.

**Not a `batch` mode.** A batch emits every point up front and runs them
independently, which is what lets one become a job array, and its module
docstring refuses to be a scheduler. Here feedback is the whole mechanism. The
stable side of a characteristic, needing none, is already a batch today::

    batch:
      values:
        operating_point.DP_adjust: [-0.10, -0.05, 0.0]

so this covers exactly the part that cannot be written that way.

**What it finds is not the surge line.** Real stall is unsteady; a steady
solver either reaches an answer or does not. This locates the lowest pressure
rise at which a steady solution exists *and is reachable from its neighbour*,
which correlates with surge without being it. Near the peak of a characteristic
``dPR/dmdot`` tends to zero as well, so a pressure bracket of a given width
spans a wide range of mass flow: the limiting *pressure* is resolved to
`step_fine`, and the limiting *mass flow* it implies is resolved rather less.
Both belong in the report rather than in a footnote someone finds later.

Nothing here runs anything or knows where a file goes: `run` is injected, as
:func:`turbigen.iterate.converge` takes it, so the march can be driven by an
analytic stand-in with no CFD at all.
"""

import dataclasses
import logging

import numpy as np

from turbigen.node import Node

logger = logging.getLogger("turbigen.chic")
"""Sweep-level messages: the points, the bracket, why it stopped."""


class Chic(Node):
    """How to march a characteristic, and how finely to pin its limit."""

    step_coarse: float = 0.05
    """Increment in ``DP_adjust`` between points until one fails, or in the
    fraction of the throat closed for a `nozzle` sweep [--].

    Positive is always *towards* the limit --- more throttled for a compressor,
    more expanded for a turbine --- because ``DP_adjust`` already carries that
    meaning through the sign of the design's own pressure change. There is no
    direction to configure, and no machine type named anywhere.
    """

    step_fine: float = 0.01
    """The fine step, marched from the last converged point once the coarse
    `step_coarse` has failed, until a point fails again [--].

    The resolution of the answer, in the units the answer is given in. It
    costs up to ``step_coarse / step_fine`` points beyond the coarse march.
    """

    step_choke: float | None = None
    """The coarse step towards choke, in ``DP_adjust``; None for `step_coarse`
    [--].

    Larger than towards stall, as a rule: lowering the back pressure moves
    the mass flow less and less as the machine nears choke, and a converged
    solution past it is still converged, so the leg has a long way to go and
    nothing to refuse on the way.
    """

    step_choke_fine: float | None = None
    """The fine step towards choke, marched from the last point that stood
    once a `step_choke` has been refused; None for `step_fine` [--].

    Coarser than towards stall, as a rule: the choke leg's coarse step is
    larger, and the fine march costs up to ``step_choke / step_choke_fine``
    points beyond it.
    """

    choke_mdot: float = 0.001
    """Least the mass flow must rise over a coarse step towards choke, as a
    fraction of it, for a point to stand [--].

    The machine has choked once lowering the back pressure no longer passes
    more flow, which a steady solver rarely signals by failing to converge.
    So a point that converged but passed too little more than the last point
    that stood is refused as one that diverged is, and the fine march refines
    from the last that stood until a point is refused again. A fine step is
    held to the same slope, the rise scaled by how far it stepped. The first
    point of the leg, with nothing before it to compare, is never judged. 0
    marches on until a point fails to converge.
    """

    max_points: int = 20
    """Most points to run before giving up.

    A machine that never destabilises would otherwise march until the exit
    pressure went negative, which `bconds.exit_pressure` refuses --- an error
    about a pressure, several minutes in, rather than a budget stated here.
    """

    nozzle: bool = False
    """Throttle with a choked exit nozzle, sweeping its throat area, instead
    of with the exit pressure.

    The throat is first sized one-dimensionally to pass the design point's
    mass flow, solved once, and the sweep closes it from there:
    `step_coarse` and `step_fine` are then fractions of that throat area,
    which near choke are fractions of the mass flow too. See :mod:`turbigen.nozzle`.
    """

    stall: bool = True
    """March from the datum towards stall. Off, with `choke` on, for a choke
    leg alone."""

    choke: bool = False
    """March from the datum the other way too, by `step_choke` and
    `step_choke_fine`, towards choke: lowering the back pressure until a point does
    not converge, or the mass flow rises by less than `choke_mdot` over it.
    With a `nozzle`, which
    cannot be opened far enough to choke the machine, the nozzle comes off
    and the open duct is first throttled to the flow the datum throat passed,
    the pressure it settles at being where the leg departs from."""

    speeds: tuple[float, ...] = (1.0,)
    """Shaft speeds to sweep a speedline at, as fractions of the design's
    [--].

    Each speedline departs from the throat sized at the design speed, which
    at another speed is where the design's throttle line crosses that
    speedline, so more than one needs a `nozzle`. The datum at a new speed is
    restarted from the nearest speed already run, behind a soft start.
    """

    def __post_init__(self):
        # Checked when the config is read, because none of it needs a design
        # and all of it can make a sweep that cannot terminate.
        for name in ("step_coarse", "step_fine", "step_choke", "step_choke_fine"):
            if getattr(self, name) is not None and getattr(self, name) <= 0.0:
                raise ValueError(
                    f"chic.{name} must be positive, got {getattr(self, name)}. "
                    f"The sweep marches towards the limit, and DP_adjust "
                    f"already carries which direction that is."
                )

        coarse, fine = self.steps(-1)
        if fine > coarse:
            raise ValueError(
                f"The fine step towards choke, {fine}, is larger than the "
                f"coarse, {coarse}, so the fine march after the first refusal "
                f"would be coarser than the march before it: set "
                f"chic.step_choke and chic.step_choke_fine to suit."
            )
        if not self.choke_mdot >= 0.0:
            raise ValueError(
                f"chic.choke_mdot must not be negative, got {self.choke_mdot}."
            )

        if self.step_fine > self.step_coarse:
            raise ValueError(
                f"chic.step_fine={self.step_fine} is larger than "
                f"chic.step_coarse={self.step_coarse}, so the fine march after "
                f"the first refusal would be coarser than the march before it."
            )

        if not (self.stall or self.choke):
            raise ValueError(
                "chic.stall and chic.choke are both off, so there is no leg to sweep."
            )

        if not self.speeds or not all(speed > 0.0 for speed in self.speeds):
            raise ValueError(
                f"chic.speeds must be positive and not empty, got {self.speeds}."
            )
        if len(set(self.speeds)) != len(self.speeds):
            raise ValueError(f"chic.speeds repeats a speed: {self.speeds}.")
        if tuple(self.speeds) != (1.0,) and not self.nozzle:
            raise ValueError(
                f"chic.speeds={self.speeds} needs chic.nozzle: a speedline "
                f"other than the design's departs from the design's throat."
            )

        if self.max_points < 1:
            raise ValueError(
                f"chic.max_points must be at least 1, got {self.max_points}."
            )

    def steps(self, direction):
        """Return the coarse and fine steps of the leg marching `direction`:
        +1 towards stall, -1 towards choke."""
        if direction > 0:
            return self.step_coarse, self.step_fine
        coarse = self.step_coarse if self.step_choke is None else self.step_choke
        fine = self.step_choke_fine
        if fine is None:
            fine = self.step_fine
        return coarse, fine


@dataclasses.dataclass(frozen=True)
class Point:
    """One run of the sweep, and where it sat."""

    setting: float
    """Where the operating point was put: `DP_adjust`, or for a nozzle the
    fraction its throat was closed by [--]."""

    result: object
    """What running it achieved."""

    ok: bool | None = None
    """Whether the point stood: converged, and passed whatever further test
    the sweep was given. None for converged alone."""

    @property
    def stood(self):
        """Whether the sweep counted this point as a success."""
        return self.converged if self.ok is None else bool(self.ok)

    @property
    def DP_adjust(self):
        """The setting, under the name a pressure sweep gives it."""
        return self.setting

    @property
    def converged(self):
        """Whether a steady solution was reached here."""
        return bool(self.result.converged)


def at(config, DP_adjust):
    """Return `config` with its operating point moved to `DP_adjust`.

    Replaces rather than adjusts what the file already carried: a sweep states
    where each point *is*, so that a member's archived config reads as an
    operating point and not as an offset from one.
    """
    from turbigen.bconds import OperatingPoint

    point = config.operating_point or OperatingPoint()

    # The throttle comes off, if there was one. A swept point is a pressure by
    # definition -- that is what a characteristic is a set of -- and a
    # controller holding the mass flow would ignore every point asked for here
    # while the table at the end still read as a map. Cleared rather than
    # refused because the design point it was converged at is not lost by
    # clearing it: `loop.converge_design` has already recorded the pressure the
    # throttle settled at as this config's own `DP_adjust`, which is the datum
    # `sweep` departs from.
    return dataclasses.replace(
        config,
        operating_point=dataclasses.replace(
            point, DP_adjust=DP_adjust, mdot_adjust=None, nozzle_area=None
        ),
    )


def at_nozzle(config, nozzle_area):
    """Return `config` with its exit choked by a throat of `nozzle_area`.

    The throttle comes off, the nozzle fixing the flow itself.
    """
    from turbigen.bconds import OperatingPoint

    point = config.operating_point or OperatingPoint()
    return dataclasses.replace(
        config,
        operating_point=dataclasses.replace(
            point, nozzle_area=nozzle_area, mdot_adjust=None
        ),
    )


def at_speed(config, speed):
    """Return `config` with every row turning at `speed` of the design's."""
    from turbigen.bconds import OperatingPoint

    point = config.operating_point or OperatingPoint()
    return dataclasses.replace(
        config, operating_point=dataclasses.replace(point, speed=speed)
    )


def setting_name(config, direction=1):
    """Return what a sweep of `config` moves `direction`, as its table heads
    the column."""
    nozzle = config.chic is not None and config.chic.nozzle and direction > 0
    return "closure" if nozzle else "DP_adjust"


def sweep(config, run, direction=1):
    """March `config` off design until it will not converge, and pin the edge;
    towards choke, until it will not converge or pass more flow.

    Parameters
    ----------
    config : Config
        The machine to sweep, at the geometry it is to keep. Not modified.
    run : callable
        Takes ``(config, start)`` and returns a :class:`~turbigen.result.
        Result`. `start` is the index in the points run so far of the one to
        restart from, the last that stood, or None for the datum. Injected
        rather than imported, so this knows nothing of output directories or
        restart files --- and a test can sweep an analytic stand-in.
    direction : int
        +1 towards stall, -1 towards choke. Towards choke is always a back
        pressure, from `config`'s own `DP_adjust`, any nozzle coming off:
        a throat cannot be opened far enough to choke the machine.

    Returns
    -------
    points : list of Point
        Every point run, in the order it was run.
    bracket : tuple of float
        ``(last success, first refusal)`` in ``DP_adjust``, or in the
        fraction of the throat closed for a nozzle sweep towards stall. The
        second is infinite when nothing refused, which is what running out of
        `max_points` looks like. Towards choke, a point that converged but
        passed no more flow is a refusal too.

    """
    spec = config.chic
    if spec is None:
        raise ValueError("Sweeping a characteristic needs a chic: section.")
    points = []

    # Where the design point sits, which is what the sweep departs from rather
    # than re-running: zero, or the pressure a throttled design point's
    # controller chose, which `bconds.achieved` wrote here. Reading it keeps
    # the first point one `step_coarse` from the machine actually converged.
    if spec.nozzle and direction > 0:
        # Towards stall a nozzle sweep instead closes the throat the caller
        # sized, by the fraction `closure`, from zero at the throat that
        # passes the design flow. A throat shut cannot be run, and `place`
        # says so with None.
        area = config.operating_point and config.operating_point.nozzle_area
        if area is None:
            raise ValueError(
                "A nozzle sweep starts from a sized throat: the config's "
                "operating_point.nozzle_area is unset."
            )

        def place(cfg, closure):
            area_now = area * (1.0 - closure)
            return at_nozzle(cfg, area_now) if area_now > 0.0 else None

        datum = 0.0
        logger.info(f"Sweeping the nozzle closed from a throat of {area:.5g}.")
    else:
        place = at
        datum = config.operating_point.DP_adjust if config.operating_point else 0.0
        if datum:
            logger.info(f"Sweeping from the design point at DP_adjust={datum:.5g}.")
    name = setting_name(config, direction)

    # Distances marched from the datum, whichever way
    lo, hi = 0.0, float("inf")
    fine = False
    step_coarse, step_fine = spec.steps(direction)

    # The mass flow the last point that stood passed, towards choke
    mdot_last = None

    def bracket():
        return datum + direction * lo, datum + direction * hi

    while len(points) < spec.max_points:
        # Coarse until something refuses, then fine from the last success. A
        # fine march that reaches the coarse refusal without refusing itself
        # has nothing left to run, the next point being one already refused.
        u = lo + (step_fine if fine else step_coarse)
        if u >= hi - 1e-9 * step_fine:
            break
        setting = datum + direction * u

        config_now = place(config, setting)
        if config_now is None:
            logger.info(f"{name}={setting:.5g} cannot be run.")
        else:
            # From the last point that stood: after a refusal, the field one
            # fine step back, not one that converged past a choke
            start = next(
                (i for i in reversed(range(len(points))) if points[i].stood), None
            )
            result = run(config_now, start)
            stood = bool(result.converged)

            # Towards choke, against the last point that stood, the rise
            # asked of a step in proportion to its length
            mdot = _mdot(result) if direction < 0 and stood else None
            if mdot is not None and mdot_last is not None and spec.choke_mdot:
                rise = mdot / mdot_last - 1.0
                least = spec.choke_mdot * (u - lo) / step_coarse
                if rise < least:
                    stood = False
                    logger.info(
                        f"{name}={setting:.5g} converged but choked: the mass "
                        f"flow rose {rise:+.3%}, against {least:.3%}."
                    )

            points.append(Point(setting, result, stood))
            if stood:
                lo, mdot_last = u, mdot
                logger.info(f"{name}={setting:.5g} stood.")
                continue
            logger.info(f"{name}={setting:.5g} did not stand.")

        hi = u
        if fine or step_fine >= step_coarse:
            break
        fine = True

    else:
        logger.warning(
            f"Stopped after {spec.max_points} point(s) with the limit still "
            "between {:.5g} and {:.5g}.".format(*bracket())
        )
        return points, bracket()

    logger.info(
        "Limit bracketed between {:.5g} and {:.5g} after ".format(*bracket())
        + f"{len(points)} point(s)."
    )
    return points, bracket()


def _mdot(result):
    """Return the mass flow `result` passed at the outlet, or None if it
    carries no flow, as a stand-in need not."""
    if getattr(result, "actual", None) is None:
        return None
    return float(result.actual.outlet.mdot)


def format_table(points, bracket, name="DP_adjust"):
    """Return the characteristic, one line per point, `name` heading the
    column of settings.

    Mass flow and pressure ratio come from each point's mixed-out mean line, so
    a point that could not be mixed out shows the run it was rather than a
    fabricated number.
    """
    lines = [
        "Characteristic:",
        f"{name:>10}  {'mdot/kg/s':>10}  {'PR_ts':>8}  ok",
        "-" * 34,
    ]

    for point in points:
        actual = getattr(point.result, "actual", None)
        if actual is None:
            mdot = pr = "--"
        else:
            mdot = f"{float(actual.outlet.mdot):10.4g}"
            pr = f"{float(actual.PR_ts):8.4g}"
        ok = "y" if point.stood else "n"
        lines.append(f"{point.setting:>10.4g}  {mdot:>10}  {pr:>8}  {ok}")

    lo, hi = bracket
    edge = f"{lo:.5g} to {hi:.5g}" if np.isfinite(hi) else f"beyond {lo:.5g}"
    lines.append("-" * 34)
    if hi >= lo:
        lines.append(
            f"Steady solutions exist up to {name} {edge}. This is where a "
            f"steady solver stops converging, which is not the surge line: "
            f"real stall is unsteady."
        )
    else:
        lines.append(
            f"Towards choke, points stand down to {name} {edge}. Beyond it "
            f"the march stopped converging, or passing more flow."
        )
    return "\n".join(lines)
