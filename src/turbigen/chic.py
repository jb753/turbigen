"""Sweeping a characteristic to the stability limit.

:attr:`~turbigen.bconds.OperatingPoint.DP_adjust` moves a fixed machine off
its design point one number at a time. The question it exists to answer is *how
far can it go*, and that cannot be asked one number at a time: you do not know
where the limit is until you have run past it.

So this marches the back pressure up until a point will not converge, then
halves the step and comes back at it from the last good field, until the
surviving bracket is narrow enough to be an answer. A continuation, where
:func:`turbigen.iterate.converge` is a root find --- the two look alike and
are not, which is why they are separate loops rather than one loop with a mode.

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
`step_min`, and the limiting *mass flow* it implies is resolved rather less.
Both belong in the report rather than in a footnote someone finds later.

Nothing here runs anything or knows where a file goes: `run` is injected, as
:func:`turbigen.iterate.converge` takes it, so the march can be driven by an
analytic stand-in with no CFD at all.
"""

import dataclasses
import logging

from turbigen.node import Node

logger = logging.getLogger("turbigen.chic")
"""Sweep-level messages: the points, the bracket, why it stopped."""


class Chic(Node):
    """How to march a characteristic, and how finely to pin its limit."""

    step: float = 0.05
    """Increment in ``DP_adjust`` between points [--].

    Positive is always *towards* the limit --- more throttled for a compressor,
    more expanded for a turbine --- because ``DP_adjust`` already carries that
    meaning through the sign of the design's own pressure change. There is no
    direction to configure, and no machine type named anywhere.
    """

    step_min: float = 0.01
    """Refine until the bracket around the limit is narrower than this [--].

    The resolution of the answer, in the units the answer is given in. Reaching
    it takes about ``log2(step / step_min)`` extra points beyond the march.
    """

    max_points: int = 20
    """Most points to run before giving up.

    A machine that never destabilises would otherwise march until the exit
    pressure went negative, which `bconds.exit_pressure` refuses --- an error
    about a pressure, several minutes in, rather than a budget stated here.
    """

    def __post_init__(self):
        # Checked when the config is read, because none of it needs a design
        # and all of it can make a sweep that cannot terminate.
        for name in ("step", "step_min"):
            if getattr(self, name) <= 0.0:
                raise ValueError(
                    f"chic.{name} must be positive, got {getattr(self, name)}. "
                    f"The sweep marches towards the limit, and DP_adjust "
                    f"already carries which direction that is."
                )

        if self.step_min > self.step:
            raise ValueError(
                f"chic.step_min={self.step_min} is larger than "
                f"chic.step={self.step}, so the first bracket would already be "
                f"finer than asked for and nothing would be refined."
            )

        if self.max_points < 1:
            raise ValueError(
                f"chic.max_points must be at least 1, got {self.max_points}."
            )


@dataclasses.dataclass(frozen=True)
class Point:
    """One run of the sweep, and where it sat."""

    DP_adjust: float
    """Where the operating point was put [--]."""

    result: object
    """What running it achieved."""

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
    from turbigen.bconds import OperatingPoint  # noqa: PLC0415 - avoids a cycle

    point = config.operating_point or OperatingPoint()

    # The throttle comes off, if there was one. A swept point is a pressure by
    # definition -- that is what a characteristic is a set of -- and a
    # controller holding the mass flow would ignore every point asked for here
    # while the table at the end still read as a map. Cleared rather than
    # refused because the design point it was converged at is not lost by
    # clearing it: `converge_design` has already recorded the pressure the
    # throttle settled at as this config's own `DP_adjust`, which is the datum
    # `sweep` departs from.
    return dataclasses.replace(
        config,
        operating_point=dataclasses.replace(
            point, DP_adjust=DP_adjust, mdot_adjust=None
        ),
    )


def sweep(config, run):
    """March `config` off design until it will not converge, and pin the edge.

    Parameters
    ----------
    config : Config
        The machine to sweep, at the geometry it is to keep. Not modified.
    run : callable
        Takes ``(config, index)`` and returns a :class:`~turbigen.result.
        Result`. Injected rather than imported, so this knows nothing of output
        directories or restarts --- and a test can sweep an analytic stand-in.

    Returns
    -------
    points : list of Point
        Every point run, in the order it was run.
    bracket : tuple of float
        ``(last converged, first refused)`` in ``DP_adjust``. The second is
        ``inf`` when nothing refused, which is what running out of
        `max_points` looks like.

    """
    spec = config.chic
    if spec is None:
        raise ValueError("Sweeping a characteristic needs a chic: section.")

    points = []

    # Where the design point sits, which is what the sweep departs from rather
    # than re-running. Usually zero, a design being run at its own pressure;
    # not zero when the design point was throttled, because the controller
    # chose a pressure and `bconds.achieved` wrote it here. Reading it rather
    # than assuming zero is what keeps the first point one `step` from the
    # machine that was actually converged, and keeps the bracket naming
    # pressures that were run.
    datum = config.operating_point.DP_adjust if config.operating_point else 0.0
    if datum:
        logger.info(f"Sweeping from the design point at DP_adjust={datum:.5g}.")

    lo, hi = datum, float("inf")
    step = spec.step

    while len(points) < spec.max_points:
        # March while nothing has refused yet; bisect the bracket once
        # something has. One expression, because they are the same move: go to
        # the midpoint of what is known, which before a refusal is a step
        # beyond the last success.
        DP_adjust = lo + step if hi == float("inf") else 0.5 * (lo + hi)

        point = Point(DP_adjust, run(at(config, DP_adjust), len(points)))
        points.append(point)

        if point.converged:
            lo = DP_adjust
            logger.info(f"DP_adjust={DP_adjust:.5g} converged.")
        else:
            hi = DP_adjust
            logger.info(f"DP_adjust={DP_adjust:.5g} did not converge.")

        if hi - lo <= spec.step_min:
            logger.info(
                f"Limit bracketed to within {spec.step_min:g} after "
                f"{len(points)} point(s)."
            )
            return points, (lo, hi)

    logger.warning(
        f"Stopped after {spec.max_points} point(s) with the limit still "
        f"between {lo:.5g} and {hi:.5g}."
    )
    return points, (lo, hi)


def format_table(points, bracket):
    """Return the characteristic, one line per point.

    Mass flow and pressure ratio come from each point's mixed-out mean line, so
    a point that could not be mixed out shows the run it was rather than a
    fabricated number.
    """
    lines = [
        "Characteristic:",
        f"{'DP_adjust':>10}  {'mdot/kg/s':>10}  {'PR_ts':>8}  ok",
        "-" * 34,
    ]

    for point in points:
        actual = getattr(point.result, "actual", None)
        if actual is None:
            mdot = pr = "--"
        else:
            mdot = f"{float(actual.outlet.mdot):10.4g}"
            pr = f"{float(actual.PR_ts):8.4g}"
        ok = "y" if point.converged else "n"
        lines.append(f"{point.DP_adjust:>10.4g}  {mdot:>10}  {pr:>8}  {ok}")

    lo, hi = bracket
    edge = f"{lo:.5g} to {hi:.5g}" if hi != float("inf") else f"beyond {lo:.5g}"
    lines.append("-" * 34)
    lines.append(
        f"Steady solutions exist up to DP_adjust {edge}. This is where a "
        f"steady solver stops converging, which is not the surge line: real "
        f"stall is unsteady."
    )
    return "\n".join(lines)
