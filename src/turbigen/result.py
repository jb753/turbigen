"""What running a machine produces."""

import dataclasses

import numpy as np

from turbigen.machine import Machine
from turbigen.meanline import MeanLine


@dataclasses.dataclass(frozen=True)
class Result:
    """The outcome of designing, and running, a machine.

    Only the flow has a nominal and an actual. There is no actual annulus:
    the annulus that was designed is the annulus, and a CFD solution does not
    produce a different one. The same holds for blades --- the deviation
    iterator changes a blade, but that yields a new *design* for the next
    iteration, not an actual version of the current one. So geometry is held
    once, on the machine, and the mean line appears twice.
    """

    machine: Machine | None = None
    """Geometry, and the flow it was designed for.

    Optional so that a result can be read back without designing, which a
    script scraping many runs may not want to pay for.
    """

    actual: MeanLine | None = None
    """Mean line mixed out from the CFD solution, once there is one."""

    Ds_mix: object | None = None
    """Specific entropy rise from mixing each cut to uniformity [J/kg/K].

    The loss the reduction to :attr:`actual` introduces, one value per station
    in the same ``(2, n_row)`` shape --- the mixed-out entropy less the
    mass-averaged entropy of the cut. Kept beside ``actual`` because it is
    measured in the same pass and is gone once the grid is, exactly like a
    :class:`~turbigen.metric.Metric`. An ``ndarray`` in memory, nested lists
    once serialised.
    """

    grid: object | None = None
    """Computational grid, once the machine has been meshed."""

    converged: bool = False
    """Whether the run met its convergence criteria."""

    history: object | None = None
    """Convergence history of the march, once there has been one.

    In memory only, like the grid: it is one record every `n_step_log` steps of
    residuals and station quantities, which is far more than a result file
    should carry, and it is worthless without the run it came from. The
    convergence plot reads it here rather than being handed it separately, so
    that a post-processor still takes only a config and a result.
    """

    operating_point: object | None = None
    """Where the machine turned out to be running, when that is not where it
    was asked to run.

    Only a throttled exit produces one: it is handed a mass flow and finds the
    pressure that passes it, so the pressure is an outcome of the run in the
    way every other boundary condition is an input to it. `None` otherwise,
    which is every unthrottled run and every throttled one whose controller
    had not settled.

    Carried here rather than returned alongside, so that the one thing a solve
    produces stays one object --- and read by `converge_design`, which has to
    hand the operating point a design converged at to whatever sweeps it next.
    Not written to the result file: it belongs in the `operating_point:`
    section of the resolved config, where it can be run again, and it is
    written there instead.
    """

    error: dict = dataclasses.field(default_factory=dict)
    """What each configured iterator measured, by name.

    Every run records these, iterating or not: the exit angle a row achieved
    and the incidence its leading edge saw are observations of the flow, worth
    keeping for their own sake, and a design that iterates is only a design
    that acts on them. Stored, unlike the grid and the history, because a
    handful of scalars describing the answer is exactly what a result file is
    for --- and because a run archived today is a sample for whatever fits
    these errors later.
    """

    metrics: dict = dataclasses.field(default_factory=dict)
    """What each configured :class:`~turbigen.metric.Metric` measured, by name.

    Kept for the same reason `error` is, and written beside it under
    ``result: metrics:``. Nothing acts on these --- a metric is a passive
    observation of the field, a surface integral or a loss breakdown that would
    otherwise be lost when the grid goes out of scope. Values are numbers or
    nested lists of them.
    """

    @property
    def nominal(self) -> MeanLine:
        """The mean line as designed."""
        if self.machine is None:
            raise ValueError(
                "This result carries no machine, so there is no nominal mean "
                "line to compare against. Read it with design=True, or design "
                "the config yourself."
            )
        return self.machine.mean_line

    #
    # SERIALISATION
    #
    # A result is not a config node and never appears among a config's own
    # keys. It is written beside them, under `result:`, because a run's answer
    # has to be readable back without repeating the CFD that produced it.
    #
    # Only the mean line and the verdict are stored. The machine is
    # reproducible from the config, and the grid is far too large -- carrying a
    # decimated one instead and re-deriving `actual` from it was considered and
    # rejected: it costs a re-mesh and an interpolation, and decimation error
    # lands hardest at the wall, which is exactly where mixed-out efficiency
    # comes from. The convergence history is left out on the same grounds: it
    # describes how the answer was reached, not what it is.
    #
    # `to_dict` names what it writes rather than dumping the fields, so a field
    # added above stays out of the file until someone decides it belongs there.
    # `Ds_mix`, `error` and `metrics` are the ones that do: all handfuls of
    # numbers measured from the field, kept because that is what a result file
    # is for.
    #
    # Nothing derived is stored either. `eta_tt`, `PR_tt` and the whole
    # `backward()` dict are recomputed from the mean line, so an archived file
    # cannot hold a value that no longer matches the definition that made it.
    #

    @classmethod
    def from_dict(cls, data, fluid, machine=None):
        """Rebuild a result from `data`, an equation of state and a machine."""
        actual = data.get("actual")
        Ds_mix = data.get("Ds_mix")
        return cls(
            machine=machine,
            actual=MeanLine.from_dict(actual, fluid) if actual else None,
            Ds_mix=np.asarray(Ds_mix, dtype=float) if Ds_mix is not None else None,
            converged=bool(data.get("converged", False)),
            error=dict(data.get("error", {})),
            metrics=dict(data.get("metrics", {})),
        )

    def to_dict(self):
        """Return everything worth keeping from a run."""
        data = {"converged": bool(self.converged)}
        if self.actual is not None:
            data["actual"] = self.actual.to_dict()
        if self.Ds_mix is not None:
            data["Ds_mix"] = np.asarray(self.Ds_mix, dtype=float).tolist()
        if self.error:
            # Floats, not numpy scalars, so the file reads as YAML rather than
            # as a pickle of array types.
            data["error"] = {name: float(value) for name, value in self.error.items()}
        if self.metrics:
            # Same, but a metric may measure a profile, so coerce recursively.
            data["metrics"] = {
                name: np.asarray(value, dtype=float).tolist()
                for name, value in self.metrics.items()
            }
        return data
