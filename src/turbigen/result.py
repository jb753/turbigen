"""What running a machine produces."""

import dataclasses

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
    #
    # Nothing derived is stored either. `eta_tt`, `PR_tt` and the whole
    # `backward()` dict are recomputed from the mean line, so an archived file
    # cannot hold a value that no longer matches the definition that made it.
    #

    @classmethod
    def from_dict(cls, data, fluid, machine=None):
        """Rebuild a result from `data`, an equation of state and a machine."""
        actual = data.get("actual")
        return cls(
            machine=machine,
            actual=MeanLine.from_dict(actual, fluid) if actual else None,
            converged=bool(data.get("converged", False)),
            error=dict(data.get("error", {})),
        )

    def to_dict(self):
        """Return everything worth keeping from a run."""
        data = {"converged": bool(self.converged)}
        if self.actual is not None:
            data["actual"] = self.actual.to_dict()
        if self.error:
            # Floats, not numpy scalars, so the file reads as YAML rather than
            # as a pickle of array types.
            data["error"] = {name: float(value) for name, value in self.error.items()}
        return data
