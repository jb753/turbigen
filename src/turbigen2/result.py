"""What running a machine produces."""

import dataclasses

from turbigen2.machine import Machine
from turbigen2.meanline import MeanLine


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
    # comes from.
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
        )

    def to_dict(self):
        """Return everything worth keeping from a run."""
        data = {"converged": bool(self.converged)}
        if self.actual is not None:
            data["actual"] = self.actual.to_dict()
        return data
