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

    machine: Machine
    """Geometry, and the flow it was designed for."""

    actual: MeanLine | None = None
    """Mean line mixed out from the CFD solution, once there is one."""

    grid: object | None = None
    """Computational grid, once the machine has been meshed."""

    converged: bool = False
    """Whether the run met its convergence criteria."""

    @property
    def nominal(self) -> MeanLine:
        """The mean line as designed."""
        return self.machine.mean_line
