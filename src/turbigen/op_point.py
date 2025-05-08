"""Configuration for changing machine operating point."""

import dataclasses
import numpy as np


@dataclasses.dataclass
class OperatingPoint:
    mdot_adjust: float = 0.0
    """Mass flow rate change relative to design value."""

    Omega_adjust: float = 0.0
    """Shaft angular velocity change relative to design value."""

    PR_ts_adjust: float = 0.0
    """Total to static pressure ratio change relative to design."""

    pid: tuple = (0.0, 0.0, 0.0)
    """PID controller parameters for mass flow rate setpoint."""

    def __post_init__(self):
        # Cannot set mdot and PR_ts at the same time
        if self.mdot_adjust and self.PR_ts_adjust:
            raise ValueError("Cannot set off-design mdot and PR_ts at the same time.")

        if self.mdot_adjust and not np.any(self.pid):
            raise ValueError("PID parameters must be set if mdot is adjusted.")
