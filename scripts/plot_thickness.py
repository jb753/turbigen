"""Compare two thickness distributions from qthick parameters."""

import numpy as np
import matplotlib.pyplot as plt
from turbigen.thickness import Taylor

# Setup the figure
fig, ax = plt.subplots(layout="constrained")
ax.set_ylim(0, 0.05)
ax.set_xlim(0, 1.0)
ax.set_xlabel(r"Meridional Distance, $m/c_m$")
ax.set_ylabel(r"Thickness, $\tau/c_m$")

# Rotor blade thickness parameters for the two designs
qthick = {
    "Baseline": [0.001, 0.02, 0.6, 0.01, 0.008, 0.15],
    "Final": [0.01, 0.04, 0.6, 0.01, 0.032, 0.15],
}

# Loop through the designs and plot
x = np.linspace(0, 1, 200)
for k, v in qthick.items():
    thick = Taylor(v)
    ax.plot(x, thick.t(x), label=k)

ax.legend()
plt.show()
