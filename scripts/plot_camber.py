"""Compare two camber distributions from qcamber parameters."""

import numpy as np
import matplotlib.pyplot as plt
from turbigen.camber import Brind

# Setup the figure
fig, ax = plt.subplots(layout="constrained")
# ax.set_ylim(0, 1.)
ax.set_xlim(0, 1.0)
ax.set_xlabel(r"Meridional Distance, $m/c_m$")
ax.set_ylabel(r"Normalised Camber, $\hat{\chi}$")

tanchiLE = np.tan(np.radians(0.))
tanchiTE = np.tan(np.radians(60.))

qcamber = [tanchiLE, tanchiTE, 1, 1, 0]

# Loop through the designs and plot
x = np.linspace(0, 1, 200)
cam = Brind(qcamber)
ax.plot(x, cam.chi(x))
plt.show()
