"""Interactive demo for Camber distribution."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from turbigen.new_geometry import Camber

ORDER = 2

m = np.linspace(0.0, 1.0, 300)

fig, ax = plt.subplots(layout="constrained")
fig.subplots_adjust(bottom=0.05 + (ORDER - 1) * 0.055)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect("equal")
ax.set_xlabel(r"Meridional Distance, $m/c_m$")
ax.set_ylabel(r"Normalised angle tangent")
ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="linear")
(line,) = ax.plot([], [], label="camber")
ax.legend()

sliders = []
for i in range(ORDER - 1):
    slax = fig.add_axes([0.15, 0.05 + i * 0.055, 0.7, 0.03])
    sliders.append(Slider(slax, f"c{i + 1}", -1.0, 1.0, valinit=0.0))


def update(_):
    coeff = np.array([s.val for s in sliders])
    camber = Camber.from_design_vector(coeff)
    line.set_data(m, camber.evaluate(m))
    fig.canvas.draw_idle()


for s in sliders:
    s.on_changed(update)

update(None)
plt.show()
