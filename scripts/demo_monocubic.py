"""Interactive demo for _monocubic thickness distribution."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from turbigen.thickness import _monocubic

m = np.linspace(0.0, 1.0, 300)

fig, ax = plt.subplots(layout="constrained")
fig.subplots_adjust(bottom=0.4)
ax.set_xlim(0, 1)
ax.set_ylim(0, 0.08)
ax.set_aspect("equal")
ax.set_xlabel(r"Meridional Distance, $m/c_m$")
ax.set_ylabel(r"Thickness, $\tau/c_m$")

defaults = dict(R_LE=0.003, m_tmax=0.4, t_max=0.05, t_TE=0.01, wedge=8.0)
(line,) = ax.plot([], [])

sliders = {}
for i, (name, val) in enumerate(defaults.items()):
    slax = fig.add_axes([0.15, 0.28 - i * 0.055, 0.7, 0.03])
    lims = {
        "R_LE": (0.0005, 0.02),
        "m_tmax": (0.2, 0.8),
        "t_max": (0.01, 0.1),
        "t_TE": (0.0, 0.03),
        "wedge": (0.0, 20.0),
    }
    sliders[name] = Slider(slax, name, *lims[name], valinit=val)


def update(_):
    thick = _monocubic(**{k: s.val for k, s in sliders.items()})
    line.set_data(m, thick(m))
    fig.canvas.draw_idle()


for s in sliders.values():
    s.on_changed(update)

update(None)
plt.show()
