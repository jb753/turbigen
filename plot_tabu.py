"""Plot a tabu search."""

import numpy as np
from turbigen import tabu
import matplotlib.pyplot as plt


base_dir = "phi_12_psi_26"
mem_file = base_dir + "/mem.json"

x0 = np.atleast_2d(
    [-10.0, 0.0, 8.0, 1.0, 0.07, 0.25, 0.2, 18.0, 0.12, 0.25, 0.1, 10.0]
)
dx = np.atleast_2d(
    [[2.0, 2.0, 2.0, 2.0, 0.04, 0.1, 0.1, 4.0, 0.04, 0.1, 0.1, 4.0]]
)
tol = dx / 4.0
ts = tabu.TabuSearch(None, None, x0.shape[1], 6, tol, j_obj=(0,))

ts.load_memories(mem_file)

ts.plot("tabu.pdf")
quit()

fig, ax = plt.subplots()
ax.plot(ts.mem_long.Y[:, 0])
ax.plot(ts.mem_med.Y[:, 0])

plt.savefig("hist.pdf")

quit()

x0 = ts.mem_med.get(0)[0]
