from turbigen.solvers import ts3
import turbigen.util
import numpy as np

fname = "combustor_runs/0058/output_probe_61_5.dat"
F, fs = ts3.read_probe_dat(fname)

import matplotlib.pyplot as plt

print(fs)

_, nj, nk, nt = F.shape
P1 = F[0, nj // 2, nk // 2, :].P
f, Pfft = turbigen.util.amplitude_spectrum(P1, fs)
fig, ax = plt.subplots()
ax.plot(f, np.abs(Pfft))
ax.set_xlim((0, 5000))
plt.show()
