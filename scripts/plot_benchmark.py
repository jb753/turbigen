"""Do a scaling plot of the benchmark results."""

import numpy as np
import matplotlib.pyplot as plt

# Read the data csv
dat = np.loadtxt("plots/bench.dat", delimiter=",")
isort = np.argsort(dat[:, 0])
dat = dat[isort]

size, err, tpnps, cost, rel_cost, rel_time = dat.T

# Parallel speedup
fig, ax = plt.subplots(layout="constrained")
ax.plot(size, size, "k--")
ax.plot(size, tpnps[0] / tpnps, "x-")
ax.set_xlabel("Number of Processors")
ax.set_ylabel("Relative Speedup")
ax.set_ylim(1, 8)
ax.set_xlim(1, 8)
ax.set_yticks(np.arange(0, 10, 2))
plt.savefig("plots/bench_speedup.pdf")
plt.close()

# Time relative to TS3
fig, ax = plt.subplots(layout="constrained")
ax.plot(size, rel_time, "x-")
ax.set_xlabel("Number of Processors")
ax.set_ylabel("Time/Time TS3")
ax.set_xlim(1, 8)
ax.set_ylim(0, 80)
plt.savefig("plots/bench_time.pdf")
plt.close()

# Cost relative to TS3
fig, ax = plt.subplots(layout="constrained")
ax.plot(size, rel_cost, "x-")
ax.set_xlabel("Number of Processors")
ax.set_ylabel("Cost/Cost TS3")
ax.set_xlim(1, 8)
ax.set_ylim(0, 60)
plt.savefig("plots/bench_cost_1.pdf")
plt.close()

# Cost relative to TS3
fig, ax = plt.subplots(layout="constrained")
ax.plot(size, rel_cost, "x-")
ax.set_xlabel("Number of Processors")
ax.set_ylabel("Cost/Cost TS3")
ax.set_xlim(1, 8)
ax.set_ylim(0, 2)
ax.set_yticks(np.arange(0, 2.5, 0.5))
plt.savefig("plots/bench_cost_2.pdf")
plt.close()
