import matplotlib.pyplot as plt
import numpy as np

x1 = np.load("grad.npy")
x2 = np.load("grad_2.npy")
x3 = np.load("grad_3.npy")

sp1 = [0.1, 0.2, 0.4, 0.8]
sp2 = [0.01, 0.02, 0.05]
sp3 = [0.001, 0.002, 0.005]

fig, ax = plt.subplots()
ax.plot(sp1, x1, "-x")
ax.set_prop_cycle(None)
ax.plot(sp2, x2, "-x")
ax.set_prop_cycle(None)
ax.plot(sp3, x3, "-x")
ax.set_xlim([0.0, 0.25])
plt.savefig("grad.pdf")
