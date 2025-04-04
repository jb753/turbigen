import matplotlib.pyplot as plt
import numpy as np
import clusterfunc.util


def plot(x):
    n = np.linspace(0, len(x) - 1, len(x))
    fig, ax = plt.subplots()
    ax.plot(n, x, "k-x")
    ax.plot(np.zeros_like(x), x, "k-x")
    ax.set_xlim(n[0] - 1, n[-1])
    ax.set_ylim(x.min(), x.max())
    ax.set_xlabel("Index")
    ax.set_ylabel("Coordinate")
    plt.tight_layout()


def plot_ER(x):
    ER = clusterfunc.util.ER(x)
    n = np.linspace(0, len(ER) - 1, len(ER))
    fig, ax = plt.subplots()
    ax.plot(n, ER, "k-x")
    # ax.plot(np.zeros_like(x), x, 'k-x')
    ax.set_xlim(n[0] - 1, n[-1])
    # ax.set_ylim(x.min(), x.max())
    ax.set_xlabel("Index")
    ax.set_ylabel("Coordinate")
    plt.tight_layout()


# import clusterfunc.single
# import clusterfunc.symmetric

# Dmin = 1e-1
# Dmax = 2e-1
# ERmax = 1.2
# # x = clusterfunc.single.unit_free_N(Dmin, Dmax, ERmax, mult=1)
# x = clusterfunc.symmetric.unit_free_N(Dmin, Dmax, ERmax, 1)
# plot(x)
# # plot_ER(x)
# plt.show()
