#!/usr/bin/env python
"""Plot zeta vs istep for any number of convergence histories.

Usage: plot_zeta.py A/conv.cnv B/conv.cnv ...
"""

import sys

import numpy as np
import matplotlib.pyplot as plt

from ember.convergence_history import ConvergenceHistory


def main(argv):
    if len(argv) != 3:
        sys.exit(f"Usage: {argv[0]} <conv.cnv> <conv.cnv>")

    # Cost factor per run: the second run's wall time is scaled by 55.
    cost = (1.0, 55.0)
    cost = (1.0, 1.0)

    zeta_ref = None
    fig, ax = plt.subplots()
    for filename, factor in zip(argv[1:], cost):
        conv = ConvergenceHistory.read_cnv(filename)
        n = conv.i_log + 1
        zeta = conv.zeta[:n]
        i_step = conv.i_step[:n]
        # Normalise by the mean zeta over the last 1000 steps of the first run.
        if zeta_ref is None:
            zeta_ref = zeta[i_step > i_step[-1] - 1000].mean()
        # Cumulative wall time [s], scaled by the run's cost factor.
        t = factor * conv.time[:n] * conv._TIME_SCALE
        zeta_norm = zeta / zeta_ref
        (line,) = ax.plot(t, zeta_norm, label=filename, marker="", linestyle="-")

        # Mark the earliest point after which zeta stays within 1% of its final value.
        zeta_final = zeta[-1]
        within = np.abs(zeta - zeta_final) <= 0.01 * np.abs(zeta_final)
        # First index from which all subsequent points are within 1%.
        i_conv = n - np.argmin(within[::-1]) if not within.all() else 0
        ax.plot(
            t[i_conv],
            zeta_norm[i_conv],
            marker="o",
            linestyle="",
            color=line.get_color(),
        )

    ax.set_xlabel("Cost [s]")
    ax.set_ylabel(r"$\zeta / \zeta_\mathrm{ref}$")
    ax.set_ylim(0.0, 2.0)
    ax.legend()
    plt.show()


if __name__ == "__main__":
    main(sys.argv)
