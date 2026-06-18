#!/usr/bin/env python
"""Plot zeta vs step/time/cost for any number of convergence histories.

Usage: plot_zeta.py [--xaxis {step,time,cost}] [--cost F ...] A/conv.cnv B/conv.cnv ...
"""

import argparse

import numpy as np
import matplotlib.pyplot as plt

from ember.convergence_history import ConvergenceHistory


def converged_index(zeta, tol=0.01):
    """First index i for which zeta[i:] varies by at most tol about its mean.

    For each candidate i, the tail zeta[i:] is "converged" when every sample
    lies within tol (fractional) of the tail mean, i.e.
    max(|zeta[i:] - mean|) <= tol * |mean|. Returns the earliest such i, or the
    last index if no tail qualifies.
    """
    n = len(zeta)
    for i in range(n):
        tail = zeta[i:]
        mean = tail.mean()
        if np.abs(tail - mean).max() <= tol * np.abs(mean):
            return i
    return n - 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", help="convergence history .cnv files")
    parser.add_argument(
        "--xaxis",
        choices=("step", "time", "cost"),
        default="step",
        help="x-axis quantity: step (no scaling, default), wall time, "
        "or cost-scaled wall time",
    )
    parser.add_argument(
        "--cost",
        type=float,
        nargs="+",
        metavar="F",
        help="per-file cost factors applied when --xaxis cost (default 1.0 each)",
    )
    args = parser.parse_args(argv)

    if args.cost is None:
        cost = [1.0] * len(args.files)
    elif len(args.cost) != len(args.files):
        parser.error(
            f"--cost expects {len(args.files)} factors, got {len(args.cost)}"
        )
    else:
        cost = args.cost

    xlabel = {"step": "Step", "time": "Time [s]", "cost": "Cost [s]"}[args.xaxis]

    zeta_ref = None
    fig, ax = plt.subplots()
    for filename, factor in zip(args.files, cost):
        conv = ConvergenceHistory.read_cnv(filename)
        n = conv.i_log + 1
        zeta = conv.zeta[:n]
        i_step = conv.i_step[:n]
        # Normalise by the mean zeta over the last 1000 steps of the first run.
        if zeta_ref is None:
            zeta_ref = zeta[i_step > i_step[-1] - 1000].mean()
        # Select the x-axis quantity.
        if args.xaxis == "step":
            x = i_step
        else:
            # Cumulative wall time [s], scaled by the cost factor when requested.
            x = conv.time[:n] * conv._TIME_SCALE
            if args.xaxis == "cost":
                x = factor * x
        zeta_norm = zeta / zeta_ref
        (line,) = ax.plot(x, zeta_norm, label=filename, marker="", linestyle="-")

        # Mark the earliest index i for which all of zeta[i:] lie within 1% of
        # their own mean.
        i_conv = converged_index(zeta, tol=0.01)
        print(f"{filename}: converged at step {i_step[i_conv]}")
        ax.plot(
            x[i_conv],
            zeta_norm[i_conv],
            marker="o",
            linestyle="",
            color=line.get_color(),
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$\zeta / \zeta_\mathrm{ref}$")
    ax.set_ylim(0.0, 2.0)
    ax.legend()
    plt.show()


if __name__ == "__main__":
    main()
