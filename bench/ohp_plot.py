"""Debug plots of a passage mesh: the planes where its cells are worst.

Each page is one constant-span plane of a row, drawn blade to blade in the
conformal coordinates the mesher smooths it in, `(m', theta)` with
`dm' = dm / r`, where angles are true angles on the stream surface. Every
block is in its own colour, over two pitches. Cells whose
worst corner is under a limit are filled orange, and cells of non-positive
volume, between this plane and the next, filled black. Each plane gets a page
of the whole passage, then a close-up of its worst cell.

The planes drawn are the hub and casing, mid-span, the top of the blade when
there is a tip gap, and whichever planes hold the worst corner and the most
folded cells.

Usage::

    from ohp_plot import plot_grid
    plot_grid(grid, annulus, spf, "debug.pdf")
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.collections import LineCollection, PolyCollection  # noqa: E402

from turbigen.ohpmesh import ConformalMap, _cell_angles  # noqa: E402

COLOURS = plt.get_cmap("tab10").colors


def _plane(block, j, cmap, shift=0.0):
    """Block `block` at its span index `j` as (2, ni, nk) in (m', theta)."""
    xr = np.stack((np.asarray(block.x)[:, j, :], np.asarray(block.r)[:, j, :]))
    t = np.asarray(block.t)[:, j, :] + shift
    return np.stack((cmap.mp_from_xr(xr), t))


def _offset(block, nj):
    """Global span index of the block's first plane: the blocks of a tip gap
    start at the top of the blade and run to the casing."""
    return nj - block.shape[1]


def _quads(P, mask):
    """Corners of the cells of plane `P` picked by `mask`, for PolyCollection."""
    p = P.transpose(1, 2, 0)
    c = np.stack([p[:-1, :-1], p[1:, :-1], p[1:, 1:], p[:-1, 1:]], axis=2)
    return c[mask]


def _folded(block, j):
    """Cells of non-positive volume between local plane `j` and the next."""
    vol = np.asarray(block.vol)
    return vol[:, min(j, vol.shape[1] - 1), :] <= 0.0


def _lines(P):
    """Grid lines of plane `P` both ways, for LineCollection."""
    p = P.transpose(1, 2, 0)
    return [p[i, :] for i in range(p.shape[0])] + [p[:, k] for k in range(p.shape[1])]


def _pick_planes(grid, cmaps, tip_j):
    """Span indices worth a page, with why."""
    nj = len(cmaps)
    picks = {0: "hub", nj // 2: "mid", nj - 1: "casing"}
    if tip_j is not None:
        picks.setdefault(tip_j, "top of blade")
    worst, worst_j = 90.0, None
    fold, fold_j = 0, None
    for j in range(nj):
        n_fold = 0
        for block in grid:
            jl = j - _offset(block, nj)
            if jl < 0:
                continue
            a = float(_cell_angles(_plane(block, jl, cmaps[j])).min())
            if a < worst:
                worst, worst_j = a, j
            n_fold += int(_folded(block, jl).sum())
        if n_fold > fold:
            fold, fold_j = n_fold, j
    if worst_j is not None:
        picks[worst_j] = f"worst corner {worst:.1f} deg"
    if fold_j is not None:
        picks[fold_j] = (picks.get(fold_j, "") + f" most folded ({fold})").strip()
    return dict(sorted(picks.items()))


def _draw(ax, grid, j, cmap, min_angle):
    """Draw plane `j` of every block of `grid`; return the worst cell centre."""
    nj = grid[0].shape[1]
    worst, centre, label = 90.0, None, None
    for b, block in enumerate(grid):
        jl = j - _offset(block, nj)
        if jl < 0:
            continue
        colour = COLOURS[b % len(COLOURS)]
        for n, shift in enumerate((0.0, block.pitch)):
            P = _plane(block, jl, cmap, shift)
            ax.add_collection(
                LineCollection(
                    _lines(P),
                    colors=[colour],
                    linewidths=0.15,
                    alpha=1.0 if n == 0 else 0.4,
                )
            )
            if n:
                continue
            ang = _cell_angles(P)
            bad = ang < min_angle
            fold = _folded(block, jl)
            ax.add_collection(
                PolyCollection(
                    _quads(P, bad & ~fold),
                    facecolors="tab:orange",
                    edgecolors="none",
                    alpha=0.8,
                )
            )
            ax.add_collection(
                PolyCollection(_quads(P, fold), facecolors="k", edgecolors="none")
            )
            i, k = np.unravel_index(np.argmin(ang), ang.shape)
            if ang[i, k] < worst:
                worst, label = float(ang[i, k]), block.label
                centre = P[:, i : i + 2, k : k + 2].reshape(2, -1).mean(axis=1)
    if centre is not None and worst < min_angle:
        ax.annotate(label, centre, fontsize=5)
    ax.set_aspect("equal")
    ax.autoscale_view()
    return worst, centre


def plot_grid(grid, annulus, spf, path, min_angle=20.0, tip_j=None, title=""):
    """Write debug plots of passage mesh `grid` to PDF `path`.

    `annulus` is the machine's, and `spf` the span fraction of each plane of
    the grid, from which each plane's conformal map is made.
    """
    grid = list(grid)
    cmaps = [ConformalMap(annulus, s, 0.0, annulus.m_max) for s in spf]
    picks = _pick_planes(grid, cmaps, tip_j)
    with PdfPages(path) as pdf:
        for j, why in picks.items():
            label = f"j={j} spf={spf[j]:.3f} ({why})"
            fig, ax = plt.subplots(figsize=(11, 8))
            worst, centre = _draw(ax, grid, j, cmaps[j], min_angle)
            ax.set_title(f"{title}  {label}, worst {worst:.1f} deg", fontsize=9)
            ax.set_xlabel("m' [--]")
            ax.set_ylabel("theta [rad]")
            for b, block in enumerate(grid):
                ax.plot([], [], color=COLOURS[b % len(COLOURS)], label=block.label)
            ax.legend(fontsize=6, loc="upper left", bbox_to_anchor=(1.0, 1.0))
            fig.tight_layout()
            pdf.savefig(fig)

            if centre is not None:
                # Close-up of the worst cell, a tenth of a pitch across
                half = 0.05 * grid[0].pitch
                ax.set_xlim(centre[0] - half, centre[0] + half)
                ax.set_ylim(centre[1] - half, centre[1] + half)
                ax.set_title(f"{title}  {label}, close-up of worst cell", fontsize=9)
                pdf.savefig(fig)
            plt.close(fig)
