import ember.grid
import matplotlib.pyplot as plt
import numpy as np

# Load the solution file into a Grid object
import sys

fname = sys.argv[1]

g = ember.grid.Grid.read_emb(fname)

# Contour the time-averaged CFL on the mid-span plane, one figure per equation.
# cfl_avg lives in block.working.cfl_avg, cell-centred (ni-1, nj-1, nk-1, 5), and
# is retained on the grid after run.finalise when SolverConfig.avg_cfl is set, so
# slice the geometry to cell centres exactly as the residual contour does.
labels = [
    "Mass CFL",
    "$x$-momentum CFL",
    "$r$-momentum CFL",
    "$r\\theta$-momentum CFL",
    "Energy CFL",
]
fnames = ["mass", "xmom", "rmom", "rtmom", "energy"]
lev = np.arange(0, 4.25, 0.25)
# Truncate cubehelix so the brightest top colors are not used.
import matplotlib as mpl

cmap = mpl.colors.ListedColormap(plt.get_cmap("cubehelix")(np.linspace(0, 0.925, 256)))
norm = mpl.colors.BoundaryNorm(lev, cmap.N)
for ieq, lab in enumerate(labels):
    fig, ax = plt.subplots(layout="constrained")
    for b in g:
        S = b[:, b.nj // 2, :]
        # Cell-centred coordinates: average the four corner nodes of each cell.
        cc = lambda a: 0.25 * (a[1:, 1:] + a[:-1, 1:] + a[1:, :-1] + a[:-1, :-1])
        x_cell = cc(S.x)
        rt_cell = cc(S.rt)
        r_cell = cc(S.r)
        cfl = b.working.cfl_avg[:, b.nj // 2, :, ieq]
        cfl = np.clip(cfl, 0.0, 4.0)
        for dt in (0.0, S.pitch * r_cell):
            cm = ax.pcolormesh(
                x_cell, rt_cell + dt, cfl, cmap=cmap, norm=norm, shading="nearest"
            )
            cm.set_rasterized(True)
    cb = plt.colorbar(cm, label=lab, ticks=np.arange(0, 5))
    cb.ax.minorticks_off()
    ax.axis("equal")
    ax.set_ylim([-0.1, 0.2])
    ax.set_xlim([-0.08, 0.3])
    ax.axis("off")
    fig.savefig(f"cfl_{fnames[ieq]}.pdf", dpi=800)
    plt.close(fig)
