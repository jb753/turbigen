"""Save plots of annulus lines."""
import os
import turbigen.util
import matplotlib.pyplot as plt
import numpy as np

logger = turbigen.util.make_logger()


def post(grid, machine, meanline, postdir):

    logger.info("Plotting annulus lines")

    m = np.linspace(0.0, 1.0, 100)

    fig, ax = plt.subplots()
    ax.set_xlabel("Axial Coordinate, $x$")
    ax.set_ylabel("Radial Coordinate, $r$")
    plt.tight_layout(pad=0.1)
    ann = machine.ann
    ax.plot(*ann.hub.xr(m), "k-")
    ax.plot(*ann.cas.xr(m), "k-")
    ax.plot(*ann.hub.xr(ann.hub.mctrl), "ro", fillstyle="none", ms=10)
    ax.plot(*ann.cas.xr(ann.cas.mctrl), "ro", fillstyle="none", ms=10)
    x = ann.hub.xr((0.0, 1.0))[0]
    ax.plot(x, np.zeros_like(x), "k-.")
    ax.axis("equal")
    pltname = os.path.join(postdir, "annulus.pdf")
    plt.savefig(pltname)
