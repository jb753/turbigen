"""Check cell areas and volumes are correct."""

import turbigen.grid
import numpy as np
import turbigen.compflow_native as cf
import turbigen.solvers.embsolve

typ = np.float32


def to_fort(x):
    """Convert an array to Fortran."""
    x = np.asfortranarray(x.copy()).astype(typ)
    return x


def test_box():

    # Geometry
    L = 0.1
    yoffset = 40.0 * L

    ni = 5
    nj = 7
    nk = 9

    Nb = 1
    xv = np.linspace(-L, L, ni)
    yv = np.linspace(-L, L, nj) + yoffset
    zv = -np.linspace(-L, L, nk)

    x, y, z = np.stack(np.meshgrid(xv, yv, zv, indexing="ij"))

    # Convert Cartesian coordinates to polar
    r = np.sqrt(y**2 + z**2)
    t = np.arctan2(-z, y)

    xrt1 = np.stack((x, r, t))
    xrt2 = xrt1.copy()
    xrt2[0] += np.ptp(xrt1[0])

    xrt2 = np.flip(xrt2, axis=(3, 1))

    xrt12 = [xrt1, xrt2]

    patch = [
        [
            turbigen.grid.PeriodicPatch(i=-1),
        ],
        [
            turbigen.grid.PeriodicPatch(i=-1),
        ],
    ]

    blocks = [
        turbigen.grid.PerfectBlock.from_coordinates(xrti, 1, pi)
        for xrti, pi in zip(xrt12, patch)
    ]
    g = turbigen.grid.Grid(blocks)
    g.check_coordinates()
    g.match_patches()

    bid, ijk, ijkf, d, nxbid, nxijk, nxijkf, nxd = (
        turbigen.solvers.embsolve.get_periodic_data(g[0].patches[0])
    )

    xf1 = g[0].x_face[0]
    xf2 = g[1].x_face[0]

    rf1 = g[0].r_face[0]
    rf2 = g[1].r_face[0]

    tf1 = g[0].t_face[0]
    tf2 = g[1].t_face[0]

    xrtf1 = to_fort(np.stack((xf1, rf1, tf1), axis=-1))
    xrtf2 = to_fort(np.stack((xf2, rf2, tf2), axis=-1))

    xrt1u = turbigen.solvers.embsolve.embsolve.get_by_ijk(xrtf1, ijkf)
    xrt2u = turbigen.solvers.embsolve.embsolve.get_by_ijk(xrtf2, nxijkf)

    assert np.allclose(xrt1u, xrt2u)


if __name__ == "__main__":
    test_box()
