"""Check cell areas and volumes are correct."""
import turbigen.grid
import numpy as np
import turbigen.compflow_native as cf
from turbigen.solvers.native import get_periodic_data, embsolve, to_fort

def test_box():

    # Geometry
    L = 0.1
    yoffset = 40.0*L

    nj = 3
    ni = 3
    nk = 3

    Nb = 1
    xv = np.linspace(-L, L, ni)
    yv = np.linspace(-L, L, nj) + yoffset
    zv = -np.linspace(-L, L, nk)

    x, y, z =  np.stack(np.meshgrid(xv, yv, zv, indexing='ij'))

    # Convert Cartesian coordinates to polar
    r = np.sqrt(y**2 + z**2)
    t = np.arctan2(-z, y)

    xrt1 = np.stack((x, r, t))
    xrt2 = xrt1.copy()
    xrt2[0] += xrt1[0].ptp()

    xrt2 = np.flip(xrt2,axis=(3,1))

    xrt12 = [xrt1, xrt2]

    patch = [
        [turbigen.grid.PeriodicPatch(i=-1),],
        [turbigen.grid.PeriodicPatch(i=-1),]
    ]

    blocks = [turbigen.grid.PerfectBlock.from_coordinates(xrti, 1, pi) for xrti, pi in zip(xrt12, patch)]
    g = turbigen.grid.Grid(blocks)
    g.check_coordinates()
    g.match_patches()

    bid, ijk, ijkf, d, nxbid, nxijk, nxijkf, nxd = get_periodic_data(g[0].patches[0])

    xf1 = g[0].x_face[0]
    xf2 = g[1].x_face[0]

    rf1 = g[0].r_face[0]
    rf2 = g[1].r_face[0]

    tf1 = g[0].t_face[0]
    tf2 = g[1].t_face[0]

    xrtf1 = np.stack((xf1, rf1, tf1))
    xrtf2 = np.stack((xf2, rf2, tf2))

    xrt1_out = embsolve.get_by_ijk(to_fort(xrtf1), ijkf)
    xrt2_out = embsolve.get_by_ijk(to_fort(xrtf2), nxijkf)


    assert np.allclose(xrt1_out, xrt2_out)

if __name__=='__main__':

    test_box()
