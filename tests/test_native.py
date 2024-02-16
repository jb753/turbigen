import turbigen.solvers.native
import turbigen.grid
import numpy as np
import turbigen.compflow_native as cf

np.random.seed = 0

def dot(a, b, axis=0):
    return np.sum(a * b, axis=axis)

def test_box():
    # Geometry
    L = 0.1
    yoffset = 2.1*L

    nj = 70
    ni = 50
    nk = 40

    Nb = 1
    xv = np.linspace(-L, L, ni)
    yv = np.linspace(-L, L, nj) + yoffset
    zv = -np.linspace(-L, L, nk)

    x, y, z =  np.stack(np.meshgrid(xv, yv, zv, indexing='ij'))

    # Convert Cartesian coordinates to polar
    r = np.sqrt(y**2 + z**2)
    t = np.arctan2(-z, y)

    xrt = np.stack((x, r, t))

    block = turbigen.grid.PerfectBlock.from_coordinates(xrt, 1, [])
    g = turbigen.grid.Grid([block,])
    g.check_coordinates()

    b = g[0]

    # Get polar unit vectors for each cartesian dirn
    tface = turbigen.util.node_to_face3(b.t)
    ex = np.stack(
            (
                np.ones_like(tface[0]),
                np.zeros_like(tface[0]),
                np.zeros_like(tface[0]),
            )
    )

    ek = np.stack(
            (
                np.zeros_like(tface[1]),
                np.cos(tface[1]),
                -np.sin(tface[1]),
            )
    )

    ey = np.stack(
            (
                np.zeros_like(tface[2]),
                np.sin(tface[2]),
                np.cos(tface[2]),
            )
    )


    # Check the areas have correct magnitude and direction
    A = (2*L)**2
    rtol = 2e-3
    assert np.allclose(dot(b.dAi,ex).sum(axis=(1,2)),A,rtol=rtol)
    assert np.allclose(dot(b.dAj,ek).sum(axis=(0,2)),A, rtol=rtol)
    assert np.allclose(dot(b.dAk,ey).sum(axis=(0,1)),A, rtol=rtol)

    # Check the volume
    vol = (2*L)**3
    assert np.isclose(b.vol.sum(),vol, rtol=rtol)

