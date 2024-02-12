import turbigen.solvers.native
import turbigen.grid
import numpy as np
import turbigen.compflow_native as cf

def test_cell_to_node():
    x = np.ones((8,3,5,6,7))
    xn = turbigen.solvers.native.cell_to_node(x)
    assert np.allclose(xn,1.)

def test_node_to_face():
    x = np.ones((8,3,5,6,7))
    xf = turbigen.solvers.native.node_to_face(x)
    for xfi in xf:
        assert np.allclose(xfi,1.)

def test_areas_volumes():
    # Geometry
    h = 0.1
    htr = 0.8
    rm = 0.5 * h * (1.0 + htr) / (1.0 - htr)
    rh = rm - 0.5 * h
    rt = rm + 0.5 * h

    nj = 7
    ni = 5
    nk = 3
    pitch = h/nj*(nk-1)
    Nb = int(2.0 * np.pi * rm / pitch)
    tpitch = 2.0 * np.pi / float(Nb)
    tv = np.linspace(-tpitch/2., tpitch/2., nk)
    xv = np.linspace(0., h, ni)
    rv = np.linspace(rh, rt, nj)
    xrt = np.stack(np.meshgrid(xv, rv, tv, indexing='ij'))

    patches = []

    block = turbigen.grid.PerfectBlock.from_coordinates(xrt, Nb, patches)
    g = turbigen.grid.Grid([block,])
    g.match_patches()
    g.check_coordinates()

    rtol = 1e-9

    # Check the grid is uniform
    b = g[0]

    dt = np.diff(b.t,axis=2)
    assert dt.ptp()/dt.mean() < rtol
    dt = dt.mean()

    dx = np.diff(b.x,axis=0)
    assert dx.ptp()/dx.mean() < rtol
    dx = dx.mean()

    dr = np.diff(b.r,axis=1)
    assert dr.ptp()/dr.mean() < rtol
    dr = dr.mean()

    # Check the total areas in i,j,k dirns
    Ai = (rt*2-rh*2)/2.*tpitch
    Aj = rv * tpitch * h
    Ak = h * h

    # Check the vectors are in one dirn only
    assert (b.dAi[(1,2),...] < rtol * b.dAi[0,...]).all()
    assert (b.dAj[(0,2),...] < rtol * b.dAj[1,...]).all()
    assert (b.dAk[(0,1),...] < rtol * b.dAk[2,...]).all()

    # Check the totals sum correctly
    assert np.allclose(Ai, b.dAi[0,:,:,:].sum(axis=(1,2)))
    assert np.allclose(Aj, b.dAj[1,:,:,:].sum(axis=(0,2)))
    assert np.allclose(Ak, b.dAk[2,:,:,:].sum(axis=(0,1)))

    # Check the volume sums correctly
    vol = Ai*h
    assert np.isclose(vol, b.vol.sum())

def not_test_box():
    # Geometry
    L = 0.1
    yoffset = 8*L

    nj = 7
    ni = 5
    nk = 3

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

    # Check the areas
    A = (2*L)**2
    print(A, turbigen.util.vecnorm(b.dAi).sum(axis=(1,2)))

    # # Check the volume
    # vol = (2*L)**3
    # print(b.vol.shape)
    # print(b.vol.sum(), vol)
    # assert np.isclose(b.vol.sum(),vol)


