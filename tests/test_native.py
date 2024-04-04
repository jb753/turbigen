# import turbigen.solvers.native
import turbigen.grid
import numpy as np
import turbigen.compflow_native as cf

np.random.seed = 0

def dot(a, b, axis=0):
    return np.sum(a * b, axis=axis)

def not_test_box():
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
    print('beans')
    assert np.allclose(dot(b.dAi,ex).sum(axis=(1,2)),A,rtol=rtol)
    # assert np.allclose(dot(b.dAj,ek).sum(axis=(0,2)),A, rtol=rtol)
    # assert np.allclose(dot(b.dAk,ey).sum(axis=(0,1)),A, rtol=rtol)

    # Check the volume
    vol = (2*L)**3
    assert np.isclose(b.vol.sum(),vol, rtol=rtol)


def test_cylinder():
    # Geometry
    L = 0.1
    rm = 1.
    dr = 0.1

    r1 = rm-dr/2.
    r2 = rm+dr/2.

    nj = 70
    ni = 50
    nk = 40

    pitch = 2.*np.pi*dr/rm

    Nb = 1
    xv = np.linspace(0, L, ni)
    rv = np.linspace(r1, r2, nj)
    tv = np.linspace(-pitch/2., pitch/2., nk)

    xrt = np.stack(np.meshgrid(xv, rv, tv, indexing='ij'))

    block = turbigen.grid.PerfectBlock.from_coordinates(xrt, 1, [])
    g = turbigen.grid.Grid([block,])
    g.check_coordinates()

    # Total areas should be
    Ar1 = L*r1*pitch
    Ar2 = L*r2*pitch
    Ax = np.pi*(r2**2.-r1**2.) * pitch / 2. /np.pi
    At = L*dr

    b = g[0]

    vol = Ax * L
    err = vol/np.sum(b.vol_new)-1.
    rtol_vol = 1e-12
    assert np.abs(err) < rtol_vol

    # rtol = 1e-3
    # dAi = np.sum(b.dAi_new,axis=(2,3))
    # err = dAi[0]/Ax-1.
    # print(err)
    # assert np.allclose(dAi[0],Ax, rtol=rtol)
    # print(dAi[1],Ax*rtol)
    # assert (np.abs(dAi[1])<Ax*rtol).all()
    # assert (np.abs(dAi[2])<Ax*rtol).all()

    # dAj = np.sum(b.dAj_new,axis=(1,3))
    # assert (np.abs(dAj[0])<Ar1*rtol).all()
    # print(dAj[1,0], Ar1)
    # assert np.allclose(dAj[1,0],Ar1, rtol=rtol)
    # assert np.allclose(dAj[1,-1],Ar2, rtol=rtol)
    # assert (np.abs(dAj[2])<Ar1*rtol).all()

    # dAk = np.sum(b.dAk_new,axis=(1,2))
    # assert (np.abs(dAk[0])<At*rtol).all()
    # print(dAk[1], At*rtol)
    # assert (np.abs(dAk[1])<At*rtol).all()
    # assert np.allclose(dAk[2],At, rtol=rtol)



def test_cylinder_skew():
    # Geometry
    L = 0.1
    rm = 1.

    ARr = 1.0
    dr = L * ARr

    r1 = rm-dr/2.
    r2 = rm+dr/2.

    nj = 40
    ni = 42
    nk = 44

    ARt = 1.0
    pitch = dr/rm*ARt

    Nb = 1
    xv = np.linspace(0, L, ni)
    rv = np.linspace(r1, r2, nj)
    tv = np.linspace(-pitch/2., pitch/2., nk)

    xrt = np.stack(np.meshgrid(xv, rv, tv, indexing='ij'))
    skew = 60.
    skewr = np.radians(skew)
    xrt[2] += xrt[0]*np.tan(skewr)

    block = turbigen.grid.PerfectBlock.from_coordinates(xrt, 1, [])
    g = turbigen.grid.Grid([block,])
    g.check_coordinates()

    xrrt = xrt.copy()
    xrrt[2] *= xrrt[1]

    # Total areas should be
    Ar1 = L*r1*pitch
    Ar2 = L*r2*pitch
    Ax = np.pi*(r2**2.-r1**2.) * pitch / 2. /np.pi
    At = L*dr

    b = g[0]

    # rtol = 1e-5
    # dAi = np.sum(b.dAi_new,axis=(2,3))
    # assert np.allclose(dAi[0],Ax, rtol=rtol)
    # assert (np.abs(dAi[1])<Ax*rtol).all()
    # assert (np.abs(dAi[2])<Ax*rtol).all()

    # dAj = np.sum(b.dAj_new,axis=(1,3))
    # assert (np.abs(dAj[0])<Ar1*rtol).all()
    # assert np.allclose(dAj[1,0],Ar1, rtol=rtol)
    # assert np.allclose(dAj[1,-1],Ar2, rtol=rtol)
    # assert (np.abs(dAj[2])<Ar1*rtol).all()

    # dAk = np.sum(b.dAk_new,axis=(1,2))
    # assert np.allclose(dAk[0],-At*np.tan(skewr), rtol=rtol)
    # assert (np.abs(dAk[1])<At*rtol).all()
    # assert np.allclose(dAk[2],At, rtol=rtol)

# not_test_box()
test_cylinder()
