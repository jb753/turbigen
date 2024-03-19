import turbigen.compiled
import numpy as np
import turbigen.grid
np.random.seed = 3

def make_ijk():

    ni = 10
    nj = 20
    nk = 30

    # Generate a grid of indices
    iv = np.linspace(0.,ni-1., ni)
    jv = np.linspace(0.,nj-1., nj)
    kv = np.linspace(0.,nk-1., nk)
    i, j, k = np.meshgrid(iv, jv, kv, indexing='ij')

    i = np.asfortranarray(np.expand_dims(i,-1),dtype=np.float32)
    j = np.asfortranarray(np.expand_dims(j,-1),dtype=np.float32)
    k = np.asfortranarray(np.expand_dims(k,-1),dtype=np.float32)

    return i, j, k


def test_node_to_face():

    # Make an ijk grid
    i, j, k = make_ijk()

    fnode = i + j + k
    ni, nj, nk, nv = i.shape
    shape_iface = (ni, nj-1, nk-1, nv)
    shape_jface = (ni-1, nj, nk-1, nv)
    shape_kface = (ni-1, nj-1, nk, nv)
    fi = np.empty(shape_iface,order='F',dtype=np.float32)
    fj = np.empty(shape_jface,order='F',dtype=np.float32)
    fk = np.empty(shape_kface,order='F',dtype=np.float32)

    turbigen.compiled.node_to_face(fnode, fi, fj, fk)

    # If all directions are increasing linearly, then the face-averaged value
    # is exactly one plus the value at low i,j node
    #
    # j+1 *----*
    #     |    |
    # j   *----*
    #    i    i+1
    #
    # face average = ((i + j) + (i+1 + j) + (i + j+1) + (i+1, j+1))/4
    #              # (4i + 4j + 4)/4 = i + j + 1
    #
    assert np.allclose(fi-fnode[:,:-1,:-1,:], 1.)
    assert np.allclose(fj-fnode[:-1,:,:-1,:], 1.)
    assert np.allclose(fk-fnode[:-1,:-1,:,:], 1.)


def test_node_to_cell():

    # Make an ijk grid
    i, j, k = make_ijk()

    # Uniform should stay uniform
    xn = np.ones_like(i)
    ni, nj, nk, nv = i.shape
    shape_cell = (ni-1, nj-1, nk-1, nv)
    xc = np.empty(shape_cell, order='F', dtype=np.float32)
    turbigen.compiled.node_to_cell(xn, xc)
    assert np.allclose(xn[:-1,:-1,:-1,:], xc)

    # Error should be exactly half for linear variation in each dirn
    ic = np.empty(shape_cell, order='F', dtype=np.float32)
    turbigen.compiled.node_to_cell(i, ic)
    assert np.allclose(ic-i[:-1, :-1, :-1,:], 0.5)

    jc = np.empty(shape_cell, order='F', dtype=np.float32)
    turbigen.compiled.node_to_cell(j, jc)
    assert np.allclose(jc-j[:-1, :-1, :-1,:], 0.5)

    kc = np.empty(shape_cell, order='F', dtype=np.float32)
    turbigen.compiled.node_to_cell(k, kc)
    assert np.allclose(kc-k[:-1, :-1, :-1,:], 0.5)


def test_cell_to_node():

    # Make an ijk grid
    i, j, k = make_ijk()

    # Uniform should stay uniform
    xc = np.ones_like(i)
    ni, nj, nk, nv = xc.shape
    shape_node = (ni+1, nj+1, nk+1, nv)
    xn = np.empty(shape_node, order='F', dtype=np.float32)
    turbigen.compiled.cell_to_node(xc, xn)
    assert np.allclose(xc, 1.)

    # Check linear variation in each direction
    inode = np.empty(shape_node, order='F', dtype=np.float32)
    turbigen.compiled.cell_to_node(i, inode)
    assert np.allclose(inode[0,:-1,:-1], i[0,:,:])
    assert np.allclose(inode[-1,:-1,:-1], i[-1,:,:])
    assert np.allclose(inode[1:-1,:-1,:-1]-i[:-1,:,:],0.5)

    jnode = np.empty(shape_node, order='F', dtype=np.float32)
    turbigen.compiled.cell_to_node(j, jnode)
    assert np.allclose(jnode[:-1,0,:-1], j[:,0,:])
    assert np.allclose(jnode[:-1,-1,:-1], j[:,-1,:])
    assert np.allclose(jnode[:-1,1:-1,:-1]-j[:,:-1,:],0.5)

    knode = np.empty(shape_node, order='F', dtype=np.float32)
    turbigen.compiled.cell_to_node(k, knode)
    assert np.allclose(knode[:-1,:-1,0], k[:,:,0])
    assert np.allclose(knode[:-1,:-1,-1], k[:,:,-1])
    assert np.allclose(knode[:-1,:-1,1:-1]-k[:,:,:-1],0.5)


def test_smooth_zero():
    # Zero smoothing factor should change nothing
    shape = (2,4,5,6)
    X = np.random.random_sample(shape)
    Xs = np.asfortranarray(X.copy()).astype(np.float32)
    turbigen.compiled.smooth(Xs, sf2=0., sf4=0.)
    assert np.allclose(X, Xs)


def test_smooth_const():
    # A constant value should stay constant after smoothing
    for sf2 in (0.1, 0.2):
        for sf4 in (0.1, 0.2):
            X = np.ones((10,15,20,1),order='F', dtype=np.float32)
            turbigen.compiled.smooth(X, sf2=sf2, sf4=sf4)
            assert np.allclose(X, 1.)


def test_smooth_linear():
    # Second-order smoothing a linear function should introduce no error

    # Generate a grid of indices
    i, j, k = make_ijk()

    # Define a linear test function
    f = i + 2.*j - 2.*(k-5) + 1.
    f = np.expand_dims(f, -1)

    # Check no change after smoothing
    fs = np.asfortranarray(f.copy())
    turbigen.compiled.smooth(fs, sf2=0.1, sf4=0.)
    err_abs = np.abs(fs - f)
    err_rel = err_abs/f.mean()
    assert np.allclose(f, fs)


def test_smooth_cubic():
    # Fourth-order smoothing a cubic function should introduce no error

    # Generate a grid of indices
    i, j, k = make_ijk()

    # Define a cubic test function
    f = 2.*i**3  + (j**2 -2.*j) - (k-5)**3 + 1.
    f = np.expand_dims(f, -1)

    # Check no change after smoothing
    fs = np.asfortranarray(f.copy())
    turbigen.compiled.smooth(fs, sf2=0.0, sf4=0.1)
    err_abs = np.abs(fs - f)
    err_rel = err_abs/f.mean()
    assert np.allclose(f, fs)


def test_smooth_converge():
    # Repeated smoothing should make it converge to a linear function

    ni = 10
    nj = 20
    nk = 30
    shape = (ni,nk,nk,1)
    X = 0.2*np.random.random_sample(shape)
    sf2 = 0.1
    sf4 = 0.3
    derr = np.inf
    for istep in range(10000):
        Xnew = np.asfortranarray(X.copy()).astype(np.float32)
        turbigen.compiled.smooth(Xnew, sf2, sf4)
        derr = X.ptp() - Xnew.ptp()
        X = Xnew

    assert derr < 1e-5

def test_div_uniform():
    # 

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


    ####
    x = np.asfortranarray(np.zeros((ni, nj, nk, 3)).astype(np.single))

    divx = np.asfortranarray(np.ones_like(b.vol).astype(np.single))
    dAi = np.asfortranarray(np.moveaxis(b.dAi,0,-1).astype(np.single))
    dAj = np.asfortranarray(np.moveaxis(b.dAj,0,-1).astype(np.single))
    dAk = np.asfortranarray(np.moveaxis(b.dAk,0,-1).astype(np.single))
    vol = np.asfortranarray(b.vol.astype(np.single))

    turbigen.compiled.div(x, divx, vol, dAi, dAj, dAk)
    rtol = 1e-2
    assert (np.abs(divx)<rtol).all()

    x[...,0] = 2.*b.x
    print(x[...,1].mean())
    print(x[...,2].mean())

    turbigen.compiled.div(x, divx, vol, dAi, dAj, dAk)
    print(divx.min(), divx.max())

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    # ax.plot(x[:, nj//2+1, nk//2])
    # ax.plot(divx[:, nj//2+1, nk//2])
    ax.plot(vol[:, nj//2, nk//2])
    # ax.plot(dAj[:, nj//2, nk//2])
    plt.show()
    # assert (np.abs(divx)<rtol).all()


test_div_uniform()
# test_node_to_face()
