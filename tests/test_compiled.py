import turbigen.compiled
import numpy as np
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

    return i, j, k

# def test_node_to_cell():

#     # Make an ijk grid
#     i, j, k = make_ijk()
#     i = np.asfortranarray(np.expand_dims(i,0))
#     j = np.asfortranarray(np.expand_dims(j,0))
#     k = np.asfortranarray(np.expand_dims(k,0))

#     # Uniform should stay uniform
#     xu = np.ones_like(i)
#     assert np.allclose(xu[:,:-1,:-1,:-1], turbigen.compiled.node_to_cell(xu))

#     # Error should be exactly half for linear variation in each dirn
#     fi = turbigen.compiled.node_to_cell(i)[0]
#     assert np.allclose(fi-i[0,:-1, :-1, :-1], 0.5)
#     fj = turbigen.compiled.node_to_cell(j)[0]
#     assert np.allclose(fj-j[0,:-1, :-1, :-1], 0.5)
#     fk = turbigen.compiled.node_to_cell(k)[0]
#     assert np.allclose(fk-k[0,:-1, :-1, :-1], 0.5)

# def test_cell_to_node():

#     # Make an ijk grid
#     i, j, k = make_ijk()

#     # Uniform should stay uniform
#     xu = np.expand_dims(np.ones_like(i),0)
#     xun =  turbigen.compiled.cell_to_node(xu)
#     assert np.allclose(xun, 1.)

#     # Check linear variation in each direction

#     inode = turbigen.compiled.cell_to_node(np.expand_dims(i, 0))[0]
#     assert np.allclose(inode[0,:-1,:-1], i[0,:,:])
#     assert np.allclose(inode[-1,:-1,:-1], i[-1,:,:])
#     assert np.allclose(inode[1:-1,:-1,:-1]-i[:-1,:,:],0.5)

#     jnode = turbigen.compiled.cell_to_node(np.expand_dims(j, 0))[0]
#     assert np.allclose(jnode[:-1,0,:-1], j[:,0,:])
#     assert np.allclose(jnode[:-1,-1,:-1], j[:,-1,:])
#     assert np.allclose(jnode[:-1,1:-1,:-1]-j[:,:-1,:],0.5)

#     knode = turbigen.compiled.cell_to_node(np.expand_dims(k, 0))[0]
#     assert np.allclose(knode[:-1,:-1,0], k[:,:,0])
#     assert np.allclose(knode[:-1,:-1,-1], k[:,:,-1])
#     assert np.allclose(knode[:-1,:-1,1:-1]-k[:,:,:-1],0.5)

def test_smooth_zero():
    # Zero smoothing factor should change nothing
    shape = (2,4,5,6)
    X = np.random.random_sample(shape)
    Xs = np.asfortranarray(X.copy())
    turbigen.compiled.smooth(Xs, sf2=0., sf4=0.)
    assert np.allclose(X, Xs)


def test_smooth_const():
    # A constant value should stay constant after smoothing
    for sf2 in (0.1, 0.2):
        for sf4 in (0.1, 0.2):
            X = np.ones((10,15,20,1),order='F')
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
        Xnew = np.asfortranarray(X.copy())
        turbigen.compiled.smooth(Xnew, sf2, sf4)
        derr = X.ptp() - Xnew.ptp()
        X = Xnew

    assert derr < 1e-5

test_smooth_converge()
