import turbigen.smooth
import numpy as np
np.random.seed = 3

def test_zero():
    # Zero smoothing factor should change nothing
    shape = (2,4,5,6)
    X = np.random.random_sample(shape)
    Xs = turbigen.smooth.smooth(X, sf=0.)
    assert np.allclose(X, Xs)

def test_const():
    # A constant value should stay constant
    X = np.ones((1,10,15,20))
    Xs = turbigen.smooth.smooth(X, sf=0.1)
    assert np.allclose(Xs, 1.)

def test_error():
    # Smoothing should cause no error for linear function

    # Generate a grid of indices
    ni = 10
    nj = 20
    nk = 30
    iv = np.linspace(0.,ni-1., ni)
    jv = np.linspace(0.,nj-1., nj)
    kv = np.linspace(0.,nk-1., nk)
    i, j, k = np.meshgrid(iv, jv, kv, indexing='ij')

    # Define a linear test function
    f = i + 2.*j - 2.*(k-5) + 1.
    f = np.expand_dims(f, 0)

    # Check no change after smoothing
    fs = turbigen.smooth.smooth(f, sf=0.1)
    err_abs = np.abs(fs - f)
    err_rel = err_abs/f.mean()
    assert np.allclose(f, fs)


def test_uniform():
    # Repeated smoothing should make it converge to a linear function

    ni = 10
    nj = 20
    nk = 30
    shape = (1,ni,nk,nk)
    X = 0.2*np.random.random_sample(shape)
    sf = 0.5
    derr = np.inf
    for istep in range(10000):
        Xnew = turbigen.smooth.smooth(X, sf)
        derr = X.ptp() - Xnew.ptp()
        X = Xnew

    assert derr < 1e-5
