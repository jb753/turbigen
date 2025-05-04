from turbigen.solvers.emb import embsolve
import numpy as np
import turbigen.grid
from timeit import default_timer as timer

np.random.seed = 3

typ = np.float32

# Utility functions


def make_ijk():
    """Assembly ijk 3D arrays."""

    ni = 10
    nj = 20
    nk = 30

    # Generate a grid of indices
    iv = np.linspace(0.0, ni - 1.0, ni)
    jv = np.linspace(0.0, nj - 1.0, nj)
    kv = np.linspace(0.0, nk - 1.0, nk)
    i, j, k = np.meshgrid(iv, jv, kv, indexing="ij")

    i = np.asfortranarray(np.expand_dims(i, -1), dtype=typ)
    j = np.asfortranarray(np.expand_dims(j, -1), dtype=typ)
    k = np.asfortranarray(np.expand_dims(k, -1), dtype=typ)

    return i, j, k


def to_fort(x):
    """Convert an array to Fortran."""
    x = np.asfortranarray(x.copy()).astype(typ)
    return x


def get_L_P(x):
    """Generate isotropic cell lengths and test pressure fields ."""
    # Equal weights in each dirn
    shape = x.shape
    L = to_fort(np.ones(shape[:-1] + (3,)))
    # Uniform pressure for 4th-order only
    P4 = to_fort(np.ones(shape[:-1]))
    # Wobbly pressure for 2nd-order only
    P2 = to_fort(np.ones(shape[:-1]))
    P2[::2, ::2, ::2] = 2.0
    return L, P2, P4


def make_cylinder(ni, nj, nk):
    """Assemble coordinates for a cylindrical sector."""

    # Geometry
    L = 0.1
    rm = 2.0

    ARr = 1.0
    dr = L * ARr

    r1 = rm - dr / 2.0
    r2 = rm + dr / 2.0

    ARt = 1.0
    pitch = dr / rm * ARt

    Nb = 1
    xv = np.linspace(0, L, ni)
    rv = np.linspace(r1, r2, nj)
    tv = np.linspace(-pitch / 2.0, pitch / 2.0, nk)

    xrt = np.stack(np.meshgrid(xv, rv, tv, indexing="ij"))
    skew = 60.0
    skewr = np.radians(skew)
    xrt[2] += xrt[0] * np.tan(skewr)

    block = turbigen.grid.PerfectBlock.from_coordinates(xrt, 1, [])
    g = turbigen.grid.Grid(
        [
            block,
        ]
    )
    g.check_coordinates()

    return g


# Begin test functions


def test_smooth_zero():
    """Zero smoothing factor should change nothing."""

    shape = (5, 6, 7, 3)
    X = np.random.random_sample(shape)
    Xs = to_fort(X)
    L, P2, P4 = get_L_P(Xs)

    embsolve.smooth(Xs, P2, L, sf2=0.0, sf4=0.0, sf2min=0.0)
    assert np.allclose(X, Xs)

    embsolve.smooth(Xs, P4, L, sf2=0.0, sf4=0.0, sf2min=0.0)
    assert np.allclose(X, Xs)


def test_smooth_const():
    """A constant value should stay constant after smoothing."""

    for sf2 in (0.1, 0.2):
        for sf4 in (0.1, 0.2):
            X = np.ones((10, 15, 20, 1), order="F", dtype=typ)
            L, P2, P4 = get_L_P(X)

            embsolve.smooth(X, P2, L, sf2=sf2, sf4=sf4, sf2min=0.0)
            assert np.allclose(X, 1.0)

            embsolve.smooth(X, P4, L, sf2=sf2, sf4=sf4, sf2min=0.0)
            assert np.allclose(X, 1.0)


def test_smooth_linear():
    """Smoothing a linear function should introduce no error."""

    # Generate a grid of indices
    i, j, k = make_ijk()

    # Define a linear test function
    f = i + 2.0 * j - 2.0 * (k - 5) + 1.0
    f = np.expand_dims(f, -1)

    # Check no change after smoothing
    fs = np.asfortranarray(f.copy())
    L, P2, P4 = get_L_P(fs)

    embsolve.smooth(fs, P2, L, sf2=0.1, sf4=0.0, sf2min=0.0)
    err_abs = np.abs(fs - f)
    err_rel = err_abs / f.mean()
    assert np.allclose(f, fs)

    fs = np.asfortranarray(f.copy())
    embsolve.smooth(fs, P4, L, sf2=0.0, sf4=0.05, sf2min=0.0)
    err_abs = np.abs(fs - f)
    err_rel = err_abs / f.mean()
    assert np.allclose(f, fs)

    fs = np.asfortranarray(f.copy())
    embsolve.smooth(fs, P4, L, sf2=0.1, sf4=0.05, sf2min=0.0)
    err_abs = np.abs(fs - f)
    err_rel = err_abs / f.mean()
    assert np.allclose(f, fs)


def test_smooth_cubic():
    """Fourth-order smoothing a cubic function should introduce no error."""

    # Generate a grid of indices
    i, j, k = make_ijk()

    # Define a cubic test function
    f = 2.0 * i**3 + (j**2 - 2.0 * j) - (k - 5) ** 3 + 1.0
    f = np.expand_dims(f, -1)

    # Check no change after smoothing
    fs = np.asfortranarray(f.copy())
    L, P2, P4 = get_L_P(fs)
    embsolve.smooth(fs, P4, L, sf2=0.1, sf4=0.1, sf2min=0.0)
    err_abs = np.abs(fs - f)
    err_rel = err_abs / f.mean()
    # Note because we revert to 2nd-order at boundaries the edges
    # will be wrong - exclude from comparison
    assert np.allclose(f[2:-2, 2:-2, 2:-2, 0], fs[2:-2, 2:-2, 2:-2, 0])

    # Check that the shock sensor works
    fs = np.asfortranarray(f.copy())
    embsolve.smooth(fs, P2, L, sf2=0.1, sf4=0.1, sf2min=0.0)
    err_abs = np.abs(fs - f)
    err_rel = err_abs / f.mean()
    assert not np.allclose(f, fs)

    # Check that sf2min works
    fs = np.asfortranarray(f.copy())
    embsolve.smooth(fs, P4, L, sf2=0.0, sf4=0.1, sf2min=0.1)
    err_abs = np.abs(fs - f)
    err_rel = err_abs / f.mean()
    assert not np.allclose(f, fs)


def test_smooth_converge():
    """Repeated smoothing should make it converge to a linear function."""

    ni = 10
    nj = 20
    nk = 30
    shape = (ni, nk, nk, 1)
    X = 0.2 * np.random.random_sample(shape)
    sf2 = 0.1
    sf4 = 0.3
    derr = np.inf
    L, P2, P4 = get_L_P(X)
    for istep in range(10000):
        Xnew = np.asfortranarray(X.copy()).astype(typ)
        embsolve.smooth(Xnew, P4, L, sf2, sf4, sf2min=0.0)
        derr = np.ptp(X) - np.ptp(Xnew)
        X = Xnew

    assert derr < 1e-5


if __name__ == "__main__":
    start = timer()
    for _ in range(10):
        test_smooth_zero()
        test_smooth_const()
        test_smooth_linear()
        test_smooth_cubic()
        test_smooth_converge()
    end = timer()
    print(f"All tests passed in {end - start:.3f} seconds")
