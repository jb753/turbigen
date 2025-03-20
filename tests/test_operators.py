from turbigen.solvers.emb import embsolve
import numpy as np
import turbigen.grid

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


def get_scale_fact(x):
    """Make isotropic smoothing factors."""
    # Equal weights in each dirn
    shape = x.shape
    return to_fort(np.ones(shape[:-1] + (3,)) / 3.0)


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


def test_div():
    """Check approximation of divergence for analytic functions."""

    nn = 40
    nj = nn
    ni = nn + 2
    nk = nn + 4
    g = make_cylinder(ni, nj, nk)

    b = g[0]

    x = np.asfortranarray(np.ones((ni, nj, nk, 3)).astype(typ))

    divx = np.asfortranarray(np.ones_like(b.vol).astype(typ))
    dAi = np.asfortranarray(np.moveaxis(b.dAi, 0, -1).astype(typ))
    dAj = np.asfortranarray(np.moveaxis(b.dAj, 0, -1).astype(typ))
    dAk = np.asfortranarray(np.moveaxis(b.dAk, 0, -1).astype(typ))
    vol = np.asfortranarray(b.vol.astype(typ))

    rn = np.asfortranarray(b.r.astype(typ))
    ni, nj, nk = rn.shape
    shape_cell = (ni - 1, nj - 1, nk - 1)
    rc = np.zeros(shape_cell, order="F", dtype=typ)
    embsolve.node_to_cell(rn, rc)

    print("Checking divergence of test fields...")
    print(
        "Note that in a cylindrical coordinate system:\n"
        "  div u = dux/dx + d(r*ur)/dr/r + dut/dt/r"
    )

    rtol = 1e-4

    x[..., 0] = 0.0
    x[..., 1] = 0.0
    x[..., 2] = 0.0
    embsolve.div(x, divx, vol, dAi, dAj, dAk)
    err = np.abs(divx)
    print(f"div(0)=0 error={err.max():.2e}")
    assert (err < rtol).all()

    x[..., 0] = 0.0
    x[..., 1] = 1.0
    x[..., 2] = 0.0
    embsolve.div(x, divx, vol, dAi, dAj, dAk)
    err = np.abs(divx * rc - 1)
    print(f"div(er)=1/r error={err.max():.2e}")
    assert (err < rtol).all()

    x[..., 0] = 2.0 * b.x
    x[..., 1] = 0.0
    x[..., 2] = 0.0
    embsolve.div(x, divx, vol, dAi, dAj, dAk)
    err = np.abs(divx / 2.0 - 1.0)
    print(f"div(2x ex)=2 error={err.max():.2e}")
    assert (err < rtol).all()

    x[..., 0] = 0.0
    x[..., 1] = 0.0
    x[..., 2] = -b.t
    embsolve.div(x, divx, vol, dAi, dAj, dAk)
    err = np.abs(divx / (-1.0 / rc) - 1.0)
    print(f"div(-t et)=-1/r error={err.max():.2e}")
    assert (err < rtol).all()

    x[..., 0] = 0.0
    x[..., 1] = 3.0 * b.r
    x[..., 2] = 0.0
    embsolve.div(x, divx, vol, dAi, dAj, dAk)
    err = np.abs(divx / 6.0 - 1.0)
    print(f"div(3r er)=6. error={err.max():.2e}")
    assert (err < rtol).all()


def test_grad():
    """Check approximation of gradient for analytic functions."""

    n = 40
    nj = n
    ni = n + 2
    nk = n + 4
    g = make_cylinder(ni, nj, nk)

    b = g[0]

    print("Checking grad of test fields...")

    gradq = np.asfortranarray(np.ones((ni - 1, nj - 1, nk - 1, 3)).astype(typ)) * np.nan
    dAi = np.asfortranarray(np.moveaxis(b.dAi, 0, -1).astype(typ))
    dAj = np.asfortranarray(np.moveaxis(b.dAj, 0, -1).astype(typ))
    dAk = np.asfortranarray(np.moveaxis(b.dAk, 0, -1).astype(typ))
    vol = np.asfortranarray(b.vol.astype(typ))

    rn = np.asfortranarray(b.r.astype(typ))
    tn = np.asfortranarray(b.t.astype(typ))
    xn = np.asfortranarray(b.x.astype(typ))
    ni, nj, nk = rn.shape
    shape_cell = (ni - 1, nj - 1, nk - 1)
    rc = np.zeros(shape_cell, order="F", dtype=typ)
    embsolve.node_to_cell(rn, rc)
    tc = np.zeros(shape_cell, order="F", dtype=typ)
    embsolve.node_to_cell(tn, tc)
    xc = np.zeros(shape_cell, order="F", dtype=typ)
    embsolve.node_to_cell(xn, xc)

    rtol = 2e-4

    q = np.asfortranarray(np.ones_like(b.r)).astype(typ)
    embsolve.grad(q, gradq, vol, dAi, dAj, dAk, rn, rc)
    err_x = np.abs(gradq[..., 0])
    err_r = np.abs(gradq[..., 1])
    err_t = np.abs(gradq[..., 2])
    print(
        f"grad(1)=0 err_x={err_x.max():.2e}, err_r={err_r.max():.2e},"
        f" err_t={err_t.max():.2e}"
    )
    assert (err_x < rtol).all()
    assert (err_r < rtol).all()
    assert (err_t < rtol).all()

    q = np.asfortranarray(b.x).astype(typ)
    embsolve.grad(q, gradq, vol, dAi, dAj, dAk, rn, rc)
    err_x = np.abs(gradq[..., 0] - 1.0)
    err_r = np.abs(gradq[..., 1])
    err_t = np.abs(gradq[..., 2])
    print(
        f"grad(x)=ex err_x={err_x.max():.2e}, err_r={err_r.max():.2e},"
        f" err_t={err_t.max():.2e}"
    )
    assert (err_x < rtol).all()
    assert (err_r < rtol).all()
    assert (err_t < rtol).all()

    q = np.asfortranarray(-2.0 * b.r).astype(typ)
    embsolve.grad(q, gradq, vol, dAi, dAj, dAk, rn, rc)
    err_x = np.abs(gradq[..., 0])
    err_r = np.abs(gradq[..., 1] / -2.0 - 1.0)
    err_t = np.abs(gradq[..., 2])
    print(
        f"grad(-2r)=-2er err_x={err_x.max():.2e}, err_r={err_r.max():.2e},"
        f" err_t={err_t.max():.2e}"
    )
    assert (err_x < rtol).all()
    assert (err_r < rtol).all()
    assert (err_t < rtol).all()

    q = np.asfortranarray(b.r**2).astype(typ)
    embsolve.grad(q, gradq, vol, dAi, dAj, dAk, rn, rc)
    err_x = np.abs(gradq[..., 0])
    err_r = np.abs(gradq[..., 1] / (2 * rc) - 1.0)
    err_t = np.abs(gradq[..., 2])
    print(
        f"grad(r^2)=2r er err_x={err_x.max():.2e}, err_r={err_r.max():.2e},"
        f" err_t={err_t.max():.2e}"
    )
    assert (err_x < rtol).all()
    assert (err_r < rtol).all()
    assert (err_t < rtol).all()

    q = np.asfortranarray(b.t).astype(typ)
    embsolve.grad(q, gradq, vol, dAi, dAj, dAk, rn, rc)
    err_x = np.abs(gradq[..., 0])
    err_r = np.abs(gradq[..., 1])
    err_t = np.abs(gradq[..., 2] / (1.0 / rc) - 1.0)
    print(
        f"grad(t)=1/r et err_x={err_x.max():.2e}, err_r={err_r.max():.2e},"
        f" err_t={err_t.max():.2e}"
    )
    assert (err_x < rtol).all()
    assert (err_r < rtol).all()
    assert (err_t < rtol).all()


if __name__ == "__main__":
    test_div()
    test_grad()
