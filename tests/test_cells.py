"""Check cell areas and volumes are correct."""

import turbigen.grid
import numpy as np
import turbigen.compflow_native as cf


def dot(a, b, axis=0):
    return np.sum(a * b, axis=axis)


def test_box():

    print(f"Checking discretisation of a cube...")

    # Geometry
    L = 0.1
    yoffset = 40.0 * L

    nn = 60
    nj = nn
    ni = nn + 2
    nk = nn + 4

    Nb = 1
    xv = np.linspace(-L, L, ni)
    yv = np.linspace(-L, L, nj) + yoffset
    zv = -np.linspace(-L, L, nk)

    x, y, z = np.stack(np.meshgrid(xv, yv, zv, indexing="ij"))

    # Convert Cartesian coordinates to polar
    r = np.sqrt(y**2 + z**2)
    t = np.arctan2(-z, y)

    xrt = np.stack((x, r, t))

    block = turbigen.grid.PerfectBlock.from_coordinates(xrt, 1, [])
    g = turbigen.grid.Grid(
        [
            block,
        ]
    )
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

    ez = np.stack(
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
    A = (2 * L) ** 2
    rtol_A = 1e-7
    err_x = dot(b.dAi, ex).sum(axis=(1, 2)) / A - 1.0
    err_z = dot(b.dAj, ez).sum(axis=(0, 2)) / A - 1.0
    err_y = dot(b.dAk, ey).sum(axis=(0, 1)) / A - 1.0
    print(
        f"Area errors Ax={err_x.max():.2e}, Ay={err_y.max():.2e}, Az={err_z.max():.2e}"
    )
    assert (err_x < rtol_A).all()
    assert (err_y < rtol_A).all()
    assert (err_z < rtol_A).all()

    # Check the total volume
    vol = (2 * L) ** 3
    err = vol / np.sum(b.vol) - 1.0
    rtol_vol = 1e-7
    print(f"Volume error = {err:.2e}")
    assert np.abs(err) < rtol_vol


def test_cylinder():

    # Geometry
    L = 0.1
    rm = 10.0
    dr = 0.1

    r1 = rm - dr / 2.0
    r2 = rm + dr / 2.0

    nn = 40 * 2
    nj = nn + 2
    ni = nn + 4
    nk = nn

    pitch = 2.0 * np.pi * dr / rm

    Nb = 1
    xv = np.linspace(0, L, ni)
    rv = np.linspace(r1, r2, nj)
    tv = np.linspace(0.0, pitch, nk)

    xrt = np.stack(np.meshgrid(xv, rv, tv, indexing="ij"))

    block = turbigen.grid.PerfectBlock.from_coordinates(xrt, 1, [])
    g = turbigen.grid.Grid(
        [
            block,
        ]
    )
    g.check_coordinates()

    print(f"Checking discretisation of a cylinder...")

    # Total areas should be
    Ar1 = L * r1 * pitch
    Ar2 = L * r2 * pitch
    Ax = np.pi * (r2**2.0 - r1**2.0) * pitch / 2.0 / np.pi
    At = L * dr
    Arside = (r2 - r1) * pitch * L / 2.0
    vol = Ax * L

    b = g[0]

    rtol_A = 1e-9
    dAi = np.sum(b.dAi, axis=(2, 3))
    err_x = dAi[0] / Ax - 1.0
    err_r = dAi[1] / Ax
    err_t = dAi[2] / Ax
    print(
        f"i-face errors: Ax={err_x.mean():.2e}, Ar={err_r.mean():.2e},"
        f" At={err_t.mean():.2e}"
    )
    assert (err_x < rtol_A).all()
    assert (err_r < rtol_A).all()
    assert (err_t < rtol_A).all()

    dAj = np.sum(b.dAj, axis=(1, 3))
    err_x = dAj[0] / Ar1
    err_r = np.array([dAj[1, 0] / Ar1 - 1.0, dAj[1, -1] / Ar2 - 1.0])
    err_t = dAj[2] / Ar1
    print(
        f"j-face errors: Ax={err_x.mean():.2e}, Ar={err_r.mean():.2e},"
        f" At={err_t.mean():.2e}"
    )
    assert (err_x < rtol_A).all()
    assert (err_r < rtol_A).all()
    assert (err_t < rtol_A).all()

    dAk = np.sum(b.dAk, axis=(1, 2))
    err_x = dAk[0] / At
    err_r = dAk[1] / At
    err_t = dAk[2] / At - 1.0
    print(
        f"k-face errors: Ax={err_x.mean():.2e}, Ar={err_r.mean():.2e},"
        f" At={err_t.mean():.2e}"
    )
    assert (err_x < rtol_A).all()
    assert (err_r < rtol_A).all()
    assert (err_t < rtol_A).all()

    # Check the total volume
    err = vol / np.sum(b.vol) - 1.0
    rtol_vol = 1e-12
    print(f"Volume error = {err:.2e}")
    assert np.abs(err) < rtol_vol
