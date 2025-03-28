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


def test_node_to_face():
    """Distribute linearly varying nodal values to faces."""

    # Make an ijk grid
    i, j, k = make_ijk()

    # Define the test function
    fnode = i + j + k

    # Preallocate face arrays
    ni, nj, nk, nv = i.shape
    shape_iface = (ni, nj - 1, nk - 1, nv)
    shape_jface = (ni - 1, nj, nk - 1, nv)
    shape_kface = (ni - 1, nj - 1, nk, nv)
    fi = np.zeros(shape_iface, order="F", dtype=typ)
    fj = np.zeros(shape_jface, order="F", dtype=typ)
    fk = np.zeros(shape_kface, order="F", dtype=typ)

    # Call the subroutine
    embsolve.node_to_face(fnode, fi, fj, fk)

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
    assert np.allclose(fi - fnode[:, :-1, :-1, :], 1.0)
    assert np.allclose(fj - fnode[:-1, :, :-1, :], 1.0)
    assert np.allclose(fk - fnode[:-1, :-1, :, :], 1.0)


def test_node_to_cell():
    """Check averaging of nodal values to cell centers"""

    # Make an ijk grid
    i, j, k = make_ijk()

    # Uniform should stay uniform
    xn = np.ones_like(i)
    ni, nj, nk, nv = i.shape
    shape_cell = (ni - 1, nj - 1, nk - 1, nv)
    xc = np.zeros(shape_cell, order="F", dtype=typ)
    embsolve.node_to_cell(xn, xc)
    assert np.allclose(xn[:-1, :-1, :-1, :], xc)

    # Discrepancy should be exactly half for linear variation in each dirn
    ic = np.zeros(shape_cell, order="F", dtype=typ)
    embsolve.node_to_cell(i, ic)
    assert np.allclose(ic - i[:-1, :-1, :-1, :], 0.5)

    jc = np.zeros(shape_cell, order="F", dtype=typ)
    embsolve.node_to_cell(j, jc)
    assert np.allclose(jc - j[:-1, :-1, :-1, :], 0.5)

    kc = np.zeros(shape_cell, order="F", dtype=typ)
    embsolve.node_to_cell(k, kc)
    assert np.allclose(kc - k[:-1, :-1, :-1, :], 0.5)


def test_cell_to_node():
    """Distribute a linear ramp from cell centers to nodes."""

    # Make an ijk grid
    i, j, k = make_ijk()

    # Uniform should stay uniform
    xc = np.ones_like(i)
    ni, nj, nk, nv = xc.shape
    shape_node = (ni + 1, nj + 1, nk + 1, nv)
    xn = np.zeros(shape_node, order="F", dtype=typ)
    embsolve.cell_to_node(xc, xn)
    assert np.allclose(xc, 1.0)

    # Check linear variation in each direction
    # Should have no change at boundaries
    # Offset of 1/2 along the ramping directoin

    inode = np.zeros(shape_node, order="F", dtype=typ)
    embsolve.cell_to_node(i, inode)
    assert np.allclose(inode[0, :-1, :-1], i[0, :, :])
    assert np.allclose(inode[-1, :-1, :-1], i[-1, :, :])
    assert np.allclose(inode[1:-1, :-1, :-1] - i[:-1, :, :], 0.5)

    jnode = np.zeros(shape_node, order="F", dtype=typ)
    embsolve.cell_to_node(j, jnode)
    assert np.allclose(jnode[:-1, 0, :-1], j[:, 0, :])
    assert np.allclose(jnode[:-1, -1, :-1], j[:, -1, :])
    assert np.allclose(jnode[:-1, 1:-1, :-1] - j[:, :-1, :], 0.5)

    knode = np.zeros(shape_node, order="F", dtype=typ)
    embsolve.cell_to_node(k, knode)
    assert np.allclose(knode[:-1, :-1, 0], k[:, :, 0])
    assert np.allclose(knode[:-1, :-1, -1], k[:, :, -1])
    assert np.allclose(knode[:-1, :-1, 1:-1] - k[:, :, :-1], 0.5)


def test_cell_to_face():
    """Distribute a linear function from cell center to faces."""

    # Make an ijk grid
    i, j, k = make_ijk()

    fnode = i + j + k
    ni, nj, nk, nv = i.shape
    shape_iface = (ni, nj - 1, nk - 1, nv)
    shape_jface = (ni - 1, nj, nk - 1, nv)
    shape_kface = (ni - 1, nj - 1, nk, nv)
    fi = np.zeros(shape_iface, order="F", dtype=typ)
    fj = np.zeros(shape_jface, order="F", dtype=typ)
    fk = np.zeros(shape_kface, order="F", dtype=typ)

    # End values should be unchanged
    # Offset of 1/2 along the direction of interest

    fcell = np.asfortranarray(i[:-1, :-1, :-1, :])
    embsolve.cell_to_face(fcell, fi, fj, fk)
    assert np.allclose(fi[1:-1, :, :, :], i[:-2, :-1, :-1, :] + 0.5)
    assert np.allclose(fi[0, :, :, :], 0.0)
    assert np.allclose(fi[-1, :, :, :], ni - 2.0)

    fcell = np.asfortranarray(j[:-1, :-1, :-1, :])
    embsolve.cell_to_face(fcell, fi, fj, fk)
    assert np.allclose(fj[:, 1:-1, :, :], j[:-1, :-2, :-1, :] + 0.5)
    assert np.allclose(fj[:, 0, :, :], 0.0)
    assert np.allclose(fj[:, -1, :, :], nj - 2.0)

    fcell = np.asfortranarray(k[:-1, :-1, :-1, :])
    embsolve.cell_to_face(fcell, fi, fj, fk)
    assert np.allclose(fk[:, :, 1:-1, :], k[:-1, :-1, :-2, :] + 0.5)
    assert np.allclose(fk[:, :, 0, :], 0.0)
    assert np.allclose(fk[:, :, -1, :], nk - 2.0)


if __name__ == "__main__":
    test_node_to_face()
    test_node_to_cell()
    test_cell_to_node()
    test_cell_to_face()
