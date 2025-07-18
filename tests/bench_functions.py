from turbigen.solvers.ember import ember
import numpy as np
from timeit import default_timer as timer
np.random.seed = 3
from turbigen import util



def bench_set_fluxes():
    """Distribute linearly varying nodal values to faces."""

    ni, nj, nk = shape = 81, 73, 65
    cons = np.ones(shape + (5,), order='F',dtype=np.float32)
    Vxrt = np.ones(shape + (3,), order='F',dtype=np.float32)
    P = np.ones(shape, order='F',dtype=np.float32)
    h = np.ones(shape, order='F',dtype=np.float32)
    Pref = 1.
    r = np.ones(shape, order='F',dtype=np.float32)
    ri, rj, rk = util.node_to_face3(r)
    ijk_iwall = np.ones((3,0),order='F',dtype=np.float32)
    ijk_jwall = np.ones((3,0),order='F',dtype=np.float32)
    ijk_kwall = np.ones((3,0),order='F',dtype=np.float32)
    Omega = 0.
    fluxes = [
            np.ones((ni, nj-1, nk-1, 3, 5), order='F',dtype=np.float32),
            np.ones((ni-1, nj, nk-1, 3, 5), order='F',dtype=np.float32),
            np.ones((ni-1, nj-1, nk, 3, 5), order='F',dtype=np.float32),
    ]

    Niter = 100
    tic = timer()
    for _ in range(Niter):
        ember.set_fluxes(cons, Vxrt, P, Pref, h, Omega, r, ri, rj, rk, ijk_iwall, ijk_jwall, ijk_kwall, *fluxes)
    toc = timer()
    elapsed = toc-tic
    tpc = elapsed / Niter / 1e-3
    print(f"set_fluxes() time per call: {tpc:.3g} ms")

#
#
# # subroutine set_fluxes( &
# #     cons, Vxrt, P, Pref, h, &                  ! Flow properties
# #     Omega, &                              ! Reference frame angular velocity
# #     r, ri, rj, rk, &                      ! Node and face-centered radii
# #     ijk_iwall, ijk_jwall, ijk_kwall, &    ! Wall locations
# #     fluxi, fluxj, fluxk, &                ! Fluxes out
# #     ni, nj, nk, niwall, njwall, nkwall &  ! Numbers of points dummy args
# #     )
#
#     # Make an ijk grid
#     i, j, k = make_ijk()
#
#     # Define the test function
#     fnode = i + j + k
#
#     # Preallocate face arrays
#     ni, nj, nk, nv = i.shape
#     shape_iface = (ni, nj - 1, nk - 1, nv)
#     shape_jface = (ni - 1, nj, nk - 1, nv)
#     shape_kface = (ni - 1, nj - 1, nk, nv)
#     fi = np.zeros(shape_iface, order="F", dtype=typ)
#     fj = np.zeros(shape_jface, order="F", dtype=typ)
#     fk = np.zeros(shape_kface, order="F", dtype=typ)
#
#     # Call the subroutine
#     ember.node_to_face(fnode, fi, fj, fk)
#
#     # If all directions are increasing linearly, then the face-averaged value
#     # is exactly one plus the value at low i,j node
#     #
#     # j+1 *----*
#     #     |    |
#     # j   *----*
#     #    i    i+1
#     #
#     # face average = ((i + j) + (i+1 + j) + (i + j+1) + (i+1, j+1))/4
#     #              # (4i + 4j + 4)/4 = i + j + 1
#     #
#     # assert np.allclose(fi - fnode[:, :-1, :-1, :], 1.0)
#     # assert np.allclose(fj - fnode[:-1, :, :-1, :], 1.0)
#     # assert np.allclose(fk - fnode[:-1, :-1, :, :], 1.0)
#
#
# def test_node_to_cell():
#     """Check averaging of nodal values to cell centers"""
#
#     # Make an ijk grid
#     i, j, k = make_ijk()
#
#     # Uniform should stay uniform
#     xn = np.ones_like(i)
#     ni, nj, nk, nv = i.shape
#     shape_cell = (ni - 1, nj - 1, nk - 1, nv)
#     xc = np.zeros(shape_cell, order="F", dtype=typ)
#     ember.node_to_cell(xn, xc)
#     assert np.allclose(xn[:-1, :-1, :-1, :], xc)
#
#     # Discrepancy should be exactly half for linear variation in each dirn
#     ic = np.zeros(shape_cell, order="F", dtype=typ)
#     ember.node_to_cell(i, ic)
#     assert np.allclose(ic - i[:-1, :-1, :-1, :], 0.5)
#
#     jc = np.zeros(shape_cell, order="F", dtype=typ)
#     ember.node_to_cell(j, jc)
#     assert np.allclose(jc - j[:-1, :-1, :-1, :], 0.5)
#
#     kc = np.zeros(shape_cell, order="F", dtype=typ)
#     ember.node_to_cell(k, kc)
#     assert np.allclose(kc - k[:-1, :-1, :-1, :], 0.5)
#
#
# def test_cell_to_node():
#     """Distribute a linear ramp from cell centers to nodes."""
#
#     # Make an ijk grid
#     i, j, k = make_ijk()
#
#     # Uniform should stay uniform
#     xc = np.ones_like(i)
#     ni, nj, nk, nv = xc.shape
#     shape_node = (ni + 1, nj + 1, nk + 1, nv)
#     xn = np.zeros(shape_node, order="F", dtype=typ)
#     ember.cell_to_node(xc, xn)
#     assert np.allclose(xc, 1.0)
#
#     # Check linear variation in each direction
#     # Should have no change at boundaries
#     # Offset of 1/2 along the ramping directoin
#
#     inode = np.zeros(shape_node, order="F", dtype=typ)
#     ember.cell_to_node(i, inode)
#     assert np.allclose(inode[0, :-1, :-1], i[0, :, :])
#     assert np.allclose(inode[-1, :-1, :-1], i[-1, :, :])
#     assert np.allclose(inode[1:-1, :-1, :-1] - i[:-1, :, :], 0.5)
#
#     jnode = np.zeros(shape_node, order="F", dtype=typ)
#     ember.cell_to_node(j, jnode)
#     assert np.allclose(jnode[:-1, 0, :-1], j[:, 0, :])
#     assert np.allclose(jnode[:-1, -1, :-1], j[:, -1, :])
#     assert np.allclose(jnode[:-1, 1:-1, :-1] - j[:, :-1, :], 0.5)
#
#     knode = np.zeros(shape_node, order="F", dtype=typ)
#     ember.cell_to_node(k, knode)
#     assert np.allclose(knode[:-1, :-1, 0], k[:, :, 0])
#     assert np.allclose(knode[:-1, :-1, -1], k[:, :, -1])
#     assert np.allclose(knode[:-1, :-1, 1:-1] - k[:, :, :-1], 0.5)
#
#
# def test_cell_to_face():
#     """Distribute a linear function from cell center to faces."""
#
#     # Make an ijk grid
#     i, j, k = make_ijk()
#
#     fnode = i + j + k
#     ni, nj, nk, nv = i.shape
#     shape_iface = (ni, nj - 1, nk - 1, nv)
#     shape_jface = (ni - 1, nj, nk - 1, nv)
#     shape_kface = (ni - 1, nj - 1, nk, nv)
#     fi = np.zeros(shape_iface, order="F", dtype=typ)
#     fj = np.zeros(shape_jface, order="F", dtype=typ)
#     fk = np.zeros(shape_kface, order="F", dtype=typ)
#
#     # End values should be unchanged
#     # Offset of 1/2 along the direction of interest
#
#     fcell = np.asfortranarray(i[:-1, :-1, :-1, :])
#     ember.cell_to_face(fcell, fi, fj, fk)
#     assert np.allclose(fi[1:-1, :, :, :], i[:-2, :-1, :-1, :] + 0.5)
#     assert np.allclose(fi[0, :, :, :], 0.0)
#     assert np.allclose(fi[-1, :, :, :], ni - 2.0)
#
#     fcell = np.asfortranarray(j[:-1, :-1, :-1, :])
#     ember.cell_to_face(fcell, fi, fj, fk)
#     assert np.allclose(fj[:, 1:-1, :, :], j[:-1, :-2, :-1, :] + 0.5)
#     assert np.allclose(fj[:, 0, :, :], 0.0)
#     assert np.allclose(fj[:, -1, :, :], nj - 2.0)
#
#     fcell = np.asfortranarray(k[:-1, :-1, :-1, :])
#     ember.cell_to_face(fcell, fi, fj, fk)
#     assert np.allclose(fk[:, :, 1:-1, :], k[:-1, :-1, :-2, :] + 0.5)
#     assert np.allclose(fk[:, :, 0, :], 0.0)
#     assert np.allclose(fk[:, :, -1, :], nk - 2.0)
#
#
if __name__ == "__main__":
    bench_set_fluxes()
