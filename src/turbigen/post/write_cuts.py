"""Write traverse plane or blade surface cuts to files."""

import os
import turbigen.util

logger = turbigen.util.make_logger()


def post(grid, machine, meanline, _, postdir, mnorm_traverse=(), irow_surf=()):
    """write_cuts(mnorm_traverse=[] irow_surf=[])

    Write 2D cuts of the CFD solution to npz files for external processing.

    Traverse cuts are unstructured planes at a constant streamwise position, such as
    the exit of a blade row. Blade surface cuts are a structured view over an entire aerofoil.

    Unstructured cuts are stored in a 3D data array, where:
        - The first axis indexes over properties `x, r, t, Vx, Vr, Vt, rho, u`
        - The second axis indexes over triangles
        - The third axis indexes over vertices in each triangle

    Structured cuts are stored in a 3D data array, where:
        - The first axis indexes over properties `x, r, t, Vx, Vr, Vt, rho, u`
        - The second axis indexes over chordwise position
        - The third axis indexes over spanwise position

    A example script to read in npz cut data is located in the `scripts` directory.

    Parameters
    ----------
    mnorm_traverse: list
        Normalised meridional coordinates of traverse cuts. The coordinate is defined 0 at the inlet plane, 1 at the
        first row LE, 2 at the first row TE, 3 at the second row LE, and so on. For example, to cut
        just upstream and downstream of the first row, use [0.95, 2.05].
    irow_surf: list
        Indices of rows in which to cut the blade surface. For example, to extract the first blade, use [0,].

    """

    if not mnorm_traverse and not irow_surf:
        logger.info("No cut locations specified.")

    if mnorm_traverse:
        logger.info("Writing traverse cuts...")

    # Loop over stations
    for i, ti in enumerate(mnorm_traverse):
        # Get meridional coordinates of the cut planes
        xrc = machine.ann.get_cut_plane(ti)[0]

        C = grid.unstructured_cut_marching(xrc)

        cutname = os.path.join(postdir, f"cut_traverse_{i}")

        C.write_npz(cutname)

    # Loop over rows
    if irow_surf:
        logger.info("Writing blade surface cuts...")

    surfs = grid.cut_blade_surfs()
    for i in irow_surf:
        # Loop over main/splitter
        for j, surfj in enumerate(surfs[i]):
            cutname = os.path.join(postdir, f"cut_blade_{i}{j}")
            surfj.squeeze().write_npz(cutname)
