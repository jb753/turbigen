"""Write traverse plane or blade surface cuts to files."""
import os
import turbigen.util

logger = turbigen.util.make_logger()


def post(grid, machine, meanline, postdir, mnorm_traverse=[], irow_surf=[]):
    """write_cuts(mnorm_traverse=[] irow_surf=[])

        Write 2D cuts of the CFD solution to files for external processing.

        Traverse cuts are unstructured at a constant streamwise position, such as
        the exit of a blade row.

    The first dimension indexes over variables: x, r, t, Vx, Vr, Vt, static P,
    static T, wall distance, turbulent viscosity, reference frame angular
    velocity
    The second dimension indexes over triangles
    The third dimension indexes over vertices in the triangle


        Parameters
        ----------
        mnorm_traverse: list
            Normalised meridional coordinates of traverse cuts. For example, to cut
            just upstream and downstream of the first row, use [0.95, 2.05].
        irow_surf: list
            Row indexes of blade surface cuts. For example, to cut the first blade, use [0,].

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
            surfj.write_npz(cutname)
