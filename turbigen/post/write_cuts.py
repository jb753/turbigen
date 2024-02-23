"""Write traverse plane or blade surface cuts to files."""
import os
import turbigen.util

logger = turbigen.util.make_logger()


def post(grid, machine, meanline, postdir, mnorm_traverse=(), irow_surf=()):

    if not mnorm_traverse and not irow_surf:
        logger.info("No cut locations specified.")

    if mnorm_traverse:
        logger.info("Writing traverse cuts...")

    # Loop over stations
    for i, ti in enumerate(mnorm_traverse):

        # Get meridional coordinates of the cut planes
        xrc = machine.ann.get_cut_plane(ti)[0]

        C = grid.unstructured_cut_marching(xrc)
        cutname = os.path.join(postdir, f"cut_traverse_{i}.yaml.gz")
        C.write(cutname)

    # Loop over rows
    if irow_surf:
        logger.info("Writing blade surface cuts...")

    surfs = grid.cut_blade_surfs()
    for i in irow_surf:
        # Loop over main/splitter
        for j, surfj in enumerate(surfs[i]):
            cutname = os.path.join(postdir, f"cut_blade_{i}{j}.yaml.gz")
            surfj.write(cutname)
