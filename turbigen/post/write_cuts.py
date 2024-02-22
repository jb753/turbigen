"""Write traverse plane or blade surface cuts to files."""
import numpy as np
import os

def post(grid, machine, meanline, workdir, offsets=None):

    # Default to chord offset
    if offsets is None:
        offsets = 0.05 * np.ones((grid.nrow * 2,))
        offsets[::2] *= -1.0

    # Get meridional coordinates of the cut planes
    xrc = machine.ann.get_cut_planes(offsets)

    # Loop over stations
    for i, xrci in enumerate(xrc):
        C = grid.unstructured_cut_marching(xrci)
        cutname = os.path.join(workdir, f"cut_station_{i}")
        np.savez_compressed(cutname, data=C._data)

    # Loop over rows
    for i, surfi in enumerate(grid.cut_blade_surfs()):
        # Loop over main/splitter
        for j, surfj in enumerate(surfi):
            cutname = os.path.join(workdir, f"cut_blade_{i}{j}")
            np.savez_compressed(cutname, data=surfj._data)

