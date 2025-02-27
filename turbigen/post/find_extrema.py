import numpy as np
import turbigen.util

logger = turbigen.util.make_logger()


def post(
    grid,
    machine,
    meanline,
    _,
    postdir,
):
    vnames = ["V", "Po", "To"]
    logger.info("Finding extrema...")
    for ib, b in enumerate(grid):
        logger.info(f"Block {ib} shape {b.shape}")
        for iv in range(len(vnames)):
            ijkmax = np.unravel_index(np.argmax(getattr(b, vnames[iv])), b.shape)
            logger.info(f" Max {vnames[iv]}: {ijkmax} {getattr(b, vnames[iv])[ijkmax]}")
            ijkmax = np.unravel_index(np.argmin(getattr(b, vnames[iv])), b.shape)
            logger.info(f" Min {vnames[iv]}: {ijkmax} {getattr(b, vnames[iv])[ijkmax]}")
