"""Save plots of annulus lines."""

import turbigen.util

logger = turbigen.util.make_logger()


def post(
    grid,
    machine,
    meanline,
    __,
    postdir,
):
    """check_phase()

    Print number of computational cells in the two-phase region to the log file.

    This is useful to check if real working fluids remain gaseous throughout
    the turbomachine."""

    logger.info("Checking for cells in two-phase region...")
    for ib, b in enumerate(grid):
        n = (b.is_two_phase).sum()
        if n:
            logger.info(f"Block {ib}: {n}/{b.size} cells not in gas phase")
