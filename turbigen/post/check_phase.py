"""Save plots of annulus lines."""
import turbigen.util

logger = turbigen.util.make_logger()


def post(
    grid,
    machine,
    meanline,
    postdir,
):

    logger.info("Checking for cells in two-phase region...")
    for ib, b in enumerate(grid):
        n = (b.is_two_phase).sum()
        if n:
            logger.info(f"Block {ib}: {n}/{b.size} cells not in gas phase")
