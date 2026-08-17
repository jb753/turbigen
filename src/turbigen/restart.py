"""Carrying a flow field from one run into the next.

A run writes what it reached to ``restart.npz`` beside its config, and a later
run can start from it instead of the crude meridional guess.

Not in the YAML, though `ARCHITECTURE.md` once proposed it. Nothing reads a
flow field out of a config file: automatic iteration passes the grid in memory,
and a manual restart names a path anyway. Putting megabytes of base64 in the
document would only slow down every reader that wants the twenty numbers under
``result:``, and cost a 4/3 inflation for the privilege.

Primitives are stored, not the conserved variables. Conserved energy is
measured from its fluid's datum, and the datum moves between runs --- the
mesher derives it from each design --- so a conserved field written by one run
would be quietly misread by the next. Pressure, temperature and velocity cross
unchanged.
"""

import logging

import numpy as np
import ember.block_util

logger = logging.getLogger("turbigen")

STATE = ember.block_util.STATE
"""What is stored per block: ember's own transfer variables, all dimensional.

Taken from ember rather than restated, so a field written here is exactly what
:meth:`ember.grid.Grid.interp_from_arrays` expects to be handed back.
"""


def save(path, grid):
    """Write the flow field in `grid` to `path`."""
    arrays = {}
    for i_block, block in enumerate(grid):
        for name in STATE:
            arrays[f"b{i_block}_{name}"] = np.asarray(
                getattr(block, name), dtype=np.float32
            )

    np.savez_compressed(path, **arrays)
    logger.debug(f"Wrote a restart field for {len(grid)} block(s) to {path}")


def apply(grid, path):
    """Write the flow field stored at `path` into `grid`, in place.

    Blocks are matched by position and interpolated in index space where the
    resolution differs, so a mesh that changed with the design is fine as long
    as its topology did not. Index space maps leading edge to leading edge,
    which is what makes a guess from a previous design worth having.
    """
    data = np.load(path)

    n_block = len({key.split("_", 1)[0] for key in data.files})
    if n_block != len(grid):
        raise ValueError(
            f"The restart field has {n_block} block(s) but this grid has "
            f"{len(grid)}; it was written for a different machine."
        )

    grid.interp_from_arrays(
        [[data[f"b{i}_{name}"] for name in STATE] for i in range(n_block)]
    )

    logger.info(f"Started from the restart field in {path}")
