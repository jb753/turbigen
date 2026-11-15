"""Fixtures shared by the turbigen tests."""

import os
from pathlib import Path

import numpy as np
import pytest

SNAPSHOTS = Path(__file__).parents[1] / "data" / "snapshots"
"""Where stored outputs are kept, one ``.npz`` file per snapshot."""

UPDATE_SNAPSHOTS = os.environ.get("TURBIGEN_UPDATE_SNAPSHOTS") == "1"
"""Set ``TURBIGEN_UPDATE_SNAPSHOTS=1`` to overwrite the stored outputs with the
current ones, after a deliberate change in behaviour."""


@pytest.fixture
def snapshot():
    """Return a function that loads the stored arrays of a named snapshot.

    Called as ``snapshot(name, **arrays)`` with the arrays as computed now, and
    returns the stored ones under the same keys. With
    :data:`UPDATE_SNAPSHOTS` set, the computed arrays are written first, so
    the comparison that follows passes trivially.
    """

    def load(name, **arrays):
        path = SNAPSHOTS / f"{name}.npz"
        if UPDATE_SNAPSHOTS:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(path, **arrays)
        with np.load(path) as stored:
            assert set(stored.files) == set(arrays), f"keys differ from {path}"
            return {key: stored[key] for key in stored.files}

    return load
