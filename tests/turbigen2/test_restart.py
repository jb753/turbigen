"""Tests for carrying a flow field from one run into the next.

The interpolation is ember's; what is ours is the file and the decision about
what goes in it. So these check that a field survives the round trip, that it
survives a change of mesh resolution, and that it is not quietly reinterpreted
by a change of datum -- which is the trap conserved variables would fall into,
and the reason primitives are stored instead.

Test cases:
- test_a_saved_field_comes_back_unchanged: the round trip, at one resolution
- test_a_field_can_be_restarted_onto_a_finer_mesh: interpolated in index space
- test_a_field_is_not_reinterpreted_by_a_changed_datum: why not conserved
- test_the_file_holds_primitives_only: no coordinates, no patches, no conserved
- test_a_field_for_another_machine_is_refused: block count must match
"""

import numpy as np
import pytest

from test_mixout import CASCADE
from turbigen2 import cli, restart
from turbigen2.config import Config


@pytest.fixture(scope="module")
def solved(tmp_path_factory):
    """A short march, and the field it reached written to a file."""
    config = Config.from_dict(CASCADE)
    machine, grid = cli.prepare(config)
    config.solver.solve(grid)

    path = tmp_path_factory.mktemp("restart") / "restart.npz"
    restart.save(path, grid)
    return config, grid, path


def test_a_saved_field_comes_back_unchanged(solved):
    config, grid, path = solved
    _, fresh = cli.prepare(config)

    restart.apply(fresh, path)

    for name in ("P", "T", "Vx", "Vr", "Vt"):
        np.testing.assert_allclose(
            np.asarray(getattr(fresh[0], name)),
            np.asarray(getattr(grid[0], name)),
            rtol=1e-5,
            err_msg=f"{name} did not survive the round trip",
        )


def test_a_field_can_be_restarted_onto_a_finer_mesh(solved):
    """A design iteration re-meshes, so the shapes need not match.

    Index space maps leading edge to leading edge, which is what makes a field
    from a previous design worth starting from at all.
    """
    config, grid, path = solved
    finer = Config.from_dict(
        {**CASCADE, "mesh": {**CASCADE["mesh"], "resolution_factor": 0.4}}
    )
    _, fresh = cli.prepare(finer)
    assert fresh[0].shape != grid[0].shape

    restart.apply(fresh, path)

    assert np.all(np.isfinite(fresh[0].T)) and np.all(np.asarray(fresh[0].T) > 0.0)
    # Interpolation cannot invent values outside the range it was given.
    assert float(fresh[0].T.min()) >= float(grid[0].T.min()) - 1.0
    assert float(fresh[0].T.max()) <= float(grid[0].T.max()) + 1.0


def test_a_field_is_not_reinterpreted_by_a_changed_datum(solved):
    """Why primitives are stored rather than the conserved variables.

    The mesher gives each grid a datum derived from its own design, so two runs
    of different designs do not share one. Conserved energy is measured from
    that datum; pressure and temperature are not.
    """
    config, grid, path = solved
    _, fresh = cli.prepare(config)

    shifted = fresh[0].fluid.change_datum(P_dtm=3.0e5, T_dtm=900.0)
    for block in fresh:
        block.set_fluid(shifted)
    assert fresh[0].fluid.T_dtm != grid[0].fluid.T_dtm

    restart.apply(fresh, path)

    np.testing.assert_allclose(
        np.asarray(fresh[0].T), np.asarray(grid[0].T), rtol=1e-5
    )


def test_the_file_holds_primitives_only(solved):
    """No coordinates, no patches, no conserved variables.

    Coordinates are unnecessary because the mapping is in index space, patches
    because a flow field needs no boundary alignment, and conserved because of
    the datum.
    """
    _, grid, path = solved
    data = np.load(path)

    assert set(data.files) == {f"b0_{name}" for name in restart.STATE}
    assert all(data[key].shape == grid[0].shape for key in data.files)
    assert all(data[key].dtype == np.float32 for key in data.files)


def test_a_field_for_another_machine_is_refused(solved, tmp_path):
    config, grid, path = solved
    _, fresh = cli.prepare(config)

    # A file claiming two blocks, against a one-block grid.
    data = dict(np.load(path))
    data.update({key.replace("b0_", "b1_"): value for key, value in data.items()})
    two_block = tmp_path / "two.npz"
    np.savez_compressed(two_block, **data)

    with pytest.raises(ValueError, match="2 block"):
        restart.apply(fresh, two_block)
