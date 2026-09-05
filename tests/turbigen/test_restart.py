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
- test_the_stamp_follows_the_design: what the digest does and does not depend on
- test_an_unstamped_field_reads_back_no_stamp: fields written before stamps
- test_a_stamped_field_still_restarts_onto_a_changed_design: the stamp records
  provenance and gates nothing here, which every chained restart depends on
"""

import numpy as np
import pytest
from test_mixout import CASCADE

from turbigen import cli, restart
from turbigen.config import Config


@pytest.fixture(scope="module")
def solved(tmp_path_factory):
    """A short march, and the field it reached written to a file."""
    config = Config.from_dict(CASCADE)
    _, _machine, grid = cli.prepare(config)
    config.solver.solve(grid)

    path = tmp_path_factory.mktemp("restart") / "restart.npz"
    restart.save(path, grid)
    return config, grid, path


def test_a_saved_field_comes_back_unchanged(solved):
    config, grid, path = solved
    *_, fresh = cli.prepare(config)

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
    _config, grid, path = solved
    finer = Config.from_dict(
        {**CASCADE, "mesh": {**CASCADE["mesh"], "resolution_factor": 0.4}}
    )
    *_, fresh = cli.prepare(finer)
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
    *_, fresh = cli.prepare(config)

    shifted = fresh[0].fluid.change_datum(P_dtm=3.0e5, T_dtm=900.0)
    for block in fresh:
        block.set_fluid(shifted)
    assert fresh[0].fluid.T_dtm != grid[0].fluid.T_dtm

    restart.apply(fresh, path)

    np.testing.assert_allclose(np.asarray(fresh[0].T), np.asarray(grid[0].T), rtol=1e-5)


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
    config, _grid, path = solved
    *_, fresh = cli.prepare(config)

    # A file claiming two blocks, against a one-block grid.
    data = dict(np.load(path))
    data.update({key.replace("b0_", "b1_"): value for key, value in data.items()})
    two_block = tmp_path / "two.npz"
    np.savez_compressed(two_block, **data)

    with pytest.raises(ValueError, match="2 block"):
        restart.apply(fresh, two_block)


#
# PROVENANCE STAMPS
#
# A stamp says which design a field solves. It is recorded here and read by the
# report verb, which will not write an answer down without one that matches.
# Nothing in this module acts on it, and the last test is what keeps it that
# way: every chained restart in turbigen -- iterate, chic, warm_start -- hands
# over a field from a design that has deliberately moved.
#


def test_the_stamp_follows_the_design(solved):
    """It depends on what the field is, not on how it was reached."""
    config, *_ = solved

    assert restart.design_stamp(config) == restart.design_stamp(config)

    moved = Config.from_dict(
        {**CASCADE, "mesh": {**CASCADE["mesh"], "resolution_factor": 0.4}}
    )
    assert restart.design_stamp(moved) != restart.design_stamp(config)

    # The solver decides how the answer was reached, not what it is, so raising
    # the step count must not invalidate a field that is already good.
    marched = Config.from_dict(
        {
            **CASCADE,
            "solver": {**CASCADE["solver"], "n_step": CASCADE["solver"]["n_step"] + 10},
        }
    )
    assert restart.design_stamp(marched) == restart.design_stamp(config)


def test_an_unstamped_field_reads_back_no_stamp(solved):
    """The fixture saves without a config, as every field written before this."""
    *_, path = solved

    assert restart.read_stamp(path) is None


def test_a_stamped_field_carries_its_design(solved, tmp_path):
    config, grid, _ = solved
    path = tmp_path / "stamped.npz"

    restart.save(path, grid, config)

    assert restart.read_stamp(path) == restart.design_stamp(config)


def test_a_stamped_field_still_restarts_onto_a_changed_design(solved, tmp_path):
    """The stamp must never gate application, or iterate could not chain.

    Every iteration starts from the field the last one reached, and the whole
    point of an iteration is that the design moved in between. The same goes
    for a characteristic walking its operating point along, and for a warm
    start from a neighbour's answer. A mismatch here is the normal case.
    """
    config, grid, _ = solved
    path = tmp_path / "stamped.npz"
    restart.save(path, grid, config)

    finer = Config.from_dict(
        {**CASCADE, "mesh": {**CASCADE["mesh"], "resolution_factor": 0.4}}
    )
    *_, fresh = cli.prepare(finer)
    assert restart.design_stamp(finer) != restart.read_stamp(path)

    restart.apply(fresh, path)

    assert np.all(np.isfinite(fresh[0].T)) and np.all(np.asarray(fresh[0].T) > 0.0)


def test_the_stamp_is_not_counted_as_a_block(solved, tmp_path):
    """A one-block field with a stamp in it is still a one-block field."""
    config, grid, _ = solved
    path = tmp_path / "stamped.npz"
    restart.save(path, grid, config)

    *_, fresh = cli.prepare(config)

    # Would raise "has 2 block(s)" if the stamp key were counted as one.
    restart.apply(fresh, path)
