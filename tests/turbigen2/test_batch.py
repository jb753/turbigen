"""Tests for covering a design space with runs.

No CFD. Screening designs a mean line, an annulus and blades, which the shared
fixtures already do in milliseconds, and that is the most expensive thing here.

Two boxes are used throughout. `BOX` is comfortable, so every point designs and
the indices run 0, 1, 2, ...; `WIDE` reaches past what an axial turbine can do
at `Ma2 = 2.5`, so some points are skipped and the indices have gaps. The gaps
are the interesting case, because they are what makes an index different from
a position in the batch. `GRID` and `WIDE_GRID` are their named-value
counterparts.

Test cases:
- test_the_batch_section_round_trips: it is an ordinary config node
- test_a_grid_section_round_trips: values: is a config node too
- test_a_batch_needs_a_batch_section
- test_bounds_are_respected: every value inside its box
- test_the_order_of_the_bounds_does_not_matter: sorted, not as typed
- test_a_seed_reproduces_a_batch: the same file gives the same designs
- test_a_different_seed_gives_different_designs: and the seed is read
- test_from_is_the_tail_of_a_longer_batch: the whole claim for Sobol over LHS
- test_a_grid_runs_every_value_in_order
- test_a_grid_of_two_variables_is_the_full_factorial
- test_the_last_grid_variable_moves_fastest: row-major, as paths() sorts
- test_the_order_of_the_values_does_not_matter: sorted, not as typed
- test_a_grid_needs_no_seed: named points, so nothing is drawn
- test_the_batch_key_does_not_survive: a member is a design, not a space
- test_the_batch_key_does_not_survive_a_grid
- test_nothing_else_about_the_datum_moves: only what the batch names is touched
- test_emitted_configs_design: the screen did its job
- test_an_infeasible_point_is_skipped: a gap in the indices
- test_a_skipped_index_is_never_retried: it would fail the same way
- test_an_impossible_box_fails_loudly: rather than spinning
- test_an_infeasible_grid_point_is_skipped_and_warned: you named it, so it is news
- test_an_impossible_grid_fails_loudly
- test_an_unknown_path_is_refused: before a batch of broken configs is written
- test_an_unknown_grid_path_is_refused
- test_an_integer_bound_is_refused: rounding makes duplicate designs
- test_an_integer_value_is_allowed: naming one cannot
- test_an_empty_range_is_refused
- test_an_empty_list_of_values_is_refused
- test_a_repeated_value_is_refused: it would run the same design twice
- test_a_non_number_value_is_refused
- test_bounds_and_values_together_are_refused: one way or the other
- test_a_batch_that_varies_nothing_is_refused
- test_member_names_are_the_sequence_index
- test_next_index_carries_on_from_the_highest
- test_next_index_of_nothing_is_the_start
"""

import dataclasses

import pytest

from test_blade import build
from turbigen2 import Config, batch

BOX = {"mean_line.psi": [1.4, 1.8], "mean_line.phi2": [0.6, 0.9]}
"""Comfortable: every point in here designs."""

WIDE = {"mean_line.Ma2": [0.3, 2.5]}
"""Reaches past what the design can do, so some points are skipped."""

GRID = {"mean_line.psi": [1.4, 1.6, 1.8]}
"""Named points, all of which design."""

WIDE_GRID = {"mean_line.Ma2": [0.5, 0.7, 25.0]}
"""Named points, one of which is far outside what the design can do."""


def make(bounds=None, values=None, seed=0, **kwargs):
    """Return a config carrying a batch section over `bounds` or `values`."""
    if values is None:
        spec = batch.Batch(bounds=dict(BOX if bounds is None else bounds), seed=seed)
    else:
        spec = batch.Batch(values=dict(values), seed=seed)

    return dataclasses.replace(build(), batch=spec, **kwargs)


def values_of(pairs, path="mean_line.psi"):
    """Return the value of `path` in each emitted design."""
    from turbigen2 import node  # noqa: PLC0415

    return [node.flatten(config)[path] for _, config in pairs]


#
# THE CONFIG NODE
#


def test_the_batch_section_round_trips():
    config = make()

    assert Config.from_dict(config.to_dict()) == config


def test_a_grid_section_round_trips():
    config = make(values=GRID)

    assert Config.from_dict(config.to_dict()) == config


def test_a_batch_needs_a_batch_section():
    with pytest.raises(ValueError, match="needs a batch: section"):
        batch.generate(build(), 4)


#
# THE SEQUENCE
#


def test_bounds_are_respected():
    pairs = batch.generate(make(), 8)

    for path, (lo, hi) in BOX.items():
        assert all(lo <= value <= hi for value in values_of(pairs, path))


def test_the_order_of_the_bounds_does_not_matter():
    """Sorted, so a mapping's insertion order cannot change the design."""
    forward = batch.generate(make(BOX), 4)
    backward = batch.generate(make(dict(reversed(list(BOX.items())))), 4)

    assert values_of(forward) == values_of(backward)


def test_a_seed_reproduces_a_batch():
    assert values_of(batch.generate(make(), 8)) == values_of(batch.generate(make(), 8))


def test_a_different_seed_gives_different_designs():
    assert values_of(batch.generate(make(seed=0), 8)) != values_of(
        batch.generate(make(seed=1), 8)
    )


def test_from_is_the_tail_of_a_longer_batch():
    """The reason this is Sobol and not a Latin hypercube.

    An LHS of 8 is not an LHS of 4 with 4 more: its stratification is defined
    by the batch size, so growing a database means regenerating it and
    discarding runs already paid for. A Sobol prefix is just a prefix.
    """
    whole = batch.generate(make(), 8)
    tail = batch.generate(make(), 4, start=4)

    assert [index for index, _ in tail] == [index for index, _ in whole[4:]]
    assert values_of(tail) == values_of(whole[4:])


#
# THE GRID
#


def test_a_grid_runs_every_value_in_order():
    pairs = batch.generate(make(values=GRID))

    assert [index for index, _ in pairs] == [0, 1, 2]
    assert values_of(pairs) == GRID["mean_line.psi"]


def test_a_grid_of_two_variables_is_the_full_factorial():
    """Every combination, once: the only reading that needs no equal lengths."""
    grid = {"mean_line.psi": [1.4, 1.8], "mean_line.phi2": [0.6, 0.75, 0.9]}
    pairs = batch.generate(make(values=grid))

    assert len(pairs) == 6
    points = set(
        zip(values_of(pairs, "mean_line.psi"), values_of(pairs, "mean_line.phi2"))
    )
    assert points == {
        (psi, phi) for psi in grid["mean_line.psi"] for phi in [0.6, 0.75, 0.9]
    }


def test_the_last_grid_variable_moves_fastest():
    """Row-major over the sorted paths, so the ordering is the file's."""
    grid = {"mean_line.psi": [1.4, 1.8], "mean_line.phi2": [0.6, 0.9]}
    pairs = batch.generate(make(values=grid))

    # Sorted paths put phi2 before psi, so psi is the one that moves fastest.
    assert values_of(pairs, "mean_line.phi2") == [0.6, 0.6, 0.9, 0.9]
    assert values_of(pairs, "mean_line.psi") == [1.4, 1.8, 1.4, 1.8]


def test_the_order_of_the_values_does_not_matter():
    grid = {"mean_line.psi": [1.4, 1.8], "mean_line.phi2": [0.6, 0.9]}
    forward = batch.generate(make(values=grid))
    backward = batch.generate(make(values=dict(reversed(list(grid.items())))))

    assert values_of(forward) == values_of(backward)


def test_a_grid_needs_no_seed():
    """Named points, so nothing is drawn and the seed cannot reach them."""
    assert values_of(batch.generate(make(values=GRID, seed=0))) == values_of(
        batch.generate(make(values=GRID, seed=7))
    )


#
# WHAT AN EMITTED DESIGN IS
#


def test_the_batch_key_does_not_survive():
    """A member is one design, not a design of experiments.

    Left in, batching a member would expand one design into another N.
    """
    _, member = batch.generate(make(), 1)[0]

    assert member.batch is None


def test_the_batch_key_does_not_survive_a_grid():
    _, member = batch.generate(make(values=GRID))[0]

    assert member.batch is None


def test_nothing_else_about_the_datum_moves():
    datum = make()
    _, member = batch.generate(datum, 1)[0]

    assert member.mean_line.Ma2 == datum.mean_line.Ma2
    assert member.blades == datum.blades
    assert member.fluid == datum.fluid


def test_emitted_configs_design():
    """Nothing is written that cannot be built, which is what screening buys."""
    for _, member in batch.generate(make(WIDE), 4):
        member.design()


#
# SCREENING
#


def test_an_infeasible_point_is_skipped():
    indices = [index for index, _ in batch.generate(make(WIDE), 8)]

    assert len(indices) == 8
    # Gaps, so an index is not a position in the batch.
    assert indices != list(range(8))
    assert indices == sorted(indices)


def test_a_skipped_index_is_never_retried():
    """Deterministic given the seed, so re-drawing it would fail identically."""
    first = [index for index, _ in batch.generate(make(WIDE), 4)]
    second = [index for index, _ in batch.generate(make(WIDE), 4, start=first[-1] + 1)]

    assert min(second) > max(first)
    assert not set(first) & set(second)


def test_an_impossible_box_fails_loudly():
    """A box that is mostly outside the design stops, rather than spinning."""
    with pytest.raises(ValueError, match="attempts"):
        batch.generate(make({"mean_line.Ma2": [5.0, 50.0]}), 4)


def test_an_infeasible_grid_point_is_skipped_and_warned(caplog):
    """You named this point, so its absence from the batch is news."""
    pairs = batch.generate(make(values=WIDE_GRID))

    # The index is the position in the product, so the hole is visible.
    assert [index for index, _ in pairs] == [0, 1]
    assert "does not design" in caplog.text
    assert any(record.levelname == "WARNING" for record in caplog.records)


def test_an_impossible_grid_fails_loudly():
    with pytest.raises(ValueError, match="None of the"):
        batch.generate(make(values={"mean_line.Ma2": [25.0, 50.0]}))


#
# WHAT A BATCH MAY NAME
#


def test_an_unknown_path_is_refused():
    """Caught against the datum, before a batch of broken configs is written.

    `set_by_path` builds what a path implies, so a typo would otherwise create
    the key it names and every design in the batch would be rejected by the
    strict unknown-key check, complaining about the wrong thing.
    """
    with pytest.raises(ValueError, match="not a leaf"):
        batch.generate(make({"mean_line.psii": [1.4, 1.8]}), 4)


def test_an_unknown_grid_path_is_refused():
    with pytest.raises(ValueError, match="not a leaf"):
        batch.generate(make(values={"mean_line.psii": [1.4, 1.8]}))


def test_an_integer_bound_is_refused():
    """Rounding a continuous draw makes duplicate designs.

    They then sit on top of each other in `database._predict` and are averaged
    as though they were repeat runs of one design, which they are.
    """
    from turbigen2 import Ember  # noqa: PLC0415

    config = make({"solver.n_step": [10, 100]}, solver=Ember(n_step=10))

    with pytest.raises(ValueError, match="whole number"):
        batch.generate(config, 4)


def test_an_integer_value_is_allowed():
    """Named values cannot collide, so the reason for the ban does not apply."""
    from turbigen2 import Ember, node  # noqa: PLC0415

    config = make(values={"solver.n_step": [10, 100]}, solver=Ember(n_step=10))
    pairs = batch.generate(config)

    assert [node.flatten(c)["solver.n_step"] for _, c in pairs] == [10, 100]


def test_an_empty_range_is_refused():
    with pytest.raises(ValueError, match="empty"):
        batch.generate(make({"mean_line.psi": [1.8, 1.4]}), 4)


def test_an_empty_list_of_values_is_refused():
    with pytest.raises(ValueError, match="non-empty list"):
        batch.generate(make(values={"mean_line.psi": []}))


def test_a_repeated_value_is_refused():
    """It would run the same design twice, for a whole solve and no news."""
    with pytest.raises(ValueError, match="repeats a value"):
        batch.generate(make(values={"mean_line.psi": [1.4, 1.4]}))


def test_a_non_number_value_is_refused():
    with pytest.raises(ValueError, match="not a number"):
        batch.generate(make(values={"mean_line.psi": [1.4, "wide"]}))


def test_bounds_and_values_together_are_refused():
    """One way or the other: a box to fill, or points to run."""
    config = dataclasses.replace(
        build(), batch=batch.Batch(bounds=dict(BOX), values=dict(GRID))
    )

    with pytest.raises(ValueError, match="cannot say both"):
        batch.generate(config)


def test_a_batch_that_varies_nothing_is_refused():
    config = dataclasses.replace(build(), batch=batch.Batch())

    with pytest.raises(ValueError, match="no design variables"):
        batch.generate(config)


#
# NAMING AND CARRYING ON
#


def test_member_names_are_the_sequence_index():
    """A directory each, so a member has somewhere to be run into."""
    assert batch.member_name(0) == "0000/input.yaml"
    assert batch.member_name(1234) == "1234/input.yaml"


def _write_members(batch_dir, indices):
    """Write empty members into `batch_dir`, as the batch verb lays them out."""
    batch_dir.mkdir(parents=True, exist_ok=True)
    for index in indices:
        member = batch_dir / batch.member_name(index)
        member.parent.mkdir(parents=True, exist_ok=True)
        member.write_text("{}\n")


def test_next_index_carries_on_from_the_highest(tmp_path):
    """Read from the names, not counted, so a gap still says where it reached."""
    batch_dir = tmp_path / "batch_0000"
    _write_members(batch_dir, (0, 1, 4))

    assert batch.next_index([batch_dir]) == 5


def test_next_index_ignores_what_is_not_a_member(tmp_path):
    """A log file or a submission script beside the members is not one."""
    batch_dir = tmp_path / "batch_0000"
    _write_members(batch_dir, (0, 1))
    (batch_dir / "log_turbigen2.txt").write_text("")
    (batch_dir / "submit.sh").write_text("")

    assert batch.next_index([batch_dir]) == 2


def test_next_index_across_two_batches(tmp_path):
    first, second = tmp_path / "batch_0000", tmp_path / "batch_0001"
    _write_members(first, (0, 1))
    _write_members(second, (2, 3))

    assert batch.next_index([first, second]) == 4


def test_next_index_of_nothing_is_the_start(tmp_path):
    """So --continue on an empty tree is the same as starting."""
    assert batch.next_index([]) == 0
    assert batch.next_index([tmp_path]) == 0
