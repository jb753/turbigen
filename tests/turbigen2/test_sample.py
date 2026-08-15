"""Tests for covering a design space with runs.

No CFD. Screening designs a mean line, an annulus and blades, which the shared
fixtures already do in milliseconds, and that is the most expensive thing here.

Two boxes are used throughout. `BOX` is comfortable, so every point designs and
the indices run 0, 1, 2, ...; `WIDE` reaches past what an axial turbine can do
at `Ma2 = 2.5`, so some points are skipped and the indices have gaps. The gaps
are the interesting case, because they are what makes an index different from
a position in the batch.

Test cases:
- test_the_sample_section_round_trips: it is an ordinary config node
- test_bounds_are_respected: every value inside its box
- test_the_order_of_the_bounds_does_not_matter: sorted, not as typed
- test_a_seed_reproduces_a_batch: the same file gives the same designs
- test_a_different_seed_gives_different_designs: and the seed is read
- test_from_is_the_tail_of_a_longer_batch: the whole claim for Sobol over LHS
- test_the_sample_key_does_not_survive: a member is a design, not a space
- test_nothing_else_about_the_datum_moves: only what bounds name is touched
- test_emitted_configs_design: the screen did its job
- test_an_infeasible_point_is_skipped: a gap in the indices
- test_a_skipped_index_is_never_retried: it would fail the same way
- test_an_impossible_box_fails_loudly: rather than spinning
- test_an_unknown_path_is_refused: before a batch of broken configs is written
- test_an_integer_variable_is_refused: rounding makes duplicate designs
- test_an_empty_range_is_refused
- test_sampling_needs_a_sample_section
- test_member_names_are_the_sequence_index
- test_next_index_carries_on_from_the_highest
- test_next_index_of_nothing_is_the_start
"""

import dataclasses

import pytest

from test_blade import build
from turbigen2 import Config, sample

BOX = {"mean_line.psi": [1.4, 1.8], "mean_line.phi2": [0.6, 0.9]}
"""Comfortable: every point in here designs."""

WIDE = {"mean_line.Ma2": [0.3, 2.5]}
"""Reaches past what the design can do, so some points are skipped."""


def make(bounds=None, seed=0, **kwargs):
    """Return a config carrying a sample section over `bounds`."""
    return dataclasses.replace(
        build(),
        sample=sample.Sample(bounds=dict(BOX if bounds is None else bounds), seed=seed),
        **kwargs,
    )


def values(pairs, path="mean_line.psi"):
    """Return the value of `path` in each emitted design."""
    from turbigen2 import node  # noqa: PLC0415

    return [node.flatten(config)[path] for _, config in pairs]


#
# THE CONFIG NODE
#


def test_the_sample_section_round_trips():
    config = make()

    assert Config.from_dict(config.to_dict()) == config


def test_sampling_needs_a_sample_section():
    with pytest.raises(ValueError, match="needs a sample: section"):
        sample.generate(build(), 4)


#
# THE SEQUENCE
#


def test_bounds_are_respected():
    pairs = sample.generate(make(), 8)

    for path, (lo, hi) in BOX.items():
        assert all(lo <= value <= hi for value in values(pairs, path))


def test_the_order_of_the_bounds_does_not_matter():
    """Sorted, so a mapping's insertion order cannot change the design."""
    forward = sample.generate(make(BOX), 4)
    backward = sample.generate(make(dict(reversed(list(BOX.items())))), 4)

    assert values(forward) == values(backward)


def test_a_seed_reproduces_a_batch():
    assert values(sample.generate(make(), 8)) == values(sample.generate(make(), 8))


def test_a_different_seed_gives_different_designs():
    assert values(sample.generate(make(seed=0), 8)) != values(
        sample.generate(make(seed=1), 8)
    )


def test_from_is_the_tail_of_a_longer_batch():
    """The reason this is Sobol and not a Latin hypercube.

    An LHS of 8 is not an LHS of 4 with 4 more: its stratification is defined
    by the batch size, so growing a database means regenerating it and
    discarding runs already paid for. A Sobol prefix is just a prefix.
    """
    whole = sample.generate(make(), 8)
    tail = sample.generate(make(), 4, start=4)

    assert [index for index, _ in tail] == [index for index, _ in whole[4:]]
    assert values(tail) == values(whole[4:])


#
# WHAT AN EMITTED DESIGN IS
#


def test_the_sample_key_does_not_survive():
    """A member is one design, not a design of experiments.

    Left in, sampling a member would expand one design into another N.
    """
    _, member = sample.generate(make(), 1)[0]

    assert member.sample is None


def test_nothing_else_about_the_datum_moves():
    datum = make()
    _, member = sample.generate(datum, 1)[0]

    assert member.mean_line.Ma2 == datum.mean_line.Ma2
    assert member.blades == datum.blades
    assert member.fluid == datum.fluid


def test_emitted_configs_design():
    """Nothing is written that cannot be built, which is what screening buys."""
    for _, member in sample.generate(make(WIDE), 4):
        member.design()


#
# SCREENING
#


def test_an_infeasible_point_is_skipped():
    indices = [index for index, _ in sample.generate(make(WIDE), 8)]

    assert len(indices) == 8
    # Gaps, so an index is not a position in the batch.
    assert indices != list(range(8))
    assert indices == sorted(indices)


def test_a_skipped_index_is_never_retried():
    """Deterministic given the seed, so re-drawing it would fail identically."""
    first = [index for index, _ in sample.generate(make(WIDE), 4)]
    second = [index for index, _ in sample.generate(make(WIDE), 4, start=first[-1] + 1)]

    assert min(second) > max(first)
    assert not set(first) & set(second)


def test_an_impossible_box_fails_loudly():
    """A box that is mostly outside the design stops, rather than spinning."""
    with pytest.raises(ValueError, match="attempts"):
        sample.generate(make({"mean_line.Ma2": [5.0, 50.0]}), 4)


#
# WHAT THE BOUNDS MAY NAME
#


def test_an_unknown_path_is_refused():
    """Caught against the datum, before a batch of broken configs is written.

    `set_by_path` builds what a path implies, so a typo would otherwise create
    the key it names and every design in the batch would be rejected by the
    strict unknown-key check, complaining about the wrong thing.
    """
    with pytest.raises(ValueError, match="not a leaf"):
        sample.generate(make({"mean_line.psii": [1.4, 1.8]}), 4)


def test_an_integer_variable_is_refused():
    """Rounding a continuous draw makes duplicate designs.

    They then sit on top of each other in `database._predict` and are averaged
    as though they were repeat runs of one design, which they are.
    """
    from turbigen2 import Ember  # noqa: PLC0415

    config = make({"solver.n_step": [10, 100]}, solver=Ember(n_step=10))

    with pytest.raises(ValueError, match="whole number"):
        sample.generate(config, 4)


def test_an_empty_range_is_refused():
    with pytest.raises(ValueError, match="empty"):
        sample.generate(make({"mean_line.psi": [1.8, 1.4]}), 4)


#
# NAMING AND CARRYING ON
#


def test_member_names_are_the_sequence_index():
    """A directory each, so a member has somewhere to be run into."""
    assert sample.member_name(0) == "0000/input.yaml"
    assert sample.member_name(1234) == "1234/input.yaml"


def _write_members(batch, indices):
    """Write empty members into `batch`, as the sample verb lays them out."""
    batch.mkdir(parents=True, exist_ok=True)
    for index in indices:
        member = batch / sample.member_name(index)
        member.parent.mkdir(parents=True, exist_ok=True)
        member.write_text("{}\n")


def test_next_index_carries_on_from_the_highest(tmp_path):
    """Read from the names, not counted, so a gap still says where it reached."""
    batch = tmp_path / "batch_0000"
    _write_members(batch, (0, 1, 4))

    assert sample.next_index([batch]) == 5


def test_next_index_ignores_what_is_not_a_member(tmp_path):
    """A log file or a submission script beside the members is not one."""
    batch = tmp_path / "batch_0000"
    _write_members(batch, (0, 1))
    (batch / "log_turbigen2.txt").write_text("")
    (batch / "submit.sh").write_text("")

    assert sample.next_index([batch]) == 2


def test_next_index_across_two_batches(tmp_path):
    first, second = tmp_path / "batch_0000", tmp_path / "batch_0001"
    _write_members(first, (0, 1))
    _write_members(second, (2, 3))

    assert sample.next_index([first, second]) == 4


def test_next_index_of_nothing_is_the_start(tmp_path):
    """So --continue on an empty tree is the same as starting."""
    assert sample.next_index([]) == 0
    assert sample.next_index([tmp_path]) == 0
