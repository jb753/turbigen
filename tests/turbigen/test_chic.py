"""Tests for sweeping a characteristic to its stability limit.

No CFD. The stand-in declares its own limit --- every point below it converges,
every point above it does not --- so the march and the bisection can be checked
against an answer that is known exactly, which is not true of any real machine.
`test_iterate.py` drives its loop the same way and for the same reason.

Test cases:
- test_the_first_point_is_one_step_off_design: the design point is already run
- test_it_marches_by_the_step: until something refuses
- test_the_bracket_contains_the_limit: and is narrower than step_min
- test_it_bisects_rather_than_creeping: the count, against a linear search
- test_a_diverged_point_is_not_the_next_start: what chaining must not do
- test_max_points_stops_a_sweep_that_never_fails: a machine with no limit
- test_a_sweep_needs_a_chic_section: and says so
- test_the_table_reports_every_point: and names what the limit is not
- test_non_positive_fields_are_refused: at read time, before any CFD
- test_a_step_min_larger_than_step_is_refused: nothing would be refined
- test_the_chic_section_round_trips: an ordinary config node
- test_at_replaces_the_operating_point: a point states where it is
"""

import dataclasses

import pytest

from test_blade import build
from turbigen import Chic, Config, Result, chic
from turbigen.bconds import OperatingPoint

LIMIT = 0.17
"""Where the stand-in stops converging. Deliberately not on a step boundary."""


def config(**kwargs):
    """A config carrying a chic: section, and nothing that needs solving."""
    return dataclasses.replace(build(), chic=Chic(**kwargs))


def runner(limit=LIMIT, log=None):
    """Return a `run` that converges below `limit` and refuses above it."""

    def run(config_now, i_point):
        DP_adjust = config_now.operating_point.DP_adjust
        if log is not None:
            log.append(DP_adjust)
        return Result(converged=DP_adjust < limit)

    return run


#
# THE MARCH
#


def test_the_first_point_is_one_step_off_design():
    """`DP_adjust = 0` is the design point, which the caller has just run."""
    seen = []

    chic.sweep(config(step=0.05), runner(log=seen))

    assert seen[0] == pytest.approx(0.05)


def test_it_marches_by_the_step():
    seen = []

    chic.sweep(config(step=0.05, step_min=0.02), runner(log=seen))

    # 0.05, 0.10, 0.15 all converge; 0.20 is the first refusal.
    assert seen[:4] == pytest.approx([0.05, 0.10, 0.15, 0.20])


def test_the_bracket_contains_the_limit():
    points, (lo, hi) = chic.sweep(config(step=0.05, step_min=0.005), runner())

    assert lo < LIMIT < hi
    assert hi - lo <= 0.005
    # And the bracket really is the last success and first refusal.
    assert all(point.converged for point in points if point.DP_adjust <= lo)
    assert not any(point.converged for point in points if point.DP_adjust >= hi)


def test_it_bisects_rather_than_creeping():
    """Refinement is halving, so the cost is logarithmic in the resolution.

    Creeping at `step_min` from the design point would take 0.20/0.002 = 100
    points; the march plus a bisection takes about 4 + log2(0.05/0.002) = 9.
    """
    points, _ = chic.sweep(config(step=0.05, step_min=0.002), runner())

    assert len(points) < 15


def test_a_diverged_point_is_not_the_next_start():
    """The next point after a refusal is a bisection back towards what worked,
    so it must start from the field that worked, not from the one that blew up.

    Checked here on the sweep's own contract --- it never asks for a point
    beyond one that refused --- since which field is handed over is the CLI's.
    """
    seen = []

    chic.sweep(config(step=0.05, step_min=0.005), runner(log=seen))

    refused = [DP for DP in seen if DP >= LIMIT]
    for after, before in zip(seen[1:], seen):
        if before in refused:
            assert after < before


def test_max_points_stops_a_sweep_that_never_fails():
    points, (lo, hi) = chic.sweep(
        config(step=0.05, max_points=6), runner(limit=float("inf"))
    )

    assert len(points) == 6
    assert hi == float("inf")
    assert lo == pytest.approx(0.30)


#
# WHAT IS REFUSED
#


def test_a_sweep_needs_a_chic_section():
    with pytest.raises(ValueError, match="needs a chic: section"):
        chic.sweep(build(), runner())


@pytest.mark.parametrize("field", ["step", "step_min"])
@pytest.mark.parametrize("value", [0.0, -0.05])
def test_non_positive_fields_are_refused(field, value):
    """A sweep marches towards the limit; DP_adjust already says which way."""
    with pytest.raises(ValueError, match=f"chic.{field} must be positive"):
        Chic(**{field: value})


def test_a_step_min_larger_than_step_is_refused():
    with pytest.raises(ValueError, match="larger than"):
        Chic(step=0.01, step_min=0.05)


def test_max_points_below_one_is_refused():
    with pytest.raises(ValueError, match="at least 1"):
        Chic(max_points=0)


#
# THE OPERATING POINT, AND THE REPORT
#


def test_at_replaces_the_operating_point():
    """A point states where it *is*, so a member's archived config reads as an
    operating point rather than as an offset from one."""
    already = dataclasses.replace(
        build(), operating_point=OperatingPoint(DP_adjust=0.3)
    )

    moved = chic.at(already, 0.1)

    assert moved.operating_point.DP_adjust == pytest.approx(0.1)
    # And a config that carried none gets one.
    assert chic.at(build(), 0.1).operating_point.DP_adjust == pytest.approx(0.1)


def test_the_table_reports_every_point():
    points, bracket = chic.sweep(config(step=0.05, step_min=0.02), runner())

    table = chic.format_table(points, bracket)

    assert table.count("\n") >= len(points)
    assert "DP_adjust" in table
    # The caveat travels with the answer rather than living in a docstring.
    assert "not the surge line" in table


def test_the_table_survives_a_point_that_would_not_mix_out():
    """A diverged point has no mean line, and the table says so rather than
    inventing one."""
    points, bracket = chic.sweep(config(step=0.05, step_min=0.02), runner())

    assert "--" in chic.format_table(points, bracket)


def test_the_chic_section_round_trips():
    assert Config.from_dict(config(step=0.02).to_dict()) == config(step=0.02)
