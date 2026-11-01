"""Tests for sweeping a characteristic to its stability limit.

No CFD. The stand-in declares its own limit --- every point below it converges,
every point above it does not --- so the coarse and fine marches can be checked
against an answer that is known exactly, which is not true of any real machine.
`test_iterate.py` drives its loop the same way and for the same reason.

Test cases:
- test_the_first_point_is_one_step_off_design: the design point is already run
- test_it_marches_by_the_step: until something refuses
- test_the_bracket_contains_the_limit: and is narrower than step_fine
- test_it_refines_by_the_fine_step: from the last success, after a refusal
- test_it_stops_at_the_second_refusal: march only, nothing refined after
- test_it_marches_towards_choke: the same steps the other way
- test_step_choke_sets_the_coarse_step_towards_choke: stall keeps its own
- test_step_choke_fine_sets_the_fine_step_towards_choke: stall keeps its own
- test_a_point_that_passes_no_more_flow_is_refused: and the fine march refines
- test_a_fine_step_is_judged_on_its_length: a smaller step, a smaller rise
- test_each_point_starts_from_the_last_that_stood: not from a choked one
- test_a_choke_leg_restarts_from_the_last_that_stood: through the CLI's chain
- test_choke_mdot_zero_marches_until_a_refusal: the flow ignored
- test_a_choke_fine_step_larger_than_its_coarse_is_refused: and a negative
  choke_mdot
- test_a_diverged_point_is_not_the_next_start: what chaining must not do
- test_max_points_stops_a_sweep_that_never_fails: a machine with no limit
- test_a_sweep_needs_a_chic_section: and says so
- test_the_table_reports_every_point: and names what the limit is not
- test_non_positive_fields_are_refused: at read time, before any CFD
- test_a_step_fine_larger_than_step_is_refused: the fine march would be coarse
- test_the_chic_section_round_trips: an ordinary config node
- test_at_replaces_the_operating_point: a point states where it is
- test_a_throttled_design_point_sets_the_datum: where the sweep departs from
- test_the_bracket_is_absolute_not_an_offset_from_the_datum: one scale
- test_at_takes_a_throttle_off: a swept point is a pressure
"""

import dataclasses
from types import SimpleNamespace

import pytest
from test_blade import build

from turbigen import Chic, Config, Result, chic, loop
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

    chic.sweep(config(step_coarse=0.05), runner(log=seen))

    assert seen[0] == pytest.approx(0.05)


def test_a_throttled_design_point_sets_the_datum():
    """A design converged with the throttle on did not sit at `DP_adjust = 0`:
    the controller chose a pressure, and `bconds.achieved` wrote it into the
    config. The sweep departs from there, so the first point is one step off
    the machine that was actually run."""
    seen = []
    throttled = dataclasses.replace(
        config(step_coarse=0.05),
        operating_point=OperatingPoint(DP_adjust=0.05, mdot_adjust=0.0),
    )

    chic.sweep(throttled, runner(limit=0.5, log=seen))

    assert seen[0] == pytest.approx(0.10)


def test_the_bracket_is_absolute_not_an_offset_from_the_datum():
    """One scale for the whole sweep, the design's own, so a bracket names
    pressures that were run rather than departures from a moving origin."""
    throttled = dataclasses.replace(
        config(step_coarse=0.05, step_fine=0.02),
        operating_point=OperatingPoint(DP_adjust=0.05, mdot_adjust=0.0),
    )

    points, (lo, hi) = chic.sweep(throttled, runner())

    # The stand-in refuses above 0.17 on the absolute scale, and the datum
    # itself is the lower bound before anything has converged beyond it.
    assert lo <= LIMIT <= hi
    assert lo >= 0.05
    assert points[0].DP_adjust == pytest.approx(0.10)


def test_it_marches_by_the_step():
    seen = []

    chic.sweep(config(step_coarse=0.05, step_fine=0.02), runner(log=seen))

    # 0.05, 0.10, 0.15 all converge; 0.20 is the first refusal.
    assert seen[:4] == pytest.approx([0.05, 0.10, 0.15, 0.20])


def test_the_bracket_contains_the_limit():
    points, (lo, hi) = chic.sweep(config(step_coarse=0.05, step_fine=0.005), runner())

    assert lo < LIMIT < hi
    assert hi - lo == pytest.approx(0.005)
    # And the bracket really is the last success and first refusal.
    assert all(point.converged for point in points if point.DP_adjust <= lo)
    assert not any(point.converged for point in points if point.DP_adjust >= hi)


def test_it_refines_by_the_fine_step():
    """After the coarse refusal at 0.20, back to 0.15 and on by 0.01."""
    seen = []

    chic.sweep(config(step_coarse=0.05, step_fine=0.01), runner(log=seen))

    assert seen == pytest.approx([0.05, 0.10, 0.15, 0.20, 0.16, 0.17])


def test_it_stops_at_the_second_refusal():
    """Every point run is a step on from a success; nothing narrows the
    bracket from the refused side."""
    points, (lo, hi) = chic.sweep(config(step_coarse=0.05, step_fine=0.01), runner())

    assert [p.converged for p in points] == [True] * 3 + [False] + [True, False]
    assert (lo, hi) == pytest.approx((0.16, 0.17))


def test_a_fine_march_that_reaches_the_coarse_refusal_stops():
    """With the limit just under the coarse refusal, the fine march runs up to
    it and stops, not re-running a point already refused."""
    seen = []

    points, (lo, hi) = chic.sweep(
        config(step_coarse=0.05, step_fine=0.01), runner(limit=0.195, log=seen)
    )

    assert seen == pytest.approx([0.05, 0.10, 0.15, 0.20, 0.16, 0.17, 0.18, 0.19])
    assert (lo, hi) == pytest.approx((0.19, 0.20))


def test_it_marches_towards_choke():
    """Lowering the back pressure from the datum, refusing below -0.12."""
    seen = []

    def run(config_now, _):
        DP_adjust = config_now.operating_point.DP_adjust
        seen.append(DP_adjust)
        return Result(converged=DP_adjust > -0.12)

    points, (lo, hi) = chic.sweep(
        config(step_coarse=0.05, step_fine=0.01), run, direction=-1
    )

    assert seen == pytest.approx([-0.05, -0.10, -0.15, -0.11, -0.12])
    assert (lo, hi) == pytest.approx((-0.11, -0.12))


def choking(limit=-float("inf"), choked_at=-0.3, log=None):
    """Return a `run` whose mass flow rises by 1% for each 0.1 the back
    pressure is lowered, flat beyond `choked_at`, and that refuses below
    `limit`."""

    def run(config_now, _):
        DP_adjust = config_now.operating_point.DP_adjust
        if log is not None:
            log.append(DP_adjust)
        mdot = 30.0 * (1.0 - 0.1 * max(DP_adjust, choked_at))
        actual = SimpleNamespace(outlet=SimpleNamespace(mdot=mdot), PR_ts=1.0)
        return SimpleNamespace(converged=DP_adjust > limit, actual=actual)

    return run


def test_step_choke_sets_the_coarse_step_towards_choke():
    seen_choke, seen_stall = [], []
    spec = config(step_coarse=0.05, step_fine=0.01, step_choke=0.2)

    chic.sweep(spec, choking(limit=-0.5, log=seen_choke), direction=-1)
    chic.sweep(spec, runner(log=seen_stall))

    assert seen_choke[:2] == pytest.approx([-0.2, -0.4])
    assert seen_stall[:2] == pytest.approx([0.05, 0.10])


def test_step_choke_fine_sets_the_fine_step_towards_choke():
    """Refusing below -0.5: the fine march towards choke goes by 0.05 from
    -0.4, where towards stall it goes by `step_fine`."""
    seen_choke, seen_stall = [], []
    spec = config(
        step_coarse=0.05, step_fine=0.01, step_choke=0.2, step_choke_fine=0.05
    )
    run = choking(limit=-0.5, choked_at=-1.0, log=seen_choke)

    chic.sweep(spec, run, direction=-1)
    chic.sweep(spec, runner(log=seen_stall))

    assert seen_choke == pytest.approx([-0.2, -0.4, -0.6, -0.45, -0.5])
    assert seen_stall == pytest.approx([0.05, 0.10, 0.15, 0.20, 0.16, 0.17])


def test_a_point_that_passes_no_more_flow_is_refused():
    """Every point converges. Flat from -0.3, so -0.4 is refused, and the
    fine march from -0.3 is refused at once."""
    seen = []

    points, (lo, hi) = chic.sweep(
        config(step_coarse=0.1, step_fine=0.01), choking(log=seen), direction=-1
    )

    assert seen == pytest.approx([-0.1, -0.2, -0.3, -0.4, -0.31])
    assert all(p.converged for p in points)
    assert [p.stood for p in points] == [True, True, True, False, False]
    assert (lo, hi) == pytest.approx((-0.3, -0.31))


def test_a_fine_step_is_judged_on_its_length():
    """Flat from -0.21, so -0.3 is refused. The fine step to -0.21 raises the
    flow 0.1%, under `choke_mdot` but over a tenth of it, so stands; -0.22,
    flat, does not."""
    seen = []

    points, _ = chic.sweep(
        config(step_coarse=0.1, step_fine=0.01, choke_mdot=0.002),
        choking(choked_at=-0.21, log=seen),
        direction=-1,
    )

    assert seen == pytest.approx([-0.1, -0.2, -0.3, -0.21, -0.22])
    assert [p.stood for p in points] == [True, True, False, True, False]


def test_each_point_starts_from_the_last_that_stood():
    """As `test_a_point_that_passes_no_more_flow_is_refused`: the choked -0.4
    converged, but -0.31 starts from -0.3, the third point."""
    starts = []
    stand_in = choking()

    def run(config_now, start):
        starts.append(start)
        return stand_in(config_now, start)

    chic.sweep(config(step_coarse=0.1, step_fine=0.01), run, direction=-1)

    assert starts == [None, 0, 1, 2, 2]


def test_a_choke_leg_restarts_from_the_last_that_stood(tmp_path, monkeypatch):
    """The same leg run through `loop.characteristic`: each point from the
    restart file of the one the sweep names, the first from the design's."""
    calls = []
    stand_in = choking()

    def solve(config_now, point_dir, restart_path=None, **_):
        calls.append((point_dir, restart_path))
        return stand_in(config_now, None)

    monkeypatch.setattr(loop.pipeline, "solve", solve)
    spec = config(step_coarse=0.1, step_fine=0.01, stall=False, choke=True)

    loop.characteristic(spec, tmp_path, previous="design.npz", design=Result())

    dirs = [c[0] for c in calls]
    restart = [dirs[i] / loop.RESTART_NAME for i in (0, 1, 2, 2)]
    assert [c[1] for c in calls] == ["design.npz", *restart]


def test_choke_mdot_zero_marches_until_a_refusal():
    seen = []

    chic.sweep(
        config(step_coarse=0.1, step_fine=0.1, choke_mdot=0.0),
        choking(limit=-0.55, log=seen),
        direction=-1,
    )

    assert seen == pytest.approx([-0.1, -0.2, -0.3, -0.4, -0.5, -0.6])


def test_a_diverged_point_is_not_the_next_start():
    """The next point after a refusal is a fine step on from what worked,
    so it must start from the field that worked, not from the one that blew up.

    Checked here on the sweep's own contract --- it never asks for a point
    beyond one that refused --- since which field is handed over is the CLI's.
    """
    seen = []

    chic.sweep(config(step_coarse=0.05, step_fine=0.005), runner(log=seen))

    refused = [DP for DP in seen if DP >= LIMIT]
    for after, before in zip(seen[1:], seen):
        if before in refused:
            assert after < before


def test_max_points_stops_a_sweep_that_never_fails():
    points, (lo, hi) = chic.sweep(
        config(step_coarse=0.05, max_points=6), runner(limit=float("inf"))
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


@pytest.mark.parametrize("field", ["step_coarse", "step_fine"])
@pytest.mark.parametrize("value", [0.0, -0.05])
def test_non_positive_fields_are_refused(field, value):
    """A sweep marches towards the limit; DP_adjust already says which way."""
    with pytest.raises(ValueError, match=f"chic.{field} must be positive"):
        Chic(**{field: value})


def test_a_step_fine_larger_than_step_is_refused():
    with pytest.raises(ValueError, match="larger than"):
        Chic(step_coarse=0.01, step_fine=0.05)


def test_a_choke_fine_step_larger_than_its_coarse_is_refused():
    with pytest.raises(ValueError, match="towards choke"):
        Chic(step_fine=0.01, step_choke=0.1, step_choke_fine=0.2)
    with pytest.raises(ValueError, match="towards choke"):
        Chic(step_coarse=0.05, step_fine=0.01, step_choke_fine=0.1)
    with pytest.raises(ValueError, match="chic.step_choke_fine must be positive"):
        Chic(step_choke_fine=0.0)
    with pytest.raises(ValueError, match="choke_mdot"):
        Chic(choke_mdot=-0.001)


def test_max_points_below_one_is_refused():
    with pytest.raises(ValueError, match="at least 1"):
        Chic(max_points=0)


#
# THE OPERATING POINT, AND THE REPORT
#


def test_at_takes_a_throttle_off():
    """Every swept point is a prescribed pressure. The design point's own
    throttle is not lost by this: the pressure it settled at is already this
    config's `DP_adjust`, which is where the sweep starts."""
    throttled = dataclasses.replace(
        build(), operating_point=OperatingPoint(DP_adjust=0.05, mdot_adjust=0.0)
    )

    moved = chic.at(throttled, 0.1)

    assert moved.operating_point.mdot_adjust is None
    assert moved.operating_point.DP_adjust == pytest.approx(0.1)


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
    points, bracket = chic.sweep(config(step_coarse=0.05, step_fine=0.02), runner())

    table = chic.format_table(points, bracket)

    assert table.count("\n") >= len(points)
    assert "DP_adjust" in table
    # The caveat travels with the answer rather than living in a docstring.
    assert "not the surge line" in table


def test_the_table_survives_a_point_that_would_not_mix_out():
    """A diverged point has no mean line, and the table says so rather than
    inventing one."""
    points, bracket = chic.sweep(config(step_coarse=0.05, step_fine=0.02), runner())

    assert "--" in chic.format_table(points, bracket)


def test_the_chic_section_round_trips():
    original = config(step_coarse=0.02)
    assert Config.from_dict(original.to_dict()) == original
