"""Tests for design iteration.

No CFD anywhere here. Everything is a pure function of a config and a result,
so the stand-ins declare the answer: `Fixed` and `Coupled` state their own error
as a function of their own knobs, and a run that solves nothing drives the loop.
A march would add minutes and a noise floor while testing nothing extra.

Test cases:
- test_the_iterate_section_round_trips: iterators are ordinary config nodes
- test_unknowns_round_trip: what with_unknowns sets is what unknowns reads
- test_setting_touches_nothing_else: an iterator writes only its own fields
- test_recamber_shift_keeps_the_spanwise_distribution: the knob is a row mean
- test_order_of_application_does_not_matter: knobs are disjoint, so they commute
- test_two_iterators_claiming_one_knob_is_refused: caught at assembly
- test_step_subtracts_the_error: the rule, exactly
- test_step_clips: a bad early step cannot throw the design
- test_step_leaves_unmeasured_knobs_alone: a failed run is not a reason to move
- test_unmeasured_knobs_are_not_converged: nor a reason to stop
- test_converge_reaches_the_answer: the loop, on an analytic error
- test_converge_gives_up: and stops when it cannot
- test_no_history_is_the_declared_gain: a first iteration is what it always was
- test_a_coupled_system_converges_faster: the claim of Broyden, on a stage-like
  lower-triangular Jacobian
- test_a_move_too_small_to_learn_from_is_ignored: a secant on noise is refused
- test_a_flat_response_stays_within_the_clip: a plateau cannot make a wild step
- test_the_history_holds_no_grids: the loop remembers numbers, not fields
- test_deviation_error_is_zero_for_a_machine_that_matches: the measurement datum
- test_mean_line_error_is_zero_for_its_own_design: likewise, through backward()
- test_mean_line_tolerance_scales_with_the_nominal: relative, per variable
- test_mean_line_restores_a_scalar_as_a_scalar: shapes survive a round trip
"""

import dataclasses

import numpy as np
import pytest

from test_blade import build
from turbigen2 import Config, Result, iterate


@pytest.fixture
def config():
    """A two-row config with blades, iterating both recambers."""
    return dataclasses.replace(
        build(),
        iterate=(iterate.Deviation(), iterate.Incidence()),
    )


class Fixed(iterate.Iterator):
    """A stand-in whose error is an analytic function of its own knob.

    No `type`, so it stays out of the registry: it is built by hand here and
    never read from a file.
    """

    target: float = 3.0
    slope: float = 1.0
    name: str = "toy"

    def unknowns(self, config):
        return {self.name: float(config.mean_line.psi)}

    def with_unknowns(self, config, values):
        return dataclasses.replace(
            config,
            mean_line=dataclasses.replace(config.mean_line, psi=values[self.name]),
        )

    def error(self, config, result):
        return {self.name: self.slope * (config.mean_line.psi - self.target)}


class Coupled(iterate.Iterator):
    """Two knobs whose errors are an affine function of both.

    Lower-triangular by default, which is the structure a row feeding the next
    produces: moving the first knob moves the second's error nearly as much as
    its own, and not the other way about. The diagonal is not unity either,
    because the declared gains are only ever a guess at it.
    """

    jacobian: tuple = ((1.5, 0.0), (0.8, 0.6))
    target: tuple = (3.0, 1.0)
    gain: float = 1.0
    tolerance: float = 1e-3

    def unknowns(self, config):
        return {
            "a": float(config.mean_line.psi),
            "b": float(config.mean_line.phi2),
        }

    def with_unknowns(self, config, values):
        return dataclasses.replace(
            config,
            mean_line=dataclasses.replace(
                config.mean_line,
                psi=values.get("a", config.mean_line.psi),
                phi2=values.get("b", config.mean_line.phi2),
            ),
        )

    def error(self, config, result):
        offset = np.array([config.mean_line.psi, config.mean_line.phi2]) - np.asarray(
            self.target
        )
        error = np.asarray(self.jacobian) @ offset
        return {"a": float(error[0]), "b": float(error[1])}


def drive(config, remember, max_iter=50):
    """Iterate to convergence, with or without a history, and count the passes.

    Both routes go through the same `step`, so the comparison is of the rule
    alone: handed no history it is the fixed-gain rule this replaces.
    """
    history = []

    for i_iter in range(max_iter):
        result = Result()
        if iterate.converged(config, result):
            return i_iter, config

        stepped = iterate.step(config, result, history if remember else ())
        history.append(
            (iterate.unknowns(config), iterate.measured_errors(config, result))
        )
        config = stepped

    return max_iter, config


#
# THE PROTOCOL
#


def test_the_iterate_section_round_trips(config):
    assert Config.from_dict(config.to_dict()) == config


def test_unknowns_round_trip(config):
    moved = {"dchi_TE[0]": -3.0, "dchi_TE[1]": 1.5}

    after = config.iterate[0].with_unknowns(config, moved)

    assert config.iterate[0].unknowns(after) == pytest.approx(moved)


def test_setting_touches_nothing_else(config):
    """An iterator owns its fields and writes only those."""
    before = config.to_dict()

    after = config.iterate[0].with_unknowns(config, {"dchi_TE[0]": -3.0}).to_dict()

    for i_row, (blade_before, blade_after) in enumerate(
        zip(before["blades"], after["blades"])
    ):
        for section_before, section_after in zip(
            blade_before["sections"], blade_after["sections"]
        ):
            changed = {
                key
                for key in section_before
                if section_before[key] != section_after[key]
            }
            assert changed == ({"dchi_TE"} if i_row == 0 else set())

    assert after["mean_line"] == before["mean_line"]
    assert after["annulus"] == before["annulus"]


def test_recamber_shift_keeps_the_spanwise_distribution(config):
    """The knob is a row mean; how it varies over the span is a design choice."""
    before = [s.dchi_TE for s in config.blades[0].sections]

    after = config.iterate[0].with_unknowns(config, {"dchi_TE[0]": -3.0})

    shifted = [s.dchi_TE for s in after.blades[0].sections]
    assert np.ptp(np.array(shifted) - np.array(before)) == pytest.approx(0.0)


def test_order_of_application_does_not_matter(config):
    deviation, incidence = config.iterate

    one = incidence.with_unknowns(
        deviation.with_unknowns(config, {"dchi_TE[0]": -3.0}), {"dchi_LE[1]": 4.0}
    )
    other = deviation.with_unknowns(
        incidence.with_unknowns(config, {"dchi_LE[1]": 4.0}), {"dchi_TE[0]": -3.0}
    )

    assert one == other


def test_two_iterators_claiming_one_knob_is_refused(config):
    doubled = dataclasses.replace(
        config, iterate=(iterate.Deviation(), iterate.Deviation())
    )

    with pytest.raises(ValueError, match="both claim"):
        iterate.unknowns(doubled)


#
# THE STEPPER
#


def test_step_subtracts_the_error():
    config = dataclasses.replace(build(), iterate=(Fixed(slope=1.0, target=3.0),))
    psi = config.mean_line.psi

    stepped = iterate.step(config, Result())

    assert stepped.mean_line.psi == pytest.approx(psi - (psi - 3.0))


def test_step_clips():
    """A big early error cannot throw the design further than the clip."""
    config = dataclasses.replace(
        build(), iterate=(Fixed(slope=100.0, target=3.0, clip=0.1),)
    )
    psi = config.mean_line.psi

    stepped = iterate.step(config, Result())

    # psi starts below the target, so the error is negative and the step up is
    # what the clip has to hold: without it the knob would move by 140.
    assert stepped.mean_line.psi == pytest.approx(psi + 0.1)


def test_step_leaves_unmeasured_knobs_alone(config):
    """A run with nothing to measure is not a reason to move the design."""
    empty = Result()

    assert iterate.step(config, empty) == config


def test_unmeasured_knobs_are_not_converged(config):
    """Nor a reason to stop: silence is not agreement."""
    assert iterate.converged(config, Result()) is False


def test_converge_reaches_the_answer():
    config = dataclasses.replace(
        build(), iterate=(Fixed(slope=1.0, target=3.0, gain=0.5, tolerance=1e-3),)
    )
    seen = []

    def run(config_now, i_iter):
        seen.append(float(config_now.mean_line.psi))
        return Result()

    final, _, converged = iterate.converge(config, run, max_iter=50)

    assert converged
    assert final.mean_line.psi == pytest.approx(3.0, abs=1e-3)
    # A gain of 0.5 against a true slope of 1 would halve the error each pass,
    # which from psi = 1.6 is about ten of them. The secant learns the slope
    # from the first move and steps straight to the root.
    assert len(seen) <= 4


def test_converge_gives_up():
    config = dataclasses.replace(
        build(), iterate=(Fixed(slope=1.0, target=3.0, gain=0.0),)
    )

    _, _, converged = iterate.converge(config, lambda c, i: Result(), max_iter=3)

    assert not converged


#
# BROYDEN
#


def test_no_history_is_the_declared_gain(config):
    """The first iteration of every run must be what it always was."""
    machine = config.design()
    result = Result(machine=machine, actual=machine.mean_line, error={})

    for gain in (1.0, -1.0, 0.5):
        one = dataclasses.replace(build(), iterate=(Fixed(gain=gain, slope=1.0),))
        psi = one.mean_line.psi
        error = one.iterate[0].error(one, result)["toy"]

        stepped = iterate.step(one, result)

        assert stepped.mean_line.psi == pytest.approx(psi - gain * error)


def test_a_coupled_system_converges_faster(config):
    """The claim of the whole change, on the structure a stage produces."""
    coupled = dataclasses.replace(build(), iterate=(Coupled(),))

    without, _ = drive(coupled, remember=False)
    with_history, final = drive(coupled, remember=True)

    assert with_history < without
    assert with_history <= 5
    assert final.mean_line.psi == pytest.approx(3.0, abs=1e-3)
    assert final.mean_line.phi2 == pytest.approx(1.0, abs=1e-3)


def test_a_move_too_small_to_learn_from_is_ignored():
    """Below the threshold a secant reports noise, so the prior stands."""
    config = dataclasses.replace(build(), iterate=(Fixed(slope=1.0, target=3.0),))
    result = Result()

    values = iterate.unknowns(config)
    errs = iterate.measured_errors(config, result)
    # A knob that has barely moved, with an error that has: exactly the
    # arrangement that would infer an enormous slope.
    negligible = [
        (
            {"toy": values["toy"] - 1e-6},
            {"toy": errs["toy"] - 1.0},
        )
    ]

    assert iterate.step(config, result, negligible) == iterate.step(config, result)


def test_a_flat_response_stays_within_the_clip():
    """A plateau makes the Jacobian singular, which must not make a wild step."""
    config = dataclasses.replace(
        build(), iterate=(Fixed(slope=1.0, target=3.0, clip=0.2),)
    )
    result = Result()

    values = iterate.unknowns(config)
    errs = iterate.measured_errors(config, result)
    # The knob moved a long way and the error did not move at all.
    flat = [({"toy": values["toy"] - 2.0}, {"toy": errs["toy"]})]

    stepped = iterate.step(config, result, flat)

    change = stepped.mean_line.psi - config.mean_line.psi
    assert np.isfinite(change)
    assert abs(change) <= 0.2 + 1e-9


def test_the_history_holds_no_grids():
    """A Result pins a live grid; keeping one per iteration would pin them all.

    Asserted by weak reference rather than by inspection, because the failure
    would otherwise be silent until a large machine ran out of memory.
    """
    import gc  # noqa: PLC0415
    import weakref  # noqa: PLC0415

    class Field:
        """Stand-in for the megabytes an ember Grid holds."""

    config = dataclasses.replace(
        build(), iterate=(Fixed(slope=1.0, target=3.0, gain=0.5),)
    )
    fields = []

    def run(config_now, i_iter):
        field = Field()
        fields.append(weakref.ref(field))
        return Result(grid=field)

    _, result, _ = iterate.converge(config, run, max_iter=4)

    del result
    gc.collect()

    assert [ref() for ref in fields] == [None] * len(fields)


#
# THE MEASUREMENTS
#


def test_deviation_error_is_zero_for_a_machine_that_matches(config):
    """The datum of the whole scheme: no mismatch, no correction."""
    machine = config.design()
    result = Result(machine=machine, actual=machine.mean_line)

    error = config.iterate[0].error(config, result)

    assert error == pytest.approx({"dchi_TE[0]": 0.0, "dchi_TE[1]": 0.0})


def test_mean_line_error_is_zero_for_its_own_design():
    """Measured through backward(), so this also pins the design round trip."""
    config = dataclasses.replace(
        build(), iterate=(iterate.MeanLine(variables=("psi", "Ys")),)
    )
    machine = config.design()
    result = Result(machine=machine, actual=machine.mean_line)

    error = config.iterate[0].error(config, result)

    assert set(error) == {"mean_line.psi", "mean_line.Ys[0]", "mean_line.Ys[1]"}
    assert error["mean_line.psi"] == pytest.approx(0.0, abs=1e-3)


def test_mean_line_tolerance_scales_with_the_nominal():
    """One absolute number cannot serve a loss coefficient and a loading."""
    config = dataclasses.replace(
        build(), iterate=(iterate.MeanLine(variables=("psi", "Ys"), tolerance=0.01),)
    )

    tolerances = config.iterate[0].tolerances(config)

    assert tolerances["mean_line.psi"] == pytest.approx(0.01 * config.mean_line.psi)
    assert tolerances["mean_line.Ys[0]"] == pytest.approx(0.01 * config.mean_line.Ys[0])


def test_mean_line_restores_a_scalar_as_a_scalar():
    config = dataclasses.replace(
        build(), iterate=(iterate.MeanLine(variables=("psi", "Ys")),)
    )

    moved = config.iterate[0].with_unknowns(
        config, {"mean_line.psi": 1.9, "mean_line.Ys[1]": 0.06}
    )

    assert moved.mean_line.psi == pytest.approx(1.9)
    assert len(moved.mean_line.Ys) == 2
    assert moved.mean_line.Ys[1] == pytest.approx(0.06)
    # Round-tripping through a file is what would catch a stray array here.
    assert Config.from_dict(moved.to_dict()) == moved
