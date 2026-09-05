"""Tests for design iteration.

No CFD anywhere here. Everything is a pure function of a config and a result,
so the stand-ins declare the answer: `Fixed` and `Coupled` state their own error
as a function of their own knobs, and a run that solves nothing drives the loop.
A march would add minutes and a noise floor while testing nothing extra.

Test cases:
- test_the_iterate_section_round_trips: iterators are ordinary config nodes
- test_unknowns_round_trip: what with_unknowns sets is what unknowns reads
- test_setting_touches_nothing_else: an iterator writes only its own fields
- test_recamber_shift_keeps_the_spanwise_distribution: a deviation knob is a row mean
- test_incidence_moves_each_section_independently: an incidence knob is per section
- test_order_of_application_does_not_matter: knobs are disjoint, so they commute
- test_two_iterators_claiming_one_knob_is_refused: caught at assembly
- test_paths_match_what_is_moved: declared ownership is real ownership
- test_paths_are_real_leaves: and every path names a leaf that exists
- test_step_subtracts_the_error: the rule, exactly
- test_step_clips: a bad early step cannot throw the design
- test_step_leaves_unmeasured_knobs_alone: a failed run is not a reason to move
- test_unmeasured_knobs_are_not_converged: nor a reason to stop
- test_converge_reaches_the_answer: the loop, on an analytic error
- test_converge_stops_on_a_diverged_march: a blown-up run measures nothing
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

from test_blade import FLUID, MEAN_LINE, blade, build
import turbigen.loading
from turbigen import Config, Result, iterate, node


@pytest.fixture
def config():
    """A two-row config with blades, iterating both recambers."""
    return dataclasses.replace(
        build(),
        iterate=iterate.Iteration(correct=(iterate.Deviation(), iterate.Incidence())),
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

    after = config.iterate.correct[0].with_unknowns(config, moved)

    assert config.iterate.correct[0].unknowns(after) == pytest.approx(moved)


def test_setting_touches_nothing_else(config):
    """An iterator owns its fields and writes only those."""
    before = config.to_dict()

    after = config.iterate.correct[0].with_unknowns(config, {"dchi_TE[0]": -3.0}).to_dict()

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
    """A deviation knob is a row mean; how it varies over the span is a design
    choice, so a move shifts every section by the same amount."""
    before = [s.dchi_TE for s in config.blades[0].sections]

    after = config.iterate.correct[0].with_unknowns(config, {"dchi_TE[0]": -3.0})

    shifted = [s.dchi_TE for s in after.blades[0].sections]
    assert np.ptp(np.array(shifted) - np.array(before)) == pytest.approx(0.0)


def test_incidence_moves_each_section_independently(config):
    """An incidence knob is one section's leading edge, set outright: the
    spanwise distribution is what the iterator is there to change."""
    incidence = config.iterate.correct[1]
    before = [s.dchi_LE for s in config.blades[0].sections]

    after = incidence.with_unknowns(
        config, {"dchi_LE[0][0]": 1.0, "dchi_LE[0][2]": -1.0}
    )

    moved = [s.dchi_LE for s in after.blades[0].sections]
    assert moved[0] == pytest.approx(1.0)
    assert moved[1] == pytest.approx(before[1])
    assert moved[2] == pytest.approx(-1.0)
    assert np.ptp(np.array(moved) - np.array(before)) > 1.0


def test_incidence_has_one_unknown_per_section(config):
    """Two rows of three sections, so six knobs, named row then section."""
    incidence = config.iterate.correct[1]

    assert set(incidence.unknowns(config)) == {
        f"dchi_LE[{i}][{j}]" for i in range(2) for j in range(3)
    }


def test_order_of_application_does_not_matter(config):
    deviation, incidence = config.iterate.correct

    one = incidence.with_unknowns(
        deviation.with_unknowns(config, {"dchi_TE[0]": -3.0}), {"dchi_LE[1][0]": 4.0}
    )
    other = deviation.with_unknowns(
        incidence.with_unknowns(config, {"dchi_LE[1][0]": 4.0}), {"dchi_TE[0]": -3.0}
    )

    assert one == other


def test_two_iterators_claiming_one_knob_is_refused(config):
    doubled = dataclasses.replace(
        config,
        iterate=iterate.Iteration(correct=(iterate.Deviation(), iterate.Deviation())),
    )

    with pytest.raises(ValueError, match="both claim"):
        iterate.unknowns(doubled)


def _probe(iterator, config):
    """Return the config leaves `with_unknowns` actually writes."""
    before = node.flatten(config)
    shifted = {name: value + 1.0 for name, value in iterator.unknowns(config).items()}
    after = node.flatten(iterator.with_unknowns(config, shifted))
    return {path for path, value in before.items() if after.get(path) != value}


def test_paths_match_what_is_moved(config):
    """What an iterator declares it owns is what it writes.

    The two are separate methods because a knob is a reduction --- one number
    per row, spread over its sections --- so neither naming can be derived from
    the other. That leaves them free to disagree, which is what this refuses:
    a knob whose leaves went unnamed would be read as a design variable, and
    `database` would use the recamber it is predicting as an input.
    """
    variables = ("psi", "Ys")
    config = dataclasses.replace(
        config,
        iterate=dataclasses.replace(
            config.iterate,
            correct=config.iterate.correct
            + (
                iterate.MeanLine(variables=variables),
                # The hardest case for this: its knob is log(mu) while its leaf
                # is mu, so the two namings are not even in the same units.
                iterate.SurfaceReynolds(target=4e5),
            ),
        ),
    )

    for iterator in config.iterate.correct:
        assert iterator.paths(config) == _probe(iterator, config)


def test_paths_are_real_leaves(config):
    """A misspelled path would silently exclude nothing at all."""
    leaves = set(node.flatten(config))

    for iterator in config.iterate.correct:
        assert iterator.paths(config) <= leaves


#
# THE STEPPER
#


def test_step_subtracts_the_error():
    config = dataclasses.replace(
        build(), iterate=iterate.Iteration(correct=(Fixed(slope=1.0, target=3.0),))
    )
    psi = config.mean_line.psi

    stepped = iterate.step(config, Result())

    assert stepped.mean_line.psi == pytest.approx(psi - (psi - 3.0))


def test_step_clips():
    """A big early error cannot throw the design further than the clip."""
    config = dataclasses.replace(
        build(),
        iterate=iterate.Iteration(correct=(Fixed(slope=100.0, target=3.0, clip=0.1),)),
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
        build(),
        iterate=iterate.Iteration(
            correct=(Fixed(slope=1.0, target=3.0, gain=0.5, tolerance=1e-3),)
        ),
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


def test_converge_stops_on_a_diverged_march():
    """A blown-up march measures nothing, so its numbers must not be stepped on.

    Its mixed-out mean line is whatever the NaNs averaged to, and correcting
    towards that would move the design somewhere arbitrary and call it an
    iteration.
    """
    config = dataclasses.replace(
        build(), iterate=iterate.Iteration(correct=(Fixed(slope=1.0, target=3.0),))
    )
    seen = []

    def run(config_now, i_iter):
        seen.append(i_iter)
        # A history is what says a march happened at all; without one there is
        # nothing to have diverged.
        return Result(history=object(), converged=False)

    _, _, converged = iterate.converge(config, run, max_iter=5)

    assert not converged
    assert seen == [0]


def test_converge_gives_up():
    config = dataclasses.replace(
        build(),
        iterate=iterate.Iteration(correct=(Fixed(slope=1.0, target=3.0, gain=0.0),)),
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
        one = dataclasses.replace(
            build(), iterate=iterate.Iteration(correct=(Fixed(gain=gain, slope=1.0),))
        )
        psi = one.mean_line.psi
        error = one.iterate.correct[0].error(one, result)["toy"]

        stepped = iterate.step(one, result)

        assert stepped.mean_line.psi == pytest.approx(psi - gain * error)


def test_a_coupled_system_converges_faster(config):
    """The claim of the whole change, on the structure a stage produces."""
    coupled = dataclasses.replace(
        build(), iterate=iterate.Iteration(correct=(Coupled(),))
    )

    without, _ = drive(coupled, remember=False)
    with_history, final = drive(coupled, remember=True)

    assert with_history < without
    assert with_history <= 5
    assert final.mean_line.psi == pytest.approx(3.0, abs=1e-3)
    assert final.mean_line.phi2 == pytest.approx(1.0, abs=1e-3)


def test_a_move_too_small_to_learn_from_is_ignored():
    """Below the threshold a secant reports noise, so the prior stands."""
    config = dataclasses.replace(
        build(), iterate=iterate.Iteration(correct=(Fixed(slope=1.0, target=3.0),))
    )
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
        build(),
        iterate=iterate.Iteration(correct=(Fixed(slope=1.0, target=3.0, clip=0.2),)),
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
        build(),
        iterate=iterate.Iteration(correct=(Fixed(slope=1.0, target=3.0, gain=0.5),)),
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

    error = config.iterate.correct[0].error(config, result)

    assert error == pytest.approx({"dchi_TE[0]": 0.0, "dchi_TE[1]": 0.0})


def test_mean_line_error_is_zero_for_its_own_design():
    """Measured through backward(), so this also pins the design round trip."""
    config = dataclasses.replace(
        build(),
        iterate=iterate.Iteration(correct=(iterate.MeanLine(variables=("psi", "Ys")),)),
    )
    machine = config.design()
    result = Result(machine=machine, actual=machine.mean_line)

    error = config.iterate.correct[0].error(config, result)

    assert set(error) == {"mean_line.psi", "mean_line.Ys[0]", "mean_line.Ys[1]"}
    assert error["mean_line.psi"] == pytest.approx(0.0, abs=1e-3)


def test_mean_line_tolerance_scales_with_the_nominal():
    """One absolute number cannot serve a loss coefficient and a loading."""
    config = dataclasses.replace(
        build(),
        iterate=iterate.Iteration(
            correct=(iterate.MeanLine(variables=("psi", "Ys"), tolerance=0.01),)
        ),
    )

    tolerances = config.iterate.correct[0].tolerances(config)

    assert tolerances["mean_line.psi"] == pytest.approx(0.01 * config.mean_line.psi)
    assert tolerances["mean_line.Ys[0]"] == pytest.approx(0.01 * config.mean_line.Ys[0])


def test_mean_line_restores_a_scalar_as_a_scalar():
    config = dataclasses.replace(
        build(),
        iterate=iterate.Iteration(correct=(iterate.MeanLine(variables=("psi", "Ys")),)),
    )

    moved = config.iterate.correct[0].with_unknowns(
        config, {"mean_line.psi": 1.9, "mean_line.Ys[1]": 0.06}
    )

    assert moved.mean_line.psi == pytest.approx(1.9)
    assert len(moved.mean_line.Ys) == 2
    assert moved.mean_line.Ys[1] == pytest.approx(0.06)
    # Round-tripping through a file is what would catch a stray array here.
    assert Config.from_dict(moved.to_dict()) == moved


#
# TWO SPEEDS OF ITERATION
#
# `Re_surf` is an iterate like any other -- a knob, an error, a target -- that
# happens to close without CFD. So it converges inside every pass rather than
# across them, and the solution iterators never see it.
#


def with_Re(target=4e5, **kwargs):
    """A bladed two-row config asking for a surface Reynolds number."""
    return dataclasses.replace(
        build(),
        iterate=iterate.Iteration(
            correct=(iterate.SurfaceReynolds(target=target, **kwargs),)
        ),
    )


def test_resolve_reaches_the_target():
    config = with_Re(target=4e5)

    resolved = iterate.resolve(config)

    Re_surf = resolved.design().Re_surf()
    assert Re_surf[0] == pytest.approx(4e5, rel=1e-6)


def test_resolve_moves_the_viscosity_and_nothing_else():
    """The knob is log(mu), but the leaf that moves is mu."""
    config = with_Re()

    resolved = iterate.resolve(config)

    before, after = node.flatten(config), node.flatten(resolved)
    moved = {path for path, value in before.items() if after.get(path) != value}
    assert moved == {"fluid.mu"}
    assert resolved.fluid.mu != config.fluid.mu


def test_resolve_is_exact_in_one_move():
    """Re_surf is exactly proportional to 1/mu at fixed geometry, so in the log
    the residual is linear with unit slope and gain=-1 is the Newton step.

    Asserted by driving the same design from a viscosity two orders out and
    checking it still lands: an approximate step would take many passes from
    there, or overshoot.
    """
    config = with_Re(target=4e5)
    far = dataclasses.replace(
        config, fluid=dataclasses.replace(config.fluid, mu=1.8e-3)
    )

    resolved = iterate.resolve(far, max_iter=2)

    assert resolved.design().Re_surf()[0] == pytest.approx(4e5, rel=1e-6)


def test_resolve_selects_the_row():
    """One viscosity cannot place two Reynolds numbers, so i_row says which."""
    first = iterate.resolve(with_Re(i_row=0)).design().Re_surf()
    second = iterate.resolve(with_Re(i_row=1)).design().Re_surf()

    assert first[0] == pytest.approx(4e5, rel=1e-6)
    assert second[1] == pytest.approx(4e5, rel=1e-6)
    # The other row follows from the design rather than being placed too.
    assert second[0] != pytest.approx(4e5, rel=1e-3)


def test_resolve_without_a_design_only_iterator_is_the_identity():
    config = dataclasses.replace(
        build(), iterate=iterate.Iteration(correct=(iterate.Deviation(),))
    )

    assert iterate.resolve(config) is config


def test_resolve_keeps_the_whole_iterate_section():
    """It steps a subset, but what comes back must still carry the iterators
    the outer loop is about to need."""
    config = dataclasses.replace(
        build(),
        iterate=iterate.Iteration(
            correct=(iterate.SurfaceReynolds(target=4e5), iterate.Deviation())
        ),
    )

    resolved = iterate.resolve(config)

    assert len(resolved.iterate.correct) == 2
    assert Config.from_dict(resolved.to_dict()) == resolved


def test_resolve_reports_a_target_it_cannot_reach():
    stuck = with_Re(target=4e5, gain=0.0)

    with pytest.raises(ValueError, match="did not converge"):
        iterate.resolve(stuck, max_iter=3)


def test_the_outer_loop_does_not_step_a_design_only_knob():
    """The guard that fails silently if it is wrong.

    A resolved knob has ~0 error while its *value* has moved, because `resolve`
    moved it inside the run. That is a zero slope, and feeding it to the
    Broyden update spends a least-change correction explaining a knob that
    needs none -- at the expense of the ones that do.
    """
    config = dataclasses.replace(
        build(),
        iterate=iterate.Iteration(
            correct=(iterate.SurfaceReynolds(target=4e5), Fixed(target=3.0))
        ),
    )

    stepping = iterate.selected(config, from_solution=True)

    assert set(iterate.unknowns(stepping)) == {"toy"}
    assert "fluid.log_mu" not in iterate.unknowns(stepping)
    # And the other way round, so the split is a partition rather than a filter
    # that could drop an iterator entirely.
    assert set(iterate.unknowns(iterate.selected(config, from_solution=False))) == {
        "fluid.log_mu"
    }


def test_a_design_only_error_is_still_recorded():
    """Stepped by one loop, but observed by both.

    `errors` is the record a run writes into `result.error` for the archive, so
    the Reynolds number a design achieved belongs in it whether or not anything
    stepped towards it.
    """
    config = with_Re()
    result = Result(machine=config.design())

    assert "fluid.log_mu" in iterate.errors(config, result)


def test_the_outer_loop_leaves_the_viscosity_alone():
    """End to end on the loop itself: a run whose design-only knob is already
    resolved must come back with it untouched."""
    config = dataclasses.replace(
        iterate.resolve(with_Re()),
        iterate=iterate.Iteration(
            correct=(iterate.SurfaceReynolds(target=4e5), Fixed(target=3.0, gain=0.5))
        ),
    )
    mu_resolved = config.fluid.mu

    final, _, converged = iterate.converge(
        config, lambda c, i: Result(machine=c.design()), max_iter=20
    )

    assert converged
    assert final.fluid.mu == pytest.approx(mu_resolved)


def test_re_surf_survives_a_design_that_moves_under_it():
    """Why this is resolved inside every pass rather than once before them.

    Recambering a blade changes its surface length, so a viscosity fixed by a
    single pre-pass would drift off target for every iteration after the first.
    """
    config = iterate.resolve(with_Re())
    assert config.design().Re_surf()[0] == pytest.approx(4e5, rel=1e-6)

    # Stand in for what a solution iterator does between passes.
    recambered = iterate._with_recamber(config, "dchi_TE", {"dchi_TE[0]": -4.0})
    assert recambered.design().Re_surf()[0] != pytest.approx(4e5, rel=1e-4)

    assert iterate.resolve(recambered).design().Re_surf()[0] == pytest.approx(
        4e5, rel=1e-6
    )


def test_re_surf_needs_blades():
    """It is measured against a blade surface, so a mean line alone cannot."""
    config = Config.from_dict(
        {
            "fluid": FLUID,
            "mean_line": MEAN_LINE,
            "iterate": {"correct": [{"type": "Re_surf", "target": 4e5}]},
        }
    )

    with pytest.raises(ValueError, match="needs a blades: section"):
        iterate.resolve(config)


def test_re_surf_rejects_a_row_that_is_not_there():
    with pytest.raises(ValueError, match="i_row=5 is out of range"):
        iterate.resolve(with_Re(i_row=5))


def test_the_iterate_section_round_trips_a_design_only_iterator():
    config = with_Re(target=4e5, i_row=1)

    assert Config.from_dict(config.to_dict()) == config
    assert config.to_dict()["iterate"]["correct"][0]["type"] == "Re_surf"


#
# THE REPEATING STAGE
#
# A stage in the middle of a machine is fed by its own exit, so the inlet
# profile is a fixed point rather than something to state. The knob is Legendre
# coefficients, because a sampled profile would be as many unknowns as the mesh
# has span stations and would archive a mesh artefact into every output.yaml.
#


def repeating(**kwargs):
    """A config carrying a repeat iterator and nothing that needs solving."""
    return dataclasses.replace(
        build(), iterate=iterate.Iteration(correct=(iterate.Repeat(**kwargs),))
    )


def test_the_knobs_are_coefficients_not_samples():
    config = repeating(order=3)

    names = set(iterate.unknowns(config))

    assert len(names) == 9
    assert "inlet_profile.DPo[0]" in names
    assert "inlet_profile.DBeta[0]" not in names, "pitch angle is not carried"


def test_order_sets_the_number_of_knobs():
    assert len(iterate.unknowns(repeating(order=2))) == 6
    assert len(iterate.unknowns(repeating(order=5))) == 15


def test_an_absent_profile_reads_as_uniform():
    """The first iteration: no profile yet means no perturbation."""
    config = repeating()

    assert config.inlet_profile is None
    assert set(iterate.unknowns(config).values()) == {0.0}


def test_the_knobs_round_trip():
    config = repeating(order=2)
    moved = {
        "inlet_profile.DPo[0]": 0.4,
        "inlet_profile.DPo[1]": -0.25,
        "inlet_profile.DAlpha[0]": 1.5,
    }

    after = config.iterate.correct[0].with_unknowns(config, moved)

    for name, value in moved.items():
        assert iterate.unknowns(after)[name] == pytest.approx(value)
    # And it wrote a Legendre profile, not a sampled one.
    assert after.inlet_profile.type == "legendre"


def test_a_written_profile_carries_no_level():
    """Modes start at 1, so a profile cannot acquire a mean."""
    config = repeating(order=3)

    after = config.iterate.correct[0].with_unknowns(
        config, {"inlet_profile.DPo[0]": 0.4, "inlet_profile.DPo[1]": -0.2}
    )

    spf = np.linspace(0.0, 1.0, 2001)
    column = after.inlet_profile.column("DPo", spf)
    assert np.trapezoid(column, spf) == pytest.approx(0.0, abs=1e-6)


def test_paths_are_the_knobs_themselves():
    """The one iterator whose knobs are its leaves one for one."""
    config = repeating()
    written = config.iterate.correct[0].with_unknowns(config, {"inlet_profile.DPo[0]": 0.3})

    assert config.iterate.correct[0].paths(written) == set(iterate.unknowns(written))


def test_paths_match_what_repeat_writes():
    """Through the same probe the other iterators are held to."""
    config = repeating(order=2)
    seeded = config.iterate.correct[0].with_unknowns(
        config, {name: 0.1 for name in iterate.unknowns(config)}
    )

    assert seeded.iterate.correct[0].paths(seeded) == _probe(seeded.iterate.correct[0], seeded)


#
# TWO SCALES, NOT ONE
#


def test_the_angle_tolerance_is_not_the_pressure_tolerance():
    """DPo is a fraction of dynamic head and DAlpha is degrees, so one number
    cannot serve both: 0.01 is slack on the first and absurd on the second."""
    config = repeating(atol_head=0.02, atol_angle=0.5)

    tolerances = config.iterate.correct[0].tolerances(config)

    assert tolerances["inlet_profile.DPo[0]"] == pytest.approx(0.02)
    assert tolerances["inlet_profile.DTo[1]"] == pytest.approx(0.02)
    assert tolerances["inlet_profile.DAlpha[0]"] == pytest.approx(0.5)


def test_the_angle_clip_is_not_the_pressure_clip():
    config = repeating(clip_head=0.1, clip_angle=4.0)

    clips = config.iterate.correct[0].clips(config)

    assert clips["inlet_profile.DPo[0]"] == pytest.approx(0.1)
    assert clips["inlet_profile.DAlpha[2]"] == pytest.approx(4.0)


def test_the_inherited_tolerance_is_ignored():
    """Setting it cannot quietly do half of something."""
    config = repeating(tolerance=99.0, clip=99.0, atol_head=0.02, atol_angle=0.5)

    assert config.iterate.correct[0].tolerances(config)["inlet_profile.DPo[0]"] == 0.02
    assert config.iterate.correct[0].clips(config)["inlet_profile.DAlpha[0]"] == 5.0


#
# HOW MUCH COMES ROUND AGAIN
#


def test_the_whole_exit_profile_comes_round_by_default():
    """A strictly repeating stage, which is what the loop meant before."""
    assert iterate.Repeat().transfers() == {"DPo": 1.0, "DTo": 1.0, "DAlpha": 1.0}


def test_only_the_temperature_is_damped():
    """Pressure and angle are re-established by the row; temperature mixes."""
    transfers = iterate.Repeat(transfer_To=0.5).transfers()

    assert transfers["DTo"] == pytest.approx(0.5)
    assert transfers["DPo"] == 1.0
    assert transfers["DAlpha"] == 1.0


def test_a_damped_temperature_moves_the_fixed_point_not_the_path(monkeypatch):
    """The error is what the loop nulls, so halving the transfer has to leave a
    null error at an inlet carrying half the exit profile -- not merely take
    smaller steps towards carrying all of it, which is what a gain would do."""
    exit_profile = {"DPo": (0.4, 0.0), "DTo": (0.6, 0.0), "DAlpha": (2.0, 0.0)}
    monkeypatch.setattr(
        iterate, "exit_profile", lambda result, order, offset: exit_profile
    )

    config = repeating(order=2, transfer_To=0.5)
    repeat = config.iterate.correct[0]

    # An inlet carrying half the exit temperature profile and all of the rest.
    settled = repeat.with_unknowns(
        config,
        {
            "inlet_profile.DPo[0]": 0.4,
            "inlet_profile.DTo[0]": 0.3,
            "inlet_profile.DAlpha[0]": 2.0,
        },
    )
    result = Result(machine=settled.design(), grid=object())

    errors = settled.iterate.correct[0].error(settled, result)

    assert errors["inlet_profile.DTo[0]"] == pytest.approx(0.0)
    assert errors["inlet_profile.DPo[0]"] == pytest.approx(0.0)
    assert errors["inlet_profile.DAlpha[0]"] == pytest.approx(0.0)

    # And the undamped loop is not settled there: it wants the whole profile.
    undamped = dataclasses.replace(
        settled, iterate=iterate.Iteration(correct=(iterate.Repeat(order=2),))
    )
    assert undamped.iterate.correct[0].error(undamped, result)[
        "inlet_profile.DTo[0]"
    ] == pytest.approx(-0.3)


def test_a_transfer_outside_zero_to_one_is_refused():
    for transfer_To in (-0.1, 1.5):
        with pytest.raises(ValueError, match="between 0 and 1"):
            iterate.Repeat(transfer_To=transfer_To)


def test_order_below_one_is_refused():
    with pytest.raises(ValueError, match="at least 1"):
        iterate.Repeat(order=0)


def test_the_repeat_section_round_trips():
    config = repeating(order=4, atol_angle=0.25, transfer_To=0.5)

    assert Config.from_dict(config.to_dict()) == config


#
# THE FIT
#


def test_the_fit_recovers_blockage_but_not_the_wall():
    """The trade the default order is chosen under, pinned so that changing it
    later is a decision rather than an accident.

    A Legendre fit to an endwall boundary layer is pointwise poor and
    integrally good. Low order is defensible only because what propagates round
    a repeating loop is the integrated deficit, the near-wall flow being
    re-established by the no-slip wall just downstream of the inlet plane.
    """
    from numpy.polynomial import legendre  # noqa: PLC0415

    spf = np.linspace(0.0, 1.0, 401)
    delta = 0.05
    u = np.minimum(np.minimum(spf / delta, 1.0), np.minimum((1 - spf) / delta, 1.0))
    u = u ** (1 / 7)
    DPo = u**2 - 1.0

    fit = legendre.legval(2 * spf - 1, legendre.legfit(2 * spf - 1, DPo, 3))

    # Pointwise it misses most of the wall deficit.
    assert fit[0] > -0.5, "a cubic should not resolve the wall value"
    assert DPo[0] == pytest.approx(-1.0, abs=1e-6)

    # Integrally it is close, which is what the scheme relies on.
    blockage = np.trapezoid(1 - u, spf)
    fitted = np.trapezoid(1 - np.sqrt(np.clip(fit + 1, 0, None)), spf)
    assert abs(fitted - blockage) / blockage < 0.1


#
# THE TRUST BOUND
#


@pytest.mark.parametrize(
    "change, limit, expected",
    [
        # Nothing over its limit passes through untouched.
        ([0.05, -0.02], [0.1, 0.1], [0.05, -0.02]),
        # One knob over: everything shrinks by the same factor.
        ([0.3, 0.05], [0.1, 0.1], [0.1, 0.05 / 3.0]),
        # Both over, which is the case that used to be projected onto a corner.
        ([0.3, -0.2], [0.1, 0.1], [0.1, -0.2 / 3.0]),
        # A knob with no clip does not constrain the others.
        ([5.0, -2.0], [np.inf, np.inf], [5.0, -2.0]),
        ([0.3, -0.2], [0.1, np.inf], [0.1, -0.2 / 3.0]),
        # A step of nothing stays nothing rather than dividing by zero.
        ([0.0, 0.0], [0.1, 0.1], [0.0, 0.0]),
    ],
)
def test_the_trust_bound_scales_rather_than_clipping(change, limit, expected):
    """The direction survives; only the length is cut."""
    np.testing.assert_allclose(
        iterate._bounded(np.array(change), np.array(limit)), expected
    )


def test_the_trust_bound_keeps_the_direction():
    """The ratio between knobs is what per-component clipping destroyed."""
    change = np.array([0.31, -0.17, 0.02])
    bounded = iterate._bounded(change, np.full(3, 0.1))

    # Same direction, shorter.
    np.testing.assert_allclose(
        bounded / np.linalg.norm(bounded), change / np.linalg.norm(change)
    )
    assert np.linalg.norm(bounded) < np.linalg.norm(change)

    # And the knob that binds moves exactly its clip, which is what the
    # setting promises.
    assert np.max(np.abs(bounded)) == pytest.approx(0.1)


def test_a_saturated_step_does_not_cycle():
    """Two knobs whose Newton step exceeds both clips must not orbit.

    Clipping each on its own sent a two-knob loading iterator into a period-2
    orbit: the corner of the box overshoots, the next step wants the opposite
    corner, and the design lands exactly where it was two iterations before.
    Scaling cannot do that, because it never changes the direction it was
    handed.
    """
    change = np.array([0.4, -0.25])
    limit = np.full(2, 0.1)

    first = iterate._bounded(change, limit)
    # The corner would have been (+0.1, -0.1): equal magnitudes, no memory of
    # the 0.4-to-0.25 ratio that the Jacobian actually asked for.
    assert abs(first[0]) != pytest.approx(abs(first[1]))
    np.testing.assert_allclose(first[0] / first[1], change[0] / change[1])


#
# THE LOADING DISTRIBUTION
#
# One point on the curve rather than end angles, so its knob is a camber line
# and its error is a single number: the front acceleration. Everything below
# stands in for the CFD that would measure it: `Shaped` declares the answer as
# a function of its own knob, exactly as `Fixed` and `Coupled` do above.
#

BERNSTEIN = {"type": "bernstein", "order": 2, "coeff": [0.0]}
"""A camber line carrying exactly the one coefficient the iterator moves."""


def shaped(camber=None, **kwargs):
    """Return a blade whose sections carry a Bernstein camber line.

    `test_blade.blade` writes a quadratic camber and takes no argument to say
    otherwise; rather than widen the shared factory for one caller, this
    rewrites the sections it produces.
    """
    built = blade(**kwargs)
    built["sections"] = [
        {**section, "camber": dict(camber or BERNSTEIN)} for section in built["sections"]
    ]
    return built


@pytest.fixture
def loading():
    """A two-row config whose first row has its loading shaped."""
    return dataclasses.replace(
        build(blades=[shaped(), shaped()]),
        iterate=iterate.Iteration(correct=(iterate.LoadingDistribution(fac_front=1.8),)),
    )


class Shaped(iterate.LoadingDistribution):
    """The real knob, against a declared response instead of a solved grid.

    Only `error` is overridden, so `unknowns`, `with_unknowns` and `paths` are
    the shipped ones. No `type`, so it stays out of the registry.

    The response is the one measured on a cascade, so what the loop is tested
    against is the coupling it will actually meet: the coefficient softens the
    front, and the circulation coefficient moves it too --- appreciably, which
    is why the level had to become a target of its own rather than being left
    to float.
    """

    def response(self, config):
        (c0,) = (self.unknowns(config)[name] for name in self.names())
        Co = config.blades[self.i_row].count.Co
        return 1.900 - 0.500 * c0 + 2.600 * (Co - 0.7)

    def error(self, config, result):
        del result
        return {self.names()[0]: self.response(config) - self.fac_front}


def _measured(fac_front=1.95, fac_peak=1.25, zeta_peak=0.62):
    """A `Loading` with the fields a test cares about and plausible rest."""
    return turbigen.loading.Loading(
        zeta_peak=zeta_peak, fac_front=fac_front, fac_peak=fac_peak,
        ma_peak=0.75, ma_TE=0.60, ma_max=0.74, zeta_max=zeta_peak + 0.02,
    )


class Levelled(iterate.PeakMach):
    """The `peak_Ma` stand-in, whose slope is positive where the camber's is not.

    Its whole reason for being a separate member: `Iterator.gain` carries one
    sign per iterator, and this knob's is the opposite of the shape knob's.
    """

    def response(self, config):
        c0 = float(
            np.mean([s.camber.coeff[0] for s in config.blades[self.i_row].sections])
        )
        Co = config.blades[self.i_row].count.Co
        return 1.160 - 0.020 * c0 + 0.500 * (Co - 0.7)

    def error(self, config, result):
        del result
        return {f"Co[{self.i_row}]": self.response(config) - self.fac_peak}


def designing(config, i_iter):
    """A run that designs and solves nothing."""
    del i_iter
    return Result(machine=config.design())


def test_loading_owns_one_coefficient_of_its_own_row(loading):
    """One knob, named by row so a second entry cannot collide with it."""
    assert set(loading.iterate.correct[0].unknowns(loading)) == {
        "camber_coeff[0][0]",
    }


def test_loading_reads_the_mean_across_sections(loading):
    """One number per row, as a deviation is one per row.

    The distribution is measured at a single span fraction, so it has nothing
    to say about how the shape should vary up the blade.
    """
    varied = dataclasses.replace(
        loading.blades[0],
        sections=tuple(
            dataclasses.replace(
                section, camber=dataclasses.replace(section.camber, coeff=(c,))
            )
            for section, c in zip(loading.blades[0].sections, (0.1, 0.2, 0.6))
        ),
    )
    config = dataclasses.replace(loading, blades=(varied, loading.blades[1]))

    assert loading.iterate.correct[0].unknowns(config)[
        "camber_coeff[0][0]"
    ] == pytest.approx(0.3)


def test_loading_shifts_every_section_alike(loading):
    """A uniform shift, so a spanwise distribution survives being iterated."""
    before = (0.1, 0.2, 0.6)
    varied = dataclasses.replace(
        loading.blades[0],
        sections=tuple(
            dataclasses.replace(
                section, camber=dataclasses.replace(section.camber, coeff=(c,))
            )
            for section, c in zip(loading.blades[0].sections, before)
        ),
    )
    config = dataclasses.replace(loading, blades=(varied, loading.blades[1]))

    moved = loading.iterate.correct[0].with_unknowns(
        config, {"camber_coeff[0][0]": 0.5}
    )
    after = [section.camber.coeff[0] for section in moved.blades[0].sections]

    # The mean lands on what was asked for, and the spread is untouched.
    assert np.mean(after) == pytest.approx(0.5)
    np.testing.assert_allclose(np.array(after) - np.array(before), 0.2)


def test_loading_leaves_the_rows_it_does_not_own_alone(loading):
    """One row per iterator, so the other row's camber must not move."""
    moved = loading.iterate.correct[0].with_unknowns(
        loading, {"camber_coeff[0][0]": 0.3}
    )
    assert moved.blades[1] == loading.blades[1]


def test_peak_owns_the_circulation_coefficient(loading):
    """The lever `LoadingDistribution` has none of.

    At a fixed blade count the loop area is fixed, so a camber line cannot set
    the peak height on its own; freeing `Co` is what lifts it.
    """
    peak = iterate.PeakMach(i_row=0)

    assert peak.unknowns(loading) == {"Co[0]": pytest.approx(loading.blades[0].count.Co)}

    moved = peak.with_unknowns(loading, {"Co[0]": 0.62})
    assert moved.blades[0].count.Co == pytest.approx(0.62)
    # It owns the count and nothing else; the camber is the other member's.
    assert moved.blades[0].sections == loading.blades[0].sections


def test_peak_declares_the_opposite_sign_to_the_shape(loading):
    """Why this is a member of its own rather than a third knob on the other.

    `gain` carries one sign per iterator. The peak rises with the circulation
    coefficient while the front value falls with the camber coefficient, so a
    single scalar cannot serve both -- folded together it drove this knob the
    wrong way every iteration.
    """
    config = dataclasses.replace(
        loading,
        iterate=iterate.Iteration(
            correct=(*loading.iterate.correct, iterate.PeakMach(i_row=0))
        ),
    )
    gain, _, _ = iterate.properties(config)

    assert gain["camber_coeff[0][0]"] < 0.0
    assert gain["Co[0]"] > 0.0


def test_peak_paths_match_what_is_moved(loading):
    peak = iterate.PeakMach(i_row=0)
    assert peak.paths(loading) == _probe(peak, loading)
    assert peak.paths(loading) <= set(node.flatten(loading))


def test_peak_round_trips_through_a_config_dict(loading):
    config = dataclasses.replace(
        loading, iterate=iterate.Iteration(correct=(iterate.PeakMach(fac_peak=1.3),))
    )
    assert Config.from_dict(config.to_dict()) == config
    assert config.to_dict()["iterate"]["correct"][0]["type"] == "peak_Ma"


def test_peak_without_a_grid_measures_nothing(loading):
    assert iterate.PeakMach().error(loading, Result()) == {}


def test_peak_delegates_to_the_measurement(loading, monkeypatch):
    monkeypatch.setattr(
        turbigen.loading, "measure", lambda *a: _measured(fac_peak=1.31)
    )
    error = iterate.PeakMach(fac_peak=1.2).error(
        loading, Result(machine=loading.design(), grid=object())
    )
    assert error["Co[0]"] == pytest.approx(1.31 - 1.2)


def test_peak_needs_a_circulation_count():
    """A fixed blade count has no continuous lever to move."""
    fixed = shaped()
    fixed["count"] = {"type": "Nb", "Nb": 40}
    config = build(blades=[fixed, shaped()])

    with pytest.raises(ValueError, match="FixedCount"):
        iterate.PeakMach().unknowns(config)


@pytest.mark.parametrize("kwargs", [{"fac_peak": 0.0}, {"fac_peak": -1.0},
                                    {"zeta_front": 0.0}, {"zeta_front": 0.99}])
def test_peak_refuses_an_impossible_setting(kwargs):
    with pytest.raises(ValueError):
        iterate.PeakMach(**kwargs)


def test_loading_paths_match_what_is_moved(loading):
    """The `paths`/`with_unknowns` contract, on a knob nested two deep.

    A camber coefficient is a leaf inside a tuple inside a camber inside a
    section, which is further down than any other iterator reaches.
    """
    iterator = loading.iterate.correct[0]

    assert iterator.paths(loading) == _probe(iterator, loading)
    assert iterator.paths(loading) <= set(node.flatten(loading))


def test_loading_round_trips_through_a_config_dict(loading):
    assert Config.from_dict(loading.to_dict()) == loading
    assert loading.to_dict()["iterate"]["correct"][0]["type"] == "loading"


def test_loading_two_rows_do_not_collide():
    """A stator and a rotor want different loading, so two entries."""
    config = dataclasses.replace(
        build(blades=[shaped(), shaped()]),
        iterate=iterate.Iteration(
            correct=(
                iterate.LoadingDistribution(i_row=0, fac_front=1.8),
                iterate.LoadingDistribution(i_row=1, fac_front=1.9),
            )
        ),
    )

    assert set(iterate.unknowns(config)) == {
        "camber_coeff[0][0]",
        "camber_coeff[1][0]",
    }


def test_loading_shares_a_table_with_the_recambers():
    """Nothing else claims a camber coefficient, so all three iterate together."""
    config = dataclasses.replace(
        build(blades=[shaped(), shaped()]),
        iterate=iterate.Iteration(
            correct=(
                iterate.Deviation(),
                iterate.Incidence(),
                iterate.LoadingDistribution(),
            )
        ),
    )

    unknowns = iterate.unknowns(config)
    assert "camber_coeff[0][0]" in unknowns
    assert "dchi_TE[0]" in unknowns
    assert iterate.format_table(config, Result())


def test_loading_uses_the_inherited_tolerance(loading):
    """One knob, one error, so the base class's scalar tolerance is enough."""
    iterator = dataclasses.replace(loading.iterate.correct[0], tolerance=0.07)
    assert iterator.tolerances(loading) == {"camber_coeff[0][0]": 0.07}


def test_loading_without_a_grid_measures_nothing(loading):
    """A run with no field says nothing about a distribution."""
    assert loading.iterate.correct[0].error(loading, Result()) == {}


def test_loading_needs_a_bernstein_camber():
    """A quadratic camber line has no interior coefficient to move."""
    config = build(blades=[blade(), blade()])
    with pytest.raises(ValueError, match="Quadratic"):
        iterate.LoadingDistribution().unknowns(config)


@pytest.mark.parametrize(
    "camber",
    [
        {"type": "bernstein", "order": 3, "coeff": [0.0, 0.0]},
        {"type": "bernstein", "order": 2, "coeff": []},
    ],
)
def test_loading_needs_exactly_one_coefficient(camber):
    """One is what a camber line has to give to move a single front value.

    A short `coeff` is refused as well as a long one: `Bernstein` zero-pads it
    to evaluate, but an unwritten coefficient is no leaf of the config, and
    `paths` would then name something that does not exist.
    """
    config = build(blades=[shaped(camber=camber), shaped()])
    with pytest.raises(ValueError, match="exactly 1"):
        iterate.LoadingDistribution().unknowns(config)


def test_loading_refuses_a_row_that_is_not_there():
    config = build(blades=[shaped(), shaped()])
    with pytest.raises(ValueError, match="out of range"):
        iterate.LoadingDistribution(i_row=5).unknowns(config)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"zeta_front": 0.0},  # the stagnation point itself
        {"zeta_front": 1.5},  # past the trailing edge
        {"fac_front": 0.0},  # a leading edge that is not moving
        {"fac_front": -0.5},  # a Mach number ratio below zero
    ],
)
def test_loading_refuses_an_unreachable_target(kwargs):
    """Caught where it is written, rather than as a NaN eight solves later."""
    with pytest.raises(ValueError):
        iterate.LoadingDistribution(**kwargs)


def test_loading_converges_onto_its_target(loading):
    """The closed loop, against a declared response instead of a solver."""
    config = dataclasses.replace(
        loading,
        iterate=iterate.Iteration(
            correct=(
                Shaped(fac_front=1.85),
                Levelled(fac_peak=1.19),
            )
        ),
    )

    final, _, converged = iterate.converge(config, designing, max_iter=20)
    assert converged

    assert final.iterate.correct[0].response(final) == pytest.approx(1.85, abs=0.05)
    assert final.iterate.correct[1].response(final) == pytest.approx(1.19, abs=0.02)


def test_loading_moves_the_camber_it_started_from(loading):
    """Convergence has to come from the coefficient, not from nowhere."""
    config = dataclasses.replace(
        loading, iterate=iterate.Iteration(correct=(Shaped(fac_front=1.85),))
    )
    final, _, _ = iterate.converge(config, designing, max_iter=20)

    assert final.blades[0].sections[0].camber.coeff != (0.0,)
    assert final.blades[1] == config.blades[1]


def test_loading_is_clipped(loading):
    """The trust bound holds on the first step, taken before anything is known."""
    iterator = Shaped(fac_front=1.20, clip=0.02)
    config = dataclasses.replace(
        loading, iterate=iterate.Iteration(correct=(iterator,))
    )

    stepped = iterate.step(config, Result(machine=config.design()))

    for name, value in iterator.unknowns(stepped).items():
        assert abs(value - iterator.unknowns(config)[name]) <= 0.02 + 1e-9


def test_loading_delegates_to_the_measurement(loading, monkeypatch):
    """`error` guards and subtracts; everything else is `measure`.

    Replaced wholesale, which is what having it at module level is for: the
    cutting it does needs a solved grid, and the arithmetic either side of it
    does not.
    """
    monkeypatch.setattr(turbigen.loading, "measure", lambda *a: _measured(fac_front=1.95))

    error = loading.iterate.correct[0].error(
        loading, Result(machine=loading.design(), grid=object())
    )

    assert error["camber_coeff[0][0]"] == pytest.approx(1.95 - 1.8)


def test_loading_drops_a_distribution_with_no_suction_surface(loading, monkeypatch):
    """An unmeasured knob is held, not stepped on a NaN."""
    monkeypatch.setattr(turbigen.loading, "measure", lambda *a: None)

    result = Result(machine=loading.design(), grid=object())
    assert loading.iterate.correct[0].error(loading, result) == {}
    assert not iterate.converged(loading, result)


#
# THE LOADING PROFILE
#
# The whole curve rather than one point, so its knobs are every interior
# camber coefficient plus the circulation coefficient that owns the level.
# `measure_profile` stands in for the CFD, exactly as `measure` does above.
#

PROFILE_BERNSTEIN = {"type": "bernstein", "order": 3, "coeff": [0.0, 0.0]}
"""A camber line carrying exactly the two coefficients a `LoadingProfile` of
order 3 moves."""


@pytest.fixture
def profile():
    """A two-row config whose first row has its loading profile shaped."""
    return dataclasses.replace(
        build(blades=[shaped(camber=PROFILE_BERNSTEIN), shaped(camber=PROFILE_BERNSTEIN)]),
        iterate=iterate.Iteration(correct=(iterate.LoadingProfile(order=3),)),
    )


def test_profile_owns_two_coefficients_and_the_level(profile):
    """order=3 gives two interior coefficients, plus the row's own Co."""
    assert set(profile.iterate.correct[0].unknowns(profile)) == {
        "camber_coeff[0][0]",
        "camber_coeff[0][1]",
        "Co[0]",
    }


def test_profile_reads_the_mean_across_sections(profile):
    """One number per row and per coefficient, as a deviation is one per row."""
    varied = dataclasses.replace(
        profile.blades[0],
        sections=tuple(
            dataclasses.replace(
                section, camber=dataclasses.replace(section.camber, coeff=(c, 0.0))
            )
            for section, c in zip(profile.blades[0].sections, (0.1, 0.2, 0.6))
        ),
    )
    config = dataclasses.replace(profile, blades=(varied, profile.blades[1]))

    unknowns = profile.iterate.correct[0].unknowns(config)
    assert unknowns["camber_coeff[0][0]"] == pytest.approx(0.3)
    assert unknowns["camber_coeff[0][1]"] == pytest.approx(0.0)


def test_profile_shifts_every_section_alike(profile):
    """A uniform shift, so a spanwise distribution survives being iterated."""
    moved = profile.iterate.correct[0].with_unknowns(
        profile, {"camber_coeff[0][0]": 0.4, "Co[0]": 0.65}
    )

    coeffs0 = [s.camber.coeff[0] for s in moved.blades[0].sections]
    np.testing.assert_allclose(coeffs0, 0.4)
    assert moved.blades[0].count.Co == pytest.approx(0.65)
    # The other coefficient and the other row are untouched.
    assert all(s.camber.coeff[1] == 0.0 for s in moved.blades[0].sections)
    assert moved.blades[1] == profile.blades[1]


def test_profile_paths_match_what_is_moved(profile):
    """A knob nested two deep, plus a sibling one on the blade count."""
    iterator = profile.iterate.correct[0]
    assert iterator.paths(profile) == _probe(iterator, profile)
    assert iterator.paths(profile) <= set(node.flatten(profile))


def test_profile_round_trips_through_a_config_dict(profile):
    assert Config.from_dict(profile.to_dict()) == profile
    assert profile.to_dict()["iterate"]["correct"][0]["type"] == "loading_profile"


def test_profile_needs_a_bernstein_camber():
    """A quadratic camber line has no interior coefficient to move."""
    config = build(blades=[blade(), blade()])
    with pytest.raises(ValueError, match="Quadratic"):
        iterate.LoadingProfile().unknowns(config)


def test_profile_needs_coefficients_written_out_in_full():
    """A short or mismatched-order camber line is refused, not padded."""
    config = build(
        blades=[shaped(camber={"type": "bernstein", "order": 4, "coeff": [0.0, 0.0, 0.0]})]
        * 2
    )
    with pytest.raises(ValueError, match="written out in full"):
        iterate.LoadingProfile(order=3).unknowns(config)


def test_profile_refuses_a_row_that_is_not_there():
    config = build(blades=[shaped(camber=PROFILE_BERNSTEIN), shaped(camber=PROFILE_BERNSTEIN)])
    with pytest.raises(ValueError, match="out of range"):
        iterate.LoadingProfile(i_row=5).unknowns(config)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"order": 1},  # no interior coefficient at all
        {"zeta_front": 0.6, "zeta_peak": 0.5},  # peak ahead of the front anchor
        {"zeta_front": 0.0},  # the stagnation point itself
        {"zeta_peak": 1.0},  # the trailing edge itself
        {"fac_front": 0.0},
        {"fac_peak": -0.5},
    ],
)
def test_profile_refuses_an_impossible_setting(kwargs):
    with pytest.raises(ValueError):
        iterate.LoadingProfile(**kwargs)


def test_profile_without_a_grid_measures_nothing(profile):
    assert profile.iterate.correct[0].error(profile, Result()) == {}


def test_profile_gains_clips_and_tolerances_differ_for_shape_and_level():
    """Two knobs, not one: the level and the shape need not agree."""
    iterator = iterate.LoadingProfile(
        order=3, gain=-0.5, gain_Co=1.5, clip=0.1, clip_Co=0.05,
        tolerance=0.05, tolerance_Co=0.02,
    )
    config = build(blades=[shaped(camber=PROFILE_BERNSTEIN)] * 2)

    assert iterator.gains(config) == {
        "camber_coeff[0][0]": -0.5, "camber_coeff[0][1]": -0.5, "Co[0]": 1.5,
    }
    assert iterator.clips(config) == {
        "camber_coeff[0][0]": 0.1, "camber_coeff[0][1]": 0.1, "Co[0]": 0.05,
    }
    assert iterator.tolerances(config) == {
        "camber_coeff[0][0]": 0.05, "camber_coeff[0][1]": 0.05, "Co[0]": 0.02,
    }


def test_profile_delegates_to_the_measurement(profile, monkeypatch):
    """`error` guards, builds the target, and splits level from shape; the rest is `measure_profile`."""
    monkeypatch.setattr(turbigen.loading, "mach_ratio", lambda *a: 1.1)
    monkeypatch.setattr(
        turbigen.loading,
        "measure_profile",
        lambda *a: (np.array([0.3, 0.7]), np.array([2.0, 2.5])),
    )

    iterator = profile.iterate.correct[0]
    error = iterator.error(profile, Result(machine=profile.design(), grid=object()))

    # zeta=0.3 sits on the front line (target 2.0), zeta=0.7 on the aft line
    # (target 1.88); the mean of the two residuals is the level, and each
    # point's own residual less that mean is its shape error.
    assert error["Co[0]"] == pytest.approx(0.31)
    assert error["camber_coeff[0][0]"] == pytest.approx(-0.31)
    assert error["camber_coeff[0][1]"] == pytest.approx(0.31)


def test_profile_holds_a_knob_below_zeta_front(profile, monkeypatch):
    """A coefficient whose m maps inside zeta_front is held, not stepped."""
    monkeypatch.setattr(turbigen.loading, "mach_ratio", lambda *a: 1.1)
    monkeypatch.setattr(
        turbigen.loading,
        "measure_profile",
        lambda *a: (np.array([0.15, 0.7]), np.array([0.0, 2.5])),
    )

    iterator = profile.iterate.correct[0]
    error = iterator.error(profile, Result(machine=profile.design(), grid=object()))

    assert "camber_coeff[0][0]" not in error
    assert error["camber_coeff[0][1]"] == pytest.approx(0.0)
    assert error["Co[0]"] == pytest.approx(0.62)


def test_profile_drops_a_distribution_with_no_suction_surface(profile, monkeypatch):
    """An unmeasured row is held, not stepped on a NaN."""
    monkeypatch.setattr(turbigen.loading, "measure_profile", lambda *a: None)

    result = Result(machine=profile.design(), grid=object())
    assert profile.iterate.correct[0].error(profile, result) == {}
    assert not iterate.converged(profile, result)
