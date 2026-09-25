"""Tests for design iteration.

No CFD anywhere here. Everything is a pure function of a config and a result,
so the stand-ins declare the answer: `Fixed` and `Other` state their own error
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
- test_step_refuses_to_move_a_design_it_could_not_measure: a failed run stops it
- test_unmeasured_knobs_are_not_converged: nor is silence agreement
- test_a_knob_on_its_clip_does_not_shrink_another: the trust bound is per knob
- test_an_unmeasurable_section_says_where_it_is: a failure names what to fix
- test_converge_reaches_the_answer: the loop, on an analytic error
- test_converge_stops_on_a_diverged_march: a blown-up run measures nothing
- test_converge_gives_up: and stops when it cannot
- test_the_step_is_the_declared_gain: sign and size, every iteration
- test_the_loop_holds_no_grids: a finished pass frees its field
- test_deviation_error_is_zero_for_a_machine_that_matches: the measurement datum
- test_mean_line_error_is_zero_for_its_own_design: likewise, through backward()
- test_mean_line_tolerance_scales_with_the_nominal: relative, per variable
- test_mean_line_restores_a_scalar_as_a_scalar: shapes survive a round trip
- test_clark_recamber_error_flips_with_the_turning: one gain for rows turning either way
- test_clark_recambers_the_way_incidence_would: an over-loaded nose is pressure-side incidence
- test_clark_refuses_to_share_the_recamber_with_incidence: one owner per leaf
"""

import dataclasses

import numpy as np
import pytest
from test_blade import FLUID, MEAN_LINE, blade, build

import turbigen.loading
import turbigen.util
from turbigen import Config, DesignError, Result, iterate, node, shapespace


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


class Other(iterate.Iterator):
    """A second stand-in on a different leaf, so two iterators are disjoint.

    `spill` is how much of the other iterator's knob leaks into this one's
    error.
    """

    target: float = 1.0
    slope: float = 1.0
    spill: float = 0.0
    name: str = "other"

    def unknowns(self, config):
        return {self.name: float(config.mean_line.phi2)}

    def with_unknowns(self, config, values):
        return dataclasses.replace(
            config,
            mean_line=dataclasses.replace(config.mean_line, phi2=values[self.name]),
        )

    def error(self, config, result):
        return {
            self.name: self.slope * (config.mean_line.phi2 - self.target)
            + self.spill * config.mean_line.psi
        }


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

    after = (
        config.iterate.correct[0].with_unknowns(config, {"dchi_TE[0]": -3.0}).to_dict()
    )

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


def test_step_refuses_to_move_a_design_it_could_not_measure(config):
    """Stepping knobs nothing was read off is worse than stopping."""
    with pytest.raises(iterate.MeasurementError):
        iterate.step(config, Result())


def test_unmeasured_knobs_are_not_converged(config):
    """Silence is not agreement, and now it is not silence either."""
    with pytest.raises(iterate.MeasurementError):
        iterate.converged(config, Result())

    # The tolerant reading still describes the run rather than raising, which
    # is what a report of a diverged march needs.
    assert iterate.errors(config, Result(), strict=False) == {}


class Blows(Fixed):
    """A stand-in that fails the way a cut through a field of NaNs does."""

    def error(self, config, result):
        raise ValueError("Pressure must be positive and finite.")


def test_tolerant_errors_survive_any_failure_to_measure():
    """A diverged field raises wherever it is touched, not as MeasurementError."""
    config = dataclasses.replace(
        build(),
        iterate=iterate.Iteration(correct=(Blows(), Other())),
    )

    with pytest.raises(ValueError):
        iterate.errors(config, Result())

    # The knobs that could be measured are still reported.
    assert set(iterate.errors(config, Result(), strict=False)) == {"other"}


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
    # A gain of 0.5 against a true slope of 1 halves the error each pass,
    # which from psi = 1.6 is about ten of them.
    assert len(seen) <= 12


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
# THE STEP
#


def test_the_step_is_the_declared_gain(config):
    """`u -= gain * e`, sign and size, on every iteration."""
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


def test_a_knob_on_its_clip_does_not_shrink_another():
    """A trust bound is a statement about one knob's step, not everyone's.

    Scaling the step as a whole let one thickness coefficient marching
    against its clip hold a circulation coefficient to a fifth of its own
    step, every pass.
    """
    far = dataclasses.replace(
        build(),
        iterate=iterate.Iteration(
            correct=(Fixed(name="toy", target=100.0, clip=0.5), Other(name="other"))
        ),
    )
    alone = dataclasses.replace(
        far, iterate=iterate.Iteration(correct=(Other(name="other"),))
    )

    both = iterate.step(far, Result())
    only = iterate.step(alone, Result())

    # `toy` is a hundred away and held to its clip; `other` must take the
    # same step it would have taken with nothing beside it.
    assert both.mean_line.phi2 == pytest.approx(only.mean_line.phi2)
    assert abs(both.mean_line.psi - far.mean_line.psi) == pytest.approx(0.5)


def test_an_unmeasurable_section_says_where_it_is(config, monkeypatch):
    """A failure the config can act on names the row, the section and the span.

    A section sitting above a clearance gap has no blade surface to stagnate
    on, and that is a config to fix rather than a knob to hold.
    """
    monkeypatch.setattr(turbigen.util, "cut_blade_surfs", lambda grid: [None, None])
    monkeypatch.setattr(iterate, "_incidence", lambda *a, **k: np.nan)

    result = Result(machine=config.design(), grid=object())
    with pytest.raises(iterate.MeasurementError, match=r"row 0 section 0") as raised:
        iterate.Incidence().error(config, result)
    assert "clearance gap" in str(raised.value)


def test_an_incidence_target_is_onto_the_pressure_surface_on_every_row(
    config, monkeypatch
):
    """One target, the same physical incidence on rows turning opposite ways.

    The two rows here do: the metal angle rises through the first and falls
    through the second. Flow arriving from the pressure side of the metal is
    then below it on the first row and above it on the second, so a positive
    target is met by flow less metal of opposite signs.
    """
    machine = config.design()
    rows = machine.rows
    assert not rows[0].blade.suction_is_upper
    assert rows[1].blade.suction_is_upper

    target = 5.0
    onto_pressure = {0: -target, 1: target}
    monkeypatch.setattr(turbigen.util, "cut_blade_surfs", lambda grid: [None, None])
    monkeypatch.setattr(
        iterate,
        "_incidence",
        lambda result, surface, i_row, *a, **k: onto_pressure[i_row],
    )

    result = Result(machine=machine, grid=object())
    errors = iterate.Incidence(target=target).error(config, result)
    assert errors
    assert all(e == pytest.approx(0.0) for e in errors.values())


def test_the_loop_holds_no_grids():
    """A Result pins a live grid; keeping one per iteration would pin them all.

    Asserted by weak reference rather than by inspection, because the failure
    would otherwise be silent until a large machine ran out of memory.
    """
    import gc
    import weakref

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


def test_mean_line_clip_scales_with_the_nominal():
    """The clip is relative for the reason the tolerance is, and to the same
    nominal --- two conventions in one config would have `clip: 0.01` permit a
    third of one loss coefficient and an eighth of the next."""
    config = dataclasses.replace(
        build(),
        iterate=iterate.Iteration(
            correct=(iterate.MeanLine(variables=("psi", "Ys"), clip=0.05),)
        ),
    )

    clips = config.iterate.correct[0].clips(config)

    assert clips["mean_line.psi"] == pytest.approx(0.05 * config.mean_line.psi)
    assert clips["mean_line.Ys[0]"] == pytest.approx(0.05 * config.mean_line.Ys[0])
    assert clips["mean_line.Ys[1]"] == pytest.approx(0.05 * config.mean_line.Ys[1])


def test_mean_line_no_clip_stays_no_clip():
    """Zero is what the stepper reads as unbounded, so it must not be scaled."""
    config = dataclasses.replace(
        build(),
        iterate=iterate.Iteration(
            correct=(iterate.MeanLine(variables=("Ys",), clip=0.0),)
        ),
    )

    assert set(config.iterate.correct[0].clips(config).values()) == {0.0}


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

    A resolved knob has ~0 error while its *value* in this config is stale,
    because `resolve` moved it on a copy inside the run. Stepping the pair
    would move it somewhere neither describes.
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
    written = config.iterate.correct[0].with_unknowns(
        config, {"inlet_profile.DPo[0]": 0.3}
    )

    assert config.iterate.correct[0].paths(written) == set(iterate.unknowns(written))


def test_paths_match_what_repeat_writes():
    """Through the same probe the other iterators are held to."""
    config = repeating(order=2)
    seeded = config.iterate.correct[0].with_unknowns(
        config, {name: 0.1 for name in iterate.unknowns(config)}
    )

    assert seeded.iterate.correct[0].paths(seeded) == _probe(
        seeded.iterate.correct[0], seeded
    )


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


def test_the_whole_exit_profile_comes_round_undamped():
    """A strictly repeating stage, which is what the loop meant before."""
    assert iterate.Repeat(transfer_To=1.0).transfers() == {
        "DPo": 1.0,
        "DTo": 1.0,
        "DAlpha": 1.0,
    }


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
        iterate, "exit_profile", lambda result, modes, order, offset: exit_profile
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
        settled,
        iterate=iterate.Iteration(
            correct=(iterate.Repeat(order=2, transfer_To=1.0),)
        ),
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
    from numpy.polynomial import legendre

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
# THE CLARK PROFILE
#
# Both surfaces at once, driven by thickness rather than camber, against the
# distribution `turbigen.clark` draws. `measure_clark_profile` stands in for
# the CFD, as `measure` does above.
#

CLARK_THICKNESS = {
    "type": "clark",
    "R_LE": 0.05,
    "tanwedge": 0.18,
    "t_TE": 0.03,
    "coeff": [[0.0, 0.0], [0.0, 0.0]],
}
"""A symmetric two-sided section of order 3, so a `ClarkProfile` has two
interior coefficients per surface to move."""

CLARK_MEAN_LINE = {**MEAN_LINE, "Ma2": 0.75}
"""Slow enough that the default `Ma_peak` asks for a subsonic suction peak,
which `ClarkProfile.check` insists on before anything is designed."""


def thickened(thickness=None, **kwargs):
    """Return a blade whose sections carry a two-sided thickness."""
    built = blade(**kwargs)
    built["sections"] = [
        {**section, "thickness": dict(thickness or CLARK_THICKNESS)}
        for section in built["sections"]
    ]
    return built


@pytest.fixture
def clark():
    """A two-row config whose first row has its thickness shaped."""
    return dataclasses.replace(
        build(blades=[thickened(), thickened()], mean_line=CLARK_MEAN_LINE),
        iterate=iterate.Iteration(correct=(iterate.ClarkProfile(),)),
    )


def measured_as(z, fac, ratio=1.0, Co=None):
    """A `ClarkMeasurement` standing in for a solved row.

    `Co` defaults to the loop those samples would give if they were the whole
    distribution, which is what the real measurement integrates from the cut.
    Given explicitly where a test wants the two to disagree.
    """
    z, fac = np.asarray(z, dtype=float), np.asarray(fac, dtype=float)
    if Co is None:
        Co = float(np.trapezoid(fac[0], z[0]) - ratio * np.trapezoid(fac[1], z[1]))
    return turbigen.loading.ClarkMeasurement(z=z, fac=fac, Co=Co, length_ratio=ratio)


def target_loop(iterator, machine, ratio):
    """The circulation the target asks for, as `error` forms it."""
    dense = np.linspace(0.0, 1.0, iterate.N_LOOP)
    wanted = iterator.target(np.stack((dense, dense)), machine)
    return float(
        np.trapezoid(wanted[0], dense) - ratio * np.trapezoid(wanted[1], dense)
    )


def test_clark_owns_both_ends_both_surfaces_the_level_and_the_recamber(clark):
    """Order 3 gives four control points, of which only the ends are shared."""
    assert set(clark.iterate.correct[0].unknowns(clark)) == {
        "Co[0]",
        "dchi_LE[0]",
        "tau_LE[0]",
        "tau_TE[0]",
        "tau[0][0][1]",
        "tau[0][0][2]",
        "tau[0][1][1]",
        "tau[0][1][2]",
    }


def test_clark_puts_the_level_first_and_the_ends_before_the_interior(clark):
    """The order a sequence `gain` is matched to, so it is load-bearing.

    `Co` and the recamber first and the two ends next means none of them
    moves when a design changes order --- only the interior grows, at the
    tail. Written the other way round, a gain written for one design would be
    read back against different knobs on the next.
    """
    assert list(clark.iterate.correct[0].unknowns(clark))[:4] == [
        "Co[0]",
        "dchi_LE[0]",
        "tau_LE[0]",
        "tau_TE[0]",
    ]


def test_clark_reads_the_ends_off_the_shape_space_curve(clark):
    """The knobs are coefficients, and the ends of that curve are the physics.

    Which is what buys one gain sign for every knob: a nose radius written as
    `sqrt(2 R_LE)` thickens its surface the same way an interior coefficient
    does, where the radius itself would have needed a prior of its own.
    """
    unknowns = clark.iterate.correct[0].unknowns(clark)

    assert unknowns["tau_LE[0]"] == pytest.approx(
        shapespace.tau_LE(CLARK_THICKNESS["R_LE"])
    )
    assert unknowns["tau_TE[0]"] == pytest.approx(
        shapespace.tau_TE(CLARK_THICKNESS["t_TE"], CLARK_THICKNESS["tanwedge"])
    )


def test_clark_writes_what_it_says_it_writes(clark):
    """`paths` is declared rather than inferred, so it has to be checked.

    A leaf this moves without naming would be read as a design variable by
    anything mining an archive of runs, and a predictor would then take the
    thickness it is trying to predict as an input.
    """
    iterator = clark.iterate.correct[0]
    before = node.flatten(clark)

    moved = set()
    for name, value in iterator.unknowns(clark).items():
        after = node.flatten(iterator.with_unknowns(clark, {name: value + 0.02}))
        moved |= {path for path in before if before[path] != after.get(path)}

    assert moved == iterator.paths(clark)


def test_clark_shifts_every_section_together(clark):
    """One span fraction is measured, so one shift is all it can justify.

    Whatever spanwise variation of the thickness a design asked for therefore
    survives being iterated, exactly as it does for a camber line.
    """
    iterator = clark.iterate.correct[0]
    unknowns = iterator.unknowns(clark)

    moved = iterator.with_unknowns(
        clark, {"tau[0][0][1]": unknowns["tau[0][0][1]"] + 0.1}
    )

    shifts = [
        section.thickness.tau_coeff[0][1] - original.thickness.tau_coeff[0][1]
        for section, original in zip(moved.blades[0].sections, clark.blades[0].sections)
    ]
    assert shifts == pytest.approx([0.1] * len(shifts))


def test_clark_keeps_one_nose_and_one_wedge(clark):
    """Moving an end knob moves it on both surfaces, that being what it is.

    A `ClarkThickness` has one leading edge radius serving two surfaces, so a
    knob on it cannot mean one thing to one side and another to the other.
    """
    iterator = clark.iterate.correct[0]
    unknowns = iterator.unknowns(clark)

    moved = iterator.with_unknowns(clark, {"tau_LE[0]": unknowns["tau_LE[0]"] + 0.05})

    for section in moved.blades[0].sections:
        c = section.thickness.tau_coeff
        assert c[0][0] == pytest.approx(c[1][0])
        assert c[0][0] == pytest.approx(unknowns["tau_LE[0]"] + 0.05)


def test_clark_splits_the_level_from_the_shape(clark, monkeypatch):
    """`error` builds the target and divides it; the rest is the measurement.

    The level is the circulation the blade drew less the one the target asks
    for, and the shape is what is left once the surface offset that would
    cause that level is taken back off --- `level / (1 + ratio)` on one side
    and its negative on the other, which is a half each only when the two
    surfaces are the same length. A ratio away from one here, so the weighting
    is pinned rather than cancelling.
    """
    ratio = 0.8
    monkeypatch.setattr(turbigen.loading, "mach_ratio", lambda *a: 1.0)
    z = np.array([[0.2, 0.5, 0.7, 0.9], [0.2, 0.5, 0.7, 0.9]])
    monkeypatch.setattr(
        turbigen.loading,
        "measure_clark_profile",
        lambda *a: measured_as(z, np.zeros((2, 4)), ratio),
    )

    iterator = clark.iterate.correct[0]
    machine = clark.design()
    error = iterator.error(clark, Result(machine=machine, grid=object()))

    # Everything measured zero, so each residual is minus its own target.
    residual = -iterator.target(z, machine)
    level = measured_as(z, np.zeros((2, 4)), ratio).Co - target_loop(
        iterator, machine, ratio
    )
    delta = level / (1.0 + ratio)
    shape = residual - np.array([[delta], [-delta]])

    assert error["Co[0]"] == pytest.approx(level)
    assert error["tau_LE[0]"] == pytest.approx(0.5 * (shape[0][0] + shape[1][0]))
    # Row 0's metal angle rises, so its pressure surface is the upper one and
    # the loading error is already in the recamber's frame.
    assert not machine.rows[0].blade.suction_is_upper
    assert error["dchi_LE[0]"] == pytest.approx(shape[0][0] - shape[1][0])
    assert error["tau_TE[0]"] == pytest.approx(0.5 * (shape[0][-1] + shape[1][-1]))
    assert error["tau[0][0][1]"] == pytest.approx(shape[0][1])
    assert error["tau[0][1][2]"] == pytest.approx(shape[1][2])


def test_clark_ends_cannot_be_driven_by_the_level(clark, monkeypatch):
    """The property that leaves the loop determined.

    A shared end reports the *mean* of the two surfaces' residuals there, and
    the offset is taken off one surface and added to the other --- so it
    cancels exactly. The nose and the wedge answer only for the common mode at
    their end, and can neither be moved by the blade count nor fight it.

    A surface lifted by `d` and the other dropped by `d` opens the loop by
    `d (1 + ratio)`, the two surfaces entering the circulation weighted by
    their own lengths. The ratio is away from one here, so that weighting is
    what the number below tests rather than something that cancels.
    """
    ratio = 0.8
    monkeypatch.setattr(turbigen.loading, "mach_ratio", lambda *a: 1.0)

    z = np.array([[0.2, 0.5, 0.7, 0.9], [0.2, 0.5, 0.7, 0.9]])
    iterator = clark.iterate.correct[0]
    machine = clark.design()
    target = iterator.target(z, machine)

    # Two measurements differing by a pure level: one surface lifted and the
    # other dropped, which is what opening the loop does and nothing else. The
    # circulation is stated rather than integrated from these four samples,
    # which span only part of the surface -- the real one integrates the whole
    # cut, so a uniform offset reaches it whole.
    base = target_loop(iterator, machine, ratio)
    errors = []
    for offset in (0.0, 0.3):
        fac = target + np.array([[offset], [-offset]])
        loop = base + offset * (1.0 + ratio)
        monkeypatch.setattr(
            turbigen.loading,
            "measure_clark_profile",
            lambda *a, f=fac, c=loop: measured_as(z, f, ratio, Co=c),
        )
        errors.append(iterator.error(clark, Result(machine=machine, grid=object())))

    # The first is the target's own loop, so it reports no circulation error.
    assert errors[0]["Co[0]"] == pytest.approx(0.0, abs=1e-12)

    assert errors[1]["Co[0]"] - errors[0]["Co[0]"] == pytest.approx(0.3 * (1.0 + ratio))
    for name in ("tau_LE[0]", "tau_TE[0]", "dchi_LE[0]"):
        assert errors[1][name] == pytest.approx(errors[0][name], abs=1e-12)


#
# A TRAILING EDGE THAT IS ONE KNOB, LIKE THE NOSE
#


def test_clark_cannot_see_an_antisymmetric_trailing_edge_error(clark, monkeypatch):
    """The null a shared wedge angle has, at both ends now, and its price.

    Two surfaces equally wrong in opposite directions read as converged
    through a single knob reporting their mean, because that is the part one
    knob cannot reach. The trailing edge was a knob per surface and could see
    this; it gave that up so the thickness would stop laying a second claim on
    the exit angle --- see `ClarkThickness.tanwedge`. Kept as a test because
    the blind spot is a cost that should be visible, not a detail.
    """
    monkeypatch.setattr(turbigen.loading, "mach_ratio", lambda *a: 1.0)

    z = np.array([[0.2, 0.5, 0.7, 0.9], [0.2, 0.5, 0.7, 0.9]])
    iterator = clark.iterate.correct[0]
    machine = clark.design()

    # Right everywhere but the two ends, where the surfaces are wrong by the
    # same amount in opposite directions.
    skew = np.array([[0.1, 0.0, 0.0, 0.1], [-0.1, 0.0, 0.0, -0.1]])
    fac = iterator.target(z, machine) + skew
    monkeypatch.setattr(
        turbigen.loading,
        "measure_clark_profile",
        lambda *a: measured_as(z, fac),
    )
    errors = iterator.error(clark, Result(machine=machine, grid=object()))

    # Whatever the skew does to the level, the offset takes the same amount
    # off one surface and adds it to the other --- so at the trailing edge the
    # antisymmetric part survives untouched, and the one knob there reports
    # the mean of it, which is zero.
    assert errors["tau_TE[0]"] == pytest.approx(0.0, abs=1e-12)

    # Nor can the nose radius, one knob for two surfaces; but at the nose the
    # recamber reads the difference: the skew twice over, less the part of it
    # the level took off both surfaces (a half each, at equal lengths).
    assert errors["tau_LE[0]"] == pytest.approx(0.0, abs=1e-12)
    assert errors["dchi_LE[0]"] == pytest.approx(0.2 - errors["Co[0]"])
    assert errors["dchi_LE[0]"] != pytest.approx(0.0)


def test_clark_takes_the_whole_level_off_the_shape(clark, monkeypatch):
    """What the offset is for: the shape never sees the loop.

    A circulation error is the blade count's alone, a thickness being able to
    move loading about but not to create it. So the surface offset that would
    have caused the measured level is taken back off before any coefficient
    reads its own residual --- and at unequal surface lengths that offset is
    `level / (1 + ratio)`, not half each.
    """
    ratio = 0.6
    delta = 0.25
    monkeypatch.setattr(turbigen.loading, "mach_ratio", lambda *a: 1.0)

    z = np.array([[0.2, 0.5, 0.7, 0.9], [0.2, 0.5, 0.7, 0.9]])
    iterator = clark.iterate.correct[0]
    machine = clark.design()

    # On target everywhere, then lifted and dropped by a pure offset -- so
    # every residual is the offset and nothing else.
    fac = iterator.target(z, machine) + np.array([[delta], [-delta]])
    loop = target_loop(iterator, machine, ratio) + delta * (1.0 + ratio)
    monkeypatch.setattr(
        turbigen.loading,
        "measure_clark_profile",
        lambda *a: measured_as(z, fac, ratio, Co=loop),
    )
    errors = iterator.error(clark, Result(machine=machine, grid=object()))

    # The whole of it lands on the count, and every shape knob reads zero.
    assert errors["Co[0]"] == pytest.approx(delta * (1.0 + ratio))
    for name, value in errors.items():
        if name != "Co[0]":
            assert value == pytest.approx(0.0, abs=1e-12), name


def test_clark_measures_a_level_the_curve_order_cannot_move(monkeypatch):
    """The circulation is a property of the flow, not of the parameterisation.

    It used to be a mean over the control points, so raising the order moved
    every sample and reported a different loop for an unchanged flow --- a
    design variable reading differently because of how the thickness happened
    to be written down. Measured over the cut, the order cannot reach it.
    """
    monkeypatch.setattr(turbigen.loading, "mach_ratio", lambda *a: 1.0)

    levels, orders = [], []
    for coeff in ([[0.0, 0.0], [0.0, 0.0]], [[0.0] * 4, [0.0] * 4]):
        section = {**CLARK_THICKNESS, "coeff": coeff}
        config = dataclasses.replace(
            build(
                blades=[thickened(section), thickened(section)],
                mean_line=CLARK_MEAN_LINE,
            ),
            iterate=iterate.Iteration(correct=(iterate.ClarkProfile(),)),
        )
        iterator = config.iterate.correct[0]
        machine = config.design()
        orders.append(config.blades[0].sections[0].thickness.order)

        # The same flow either way, stated as one circulation and a
        # distribution sitting exactly on target wherever it is sampled.
        m_ctl = config.blades[0].sections[0].thickness.m_ctl
        z = np.stack((m_ctl, m_ctl))
        monkeypatch.setattr(
            turbigen.loading,
            "measure_clark_profile",
            lambda *a, zz=z, it=iterator, mc=machine: measured_as(
                zz, it.target(zz, mc), 1.0, Co=0.7
            ),
        )
        levels.append(
            iterator.error(config, Result(machine=machine, grid=object()))["Co[0]"]
        )

    assert orders[0] != orders[1]
    assert levels[0] == pytest.approx(levels[1])


def test_clark_holds_the_nose_inside_its_bounds(clark):
    """A bound on where the knob arrives, which no clip provides.

    `clip` limits one step and not a run of them, so a nose thickened a little
    every pass reaches a radius no single step would have been allowed. Held
    at the bound rather than refused: refusing the step would freeze every
    other knob in the row because this one reached a limit.
    """
    iterator = clark.iterate.correct[0]
    lo, hi = iterator.R_LE_lim
    tau = iterator.unknowns(clark)["tau_LE[0]"]

    for asked, bound in ((10.0, hi), (-tau + (2.0 * lo) ** 0.5 * 0.5, lo)):
        moved = iterator.with_unknowns(clark, {"tau_LE[0]": tau + asked})
        for section in moved.blades[0].sections:
            assert section.thickness.R_LE == pytest.approx(bound)


def test_clark_leaves_a_nose_inside_its_bounds_alone(clark):
    """The bound is a limit, not a target: within it, nothing is held."""
    iterator = clark.iterate.correct[0]
    tau = iterator.unknowns(clark)["tau_LE[0]"]

    # Half way between where it starts and the upper bound, so the move is
    # real but lands inside.
    wanted = 0.5 * (clark.blades[0].sections[0].thickness.R_LE + iterator.R_LE_lim[1])
    moved = iterator.with_unknowns(clark, {"tau_LE[0]": (2.0 * wanted) ** 0.5})

    assert moved.blades[0].sections[0].thickness.R_LE == pytest.approx(wanted)
    assert tau != pytest.approx((2.0 * wanted) ** 0.5)


def _nose_error_at(clark, R_LE, offset, monkeypatch):
    """Return row 0's `tau_LE` error with its nose at `R_LE`, both surfaces
    running `offset` fast at the first station and on target elsewhere."""
    monkeypatch.setattr(turbigen.loading, "mach_ratio", lambda *a: 1.0)
    iterator = clark.iterate.correct[0]

    def at(section):
        c = section.thickness.tau_coeff.copy()
        c[:, 0] = shapespace.tau_LE(R_LE)
        return dataclasses.replace(
            section, thickness=section.thickness.with_tau_coeff(c)
        )

    blade = clark.blades[0]
    config = dataclasses.replace(
        clark,
        blades=(
            dataclasses.replace(blade, sections=tuple(map(at, blade.sections))),
            *clark.blades[1:],
        ),
    )
    machine = config.design()

    z = np.array([[0.2, 0.5, 0.7, 0.9], [0.2, 0.5, 0.7, 0.9]])
    fac = iterator.target(z, machine) + np.array(
        [[offset, 0.0, 0.0, 0.0], [offset, 0.0, 0.0, 0.0]]
    )
    loop = target_loop(iterator, machine, 1.0)
    monkeypatch.setattr(
        turbigen.loading,
        "measure_clark_profile",
        lambda *a: measured_as(z, fac, Co=loop),
    )
    return iterator.error(config, Result(machine=machine, grid=object()))[
        "tau_LE[0]"
    ]


@pytest.mark.parametrize(
    "bound, offset, nulled",
    [
        ("lower", 0.05, True),
        ("lower", -0.05, False),
        ("upper", -0.05, True),
        ("upper", 0.05, False),
    ],
)
def test_clark_counts_a_nose_held_at_its_bound_as_converged(
    clark, monkeypatch, bound, offset, nulled
):
    """A nose that may go no further has converged; one that can come back
    off the bound has not, and keeps the error that takes it there."""
    lo, hi = clark.iterate.correct[0].R_LE_lim
    R_LE = lo if bound == "lower" else hi

    error = _nose_error_at(clark, R_LE, offset, monkeypatch)

    if nulled:
        assert error == 0.0
    else:
        assert error == pytest.approx(offset)


def test_clark_reports_a_free_nose_whole(clark, monkeypatch):
    """Inside its bounds the nose error is the plain mean residual."""
    lo, hi = clark.iterate.correct[0].R_LE_lim
    error = _nose_error_at(clark, 0.5 * (lo + hi), 0.05, monkeypatch)

    assert error == pytest.approx(0.05)


def test_clark_refuses_a_step_that_closes_the_section(clark):
    """A knob has no bound of its own that keeps an aerofoil open.

    `clip` limits one step, not where a run of them arrives, and a thickness
    driven through the camber line fails in the mesher rather than here --- a
    long way from the step that caused it.
    """
    iterator = clark.iterate.correct[0]
    unknowns = iterator.unknowns(clark)

    held = iterator.with_unknowns(
        clark, {"tau[0][0][1]": unknowns["tau[0][0][1]"] - 5.0}
    )

    assert held.blades[0].sections == clark.blades[0].sections


def test_clark_without_a_grid_raises(clark):
    iterator = clark.iterate.correct[0]
    with pytest.raises(iterate.MeasurementError, match="no solved grid"):
        iterator.error(clark, Result(machine=clark.design()))


def test_clark_leading_edge_tolerance_is_its_own(clark):
    """The nose is often held against its R_LE bound, so it converges to a
    wider criterion than the other shape knobs without slackening them."""
    config = dataclasses.replace(
        clark,
        iterate=iterate.Iteration(
            correct=(iterate.ClarkProfile(tolerance=0.01, tolerance_tau_LE=0.05),)
        ),
    )
    tolerances = config.iterate.correct[0].tolerances(config)

    assert tolerances["tau_LE[0]"] == pytest.approx(0.05)
    assert tolerances["tau_TE[0]"] == pytest.approx(0.01)
    assert tolerances["tau[0][0][1]"] == pytest.approx(0.01)


def test_clark_tolerances_default(clark):
    """Shape knobs converge to 0.025 and the rest to 0.02 unless a config says otherwise."""
    tolerances = clark.iterate.correct[0].tolerances(clark)

    shape = {name for name in tolerances if name.startswith(("tau[", "tau_TE["))}
    assert shape
    for name, value in tolerances.items():
        assert value == pytest.approx(0.025 if name in shape else 0.02), name


def test_clark_recambers_every_section_together(clark):
    """One span fraction is measured, so the recamber is a uniform shift too."""
    iterator = clark.iterate.correct[0]
    dchi = iterator.unknowns(clark)["dchi_LE[0]"]

    moved = iterator.with_unknowns(clark, {"dchi_LE[0]": dchi + 0.7})

    shifts = [
        section.dchi_LE - original.dchi_LE
        for section, original in zip(moved.blades[0].sections, clark.blades[0].sections)
    ]
    assert shifts == pytest.approx([0.7] * len(shifts))
    assert moved.blades[1] == clark.blades[1]
    for section, original in zip(moved.blades[0].sections, clark.blades[0].sections):
        assert section.thickness == original.thickness


def _nose_loading_error(clark, i_row, skew, monkeypatch):
    """Return the `dchi_LE` error of row `i_row` for a nose skewed by `skew`."""
    monkeypatch.setattr(turbigen.loading, "mach_ratio", lambda *a: 1.0)
    config = dataclasses.replace(
        clark,
        iterate=iterate.Iteration(correct=(iterate.ClarkProfile(i_row=i_row),)),
    )
    iterator = config.iterate.correct[0]
    machine = config.design()

    z = np.array([[0.2, 0.5, 0.7, 0.9], [0.2, 0.5, 0.7, 0.9]])
    fac = iterator.target(z, machine) + np.array(
        [[skew, 0.0, 0.0, 0.0], [-skew, 0.0, 0.0, 0.0]]
    )
    # The target's own loop, so no level is taken off and the skew is read
    # whole.
    loop = target_loop(iterator, machine, 1.0)
    monkeypatch.setattr(
        turbigen.loading,
        "measure_clark_profile",
        lambda *a: measured_as(z, fac, Co=loop),
    )
    return iterator.error(config, Result(machine=machine, grid=object()))[
        f"dchi_LE[{i_row}]"
    ]


def test_clark_recamber_error_flips_with_the_turning(clark, monkeypatch):
    """The same over-loaded nose, on rows turning opposite ways.

    A rising metal angle takes flow off the pressure surface where that is
    the lower one and puts it on where it is the upper, so the one positive
    gain needs the error turned into the recamber's frame row by row.
    """
    rows = clark.design().rows
    assert rows[0].blade.suction_is_upper != rows[1].blade.suction_is_upper

    first = _nose_loading_error(clark, 0, 0.1, monkeypatch)
    second = _nose_loading_error(clark, 1, 0.1, monkeypatch)

    assert abs(first) == pytest.approx(0.2)
    assert second == pytest.approx(-first)


@pytest.mark.parametrize("i_row", [0, 1])
def test_clark_recambers_the_way_incidence_would(clark, monkeypatch, i_row):
    """An over-loaded nose is flow too far onto the pressure surface.

    So the recamber this takes for it has to have the sign the incidence
    iterator takes for the same flow, on either row. Checked against that
    iterator rather than against a sign written here, because its frame is the
    one pinned to what a blade measures.
    """
    clark_step = -iterate.ClarkProfile().gain_dchi_LE * _nose_loading_error(
        clark, i_row, 0.1, monkeypatch
    )

    # Flow onto the pressure surface, in the frame the incidence is measured
    # in: positive where the suction surface is the upper one.
    machine = clark.design()
    onto_pressure = 5.0 if machine.rows[i_row].blade.suction_is_upper else -5.0
    monkeypatch.setattr(turbigen.util, "cut_blade_surfs", lambda grid: [None, None])
    monkeypatch.setattr(iterate, "_incidence", lambda *a, **k: onto_pressure)
    config = dataclasses.replace(
        clark, iterate=iterate.Iteration(correct=(iterate.Incidence(),))
    )
    incidence = iterate.Incidence()
    errors = incidence.error(config, Result(machine=machine, grid=object()))
    incidence_step = -incidence.gain * errors[f"dchi_LE[{i_row}][0]"]

    assert np.sign(clark_step) == np.sign(incidence_step) != 0.0


def test_clark_refuses_to_share_the_recamber_with_incidence(clark):
    """Two iterators moving one leaf towards different targets is a fight."""
    config = dataclasses.replace(
        clark,
        iterate=iterate.Iteration(
            correct=(iterate.ClarkProfile(), iterate.Incidence())
        ),
    )

    with pytest.raises(ValueError, match="Drop the incidence iterator"):
        iterate.unknowns(config)


def test_clark_recamber_has_its_own_gain_clip_and_tolerance(clark):
    """An angle against a loading error, so none of the shape knobs' numbers."""
    iterator = iterate.ClarkProfile(
        gain_dchi_LE=3.0, clip_dchi_LE=0.4, tolerance_dchi_LE=0.07
    )

    assert iterator.gains(clark)["dchi_LE[0]"] == pytest.approx(3.0)
    assert iterator.clips(clark)["dchi_LE[0]"] == pytest.approx(0.4)
    assert iterator.tolerances(clark)["dchi_LE[0]"] == pytest.approx(0.07)
    assert iterator.gains(clark)["tau_LE[0]"] == pytest.approx(iterator.gain)


def test_clark_needs_a_two_sided_thickness():
    """A Taylor section is the same both sides, so it has no rows to move."""
    config = dataclasses.replace(
        build(blades=[blade()]),
        iterate=iterate.Iteration(correct=(iterate.ClarkProfile(),)),
    )

    with pytest.raises(ValueError, match="differ side to side"):
        config.iterate.correct[0].unknowns(config)


def test_clark_needs_coefficients_written_out():
    """An empty row is a symmetric section, with nothing between its ends."""
    config = dataclasses.replace(
        build(blades=[thickened({**CLARK_THICKNESS, "coeff": [[], []]})]),
        iterate=iterate.Iteration(correct=(iterate.ClarkProfile(),)),
    )

    with pytest.raises(ValueError, match="no interior thickness coefficients"):
        config.iterate.correct[0].unknowns(config)


def test_clark_needs_every_section_to_agree_on_order():
    """Sections are interpolated field by field, which ragged rows cannot be."""
    row = thickened()
    row["sections"][0]["thickness"] = {**CLARK_THICKNESS, "coeff": [[0.0], [0.0]]}
    config = dataclasses.replace(
        build(blades=[row]),
        iterate=iterate.Iteration(correct=(iterate.ClarkProfile(),)),
    )

    with pytest.raises(ValueError, match="same number of thickness coefficients"):
        config.iterate.correct[0].unknowns(config)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"z_peak": 0.0}, "0 < z_peak < 1"),
        ({"z_peak": 1.0}, "0 < z_peak < 1"),
        ({"Ma_peak": 0.0}, "Ma_peak must be positive"),
        ({"Ma_peak_max": 0.0}, "Ma_peak_max must be positive"),
        ({"Ma_LE": -1.0}, "Ma_LE must be positive"),
        ({"Ma_PS": 0.0}, "Ma_PS must be positive"),
        ({"tolerance_tau_LE": 0.0}, "tolerance_tau_LE must be positive"),
    ],
)
def test_clark_rejects_a_target_off_the_surface(kwargs, match):
    with pytest.raises(ValueError, match=match):
        iterate.ClarkProfile(**kwargs)


@pytest.mark.parametrize("i_row", [0, 1])
def test_clark_refuses_a_peak_over_the_default_limit(clark, i_row):
    """The peak is `Ma_peak` times the row exit relative Mach number, checked
    at design time so a batch screens the point out before it is solved."""
    Ma_TE = float(clark.design().mean_line[:, i_row].Ma_rel[1])
    config = dataclasses.replace(
        clark,
        iterate=iterate.Iteration(
            correct=(iterate.ClarkProfile(i_row=i_row, Ma_peak=1.11 / Ma_TE),)
        ),
    )
    with pytest.raises(DesignError, match=f"Row {i_row}: .* exceeds Ma_peak_max=1.1"):
        config.design()


def test_clark_accepts_a_peak_under_the_default_limit(clark):
    Ma_TE = float(clark.design().mean_line[:, 0].Ma_rel[1])
    config = dataclasses.replace(
        clark,
        iterate=iterate.Iteration(
            correct=(iterate.ClarkProfile(Ma_peak=1.09 / Ma_TE),)
        ),
    )
    config.design()


@pytest.mark.parametrize("Ma_peak_max, refused", [(0.8, True), (1.2, False)])
def test_clark_peak_limit_is_configurable(clark, Ma_peak_max, refused):
    """The limit is on the peak itself, so the same target is refused under a
    tighter limit and accepted under a looser one."""
    Ma_TE = float(clark.design().mean_line[:, 0].Ma_rel[1])
    config = dataclasses.replace(
        clark,
        iterate=iterate.Iteration(
            correct=(
                iterate.ClarkProfile(Ma_peak=1.0 / Ma_TE, Ma_peak_max=Ma_peak_max),
            )
        ),
    )
    if refused:
        with pytest.raises(DesignError, match="exceeds Ma_peak_max=0.8"):
            config.design()
    else:
        config.design()


def test_an_iterator_checks_nothing_by_default(config):
    """The hook is opt-in, so an iterator without a design-time target is
    never the reason a design fails."""
    config.design()


#
# THE MODES A REPEATING STAGE IS FITTED IN
#


def test_a_basis_sets_the_modes_the_knobs_are_written_in(tmp_path):
    from test_pod import basis_file

    from turbigen import bconds

    path, sha = basis_file(tmp_path)
    config = repeating(order=2, basis=path)
    repeat = config.iterate.correct[0]

    assert len(iterate.unknowns(config)) == 6

    after = repeat.with_unknowns(
        config, {"inlet_profile.DPo[0]": 0.4, "inlet_profile.DPo[1]": -0.2}
    )
    assert after.inlet_profile.type == "pod"
    assert after.inlet_profile.sha256 == sha

    modes = bconds.PodModes(path, sha)
    expected = 0.4 * modes.table["DPo"][0] - 0.2 * modes.table["DPo"][1]
    assert after.inlet_profile.column("DPo", modes.spf) == pytest.approx(expected)


def test_an_order_beyond_the_basis_is_refused(tmp_path):
    from test_pod import N_MODE, basis_file

    path, _ = basis_file(tmp_path)

    with pytest.raises(ValueError, match="mode"):
        repeating(order=N_MODE + 1, basis=path)


def test_the_repeat_section_with_a_basis_round_trips(tmp_path):
    from test_pod import basis_file

    path, _ = basis_file(tmp_path)
    config = repeating(order=2, basis=path)

    assert Config.from_dict(config.to_dict()) == config


def test_the_outermost_stations_do_not_move_the_fit(monkeypatch):
    """The face at the wall reads the wall value itself; one is dropped."""
    from turbigen import bconds

    n = 137
    spf = 0.5 * (1.0 - np.cos(np.pi * (np.arange(n) + 0.5) / n))
    clean = {name: 0.1 * np.cos(np.pi * spf) for name in iterate.Repeat.COLUMNS}
    spiked = {name: values.copy() for name, values in clean.items()}
    for values in spiked.values():
        values[[0, -1]] = -50.0

    def measured(deficit):
        monkeypatch.setattr(
            iterate, "exit_deficit", lambda result, offset: (spf, deficit, {})
        )
        return iterate.exit_profile(None, bconds.LegendreModes(), 3)

    assert measured(spiked)["DPo"] == pytest.approx(measured(clean)["DPo"])
