"""Tests for the spanwise flow designs and the iterators that move them.

Built on the synthetic two-row axial turbine of `test_blade`, whose second row
rotates and whose blades have sections at span fractions 0.2, 0.5 and 0.8,
the knots of every station. The CFD is stood in for by :func:`fake_profiles`,
which measures the design's own spanwise flow on a fine grid of faces, so a
flow that follows its knobs exactly is a fixed point the iterators can be
checked against.

Test cases:
- test_omitted_spanwise_is_a_free_vortex: the default is written out
- test_vortex_expon_is_the_power_law: the formula, per row
- test_vortex_expon_needs_one_exponent_per_row: or one for all
- test_a_blade_vortex_exponent_is_an_unknown_key: the old spelling is refused
- test_the_spanwise_design_is_stamped: a new flow is a new field
- test_a_uniform_spline_is_a_free_vortex: knobs at defaults
- test_the_scale_of_a_velocity_shape_does_nothing: only the shape matters
- test_a_velocity_shape_keeps_the_mass_flow: area average is the mean line's
- test_an_angular_momentum_offset_is_additive: in r_rms * Vm_rms units
- test_defaults_are_written_out_in_full: every knot is a leaf
- test_bad_station_values_are_refused: one value per section, Vm positive
- test_a_station_count_must_match_the_rows: two per row
- test_one_section_is_constant_over_the_span
- test_the_knots_move_with_the_sections
- test_fit_knots_recovers_a_spline: exact, weights or not
- test_a_fit_ignores_the_walls_beyond_the_outermost_knots
- test_the_guess_follows_a_spline_profile: same flow as the blade
- test_deviation_and_an_outlet_iterator_are_refused_together
- test_two_outlet_iterators_on_one_row_are_refused
- test_spanwise_work_refuses_a_stator
- test_spline_iterators_refuse_a_vortex_design
- test_a_target_needs_a_value_per_section
- test_paths_match_what_is_moved / test_paths_are_real_leaves: the protocol
- test_measured_inflow_is_a_fixed_point: Vm and rVt copy what they measure
- test_inflow_steps_towards_the_measurement
- test_swirl_error_is_the_mean_the_target_cannot_set
- test_the_whole_span_mean_moves_the_target
- test_work_pairs_stream_surfaces_by_mass_fraction
- test_one_full_work_step_lands_on_the_target: gain sign and size
- test_target_plot_draws_what_the_iterator_drives_to
"""

import dataclasses

import numpy as np
import pytest
from test_blade import SPF, blade, build

from turbigen import DesignError, Result, guess, iterate, node, post, restart, spanwise

VM_SHAPE = (0.9, 1.08, 0.95)
"""A meridional velocity shape that is not uniform, one value per section."""

DELTA = np.array([0.05, -0.02, 0.08])
"""An angular momentum offset that is not uniform."""

N_FACE = 401
"""Faces across the span of a fake station profile."""


def spline_config(stations=None, correct=(), **kwargs):
    """The two-row turbine with a spline spanwise design."""
    if stations is None:
        stations = [{"Vm": VM_SHAPE}] * 4
    return build(
        spanwise={"type": "velocity_spline", "stations": stations},
        iterate={"correct": list(correct)},
        **kwargs,
    )


def chi(machine):
    """Every row's metal angles at a few span fractions."""
    return np.array(
        [
            [row.blade.evaluate_chi(spf) for spf in (0.0, 0.3, 0.7, 1.0)]
            for row in machine.rows
        ]
    )


def station_flow(config, k, spf):
    """Meridional velocity and angular momentum the design gives at station
    `k`, from the design's own spanwise flow."""
    mean_line = config.mean_line.design(config.fluid.eos())
    station = mean_line[k % 2, k // 2]
    knots = iterate._knots(config, k // 2)
    return (
        config.spanwise.Vm(station, k, knots, spf),
        config.spanwise.rVt(station, k, knots, spf),
    )


#
# A STAND-IN FOR THE CFD
#


def fake_profiles(config, ho_tilt=0.0):
    """Return what :func:`iterate.station_profile` would measure, per station.

    The design's own spanwise flow on a fine grid of faces, at uniform density.
    Stagnation enthalpy is the mean line's at each row inlet, tilted from hub
    to casing by `ho_tilt` times the square of the row's outlet blade speed,
    and rises through a row by the Euler work along each stream surface ---
    paired by mass fraction, so a test that gets the pairing wrong sees it.
    """
    mean_line = config.mean_line.design(config.fluid.eos())

    edges = np.linspace(0.0, 1.0, N_FACE + 1)
    spf = 0.5 * (edges[1:] + edges[:-1])

    profiles = {}
    for i_row in range(mean_line.n_row):
        U2 = float(mean_line[1, i_row].Omega * mean_line[1, i_row].r) ** 2
        for j in range(2):
            k = 2 * i_row + j
            r_edge = spanwise.station_radius(mean_line[j, i_row], edges)
            area = 0.5 * (r_edge[1:] + r_edge[:-1]) * np.diff(r_edge)
            Vm, rVt = station_flow(config, k, spf)
            mdot = Vm * area
            profiles[k] = {
                "spf": spf,
                "mdot": mdot,
                "area": area,
                "F": (np.cumsum(mdot) - 0.5 * mdot) / np.sum(mdot),
                "Vm": Vm,
                "rVt": rVt,
            }

        inlet, outlet = profiles[2 * i_row], profiles[2 * i_row + 1]
        inlet["ho"] = float(mean_line[0, i_row].ho) + ho_tilt * U2 * (spf - 0.5)
        Omega = float(mean_line[1, i_row].Omega)
        outlet["ho"] = np.interp(outlet["F"], inlet["F"], inlet["ho"]) + Omega * (
            outlet["rVt"] - np.interp(outlet["F"], inlet["F"], inlet["rVt"])
        )

    return profiles


@pytest.fixture
def measure(monkeypatch):
    """Stand the fake profiles in for the CFD; return a Result to measure."""

    def result_for(config, **kwargs):
        profiles = fake_profiles(config, **kwargs)
        monkeypatch.setattr(iterate, "station_profile", lambda result, k: profiles[k])
        return Result(machine=config.design(), grid=object())

    return result_for


#
# THE DESIGNS
#


def test_omitted_spanwise_is_a_free_vortex():
    config = build()

    assert config.to_dict()["spanwise"] == {"type": "vortex_expon", "n": -1.0}
    np.testing.assert_array_equal(
        chi(config.design()),
        chi(build(spanwise={"type": "vortex_expon", "n": -1.0}).design()),
    )


def test_vortex_expon_is_the_power_law():
    """The formula the blade used before there was a spanwise stage."""
    machine = build(spanwise={"type": "vortex_expon", "n": [-0.6, 0.3]}).design()

    for i_row, n in enumerate((-0.6, 0.3)):
        ml = machine.mean_line[:, i_row]
        spf = np.array([[0.0], [0.4], [1.0]])
        r = ml.r_hub * (1.0 - spf) + ml.r_cas * spf
        Vt_rel = ml.Vt * (r / ml.r) ** n - ml.Omega * r
        expected = np.degrees(np.arctan(Vt_rel / ml.Vm))

        actual = machine.spanwise.Alpha_rel(ml, i_row, SPF, spf)
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)


def test_vortex_expon_needs_one_exponent_per_row():
    with pytest.raises(ValueError, match="one vortex exponent per row"):
        build(spanwise={"type": "vortex_expon", "n": [-0.6]})


def test_a_blade_vortex_exponent_is_an_unknown_key():
    with pytest.raises(ValueError, match="vortex_exponent"):
        build(blades=[blade(vortex_exponent=-0.6), blade()])


def test_the_spanwise_design_is_stamped():
    free = build()
    forced = build(spanwise={"type": "vortex_expon", "n": -0.6})

    assert restart.design_stamp(free) != restart.design_stamp(forced)


def test_a_uniform_spline_is_a_free_vortex():
    uniform = spline_config(stations=[{}] * 4).design()

    # The mean line is single precision, and the two evaluate the same
    # angular momentum in a different order
    np.testing.assert_allclose(chi(uniform), chi(build().design()), atol=1e-5)


def test_the_scale_of_a_velocity_shape_does_nothing():
    shape = spline_config().design()
    doubled = spline_config(
        stations=[{"Vm": tuple(2.0 * v for v in VM_SHAPE)}] * 4
    ).design()

    np.testing.assert_allclose(chi(doubled), chi(shape), atol=1e-5)


def test_a_velocity_shape_keeps_the_mass_flow():
    """At uniform density, the area average is the mean line's velocity."""
    config = spline_config()
    mean_line = config.mean_line.design(config.fluid.eos())

    s = np.linspace(0.0, 1.0, 2001)
    for k in (2, 3):
        station = mean_line[k % 2, k // 2]
        Vm, _ = station_flow(config, k, s)
        r = spanwise.station_radius(station, s)
        average = np.trapezoid(Vm * r, s) / np.trapezoid(r, s)
        assert average == pytest.approx(float(station.Vm), rel=1e-6)


def test_an_angular_momentum_offset_is_additive():
    Delta = 0.1
    config = spline_config(stations=[{}, {"rVt": (Delta,) * 3}, {}, {}])
    mean_line = config.mean_line.design(config.fluid.eos())

    station = mean_line[1, 0]
    expected = float(station.r * station.Vt) + Delta * float(station.r * station.Vm)
    _, rVt = station_flow(config, 1, np.array([0.0, 0.5, 1.0]))
    np.testing.assert_allclose(rVt, expected, rtol=1e-6)


def test_defaults_are_written_out_in_full():
    data = spline_config(stations=[]).to_dict()["spanwise"]

    assert len(data["stations"]) == 4
    for station in data["stations"]:
        assert station == {"Vm": [1.0] * 3, "rVt": [0.0] * 3}


@pytest.mark.parametrize(
    "station, match",
    [
        ({"Vm": (1.0, 1.0)}, "gives 2 Vm value"),
        ({"rVt": (0.0,) * 4}, "gives 4 rVt value"),
        ({"Vm": (1.0, 0.0, 1.0)}, "must be positive"),
    ],
)
def test_bad_station_values_are_refused(station, match):
    with pytest.raises(ValueError, match=match):
        spline_config(stations=[station, {}, {}, {}])


def test_a_station_count_must_match_the_rows():
    with pytest.raises(ValueError, match="each of 4 stations"):
        spline_config(stations=[{}] * 3)


def test_one_section_is_constant_over_the_span():
    one = blade(section=None)
    one["sections"] = one["sections"][1:2]
    config = build(
        blades=[one, blade()],
        spanwise={
            "type": "velocity_spline",
            "stations": [{"Vm": (1.3,), "rVt": (0.1,)}, {}, {}, {}],
        },
    )
    mean_line = config.mean_line.design(config.fluid.eos())

    s = np.array([0.0, 0.3, 1.0])
    Vm, rVt = station_flow(config, 0, s)
    station = mean_line[0, 0]
    np.testing.assert_allclose(Vm, float(station.Vm), rtol=1e-6)
    expected = float(station.r * station.Vt) + 0.1 * float(station.r * station.Vm)
    np.testing.assert_allclose(rVt, expected, rtol=1e-6)


def test_the_knots_move_with_the_sections():
    """A knot value belongs to its section, wherever the section is put."""
    stations = [{}, {"rVt": tuple(DELTA)}, {}, {}]
    moved = blade()
    for section, spf in zip(moved["sections"], (0.1, 0.5, 0.9), strict=True):
        section["spf"] = spf
    config = spline_config(stations=stations, blades=[moved, blade()])
    mean_line = config.mean_line.design(config.fluid.eos())

    station = mean_line[1, 0]
    _, rVt = station_flow(config, 1, np.array([0.1, 0.5, 0.9]))
    Delta = (rVt - float(station.r * station.Vt)) / float(station.r * station.Vm)
    np.testing.assert_allclose(Delta, DELTA, atol=1e-6)


@pytest.mark.parametrize("weighted", [False, True])
def test_fit_knots_recovers_a_spline(weighted):
    s = np.linspace(0.2, 0.8, 57)
    weights = 1.0 + s if weighted else None

    fitted = spanwise.fit_knots(SPF, s, spanwise.spline(SPF, DELTA)(s), weights)

    np.testing.assert_allclose(fitted, DELTA, atol=1e-10)


def test_a_fit_ignores_the_walls_beyond_the_outermost_knots():
    """Fitted to the span between the outermost knots only, so a boundary
    layer outside them, however strong, leaves the knots alone."""
    face = np.linspace(0.0, 1.0, 101)
    inside = spanwise.spline(SPF, DELTA)(face)
    wall = np.where((face < 0.2) | (face > 0.8), 5.0 * (face - 0.5) ** 2, 0.0)
    profile = {"spf": face}

    fitted = iterate._fit(np.array(SPF), profile, inside + wall, np.ones_like(face))

    np.testing.assert_allclose(fitted, DELTA, atol=1e-10)


def test_the_guess_follows_a_spline_profile():
    """At a row's edges, the guess has the spanwise flow the blade was
    designed against: its meridional velocity shape and its swirl."""
    stations = [{"Vm": VM_SHAPE, "rVt": tuple(DELTA)}] * 4
    config = spline_config(stations=stations)
    machine = config.design()
    field = guess._Field(machine)

    s = np.array([0.2, 0.35, 0.5, 0.8])
    for k in range(4):
        m = np.full(s.size, k + 1.0)
        r = machine.annulus.evaluate_xr(m, s)[1]
        _, _, Vxrt = field.evaluate(m, s, r)
        Vm, rVt = station_flow(config, k, s)
        np.testing.assert_allclose(np.hypot(Vxrt[:, 0], Vxrt[:, 1]), Vm, rtol=1e-5)
        np.testing.assert_allclose(Vxrt[:, 2] * r, rVt, rtol=1e-4)


#
# CHECKS ON A CONFIG
#


def test_deviation_and_an_outlet_iterator_are_refused_together():
    work = {"type": "spanwise_work", "i_row": 1, "psi": [0.0] * 3}
    with pytest.raises(ValueError, match="deviation recambers"):
        spline_config(correct=[work, {"type": "deviation"}])


def test_two_outlet_iterators_on_one_row_are_refused():
    with pytest.raises(ValueError, match="same row"):
        spline_config(
            correct=[
                {"type": "spanwise_work", "i_row": 1, "psi": [0.0] * 3},
                {"type": "spanwise_swirl", "i_row": 1, "rVt": [0.0] * 3},
            ]
        )


def test_spanwise_work_refuses_a_stator():
    with pytest.raises(DesignError, match="row 0, which is stationary"):
        spline_config(
            correct=[{"type": "spanwise_work", "i_row": 0, "psi": [0.0] * 3}]
        ).design()


def test_spline_iterators_refuse_a_vortex_design():
    with pytest.raises(ValueError, match="velocity_spline"):
        build(iterate={"correct": [{"type": "spanwise_inflow"}]})


def test_a_target_needs_a_value_per_section():
    with pytest.raises(ValueError, match="3 blade section"):
        spline_config(correct=[{"type": "spanwise_swirl", "i_row": 0, "rVt": [0.0]}])


#
# THE ITERATOR PROTOCOL
#

ITERATORS = (
    iterate.SpanwiseInflow(),
    iterate.SpanwiseWork(i_row=1, psi=(0.0,) * 3),
    iterate.SpanwiseSwirl(i_row=0, rVt=(0.0,) * 3),
)


def _probe(iterator, config):
    """Return the config leaves `with_unknowns` actually writes."""
    before = node.flatten(config)
    shifted = {name: value + 0.1 for name, value in iterator.unknowns(config).items()}
    after = node.flatten(iterator.with_unknowns(config, shifted))
    return {path for path, value in before.items() if after.get(path) != value}


@pytest.mark.parametrize("iterator", ITERATORS, ids=[it.type for it in ITERATORS])
def test_paths_match_what_is_moved(iterator):
    config = spline_config()

    assert iterator.paths(config) == _probe(iterator, config)


def test_paths_are_real_leaves():
    config = spline_config(correct=[it.to_dict() for it in ITERATORS])
    leaves = set(node.flatten(config))

    for iterator in config.iterate.correct:
        assert iterator.paths(config) <= leaves

    # Disjoint, so the three can run together
    assert len(iterate.unknowns(config)) == sum(
        len(it.unknowns(config)) for it in config.iterate.correct
    )


#
# WHAT THEY MEASURE
#


def test_measured_inflow_is_a_fixed_point(measure):
    """A flow that is exactly the design leaves the inflow knobs where they are."""
    config = spline_config(stations=[{"Vm": VM_SHAPE, "rVt": tuple(DELTA)}] * 4)
    result = measure(config)

    inflow = iterate.SpanwiseInflow()
    error = inflow.error(config, result)
    assert set(error) == set(inflow.unknowns(config))
    np.testing.assert_allclose(list(error.values()), 0.0, atol=2e-3)


def test_inflow_steps_towards_the_measurement(measure):
    """Measured on one design and stepped from a uniform one, a full step
    copies the measured profile in: the swirl as it is, and the velocity as a
    shape, at the level the uniform knots already had."""
    shaped = spline_config(stations=[{"Vm": VM_SHAPE, "rVt": tuple(DELTA)}] * 4)
    result = measure(shaped)
    inflow = {"type": "spanwise_inflow", "rows": [1], "gain": 1.0, "clip": 0.0}
    config = spline_config(stations=[{}] * 4, correct=[inflow])

    stepped = iterate.step(config, result)

    inlet, outlet = stepped.spanwise.stations[2:]
    mean_line = result.machine.mean_line
    for station, profile in ((mean_line[0, 1], inlet), (mean_line[1, 1], outlet)):
        level = spanwise.area_average(SPF, VM_SHAPE, station)
        np.testing.assert_allclose(profile.Vm, np.array(VM_SHAPE) / level, atol=2e-3)
    np.testing.assert_allclose(inlet.rVt, DELTA, atol=2e-3)
    assert outlet.rVt == (0.0,) * 3
    assert stepped.spanwise.stations[:2] == config.spanwise.stations[:2]


def _swirl_mean(result, Delta, Vm_shape=(1.0,) * 3, k=1):
    station = result.machine.mean_line[k % 2, k // 2]
    return spanwise.mass_average(SPF, Delta, Vm_shape, station)


def test_swirl_error_is_the_mean_the_target_cannot_set(measure):
    """Only the shape of a target is taken; its level is the mean line's.

    The fake flow has no endwall bands, so the whole span's mean is the
    knots' mean extrapolated to the walls, and differs from it only by the
    straight ends beyond the outermost sections.
    """
    config = spline_config(stations=[{}, {"rVt": tuple(DELTA)}, {}, {}])
    result = measure(config)

    swirl = iterate.SpanwiseSwirl(i_row=0, rVt=tuple(DELTA))
    m = swirl.measure(config, result)
    errors = []
    for offset in (0.0, 0.3):
        target = tuple(DELTA + offset)
        error = iterate.SpanwiseSwirl(i_row=0, rVt=target).error(config, result)
        errors.append(np.array(list(error.values())))
    np.testing.assert_allclose(errors[0], errors[1], atol=1e-12)
    np.testing.assert_allclose(
        errors[0], _swirl_mean(result, DELTA) - m["shift"], atol=2e-3
    )


def test_the_whole_span_mean_moves_the_target(measure, monkeypatch):
    """A band of swirl outside the outermost sections is invisible to the
    fit, but moves every knot's target by its share of the whole span."""
    config = spline_config(stations=[{}, {}, {}, {}])
    result = measure(config)
    profiles = fake_profiles(config)
    outlet = dict(profiles[1])
    station = result.machine.mean_line[1, 0]
    band = outlet["spf"] > 0.9
    outlet["rVt"] = outlet["rVt"] + band * 0.2 * float(station.r * station.Vm)
    profiles[1] = outlet

    monkeypatch.setattr(iterate, "station_profile", lambda result, k: profiles[k])

    swirl = iterate.SpanwiseSwirl(i_row=0, rVt=(0.0,) * 3)
    m = swirl.measure(config, result)
    error = np.array(list(swirl.error(config, result).values()))

    inner = slice(iterate.WALL_TRIM, band.size - iterate.WALL_TRIM)
    w = outlet["mdot"][inner]
    whole = np.sum(w * 0.2 * band[inner]) / np.sum(w)
    np.testing.assert_allclose(m["measured"], 0.0, atol=1e-6)
    np.testing.assert_allclose(m["shift"], -whole, atol=1e-6)
    np.testing.assert_allclose(error, whole, atol=1e-6)


def test_work_pairs_stream_surfaces_by_mass_fraction(measure):
    """The work a rotor does is its own, whatever arrives with the flow.

    The inlet carries a tilted enthalpy, and the inlet and outlet velocity
    shapes differ so a stream surface changes span fraction through the row.
    Paired by mass fraction, the tilt cancels and what is left is the Euler
    work the outlet knots set: asking for exactly that leaves only the mean.
    """
    stations = [{}, {}, {"Vm": VM_SHAPE[::-1]}, {"Vm": VM_SHAPE, "rVt": tuple(DELTA)}]
    config = spline_config(stations=stations)
    result = measure(config, ho_tilt=0.2)

    ml_out = result.machine.mean_line[1, 1]
    phi = float(ml_out.Vm / (ml_out.Omega * ml_out.r))
    work = iterate.SpanwiseWork(i_row=1, psi=tuple(phi * DELTA))
    m = work.measure(config, result)
    error = np.array(list(work.error(config, result).values()))

    np.testing.assert_allclose(m["measured"] - m["measured"].mean(), phi * (
        DELTA - DELTA.mean()
    ), atol=2e-3)
    np.testing.assert_allclose(error - error.mean(), 0.0, atol=2e-3)


def test_one_full_work_step_lands_on_the_target(measure):
    """The flow follows its knots here, so a full step is exact.

    Which checks the gain's sign as well as its size, on a turbine rotor
    whose blade speed and work are both of the opposite sign to a fan's.
    """
    psi = (0.04, 0.0, -0.03)
    work = {"type": "spanwise_work", "i_row": 1, "psi": psi, "gain": 1.0, "clip": 0.0}
    config = spline_config(correct=[work])

    before = iterate.measured_errors(config, measure(config))
    stepped = iterate.step(config, measure(config))
    after = iterate.measured_errors(stepped, measure(stepped))

    assert np.max(np.abs(list(before.values()))) > 0.01
    np.testing.assert_allclose(list(after.values()), 0.0, atol=1e-4)


def test_target_plot_draws_what_the_iterator_drives_to(measure):
    """A flow on which the iterator's error is zero sits on the plotted
    target, shift and all."""
    swirl = {"type": "spanwise_swirl", "i_row": 0, "rVt": list(DELTA)}
    config = spline_config(correct=[swirl])
    result = measure(config)
    iterator = config.iterate.correct[0]

    m = iterator.measure(config, result)
    (fig,) = post.SpanwiseTargetPlot().report(config, result)
    target_knots = fig.axes[0].lines[3]

    np.testing.assert_allclose(target_knots.get_ydata(), SPF)
    error = np.array(list(iterator.error(config, result).values()))
    np.testing.assert_allclose(
        m["measured"] - target_knots.get_xdata(), error, atol=1e-12
    )
    assert post.SpanwiseTargetPlot().report(spline_config(), result) == []


def test_a_station_list_round_trips():
    config = spline_config(stations=[{"Vm": VM_SHAPE, "rVt": tuple(DELTA)}] * 4)

    assert type(config).from_dict(config.to_dict()) == config
    assert dataclasses.replace(config) == config
