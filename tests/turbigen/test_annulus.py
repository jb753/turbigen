"""Tests for annulus design.

The annulus is the first stage that takes a result rather than a config node as
its input, and the first whose result is not a fixed-size container. So besides
the geometry itself, these check the shape of the stage interface: a design
produces an Annulus and stores nothing on itself, and a config with no annulus
designs a mean line alone.
"""

import dataclasses

import numpy as np
import pytest

import turbigen_ref.annulus
from turbigen import Annulus, Config, FixedAxialChord, Machine

FLUID = {"type": "perfect", "cp": 1005.0, "gamma": 1.4, "mu": 1.8e-5}
MEAN_LINE = {
    "type": "axial_turbine",
    "psi": 1.6,
    "phi2": 0.8,
    "Ma2": 0.9,
    "fac_Ma3_rel": 0.8,
    "mdot": 10.0,
    "Ys": [0.05, 0.05],
    "r_rms": 0.3,
}
ANNULUS = {
    "type": "fixed_axial_chord",
    "cx_row": [0.04, 0.04],
    "cx_gap": [0.06, 0.02, 0.08],
}
ANNULUS_AR = {
    "type": "aspect_ratio",
    "AR_row": [1.6, 1.6],
    "AR_gap": [1.0, 3.0, 0.8],
}


def build(**annulus):
    """A two-row config, with the annulus overridden as asked."""
    return Config.from_dict(
        {
            "fluid": FLUID,
            "mean_line": MEAN_LINE,
            "annulus": {**ANNULUS, **annulus},
        }
    )


def build_AR(**annulus):
    """The same two-row config, with the aspect-ratio annulus instead."""
    return Config.from_dict(
        {
            "fluid": FLUID,
            "mean_line": MEAN_LINE,
            "annulus": {**ANNULUS_AR, **annulus},
        }
    )


def segment_average(values):
    """Average a per-station array onto the segments, as the design does."""
    inner = 0.5 * (values[:-1] + values[1:])
    return np.concatenate([[values[0]], inner, [values[-1]]])


@pytest.fixture
def machine():
    return build().design()


#
# THE STAGE INTERFACE
#


def test_design_returns_a_machine_holding_both_stages(machine):
    assert isinstance(machine, Machine)
    assert isinstance(machine.annulus, Annulus)
    assert machine.mean_line is not None


def test_a_config_without_an_annulus_designs_the_mean_line_alone():
    """Depth comes from what is configured, not from an argument."""
    config = Config.from_dict({"fluid": FLUID, "mean_line": MEAN_LINE})

    machine = config.design()

    assert machine.annulus is None
    assert machine.mean_line is not None


def test_the_design_stores_nothing_on_itself():
    """A design is a frozen config; the fitted curves live on the result.

    The package this replaces wrote the splines onto the designer, so the
    config was its own result and could hold only one.
    """
    config = build()

    first = config.annulus.design(config.design().mean_line)
    second = config.annulus.design(config.design().mean_line)

    assert first is not second
    assert not hasattr(config.annulus, "_s")


def test_row_count_comes_from_the_mean_line():
    """An annulus is generic over row count, so it declares no n_row."""
    assert not hasattr(FixedAxialChord, "n_row")

    machine = build().design()

    assert machine.annulus.n_row == machine.mean_line.n_row
    assert machine.annulus.n_segment == 2 * machine.mean_line.n_row + 1


#
# GEOMETRY
#


def test_unmerged_annulus_passes_through_the_mean_line(machine):
    """With no merging the hub and casing hit every station exactly."""
    flat = machine.mean_line.flat
    annulus = machine.annulus

    np.testing.assert_allclose(annulus.r_mid, flat.r_mid, atol=1e-9)
    np.testing.assert_allclose(annulus.r_tip - annulus.r_hub, flat.span, atol=1e-9)


@pytest.mark.parametrize("weight", [0.25, 0.5, 1.0])
def test_merging_departs_from_the_intermediate_stations(weight):
    """merge_weight trades station fidelity for curvature smoothness."""
    machine = build(merge_weight=weight).design()
    flat = machine.mean_line.flat

    departure = np.abs(machine.annulus.r_mid - flat.r_mid).max()
    assert departure > 1e-6, "merging should move the intermediate stations"

    # The ends are shared by both fits, so they are untouched.
    assert machine.annulus.r_mid[0] == pytest.approx(flat.r_mid[0], abs=1e-9)
    assert machine.annulus.r_mid[-1] == pytest.approx(flat.r_mid[-1], abs=1e-9)


def test_merge_weight_is_a_parameter_not_a_type():
    """Zero merging is the same annulus the unmerged design would give.

    This is why there is one class rather than the two the old package had.
    """
    plain = build(merge_weight=0.0).design().annulus
    merged = build(merge_weight=0.5).design().annulus

    m = np.linspace(0.0, plain.m_max, 25)
    assert not np.allclose(plain.evaluate_xr(m, 0.5), merged.evaluate_xr(m, 0.5))


def test_chords_reproduce_the_requested_axial_chords(machine):
    """Row segments have the axial chord they were given."""
    xr = machine.annulus.evaluate_xr(np.array([1.0, 2.0, 3.0, 4.0]), 0.5)
    cx_row = np.diff(xr[0])[::2]

    np.testing.assert_allclose(cx_row, ANNULUS["cx_row"], rtol=1e-6)


def test_nozzle_ratio_scales_the_exit_span():
    narrow = build(nozzle_ratio=1.0).design().annulus
    wide = build(nozzle_ratio=1.5).design().annulus

    span_narrow = (narrow.r_tip - narrow.r_hub)[-1]
    span_wide = (wide.r_tip - wide.r_hub)[-1]

    assert span_wide > span_narrow


def test_geometry_is_self_consistent(machine):
    annulus = machine.annulus

    assert np.all(annulus.r_hub < annulus.r_tip)
    assert np.all(annulus.htr < 1.0)
    np.testing.assert_allclose(
        annulus.r_mid, 0.5 * (annulus.r_hub + annulus.r_tip), atol=1e-12
    )


def test_geometry_matches_the_mean_line_it_was_designed_from(machine):
    """The fit passes through the stations, so the two agree station by station.

    Area is the one worth stating explicitly. A mean line's `Am` is the true
    area normal to the meridional flow, `2 pi r_mid span`, whereas
    `pi (r_tip**2 - r_hub**2)` is that area projected onto the axis, smaller by
    cos(Beta). The two coincide only in axial flow, so comparing the annulus
    against the projection would hide the difference rather than pin it down.
    """
    annulus, flat = machine.annulus, machine.mean_line.flat
    m = np.arange(1, 2 * annulus.n_row + 1, dtype=float)
    span = annulus.evaluate_span(m)

    # Loose tolerances throughout: a mean line stores its state as float32.
    np.testing.assert_allclose(annulus.r_mid, flat.r_mid, rtol=1e-6)
    np.testing.assert_allclose(annulus.r_rms, flat.r, rtol=1e-6)
    np.testing.assert_allclose(span, flat.span, rtol=1e-6)
    np.testing.assert_allclose(2.0 * np.pi * annulus.r_mid * span, flat.Am, rtol=1e-6)


#
# VALIDATION
#


@pytest.mark.parametrize("weight", [-0.1, 1.1])
def test_merge_weight_outside_the_unit_interval_is_rejected(weight):
    with pytest.raises(ValueError, match=r"merge_weight.*\[0, 1\]"):
        build(merge_weight=weight).design()


def test_wrong_number_of_row_chords_is_rejected():
    with pytest.raises(ValueError, match="cx_row must have one value per row"):
        build(cx_row=[0.04, 0.04, 0.04]).design()


def test_wrong_number_of_gap_chords_is_rejected():
    with pytest.raises(ValueError, match="cx_gap must have one value per gap"):
        build(cx_gap=[0.06, 0.02]).design()


#
# SERIALISATION
#


def test_config_with_an_annulus_round_trips():
    config = build(merge_weight=0.3)

    assert Config.from_dict(config.to_dict()) == config


def test_an_absent_annulus_round_trips():
    """An optional stage that was not configured survives the round trip."""
    config = Config.from_dict({"fluid": FLUID, "mean_line": MEAN_LINE})

    assert config.annulus is None
    assert Config.from_dict(config.to_dict()) == config


def test_annulus_defaults_are_written_out():
    dumped = build().to_dict()["annulus"]

    assert dumped["nozzle_ratio"] == 1.0
    assert dumped["merge_weight"] == 0.0
    assert dumped["type"] == "fixed_axial_chord"


#
# EQUIVALENCE WITH THE EXISTING IMPLEMENTATION
#


@pytest.mark.parametrize("weight", [0.0, 0.3, 1.0])
def test_matches_the_turbigen_implementation(weight):
    machine = build(merge_weight=weight).design()
    flat = machine.mean_line.flat

    cx_row = np.array(ANNULUS["cx_row"])
    cx_gap = np.array(ANNULUS["cx_gap"])
    old = turbigen_ref.annulus.MergedFixedAxialChord(
        {"cx_row": cx_row, "cx_gap": cx_gap, "merge_weight": weight}
    )
    old.forward(
        np.asarray(flat.r_mid, dtype=float),
        np.asarray(flat.span, dtype=float),
        np.asarray(flat.Beta, dtype=float),
        cx_row=cx_row,
        cx_gap=cx_gap,
        merge_weight=weight,
    )

    m = np.linspace(0.0, old.mmax, 37)
    for spf in (0.0, 0.5, 1.0):
        np.testing.assert_allclose(
            machine.annulus.evaluate_xr(m, spf),
            old.evaluate_xr(m, spf),
            atol=1e-12,
            err_msg=f"differs from the turbigen annulus at spf={spf}",
        )


#
# REPORTING
#


def test_machine_reports_every_stage(machine):
    out = machine.to_string()

    assert "Mean line:" in out
    assert "Annulus:" in out


def test_machine_reports_only_what_was_designed():
    machine = Config.from_dict({"fluid": FLUID, "mean_line": MEAN_LINE}).design()

    out = machine.to_string()

    assert "Mean line:" in out
    assert "Annulus:" not in out


#
# THE ASPECT-RATIO DESIGN
#
# The second way of saying how long a segment is. Everything below the segment
# lengths is shared with fixed_axial_chord, so these check the specification
# itself and that the shared half really is shared.
#


def requested_lengths(flat):
    """Segment arc lengths the test config's aspect ratios ask for [m]."""
    span_avg = segment_average(np.asarray(flat.span, dtype=float))
    AR = np.empty(len(span_avg))
    AR[::2] = ANNULUS_AR["AR_gap"]
    AR[1::2] = ANNULUS_AR["AR_row"]
    return span_avg / AR


def test_aspect_ratios_place_the_stations():
    """Each segment gets the length its aspect ratio asked for.

    The span is the one averaged over the segment, which is what the design
    divides by, so a row's aspect ratio is set by the mean line at both ends of
    it rather than at either one.

    Checked on the axial coordinates, because those are exact: the control
    points are placed at the cumulative axial length and the fit passes through
    them. The counterpart for `fixed_axial_chord` measures the same thing the
    same way.
    """
    machine = build_AR().design()
    flat = machine.mean_line.flat

    Beta_avg = segment_average(np.asarray(flat.Beta, dtype=float))
    Dx = requested_lengths(flat) * np.cos(np.radians(Beta_avg))

    m = np.arange(machine.annulus.n_segment + 1, dtype=float)
    x_mid = machine.annulus.evaluate_xr(m, 0.5)[0]

    np.testing.assert_allclose(np.diff(x_mid), Dx, rtol=1e-6)


def test_arc_length_lands_close_to_the_aspect_ratio():
    """And the arc length the curve actually has agrees to about 0.1%.

    Not exactly, and not a defect: the PCHIP fit iterates the *parameterisation*
    until each segment's share of arc length matches its share of the target,
    normalised by the total. A segment's own length is therefore a fixed point
    of the fit rather than a constraint on it, and the curvature the fit puts in
    to pass through the stations moves it slightly. Stated here so that the gap
    is a recorded property rather than something rediscovered as a bug.
    """
    machine = build_AR().design()

    np.testing.assert_allclose(
        machine.annulus.evaluate_chords(0.5),
        requested_lengths(machine.mean_line.flat),
        rtol=1e-3,
    )


def test_aspect_ratio_passes_through_the_mean_line():
    """The station fit is the shared half, so it holds here too."""
    machine = build_AR().design()
    flat = machine.mean_line.flat

    np.testing.assert_allclose(machine.annulus.r_mid, flat.r_mid, atol=1e-9)


@pytest.mark.parametrize("weight", [0.25, 1.0])
def test_aspect_ratio_merges(weight):
    """merge_weight is on the shared base, so it reaches both designs."""
    machine = build_AR(merge_weight=weight).design()
    flat = machine.mean_line.flat

    departure = np.abs(machine.annulus.r_mid - flat.r_mid).max()
    assert departure > 1e-6, "merging should move the intermediate stations"


def test_aspect_ratio_takes_a_nozzle_ratio():
    narrow = build_AR(nozzle_ratio=1.0).design().annulus
    wide = build_AR(nozzle_ratio=1.5).design().annulus

    assert (wide.r_tip - wide.r_hub)[-1] > (narrow.r_tip - narrow.r_hub)[-1]


def test_the_two_designs_agree_when_they_ask_for_the_same_thing():
    """Stating a chord and stating its aspect ratio are two spellings of one
    number, so a design that resolves to the same segment lengths must give
    back the same annulus --- which is the whole reason there is one body."""
    chord = build_AR().design().annulus

    flat = build_AR().design().mean_line.flat
    span_avg = segment_average(np.asarray(flat.span, dtype=float))
    Beta_avg = segment_average(np.asarray(flat.Beta, dtype=float))
    AR = np.empty(len(span_avg))
    AR[::2] = ANNULUS_AR["AR_gap"]
    AR[1::2] = ANNULUS_AR["AR_row"]
    cx = span_avg / AR * np.cos(np.radians(Beta_avg))

    equivalent = build(cx_gap=list(cx[::2]), cx_row=list(cx[1::2])).design().annulus

    m = np.linspace(0.0, chord.m_max, 31)
    np.testing.assert_allclose(
        equivalent.evaluate_xr(m, 0.5), chord.evaluate_xr(m, 0.5), atol=1e-12
    )


#
# VALIDATION OF THE ASPECT-RATIO DESIGN
#


def test_wrong_number_of_row_aspect_ratios_is_rejected():
    with pytest.raises(ValueError, match="AR_row must have one value per row"):
        build_AR(AR_row=[1.6, 1.6, 1.6]).design()


def test_wrong_number_of_gap_aspect_ratios_is_rejected():
    with pytest.raises(ValueError, match="AR_gap must have one value per gap"):
        build_AR(AR_gap=[1.0, 3.0]).design()


@pytest.mark.parametrize("name", ["AR_row", "AR_gap"])
@pytest.mark.parametrize("value", [0.0, -0.4])
def test_non_positive_aspect_ratios_are_rejected_on_reading(name, value):
    """Rejected by the config, not by the design.

    It needs no mean line to see that it is wrong, and the old package gave a
    negative value a second meaning --- a segment whose length is chosen to
    smooth the curvature --- which is not ported, so it must fail rather than
    be quietly reinterpreted.
    """
    values = list(ANNULUS_AR[name])
    values[0] = value

    with pytest.raises(ValueError, match=f"{name} must be positive"):
        build_AR(**{name: values})


def test_the_shared_body_is_not_selectable():
    """PchipAnnulus is the shared half of two designs, not a third design, so
    it declares no type and cannot be named in a file."""
    with pytest.raises(ValueError, match="Unknown AnnulusDesign type"):
        build_AR(type="pchip_annulus")


#
# SERIALISATION OF THE ASPECT-RATIO DESIGN
#


def test_config_with_an_aspect_ratio_annulus_round_trips():
    config = build_AR(merge_weight=0.3)

    assert Config.from_dict(config.to_dict()) == config


def test_aspect_ratio_defaults_are_written_out():
    dumped = build_AR().to_dict()["annulus"]

    assert dumped["type"] == "aspect_ratio"
    assert dumped["nozzle_ratio"] == 1.0
    assert dumped["merge_weight"] == 0.0
    assert "cx_row" not in dumped


#
# EQUIVALENCE OF THE ASPECT-RATIO DESIGN
#


@pytest.mark.parametrize("weight", [0.0, 0.3, 1.0])
def test_aspect_ratio_matches_the_turbigen_implementation(weight):
    machine = build_AR(merge_weight=weight).design()
    flat = machine.mean_line.flat

    AR_chord = np.array(ANNULUS_AR["AR_row"])
    AR_gap = np.array(ANNULUS_AR["AR_gap"])
    old = turbigen_ref.annulus.Merged(
        {"AR_chord": AR_chord, "AR_gap": AR_gap, "merge_weight": weight}
    )
    old.forward(
        np.asarray(flat.r_mid, dtype=float),
        np.asarray(flat.span, dtype=float),
        np.asarray(flat.Beta, dtype=float),
        AR_chord=AR_chord,
        AR_gap=AR_gap,
        merge_weight=weight,
    )

    m = np.linspace(0.0, old.mmax, 37)
    for spf in (0.0, 0.5, 1.0):
        np.testing.assert_allclose(
            machine.annulus.evaluate_xr(m, spf),
            old.evaluate_xr(m, spf),
            atol=1e-12,
            err_msg=f"differs from the turbigen annulus at spf={spf}",
        )


#
# IMMUTABILITY
#
# Freezing the mean line closed only a third of the problem: an Annulus was a
# plain class, so anything holding one could rebind its merge weight or its
# fitted curves. Results are now uniformly frozen dataclasses, like Machine and
# Result already were.
#


def test_annulus_cannot_be_rebound(machine):
    with pytest.raises(dataclasses.FrozenInstanceError):
        machine.annulus.merge_weight = 0.9

    with pytest.raises(dataclasses.FrozenInstanceError):
        machine.annulus.curves = ()


def test_row_annulus_cannot_be_rebound(machine):
    with pytest.raises(dataclasses.FrozenInstanceError):
        machine.annulus.extract_row(0).chord = 1.0


def test_station_coordinates_are_cached(machine):
    """Every station property reads them, so they were evaluated ten times over
    to print five rows of a table. Caching is only safe because it is frozen."""
    annulus = machine.annulus

    assert annulus._xr_stations is annulus._xr_stations


def test_repr_stays_readable(machine):
    """A generated repr would dump the spline objects and the whole knot array,
    so the bulky fields are excluded from it."""
    text = repr(machine.annulus)

    assert text == f"Annulus(merge_weight=0.0, n_row={machine.annulus.n_row})"
