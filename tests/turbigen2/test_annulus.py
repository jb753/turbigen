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

import turbigen.annulus
from turbigen2 import Annulus, Config, FixedAxialChord, Machine

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


def build(**annulus):
    """A two-row config, with the annulus overridden as asked."""
    return Config.from_dict(
        {
            "fluid": FLUID,
            "mean_line": MEAN_LINE,
            "annulus": {**ANNULUS, **annulus},
        }
    )


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
    np.testing.assert_allclose(
        annulus.r_tip - annulus.r_hub, flat.span, atol=1e-9
    )


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

    m = np.linspace(0.0, plain.mmax, 25)
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
    span = annulus.span(m)

    # Loose tolerances throughout: a mean line stores its state as float32.
    np.testing.assert_allclose(annulus.r_mid, flat.r_mid, rtol=1e-6)
    np.testing.assert_allclose(annulus.r_rms, flat.r, rtol=1e-6)
    np.testing.assert_allclose(span, flat.span, rtol=1e-6)
    np.testing.assert_allclose(
        2.0 * np.pi * annulus.r_mid * span, flat.Am, rtol=1e-6
    )


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
    old = turbigen.annulus.MergedFixedAxialChord(
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
    machine = Config.from_dict(
        {"fluid": FLUID, "mean_line": MEAN_LINE}
    ).design()

    out = machine.to_string()

    assert "Mean line:" in out
    assert "Annulus:" not in out


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


def test_stream_surface_cannot_be_rebound(machine):
    with pytest.raises(dataclasses.FrozenInstanceError):
        machine.annulus.row(0).chord = 1.0


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
