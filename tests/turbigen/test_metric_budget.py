"""Tests for the lost efficiency budget metric.

The same two cases as the profile and secondary breakdown, whose fixtures these
borrow. The two-row stage has a rotor, so a work to refer lost efficiency to,
a tip gap, and a gap between rows; the cascade does not rotate.

Test cases:
- test_nothing_to_measure_returns_nothing: the Metric contract
- test_a_diverged_march_is_not_measured: a field of NaN is not averaged
- test_keys_and_shape: bands per row, gaps between rows, mixing at the ends
- test_a_cascade_has_no_efficiency_to_lose: NaN without work
- test_the_fraction_leaves_a_main_band: 0 < fraction < 0.5
- test_the_budget_adds_up_to_eta_tt: the point of the metric
- test_the_bands_add_up_to_the_plane: mass averages need no mixing term
- test_a_band_is_its_mass_averaged_entropy: the metric is its definition
- test_a_gap_is_its_mass_averaged_entropy: likewise, between the rows
- test_mixing_is_the_ends_of_Ds_mix: charged at the exit, credited at the inlet
- test_values_survive_the_case_file: floats, through the config by type name
- test_the_old_type_name_still_reads: case files written as loss_span
- test_no_key_is_shared_with_the_breakdown: both are configured together
"""

import dataclasses

import ember.average
import numpy as np
import pytest
from test_metric_loss import gapped, solved  # noqa: F401

from turbigen import case, metric, util
from turbigen.metric import LossBudget, LossSpan

BANDS = ("Deta_hub", "Deta_main", "Deta_tip")
KEYS = {*BANDS, "Deta_gap", "Deta_mix_out", "Deta_mix_in"}


def _per_Ds(actual):
    """The factor taking an entropy rise to a share of 1 - eta_tt."""
    return (1.0 - float(actual.eta_tt)) / float(actual.outlet.s - actual.inlet.s)


def _plane_entropy(grid, xr, lo=0.0, hi=1.0):
    """Mass-averaged entropy of the band `lo..hi` of a plane, done by hand."""
    cut = util.cut_structured(grid, xr)
    if (lo, hi) != (0.0, 1.0):
        cut = ember.average.mass_band(cut, lo, hi, axis=0)
    return float(ember.average.mass_average(cut.s, cut))


def _total(values):
    """The budget summed: bands, gaps and mixing, the inlet's a credit."""
    return (
        sum(np.sum(values[name]) for name in BANDS)
        + np.sum(values["Deta_gap"])
        + values["Deta_mix_out"]
        - values["Deta_mix_in"]
    )


#
# THE CONTRACT
#


@pytest.mark.parametrize("missing", ["grid", "machine", "actual", "Ds_mix"])
def test_nothing_to_measure_returns_nothing(solved, missing):  # noqa: F811
    """An observation of a field that is not there is not an error."""
    config, result = solved
    result = dataclasses.replace(result, **{missing: None})

    assert LossBudget().evaluate(config, result) == {}


def test_a_diverged_march_is_not_measured(solved):  # noqa: F811
    """A blown-up field is NaN throughout and has no entropy to average."""
    config, result = solved
    history = result.history.copy()
    history.diverged = True

    values = LossBudget().evaluate(config, dataclasses.replace(result, history=history))

    assert values == {}


def test_keys_and_shape(gapped):  # noqa: F811
    """A value per row for each band, one per gap, and one for each end."""
    config, result = gapped
    n_row = result.grid.n_row

    values = LossBudget().evaluate(config, result)

    assert set(values) == KEYS
    shapes = {name: (n_row,) for name in BANDS}
    shapes.update(Deta_gap=(n_row - 1,), Deta_mix_out=(), Deta_mix_in=())
    for name, value in values.items():
        assert np.shape(value) == shapes[name], name
        assert np.all(np.isfinite(value)), name


def test_a_cascade_has_no_efficiency_to_lose(solved):  # noqa: F811
    """No row rotates, so there is no work, and no efficiency to refer to it."""
    config, result = solved

    values = LossBudget().evaluate(config, result)

    assert set(values) == KEYS
    for value in values.values():
        assert np.all(np.isnan(value))


@pytest.mark.parametrize("fraction", [0.0, 0.5, -0.1, 0.7])
def test_the_fraction_leaves_a_main_band(fraction):
    """Two endwall bands of half the flow each would leave nothing between."""
    with pytest.raises(ValueError, match="fraction"):
        LossBudget(fraction=fraction)


#
# WHAT THE NUMBERS OBEY
#


@pytest.mark.parametrize("fraction", [0.1, 0.2, 0.3])
def test_the_budget_adds_up_to_eta_tt(gapped, fraction):  # noqa: F811
    """Bands, gaps and end mixing are the mixed-out loss, in the currency of
    the efficiency itself, to the accuracy of the band split."""
    config, result = gapped

    values = LossBudget(fraction=fraction).evaluate(config, result)

    assert _total(values) == pytest.approx(1.0 - float(result.actual.eta_tt), rel=1e-4)


@pytest.mark.parametrize("fraction", [0.1, 0.2, 0.3])
def test_the_bands_add_up_to_the_plane(gapped, fraction):  # noqa: F811
    """Mass averages are additive, so the three bands are the row's whole loss.

    Mixed-out bands would not be: putting them back together would generate
    entropy of its own. Not to round-off, though: `mass_band` carries its mass
    exactly, but interpolates the entropy linearly across the partial strips at
    its ends, which the whole plane's average does not.
    """
    config, result = gapped
    per_Ds = _per_Ds(result.actual)
    planes = result.machine.annulus.cut_planes()

    values = LossBudget(fraction=fraction).evaluate(config, result)

    for i_row in range(result.grid.n_row):
        s_in, s_out = (
            _plane_entropy(result.grid, xr) for xr in planes[2 * i_row : 2 * i_row + 2]
        )
        total = sum(values[name][i_row] for name in BANDS)
        assert total == pytest.approx(per_Ds * (s_out - s_in), rel=1e-4)


def test_a_band_is_its_mass_averaged_entropy(gapped):  # noqa: F811
    """Cut, band and mass average at both planes, done here rather than by the
    metric, for the tip band of the rotor."""
    config, result = gapped
    i_row = 1
    planes = result.machine.annulus.cut_planes()

    s = [
        _plane_entropy(result.grid, xr, 0.8, 1.0)
        for xr in planes[2 * i_row : 2 * i_row + 2]
    ]

    values = LossBudget(fraction=0.2).evaluate(config, result)

    expected = 0.2 * _per_Ds(result.actual) * (s[1] - s[0])
    assert values["Deta_tip"][i_row] == pytest.approx(expected, rel=1e-12)


def test_a_gap_is_its_mass_averaged_entropy(gapped):  # noqa: F811
    """From the stator's exit plane to the rotor's inlet plane, by hand."""
    config, result = gapped
    planes = result.machine.annulus.cut_planes()

    s = [_plane_entropy(result.grid, xr) for xr in planes[1:3]]

    values = LossBudget().evaluate(config, result)

    expected = _per_Ds(result.actual) * (s[1] - s[0])
    assert values["Deta_gap"][0] == pytest.approx(expected, rel=1e-6)


def test_mixing_is_the_ends_of_Ds_mix(gapped):  # noqa: F811
    """The machine's exit plane is charged and its inlet plane credited; the
    planes between rows do not appear."""
    config, result = gapped
    per_Ds = _per_Ds(result.actual)

    values = LossBudget().evaluate(config, result)

    assert values["Deta_mix_out"] == pytest.approx(per_Ds * result.Ds_mix[1, -1])
    assert values["Deta_mix_in"] == pytest.approx(per_Ds * result.Ds_mix[0, 0])


#
# WHAT IS KEPT
#


def test_values_survive_the_case_file(gapped, tmp_path):  # noqa: F811
    """Written as plain numbers and read back as the same, with the metric
    itself found again by its type name."""
    config, result = gapped
    config = dataclasses.replace(config, metrics=(LossBudget(fraction=0.15),))
    result = dataclasses.replace(result, metrics=metric.measure(config, result))

    assert set(result.metrics) == KEYS

    path = tmp_path / "case.yaml"
    case.write(path, config, result)
    config_back, read_back = case.read(path)

    assert config_back.metrics == (LossBudget(fraction=0.15),)
    for name, value in result.metrics.items():
        np.testing.assert_allclose(read_back.metrics[name], value)


def test_the_old_type_name_still_reads(gapped, tmp_path):  # noqa: F811
    """Runs in the database were written with ``type: loss_span``, and a case
    file that does not read is skipped as if it had never run."""
    config, result = gapped
    config = dataclasses.replace(config, metrics=(LossSpan(fraction=0.15),))

    path = tmp_path / "case.yaml"
    case.write(path, config, result)
    config_back, _ = case.read(path)

    assert config_back.metrics == (LossSpan(fraction=0.15),)
    assert isinstance(config_back.metrics[0], LossBudget)


def test_no_key_is_shared_with_the_breakdown(gapped):  # noqa: F811
    """A sweep measures both, and `measure` keeps only the last of two values
    written under one name."""
    config, result = gapped

    budget = LossBudget().evaluate(config, result)
    breakdown = metric.LossBreakdown().evaluate(config, result)

    assert not set(budget) & set(breakdown)
