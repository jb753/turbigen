"""Tests for the hub, main and tip lost efficiency metric.

The same two cases as the profile and secondary breakdown, whose fixtures these
borrow. The two-row stage has a rotor, so a work to refer lost efficiency to,
and a tip gap; the cascade does not rotate.

Test cases:
- test_nothing_to_measure_returns_nothing: the Metric contract
- test_a_diverged_march_is_not_measured: a field of NaN is not averaged
- test_keys_and_shape: (n_row,), one value per row
- test_a_cascade_has_no_efficiency_to_lose: NaN without work
- test_the_fraction_leaves_a_main_band: 0 < fraction < 0.5
- test_the_bands_add_up_to_the_plane: mass averages need no mixing term
- test_a_band_is_its_mass_averaged_entropy: the metric is its definition
- test_rows_gaps_and_mixing_add_up_to_the_machine: closes on the mean line
- test_values_survive_the_case_file: floats, through the config by type name
"""

import dataclasses

import ember.average
import numpy as np
import pytest
from test_metric_loss import gapped, solved  # noqa: F401

from turbigen import case, metric, util
from turbigen.metric import LossSpan

KEYS = {"Deta_hub", "Deta_main", "Deta_tip"}


def _eta_per_Ds(actual):
    return float(actual.outlet.T) / abs(float(actual.Dho))


def _plane_entropy(grid, xr, lo=0.0, hi=1.0):
    """Mass-averaged entropy of the band `lo..hi` of a plane, done by hand."""
    cut = util.cut_structured(grid, xr)
    if (lo, hi) != (0.0, 1.0):
        cut = ember.average.mass_band(cut, lo, hi, axis=0)
    return float(ember.average.mass_average(cut.s, cut))


#
# THE CONTRACT
#


@pytest.mark.parametrize("missing", ["grid", "machine", "actual"])
def test_nothing_to_measure_returns_nothing(solved, missing):  # noqa: F811
    """An observation of a field that is not there is not an error."""
    config, result = solved
    result = dataclasses.replace(result, **{missing: None})

    assert LossSpan().evaluate(config, result) == {}


def test_a_diverged_march_is_not_measured(solved):  # noqa: F811
    """A blown-up field is NaN throughout and has no entropy to average."""
    config, result = solved
    history = result.history.copy()
    history.diverged = True

    values = LossSpan().evaluate(config, dataclasses.replace(result, history=history))

    assert values == {}


def test_keys_and_shape(gapped):  # noqa: F811
    """One value per row, for each of the three bands."""
    config, result = gapped

    values = LossSpan().evaluate(config, result)

    assert set(values) == KEYS
    for name, value in values.items():
        assert np.shape(value) == (result.grid.n_row,), name
        assert np.all(np.isfinite(value)), name


def test_a_cascade_has_no_efficiency_to_lose(solved):  # noqa: F811
    """No row rotates, so there is no work, and no efficiency to refer to it."""
    config, result = solved

    values = LossSpan().evaluate(config, result)

    assert set(values) == KEYS
    for value in values.values():
        assert np.all(np.isnan(value))


@pytest.mark.parametrize("fraction", [0.0, 0.5, -0.1, 0.7])
def test_the_fraction_leaves_a_main_band(fraction):
    """Two endwall bands of half the flow each would leave nothing between."""
    with pytest.raises(ValueError, match="fraction"):
        LossSpan(fraction=fraction)


#
# WHAT THE NUMBERS OBEY
#


@pytest.mark.parametrize("fraction", [0.1, 0.2, 0.3])
def test_the_bands_add_up_to_the_plane(gapped, fraction):  # noqa: F811
    """Mass averages are additive, so the three bands are the row's whole loss.

    Mixed-out bands would not be: putting them back together would generate
    entropy of its own. Not to round-off, though: `mass_band` carries its mass
    exactly, but interpolates the entropy linearly across the partial strips at
    its ends, which the whole plane's average does not.
    """
    config, result = gapped
    eta_per_Ds = _eta_per_Ds(result.actual)
    planes = result.machine.annulus.cut_planes()

    values = LossSpan(fraction=fraction).evaluate(config, result)

    for i_row in range(result.grid.n_row):
        s_in, s_out = (
            _plane_entropy(result.grid, xr) for xr in planes[2 * i_row : 2 * i_row + 2]
        )
        total = sum(values[name][i_row] for name in KEYS)
        assert total == pytest.approx(eta_per_Ds * (s_out - s_in), rel=1e-4)


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

    values = LossSpan(fraction=0.2).evaluate(config, result)

    expected = 0.2 * _eta_per_Ds(result.actual) * (s[1] - s[0])
    assert values["Deta_tip"][i_row] == pytest.approx(expected, rel=1e-12)


def test_rows_gaps_and_mixing_add_up_to_the_machine(gapped):  # noqa: F811
    """The bands of every row, the gaps between rows, and the mixing loss at
    the two ends of the machine are its mixed-out loss.

    The mixing loss is the mixed-out entropy less the mass average on the same
    regridded cut, so it bridges this metric's mass averages and the mixed-out
    mean line exactly.
    """
    config, result = gapped
    actual = result.actual
    eta_per_Ds = _eta_per_Ds(actual)
    planes = result.machine.annulus.cut_planes()
    s_bar = [_plane_entropy(result.grid, xr) for xr in planes]

    values = LossSpan().evaluate(config, result)

    rows = sum(np.sum(values[name]) for name in KEYS)
    gaps = eta_per_Ds * float(np.sum(np.array(s_bar[2::2]) - np.array(s_bar[1:-1:2])))
    mixing = eta_per_Ds * float(result.Ds_mix[1, -1] - result.Ds_mix[0, 0])
    machine = eta_per_Ds * float(actual.outlet.s - actual.inlet.s)
    assert rows + gaps + mixing == pytest.approx(machine, rel=1e-6)


#
# WHAT IS KEPT
#


def test_values_survive_the_case_file(gapped, tmp_path):  # noqa: F811
    """Written as plain numbers and read back as the same, with the metric
    itself found again by its type name."""
    config, result = gapped
    config = dataclasses.replace(config, metrics=(LossSpan(fraction=0.15),))
    result = dataclasses.replace(result, metrics=metric.measure(config, result))

    assert set(result.metrics) == KEYS

    path = tmp_path / "case.yaml"
    case.write(path, config, result)
    config_back, read_back = case.read(path)

    assert config_back.metrics == (LossSpan(fraction=0.15),)
    for name, value in result.metrics.items():
        np.testing.assert_allclose(read_back.metrics[name], value)
