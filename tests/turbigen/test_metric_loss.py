"""Tests for the profile and secondary loss breakdown metric.

Two cases, as for the other metrics. The single-row cascade is marched, so its
row has a real loss to split, and it does not rotate. The two-row stage is not
marched: its initial guess already carries the design entropy rise, it has a
rotor, and so it has a work to refer lost efficiency to.

Test cases:
- test_nothing_to_measure_returns_nothing: the Metric contract
- test_a_diverged_march_is_not_measured: a field of NaN has no band to mix out
- test_keys_and_shape: (n_row,), one value per row
- test_the_parts_sum_to_the_mixed_out_total: mid plus secondary is actual
- test_the_profile_loss_is_the_mixed_out_band: the metric is its definition
- test_the_whole_span_as_the_band_leaves_little_secondary: band=1 is the total
- test_lost_efficiency_is_the_same_entropy_referred_to_the_machine: one factor
- test_rows_and_gaps_add_up_to_the_machine: lost efficiency is additive
- test_a_cascade_has_no_efficiency_to_lose: NaN without work, Y still finite
- test_values_survive_the_case_file: floats and NaN both
- test_cut_structured_resolution: the brute-force regrid, and None off the grid
"""

import dataclasses

import ember.average
import numpy as np
import pytest
import yaml
from test_blade import build
from test_cli import RUN_CASE
from test_mesh import MESH, TIP

from turbigen import Config, Result, case, metric, mixout, pipeline, util
from turbigen.metric import LossBreakdown

KEYS = {"Ys_mid", "Ys_sec", "Deta_mid", "Deta_sec"}


@pytest.fixture(scope="module")
def gapped():
    """The two-row tip-gap stage on its initial guess, never marched."""
    config = build(blades=TIP, mesh=MESH)
    _, machine, grid = pipeline.prepare(config)
    actual, Ds_mix = mixout.mean_line(grid, machine)
    return config, Result(machine=machine, grid=grid, actual=actual, Ds_mix=Ds_mix)


@pytest.fixture(scope="module")
def solved():
    """The fast cascade, marched briefly for a real field and a real loss."""
    config = Config.from_dict(yaml.safe_load(RUN_CASE))
    _, machine, grid = pipeline.prepare(config)
    history = config.solver.solve(grid)
    actual, Ds_mix = mixout.mean_line(grid, machine)
    return config, Result(
        machine=machine,
        grid=grid,
        actual=actual,
        Ds_mix=Ds_mix,
        converged=True,
        history=history,
    )


def _Y_per_Ds(actual, i_row):
    """Row exit temperature over the characteristic relative dynamic head."""
    ref = actual.get_characteristic_station(i_row)
    return float(actual[:, i_row].T[-1]) / float(ref.halfVsq_rel)


def _Ds_total(actual, i_row):
    return float(actual.s[1, i_row] - actual.s[0, i_row])


#
# THE CONTRACT
#


@pytest.mark.parametrize("missing", ["grid", "machine", "actual"])
def test_nothing_to_measure_returns_nothing(solved, missing):
    """An observation of a field that is not there is not an error."""
    config, result = solved
    result = dataclasses.replace(result, **{missing: None})

    assert LossBreakdown().evaluate(config, result) == {}


def test_a_diverged_march_is_not_measured(solved):
    """A blown-up field is NaN throughout and no band can be found in it."""
    config, result = solved
    history = result.history.copy()
    history.diverged = True

    values = LossBreakdown().evaluate(
        config, dataclasses.replace(result, history=history)
    )

    assert values == {}


def test_keys_and_shape(gapped):
    """One value per row, for each of the four."""
    config, result = gapped

    values = LossBreakdown().evaluate(config, result)

    assert set(values) == KEYS
    for name, value in values.items():
        assert np.shape(value) == (result.grid.n_row,), name
        assert np.all(np.isfinite(value)), name


#
# WHAT THE NUMBERS OBEY
#


@pytest.mark.parametrize("fixture", ["solved", "gapped"])
def test_the_parts_sum_to_the_mixed_out_total(fixture, request):
    """Profile plus secondary is the row's loss in the mixed-out mean line."""
    config, result = request.getfixturevalue(fixture)

    values = LossBreakdown().evaluate(config, result)

    for i_row in range(result.grid.n_row):
        total = values["Ys_mid"][i_row] + values["Ys_sec"][i_row]
        expected = _Y_per_Ds(result.actual, i_row) * _Ds_total(result.actual, i_row)
        assert total == pytest.approx(expected, rel=1e-12)


def test_the_profile_loss_is_the_mixed_out_band(solved):
    """Cut, band and mix out at both planes, done here rather than by the metric."""
    config, result = solved
    i_row = 0
    planes = result.machine.annulus.cut_planes()

    s = []
    for xr in planes[2 * i_row : 2 * i_row + 2]:
        cut = util.cut_structured(result.grid, xr)
        band = ember.average.mass_band(cut, 0.45, 0.55)
        s.append(float(ember.average.mix_out(band).s))

    values = LossBreakdown(band=0.1).evaluate(config, result)

    expected = _Y_per_Ds(result.actual, i_row) * (s[1] - s[0])
    assert values["Ys_mid"][i_row] == pytest.approx(expected, rel=1e-12)
    assert values["Ys_mid"][i_row] > 0.0


def test_the_whole_span_as_the_band_leaves_little_secondary(solved):
    """A band of all the flow is the whole plane mixed out, which is the total.

    Not exactly: the mean line mixes out the raw cut and the band is taken from
    the regridded one. What is left over is the regrid's error, and it should
    be small beside the loss.
    """
    config, result = solved

    values = LossBreakdown(band=1.0).evaluate(config, result)

    total = values["Ys_mid"][0] + values["Ys_sec"][0]
    assert abs(values["Ys_sec"][0]) < 0.05 * abs(total)


def test_lost_efficiency_is_the_same_entropy_referred_to_the_machine(gapped):
    """Deta and Y differ by one factor per row, and no more."""
    config, result = gapped
    actual = result.actual
    eta_per_Ds = float(actual.outlet.T) / abs(float(actual.Dho))

    values = LossBreakdown().evaluate(config, result)

    for i_row in range(result.grid.n_row):
        ratio = eta_per_Ds / _Y_per_Ds(actual, i_row)
        for part in ("mid", "sec"):
            assert values[f"Deta_{part}"][i_row] == pytest.approx(
                ratio * values[f"Ys_{part}"][i_row], rel=1e-12
            )


def test_rows_and_gaps_add_up_to_the_machine(gapped):
    """Lost efficiencies add: the rows, plus the gaps between them, are the machine.

    The gaps belong to no row, so they are added here from the mean line. That
    they close the sum exactly is what says every row has the same denominator.
    """
    config, result = gapped
    actual = result.actual
    eta_per_Ds = float(actual.outlet.T) / abs(float(actual.Dho))

    values = LossBreakdown().evaluate(config, result)

    rows = np.sum(values["Deta_mid"]) + np.sum(values["Deta_sec"])
    gaps = eta_per_Ds * float(np.sum(actual.s[0, 1:] - actual.s[1, :-1]))
    machine = eta_per_Ds * float(actual.outlet.s - actual.inlet.s)
    assert rows + gaps == pytest.approx(machine, rel=1e-12)


def test_a_cascade_has_no_efficiency_to_lose(solved):
    """No row rotates, so there is no work, and no efficiency to refer to it."""
    config, result = solved

    values = LossBreakdown().evaluate(config, result)

    assert np.all(np.isnan(values["Deta_mid"]))
    assert np.all(np.isnan(values["Deta_sec"]))
    assert np.all(np.isfinite(values["Ys_mid"]))
    assert np.all(np.isfinite(values["Ys_sec"]))


#
# WHAT IS KEPT
#


def test_values_survive_the_case_file(solved, tmp_path):
    """Written as plain numbers, and read back as the same ones --- NaN too."""
    config, result = solved
    config = dataclasses.replace(config, metrics=(LossBreakdown(),))
    result = dataclasses.replace(result, metrics=metric.measure(config, result))

    assert result.metrics, "the metric has to have run for this to say anything"

    path = tmp_path / "case.yaml"
    case.write(path, config, result)
    _, read_back = case.read(path)

    for name, value in result.metrics.items():
        np.testing.assert_allclose(read_back.metrics[name], value)
    assert set(read_back.metrics) == KEYS


#
# THE CUT IT READS
#


def test_cut_structured_resolution(solved):
    """Regridded at the brute-force resolution, and None where nothing is cut."""
    _, result = solved
    planes = result.machine.annulus.cut_planes()

    cut = util.cut_structured(result.grid, planes[0])
    assert cut.shape == (util.N_CUT_SPAN, util.N_CUT_PITCH)

    far = np.asarray(planes[0]) + np.array([100.0, 0.0])
    assert util.cut_structured(result.grid, far) is None
