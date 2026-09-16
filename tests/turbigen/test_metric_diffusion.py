"""Tests for the diffusion factor metric.

Two cases, as for the surface dissipation. The single-row cascade is marched,
so it has a real surface distribution with a peak in it. The two-row tip-gap
machine is not marched: it is here for shape and for the section above the
clearance gap, which is the one place a blade has no surface to cut.

Test cases:
- test_nothing_to_measure_returns_nothing: the Metric contract
- test_a_diverged_march_is_not_measured: a field of NaN has no distribution
- test_keys_and_shape: (n_spf, n_row), span fractions then rows
- test_it_measures_what_the_surface_plot_draws: the suction peak over the exit
- test_the_peak_sits_on_the_suction_surface: argmax of the data, no fit
- test_it_reads_what_clark_profile_reads: one measurement, not a second opinion
- test_a_section_above_the_gap_is_unmeasured: NaN rather than a wrong number
- test_a_peakless_blade_still_gets_a_diffusion_factor: the maximum always exists
- test_values_survive_the_case_file: floats and NaN both
"""

import dataclasses

import numpy as np
import pytest
import yaml
from test_blade import build
from test_cli import RUN_CASE
from test_mesh import MESH, TIP

from turbigen import Config, Result, case, cli, loading, metric, mixout, util
from turbigen.metric import DiffusionFactor

TIP_ROW = 1
"""The row of the two-row fixture that has a clearance gap."""


@pytest.fixture(scope="module")
def gapped():
    """The two-row tip-gap machine on its initial guess, never marched."""
    config = build(blades=TIP, mesh=MESH)
    _, machine, grid = cli.prepare(config)
    return config, Result(machine=machine, grid=grid)


@pytest.fixture(scope="module")
def solved():
    """The fast cascade, marched briefly for a real surface distribution."""
    config = Config.from_dict(yaml.safe_load(RUN_CASE))
    _, machine, grid = cli.prepare(config)
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


#
# THE CONTRACT
#


@pytest.mark.parametrize("missing", ["grid", "machine"], ids=["no_grid", "no_machine"])
def test_nothing_to_measure_returns_nothing(solved, missing):
    """An observation of a field that is not there is not an error."""
    config, result = solved
    result = dataclasses.replace(result, **{missing: None})

    assert DiffusionFactor().evaluate(config, result) == {}


def test_a_diverged_march_is_not_measured(solved):
    """A blown-up field is NaN throughout and no state can be found in it."""
    config, result = solved
    history = result.history.copy()
    history.diverged = True

    values = DiffusionFactor().evaluate(
        config, dataclasses.replace(result, history=history)
    )

    assert values == {}


def test_keys_and_shape(solved):
    """One row per column, one span fraction per row of the array."""
    config, result = solved
    spf = (0.3, 0.5, 0.7)

    values = DiffusionFactor(spf=spf).evaluate(config, result)

    assert set(values) == {"DF", "zeta_peak", "Co"}
    for name, value in values.items():
        assert np.shape(value) == (len(spf), result.grid.n_row), name
        assert np.all(np.isfinite(value)), name


#
# WHAT THE NUMBERS OBEY
#


def test_it_measures_what_the_surface_plot_draws(solved):
    """The suction peak over the exit, less one, off the curve the plot draws.

    The exit value is checked against a cut taken here rather than by the
    metric, so `DF` is that curve reduced rather than a second definition that
    happens to be close. The peak is the suction surface's own, as `Ma_peak` is
    for `ClarkProfile` --- so on a blade that accelerates to its trailing edge,
    like this barely-marched one, `DF` can sit a little below zero, the suction
    side there reading under the mean of the two.
    """
    config, result = solved
    spf = 0.5
    i_row = 0

    mas = _distribution(result, i_row, spf)
    measured = loading.surfaces(result, i_row, spf)

    values = DiffusionFactor(spf=(spf,)).evaluate(config, result)

    assert measured.ma_TE == pytest.approx(0.5 * float(mas[0] + mas[-1]))
    assert values["DF"][0][i_row] == pytest.approx(
        float(measured.ma[0].max()) / measured.ma_TE - 1.0
    )


def _distribution(result, i_row, spf):
    """Return the isentropic Mach number round a section, cut here."""
    import ember.cut

    surface = util.cut_blade_surfs(result.grid, 0)[i_row][0][:, :, None]
    m = np.linspace(2 * i_row + 1, 2 * i_row + 2, util.N_SPAN_CUT)
    xr = result.machine.annulus.evaluate_xr(m, spf)
    cut = ember.cut.structured_meridional(surface, xr.T)[0]

    return util.isentropic_mach(cut, result.machine.mean_line[:, i_row].s[0])[:, 0]


def test_the_peak_sits_on_the_suction_surface(solved):
    """The node the maximum is on, on Clark's own axis, and nothing fitted."""
    config, result = solved
    spf = 0.5

    values = DiffusionFactor(spf=(spf,)).evaluate(config, result)
    measured = loading.surfaces(result, 0, spf)

    i_peak = int(np.argmax(measured.ma[0]))
    assert values["zeta_peak"][0][0] == pytest.approx(measured.z[0][i_peak])
    assert 0.0 <= values["zeta_peak"][0][0] <= 1.0


def test_it_reads_what_clark_profile_reads(solved):
    """The circulation is the one the iterator drives the blade count with."""
    config, result = solved
    spf = 0.5

    values = DiffusionFactor(spf=(spf,)).evaluate(config, result)
    iterated = loading.measure_clark_profile(result, 0, spf, np.array([0.5]))

    assert values["Co"][0][0] == pytest.approx(iterated.Co)
    assert values["Co"][0][0] > 0.0


def test_a_section_above_the_gap_is_unmeasured(gapped):
    """A blade with no surface at a span has no diffusion there, not zero."""
    config, result = gapped

    values = DiffusionFactor(spf=(0.5, 0.999)).evaluate(config, result)

    for name in ("DF", "zeta_peak", "Co"):
        value = np.asarray(values[name])
        assert np.isfinite(value[0, TIP_ROW]), "mid-span is below the gap"
        assert np.isnan(value[1, TIP_ROW]), "the tip section is trimmed off"


def test_a_peakless_blade_still_gets_a_diffusion_factor(gapped):
    """An accelerating cascade has no interior peak, and still diffuses.

    Its maximum is at the trailing edge, so `DF` is small rather than missing:
    a metric that has to describe every blade cannot use a number that only
    some blades have.
    """
    config, result = gapped
    values = DiffusionFactor().evaluate(config, result)

    assert np.isfinite(np.asarray(values["DF"])[0, TIP_ROW])


#
# WHAT IS KEPT
#


def test_values_survive_the_case_file(gapped, tmp_path):
    """Written as plain numbers, and read back as the same ones --- NaN too."""
    config, result = gapped
    config = dataclasses.replace(config, metrics=(DiffusionFactor(spf=(0.5, 0.999)),))
    result = dataclasses.replace(result, metrics=metric.measure(config, result))

    assert result.metrics, "the metric has to have run for this to say anything"

    path = tmp_path / "case.yaml"
    case.write(path, config, result)
    _, read_back = case.read(path)

    for name, value in result.metrics.items():
        np.testing.assert_allclose(read_back.metrics[name], value)
    assert set(read_back.metrics) == {"DF", "zeta_peak", "Co"}
