"""Tests for the diffusion factor metric.

Two cases, as for the surface dissipation. The single-row cascade is marched,
so it has a real surface distribution with a peak in it. The two-row tip-gap
machine is not marched: it is here for shape and for the section above the
clearance gap, which is the one place a blade has no surface to cut.

Test cases:
- test_nothing_to_measure_returns_nothing: the Metric contract
- test_a_diverged_march_is_not_measured: a field of NaN has no distribution
- test_keys_and_shape: (n_spf, n_row), span fractions then rows
- test_the_factor_is_made_of_the_two_mach_numbers: DF is exactly the ratio
- test_it_measures_what_the_surface_plot_draws: the same cut, the same numbers
- test_the_peak_sits_where_the_distribution_peaks: and on the plot's own axis
- test_a_section_above_the_gap_is_unmeasured: NaN rather than a wrong number
- test_values_survive_the_case_file: floats and NaN both
"""

import dataclasses

import numpy as np
import pytest
import yaml

from test_blade import build
from test_mesh import MESH, TIP
from test_cli import RUN_CASE
from turbigen import Config, Result, case, cli, metric, mixout, util
from turbigen.metric import DiffusionFactor
from turbigen.post import N_CHORD_PLOT, _isentropic_mach, _normalise_surface_distance

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

    assert set(values) == {"DF", "Mas_max", "Mas_TE", "zeta_max"}
    for name, value in values.items():
        assert np.shape(value) == (len(spf), result.grid.n_row), name
        assert np.all(np.isfinite(value)), name


#
# WHAT THE NUMBERS OBEY
#


def test_the_factor_is_made_of_the_two_mach_numbers(solved):
    """The peak over the exit, less one --- and a peak is never below an exit."""
    config, result = solved

    values = DiffusionFactor().evaluate(config, result)

    DF = np.asarray(values["DF"])
    np.testing.assert_allclose(
        DF, np.asarray(values["Mas_max"]) / np.asarray(values["Mas_TE"]) - 1.0
    )
    assert np.all(DF >= 0.0)


def test_it_measures_what_the_surface_plot_draws(solved):
    """Against the distribution itself, cut here rather than by the metric.

    The surface Mach number is the number someone reads off `SurfacePlot`, so
    the metric has to be that curve reduced --- not a second definition that
    happens to be close.
    """
    config, result = solved
    spf = 0.5
    i_row = 0

    mas, zeta = _distribution(result, i_row, spf)

    values = DiffusionFactor(spf=(spf,)).evaluate(config, result)

    assert values["Mas_max"][0][i_row] == pytest.approx(float(mas.max()))
    assert values["Mas_TE"][0][i_row] == pytest.approx(0.5 * float(mas[0] + mas[-1]))
    assert values["zeta_max"][0][i_row] == pytest.approx(
        abs(float(zeta[np.argmax(mas)]))
    )


def test_the_peak_sits_where_the_distribution_peaks(solved):
    """Between the stagnation point and the trailing edge, folded positive.

    The axis is the one `SurfacePlot` draws: zero where the flow stagnates and
    one at the trailing edge, whichever surface the point is on.
    """
    config, result = solved

    zeta_max = np.asarray(DiffusionFactor().evaluate(config, result)["zeta_max"])

    assert np.all(zeta_max >= 0.0)
    assert np.all(zeta_max <= 1.0)


def _distribution(result, i_row, spf):
    """Return ``(mas, zeta)`` cut here rather than by the metric."""
    import ember.cut  # noqa: PLC0415 - only the test's own arithmetic needs it

    surface = util.cut_blade_surfs(result.grid, 0)[i_row][0][:, :, None]
    m = np.linspace(2 * i_row + 1, 2 * i_row + 2, metric.N_SPAN_CUT)
    xr = result.machine.annulus.evaluate_xr(m, spf)
    cut = ember.cut.structured_meridional(surface, xr.T)[0]

    mas = _isentropic_mach(cut, result.machine.mean_line[:, i_row].s[0])[:, 0]
    blade = result.machine.rows[i_row].blade
    xrt_nose = blade.evaluate_section(spf, nchord=N_CHORD_PLOT)[0][:, 0]
    return mas, _normalise_surface_distance(cut, mas, xrt_nose)


def test_a_section_above_the_gap_is_unmeasured(gapped):
    """A blade with no surface at a span has no diffusion there, not zero."""
    config, result = gapped

    values = DiffusionFactor(spf=(0.5, 0.999)).evaluate(config, result)

    DF = np.asarray(values["DF"])
    assert np.isfinite(DF[0, TIP_ROW]), "mid-span is below the gap and is blade"
    assert np.isnan(DF[1, TIP_ROW]), "the tip section is trimmed off as flow"


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
    assert set(read_back.metrics) == {"DF", "Mas_max", "Mas_TE", "zeta_max"}
