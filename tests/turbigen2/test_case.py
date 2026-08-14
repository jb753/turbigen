"""Tests for reading and writing a config together with its result.

Both live in one file so that comparing what a run achieved against what it was
asked for is a single load. One document, with the answer under one `result:`
key, so that `yaml.safe_load` and everything built on it still work.

The point to protect is that this does not put results on the config: `Config`
stays frozen and unaware, and it is `case.read` that returns two objects. A file
is not an object.

Test cases:
- test_a_config_without_a_result_reads: nothing is required to have been run
- test_round_trip_gives_back_both: config equal, actual to float32
- test_config_from_file_ignores_a_result: the config half alone still loads
- test_result_is_one_key_beside_the_config: the file layout
- test_reading_without_designing_skips_the_machine: for bulk scraping
- test_nominal_without_a_machine_says_so: rather than an AttributeError
"""

import numpy as np
import pytest

from test_blade import build
from turbigen2 import Config, MeanLine, Result, case


@pytest.fixture
def config():
    return build()


@pytest.fixture
def result(config):
    """A designed machine with a fabricated `actual`, a little off the design.

    Fabricated rather than solved: what is under test is the file, and a march
    would make these tests cost seconds to prove nothing extra.
    """
    machine = config.design()
    actual = machine.mean_line.copy()
    actual.flat.set_P_T(
        np.asarray(machine.mean_line.flat.P) * 0.99,
        np.asarray(machine.mean_line.flat.T) * 1.01,
    )
    return Result(machine=machine, actual=actual, converged=True)


def test_a_config_without_a_result_reads(config, tmp_path):
    path = tmp_path / "case.yaml"
    case.write(path, config)

    read_config, read_result = case.read(path)

    assert read_result is None
    assert read_config == config


def test_round_trip_gives_back_both(config, result, tmp_path):
    path = tmp_path / "case.yaml"
    case.write(path, config, result)

    read_config, read_result = case.read(path)

    assert read_config == config
    assert read_result.converged is True
    for name in ("P", "T", "Vx", "Vt", "r", "Am", "Omega"):
        np.testing.assert_allclose(
            np.asarray(getattr(read_result.actual.flat, name), dtype=float),
            np.asarray(getattr(result.actual.flat, name), dtype=float),
            rtol=1e-6,
            err_msg=f"{name} did not survive the file",
        )

    # The comparison the whole arrangement exists to make possible.
    assert read_result.nominal.eta_tt != read_result.actual.eta_tt


def test_config_from_file_ignores_a_result(config, result, tmp_path):
    """A file with an answer in it is still a valid config on its own."""
    path = tmp_path / "case.yaml"
    case.write(path, config, result)

    assert Config.from_file(path) == config


def test_result_is_one_key_beside_the_config(config, result, tmp_path):
    """One document, one extra key -- so safe_load and yq keep working."""
    import yaml  # noqa: PLC0415

    path = tmp_path / "case.yaml"
    case.write(path, config, result)

    raw = yaml.safe_load(path.read_text())

    assert case.RESULT_KEY in raw
    assert set(raw[case.RESULT_KEY]) == {"converged", "actual"}
    assert set(raw[case.RESULT_KEY]["actual"]) == set(MeanLine.STATE)
    # The config's own keys are untouched by its neighbour.
    assert raw["mean_line"] == config.to_dict()["mean_line"]


def test_the_convergence_history_stays_out_of_the_file(config, result, tmp_path):
    """It describes how the answer was reached, not what it is.

    One record every `n_step_log` steps of residuals and station quantities is
    far more than a result file should carry, and it is worthless without the
    run that produced it.
    """
    import dataclasses  # noqa: PLC0415

    path = tmp_path / "case.yaml"
    case.write(path, config, dataclasses.replace(result, history=object()))

    _, read_result = case.read(path)

    assert read_result.history is None
    assert "history" not in path.read_text()


def test_reading_without_designing_skips_the_machine(config, result, tmp_path):
    """Designing every file is a cost a script scraping many need not pay."""
    path = tmp_path / "case.yaml"
    case.write(path, config, result)

    _, read_result = case.read(path, design=False)

    assert read_result.machine is None
    assert read_result.actual is not None


def test_nominal_without_a_machine_says_so(config, result, tmp_path):
    path = tmp_path / "case.yaml"
    case.write(path, config, result)
    _, read_result = case.read(path, design=False)

    with pytest.raises(ValueError, match="no machine"):
        read_result.nominal
