"""Tests for metrics: quantities measured from a solved field and kept.

A metric returns data, not figures, and nothing acts on what it measures, so
these tests are pure functions of a config and a result --- except the one
that needs a real grid, which marches the fast cascade briefly like the
post-processing tests do.

Test cases:
- test_metric_is_a_node: a metric is an ordinary config node
- test_metric_is_quiet_without_a_grid: no field, nothing measured
- test_measure_merges_and_yamlifies: scalars and profiles come back as floats
- test_a_raising_metric_is_logged_not_raised: an observation cannot sink a run
- test_a_key_collision_keeps_the_last: and says so
- test_metrics_round_trip_through_the_case_file: including a profile value
"""

import dataclasses
import logging

import numpy as np
import pytest
import yaml
from test_cli import RUN_CASE

from turbigen import Config, Metric, Result, case, cli, metric


class GridStats(Metric):
    """A stand-in that reads the grid, returning a scalar and a profile."""

    type = "_test_grid_stats"

    def evaluate(self, config, result):
        if result.grid is None:
            return {}
        blocks = list(result.grid)
        return {
            "n_block": len(blocks),
            "shapes": [list(block.shape) for block in blocks],
        }


class Boom(Metric):
    """A metric that cannot be measured, to prove one is not fatal."""

    type = "_test_boom"

    def evaluate(self, config, result):
        raise RuntimeError("no")


@pytest.fixture(scope="module")
def solved():
    """The fast single-row cascade, marched briefly for a real field."""
    config = Config.from_dict(yaml.safe_load(RUN_CASE))
    _, machine, grid = cli.prepare(config)
    history = config.solver.solve(grid)
    result = Result(machine=machine, grid=grid, converged=True, history=history)
    return config, result


def test_metric_is_a_node():
    base = Config.from_dict(yaml.safe_load(RUN_CASE)).to_dict()
    config = Config.from_dict({**base, "metrics": [{"type": GridStats.type}]})
    assert isinstance(config.metrics, tuple)
    assert isinstance(config.metrics[0], GridStats)
    assert Config.from_dict(config.to_dict()) == config


def test_metric_is_quiet_without_a_grid():
    config = Config.from_dict(yaml.safe_load(RUN_CASE))
    assert GridStats().evaluate(config, Result()) == {}

    config = dataclasses.replace(config, metrics=(GridStats(),))
    assert metric.measure(config, Result()) == {}


def test_measure_merges_and_yamlifies(solved):
    config, result = solved
    config = dataclasses.replace(config, metrics=(GridStats(),))

    values = metric.measure(config, result)

    assert isinstance(values["n_block"], float)
    assert values["n_block"] >= 1.0
    assert all(
        isinstance(axis, float) for shape in values["shapes"] for axis in shape
    )


def test_a_raising_metric_is_logged_not_raised(solved, caplog):
    config, result = solved
    config = dataclasses.replace(config, metrics=(Boom(), GridStats()))

    with caplog.at_level(logging.WARNING, logger="turbigen"):
        values = metric.measure(config, result)

    assert "n_block" in values
    assert "could not be measured" in caplog.text


def test_a_key_collision_keeps_the_last(solved, caplog):
    config, result = solved
    config = dataclasses.replace(config, metrics=(GridStats(), GridStats()))

    with caplog.at_level(logging.WARNING, logger="turbigen"):
        values = metric.measure(config, result)

    assert set(values) == {"n_block", "shapes"}
    assert "both write" in caplog.text


def test_metrics_round_trip_through_the_case_file(tmp_path):
    config = Config.from_dict(yaml.safe_load(RUN_CASE))
    result = Result(
        machine=config.design(),
        converged=True,
        metrics={"a": 1, "profile": np.array([[1, 2], [3, 4]])},
    )

    path = tmp_path / "case.yaml"
    case.write(path, config, result)
    _, read_back = case.read(path)

    assert read_back.metrics == {"a": 1.0, "profile": [[1.0, 2.0], [3.0, 4.0]]}
    raw = yaml.safe_load(path.read_text())
    assert set(raw["result"]) == {"converged", "metrics"}
