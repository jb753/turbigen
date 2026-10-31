"""Tests for the Turbostream 3 solver, through the ember-cfd-ts plugin.

Turbostream itself needs a GPU node, so these check what turbigen adds: that a
config names it by type and round-trips, that the soft start can set its step
count as it does ember's, and that it is handed the run's directory. A run that
leaves no history is the plugin's to refuse.
"""

import dataclasses

import pytest

from turbigen import solver as solver_mod
from turbigen.solver import Solver

pytestmark = pytest.mark.skipif(
    solver_mod.ember_ts is None, reason="ember-cfd-ts is not installed"
)

CONFIG = {"type": "ts3", "n_step": 2000, "n_step_avg": 500, "n_step_soft": 100}


def test_a_config_names_it_by_type():
    from turbigen.solver import TS3

    ts3 = Solver.from_dict(dict(CONFIG))
    assert isinstance(ts3, TS3)
    assert (ts3.n_step, ts3.n_step_avg, ts3.n_step_soft) == (2000, 500, 100)


def test_it_round_trips_through_its_config():
    ts3 = Solver.from_dict(dict(CONFIG))
    assert Solver.from_dict(ts3.to_dict()) == ts3


def test_the_soft_start_sets_its_step_count():
    ts3 = Solver.from_dict(dict(CONFIG))
    soft = dataclasses.replace(ts3.soft(), n_step=ts3.n_step_soft)
    assert soft.n_step == 100 and soft.n_step_avg == 1 and soft.cfl < ts3.cfl


def test_it_runs_in_the_directory_it_is_given(monkeypatch, tmp_path):
    ts3 = Solver.from_dict(dict(CONFIG))
    seen = {}

    def run(self, grid, workdir=None):
        seen["workdir"] = workdir
        return "history"

    monkeypatch.setattr(type(ts3), "run", run)
    assert ts3.solve(None, tmp_path) == "history"
    assert seen["workdir"] == tmp_path
