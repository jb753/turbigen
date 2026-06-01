"""Tests for the TS3 solver config, built up during the ember.grid migration.

These cover the pieces that are testable without a Turbostream binary or the
HPC: the module imports, the config dataclass behaves, and the solver resolves
through the same dynamic-dispatch path that ``config.py`` uses for ``type: ts3``.
"""

import importlib
import dataclasses

import turbigen.solvers.base
import turbigen.solvers.ts3 as ts3_mod
import turbigen.util as util


def test_module_imports():
    """The solver module imports without the deleted turbigen.grid dependency."""
    assert hasattr(ts3_mod, "ts3")


def test_config_instantiation_defaults():
    """The ts3 config dataclass instantiates with its documented defaults."""
    cfg = ts3_mod.ts3()
    assert cfg._name == "ts3"
    assert cfg.cfl == 0.4
    assert cfg.ilos == 2
    assert cfg.nstep == 10000
    assert cfg.Lref_xllim == "pitch"


def test_robust_returns_more_stable_config():
    """robust() lowers CFL and switches to the mixing-length model."""
    cfg = ts3_mod.ts3()
    robust = cfg.robust()
    assert isinstance(robust, ts3_mod.ts3)
    assert robust.ilos == 1
    assert robust.cfl == 0.3
    assert robust.soft_start is False
    # Original is unchanged (replace returns a new instance).
    assert cfg.cfl == 0.4


def test_restart_zeros_nchange():
    """restart() disables the start-up smoothing/damping ramp."""
    assert ts3_mod.ts3().restart().nchange == 0


def test_solver_resolves_via_dynamic_dispatch():
    """`type: ts3` resolves to the ts3 class through the config dispatch path."""
    importlib.import_module(".ts3", package="turbigen.solvers")
    cls = util.get_subclass_by_name(turbigen.solvers.base.BaseSolver, "ts3")
    assert cls is ts3_mod.ts3


def test_input_writing_not_yet_migrated():
    """Until the write side is ported to ember.ts3, _run raises clearly."""
    import pytest

    with pytest.raises(NotImplementedError):
        ts3_mod._run(grid=None, ts3_config=ts3_mod.ts3())
