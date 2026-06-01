"""Tests for the TS3 solver config, built up during the ember.grid migration.

These cover the pieces that are testable without a Turbostream binary or the
HPC: the module imports, the config dataclass behaves, and the solver resolves
through the same dynamic-dispatch path that ``config.py`` uses for ``type: ts3``.
"""

import importlib

import h5py
import numpy as np
import pytest

import ember.util
from ember.block import Block
from ember.fluid import PerfectFluid
from ember.grid import Grid

import turbigen.solvers.base
import turbigen.solvers.ts3 as ts3_mod
import turbigen.util as util
from turbigen.exceptions import ConvergenceError


def _make_solved_grid(T_dtm=300.0):
    """Single-block ember grid on a general datum with a known flow field."""
    shape = (3, 4, 5)
    block = Block(shape=shape)
    xrt = ember.util.linmesh3([0.0, 1.0], [0.5, 1.5], [0.0, 0.2], shape)
    block.set_x(xrt[..., 0]).set_r(xrt[..., 1]).set_t(xrt[..., 2])
    block.set_fluid(PerfectFluid(cp=1005.0, gamma=1.4, mu=1.8e-5, Pr=0.7, T_dtm=T_dtm))
    block.set_rpm(0.0).set_Nb(1)
    block.set_P_T(np.full(shape, 1.2e5), np.full(shape, 320.0))
    block.set_Vxrt(
        np.stack(
            [120.0 * np.ones(shape), 5.0 * np.ones(shape), -20.0 * np.ones(shape)],
            axis=-1,
        )
    )
    block.set_mu_turb(np.full(shape, 1e-3))
    block.set_wdist(np.full(shape, 0.01, dtype=np.float32))
    return Grid([block])


def _write_outputs(grid, workdir):
    """Write the grid as the two TS3 output files _read_hdf5 expects."""
    grid.write_ts3(str(workdir / "output_avg.hdf5"), zero_datum=True)
    grid.write_ts3(str(workdir / "output.hdf5"), zero_datum=True)


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
    with pytest.raises(NotImplementedError):
        ts3_mod._run(grid=None, ts3_config=ts3_mod.ts3())


def test_read_hdf5_roundtrip(tmp_path):
    """_read_hdf5 loads the TS3 output back onto an existing grid via ember."""
    # Reference solution and the dimensional state we expect to recover.
    ref = _make_solved_grid()
    P, T = ref[0].P.copy(), ref[0].T.copy()
    Vx, mu = ref[0].Vx.copy(), ref[0].mu_turb.copy()

    # Writing zeroes the datum in place, so use a separate grid for the files.
    _write_outputs(_make_solved_grid(), tmp_path)

    # A blank target grid on the original datum receives the solution.
    target = _make_solved_grid()
    target[0].set_P_T(np.full(target[0].shape, 1e5), np.full(target[0].shape, 300.0))

    cfg = ts3_mod.ts3()
    cfg.workdir = tmp_path
    ts3_mod._read_hdf5(target, cfg)

    np.testing.assert_allclose(target[0].P, P, rtol=1e-4)
    np.testing.assert_allclose(target[0].T, T, rtol=1e-4)
    np.testing.assert_allclose(target[0].Vx, Vx, rtol=1e-4, atol=1e-3)
    np.testing.assert_allclose(target[0].mu_turb, mu, rtol=1e-5)


def test_read_hdf5_diverged_raises_convergence_error(tmp_path):
    """A diverged (negative-density) output surfaces as ConvergenceError."""
    _write_outputs(_make_solved_grid(), tmp_path)
    with h5py.File(tmp_path / "output_avg.hdf5", "r+") as f:
        ro = f["block0"]["ro_bp"]
        ro[...] = -np.abs(ro[...]) - 1.0

    cfg = ts3_mod.ts3()
    cfg.workdir = tmp_path
    with pytest.raises(ConvergenceError):
        ts3_mod._read_hdf5(_make_solved_grid(), cfg)


def test_read_hdf5_missing_output_raises(tmp_path):
    """No output file present raises a clear error."""
    cfg = ts3_mod.ts3()
    cfg.workdir = tmp_path
    with pytest.raises(Exception, match="No Turbostream output file"):
        ts3_mod._read_hdf5(_make_solved_grid(), cfg)
