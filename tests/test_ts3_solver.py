"""Tests for the TS3 solver config, built up during the ember.grid migration.

These cover the pieces that are testable without a Turbostream binary or the
HPC: the module imports, the config dataclass behaves, and the solver resolves
through the same dynamic-dispatch path that ``config.py`` uses for ``type: ts3``.
"""

import importlib
from pathlib import Path

import h5py
import numpy as np
import pytest

from ember.convergence_history import ConvergenceHistory

import ember.util
from ember.block import Block
from ember.fluid import PerfectFluid
from ember.grid import Grid
from ember.patch import InletPatch, OutletPatch

import turbigen.solvers.base
import turbigen.solvers.ts3 as ts3_mod
import turbigen.util as util
from turbigen.exceptions import ConvergenceError

_DATA = Path(__file__).parent / "data"


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


def _make_grid_with_inlet():
    """Solved single-block grid with an inlet and outlet patch attached."""
    grid = _make_solved_grid()
    block = grid[0]
    inlet = InletPatch(i=0, j=(0, -1), k=(0, -1), label="inlet")
    outlet = OutletPatch(i=-1, j=(0, -1), k=(0, -1), label="outlet")
    block.patches.append(inlet)
    block.patches.append(outlet)
    inlet.set_Po_To_Alpha_Beta(
        Po=np.full(inlet.shape, 1.2e5),
        To=np.full(inlet.shape, 320.0),
        Alpha=np.zeros(inlet.shape),
        Beta=np.zeros(inlet.shape),
    )
    outlet.set_P(1.1e5)
    return grid


def _write_outputs(grid, workdir):
    """Write the grid as the two TS3 output files _read_hdf5 expects."""
    grid.write_ts3(str(workdir / "output_avg.hdf5"))
    grid.write_ts3(str(workdir / "output.hdf5"))


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


def _av(f, name):
    """Read a scalar application variable from an open TS3 hdf5 file."""
    return float(np.squeeze(f[f"{name}_av"][...]))


def _bv(f, bid, name):
    """Read a scalar block variable from an open TS3 hdf5 file."""
    return float(np.squeeze(f[f"block{bid}"][f"{name}_bv"][...]))


def test_write_input_av_bv_roundtrip(tmp_path):
    """_write_input forwards typed and derived av/bv into input.hdf5."""
    cfg = ts3_mod.ts3(nstep=8000, nstep_avg=2000, cfl=0.35, fmgrid=0.15)
    cfg.workdir = tmp_path
    ts3_mod._write_input(_make_grid_with_inlet(), cfg)

    with h5py.File(tmp_path / "input.hdf5", "r") as f:
        assert _av(f, "cfl") == pytest.approx(0.35)
        assert _av(f, "nstep") == pytest.approx(8000)
        assert _av(f, "nstep_save_start") == pytest.approx(6000)
        assert _av(f, "restart") == pytest.approx(1)
        assert _bv(f, 0, "fmgrid") == pytest.approx(0.15)


def test_write_input_rfin_pv(tmp_path):
    """rfin is applied to inlet patches as a patch variable."""
    cfg = ts3_mod.ts3(rfin=0.25)
    cfg.workdir = tmp_path
    ts3_mod._write_input(_make_grid_with_inlet(), cfg)

    with h5py.File(tmp_path / "input.hdf5", "r") as f:
        # Inlet is patch0 (rotating patches excluded; none here).
        rfin = float(np.squeeze(f["block0"]["patch0"]["rfin_pv"][...]))
    assert rfin == pytest.approx(0.25)


def test_write_input_raw_av_override(tmp_path):
    """A raw av override with no typed field lands in the file."""
    cfg = ts3_mod.ts3(av={"sfin_sa": 0.1})
    cfg.workdir = tmp_path
    ts3_mod._write_input(_make_grid_with_inlet(), cfg)

    with h5py.File(tmp_path / "input.hdf5", "r") as f:
        assert _av(f, "sfin_sa") == pytest.approx(0.1)


def test_write_input_raw_bv_override(tmp_path):
    """A raw bv override with no typed field lands in the file."""
    cfg = ts3_mod.ts3(bv={0: {"free_turb": 0.02}})
    cfg.workdir = tmp_path
    ts3_mod._write_input(_make_grid_with_inlet(), cfg)

    with h5py.File(tmp_path / "input.hdf5", "r") as f:
        assert _bv(f, 0, "free_turb") == pytest.approx(0.02)


def test_write_input_av_overlap_errors(tmp_path):
    """A raw av override colliding with a non-default typed field errors."""
    cfg = ts3_mod.ts3(cfl=0.2, av={"cfl": 0.3})
    cfg.workdir = tmp_path
    with pytest.raises(ValueError, match="cfl"):
        ts3_mod._write_input(_make_grid_with_inlet(), cfg)


def test_write_input_bv_overlap_errors(tmp_path):
    """A raw bv override colliding with a non-default typed field errors."""
    cfg = ts3_mod.ts3(fmgrid=0.1, bv={0: {"fmgrid": 0.0}})
    cfg.workdir = tmp_path
    with pytest.raises(ValueError, match="fmgrid"):
        ts3_mod._write_input(_make_grid_with_inlet(), cfg)


def test_write_input_av_overlap_ok_when_default(tmp_path):
    """An av override matching a typed field at its default is allowed."""
    # cfl left at its default, so the av dict is the sole source.
    cfg = ts3_mod.ts3(av={"cfl": 0.3})
    cfg.workdir = tmp_path
    ts3_mod._write_input(_make_grid_with_inlet(), cfg)
    with h5py.File(tmp_path / "input.hdf5", "r") as f:
        assert _av(f, "cfl") == pytest.approx(0.3)


def test_run_writes_input(tmp_path, monkeypatch):
    """_run writes input.hdf5 (no longer raising NotImplementedError)."""
    monkeypatch.setattr(ts3_mod, "_execute", lambda cfg: None)
    monkeypatch.setattr(ts3_mod, "_read_hdf5", lambda grid, cfg: None)
    cfg = ts3_mod.ts3()
    cfg.workdir = tmp_path
    ts3_mod._run(_make_grid_with_inlet(), cfg)
    assert (tmp_path / "input.hdf5").exists()


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


def test_convergence_history_from_log():
    """run() builds an ember ConvergenceHistory from the TS3 log + grid.

    Exercises the exact post-run call run() makes — log parsing lives in
    ember.ts3; the grid supplies the reference scales the log lacks.
    """
    conv = ConvergenceHistory.from_ts3(_DATA / "log_duct.txt", _make_grid_with_inlet())
    assert isinstance(conv, ConvergenceHistory)
    # All 399 step blocks parsed, with the grid-derived reference scales finite.
    assert conv.i_log + 1 == 399
    assert np.isfinite(conv._get_metadata_by_key("V_ref"))
    assert np.isfinite(conv._get_metadata_by_key("T_ref"))
