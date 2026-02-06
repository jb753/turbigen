"""Tests for read_probe_dat function in ts3 solver."""

import numpy as np
import pytest
import h5py
import os
from pathlib import Path
import time
from unittest.mock import patch
from turbigen.solvers.ts3 import read_probe_dat
from turbigen import yaml_utils


def create_minimal_hdf5(
    filepath,
    cp=1005.0,
    ga=1.4,
    mu=1.8e-5,
    freq=100.0,
    nstep_cycle=72,
    nstep_save_probe=1,
):
    """Create minimal HDF5 file with required scalar fields."""
    with h5py.File(filepath, "w") as f:
        f.create_dataset("cp_av", data=np.array([cp], dtype=np.float32))
        f.create_dataset("ga_av", data=np.array([ga], dtype=np.float32))
        f.create_dataset("viscosity_av", data=np.array([mu], dtype=np.float32))
        f.create_dataset("frequency_av", data=np.array([freq], dtype=np.float32))
        f.create_dataset("nstep_cycle_av", data=np.array([nstep_cycle], dtype=np.int32))
        f.create_dataset(
            "nstep_save_probe_av", data=np.array([nstep_save_probe], dtype=np.int32)
        )


def create_probe_dat(filepath, shape, nsteps=10):
    """Create synthetic probe .dat file with realistic data.

    Parameters
    ----------
    filepath : Path or str
        Output file path
    shape : tuple
        Spatial shape of probe (e.g., (5, 2))
    nsteps : int
        Number of time steps

    Returns
    -------
    expected_data : dict
        Dictionary with expected values for verification
    """
    nspatial = np.prod(shape)

    # Create recognizable synthetic data
    # x coordinates: 0, 1, 2, ..., nspatial-1
    x = np.arange(nspatial, dtype=np.float32)
    # r coordinates: constant at 0.5
    r = np.full(nspatial, 0.5, dtype=np.float32)
    # rt: r * theta, theta from 0 to 2*pi
    rt = r * np.linspace(0, 2 * np.pi, nspatial, dtype=np.float32)

    # Density: constant at 1.2
    ro = np.full(nspatial, 1.2, dtype=np.float32)
    # Velocities: simple patterns
    rovx = ro * 10.0  # Vx = 10 m/s
    rovr = ro * 0.0  # Vr = 0 m/s
    rorvt = ro * r * 5.0  # Vt = 5 m/s

    # Total energy: internal + kinetic
    # For perfect gas: u = P/(rho*(gamma-1)) = cv*T
    # Use P = 1e5, gamma = 1.4 => u = 1e5/(1.2*0.4) = 208333.33
    u = 208333.33
    Vx = rovx / ro
    Vr = rovr / ro
    Vt = rorvt / (ro * r)
    ke = 0.5 * (Vx**2 + Vr**2 + Vt**2)
    roe = ro * (u + ke)

    # Stack conserved variables
    conserved = np.stack([x, r, rt, ro, rovx, rovr, rorvt, roe])

    # Repeat for nsteps (time dimension)
    # Add small time variation to density
    data_all_steps = []
    for istep in range(nsteps):
        conserved_t = conserved.copy()
        # Add 1% sinusoidal variation to density and energy
        factor = 1.0 + 0.01 * np.sin(2 * np.pi * istep / nsteps)
        conserved_t[3] *= factor  # ro
        conserved_t[4] *= factor  # rovx
        conserved_t[5] *= factor  # rovr
        conserved_t[6] *= factor  # rorvt
        conserved_t[7] *= factor  # roe
        data_all_steps.append(conserved_t.T)  # Transpose to (nspatial, 8)

    # Stack all time steps: (nsteps, nspatial, 8)
    data_all = np.concatenate(data_all_steps, axis=0)

    # Write to file with header
    with open(filepath, "w") as f:
        f.write("x r rt ro rovx rovr rorvt roe\n")
        np.savetxt(f, data_all, fmt="%.8e")

    return {
        "x": x,
        "r": r,
        "ro_initial": ro,
        "Vx": Vx[0],
        "nsteps": nsteps,
        "nspatial": nspatial,
    }


def test_basic_dat_loading(tmp_path):
    """Test 1: Basic .dat file loading with minimal setup."""
    # Setup
    shape = [5, 2]
    nsteps = 10
    bid, pid = 75, 16

    # Create probe metadata
    probe_meta = {
        bid: {pid: {"Nb": 10, "Omega": 0.0, "label": "test_probe", "shape": shape}}
    }
    yaml_utils.write_yaml(probe_meta, tmp_path / "probe_meta.yaml")

    # Create HDF5 with gas properties
    freq = 100.0
    nstep_cycle = 72
    nstep_save_probe = 1
    create_minimal_hdf5(
        tmp_path / "input.hdf5",
        freq=freq,
        nstep_cycle=nstep_cycle,
        nstep_save_probe=nstep_save_probe,
    )

    # Create probe .dat file
    dat_file = tmp_path / f"output_probe_{bid}_{pid}.dat"
    expected = create_probe_dat(dat_file, shape, nsteps)

    # Test
    F, fs = read_probe_dat(str(dat_file))

    # Verify
    assert F is not None, "FlowField should not be None"
    assert fs == freq * nstep_cycle / nstep_save_probe, "Sampling frequency incorrect"

    # Check shape: (5, 2, 10)
    assert F.shape == tuple(shape + [nsteps]), (
        f"Expected shape {tuple(shape + [nsteps])}, got {F.shape}"
    )

    # Check that .hdf5 cache was created (default cache_format)
    hdf5_file = tmp_path / f"output_probe_{bid}_{pid}.hdf5"
    assert hdf5_file.exists(), ".hdf5 file should be created"

    # Verify some data values
    # x-coordinate should be recognizable pattern
    assert F.x[0, 0, 0] == 0.0, "First x-coordinate should be 0"
    assert F.x[1, 0, 0] == 1.0, "Second x-coordinate should be 1"

    # Density should be around 1.2
    assert np.abs(F.rho[0, 0, 0] - 1.2) < 0.1, "Density should be close to 1.2"

    # Velocity should be computed correctly
    assert np.abs(F.Vx[0, 0, 0] - expected["Vx"]) < 0.1, (
        "Vx should be computed correctly"
    )


def test_npz_caching(tmp_path):
    """Test 2: Caching behavior - .dat should be deleted once cached, data remains accessible."""
    # Setup
    shape = [3, 3]
    nsteps = 10
    bid, pid = 76, 17

    # Create metadata and HDF5
    probe_meta = {
        bid: {pid: {"Nb": 10, "Omega": 0.0, "label": "cache_test", "shape": shape}}
    }
    yaml_utils.write_yaml(probe_meta, tmp_path / "probe_meta.yaml")
    create_minimal_hdf5(tmp_path / "input.hdf5")

    # Create probe .dat file
    dat_file = tmp_path / f"output_probe_{bid}_{pid}.dat"
    create_probe_dat(dat_file, shape, nsteps)

    # Record initial state
    initial_files = set(tmp_path.glob(f"output_probe_{bid}_{pid}*"))

    # Mock the file to be 48+ hours old so it gets deleted on first call
    old_time = time.time() - (49 * 3600)  # 49 hours ago

    with patch("os.path.getmtime", return_value=old_time):
        # First call: should load from .dat, create cache, and delete .dat (because it's "old")
        F1, fs1 = read_probe_dat(str(dat_file))

    # Record files after first call
    files_after_first = set(tmp_path.glob(f"output_probe_{bid}_{pid}*"))
    new_files = files_after_first - initial_files

    # Should have created at least one new file (the cache)
    assert len(new_files) > 0, "Cache file should be created on first call"

    # .dat should be deleted after first call (because we mocked it as old)
    assert not dat_file.exists(), (
        ".dat should be deleted on first call when file is >48 hours old"
    )

    # Second call: should load from cache even though .dat is gone
    F2, fs2 = read_probe_dat(str(dat_file))

    # Third call: should still work, loading from cache
    F3, fs3 = read_probe_dat(str(dat_file))

    # Verify all calls return identical data
    assert F1.shape == F2.shape == F3.shape, (
        "Shape should be identical across all calls"
    )
    assert fs1 == fs2 == fs3, "Sampling frequency should be identical across all calls"

    # Check that data values match across all calls
    np.testing.assert_allclose(
        F1.x, F2.x, rtol=1e-6, err_msg="x-coordinates should match (call 1 vs 2)"
    )
    np.testing.assert_allclose(
        F1.rho, F2.rho, rtol=1e-6, err_msg="Density should match (call 1 vs 2)"
    )
    np.testing.assert_allclose(
        F1.Vx, F2.Vx, rtol=1e-6, err_msg="Vx should match (call 1 vs 2)"
    )

    np.testing.assert_allclose(
        F1.x, F3.x, rtol=1e-6, err_msg="x-coordinates should match (call 1 vs 3)"
    )
    np.testing.assert_allclose(
        F1.rho, F3.rho, rtol=1e-6, err_msg="Density should match (call 1 vs 3)"
    )
    np.testing.assert_allclose(
        F1.Vx, F3.Vx, rtol=1e-6, err_msg="Vx should match (call 1 vs 3)"
    )


def test_fortran_order_reshape(tmp_path):
    """Test 3: Correct reshaping with Fortran order for non-trivial shapes."""
    # Setup with non-trivial shape
    shape = [2, 3, 5]  # 30 spatial points
    nsteps = 10
    bid, pid = 77, 18
    nspatial = np.prod(shape)

    # Create metadata and HDF5
    probe_meta = {
        bid: {pid: {"Nb": 10, "Omega": 0.0, "label": "reshape_test", "shape": shape}}
    }
    yaml_utils.write_yaml(probe_meta, tmp_path / "probe_meta.yaml")
    create_minimal_hdf5(tmp_path / "input.hdf5")

    # Create probe .dat file
    dat_file = tmp_path / f"output_probe_{bid}_{pid}.dat"
    expected = create_probe_dat(dat_file, shape, nsteps)

    # Test
    F, fs = read_probe_dat(str(dat_file))

    # Verify shape
    expected_shape = tuple(shape + [nsteps])
    assert F.shape == expected_shape, f"Expected shape {expected_shape}, got {F.shape}"

    # Verify Fortran-order reshaping by checking x-coordinates
    # The x-coordinates were created as 0, 1, 2, ..., 29
    # With Fortran order reshape to (2, 3, 5), the indexing should be:
    # x[i, j, k] = i + 2*j + 2*3*k

    # Check specific indices
    assert F.x[0, 0, 0, 0] == 0, "x[0,0,0] should be 0"
    assert F.x[1, 0, 0, 0] == 1, "x[1,0,0] should be 1"  # i+1
    assert F.x[0, 1, 0, 0] == 2, "x[0,1,0] should be 2"  # 2*j
    assert F.x[0, 0, 1, 0] == 6, "x[0,0,1] should be 6"  # 2*3*k
    assert F.x[1, 1, 1, 0] == 9, "x[1,1,1] should be 9"  # 1 + 2*1 + 6*1

    # Verify density is positive everywhere
    assert np.all(F.rho > 0), "Density should be positive everywhere"

    # Verify velocities are computed correctly
    # We set Vx = 10 m/s initially
    assert np.abs(F.Vx[0, 0, 0, 0] - 10.0) < 0.1, "Vx should be ~10 m/s"

    # Verify internal energy is positive (physical requirement)
    # Internal energy u = roe/rho - 0.5*V^2
    u = F.rhoe / F.rho - 0.5 * (F.Vx**2 + F.Vr**2 + F.Vt**2)
    assert np.all(u > 0), "Internal energy should be positive everywhere"


def test_time_dimension_validation(tmp_path):
    """Test 4: Time dimension validation against log.txt parameters."""
    # Setup
    shape = [3, 2]
    bid, pid = 78, 19

    # Create metadata and HDF5
    probe_meta = {
        bid: {
            pid: {
                "Nb": 10,
                "Omega": 0.0,
                "label": "time_validation_test",
                "shape": shape,
            }
        }
    }
    yaml_utils.write_yaml(probe_meta, tmp_path / "probe_meta.yaml")
    create_minimal_hdf5(tmp_path / "input.hdf5")

    # Create log.txt with specific parameters
    ncycle = 8
    nstep_cycle = 72
    nstep_save_probe = 2
    nstep_save_start_probe = 0

    log_content = f"""
Application variables
    ncycle: {ncycle}
    nstep_cycle: {nstep_cycle}
    nstep_save_probe: {nstep_save_probe}
    nstep_save_start_probe: {nstep_save_start_probe}
"""

    with open(tmp_path / "log.txt", "w") as f:
        f.write(log_content)

    # Expected number of time steps: (ncycle * nstep_cycle - nstep_save_start_probe) // nstep_save_probe
    expected_nsteps = (
        ncycle * nstep_cycle - nstep_save_start_probe
    ) // nstep_save_probe  # = 288

    # Test 1: Correct number of time steps - should pass
    dat_file_correct = tmp_path / f"output_probe_{bid}_{pid}.dat"
    create_probe_dat(dat_file_correct, shape, nsteps=expected_nsteps)

    F, fs = read_probe_dat(str(dat_file_correct))
    assert F.shape[-1] == expected_nsteps, "Should accept correct number of time steps"

    # Test 2: Wrong number of time steps - should raise ValueError
    bid2, pid2 = 79, 20
    probe_meta[bid2] = {
        pid2: {"Nb": 10, "Omega": 0.0, "label": "wrong_time_test", "shape": shape}
    }
    yaml_utils.write_yaml(probe_meta, tmp_path / "probe_meta.yaml")

    wrong_nsteps = 100  # Different from expected
    dat_file_wrong = tmp_path / f"output_probe_{bid2}_{pid2}.dat"
    create_probe_dat(dat_file_wrong, shape, nsteps=wrong_nsteps)

    # Should raise ValueError with informative message
    with pytest.raises(ValueError, match="Time dimension mismatch"):
        read_probe_dat(str(dat_file_wrong))


def test_skip_age_check(tmp_path):
    """Test 5: skip_age_check flag bypasses 48-hour age check."""
    # Setup
    shape = [2, 2]
    nsteps = 10
    bid, pid = 80, 21

    # Create metadata and HDF5
    probe_meta = {
        bid: {pid: {"Nb": 10, "Omega": 0.0, "label": "skip_age_test", "shape": shape}}
    }
    yaml_utils.write_yaml(probe_meta, tmp_path / "probe_meta.yaml")
    create_minimal_hdf5(tmp_path / "input.hdf5")

    # Create probe .dat file
    dat_file = tmp_path / f"output_probe_{bid}_{pid}.dat"
    hdf5_file = tmp_path / f"output_probe_{bid}_{pid}.hdf5"
    create_probe_dat(dat_file, shape, nsteps)

    # First call with skip_age_check=True (file is recent, normally wouldn't delete)
    # Don't mock time - file will be fresh
    F1, fs1 = read_probe_dat(str(dat_file), skip_age_check=True)

    # Verify .dat was deleted despite being fresh (< 48 hours old)
    assert not dat_file.exists(), ".dat should be deleted when skip_age_check=True"

    # Verify .hdf5 cache exists (default cache_format)
    assert hdf5_file.exists(), ".hdf5 should exist"

    # Second call should still work (loading from cache)
    F2, fs2 = read_probe_dat(str(dat_file), skip_age_check=True)

    # Verify data is identical
    assert F1.shape == F2.shape, "Shape should be identical"
    assert fs1 == fs2, "Sampling frequency should be identical"
    np.testing.assert_allclose(F1.x, F2.x, rtol=1e-6, err_msg="Data should match")


def test_skip_age_check_default_false(tmp_path):
    """Test 6: Default behavior (skip_age_check=False) preserves recent files."""
    # Setup
    shape = [2, 2]
    nsteps = 10
    bid, pid = 81, 22

    # Create metadata and HDF5
    probe_meta = {
        bid: {
            pid: {"Nb": 10, "Omega": 0.0, "label": "default_age_test", "shape": shape}
        }
    }
    yaml_utils.write_yaml(probe_meta, tmp_path / "probe_meta.yaml")
    create_minimal_hdf5(tmp_path / "input.hdf5")

    # Create probe .dat file
    dat_file = tmp_path / f"output_probe_{bid}_{pid}.dat"
    hdf5_file = tmp_path / f"output_probe_{bid}_{pid}.hdf5"
    create_probe_dat(dat_file, shape, nsteps)

    # First call with default skip_age_check=False
    # File is fresh, should NOT be deleted
    F1, fs1 = read_probe_dat(str(dat_file))

    # Verify .dat was NOT deleted (file is fresh, < 48 hours)
    assert dat_file.exists(), (
        ".dat should be preserved when file is recent and skip_age_check=False"
    )

    # Verify .hdf5 cache exists (default cache_format)
    assert hdf5_file.exists(), ".hdf5 should exist"


def test_hdf5_cache_format(tmp_path):
    """Test 7: HDF5 cache format works correctly."""
    # Setup
    shape = [3, 3]
    nsteps = 10
    bid, pid = 82, 23

    # Create metadata and HDF5
    probe_meta = {
        bid: {pid: {"Nb": 10, "Omega": 0.0, "label": "hdf5_test", "shape": shape}}
    }
    yaml_utils.write_yaml(probe_meta, tmp_path / "probe_meta.yaml")
    create_minimal_hdf5(tmp_path / "input.hdf5")

    # Create probe .dat file
    dat_file = tmp_path / f"output_probe_{bid}_{pid}.dat"
    hdf5_file = tmp_path / f"output_probe_{bid}_{pid}.hdf5"
    npz_file = tmp_path / f"output_probe_{bid}_{pid}.npz"
    create_probe_dat(dat_file, shape, nsteps)

    # Mock file to be old so it gets deleted
    old_time = time.time() - (49 * 3600)

    with patch("os.path.getmtime", return_value=old_time):
        # First call with cache_format='hdf5'
        F1, fs1 = read_probe_dat(str(dat_file), cache_format="hdf5")

    # Verify .hdf5 was created and .dat was deleted
    assert hdf5_file.exists(), ".hdf5 cache file should be created"
    assert not dat_file.exists(), ".dat should be deleted when old"
    assert not npz_file.exists(), ".npz should not be created when using hdf5 format"

    # Second call should load from .h5 cache
    F2, fs2 = read_probe_dat(str(dat_file), cache_format="hdf5")

    # Verify data is identical
    assert F1.shape == F2.shape, "Shape should be identical"
    assert fs1 == fs2, "Sampling frequency should be identical"
    np.testing.assert_allclose(
        F1.x, F2.x, rtol=1e-6, err_msg="x-coordinates should match"
    )
    np.testing.assert_allclose(
        F1.rho, F2.rho, rtol=1e-6, err_msg="Density should match"
    )
    np.testing.assert_allclose(F1.Vx, F2.Vx, rtol=1e-6, err_msg="Vx should match")
