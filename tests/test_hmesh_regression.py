import pickle
import pathlib
import tempfile
import numpy as np
import pytest

import turbigen_ref.config
import turbigen_ref.yaml_utils

DATA = pathlib.Path(__file__).parent / "data"


def _build_grid(inputs):
    """Reconstruct mac from conf_dict and run make_grid."""
    with tempfile.TemporaryDirectory() as tmp:
        conf_dict = dict(inputs["conf_dict"])
        conf_dict["work_dir"] = tmp
        conf = turbigen_ref.config.TurbigenConfig(**conf_dict)
        conf.get_mean_line_nominal()
        conf.get_geometry()
        conf.adjust_ref()
        conf.apply_recamber()
        mac = conf.get_machine()
        return inputs["mesh_cfg"].make_grid(
            pathlib.Path(tmp),
            mac,
            inputs["dhub"],
            inputs["dcas"],
            inputs["dsurf"],
            inputs["Omega"],
        )


@pytest.fixture(scope="module")
def inputs():
    with open(DATA / "axial_turbine_mesh_inputs.pkl", "rb") as f:
        return pickle.load(f)


@pytest.fixture(scope="module")
def grid(inputs):
    return _build_grid(inputs)


@pytest.fixture(scope="module")
def snapshot():
    return np.load(DATA / "axial_turbine_mesh_xrt.npz")


# --- Snapshot ---

def test_xrt_matches_snapshot(grid, snapshot):
    for i in range(len(grid)):
        np.testing.assert_allclose(
            grid[i].xrt,
            snapshot[f"block{i}_xrt"],
            atol=1e-10,
            rtol=1e-10,
            err_msg=f"block {i} xrt mismatch",
        )


# --- Coordinate invariants ---

def test_all_blocks_xrt_finite(grid):
    for i in range(len(grid)):
        assert np.isfinite(grid[i].xrt).all(), f"block {i} has non-finite xrt"


def test_mg_divisibility(grid):
    for i in range(len(grid)):
        ni, nj, nk = grid[i].shape
        assert (ni - 1) % 8 == 0, f"block {i}: ni-1={ni-1} not divisible by 8"
        assert (nj - 1) % 8 == 0, f"block {i}: nj-1={nj-1} not divisible by 8"
        assert (nk - 1) % 8 == 0, f"block {i}: nk-1={nk-1} not divisible by 8"


def test_x_monotonic_midspan(grid):
    for i in range(len(grid)):
        b = grid[i]
        jmid = b.shape[1] // 2
        kmid = b.shape[2] // 2
        x_line = b.xrt[:, jmid, kmid, 0]
        assert (np.diff(x_line) > 0).all(), f"block {i} x not monotonic at midspan"


def test_theta_ordering(grid):
    for i in range(len(grid)):
        theta_lo = grid[i].xrt[..., 0, 2]
        theta_hi = grid[i].xrt[..., -1, 2]
        assert (theta_hi > theta_lo).all(), f"block {i} theta ordering violated"


def test_mixing_plane_xr_match(grid):
    for i in range(len(grid) - 1):
        xr0 = grid[i].xrt[-1, :, 0, :2]
        xr1 = grid[i + 1].xrt[0, :, 0, :2]
        np.testing.assert_allclose(
            xr0, xr1, atol=1e-10,
            err_msg=f"mixing plane mismatch between blocks {i} and {i+1}",
        )


def test_periodic_connectivity(grid):
    grid.connectivity.periodic.pair()
