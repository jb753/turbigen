"""Run once to capture make_grid inputs and known-good output.

Usage:
    uv run python scripts/capture_hmesh_inputs.py
"""
import pickle
import tempfile
import pathlib
import numpy as np
import turbigen.config
import turbigen.yaml_utils

YAML = pathlib.Path("examples/axial_turbine.yaml")
OUT_DIR = pathlib.Path("tests/data")
OUT_DIR.mkdir(exist_ok=True)


def build_conf(tmp):
    conf_dict = turbigen.yaml_utils.read_yaml(YAML)
    conf_dict["work_dir"] = tmp
    conf = turbigen.config.TurbigenConfig(**conf_dict)
    conf.get_mean_line_nominal()
    conf.adjust_ref()
    conf.get_geometry()
    conf.apply_recamber()
    return conf


with tempfile.TemporaryDirectory() as tmp:
    conf = build_conf(tmp)

    dsurf = conf.calculate_d_wall()
    dhub = dcas = float(np.mean(dsurf))
    Omega = conf.mean_line.nominal.Omega[::2].copy()
    mesh_cfg = conf.mesh

    # Store picklable scalar inputs plus the original YAML dict (no work_dir)
    conf_dict_clean = turbigen.yaml_utils.read_yaml(YAML)
    conf_dict_clean.pop("work_dir", None)

    inputs = dict(
        dhub=dhub,
        dcas=dcas,
        dsurf=dsurf,
        Omega=Omega,
        mesh_cfg=mesh_cfg,
        conf_dict=conf_dict_clean,
    )
    with open(OUT_DIR / "axial_turbine_mesh_inputs.pkl", "wb") as f:
        pickle.dump(inputs, f)
    print("Saved inputs pickle.")

    mac = conf.get_machine()
    grid = mesh_cfg.make_grid(pathlib.Path(tmp), mac, dhub, dcas, dsurf, Omega)
    arrays = {f"block{i}_xrt": grid[i].xrt for i in range(len(grid))}
    np.savez(OUT_DIR / "axial_turbine_mesh_xrt.npz", **arrays)
    print(f"Saved xrt for {len(grid)} blocks.")
