"""Sweep thickness."""

from turbigen import submit, turbostream

params = submit.read_params("turbigen/default_params.json")

base_dir = 'A'

Aref = params["mesh"]["A"] + 0.0

for mult in (0.1, 0.5, 1., 1.5):
    params["mesh"]["A"] = mult * Aref
    submit.run(turbostream.write_grid_from_dict, params, base_dir)
