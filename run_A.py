"""Sweep thickness."""

from turbigen import submit, turbostream

params = submit.read_params("turbigen/default_params.json")

base_dir = "A"

Aref = params["mesh"]["A"] + 0.0

params["run"]["guess_file"] = 'A/0011/output_1_avg.hdf5'

for mult in (1.2, ):
    params["mesh"]["A"] = mult * Aref
    submit.run(turbostream.write_grid_from_dict, params, base_dir)
