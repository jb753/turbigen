"""Sweep thickness."""

import numpy as np
from turbigen import submit, turbostream


params_default = submit.read_params("turbigen/default_params.json")
base_dir = "A"
params_default["run"]["guess_file"] = "A/0011/output_1_avg.hdf5"


func, x0 = submit.make_objective(turbostream.write_grid_from_dict, params_default,
        base_dir)

x = np.vstack([x0*mult for mult in (0.1, 0.5, 1., 2.)])

eta = func(x)

print(eta)
# params_all = [deepcopy(params_default) for _ in range(4)]
# mult = (0.6,0.8,1.2,1.5)
# for i in range(4):
#     params_all[i]["mesh"]["A"] = Aref * mult[i]



# meta = submit.run_parallel(turbostream.write_grid_from_dict, params_all, base_dir)

# print([m["eta"] for m in meta])
