"""Sweep thickness."""

import numpy as np
from turbigen import submit, turbostream


base_dir = "A"
P = submit.ParameterSet.from_json("turbigen/default_params.json")

obj, constr, x0 = submit.make_objective_and_constraint(turbostream.write_grid_from_params, P, base_dir)

x = np.vstack([x0 * mult for mult in (0.1, 0.5, 1.0, 2.0)])

print(constr(x))

# eta = obj(x)
# print(eta)
# params_all = [deepcopy(params_default) for _ in range(4)]
# mult = (0.6,0.8,1.2,1.5)
# for i in range(4):
#     params_all[i]["mesh"]["A"] = Aref * mult[i]


# meta = submit.run_parallel(turbostream.write_grid_from_dict, params_all, base_dir)

# print([m["eta"] for m in meta])
