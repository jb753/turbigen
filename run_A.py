"""Sweep thickness."""

import numpy as np
from turbigen import submit, turbostream, tabu


base_dir = "recam"
P = submit.ParameterSet.from_json("turbigen/default_params.json")

obj, constr, x0 = submit.make_objective_and_constraint(
    turbostream.write_grid_from_params, P, base_dir
)

constr(np.atleast_2d([-2., 0., 0., 0.]))

quit()

dx = 2. * np.ones_like(x0)
tol = dx / 8.0
ts = tabu.TabuSearch(obj, constr, x0.shape[1], 1, tol)

ts.max_fevals = 1000
ts.max_parallel = 4
x_opt, y_opt = ts.search(x0, dx)

print(x_opt, y_opt)
