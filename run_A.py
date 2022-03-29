"""Sweep thickness."""

import numpy as np
from turbigen import submit, turbostream, tabu
import pickle


base_dir = "A"
P = submit.ParameterSet.from_json("turbigen/default_params.json")

obj, constr, x0 = submit.make_objective_and_constraint(
    turbostream.write_grid_from_params, P, base_dir
)

dx = 0.05 * np.ones_like(x0)
tol = dx/8.
ts = tabu.TabuSearch(obj, constr, x0.shape[1], 1, tol)

ts.max_fevals = 10
ts.max_parallel = 4
ts.search(x0, dx)

with open('test.pickle','w') as f:
    pickle.dump(ts, f)
