"""Sweep thickness."""

import numpy as np
from turbigen import submit, turbostream, tabu


base_dir = "recam"
mem_file = "mem_recam.json"
P = submit.ParameterSet.from_json("turbigen/default_params.json")

P.guess_file = 'recam/0670/output_avg.hdf5'

obj, constr, x0 = submit.make_objective_and_constraint(
    turbostream.write_grid_from_params, P, base_dir
)


dx = np.atleast_2d([[1.,1.,1.,1.,0.05, 0.05, 0.05, 2., 0.05, 0.05, 0.05,2.]])
tol = dx / 4.0
ts = tabu.TabuSearch(obj, constr, x0.shape[1], 5, tol, j_obj=(0,))

ts.max_fevals = 10
ts.max_parallel = 4

ts.load_memories(mem_file)
# x0 = np.atleast_2d(ts.mem_med.get(0)[0])

x_opt, y_opt = ts.search(x0 , dx)

ts.save_memories(mem_file)
