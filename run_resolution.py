"""Run a resolution study on datum design."""

from turbigen import submit, turbostream, geometry
import numpy as np
import matplotlib.pyplot as plt

# y = geometry.cluster_wall_solve_ER(97, 0.0006)
# y2 = geometry.cluster_cosine(97)
# dy = np.diff(y)[0]
# dy2 = np.diff(y2)[0]
# print(dy, dy2)
# f, a = plt.subplots()
# a.plot(y, "-")
# a.plot(y2, "-")
# plt.savefig("test.pdf")
# rstrt


param_default = submit.ParameterSet.from_default()
param_default.dampin = 10.0
param_default.ilos = 1
param_default.Ma2 = 0.3
param_default.set_stag()

params_res = param_default.sweep("resolution", [1, 2, 3])
submit._run_parameters(turbostream.write_grid_from_params, params_res, "res")
rstrt
# # artat

guess_id = [0, 1, 2, 3]
params_sa = param_default.sweep("resolution", [1.0, 2.0, 3.0, 4.0])
for p, gid in zip(params_sa, guess_id):
    p.ilos = 2
    p.guess_file = "res/%04d/output_avg.hdf5" % gid

submit._run_parameters(turbostream.write_grid_from_params, params_sa, "res")
