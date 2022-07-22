"""Run a resolution study on datum design."""


OBJECTIVE_KEYS = ["eta_lost", "psi", "phi", "Lam", "Ma2", "runid", "resid"]
CONSTR_KEYS = ["psi", "phi", "Lam", "Ma2"]


from turbigen import submit, turbostream, geometry
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
import subprocess
import os, sys

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
# param_default.Ma2 = 0.3
param_default.set_stag()
param_default.rtol = 0.05
param_default.min_Rins = 0.0

base_dir = "grad"
datum_file = os.path.join(base_dir, "datum_param.json")

xg = np.zeros((7,))
# xg[0] = -0.1

eps = 0.1
obj, grad, constr, cache = submit._wrap_for_grad(
    turbostream.write_grid_from_params, param_default, base_dir, 0, eps, False
)

bnd = submit._assemble_bounds_rel(1)


def do_callback(xk):
    print("Lost effy %.4f" % obj(xk))
    sys.stdout.flush()


scipy.optimize.minimize(
    fun=obj,
    x0=xg,
    method="SLSQP",
    jac=grad,
    tol=0.05,
    bounds=bnd,
    constraints=constr,
    options={"disp": True},
    callback=do_callback,
)

effy = np.array([cache[k][0] for k in cache])
print(effy.min(), effy.max())

# print(obj(xg))
# print(grad(xg))
# print(grad(xg))
# # print(grad(xg))

quit()

param_default.write_json(datum_file)

param_default.guess_file = "/rds/project/gp10006/rds-gp10006-pullan-mhi/jb753/turbigen/grad/0000/output_avg.hdf5"

# param_corrected = submit._initial_search(turbostream.write_grid_from_params)

submit._grad("grad", param_default, turbostream.write_grid_from_params, 0)
