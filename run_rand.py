"""Run a set of random design vectors about datum geometry."""


from turbigen import submit, turbostream
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
import subprocess
import os, sys

# from scipy.stats.qmc import LatinHypercube

param_default = submit.ParameterSet.from_default()
param_default.set_stag()
param_default.rtol = 0.05
param_default.min_Rins = 0.0

base_dir = "rand"
datum_file = os.path.join(base_dir, "datum_param.json")

xg = np.zeros((7,))

eps = 0.1
obj, grad, constr, cache = submit._wrap_for_grad(
    turbostream.write_grid_from_params, param_default, base_dir, 0, eps, False
)

# bnd = np.array(submit._assemble_bounds_rel(1)).T
# print(bnd)


N = 1024
# q = LatinHypercube(d=7, optimization="random-cd").random(N)
# xn = bnd[0] * (1.0 - q) + bnd[1] * q
# np.savetxt('random_designs.txt',xn)
xn = np.loadtxt("random_designs.txt")
xns = np.array_split(xn, N / 4)

for xni in xns:
    obj(xni)


quit()


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
