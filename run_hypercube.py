"""Run a set of tabu searches over a hypercube of parameters."""

from turbigen import submit
from scipy.stats.qmc import LatinHypercube
import numpy as np
import uuid

import matplotlib.pyplot as plt

Ma2_lim = (0.3, 1.1)
phi_lim = (0.4, 1.2)
psi_lim = (1.0, 2.6)
Lam_lim = (0.2, 0.8)
Co_lim = (0.4, 0.8)

xmin, xmax = np.column_stack((phi_lim, psi_lim, Lam_lim, Ma2_lim, Co_lim))

N = 128
q = LatinHypercube(d=5, optimization="random-cd").random(N)
v = xmin * (1.0 - q) + xmax * q

param_default = submit.ParameterSet.from_default()

# phi, psi, Lam, Ma2 = v.T
# fig, ax = plt.subplots(1, 4)
# labs = ["phi", "psi", "Lam", "Ma2"]
# for vi, ai, li in zip(v.T, ax, labs):
#     ai.hist(vi)
#     ai.set_xlabel(li)
# plt.tight_layout()
# plt.savefig("hist.pdf")

# quit()

for vi in v:
    param_now = param_default.copy()
    phi, psi, Lam, Ma2, Co = vi

    param_now.phi = phi
    param_now.psi = psi
    param_now.Lam = Lam
    param_now.Ma2 = Ma2
    param_now.Co = list([Co + 0.0, Co + 0.0])
    case_str = str(uuid.uuid4())[:8]
    submit.run_search(param_now, case_str, "run5")
