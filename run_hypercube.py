"""Run a set of tabu searches over a hypercube of parameters."""

from turbigen import submit
from scipy.stats.qmc import LatinHypercube
import numpy as np
import uuid

Ma2_lim = (0.4, 0.8)
phi_lim = (0.4, 1.2)
psi_lim = (1.0, 2.6)
Lam_lim = (.3, .7)

xmin, xmax = np.column_stack((phi_lim, psi_lim, Lam_lim, Ma2_lim))

q = LatinHypercube(d=4).random(8)
v = xmin * (1.-q) + xmax*q

param_default = submit.ParameterSet.from_default()

for vi in v:
    param_now =  param_default.copy()
    phi, psi, Lam, Ma2 = vi
    param_now.phi = phi
    param_now.psi = psi
    param_now.Lam = Lam
    param_now.Ma2 = Ma2
    case_str = str(uuid.uuid4())[:8]
    submit.run_search(param_now, case_str)
