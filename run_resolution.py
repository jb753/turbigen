"""Run a resolution study on datum design."""

from turbigen import hmesh, geometry
# from turbigen import submit, turbostream, hmesh, geometry
import numpy as np
import matplotlib.pyplot as plt

y = geometry.cluster_wall_solve_npts(1.1, 0.001)

y = geometry.cluster_wall_solve_ER(65, 0.001)

# rstrt
y = geometry.cluster_wall(65, 1.11, 0.001)
# dy = np.diff(y)

f, a = plt.subplots()
a.plot(y,'k-x')
plt.savefig('test.pdf')




# param_default = submit.ParameterSet.from_default()
# param_default.set_stag()
# param_default.resolution=5
# submit._run_parameters(turbostream.write_grid_from_params,param_default, 'res')
