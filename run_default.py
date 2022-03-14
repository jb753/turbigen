from turbigen import design, hmesh
import json
import numpy as np


with open("turbigen/default_params.json", "r") as f:
    params = json.load(f)

# Mean-line design using the non-dimensionals
stg = design.nondim_stage_from_Lam(**params["mean-line"])

for ai in stg._fields:
    print(ai)

# Set geometry using dimensional bcond and 3D design parameter
bcond_and_3d_params = dict(params["bcond"], **params["3d"])
Dstg = design.get_geometry(stg, **bcond_and_3d_params)

Dstg.free_vortex_vane([0.,0.5,1.0])

print(Dstg)

quit()

# Evaluate blade angles
r = np.vstack([np.linspace(rm-Dri/2.,rm+Dri/2.,3) for Dri in Dr])
chi = design.free_vortex(stg, r/rm, (0.0, 0.0))

sec_xrt = hmesh.section_coords(r, chi[0])
print(sec_xrt.shape)


# Change the base run dir to separate different groups of runs
# base_run_dir = "run"

# params["mean-line"]["phi"] = 0.392
# params["mean-line"]["psi"] = 1.009

# params["mesh"]["Asta"] = [[0.36890363, 0.5244426 , 0.29232573, 0.29764035],[0.36890363, 0.5244426 , 0.29232573, 0.29764035]]
# params["mesh"]["Arot"] = [[0.36890363, 0.5244426 , 0.29232573, 0.29764035],[0.36890363, 0.5244426 , 0.29232573, 0.29764035]]

# Submit a job to the cluster
# submit.run(params, base_run_dir, plot_stuff=True)
