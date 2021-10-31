import numpy as np
import design, turbostream, hmesh

# Mean-line design
nondim_params = {
    "phi": 0.8,
    "psi": 1.6,
    "Lam": 0.5,
    "Al1": 0.,
    "Ma": 0.75,
    "eta": 0.95,
    "ga": 1.33,
    }
stg = design.nondim_stage_from_Lam(**nondim_params)

# Dimensional conditions set the geometry
dim_params = {
    "To1": 1600.,
    "Po1": 16e5,
    "rgas": 287.14,
    "Omega": 2.0 * np.pi * 50.0,
    "htr": 0.9,
    "Re": 4.0e6,
    "Co": 0.65
    }
geometry_params = design.get_geometry(stg, **dim_params)

# Add axial spacings
geometry_params['dx_c'] = (2.0, 1.0, 3.0)
# Add deviations
geometry_params['dev'] = (0., 0.)

# Generate grid
grid = hmesh.stage_grid( stg, **geometry_params)

# Make Turbostream grid and write out
g = turbostream.generate( stg, grid, dim_params )
g.write_hdf5("run/input_1.hdf5")
