import numpy as np
import design, turbostream, hmesh

ga = 1.33
To1 = 1600.0
Po1 = 16.0e5
rgas = 287.14
cp = rgas * ga / (ga - 1.0)
Omega = 2.0 * np.pi * 50.0
Z = 0.8
Co = 0.65
phi = 0.8
psi = 1.6
Lam = 0.5
Al1 = 0.0
Ma = 0.75
eta = 0.95
Re = 4.0e6
rpm = Omega/ 2. /np.pi * 60.

htr = 0.9

# Turbine stage design
stg = design.nondim_stage_from_Lam(
    phi, psi, Lam, Al1, Ma, ga, eta
)

PR = 0.5

print(design.pitch_circ(stg, 0.65))
print(design.pitch_Zweifel(stg, (0.8,0.8)))


g = turbostream.generate(*hmesh.stage_grid(stg, cp*To1, htr, Omega, Po1, Re, rgas, (0.,0.), (2.,1.,3.), Co), rpm_rotor=rpm, Po1=Po1, To1=To1,P3=PR*Po1, stg=stg, ga=ga, rgas=rgas)

g.write_hdf5('run/input_1.hdf5')
