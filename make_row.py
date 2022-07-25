import numpy as np
import turbigen.hmesh, turbigen.turbostream, turbigen.design
import matplotlib.pyplot as plt
import os

Po1 = 1e5
To1 = 300.0
rgas = 287.14
ga = 1.4
htr = 0.4
Omega = 50.0
tte = 0.015
Al1 = 0.0

fan = turbigen.design.nondim_fan_total_static(
    phi=0.2,  # Flow coefficient [--]
    Psi_ts=0.1,  # Total-to-static pressure rise coefficient [--]
    Al1=Al1,  # Inlet yaw angle [deg]
    Mab=0.3,  # Blade Mach number [--]
    ga=ga,  # Ratio of specific heats [--]
    eta=0.9,  # Polytropic efficiency [--]
)

# Assuming perfect gas get cp
cp = rgas * ga / (ga - 1.0)
cpTo1 = cp * To1

# Annulus line
rm, Dr = turbigen.design.annulus_line(fan, htr, cpTo1, Omega)

# Specify flow angles at nr points across span
nrad = 10
spf = np.linspace(0.0, 1.0, nrad)
rhub = rm - Dr[0] / 2.0
rtip = rm + Dr[0] / 2.0
r_spf = rhub + spf * (rtip - rhub)
r_rm_spf = r_spf / rm

# Evaluate free vortex swirl distribution
chi = turbigen.design.fan_free_vortex(fan, r_rm_spf)

# Stagger angles are mean camber angle
stag = np.degrees(np.arctan(np.mean(np.tan(np.radians(chi)), axis=1)))

# Repeat a datum thickness distribution for all radial points
A_datum = np.ones((2, 3)) * 0.1
A_datum[:, 0] = 0.2
A = np.tile(A_datum, (nrad, 1, 1))

dx_c = (4.0, 4.0)  # Spacing to inlet and exit boundaries in axial chords
c = 0.2  # Axial chord
nb = 13  # Number of blades

# Generate grid for single blade row
x, r, rt, ilte = turbigen.hmesh.row_grid(
    dx_c, c, rm, Dr, nb, chi, stag, A, tte, spf
)

# Write turbostream input file
run_directory = "fan_example"
if not os.path.exists(run_directory):
    os.mkdir(run_directory)
fname = os.path.join(run_directory, "input.hdf5")
Pout = fan.P_Po1[-1] * Po1
turbigen.turbostream.make_row(
    fname, x, r, rt, ilte, Po1, To1, Al1, Pout, Omega, rgas, ga
)

# pitch_t = 2.*np.pi/nb
# pitch_rt = pitch_t * rm

# iplt = 40
# fig, ax = plt.subplots()
# ax.plot(x,rt[:,iplt,0])
# ax.plot(x,rt[:,iplt,-1]-pitch_t*r[ilte[0],iplt])
# ax.axis("equal")
# plt.savefig("test.pdf")
