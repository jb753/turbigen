import numpy as np
import matplotlib.pyplot as plt
import turbigen.flowfield

cons0 = np.moveaxis(np.loadtxt("block_0_mixing.txt").reshape((-1, 5, 1, 41, 49)), 0, -1)
# cons0 = np.moveaxis(np.loadtxt("block_1_mixing.txt").reshape((-1, 5, 1, 41, 41)), 0, -1)

cp = 1100.0
ga = 1.33
mu = 1.8e-5

xrt = np.ones((3,) + cons0.shape[1:])

dat = np.concatenate((xrt, cons0), axis=0)

# Get the raw data and split into conserved variables
x, r, rt, ro, rovx, rovr, rorvt, roe = dat

# Make a perfect flowfield object
F = turbigen.flowfield.PerfectFlowField(x.shape)
F.cp = cp
F.gamma = ga
F.mu = mu
F.Omega = 0.0

# Insert the coordinates and velocities
F.xrt = np.stack((x, r, rt / r))
F.Vxrt = np.stack((rovx, rovr, rorvt / r)) / ro

# Insert the thermodynamic state
u = roe / ro - 0.5 * F.V**2.0
F.set_rho_u(ro, u)

ni, nj, nk, nt = F.shape
plt.plot(np.max(F.s, axis=(0, 1, 2)))
plt.plot(np.min(F.s, axis=(0, 1, 2)))

print(F.shape)
Flast = F[0, :, :, -10].squeeze()
pltvar = Flast.To
print(pltvar.min(), pltvar.max())

plt.figure(layout="constrained")
plt.contourf(
    pltvar,
)
plt.colorbar()
plt.axis("equal")
plt.show()
