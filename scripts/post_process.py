import ember.grid
import matplotlib.pyplot as plt
import numpy as np

# Run a simulation first using
#   turbigen mean_line.yaml
# and then change file name below

# Load the solution file into a Grid object
fname = "test_run/0409/soln.pkl"
g = ember.grid.Grid.read_emb(fname)

# Rotor only has a single block, extract it by indexing into grid
b = g[0]

# The block is a 3D array
# i is x, streamwise
# j is r, spanwise
# k is t, pitchwise
ni, nj, nk = b.shape
print(f"Block shape: ni={ni}, nj={nj}, nk={nk}")

# Take planes at mid-span and mid-pitch and downstream of blade
Cjm = b[:, nj // 2, :]
Cmk = b[:, :, nk // 2]
C2 = b[-33, :, :]

# Plot the mid-span grid lines in x/rt blade-to-blade plane
fig, ax = plt.subplots()
ax.plot(Cjm.x, Cjm.rt, "k-", lw=0.5)
ax.plot(Cjm.x.T, Cjm.rt.T, "k-", lw=0.5)
ax.plot(C2[nj // 2, :].x, C2[nj // 2, :].rt, "r-")  # Downstream cut plane
ax.axis("equal")

# Plot the mid-pitch grid lines in x/r meridional plane
fig, ax = plt.subplots()
ax.plot(Cmk.x, Cmk.r, "k-", lw=0.5)
ax.plot(Cmk.x.T, Cmk.r.T, "k-", lw=0.5)
ax.axis("equal")

# Contours of relative Mach number on mid-span plane
fig, ax = plt.subplots()
lev_Ma = np.arange(0.0, 1.0, 0.1)
ax.contourf(Cjm.x, Cjm.rt, Cjm.Ma_rel, lev_Ma, cmap="cubehelix")
ax.axis("equal")

# Contours of Ma downstream of blade
fig, ax = plt.subplots()
ax.contourf(C2.z, C2.y, C2.Ma, lev_Ma, cmap="cubehelix")
ax.axis("equal")

# Pressure on the blade
fig, ax = plt.subplots()
ax.plot(Cjm.x[:, 0], Cjm.P[:, (0, -1)], "k-")

plt.show()
