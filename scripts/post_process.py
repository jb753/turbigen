import ember.grid
import matplotlib.pyplot as plt
import numpy as np

# Run a simulation first using
#   turbigen mean_line.yaml
# and then change file name below

# Load the solution file into a Grid object
import sys

fname = sys.argv[1]
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
C2 = g[-1][-9, :, :]

fig, ax = plt.subplots()
ax.plot(Cjm.x[:, 0], Cjm.P[:, (0, -1)], "k-x")
ax.set_ylabel("P")

# fig, ax = plt.subplots()
# ax.plot(Cjm.x[:, 0], Cjm.To[:, (0, -1)], "k-x")
# ax.set_ylabel("To")

fig, ax = plt.subplots()
ax.plot(Cjm.x[:, 0], Cjm.conserved_nd[:, 0, :], "-x")
ax.set_ylabel("conserved")

fig, ax = plt.subplots()
ax.contourf(Cmk.x, Cmk.r, Cmk.Beta)
ax.axis("equal")


print(C2.shape)

# Contours of relative Mach number on mid-span plane
fig, ax = plt.subplots()
for bi in g:
    Ci = bi[:, nj // 2, :]
    ax.contourf(Ci.x, Ci.rt, Ci.Alpha_rel, cmap="cubehelix")
ax.axis("equal")
plt.show()

# Contours of relative Mach number on mid-span plane
fig, ax = plt.subplots()
for bi in g:
    Ci = bi[:, nj // 2, :]
    ax.contourf(Ci.x, Ci.rt, Ci.To, cmap="cubehelix")
ax.axis("equal")
plt.show()


# Contours of relative Mach number on mid-span plane
fig, ax = plt.subplots()
for bi in g:
    Ci = bi[:, nj // 2, :]
    ax.contourf(Ci.x, Ci.rt, Ci.P, cmap="cubehelix")
ax.axis("equal")
plt.show()

# Contours of relative Mach number on mid-span plane
fig, ax = plt.subplots()
lev_Ma = np.arange(0.0, 0.8, 0.1)
for bi in g:
    Ci = bi[:, nj // 2, :]
    ax.contourf(Ci.x, Ci.rt, Ci.Ma_rel, lev_Ma, cmap="cubehelix")
ax.axis("equal")
plt.show()

# Contours of Ma downstream of blade
fig, ax = plt.subplots()
lev_Ma = np.arange(0.0, 0.8, 0.05)
cm = ax.contourf(C2.z, C2.y, C2.Max, lev_Ma, cmap="cubehelix")
plt.colorbar(cm)
ax.contour(C2.z, C2.y, C2.Max, [1], colors="k")
ax.set_title("outlet Max")
ax.axis("equal")

# Contours of Ma downstream of blade
fig, ax = plt.subplots()
lev_Ma = np.arange(-0.4, 0.8, 0.05)
cm = ax.contourf(C2.z, C2.y, C2.Vt / C2.a, lev_Ma, cmap="cubehelix")
plt.colorbar(cm)
ax.contour(C2.z, C2.y, C2.Max, [1], colors="k")
ax.set_title("Vt/a")
ax.axis("equal")

# Contours of Ma downstream of blade
fig, ax = plt.subplots()
cm = ax.contourf(C2.z, C2.y, C2.P, cmap="cubehelix")
ax.set_title("outlet P")
ax.axis("equal")
plt.colorbar(cm)
plt.show()
quit()

fig, ax = plt.subplots()
ax.plot(C2.t[-1, :], C2.Vt[-1, :], "-x")
ax.plot(C2.pitch + C2.t[-1, :], C2.Vt[-1, :], "-x")
ax.set_title("Vt")
plt.show()


# pressure on the blade
fig, ax = plt.subplots()
ax.plot(cjm.x[:, 0], cjm.p[:, (0, -1)], "k-x")

# pressure on the blade
fig, ax = plt.subplots()
ax.plot(cjm.x[:, 0], cjm.to[:, (0, -1)], "k-x")

plt.show()
