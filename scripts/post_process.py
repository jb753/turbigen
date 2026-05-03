import ember.grid
import matplotlib.pyplot as plt
import numpy as np
import ember.average

# Run a simulation first using
#   turbigen mean_line.yaml
# and then change file name below

# Load the solution file into a Grid object
import sys

fname = sys.argv[1]
g = ember.grid.Grid.read_emb(fname)


# Mass flow plot
x = []
mdot = []
for b in g:
    for i in range(b.ni):
        x.append(b.x[i, 0, 0])
        mdot.append(ember.average.flow_mass(b[i]) * b.Nb)
x = np.array(x)
mdot = np.array(mdot)

fig, ax = plt.subplots()
ax.plot(x, mdot, "-x")
ax.set_ylim(bottom=0)
# plt.show()
# quit()


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
ax.plot(g[0].Vx[-1, 9], g[0].t[-1, 9], "-x")
plt.show()

fig, ax = plt.subplots()
ax.set_title("mixer")
ax.plot(g[0].rhoVx[-1, :, nk // 2].T, g[0].r[-1, :, 0], "-x")
ax.plot(g[1].rhoVx[0, :, nk // 2].T, g[1].r[0, :, 0], "--o")
fig, ax = plt.subplots()
ax.set_title("inlet")
ax.plot(g[0].rhoVx[0, :, (0, nk // 2, -1)].T, g[0].r[0, :, 0], "-x")
fig, ax = plt.subplots()
ax.set_title("outlet")
ax.plot(g[1].rhoVx[-1, :, (0, nk // 2, -1)].T, g[1].r[-1, :, 0], "-x")
plt.show()

plt.show()
quit()

fig, ax = plt.subplots()
lev_Ma = np.linspace(0.0, 0.5, 11) * g[0].pitch
ax.axis("equal")
for b in g:
    Ck = b[:, :, nk // 2]
    ax.contourf(Ck.x, Ck.r, Ck.wdist, lev_Ma, cmap="cubehelix")
fig, ax = plt.subplots()
ax.axis("equal")
for b in g:
    Cj = b[:, nj // 2, :]
    ax.contourf(Cj.x, Cj.rt, Cj.wdist, lev_Ma, cmap="cubehelix")


plt.show()


fig, ax = plt.subplots()
lev_Ma = np.arange(0.0, 0.8, 0.1)
ax.axis("equal")
for b in g:
    Ck = b[:, :, nk // 2]
    ax.contourf(Ck.x, Ck.r, Ck.Max, lev_Ma, cmap="cubehelix")
plt.show()


# ii = (127, 132)
# ax.plot(Cjm.x[ii[0], (0, 0)], Cjm.P[ii[0], (1, -2)], "r-o")
# ax.plot(Cjm.x[ii[1], (0, 0)], Cjm.P[ii[1], (1, -2)], "b-^")
ax.set_ylabel("P")

fig, ax = plt.subplots()
ax.plot(Cjm.x[:, 0], Cjm.To[:, (0, -1)], "k-x")
# ii = (127, 132)
# ax.plot(Cjm.x[ii[0], (0, 0)], Cjm.To[ii[0], (1, -2)], "r-o")
# ax.plot(Cjm.x[ii[1], (0, 0)], Cjm.To[ii[1], (1, -2)], "b-^")
ax.set_ylabel("To")

# fig, ax = plt.subplots()
# ax.plot(Cjm.x[:, 0], Cjm.To[:, (0, -1)], "k-x")
# ax.set_ylabel("To")

fig, ax = plt.subplots()
ax.plot(Cjm.x[:, 0], Cjm.conserved_nd[:, 0, :], "-x")
ax.set_ylabel("conserved")

print(C2.shape)

# Contours of relative Mach number on mid-span plane
fig, ax = plt.subplots()
lev_Ma = np.arange(0.0, 0.8, 0.1)
jplot = 9
for bi in g:
    Ci = bi[:, jplot, :]
    ax.contourf(Ci.x, Ci.rt, Ci.Ma_rel, lev_Ma, cmap="cubehelix")
    ax.contourf(Ci.x, Ci.rt + Ci.r * Ci.pitch, Ci.Ma_rel, lev_Ma, cmap="cubehelix")
ax.axis("equal")

# Contours of relative Mach number on mid-span plane
fig, ax = plt.subplots()
lev_ent = np.arange(0.0, 0.8, 0.1)
for bi in g:
    Ci = bi[:, nj // 2, :]
    # ax.contourf(Ci.x, Ci.rt, Ci.Ma_rel, lev_Ma, cmap="cubehelix")
    ax.contourf(Ci.x, Ci.rt + Ci.r * Ci.pitch, np.exp(Ci.s / Ci.rgas))
ax.set_title("entropy")
ax.axis("equal")

# Contours of relative Mach number on mid-span plane
fig, ax = plt.subplots()
for bi in g:
    Ci = bi[:, nj // 2, :]
    ax.plot(Ci.x, Ci.rt, "k-", lw=0.5)
    ax.plot(Ci.x.T, Ci.rt.T, "k-", lw=0.5)
ax.axis("equal")
plt.show()
quit()


quit()

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
