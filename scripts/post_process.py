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
print(g.patches.outlet[0]._adjustment)

b = g[1]
C = b[-9, :, :]

fig, ax = plt.subplots()
lev = np.arange(0.0, 0.50001, 0.05)
print(lev)
lev = np.arange(0.2, 0.4001, 0.02)
print(C.x.mean())
cm = ax.contourf(C.r * C.t, C.r, C.Vx, cmap="cubehelix")
ax.axis("equal")
plt.colorbar(cm)
plt.show()
# quit()

# quit()


# Mass flow plot
x = []
mdot = []
rVt = []
for b in g:
    for i in range(b.ni):
        x.append(b.x[i, 0, 0])
        flow = ember.average.flow_conserved(b[i]) * b.Nb
        mdot.append(flow[0])
        rVt.append(flow[3])

x = np.array(x)
mdot = np.array(mdot)
rVt = np.array(rVt)

fig, ax = plt.subplots()
ax.plot(x, mdot / mdot[0], "-x", label="mass")
ax.plot(x, rVt / rVt[0], "-x", label="mass")
ax.set_ylim(bottom=0)
ax.axhline(0.0)
ax.set_ylim(bottom=-0.5)


b = g[0]
ni, nj, nk = b.shape
C = b[:-1, nj // 2, :-1]

fig, ax = plt.subplots()

lev = np.arange(0.0, 4.0, 0.1)
cm = ax.contourf(C.x, C.rt, b.working.cfl[:, b.nj // 2, :, 3], cmap="cubehelix")
plt.colorbar(cm)
# # ax.plot(C.rt[-9, :], b.working.cfl[-9, b.nj // 2, :, -1])
# # fig, ax = plt.subplots()
# # ax.plot(C.rt[-9, :], C.V[-9, :])
plt.show()


C = b[-9, :, :-1]
fig, ax = plt.subplots()
cm = ax.contourf(C.y, C.z, C.To, cmap="cubehelix")
plt.colorbar(cm)
plt.show()


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
Cjm2 = g[-1][:, nj // 2, :]

fig, ax = plt.subplots()
ax.set_title("i=ni, j=9 Vx vs t")
ax.plot(g[0].Vx[-1, 9], g[0].t[-1, 9], "-x")
plt.show()

fig, ax = plt.subplots()
ax.set_title("i=ni, j=nj//2 To vs t")
ax.plot(g[0].To[-9, -25], g[0].t[-9, -25], "-x")
plt.show()

fig, ax = plt.subplots()
Max_mix = g[1].Ma[0, :, nk // 2].mean()
ax.set_title(f"mixer Max = {Max_mix:.3f}")
ax.plot(g[0].rhoVx[-1, :, nk // 2].T, g[0].r[-1, :, 0], "-x", label="upstream")
ax.plot(g[1].rhoVx[0, :, nk // 2].T, g[1].r[0, :, 0], "--o", label="downstream")
fig, ax = plt.subplots()
ax.set_title(f"mixer rVt")
ax.plot(g[0].rhorVt[-1, :].mean(axis=-1).T, g[0].r[-1, :, 0], "-x", label="upstream")
ax.plot(g[1].rhorVt[0, :].mean(axis=-1).T, g[1].r[0, :, 0], "--o", label="downstream")
ax.set_xlim(left=0)
ax.legend()
fig, ax = plt.subplots()
ax.set_title(f"mixer Vx")
ax.plot(g[0].Vx[-1, :].mean(axis=-1).T, g[0].r[-1, :, 0], "-x", label="upstream")
ax.plot(g[1].Vx[0, :].mean(axis=-1).T, g[1].r[0, :, 0], "--o", label="downstream")
ax.legend()
fig, ax = plt.subplots()
ax.set_title(f"mixer swirl")
ax.plot(g[0].Alpha[-1, :].mean(axis=-1).T, g[0].r[-1, :, 0], "-x", label="upstream")
ax.plot(g[1].Alpha[0, :].mean(axis=-1).T, g[1].r[0, :, 0], "--o", label="downstream")
ax.legend()
fig, ax = plt.subplots()
Max_mix = g[1].Ma[0, :, nk // 2].mean()
ax.set_title(f"mixer P")
ax.plot(g[0].P[-1, :, nk // 2].T, g[0].r[-1, :, 0], "-x", label="upstream")
ax.plot(g[1].P[0, :, nk // 2].T, g[1].r[0, :, 0], "--o", label="downstream")
fig, ax = plt.subplots()
ax.set_title("inlet")
ax.plot(g[0].rhoVx[0, :, (0, nk // 2, -1)].T, g[0].r[0, :, 0], "-x")
fig, ax = plt.subplots()
ax.set_title("outlet rhoVx")
ax.plot(g[1].rhoVx[-1, :, (0, nk // 2, -1)].T, g[1].r[-1, :, 0], "-x")
fig, ax = plt.subplots()
ax.set_title("outlet swirl")
ax.plot(g[1].Alpha[-1, :, (0, nk // 2, -1)].T, g[1].r[-1, :, 0], "-x")
fig, ax = plt.subplots()
ax.set_title("outlet P")
ax.plot(g[1].P[-1, :, (0, nk // 2, -1)].T, g[1].r[-1, :, 0], "-x")
plt.show()

plt.show()

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
lev_Ma = np.arange(0.0, 1.3, 0.1)
ax.axis("equal")
for b in g:
    Ck = b[:, :, nk // 2]
    ax.contourf(Ck.x, Ck.r, Ck.Max, lev_Ma, cmap="cubehelix")
    ax.contour(Ck.x, Ck.r, Ck.Max, [1.0], linestyles="--", colors="k")
plt.show()


# ii = (127, 132)
# ax.plot(Cjm.x[ii[0], (0, 0)], Cjm.P[ii[0], (1, -2)], "r-o")
# ax.plot(Cjm.x[ii[1], (0, 0)], Cjm.P[ii[1], (1, -2)], "b-^")
ax.set_ylabel("P")

fig, ax = plt.subplots()
ax.plot(Cjm.x[:, 0], Cjm.To[:, (0, -1)], "k-x")
ax.plot(Cjm.x[:, 0], Cjm.To[:, (4, -4)], "r-o")
# ii = (127, 132)
# ax.plot(Cjm.x[ii[0], (0, 0)], Cjm.To[ii[0], (1, -2)], "r-o")
# ax.plot(Cjm.x[ii[1], (0, 0)], Cjm.To[ii[1], (1, -2)], "b-^")
ax.set_ylabel("To")

# fig, ax = plt.subplots()
# ax.plot(Cjm.x[:, 0], Cjm.To[:, (0, -1)], "k-x")
# ax.set_ylabel("To")

fig, ax = plt.subplots()
for ii, lab in enumerate(["rho", "rhoVx", "rhoVr", "rhorVt", "rhoE"]):
    ax.plot(Cjm.x[:, 0], Cjm.conserved_nd[:, 0, ii], "-x", label=lab)
ax.set_ylabel("conserved")
ax.set_title("stator")
ax.legend()

fig, ax = plt.subplots()
for ii, lab in enumerate(["rho", "rhoVx", "rhoVr", "rhorVt", "rhoE"]):
    ax.plot(Cjm2.x[:, 0], Cjm2.conserved_nd[:, 0, ii], "-x", label=lab)
ax.set_ylabel("conserved")
ax.set_title("rotor")
ax.legend()

print(C2.shape)
plt.show()
# quit()

# Contours of relative Mach number on mid-span plane
fig, ax = plt.subplots()
lev_Ma = np.arange(0.0, 0.8, 0.1)
jplot = g[0].nj // 2
# jplot = -9
for bi in g:
    Ci = bi[:, jplot, :]
    ax.contourf(Ci.x, Ci.rt, Ci.Ma_rel, lev_Ma, cmap="cubehelix")
    ax.contourf(Ci.x, Ci.rt + Ci.r * Ci.pitch, Ci.Ma_rel, lev_Ma, cmap="cubehelix")
ax.axis("equal")

fig, ax = plt.subplots()
for bi in g:
    Ci = bi[:, jplot, :]
    ax.contourf(Ci.x, Ci.rt, Ci.Max, lev_Ma, cmap="cubehelix")
    ax.contourf(Ci.x, Ci.rt + Ci.r * Ci.pitch, Ci.Max, lev_Ma, cmap="cubehelix")
    ax.contourf(Ci.x, Ci.rt + 2 * Ci.r * Ci.pitch, Ci.Max, lev_Ma, cmap="cubehelix")
ax.axis("equal")

fig, ax = plt.subplots()
for bi in g:
    Ci = bi[:, jplot, :]
    ax.contourf(Ci.x, Ci.rt, Ci.To, cmap="cubehelix")
    cm = ax.contourf(Ci.x, Ci.rt + Ci.r * Ci.pitch, Ci.To, cmap="cubehelix")
ax.axis("equal")
ax.set_title("To")
plt.colorbar(cm)
# Contours of relative Mach number on mid-span plane
fig, ax = plt.subplots()
lev_ent = np.arange(0.0, 0.8, 0.1)
for bi in g:
    Ci = bi[:, nj // 2, :]
    # ax.contourf(Ci.x, Ci.rt, Ci.Ma_rel, lev_Ma, cmap="cubehelix")
    ax.contourf(Ci.x, Ci.rt + Ci.r * Ci.pitch, np.exp(Ci.s / Ci.Rgas))
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
