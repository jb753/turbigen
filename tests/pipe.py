import turbigen.solvers.native
import turbigen.compflow_native as cf
import turbigen.grid
import numpy as np
from timeit import default_timer as timer
import sys

# Check our MPI rank
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Jump to solver slave process if not first rank
if rank > 0:
    turbigen.solvers.native.run_slave()
    sys.exit(0)


# Geometry
h = 0.1
L = h * 4.0
skew = 0.0
htr = 0.9
rm = 0.5 * h * (1.0 + htr) / (1.0 - htr)
rh = rm - 0.5 * h
rt = rm + 0.5 * h

# Boundary conditions
ga = 1.4
cp = 1005.0
mu = 1.8e-5
Alpha = 0.0
Beta = 0.0
Po1 = 1e5
To1 = 300.0

M = 0.3
rgas = cp * (ga - 1.0) / ga
V = cf.V_cpTo_from_Ma(M, ga) * np.sqrt(cp * To1)
P1 = Po1 / cf.Po_P_from_Ma(M, ga)
T1 = To1 / cf.To_T_from_Ma(M, ga)


nj = 17
nk = 17

AR = 1.0
ni = int(nj / h * L)
print(f"ni={ni}, nj={nj}, nk={nk}")
print(f"ncell/1e6={ni*nj*nk/1e6}")
# quit()
pitch = h / (nj - 1) * (nk - 1) * AR
Nb = int(2.0 * np.pi * rm / pitch)
print(Nb)
dt = 2.0 * np.pi / float(Nb)
dk = rm * dt / (nk - 1)
dj = h / nj
di = L / ni
print("AR", dj / dk)
print("AR", dj / di)
# quit()
tv = np.linspace(-dt / 2.0, dt / 2.0, nk)
xv = np.linspace(0.0, L, ni)
rv = np.linspace(rh, rt, nj)

xrt = np.stack(np.meshgrid(xv, rv, tv, indexing="ij"))

xrt[2] += xrt[0] * np.tan(np.radians(skew)) / xrt[1]

# squeeze the nozzle
fac_noz = np.interp(xv, [0.0, L / 2, L], [1.0, 0.65, 1.0])[:, None, None]
xrt[1] = (xrt[1] - rm) * fac_noz + rm

patches = [
    turbigen.grid.InletPatch(i=0),
    turbigen.grid.OutletPatch(i=-1),
    turbigen.grid.PeriodicPatch(k=0),
    turbigen.grid.PeriodicPatch(k=-1),
]

blocks = []
nblock = 4

istb = [ni // nblock * iblock for iblock in range(nblock)]
ienb = [ni // nblock * (iblock + 1) + 1 for iblock in range(nblock)]
ienb[-1] = ni

for iblock in range(nblock):

    # Special case for only one block
    if nblock == 1:
        patches = [
            turbigen.grid.InletPatch(i=0),
            turbigen.grid.OutletPatch(i=-1),
            turbigen.grid.PeriodicPatch(k=0),
            turbigen.grid.PeriodicPatch(k=-1),
        ]

    # First block has an inlet
    elif iblock == 0:
        patches = [
            turbigen.grid.InletPatch(i=0),
            turbigen.grid.PeriodicPatch(i=-1),
            turbigen.grid.PeriodicPatch(k=0),
            turbigen.grid.PeriodicPatch(k=-1),
        ]

    # Last block has outlet
    elif iblock == (nblock - 1):
        patches = [
            turbigen.grid.PeriodicPatch(i=0),
            turbigen.grid.OutletPatch(i=-1),
            turbigen.grid.PeriodicPatch(k=0),
            turbigen.grid.PeriodicPatch(k=-1),
        ]

    # Middle blocks are both periodic
    else:
        patches = [
            turbigen.grid.PeriodicPatch(i=0),
            turbigen.grid.PeriodicPatch(i=-1),
            turbigen.grid.PeriodicPatch(k=0),
            turbigen.grid.PeriodicPatch(k=-1),
        ]
    blocks.append(
        turbigen.grid.PerfectBlock.from_coordinates(
            xrt[:, istb[iblock] : ienb[iblock], :, :], Nb, patches
        )
    )
    blocks[-1].label = f"b{iblock}"

g = turbigen.grid.Grid(blocks)


def split_block(b, ind, axis):
    # Split this block into two blocks
    # b1 is from start to ind
    # b2 is from ind to end

    # Get patches that are intersected
    for p in b.patches:
        print(p)


# split_block(g[0])
# quit()


g.match_patches()
g.check_coordinates()

print("i", turbigen.util.vecnorm(g[0].dli).min())
print("j", turbigen.util.vecnorm(g[0].dlj).min())
print("k", turbigen.util.vecnorm(g[0].dlk).min())


# for b in g:
#     print(b.x.mean())
# quit()

# print(g[0].patches[1].get_cut().x.mean())
# print(g[1].patches[0].get_cut().x.mean())
# ind1 = g[1].patches[0].get_flat_indices('F')
# assert np.allclose(
#     g[0].t.ravel(order='F')[ind0],
#     g[1].t.ravel(order='F')[ind1]
#     )

# quit()

So1 = turbigen.fluid.PerfectState.from_properties(cp, ga, mu)
So1.set_P_T(Po1, To1)
g.apply_inlet(So1, Alpha, Beta)
g.calculate_wall_distance()
g.apply_outlet(P1)

for b in g:
    b.Vx = V
    b.Vr = 0.0
    b.Vt = V * np.tan(np.radians(Alpha))
    b.cp = cp
    b.gamma = ga
    b.mu = mu
    b.Omega = 0.0
    b.set_P_T(P1, T1)

# dt = turbigen.solvers.native.get_timestep(g[0])

g.apply_periodic()

import matplotlib.pyplot as plt

np.set_printoptions(precision=3)

settings = {
    "n_step": 10000,
    "n_step_avg": 1000,
    "n_step_log": 100,
    "i_loss": 0,
    "damping_factor": 25.0,
}

tst = timer()
turbigen.solvers.native.run(g, settings)
ten = timer()
print(ten - tst)

# b = g[0][ni//2,:,:]
fig, ax = plt.subplots()
for b in g:
    bc = b[:, 0, -1]
    hm = ax.plot(bc.x, bc.P / Po1, "-x")
ax.plot(bc.x[-1], P1 / Po1, "k*")
ax.set_ylim((0.4, 1.0))

# b = g[0][ni//2,:,:]
fig, ax = plt.subplots()
for b in g:
    bc = b[:, 0, -1]
    hm = ax.plot(bc.x, bc.To / To1, "-x")
# ax.set_ylim((0.4,2.))

# b = g[0][ni//2,:,:]
fig, ax = plt.subplots()
for b in g:
    bc = b[:, 0, -1]
    hm = ax.plot(bc.x, bc.Ma, "-x")
ax.set_ylabel("Ma")
# ax.set_ylim((0.4,2.))

# b = g[0][ni//2,:,:]
fig, ax = plt.subplots()
for b in g:
    bc = b[:, 0, -1]
    hm = ax.plot(bc.x, bc.Alpha, "-x")
ax.set_ylabel("Alpha")
# ax.set_ylim((0.4,2.))

# b = g[0][ni//2,:,:]
fig, ax = plt.subplots()
for b in g:
    ni, nj, nk = b.shape
    bc = b[ni // 2, :, nk // 2]
    hm = ax.plot(bc.Vx, bc.r)
    ax.set_title("Vprof")
    plt.tight_layout()

# ax.axis('equal')
# plt.colorbar(hm)
plt.show()
