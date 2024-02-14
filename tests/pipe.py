import turbigen.solvers.native
import turbigen.compflow_native as cf
import turbigen.grid
import numpy as np



# Geometry
h = 0.1
L = h * 4.0
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
rgas = cp * (ga-1.)/ga
V = cf.V_cpTo_from_Ma(M,ga)*np.sqrt(cp*To1)
P1 = Po1/cf.Po_P_from_Ma(M,ga)
T1 = To1/cf.To_T_from_Ma(M,ga)


nj = 20
AR = 1.
ni = int(nj/h*L)
nk = 10
pitch = h/nj*(nk-1)
Nb = int(2.0 * np.pi * rm / pitch)
dt = 2.0 * np.pi / float(Nb)
tv = np.linspace(-dt / 2., dt / 2., nk)
xv = np.linspace(0., L, ni)
rv = np.linspace(rh, rt, nj)

xrt = np.stack(np.meshgrid(xv, rv, tv, indexing='ij'))

# squeeze the nozzle
fac_noz = np.interp(xv, [0., L/2, L], [1., 0.5, 1.])[:,None,None]
xrt[1] = (xrt[1] - rm)*fac_noz + rm

# import matplotlib.pyplot as plt
# plt.plot(xrt[0,:,0,0], xrt[1,:,0,0])
# plt.plot(xrt[0,:,0,0], xrt[1,:,-1,0])
# plt.axis('equal')
# plt.show()


patches = [
    turbigen.grid.InletPatch(i=0),
    turbigen.grid.OutletPatch(i=-1),
    turbigen.grid.PeriodicPatch(k=0),
    turbigen.grid.PeriodicPatch(k=-1),
]

block = turbigen.grid.PerfectBlock.from_coordinates(xrt, Nb, patches)
g = turbigen.grid.Grid([block,])

g.match_patches()
g.check_coordinates()

print('nijk',g[0].shape)
print('dAi', g[0].dAi.shape, g[0].dAi.mean(axis=(-1,-2,-3)))
print('dAj', g[0].dAj.shape, g[0].dAj.mean(axis=(-1,-2,-3)))
print('dAk', g[0].dAk.shape, g[0].dAk.mean(axis=(-1,-2,-3)))

So1 = turbigen.fluid.PerfectState.from_properties(cp, ga, mu)
So1.set_P_T(Po1, To1)
g.apply_inlet(So1, Alpha, Beta)
g.calculate_wall_distance()
g.apply_outlet(P1)

for b in g:
    b.Vx = V
    b.Vr = 0.
    b.Vt = 0.
    b.cp = cp
    b.gamma = ga
    b.mu = mu
    b.Omega = 0.0
    b.set_P_T(P1, T1)


dt = turbigen.solvers.native.get_timestep(g[0])

g.apply_periodic()

import matplotlib.pyplot as plt

wall = turbigen.solvers.native.get_wall(g[0])
# wi, wj, wk = wall

# # plt.plot(wj[1,:,2])
# b = g[0][:,:,0]
# iw = g[0].get_wall()[...,1]
# plt.plot(b.x, b.r, 'k-',lw=0.5)
# plt.plot(b.x.T, b.r.T, 'k-',lw=0.5)
# plt.plot(b.x[iw], b.r[iw], 'b*')
# plt.show()
# quit()


np.set_printoptions(precision=3)
Unow = []
for i in range(5000):

    if not np.mod(i, 100):
        dt = turbigen.solvers.native.get_timestep(g[0])

    dU = turbigen.solvers.native.step(g[0], dt, wall)

    if not np.mod(i, 50):
        b = g[0][ni//2, nj//2, nk//2]
        print(i, np.abs(dU).mean(axis=(-1,-2,-3)))

b = g[0][ni//2,:,:]
fig, ax = plt.subplots()
hm = ax.contourf(b.y, b.z, b.P)
ax.axis('equal')
plt.colorbar(hm)

b = g[0][:,:,nk//2]
fig, ax = plt.subplots()
hm = ax.contourf(b.x, b.r, b.Vr)
ax.axis('equal')
plt.colorbar(hm)

plt.show()
