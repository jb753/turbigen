import turbigen.solvers.native
import turbigen.compflow_native as cf
import turbigen.grid
import numpy as np



# Geometry
h = 0.1
L = h * 4.0
htr = 0.99
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
nk = 6
pitch = h/nj*(nk-1)
Nb = int(2.0 * np.pi * rm / pitch)
dt = 2.0 * np.pi / float(Nb)
tv = np.linspace(-dt / 2., dt / 2., nk)
xv = np.linspace(0., L, ni)
rv = np.linspace(rh, rt, nj)
xrt = np.stack(np.meshgrid(xv, rv, tv, indexing='ij'))

patches = [
    turbigen.grid.PeriodicPatch(k=0),
    turbigen.grid.PeriodicPatch(k=-1),
    turbigen.grid.InletPatch(i=0),
    turbigen.grid.OutletPatch(i=-1),
]

block = turbigen.grid.PerfectBlock.from_coordinates(xrt, Nb, patches)
g = turbigen.grid.Grid([block,])
g.match_patches()
g.check_coordinates()

So1 = turbigen.fluid.PerfectState.from_properties(cp, ga, mu)
So1.set_P_T(Po1, To1)
g.apply_inlet(So1, Alpha, Beta)
g.calculate_wall_distance()
g.apply_outlet(P1-2.)

for b in g:
    b.Vx = V
    b.Vr = 0.
    b.Vt = 0.
    b.cp = cp
    b.gamma = ga
    b.mu = mu
    b.Omega = 0.0
    b.set_P_T(P1, T1)

CFL=0.4

dt = turbigen.solvers.native.get_timestep(g[0], CFL)

g.apply_periodic()

import matplotlib.pyplot as plt

for i in range(10000):
    print(g[0].P.mean())
    print(i)
    try:
        turbigen.solvers.native.step(g[0], dt)
    except:
        b = g[0][:,:,0]
        fig, ax = plt.subplots()
        ax.contourf(b.x, b.r, b.P)
        plt.show()
        break


    if not np.mod(i, 10):
        dt = turbigen.solvers.native.get_timestep(g[0], CFL)


        # b = g[0][-1,g[0].nj, lev_P//2,:]
        # fig, ax = plt.subplots()
        # ax.plot(b.rt, b.P)
        # plt.show()

