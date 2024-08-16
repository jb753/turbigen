import numpy as np
import turbigen.compflow_native as cf
import matplotlib.pyplot as plt
from turbigen import util
import turbigen.clusterfunc
import turbigen.grid
import turbigen.fluid
import turbigen.solvers.embsolve

from turbigen import mesh2d as m2d

# Check our MPI rank
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Jump to solver slave process if not first rank
if rank > 0:
    from turbigen.solvers import embsolve

    embsolve.run_slave()
    quit()

dmax = 0.05

#
# Define points
#
pitch = 1.
p = util.AttrDict()
yup = pitch/2.
ydn = -pitch/2.
L = 1.
p.A = m2d.Point(0,yup)
p.B = m2d.Point(L,yup)
p.G = m2d.Point(2*L,yup)
p.H = m2d.Point(3*L,yup)
p.E = m2d.Point(L,0.)
p.F = m2d.Point(2*L,0.)
p.D = m2d.Point(0,ydn)
p.C = m2d.Point(L,ydn)
p.J = m2d.Point(2*L,ydn)
p.I = m2d.Point(3*L,ydn)

#
# Define curves
#
c = util.AttrDict()
ni = 33
nk = 17

c.AB = m2d.Curve.from_uniform(p.A, p.B, ni-4)
c.BG = m2d.Curve.from_uniform(p.B, p.G, ni)
c.GH = m2d.Curve.from_uniform(p.G, p.H, ni)

c.DC = m2d.Curve.from_uniform(p.D, p.C, ni-4)
c.CJ = m2d.Curve.from_uniform(p.C, p.J, ni)
c.JI = m2d.Curve.from_uniform(p.J, p.I, ni)

c.DA = m2d.Curve.from_uniform(p.D, p.A, ni)
c.CB = m2d.Curve.from_uniform(p.C, p.B, ni)
c.JG = m2d.Curve.from_uniform(p.J, p.G, ni)
c.IH = m2d.Curve.from_uniform(p.I, p.H, ni)

c.CE = m2d.Curve.from_uniform(p.C, p.E, nk)
c.EB = m2d.Curve.from_uniform(p.E, p.B, nk)
c.JF = m2d.Curve.from_uniform(p.J, p.F, nk)
c.FG = m2d.Curve.from_uniform(p.F, p.G, nk)
c.EF = m2d.Curve.from_uniform(p.E, p.F, ni)

#
# Define Blocks
#
b = util.AttrDict()
b.ABCD = m2d.Block.from_transfinite(c.AB, c.CB, c.DC, c.DA)
b.BGFE = m2d.Block.from_transfinite(c.BG, c.FG, c.EF, c.EB)
b.EFJC = m2d.Block.from_transfinite(c.EF, c.JF, c.CJ, c.CE)
b.GHIJ = m2d.Block.from_transfinite(c.GH, c.IH, c.JI, c.JG)
b.GHIJ = m2d.Block.from_transfinite(c.GH, c.IH, c.JI, c.JG)
b.ABCD.label = 'ABCD'
b.BGFE.label = 'BGFE'
b.EFJC.label = 'EFJC'
b.GHIJ.label = 'GHIJ'

m2d.find_periodic(b.ABCD, b.BGFE, pitch, None)

conn = m2d.find_periodics(list(b.values()), pitch)

fig, ax = plt.subplots()
ax.axis('equal')
for bi in b.values():
    bi.plot(ax)
for ci in c.values():
    ci.plot(ax)
# plt.show()

g = turbigen.grid.from_mesh2d_xrt(b.values(), conn, c.DA.ds.mean(), pitch)
g.check_coordinates()

# Now apply boundary conditions
g['ABCD'].add_patch(turbigen.grid.InletPatch(i=0))
g['GHIJ'].add_patch(turbigen.grid.OutletPatch(i=-1))

Po1 = 1e5
To1 = 300.
cp = 1005.
ga = 1.4
mu = 1.84e-5
Tu0 = 300.
Alpha1 = 0.
Beta = 0.
Ma1 = 0.3

# Set inlet Ma to get inlet static state
V = cf.V_cpTo_from_Ma(Ma1, ga) * np.sqrt(cp * To1)
P1 = Po1 / cf.Po_P_from_Ma(Ma1, ga)
T1 = To1 / cf.To_T_from_Ma(Ma1, ga)


# Boundary conditions
So1 = turbigen.fluid.PerfectState.from_properties(cp, ga, mu)
So1.set_P_T(Po1, To1)
So1.set_Tu0(Tu0)
g.apply_inlet(So1, Alpha1, Beta)
g.calculate_wall_distance()
g.apply_outlet(P1)

# Initial guess
for b in g:
    b.Vx = V
    b.Vr = 0.0
    b.Vt = 0.0
    b.cp = cp
    b.gamma = ga
    b.mu = mu
    b.Omega = 0.0
    b.set_P_T(P1, T1)
    b.set_Tu0(Tu0)

# Check walls on inlet
iwall, jwall, kwall, wall = g['ABCD'].get_wall()
assert not iwall.any()
assert not kwall.any()
assert jwall[:,(0,-1),:].all()
assert not jwall[:,1:-1,:].any()

# Check walls on outlet

iwall, jwall, kwall, wall = g['GHIJ'].get_wall()
assert not iwall.any()
assert not kwall.any()
assert jwall[:,(0,-1),:].all()
assert not jwall[:,1:-1,:].any()

settings = {
    "i_loss": 0,
    "n_step": 5000,
    "n_step_avg": 1000,
    "n_step_log": 100,
    "plot_conv": True,
}
turbigen.solvers.embsolve.run(g, settings)

#fig, ax = plt.subplots()
#C = g['inlet'][0,:,:]
#ax.plot(C.r, C.t, 'k-', lw=0.2)
#ax.plot(C.r.T, C.t.T, 'k-', lw=0.2)
#ax.axis('equal')
#plt.show()
jplot = 5

fig, ax = plt.subplots()
C = g['ABCD'][:,jplot,-1]
ax.plot(C.x, C.Vx)
C = g['GHIJ'][:,jplot,-1]
ax.plot(C.x, C.Vx)
C = g['BGFE'][:,jplot,-1]
ax.plot(C.x, C.Vx)

fig, ax = plt.subplots()
levP = np.linspace(P1*0.9, Po1, 17)
for b in g:
    C = b[:,jplot,:]
    ax.contourf(C.x, C.rt, C.P, levP)
ax.axis('equal')

fig, ax = plt.subplots()
levV = np.linspace(V*0.5, V*1.5, 17)
for b in g:
    C = b[:,jplot,:]
    ax.contourf(C.x, C.rt, C.Vx, levV)
ax.axis('equal')

fig, ax = plt.subplots()
levV = np.linspace(-V*0.5, V*0.5, 17)
for b in g:
    C = b[:,jplot,:]
    ax.contourf(C.x, C.rt, C.Vt, levV)
ax.axis('equal')
plt.show()

plt.show()
