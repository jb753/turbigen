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

def thickness(m, R, mmax):
    t = np.full_like(m, R)
    t[m<R] = np.sqrt(R**2 - (m[m<R]-R)**2)
    t[m>(1.-R)] = np.sqrt(R**2 - (m[m>(1.-R)]-(1.-R))**2 + 1e-16)
    # t[m>mmax] = R*(1.-(m[m>mmax]-mmax)/(1.-mmax))
    return t

def blade(m, R, mmax, xi):
    cosxi = util.cosd(xi)
    sinxi = util.sind(xi)
    t = thickness(m, R, mmax)
    xc = m*cosxi
    yc = m*sinxi
    xyu = np.stack((xc - t*sinxi, yc + t*cosxi))
    xyl = np.stack((xc + t*sinxi, yc - t*cosxi))
    xyLE = 0.5*(xyu[:,0] + xyl[:,0])
    xyTE = 0.5*(xyu[:,-1] + xyl[:,-1])
    xyu[:,0] = xyLE
    xyl[:,0] = xyLE
    xyu[:,-1] = xyTE
    xyl[:,-1] = xyTE
    return np.stack((xyu, xyl))


def extend_x_from_point(xr0, x1, d0, dmax, ni, ER=1.2):
    xr1 = xr0.copy()
    xr1[0] = x1
    xr12 = np.stack((xr0, xr1),1)
    L = turbigen.util.arc_length(xr12)
    s = turbigen.clusterfunc.single.fixed(d0, dmax, ER, ni, x0=0., x1=L)/L
    if x1 < xr0[0]:
        s = np.flip(s)
    return turbigen.util.interpolate_curve_1d(xr12, s)

def extend_y_from_point(xr0, r1, d0, dmax, ni, ER=1.2):
    xr1 = xr0.copy()
    xr1[1] = r1
    xr12 = np.stack((xr0, xr1),1)
    L = turbigen.util.arc_length(xr12)
    s = turbigen.clusterfunc.single.fixed(d0, dmax, ER, ni, x0=0., x1=L)/L
    if r1 < xr0[1]:
        s = np.flip(s)
    return turbigen.util.interpolate_curve_1d(xr12, s)


fig, ax = plt.subplots()
ax.axis('equal')

nchord = 65
nperiodic = nchord + 33 -1
nomesh = 17
ninlet = 21
noutlet = 21

ER = 1.2

dwall = 0.005
R = 0.05
mmax = 0.5
stag = 0.
dinf = 0.04
Linf = 0.12

#
# Define vertical periodic boundaries
#
pitch = 1.
xlim = [-1.0, 2.0]
x_peridodic = np.linspace(*xlim, nperiodic)
c_pup = m2d.Curve(x_peridodic, pitch/2.).split_by_index((ninlet, -noutlet))
c_pdn = m2d.Curve(x_peridodic, -pitch/2.).split_by_index((ninlet, -noutlet))


#
# Get blade surface coordinates
#
m = util.cluster_cosine(nchord)
xyu, xyl = blade(m, R, mmax, stag)
c_sup = m2d.Curve(*xyu)
c_sdn = m2d.Curve(*xyl)
c_srf = m2d.Curve.from_join(c_sup, c_sdn).roll(c_sup.n-1)

#
# Make the O-block
#
d_omesh = turbigen.clusterfunc.single.fixed(dwall, dinf, ER, nomesh, x0=0., x1=Linf)
b_omesh = m2d.Block.from_offset(c_srf, d_omesh)

#
# Get the points at ends of in/exit h blocks
#
p_end = [c_pdn[1][-1], c_pdn[1][0], c_pup[1][0], c_pup[1][-1]]

#
# Get splits for the o blocks on the outer j line
#
angles = [-135, 135, 45, -45]
curves, isplit, dsplit = m2d.split_by_angle(b_omesh, angles)
c_odn = curves[0]
c_oup = curves[2]

#
# Make the four spokes
#
c_spoke = []
for c, p, d in zip(curves, p_end, dsplit):
    c_spoke.append(m2d.Curve.from_cluster_single(c[0], p, d, dinf))

#
# Join up the cross-passage curves
c_tin = m2d.Curve.from_join(c_spoke[1],curves[1], c_spoke[2] )
c_tout = m2d.Curve.from_join(c_spoke[0],curves[3], c_spoke[3] )
c_iin = m2d.Curve.from_uniform(c_pdn[0][0],c_pup[0][0], c_tin.n)
c_oout = m2d.Curve.from_uniform(c_pdn[-1][-1],c_pup[-1][-1], c_tout.n)
#

#
# Draw curves from
#

b_inlet = m2d.Block.from_transfinite(c_iin, c_pup[0], c_pdn[0], c_tin)
b_outlet = m2d.Block.from_transfinite(c_oout, c_pup[2], c_pdn[2], c_tout)
b_up = m2d.Block.from_transfinite(c_pup[1], c_oup, c_spoke[3], c_spoke[2])
b_dn = m2d.Block.from_transfinite(c_pdn[1], c_odn, c_spoke[0], c_spoke[1])

#
# Plotting
#
c_all = c_pup + c_pdn + [c_srf, c_tin, c_tout, c_odn, c_oup, c_iin, c_oout]# + c_spoke
# for c in c_all:
#     c.plot(ax)

# p_all = p_end + []
# for p in p_all:
#     p.plot(ax)
#
b_omesh.label = 'omesh'
b_inlet.label = 'inlet'
b_outlet.label = 'outlet'
b_up.label = 'upper'
b_dn.label = 'lower'


b_all = [b_omesh, b_inlet, b_outlet, b_up, b_dn]
conn = m2d.find_periodics(b_all, pitch)

# conn = m2d.find_periodic(b_omesh,b_outlet, pitch, None)
# for c in conn:
#     print('***')
#     print(c[0])
#     print(c[1])
# quit()

g = turbigen.grid.from_mesh2d_xrt(b_all, conn, c_iin.ds.mean(), pitch)
g.check_coordinates()

# Now apply boundary conditions
g['inlet'].add_patch(turbigen.grid.InletPatch(i=0))
g['outlet'].add_patch(turbigen.grid.OutletPatch(i=-1))

Po1 = 1e5
To1 = 300.
cp = 1005.
ga = 1.4
mu = 1.84e-5
Tu0 = 300.
Alpha1 = 5.
Beta = 0.
Ma1 = 0.5


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

settings = {
    "i_loss": 1,
    "n_step": 10000,
    "n_step_avg": 1000,
    "n_step_log": 100,
    "nstep_damp": -1,
    # "i_scheme": 1,
    # "smooth4": 0.01,
    # "CFL": 0.4,
    "plot_conv": True,
}
turbigen.solvers.embsolve.run(g, settings)

jplot = 5

fig, ax = plt.subplots()
C = g['omesh'][:,jplot,0]
ax.plot(C.x, C.P)
ts3_dat = np.loadtxt('body_ts3.dat')
ax.plot(*ts3_dat,'k--')


fig, ax = plt.subplots()
levP = np.linspace(P1*0.8, Po1, 17)
for b in g:
    C = b[:,jplot,:]
    ax.contourf(C.x, C.rt, C.P, levP)
ax.axis('equal')

# fig, ax = plt.subplots()
# levw = np.linspace(0., 0.1*pitch, 17)
# for b in g:
#     C = b[:,jplot,:]
#     ax.contourf(C.x, C.rt, C.w, levw)
# for c in c_all:
#     c.plot(ax)
# ax.axis('equal')

plt.show()

