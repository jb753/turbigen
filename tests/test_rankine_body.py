import numpy as np
import matplotlib.pyplot as plt
from turbigen import util
import turbigen.clusterfunc

from turbigen import mesh2d as m2d

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
for c in c_all:
    c.plot(ax)

# p_all = p_end + []
# for p in p_all:
#     p.plot(ax)
#

b_all = [b_omesh, b_inlet, b_outlet, b_up, b_dn]
for b in b_all:
    b.plot(ax)

#
plt.show()
xy = b_omesh.xy
print(b_omesh.shape)
print(b_omesh.ij[0,0,:])
print(b_omesh.ij[0,:,0])

quit()

#
# plt.show()
#
#
#
# d1 = d_omesh[-1]-d_omesh[-2]
# d2 = d1*4
#
# CO = bO[:,-1]
# angles = [-135, 135, 45, -45]
# curves, isplit = CO.split_by_angle(angles)
#
# dW = bO[isplit[1]:isplit[2],-2:].dsj.mean()
# dE = bO[isplit[3]:isplit[4],-2:].dsj.mean()
#
# bW = m2d.Block.from_project_to_x(curves[1], xlim[0], dW, 2.)
# bE = m2d.Block.from_project_to_x(curves[3], xlim[1], dE, 2.)
# # Cin = curves[1].project_to_x(xlim[0])
# # Cout = curves[-1].project_to_x(xlim[1])
#
#
# fig, ax = plt.subplots()
# ax.axis('equal')
# ax.plot(*bO.xy, 'k-')
# ax.plot(*bO.T.xy, 'k-')
# for c in curves:
#     ax.plot(*c.xy, '-o')
# # ax.plot(*Cin.xy, 'b-x')
# # ax.plot(*Cout.xy, 'b-x')
# ax.plot(*bW.xy, 'k-')
# ax.plot(*bW.T.xy, 'k-')
# ax.plot(*bE.xy, 'k-')
# ax.plot(*bE.T.xy, 'k-')
# plt.show()
# quit()
#
#
# # # Find split points
# # ang = turbigen.util.angle_curve(xyso[:,:,-1])
# # isplit = [np.argmin(np.abs(ang - si)) for si in [45, -45, 135, -135]]
# # displit = np.diff(isplit)
# # print(displit)
#
# # ni1 = 25
# # xy_WNW = extend_x_from_point(xyso[:, isplit[0], -1], xlim[0], d1, d2, ni1)
# # xy_ENE = extend_x_from_point(xyso[:, isplit[1], -1], xlim[1], d1, d2, ni1)
# # xy_ESE = extend_x_from_point(xyso[:, isplit[2], -1], xlim[0], d1, d2, ni1)
# # xy_WSW = extend_x_from_point(xyso[:, isplit[3], -1], xlim[1], d1, d2, ni1)
#
# # ni_per = 2*ni1 + displit[0] - 1
# # xy12_top = np.array((xlim, (pitch/2, pitch/2)))
#
# # xy_bot = xy_top.copy()
# # xy_bot[1] *= -1.
#
# # xy_top_match = np.concatenate(
# #     (
# #         xy_WNW[:,:-1],
# #         xyso[:,isplit[0]:(isplit[1]+1),-1],
# #         xy_ENE[:,1:]
# #     ),
# #     axis=1
# # )
# # s_top = turbigen.util.cum_arc_length(xy_top_match)
# # s_top /= s_top[-1]
# # s_top = turbigen.util.smooth_1d(s_top, 1.0, 100)
#
# xy_top = turbigen.util.interpolate_curve_1d(xy12_top, s_top)
#
# nk1 = 9
#
# xy12_in_N = np.stack((xy_top_match[:,0],xy_top[:,0]),axis=1)
# xy_in_N = turbigen.util.interpolate_curve_1d(xy12_in_N, np.linspace(0., 1., nk1))
# xy12_out_N = np.stack((xy_top_match[:,-1],xy_top[:,-1]),axis=1)
# xy_out_N = turbigen.util.interpolate_curve_1d(xy12_out_N, np.linspace(0., 1., nk1))
#
#
# print(xy_top.shape)
# print(xy_top_match.shape)
# xy_N = turbigen.util.interpolate_transfinite(
#         (
#             xy_top_match,
#             xy_in_N,
#             xy_top,
#             xy_out_N,
#         )
# )
#
# # xy_in_W = turbigen.util.interpolate_curve_1d(
# # print(isplit)
# # fig, ax = plt.subplots()
# # ax.plot(ang)
# # ax.plot(np.arange(len(ang))[isplit],ang[isplit],'kx')
# # plt.show()
#
# fig, ax = plt.subplots()
# # ax.plot(*xyu, '-x')
# print(xyso.shape)
# ax.plot(*xyso, 'k-')
# ax.plot(*xyso.transpose(0,2,1), 'k-')
# ax.plot(*xyso[:,isplit,-1], 'bs')
# ax.plot(*xy_WNW, '.-')
# ax.plot(*xy_ENE, '.-')
# ax.plot(*xy_ESE, '.-')
# ax.plot(*xy_WSW, '.-')
# ax.plot(*xy_top, 'r.-')
#
# ax.plot(*xy_N, 'k-')
# ax.plot(*xy_N.transpose(0,2,1), 'k-')
#
# # ax.plot(*xy_top_match, 'r.-')
# # ax.plot(*xy_bot, 'b.-')
# # ax.plot(*xy_in_N, 'm.-')
# # ax.plot(*xy_out_N, 'm.-')
#
# # ax.plot(xlim, pitch2/2, 'k-')
# # ax.plot(xlim, -pitch2/2, 'k-')
# ax.axis('equal')
# plt.show()
#
