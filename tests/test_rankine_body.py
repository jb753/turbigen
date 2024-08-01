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

# Define bounding box
pitch = 1.
xlim = [-1.0, 2.0]

# Get blade surface coordinates for each side
R = 0.05
mmax = 0.5
stag = 0.
nchord = 65
m = util.cluster_cosine(nchord)
xyu, xyl = blade(m, R, mmax, stag)

cU = m2d.Curve(*xyu)
cL = m2d.Curve(*xyl)
cS = m2d.Curve.from_join(cU, cL).roll(cU.n-1)


nk_omesh = 17
dw = 0.005
dinf = 0.05
ER = 1.2
L_omesh = 0.16
d_omesh = turbigen.clusterfunc.single.fixed(dw, dinf, ER, nk_omesh, x0=0., x1=L_omesh)

d1 = d_omesh[-1]-d_omesh[-2]
d2 = d1*4
bO = m2d.Block.from_offset(cS, d_omesh)

CO = bO[:,-1]
angles = [-135, 135, 45, -45]
curves = CO.split_by_angle(angles)
Cin = curves[1].project_to_x(xlim[0])
Cout = curves[-1].project_to_x(xlim[1])


fig, ax = plt.subplots()
ax.axis('equal')
ax.plot(*bO.xy, 'k-')
ax.plot(*bO.T.xy, 'k-')
for c in curves:
    ax.plot(*c.xy, '-o')
ax.plot(*Cin.xy, 'b-x')
ax.plot(*Cout.xy, 'b-x')
plt.show()
quit()


# # Find split points
# ang = turbigen.util.angle_curve(xyso[:,:,-1])
# isplit = [np.argmin(np.abs(ang - si)) for si in [45, -45, 135, -135]]
# displit = np.diff(isplit)
# print(displit)

# ni1 = 25
# xy_WNW = extend_x_from_point(xyso[:, isplit[0], -1], xlim[0], d1, d2, ni1)
# xy_ENE = extend_x_from_point(xyso[:, isplit[1], -1], xlim[1], d1, d2, ni1)
# xy_ESE = extend_x_from_point(xyso[:, isplit[2], -1], xlim[0], d1, d2, ni1)
# xy_WSW = extend_x_from_point(xyso[:, isplit[3], -1], xlim[1], d1, d2, ni1)

# ni_per = 2*ni1 + displit[0] - 1
# xy12_top = np.array((xlim, (pitch/2, pitch/2)))

# xy_bot = xy_top.copy()
# xy_bot[1] *= -1.

# xy_top_match = np.concatenate(
#     (
#         xy_WNW[:,:-1],
#         xyso[:,isplit[0]:(isplit[1]+1),-1],
#         xy_ENE[:,1:]
#     ),
#     axis=1
# )
# s_top = turbigen.util.cum_arc_length(xy_top_match)
# s_top /= s_top[-1]
# s_top = turbigen.util.smooth_1d(s_top, 1.0, 100)

xy_top = turbigen.util.interpolate_curve_1d(xy12_top, s_top)

nk1 = 9

xy12_in_N = np.stack((xy_top_match[:,0],xy_top[:,0]),axis=1)
xy_in_N = turbigen.util.interpolate_curve_1d(xy12_in_N, np.linspace(0., 1., nk1))
xy12_out_N = np.stack((xy_top_match[:,-1],xy_top[:,-1]),axis=1)
xy_out_N = turbigen.util.interpolate_curve_1d(xy12_out_N, np.linspace(0., 1., nk1))


print(xy_top.shape)
print(xy_top_match.shape)
xy_N = turbigen.util.interpolate_transfinite(
        (
            xy_top_match,
            xy_in_N,
            xy_top,
            xy_out_N,
        )
)

# xy_in_W = turbigen.util.interpolate_curve_1d(
# print(isplit)
# fig, ax = plt.subplots()
# ax.plot(ang)
# ax.plot(np.arange(len(ang))[isplit],ang[isplit],'kx')
# plt.show()

fig, ax = plt.subplots()
# ax.plot(*xyu, '-x')
print(xyso.shape)
ax.plot(*xyso, 'k-')
ax.plot(*xyso.transpose(0,2,1), 'k-')
ax.plot(*xyso[:,isplit,-1], 'bs')
ax.plot(*xy_WNW, '.-')
ax.plot(*xy_ENE, '.-')
ax.plot(*xy_ESE, '.-')
ax.plot(*xy_WSW, '.-')
ax.plot(*xy_top, 'r.-')

ax.plot(*xy_N, 'k-')
ax.plot(*xy_N.transpose(0,2,1), 'k-')

# ax.plot(*xy_top_match, 'r.-')
# ax.plot(*xy_bot, 'b.-')
# ax.plot(*xy_in_N, 'm.-')
# ax.plot(*xy_out_N, 'm.-')

# ax.plot(xlim, pitch2/2, 'k-')
# ax.plot(xlim, -pitch2/2, 'k-')
ax.axis('equal')
plt.show()

