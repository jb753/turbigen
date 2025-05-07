"""Miscellaneous utility functions that don't fit anywhere else."""

from abc import ABC, abstractmethod
import numpy as np
import os
import inspect
import tarfile
import scipy.interpolate

from turbigen.exceptions import ConfigError
from scipy.integrate import cumulative_trapezoid as cumtrapz
from scipy.interpolate import griddata
import re

import logging

logging.ITER = 25
logging.raiseExceptions = True
logging.addLevelName(logging.ITER, "ITER")
logging.basicConfig(format="%(message)s")


def check_scalar(**kwargs):
    """Raise a helpful error message if any of the inputs are not scalar."""
    for k, v in kwargs.items():
        if not np.isscalar(v):
            raise ConfigError(f"{k}={v} is a vector, but expected a scalar.")


def check_vector(shape, **kwargs):
    """Raise a helpful error message if any inputs do not have specified shape."""
    for k, v in kwargs.items():
        shape_in = np.atleast_1d(v).shape
        if not shape_in == shape:
            raise ConfigError(f"{k}={v} has shape {shape_in}, but expected {shape}")


def node_to_cell(var):
    """For a (...,ni,nj,nk) matrix of some property, average over eight corners of
    each cell to produce an (...,ni-1,nj-1,nk-1) matrix of cell-centered properties."""
    return np.mean(
        np.stack(
            (
                var[..., :-1, :-1, :-1],  # i, j, k
                var[..., 1:, :-1, :-1],  # i+1, j, k
                var[..., :-1, 1:, :-1],  # i, j+1, k
                var[..., 1:, 1:, :-1],  # i+1, j+1, k
                var[..., :-1, :-1, 1:],  # i, j, k+1
                var[..., 1:, :-1, 1:],  # i+1, j, k+1
                var[..., :-1, 1:, 1:],  # i, j+1, k+1
                var[..., 1:, 1:, 1:],  # i+1, j+1, k+1
            ),
        ),
        axis=0,
    )


def cell_to_node(x):
    """One-dimensional centered values to nodal values."""
    return np.concatenate(
        (
            x[
                (0,),
            ],
            0.5 * (x[1:] + x[:-1]),
            x[
                (-1,),
            ],
        )
    )


def cluster_cosine(npts):
    """Cosinusoidal cluster on unit interval with a set number of points."""
    # Define a non-dimensional clustering function
    xc = 0.5 * (1.0 - np.cos(np.pi * np.linspace(0.0, 1.0, npts)))
    xc -= xc[0]
    xc /= xc[-1]
    return xc


def cumsum0(x, axis=None):
    """Cumulative summation including an inital zero, input same length as output."""
    return np.insert(np.cumsum(x, axis=axis), 0, 0.0, axis=axis)


def cumtrapz0(x, *args):
    """Cumulative integration including an inital zero, input same length as output."""
    return np.insert(cumtrapz(x, *args), 0, 0.0)


def arc_length(xr):
    """Arc length along second axis, assuming x/r on first axis"""
    dxr = np.diff(xr, n=1, axis=1) ** 2.0
    return np.sum(np.sqrt(np.sum(dxr, axis=0, keepdims=True)), axis=1).squeeze()


def cum_arc_length(xr, axis=1):
    """Cumulative arc length along a given axis, assuming x/r on first axis"""
    dxr = np.diff(xr, n=1, axis=axis) ** 2.0
    ds = np.sqrt(np.sum(dxr, axis=0, keepdims=True))
    s = cumsum0(ds, axis=axis)[0]
    return s


def tand(x):
    """Tangent of degree angle"""
    return np.tan(np.radians(x))


def atand(x):
    """Arctangent to degree angle"""
    return np.degrees(np.arctan(x))


def atan2d(y, x):
    """2D arctangent to degree angle"""
    return np.degrees(np.arctan2(y, x))


def cosd(x):
    """Cosine of degree angle"""
    return np.cos(np.radians(x))


def sind(x):
    """Sine of degree angle"""
    return np.sin(np.radians(x))


def tolist(x):
    if np.shape(x) == ():
        return [
            x,
        ]
    else:
        return x


def vecnorm(x):
    return np.sqrt(np.einsum("i...,i...", x, x))


def angles_to_velocities(V, Alpha, Beta):
    tanAl = tand(Alpha)
    tanBe = tand(Beta)
    tansqAl = tanAl**2.0
    tansqBe = tanBe**2.0
    Vm = V / np.sqrt(1.0 + tansqAl)
    Vx = V / np.sqrt((1.0 + tansqBe) * (1.0 + tansqAl))
    Vt = Vm * tanAl
    Vr = Vx * tanBe

    assert np.allclose(atan2d(Vt, Vm), Alpha)
    assert np.allclose(atan2d(Vr, Vx), Beta)

    return Vx, Vr, Vt


def resample_critical_indices(ni, ic, f):
    # Spans between each critical index
    dic = np.diff(ic)

    # Assemble segments between each critical point
    segs = []
    nseg = len(dic)
    for iseg in range(nseg):
        niseg = int(np.round(dic[iseg] * f).item() + 1)
        segs.append(np.round(np.linspace(ic[iseg], ic[iseg + 1], niseg)).astype(int))

    i = np.unique(np.concatenate(segs))
    assert np.all(np.isin(ic, i))
    return i


def resample(x, f, mult=None):
    """Multiply number of points in x by f, keeping relative spacings."""
    if np.isclose(f, 1.0):
        return x
    xnorm = (x - x[0]) / np.ptp(x)
    npts = len(x)
    npts_new = np.round((npts - 1) * f).astype(int) + 1
    if mult:
        npts_new = int(mult * np.ceil((npts_new - 1) / mult)) + 1
    inorm = np.linspace(0.0, 1.0, npts)
    inorm_new = np.linspace(0.0, 1.0, npts_new)
    xnorm_new = np.interp(inorm_new, inorm, xnorm)
    xnew = xnorm_new * np.ptp(x) + x[0]

    assert np.allclose(
        xnew[
            (0, -1),
        ],
        x[
            (0, -1),
        ],
    )

    return xnew


def zero_crossings(x):
    ind_up = np.where(np.logical_and(x[1:] > 0.0, x[:-1] < 0.0))[0] + 1
    ind_down = np.where(np.logical_and(x[1:] < 0.0, x[:-1] > 0.0))[0] + 1
    return ind_up, ind_down


def replace_nan(x, y, z, kind):
    xy = np.stack((x.reshape(-1), y.reshape(-1)), axis=1)
    zrow = z.reshape(-1)

    # Check for missing values
    is_nan = np.isnan(zrow)
    not_nan = np.logical_not(is_nan)
    if np.sum(is_nan):
        # Replace missing with nearest
        zrow[is_nan] = griddata(xy[not_nan], zrow[not_nan], xy[is_nan], method=kind)


def _match(x, y):
    if x is None and y is None:
        return True
    elif x is None and y is not None:
        return False
    elif x is not None and y is None:
        return False
    elif np.isclose(x, y).all():
        return True
    else:
        return False


def node_to_face(var):
    """For a (...,n,m) matrix of some property, average over the four corners of
    each face to produce an (...,n-1,m-1) matrix of face-centered properties."""
    return np.mean(
        np.stack(
            (
                var[..., :-1, :-1],
                var[..., 1:, 1:],
                var[..., :-1, 1:],
                var[..., 1:, :-1],
            ),
        ),
        axis=0,
    )


def make_logger():
    # Add a special logging level above INFO for iterations
    logger = logging.getLogger("turbigen")

    def _log_iter(message, *args, **kwargs):
        logger.log(logging.ITER, message, *args, **kwargs)

    logger.iter = _log_iter
    return logger


def interpolate_transfinite(c, plot=False):
    #         c3
    #     B--->---C
    #     |       |
    #  c2 ^       ^ c4   y
    #     |       |      ^
    #     A--->---D      |
    #         c1         +--> x

    if plot:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        labels = ["C1", "C2", "C3", "C4"]
        markers = ["x", "+", "^", "o"]
        for i, ci in enumerate(c):
            if ci is not None:
                ax.plot(*ci, color=f"C{i}")
                ax.plot(
                    *ci[:, (0,)],
                    markers[i],
                    color=f"C{i}",
                    label=f"{labels[i]},{ci.shape[1]}",
                )
        ax.legend()
        plt.savefig("beans.pdf")
        plt.show()

    # Check corners are coincident
    assert np.allclose(c[0][:, 0], c[1][:, 0])
    assert np.allclose(c[1][:, -1], c[2][:, 0])
    assert np.allclose(c[2][:, -1], c[3][:, -1])
    assert np.allclose(c[0][:, -1], c[3][:, 0])

    # Check lengths are the same
    ni = c[0].shape[1]
    nj = c[1].shape[1]
    assert c[2].shape[1] == ni
    assert c[3].shape[1] == nj

    # Calculate arc lengths
    s = [cum_arc_length(ci) for ci in c]
    # Normalise
    sn = [si / si[-1] for si in s]

    # Parameterise by the mean arc length of each pair of curves
    u = np.mean(np.stack((sn[0], sn[2])), axis=0).reshape(1, -1, 1)
    v = np.mean(np.stack((sn[1], sn[3])), axis=0).reshape(1, 1, -1)

    # For brevity
    u1 = 1.0 - u
    v1 = 1.0 - v
    A = c[0][:, None, None, 0]
    B = c[2][:, None, None, 0]
    C = c[2][:, None, None, -1]
    D = c[0][:, None, None, -1]

    c0 = c[0].reshape(2, -1, 1)
    c1 = c[1].reshape(2, 1, -1)
    c2 = c[2].reshape(2, -1, 1)
    c3 = c[3].reshape(2, 1, -1)

    return (
        v1 * c0
        + v * c2
        + u1 * c1
        + u * c3
        - (u1 * v1 * A + u * v * C + u * v1 * D + v * u1 * B)
    )


logger = make_logger()


def signed_distance(xrc, xr):
    """Distance above or below a straight line in meridional plane.

    Parameters
    ----------
    xrc: (2, 2)
        Coordinates of the cut plane.
    xr: (2,ni,nj,nk) array
        Meridional coordinates to cut.

    """

    dxrc = np.diff(xrc, axis=1)

    return dxrc[0] * (xrc[1, 0] - xr[1]) - (xrc[0, 0] - xr[0]) * dxrc[1]


def signed_distance_piecewise(xrc, xr):
    """Distance above or below a piecewise line in meridional plane.

    Note that this becomes increasingly inaccurate far away from the
    curve but the zero level is correct (which is sufficient for cutting).

    Parameters
    ----------
    xrc: (2, ns)
        Coordinates of the cut plane with ns segments.
    xr: (2,...) array
        Meridional coordinates to cut.

    Returns
    ------
    ds: (...) array
        Signed distance above or below the cut.

    """

    assert xrc.shape[0] == 2
    assert xrc.ndim == 2
    assert xr.shape[0] == 2

    # Preallocate the signed distance
    d = np.full(xr.shape[1:], np.inf)

    # Expand dimensions of cut line so it broadcasts
    add_dims = [i for i in range(2, xr.ndim + 1)]
    xrce = np.expand_dims(xrc, add_dims)

    # Dot product over the first axis
    def dot(a, b):
        return np.einsum("i...,i...", a, b)

    # Loop over line segments
    ni = xrc.shape[1]
    for i in range(ni - 1):
        # Calculate absolute distance field for this segment
        a = xr - xrce[:, i]  # Segment start to point
        b = xrce[:, i + 1] - xrce[:, i]  # Parallel to segment
        L = np.maximum(dot(b, b), 1e-9)
        h = np.clip(dot(a, b) / L, 0.0, 1.0)  # Distance along segment
        l = a - b * h  # Subtract parallel component to get perp distance
        di = np.sqrt(dot(l, l))  # Get length

        # Get the smallest absolute value
        ind = np.where(di < np.abs(d))

        # Make the distance signed
        c = np.array([-b[1], b[0]])  # Vector perp to segment
        di *= np.sign(dot(l, c))

        # Assign where we have a new smallest absolute distance
        d[ind] = di[ind]

    return d


def next_numbered_dir(basename):
    # Find the ids of existing directories
    base_dir, stem = os.path.split(basename)

    # Check the placeholder is there
    if "*" not in stem:
        raise Exception(
            f"Directory stem {stem} missing * placeholder for automatic numbering"
        )

    # Make a regular expression to extract the id from a dir name
    restr = stem.replace("*", r"(\d*)")
    re_id = re.compile(restr)

    cur_id = -1

    # Get all dirs matching placeholder using glob
    try:
        dirs = next(os.walk(base_dir))[1]
        for d in dirs:
            try:
                now_id = int(re_id.match(d).groups()[0])
                cur_id = np.maximum(now_id, cur_id)
            except (AttributeError, ValueError):
                continue
    except StopIteration:
        pass
    next_id = cur_id + 1
    return os.path.join(base_dir, stem.replace("*", f"{next_id:04d}"))


def node_to_face3(x):
    # x has shape [?,ni,nj,nk]
    # return averaged values on const i, const j, const k faces
    # xi [?,ni,nj-1, nk-1]
    # xj [?,ni-1,nj, nk-1]
    # xk [?,ni-1,nj-1, nk]

    xi = np.stack(
        (
            x[..., :, :-1, :-1],
            x[..., :, 1:, :-1],
            x[..., :, 1:, 1:],
            x[..., :, :-1, 1:],
        ),
    ).mean(axis=0)

    xj = np.stack(
        (
            x[..., :-1, :, :-1],
            x[..., 1:, :, :-1],
            x[..., 1:, :, 1:],
            x[..., :-1, :, 1:],
        ),
    ).mean(axis=0)

    xk = np.stack(
        (
            x[..., :-1, :-1, :],
            x[..., 1:, :-1, :],
            x[..., 1:, 1:, :],
            x[..., :-1, 1:, :],
        ),
    ).mean(axis=0)

    return xi, xj, xk


def incidence_unstructured(grid, machine, ml, irow, spf, plot=False):
    # Pull out 2D cuts of blades and splitters
    surfs = grid.cut_blade_surfs()[irow]

    nspf = len(spf)

    # Meridional curves for target span fractions
    ist = irow * 2 + 1
    ien = ist + 1
    m = np.linspace(ist, ien, 101)
    xr_spf = machine.ann.evaluate_xr(m.reshape(-1, 1), spf.reshape(1, -1)).reshape(
        2, -1, nspf
    )

    # Meridional velocity vector at inlet to this row
    Vxrt = ml[irow * 2].Vxrt_rel

    # Loop over main/splitter
    chi = []
    for jbld, surfj in enumerate(surfs):
        surf = surfj.squeeze()

        # Get the current blade object
        bldnow = machine.bld[irow][jbld]

        # Loop over span fractions
        # Unstructure cut through current surface along the
        # target span fraction curves
        xrt_stag = np.zeros((3, nspf))
        xrt_nose = np.zeros((3, nspf))
        xrt_cent = np.zeros((3, nspf))
        for k in range(len(spf)):
            # Cut at this span fraction
            C = surf[..., None].meridional_slice(xr_spf[:, :, k])

            # Stag point coordinates
            xrt_stag[:, k] = C.xrt_stag.squeeze()

            # Geometric nose coordinates
            xrt_nose[:, k] = bldnow.get_nose(spf[k])

            # Leading edge centre
            xrt_cent[:, k] = bldnow.get_LE_cent(spf[k], 5.0)

        # Calculate the angles
        chi_metal = yaw_from_xrt(xrt_nose, xrt_cent, Vxrt)
        chi_flow = yaw_from_xrt(xrt_stag, xrt_cent, Vxrt, yaw_ref=chi_metal)

        chi.append(np.stack((chi_metal, chi_flow)))

    return chi


def stagnation_point_angle(grid, machine, meanline, fac_Rle=1.0):
    surfs = grid.cut_blade_surfs()

    chi_stag = []

    # Loop over rows
    for irow, surfi in enumerate(surfs):
        chi_stag.append([])

        if surfi is None:
            continue

        # Loop over main/splitter
        for jbld, surfj in enumerate(surfi):
            surf = surfj.squeeze()
            _, nj = surf.shape

            istag_mean = np.round(np.nanmean(surf.i_stag)).astype(int)
            spf = np.array([surf.spf[istag_mean, j] for j in range(nj)])

            # spf_mesh = [surf.spf[surf.i_stag[j], j] for j in range(nj)]

            # Get coordinates of stagnation point
            xrt_stag = surf.xrt_stag

            # Set up a conversion from mesh spf to blade spf at LE
            bldnow = machine.split[irow] if jbld else machine.bld[irow]

            # import matplotlib.pyplot as plt
            # fig, ax = plt.subplots()
            # fig2, ax2 = plt.subplots()
            # for spf_target in [0.,0.5,1.]:
            #     jplot = np.argmin(np.abs(spf_mesh-spf_target))
            #     spf_plot = spf_mesh[jplot]
            #     print(spf_plot)
            #     ax.plot(*surf[:,jplot].yz,'k-')
            #     xrtsect = np.stack(bldnow.evaluate_section(spf_plot-0.01), axis=0)
            #     ysect = xrtsect[:,1,:]*np.sin(xrtsect[:,2,:])
            #     zsect = xrtsect[:,1,:]*np.cos(xrtsect[:,2,:])
            #     ax.plot(ysect[0], zsect[0],'-')
            #     ax.plot(ysect[1], zsect[1],'-')
            #     ax.axis('equal')
            # ax2.plot(*surf[:,jplot].xr,'k-')
            # ax2.plot(*xrtsect[0][:2],'-')
            # ax2.plot(*xrtsect[1][:2],'-')
            # ax2.axis('equal')
            # plt.show()
            # # quit()
            # fig, ax = plt.subplots()
            # ax.plot(*xrt_stag[:2],'-x')
            # ax.plot(*xrtLE_blade[:2],'-+')
            # fig, ax = plt.subplots()
            # ax.plot(*xrt_stag[(0,2),],'-x')
            # ax.plot(*xrtLE_blade[(0,2),],'-+')
            # plt.show()
            # quit()

            # Get coordinates of LE center
            xrt_cent = np.stack(
                [bldnow.get_LE_cent(spf[j], fac_Rle).squeeze() for j in range(nj)],
                axis=-1,
            )

            xrt_nose = np.stack(
                [bldnow.get_nose(spf[j]).squeeze() for j in range(nj)],
                axis=-1,
            )

            # Get vector between stagnation point and centre of LE
            dxrt = xrt_cent - xrt_stag

            # Get vector between nose and centre of LE
            dxrt_nose = xrt_cent - xrt_nose

            # Multiply theta component by reference radius
            dxrrt = dxrt.copy()
            dxrrt_nose = dxrt_nose.copy()
            rref = 0.5 * (xrt_cent + xrt_stag)[1]
            dxrrt[2] *= rref
            dxrrt_nose[2] *= rref

            # Calculate angle
            denom = np.sqrt(dxrrt[0] ** 2 + dxrrt[1] ** 2)
            chi_stag_now = np.degrees(np.arctan2(dxrrt[2], denom))
            denom_nose = np.sqrt(dxrrt_nose[0] ** 2 + dxrrt_nose[1] ** 2)
            chi_metal_now = np.degrees(np.arctan2(dxrrt_nose[2], denom_nose))

            chi_stag[-1].append(np.stack((spf, chi_stag_now, chi_metal_now)))

    return chi_stag


def yaw_from_xrt(xrt1, xrt2, Vxrt, yaw_ref=None):
    # Vector between the points
    dxrt = xrt2 - xrt1

    # Midpoint radius
    rmid = 0.5 * (xrt1[1] + xrt2[1])

    # Distances in each direction
    dist_merid = vecnorm(dxrt[:2])
    dist_theta = rmid * dxrt[2]

    # As of now, dist_merid is always positive, which is not what we want
    # So if the meridional component is going against flow, switch the sign
    dist_merid *= np.sign(np.sum(np.reshape(Vxrt[:2], (2, 1)) * dxrt[:2]))

    # Trigonometry
    yaw = np.degrees(np.arctan2(dist_theta, dist_merid))

    # Out of arctan2, yaw is always -180 to 180
    # But we need to wrap with respect to the reference angle
    if yaw_ref is not None:
        # Calculate angle relative to the wrap angle
        yaw_rel = yaw - yaw_ref
        yaw_rel[yaw_rel < 180.0] += 360.0
        yaw_rel[yaw_rel > 180.0] -= 360.0
        yaw = yaw_rel + yaw_ref

    return yaw


def qinv(x, q):
    xs = np.sort(x)
    n = len(x)
    irel = np.linspace(0.0, n - 1, n) / (n - 1)
    return np.interp(q, irel, xs)


def clipped_levels(x, dx=None, thresh=0.001):
    xmin = qinv(x, thresh)
    xmax = qinv(x, 1.0 - thresh)
    if dx:
        xmin = np.floor(xmin / dx) * dx
        xmax = np.ceil(xmax / dx) * dx
        xlev = np.arange(xmin, xmax + dx, dx)
    else:
        xlev = np.linspace(xmin, xmax, 20)

    return xlev


def get_mp_from_xr(grid, machine, irow, spf, mlim):
    # Start by choosing a j-index to plot along
    jspf = grid.spf_index(spf)

    xr_row = machine.ann.xr_row(irow)

    surf = grid.cut_blade_surfs()[irow][0].squeeze()
    spf_blade = surf.spf[:, jspf]
    spf_actual = spf_blade[surf.i_stag[jspf]]

    # We want to plot along a general meridional surface
    # So brute force a mapping from x/r to meridional distance

    # Evaluate xr as a function of meridonal distance using machine geometry
    m_ref = np.linspace(*mlim, 5000)
    xr_ref = xr_row(spf_actual, m_ref)

    # Calculate normalised meridional distance (angles are angles)
    dxr = np.diff(xr_ref, n=1, axis=1)
    dm = np.sqrt(np.sum(dxr**2.0, axis=0))
    rc = 0.5 * (xr_ref[1, 1:] + xr_ref[1, :-1])
    mp_ref = cumsum0(dm / rc)
    assert (np.diff(mp_ref) > 0.0).all()

    # Calculate location of stacking axis
    mp_stack = np.interp(machine.bld[irow].mstack, m_ref, mp_ref)

    def mp_from_xr(xr):
        func = scipy.interpolate.NearestNDInterpolator(xr_ref.T, mp_ref)
        xru = xr.reshape(2, -1)
        mpu = func(xru.T) - mp_stack
        return mpu.reshape(xr.shape[1:])

    return mp_from_xr, spf_actual


def dA_Gauss(A, B, C, D):
    # Assemble all vertices together (stack along second axis)
    # xrrt[4, 3, ni, nj, nk]
    xrrt = np.stack((A, B, C, D), axis=0).copy()

    # Shift theta origin to face center
    # This is important so that constant-theta faces have no radial area
    t = xrrt[:, 2] / xrrt[:, 1]
    t -= t.mean(axis=0)
    xrrt[:, 2] = xrrt[:, 1] * t

    # Subtract face-center coords to reduce round-off error
    xrrtc = xrrt.mean(axis=0)
    xrrt -= xrrtc

    # Circular array of vertices
    v = np.concatenate((xrrt, xrrt[0][None, ...]), axis=0)

    # Edges
    dv = np.diff(v, axis=0)

    # Edge midpoint vertices
    vm = 0.5 * (v[:-1] + v[1:])

    # Vector field
    Fx = vm.copy()
    Fr = vm.copy()
    Ft = vm.copy()
    Fx[:, 0, :, :, :] = 0.0
    Fr[:, 1, :, :, :] = 0.0
    Ft[:, 2, :, :, :] = 0.0
    F = np.stack((Fx, Fr, Ft))

    # Edge normals
    dlx = np.stack(
        (
            dv[:, 0, :, :, :],
            -dv[:, 2, :, :, :],
            dv[:, 1, :, :, :],
        ),
        axis=1,
    )
    dlr = np.stack(
        (
            dv[:, 2, :, :, :],
            dv[:, 1, :, :, :],
            -dv[:, 0, :, :, :],
        ),
        axis=1,
    )
    dlt = np.stack(
        (
            -dv[:, 1, :, :, :],
            dv[:, 0, :, :, :],
            dv[:, 2, :, :, :],
        ),
        axis=1,
    )
    dl = np.stack((dlx, dlr, dlt))

    # Apply Gauss' theorem for area
    dA = 0.5 * np.sum(F * dl, axis=(2, 1))

    return dA


def moving_average_1d(arr, window_size):
    if window_size < 1:
        raise ValueError("Window size must be at least 1")
    if window_size % 2 == 0:
        raise ValueError("Window size must be odd to preserve shape")

    kernel = np.ones(window_size) / window_size
    return np.convolve(arr, kernel, mode="same")


def relax(x_old, x_new, rf):
    return x_new * rf + x_old * (1.0 - rf)


def interp1d_linear_extrap(x, y, axis=0):
    """Extend the default scipy interp1d with linear end extrapolation."""

    N = len(x)

    if N == 1:
        # Define a function that returns only point
        def spline(xq):
            return np.take(y.copy(), 0, axis=axis)

    elif N == 2:
        spline = scipy.interpolate.interp1d(
            x, y, fill_value="extrapolate", axis=axis, kind="linear"
        )

    else:
        spline = scipy.interpolate.CubicSpline(x, y, axis=axis, bc_type="natural")

        # determine the slope at the left edge
        leftx = np.atleast_1d(spline.x[0])
        lefty = spline(leftx)
        leftslope = spline(leftx, nu=1)

        # add a new breakpoint just to the left and use the
        # known slope to construct the PPoly coefficients.
        leftxnext = np.nextafter(leftx, leftx - 1)
        leftynext = lefty + leftslope * (leftxnext - leftx)
        Z = np.zeros_like(leftslope)
        leftcoeffs = np.expand_dims(
            np.concatenate([Z, Z, leftslope, leftynext], axis=0), 1
        )
        spline.extend(leftcoeffs, leftxnext)

        # repeat with additional knots to the right
        rightx = np.atleast_1d(spline.x[-1])
        righty = spline(rightx)
        rightslope = spline(rightx, nu=1)
        rightxnext = np.nextafter(rightx, rightx + 1)
        rightynext = righty + rightslope * (rightxnext - rightx)
        rightcoeffs = np.expand_dims(
            np.concatenate([Z, Z, rightslope, rightynext]), axis=1
        )
        spline.extend(rightcoeffs, rightxnext)

    return spline


def fit_plane(xyz):
    """Find the normal vector of a flat surface fitted to the input points"""

    # Center the curve around the origin
    xyz = xyz - np.mean(xyz, axis=1, keepdims=True)

    # Compute the covariance matrix of the centered points
    covariance_matrix = np.cov(xyz)

    # Compute the eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    # The normal vector is the eigenvector corresponding to the smallest eigenvalue
    normal_vector = eigenvectors[:, np.argmin(eigenvalues)]

    return normal_vector


def basis_from_normal(normal):
    # Find two vectors orthogonal to the normal to form a basis for the plane
    if np.allclose(normal, np.array([1, 0, 0])) or np.allclose(
        normal, np.array([-1, 0, 0])
    ):
        basis1 = np.array([0, 1, 0])
    else:
        basis1 = np.cross(normal, np.array([1, 0, 0]))
        basis1 /= np.linalg.norm(basis1)

    basis2 = np.cross(normal, basis1)

    return basis1, basis2


def dot(a, b):
    # Dot product over the first axis
    return np.einsum("i...,i...", a, b)


def project_onto_plane(points, basis1, basis2):
    # Center the curve around the origin
    xyz_mean = np.mean(points, axis=1, keepdims=True)
    points = points - xyz_mean

    # Dot product over the first axis
    def dot(a, b):
        return np.einsum("i...,i...", a, b)

    # Project the points onto the plane and express in the plane's basis
    projected_points = np.stack((dot(points, basis1), dot(points, basis2)))

    return projected_points


def shoelace_formula(xy):
    x, y = xy
    return 0.5 * np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])


def asscalar(x):
    if isinstance(x, np.ndarray):
        return x.item()
    elif isinstance(x, (np.float32, np.float64, float)):
        return x
    else:
        raise NotImplementedError()


def save_source_tar_gz(output_filename):
    """Creates a tar.gz archive containing all Python source files"""

    # Set directory to the package location
    directory = os.path.dirname(os.path.abspath(__file__))

    logger.debug(f"Saving source code backup to {output_filename}")
    with tarfile.open(output_filename, "w:gz") as tar:
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(".py") or file.endswith(
                    ".toml"
                ):  # Only include Python source files
                    file_path = os.path.join(root, file)
                    logger.debug(f"{file_path}")
                    tar.add(file_path, arcname=os.path.relpath(file_path, directory))


def camel_to_snake(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def get_subclass_by_name(cls, name):
    """Match a subclass by name."""

    # Loop over all subclasses of the abstract base class
    for subclass in cls.__subclasses__():
        # Check if the subclass name matches the input name
        subname = camel_to_snake(subclass.__name__)
        if subname == name or subname.replace("_", "") == name:
            return subclass

    # If no subclass matches the input name, raise an error
    # and list the available subclasses
    error_message = f"No {cls.__name__} named {name}.\n"
    error_message += "Available subclasses are:"
    for subclass in cls.__subclasses__():
        error_message += f"\n{camel_to_snake(subclass.__name__)}"
    raise ValueError(error_message)


def init_subclass_by_signature(cls, kwargs):
    """Automatically select a subclass by matching a signature."""

    # Loop over all subclasses of the abstract base class
    for subclass in cls.__subclasses__():
        # Check if the subclass signature matches the input arguments
        try:
            return subclass(**kwargs)
        except TypeError:
            continue

    # If no subclass matches the input arguments, raise an error
    # and list the available subclasses and their signatures
    error_message = f"No subclass of {cls.__name__} matches the input arguments.\n"
    error_message += str(kwargs) + "\n"
    error_message += "Available subclasses are:"
    for subclass in cls.__subclasses__():
        error_message += (
            f"\n{subclass.__name__}({inspect.signature(subclass.__init__)})"
        )

    raise ValueError(error_message)


class BaseDesigner(ABC):
    """A general class for storing and serialising design varaiables."""

    _supplied_design_vars = ()

    def __init__(self, design_vars):
        """Initialise by saving the design variables dict."""
        self.design_vars = design_vars
        self.check_design_vars()
        # Make any vector design variables into arrays
        for var in self.design_vars:
            if isinstance(self.design_vars[var], (tuple, list)):
                self.design_vars[var] = np.array(self.design_vars[var])

    def to_dict(self):
        """Convert the designer to a dictionary."""
        dvars = {
            k: v.tolist() if isinstance(v, np.ndarray) else v
            for k, v in self.design_vars.items()
        }
        return {
            "type": camel_to_snake(self.__class__.__name__),
            **dvars,
        }

    def check_design_vars(self):
        """Verify that the input design variables match the forward signature.

        We do it this way so the user only has to touch the forward and backward
        methods to implement a new designer. Instead of a new init
        method or defining their design variables as dataclass attributes."""

        # Get the signature of the forward method
        forward_sig = inspect.signature(self.forward)

        # Check for any design variables that are not in the forward signature
        valid_vars = [
            v
            for v in list(forward_sig.parameters.keys())
            if v not in self._supplied_design_vars
        ]
        for var in self.design_vars:
            if var not in valid_vars:
                raise ValueError(
                    f"Design variable '{var}' invalid, expected one of {valid_vars}"
                )

        # Check for any forward method parameters that are not in the design variables
        func_params = list(forward_sig.parameters.values())
        for param in func_params:
            # Ignore the design variable that is the inlet stagnation state
            if str(param) in self._supplied_design_vars:
                continue
            if str(param) not in self.design_vars and param.default is param.empty:
                raise ValueError(f"Required design variable '{param}' not supplied.")

    @staticmethod
    @abstractmethod
    def forward(*args, **kwargs):
        raise NotImplementedError


def format_sf(x, sig=3):
    return f"{x:.{sig}g}"


def format_array(x, precision=3):
    return "[" + ", ".join(format_sf(xi, precision) for xi in x) + "]"
