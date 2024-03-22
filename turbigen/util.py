"""Miscellaneous utility functions that don't fit anywhere else."""
# from . import compflow as cf
import numpy as np
import gzip
import os
import sys
import importlib
import scipy.interpolate
from scipy.integrate import cumulative_trapezoid as cumtrapz
from scipy.optimize import minimize, leastsq, root_scalar
from scipy.spatial import Voronoi, ConvexHull
from scipy.signal import medfilt
from scipy.interpolate import griddata
import yaml
import re

import logging

logging.ITER = 25
logging.raiseExceptions = True
logging.addLevelName(logging.ITER, "ITER")
logging.basicConfig(format="%(message)s")


# Logic to allow dumping of numpy float64
def represent_float(dumper, data):
    return dumper.represent_scalar("tag:yaml.org,2002:float", str(data))


yaml.representer.SafeRepresenter.add_representer(np.float64, represent_float)


# Logic to allow dumping int to yaml
def represent_int(dumper, data):
    return dumper.represent_scalar("tag:yaml.org,2002:int", str(data))


yaml.representer.SafeRepresenter.add_representer(np.int64, represent_int)


def medial_axis(xy, plot=False):
    """ "Find medial axis using Voronoi triangulation.

    Parameters
    ----------
    xy: (2, N) array
        Plane coordinates for loop of upper and lower surfaces"""

    vor = Voronoi(xy.T)

    # Form medial axis by discarding Voronoi vertices outside section
    # https://stackoverflow.com/a/69485899
    hull = ConvexHull(xy.T)
    A, b = hull.equations[:, :-1], hull.equations[:, -1:]
    tol = xy[0].ptp() * 1e-3
    xy_med = vor.vertices

    xy_med = xy_med[np.all(np.matmul(xy_med, A.T) + b.T < -tol, axis=1)].T

    # Arrange coordinates in order of increasing meridional coodinate
    xy_med = xy_med[:, np.argsort(xy_med[0])]

    # if plot:
    #     from IPython import embed; embed()
    #     import matplotlib.pyplot as plt
    #     fig, ax = plt.subplots()
    #     ax.plot(*xy, "kx")
    #     ax.plot(*xy_med, "ro")
    #     ax.axis("equal")
    #     ax.set_xlim((0.,0.02))
    #     ax.set_ylim((0.7,0.8))
    #     plt.savefig("test2.pdf")
    #     quit()

    # Remove outliers with median filter
    xy_med = xy_med[:, np.argsort(xy_med[0])]
    if xy_med.shape[1] == 0:
        raise ValueError("Failed to find medial axis")
    xy_med[1] = medfilt(xy_med[1])

    return xy_med


# def boundary_layer_Po_Poinf(spf, delta, Mainf, ga, verbose=False):
#     """Return stagnation pressure ratio in a boundary layer."""

#     expon = 1.0 / 7.0
#     d99 = delta * 0.99

#     # Specify velocity
#     V_Vinf = np.ones_like(spf)
#     V_Vinf[spf < d99] = (spf[spf < d99] / delta) ** expon
#     V_Vinf[spf > (1.0 - d99)] = ((1.0 - spf[spf > (1.0 - d99)]) / delta) ** expon

#     # Evaluate thicknesses and shape factor
#     del_star = np.trapz(1.0 - V_Vinf[spf < d99], spf[spf < d99])
#     theta = np.trapz((1.0 - V_Vinf[spf < d99]) * V_Vinf[spf < d99], spf[spf < d99])
#     H = del_star / theta

#     if verbose:
#         print("H", H)
#         print("del_star", del_star)
#         print("theta", theta)

#     # Get Mach number
#     V_cpTo_inf = cf.V_cpTo_from_Ma(Mainf, ga)
#     Po_Pinf = cf.Po_P_from_Ma(Mainf, ga)
#     V_cpTo = V_cpTo_inf * V_Vinf
#     Ma = cf.Ma_from_V_cpTo(V_cpTo, ga)

#     # Transform to stagnation pressure ratio
#     Po_P = cf.Po_P_from_Ma(Ma, ga)
#     Po_Poinf = Po_P / Po_Pinf

#     return Po_Poinf


def reduce_scalar(x):
    """Convert something to a scalar float if possible."""
    if np.shape(x) in ((), (1,)):
        return x.item()
    else:
        return x


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


class DiscontinuousBPoly:
    def __init__(self, x, yl=None, yr=None, p=None):
        """Initialise a discontinous piecewise Bernstein polynomial.

        Parameters
        ----------
        x: (n,) array
            Sample points.
        yl: (n-1,m) nested list
            yl[i][j] is the j-th derivative to the left of x[i+1].
        yr: (n-1,m) nested list
            yl[i][j] is the j-th derivative to the right of x[i].
        """

        self.x = x.reshape(-1)
        self.n = len(self.x)

        if p:
            self._p = p

        else:
            self._p = []
            for i in range(self.n - 1):
                xnow = x[i : i + 2]
                ynow = np.stack((yr[i], yl[i]))
                self._p.append(scipy.interpolate.BPoly.from_derivatives(xnow, ynow))

    def derivative(self, nu=1):
        """Take derivative and return a new object."""
        return DiscontinuousBPoly(self.x, p=[p.derivative(nu) for p in self._p])

    def __call__(self, x):
        """Evaluate the piecewise polynomial."""
        # Assign the input points to a polynomial bin
        ind = np.digitize(x, self.x) - 1

        # Evalute each polynomial
        y = np.empty_like(x)
        for i in range(self.n - 1):
            y[ind == i] = self._p[i](x[ind == i])

        # Special case for last bin end
        y[x == self.x[-1]] = self._p[-1](x[x == self.x[-1]])

        return y


def cluster_cosine(npts):
    """Return a cosinusoidal clustering function with a set number of points."""
    # Define a non-dimensional clustering function
    return 0.5 * (1.0 - np.cos(np.pi * np.linspace(0.0, 1.0, npts)))


def _invert_sinhx_x(y):
    """Return solution x for y = sinh(x)/x in Eqns. (62-67)."""

    y1 = y - 1.0
    x_low = np.sqrt(6.0 * y1) * (
        1.0
        - 0.15 * y1
        + 0.057321429 * y1**2.0
        - 0.024907295 * y1**3.0
        + 0.0077424461 * y1**4.0
        - 0.0010794123 * y1**5.0
    )

    v = np.log(y)
    w = 1.0 / y - 0.028527431
    x_high = (
        v
        + (1.0 + 1.0 / v) * np.log(2.0 * v)
        - 0.02041793
        + 0.24902722 * w
        + 1.9496443 * w**2.0
        - 2.6294547 * w**3.0
        + 8.56795911 * w**4.0
    )

    return np.where(y < 2.7829681, x_low, x_high)


def _invert_sinx_x(y):
    """Return solution x for y = sin(x)/x from Eqns. (68-71)."""

    x_low = np.pi * (
        1.0
        - y
        + y**2.0
        - (1.0 + np.pi**2.0 / 6.0) * y**3.0
        + 6.794732 * y**4.0
        - 13.205501 * y**5.0
        + 11.726095 * y**6.0
    )

    y1 = 1.0 - y
    x_high = np.sqrt(6.0 * y1) * (
        1.0
        + 0.15 * y1
        + 0.057321429 * y1**2.0
        + 0.048774238 * y1**3.0
        - 0.053337753 * y1**4.0
        + 0.075845134 * y1**5.0
    )

    return np.where(y < 0.26938972, x_low, x_high)


def cluster_two_sided_slope(npts, s0, s1):
    """Two sided analytic clustering function after Vinokur."""

    A = np.sqrt(s0 * s1)
    B = np.sqrt(s0 / s1)

    xi = np.linspace(0.0, 1.0, npts)

    if np.abs(B - 1.0) < 0.001:
        # Eqn. (52)
        u = xi * (1.0 + 2.0 * (B - 1.0) * (xi - 0.5) * (1.0 - xi))
    elif B < 1.0:
        # Solve Eqn. (49)
        Dx = _invert_sinx_x(B)
        assert np.isclose(np.sin(Dx) / Dx, B, rtol=1e-1)
        # Eqn. (50)
        u = 0.5 * (1.0 + np.tan(Dx * (xi - 0.5)) / np.tan(Dx / 2.0))
    elif B >= 1.0:
        # Solve Eqn. (46)
        Dy = _invert_sinhx_x(B)
        assert np.isclose(np.sinh(Dy) / Dy, B, rtol=1e-1)
        # Eqn. (47)
        u = 0.5 * (1.0 + np.tanh(Dy * (xi - 0.5)) / np.tanh(Dy / 2.0))
    else:
        breakpoint()
        raise Exception(f"Unexpected B={B}, s0={s0}, s1={s1}")

    t = u / (A + (1.0 - A) * u)

    return t


def cluster_two_sided_step(npts, dt0, dt1):
    """Two-sided clustering by cell width."""

    def _iter(s):
        t = cluster_two_sided_slope(npts, *(10.0**s))
        err_le = np.abs((t[1] - t[0]) / dt0 - 1.0)
        err_te = np.abs((t[-1] - t[-2]) / dt1 - 1.0)
        return err_le + err_te

    dunif = 1.0 / npts
    s0 = dunif / dt0
    s1 = dt1 / dunif
    x0 = np.log10([s0, s1])
    bnd = ((-6, 6),) * 2
    res = minimize(_iter, x0, bounds=bnd, tol=1e-9)
    s_opt = 10.0 ** (res.x)
    t = cluster_two_sided_slope(npts, *s_opt)
    err_dt0 = np.abs((t[1] - t[0]) / dt0 - 1.0)
    err_dt1 = np.abs((t[-1] - t[-2]) / dt1 - 1.0)
    if err_dt0 > 0.1 or err_dt1 > 0.1:
        raise Exception(
            f"Clustering failed, npts={npts}, dt0={dt0}, dt1={dt1},"
            f"actual_dt0={t[1]-t[0]}, , actual_dt1={t[-1]-t[-2]}, "
        )
    return cluster_two_sided_slope(npts, *s_opt)


def cumsum0(x, axis=None):
    return np.insert(np.cumsum(x, axis=axis), 0, 0.0, axis=axis)


def cumtrapz0(x, *args):
    return np.insert(cumtrapz(x, *args), 0, 0.0)


def arc_length(xr):
    dxr = np.diff(xr, n=1, axis=1) ** 2.0
    return np.sum(np.sqrt(np.sum(dxr, axis=0)))


def cum_arc_length(xr, axis=1):
    dxr = np.diff(xr, n=1, axis=axis) ** 2.0
    ds = np.sqrt(np.sum(dxr, axis=0, keepdims=True))
    s = cumsum0(ds, axis=axis)[0]
    return s


def tand(x):
    return np.tan(np.radians(x))


def atand(x):
    return np.degrees(np.arctan(x))


def atan2d(y, x):
    return np.degrees(np.arctan2(y, x))


def cosd(x):
    return np.cos(np.radians(x))


def sind(x):
    return np.sin(np.radians(x))


def tolist(x):
    if np.shape(x) == ():
        return [
            x,
        ]
    else:
        return x


class UniqueKeyLoader(yaml.SafeLoader):
    def construct_mapping(self, node, deep=False):
        mapping = []
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in mapping:
                raise Exception(f'Config: duplicate key "{key}"')
            mapping.append(key)
        return super().construct_mapping(node, deep)


def read_yaml(fname):
    """Read a dictionary from file."""

    # Patch YAML loader to get scientific notation correct
    loader = UniqueKeyLoader
    loader.add_implicit_resolver(
        "tag:yaml.org,2002:float",
        re.compile(
            """^(?:
        [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$""",
            re.X,
        ),
        list("-+0123456789."),
    )

    # Read the YAML
    with open(fname, "r") as f:
        config = yaml.load(f, Loader=loader)

    return config


def read_yaml_list(fname):
    """Read a list of dictionaries from YAML file."""
    # Patch YAML loader to get scientific notation correct
    loader = UniqueKeyLoader
    loader.add_implicit_resolver(
        "tag:yaml.org,2002:float",
        re.compile(
            """^(?:
        [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$""",
            re.X,
        ),
        list("-+0123456789."),
    )
    # Read the YAML
    with open(fname, "r") as f:
        config = list(yaml.load_all(f, Loader=loader))

    return config


def write_yaml(d, fname, mode="w"):
    """Write a dictionary to file."""
    with open(fname, mode) as f:
        yaml.safe_dump(d, f, explicit_start=True, explicit_end=True)


def write_yaml_compressed(d, fname):
    """Write a dictionary to compressed file."""
    with gzip.open(fname, "wt") as f:
        yaml.safe_dump(d, f, explicit_start=True, explicit_end=True)


def list_of_dict_to_dict_of_list(ldict):
    # Get unique blade keys
    all_keys = []
    for d in ldict:
        if d:
            all_keys += d.keys()
    keys = list(set(all_keys))

    dictl = {}
    for k in keys:
        dictl[k] = []

    for d in ldict:
        for k in keys:
            if not d or k not in d:
                dictl[k].append(None)
            else:
                dictl[k].append(d[k])

    return dictl


class NoneDict(dict):
    def __init__(self, *args, **kwargs):
        self.update(*args, **kwargs)

    def __getitem__(self, key):
        return dict.__getitem__(self, key) if key in self else None


def find(path, pattern=None):
    """Return all files under `path` with the substring `pattern`."""
    results = []
    for root, dirs, files in os.walk(path):
        for f in files:
            if not pattern or (pattern in f):
                results.append(os.path.join(root, f))
    return results


def circle_fit(x, y):
    def _iter(xyc):
        xc, yc = xyc
        R = np.sqrt((x - xc) ** 2 + (y - yc) ** 2)
        return R - R.mean()

    xyc0 = (np.mean(x), np.mean(y))

    xyc_opt = leastsq(_iter, xyc0)[0]
    R_opt = np.mean(np.sqrt((x - xyc_opt[0]) ** 2 + (y - xyc_opt[1]) ** 2))

    return xyc_opt, R_opt


def centroid(loop_xrt):
    # Area and centroid of the loop
    terms_cross = (
        loop_xrt[0, :-1] * loop_xrt[1, 1:] - loop_xrt[0, 1:] * loop_xrt[1, :-1]
    )
    terms_rt = loop_xrt[1, :-1] + loop_xrt[1, 1:]
    terms_x = loop_xrt[0, :-1] + loop_xrt[0, 1:]
    Area = 0.5 * np.sum(terms_cross)
    rt_cent = np.sum(terms_rt * terms_cross) / 6.0 / Area
    x_cent = np.sum(terms_x * terms_cross) / 6.0 / Area

    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots()
    # ax.plot(*loop_xrt,'k-x')
    # ax.plot(x_cent, rt_cent,'ro')
    # plt.savefig('cent.pdf')

    return (x_cent, rt_cent)


def to_basic_type(x):
    try:
        if np.shape(x) == ():
            return x.item()
        else:
            return x.tolist()
    except (AttributeError, TypeError):
        return x


def vecnorm(x):
    return np.sqrt(np.einsum("i...,i...", x, x))


def cluster_ER_by_npts(Dstart, Dend, ER, npts, exact_length):
    if Dstart > Dend:
        raise ValueError(f"Dstart={Dstart} should be < than Dend={Dend}")

    # Work out how many expanding points we need to hit Dend
    n1 = np.floor(1 + np.log(Dend / Dstart) / np.log(ER)).astype(int)

    # Subtract expanding from total to get number uniform points
    n2 = npts - n1

    # Spacings in expanding section
    Dn1 = Dstart * ER ** np.arange(0, n1)

    # Spacings in uniform section
    if exact_length:
        Lt = 1.0 - np.sum(Dn1)
        N = n2

        if N <= 2:
            raise ValueError(f"Clustering failed with Dstart={Dstart}, Dend={Dend}")

        Dst = Dn1[-1]
        Den = Dend
        A = 3 * (-2 * Lt + N * (Den + Dst)) * (N - 1) / (N * (N - 2))
        B = -(6 * Lt + N**2 * (2 * Den + 4 * Dst) + N * (-Den - 5 * Dst - 6 * Lt)) / (
            N * (N - 2)
        )
        C = Dst
        inorm = np.linspace(0.0, 1.0, N)
        Dn2 = A * inorm**2.0 + B * inorm + C
    else:
        Dn2 = np.ones((n2,)) * Dn1[-1]

    # Concatenate
    Dout = np.concatenate([Dn1, Dn2])
    yout = cumsum0(Dout)

    # yout /= yout[-1]

    return yout


def cluster_one_sided_ER(Dstart, Dend, ER, rtol=0.01):
    if Dstart > Dend:
        flip = True
        Dstart, Dend = Dend, Dstart
    else:
        flip = False

    def _iter_npts(N):
        return cluster_ER_by_npts(Dstart, Dend, ER, N, exact_length=False)[-1] - 1.0

    # Minimum number of points if all expanding
    nlow = np.floor(1 + np.log(Dend / Dstart) / np.log(ER)).astype(int) + 1
    errlow = _iter_npts(nlow)
    if errlow > 0.0:
        raise ValueError(
            f"Clustering failed with Dstart={Dstart} Dend={Dend} ER={ER}, increase ER"
        )

    # Increase maximum number of points until target length reached
    nhigh = nlow + 0
    max_iter = 50
    for _ in range(max_iter):
        nhigh = np.round(nhigh * 1.5).astype(int)
        errhigh = _iter_npts(nhigh)
        if errhigh > 1.0:
            break
    if errhigh < 0.0:
        raise ValueError("Clustering failed")

    # Find number of points by binary search
    dn = nhigh - nlow
    for _ in range(max_iter):
        nnew = (nlow + nhigh) // 2
        errnew = _iter_npts(nnew)
        if errnew < 0.0:
            nlow = nnew
            errlow = errnew
        else:
            nhigh = nnew
            errhigh = errnew
        dn = nhigh - nlow
        if dn == 1:
            break

    # Finally, evaluate the distribution and scale to unit interval
    x = cluster_ER_by_npts(Dstart, Dend, ER, nlow, exact_length=True)

    dx = np.diff(x)
    assert np.isclose(dx[0], Dstart)
    assert np.isclose(dx[-1], Dend)
    assert np.isclose(x[0], 0.0)
    assert np.isclose(x[-1], 1.0)
    assert (dx > 0.0).all()

    if flip:
        x = 1.0 - np.flip(x)

    dx = np.diff(x)
    assert np.isclose(x[0], 0.0)
    assert np.isclose(x[-1], 1.0)
    assert (dx > 0.0).all()

    # ERout = dx[1:]/dx[:-1]
    # ERout[ERout<1.] = 1./ERout[ERout<1.]
    # rtol = 1e-1
    # assert (ERout<ER*(1+rtol)).all()

    return x


def cluster_two_sided_ER(Dstart, Dmid, Dend, ER, rtol=0.01, fix_mid=True):
    if not fix_mid:
        Dmid_now = Dmid + 0.0
        maxiter = 1000
        for _ in range(maxiter):
            logger
            try:
                return cluster_two_sided_ER(
                    Dstart, Dmid_now, Dend, ER, rtol, fix_mid=True
                )
            except ValueError:
                Dmid_now *= 0.9
                logger.debug(f"Reducing Dmid={Dmid_now}")
        raise Exception(f"failed, Dstart={Dstart}, Dmid={Dmid}, Dend={Dend}, ER={ER}")

    x1 = cluster_one_sided_ER(Dstart * 2.0, Dmid * 2.0, ER, rtol) * 0.5
    x2 = cluster_one_sided_ER(Dend * 2.0, Dmid * 2.0, ER, rtol) * 0.5

    x = np.concatenate([x1[:-1], 1.0 - np.flip(x2)])
    dx = np.diff(x)
    assert np.isclose(dx[0], Dstart, rtol=rtol)
    assert np.isclose(dx[-1], Dend, rtol=rtol)
    assert (dx > 0.0).all()
    imid = len(x1) - 1
    assert np.allclose(
        dx[
            (imid, imid - 1),
        ],
        Dmid,
        rtol=rtol,
    )
    assert np.allclose(
        x[
            (0, -1),
        ],
        (0.0, 1.0),
        rtol=rtol,
    )
    return x


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


def resample(x, f, mult=None):
    """Multiply number of points in x by f, keeping relative spacings."""
    if np.isclose(f, 1.0):
        return x
    xnorm = (x - x[0]) / x.ptp()
    npts = len(x)
    npts_new = np.round((npts - 1) * f).astype(int) + 1
    if mult:
        npts_new = round_mg(npts_new, mult)
    inorm = np.linspace(0.0, 1.0, npts)
    inorm_new = np.linspace(0.0, 1.0, npts_new)
    xnorm_new = np.interp(inorm_new, inorm, xnorm)
    xnew = xnorm_new * x.ptp() + x[0]

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


def relax(xold, xnew, rf):
    return xold * (1.0 - rf) + xnew * rf


def cluster_new_free(Dst, Dmax, ERmax):
    # Number of points needed to expand Dst to Dmax at full stretch
    N1 = np.ceil(np.log(Dmax / Dst) / np.log(ERmax)) + 1

    # Evaluate coordinates
    dx1 = Dst * ERmax ** np.arange(0, N1 - 1)
    L1 = np.sum(dx1)

    # Check if we will need a unifor section
    if L1 < 1.0:
        # Uniform needed
        dx1max = dx1[-1]
        L1 = np.sum(dx1)
        N2 = np.ceil((1.0 - L1) / 0.5 / (dx1max + Dmax)).astype(int)
        if N2 == 1:
            # Reduce ER a bit
            return cluster_new_free(Dst, Dmax, ERmax * 0.99)
        else:
            Dmax_adjust = (1.0 - L1) / N2 / 0.5 - dx1max
            dx2 = np.linspace(dx1max, Dmax_adjust, N2)

        dx = np.concatenate((dx1, dx2))

    else:
        # Truncate the expansion
        x1 = cumsum0(dx1)
        N = np.argmax(x1 > 1.0) + 1

        # Closure to return L and dLdER as function of ER
        def _iter(ERi):
            # Length error
            L = Dst * (1.0 - ERi ** (N - 1)) / (1.0 - ERi) - 1.0
            # Derivative
            dL = (
                Dst
                * ((N - 2.0) * ERi**N - (N - 1.0) * ERi ** (N - 1.0) + ERi)
                / (1.0 - ERi) ** 2.0
                / ERi
            )
            return L, dL

        # Solve for the ER that gives unit length
        ER = root_scalar(_iter, x0=ERmax, fprime=True).root
        dx = Dst * ER ** np.arange(0, N - 1)

    x = cumsum0(dx)

    assert x[0] == 0.0
    assert np.isclose(x[-1], 1.0)
    assert np.isclose(dx[0], Dst, rtol=1e-2)
    assert np.all(dx <= Dmax)
    assert np.all(dx > 0.0)

    return x


def cluster_two_sided_free(x1, x2, dx1, dx2, dxmax, ER):
    L = np.abs(x2 - x1)
    D1 = dx1 / L * 2.0
    D2 = dx2 / L * 2.0
    Dmax = dxmax / L * 2.0

    xh1 = cluster_new_free(D1, Dmax, ER) * 0.5
    xh2 = np.flip(1.0 - cluster_new_free(D2, Dmax, ER)) * 0.5 + 0.5
    x12 = np.concatenate((xh1[:-1], xh2))
    x = x1 + (x2 - x1) * x12
    dx = np.diff(x) * np.sign(x2 - x1)

    assert np.isclose(x[0], x1)
    assert np.isclose(x[-1], x2)
    assert np.isclose(dx[0], dx1)
    assert np.isclose(dx[-1], dx2)
    assert (np.abs(dx) > 0.0).all()
    assert (np.abs(dx) < dxmax).all()

    return x


def cluster_two_sided(x1, x2, dxmin, dxmax, ER, N):
    L = np.abs(x2 - x1)
    Dst = dxmin / L * 2.0
    Den = dxmax / L * 2.0
    if np.mod(N, 2):
        npts1 = N // 2 + 1
        xh = 0.5 * cluster_new(Dst, Den, ER, npts1)
        x = np.concatenate((xh[:-1], 1.0 - np.flip(xh)))
    else:
        npts1 = (N + 1) // 2 + 1
        xh1 = 0.5 * cluster_new(Dst, Den, ER, npts1)
        xh2 = 0.5 * cluster_new(Dst, Den, ER, npts1 - 1)
        x = np.concatenate((xh1[:-1], 1.0 - np.flip(xh2)))
    dx = np.diff(x)
    assert np.isclose(x[0], 0.0)
    assert np.isclose(x[-1], 1.0)
    assert np.all(dx > 0.0)
    if not len(x) == N:
        raise Exception(
            f"Wrong number of points, N={N}, npts1={npts1}, len(x)={len(x)}"
        )
    assert np.isfinite(x).all()
    xdim = x1 + (x2 - x1) * x
    return xdim


def cluster_one_sided(x1, x2, dxmin, dxmax, ER, N):
    L = np.abs(x2 - x1)
    Dst = dxmin / L
    Den = dxmax / L
    x = cluster_new(Dst, Den, ER, N)
    dx = np.diff(x)
    assert np.isclose(x[0], 0.0)
    assert np.isclose(x[-1], 1.0)
    assert np.all(dx > 0.0)
    assert len(x) == N
    assert np.isfinite(x).all()
    xdim = x1 + (x2 - x1) * x
    return xdim


def cluster_new(Dst, Dmax, ERmax, N):
    # Number of points needed to expand Dst to Dmax at full stretch
    N1 = np.ceil(np.log(Dmax / Dst) / np.log(ERmax)) + 1

    if N1 >= N:
        # We will not reach Dmax before using up all points, so just expansion
        # needed

        # Closure to return L and dLdER as function of ER
        def _iter(ERi):
            # Length error
            L = Dst * (1.0 - ERi ** (N - 1)) / (1.0 - ERi) - 1.0
            # Derivative
            dL = (
                Dst
                * ((N - 2.0) * ERi**N - (N - 1.0) * ERi ** (N - 1.0) + ERi)
                / (1.0 - ERi) ** 2.0
                / ERi
            )
            return L, dL

        # Solve for the ER that gives unit length
        ER = root_scalar(_iter, x0=ERmax, fprime=True).root

        # Evaluate coordinates
        dx = Dst * ER ** np.arange(0, N - 1)
        x = cumsum0(dx)

    else:
        # If we reach Dmax before using up all points, then a uniform section
        # shall be needed.

        dx1 = Dst * ERmax ** np.arange(0, N1 - 1)
        dx1max = Dst * ERmax ** (N1 - 2)
        L1 = np.sum(dx1)

        # Linear increase in spacing
        N2 = N - N1
        L2 = N2 * 0.5 * (dx1max + Dmax)

        if L1 + L2 < 1.0:
            raise ValueError(
                f"""Not enough points to cluster.
{L1 + L2},
ERmax={ERmax}, Dst={Dst}, Dmax={Dmax}, N={N}
Increase ERmax, Dst, Dmax, or N."""
            )
        else:
            # We have too many points
            # reduce Dmax by reducing N1
            for dn in range(0, int(N1)):
                N1_now = N1 - dn

                # dx1now = Dst * ERmax ** np.arange(0, N1_now - 1)
                # L1 = np.sum(dx1now)

                L1 = Dst * (1.0 - ERmax ** (N1_now - 1)) / (1.0 - ERmax)
                dx1max = Dst * ERmax ** (N1_now - 2)
                L2 = 1.0 - L1
                N2 = N - N1_now
                if L2 / N2 > dx1max:
                    Dmax_now = 2.0 * L2 / N2 - dx1max
                    if Dmax_now > Dmax:
                        # We need to use one more expanding point but solve for
                        N1_now += 1
                        N2 = N - N1_now
                        L1 = Dst * (1.0 - ERmax ** (N1_now - 1)) / (1.0 - ERmax)
                        dx1max = Dst * ERmax ** (N1_now - 2)
                        L2 = 1.0 - L1
                        Dmax_now = 2.0 * L2 / N2 - dx1max
                    dx1 = Dst * ERmax ** np.arange(0, N1_now - 1)
                    dx2 = np.linspace(dx1max, Dmax_now, int(N2))
                    dx = np.concatenate((dx1, dx2))
                    x = cumsum0(dx)
                    break

            # If we reach this point, then even with a uniform grid the
            # spacings will be smaller than those requested.
            if N1 == 0 or N1_now == 1:
                x = np.linspace(0.0, 1.0, N)

        assert len(x) == N
        assert x[0] == 0.0
        assert np.isclose(x[-1], 1.0)
        dx = np.diff(x)
        # assert np.isclose(dx[0], Dst, rtol=1e-2)
        ER = dx[1:] / dx[:-1]
        ER[ER < 1.0] = 1.0 / ER[ER < 1.0]
        if np.any(ER > ERmax * 1.01):
            raise ValueError(f"Expansion ratio {ER.max()} exceeds ERmax={ERmax}")
        assert np.all(dx <= Dmax)

    return x


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


def subsample_cases(c, k, K):
    """Split into K parts, extract kth and other K-1 subsamples."""
    di = int(np.floor(len(c) / K))
    c_test = []
    c_train = []
    for i in range(K):
        ist = di * i
        if i == K - 1:
            ien = len(c)
        else:
            ien = di * (i + 1)
        cnow = c[ist:ien]
        if i == k:
            c_test += cnow
        else:
            c_train += cnow
    return c_test, c_train


def hyperfaces(x):
    """Unstructured copy of all elements on hypercube faces.

    This function is the multidimensional equivalent of the following:

    xf = np.unique(
        np.concatenate(
            (
                x[:, 0, :],
                x[:, -1, :],
                x[:, :, 0],
                x[:, :, -1],
            ),
            axis=-1,
        ),
        axis=-1,
    )

    Parameters
    ----------
    x: (M, N0, N1, ..., NM) array
        Points in an M-dimensional hypercube; x.ndim == M+1.

    Returns
    -------
    xf: (M, Nf) array
        All points that are located on faces of the hypercube.

    """

    M = x.shape[0]
    assert x.ndim == M + 1

    xf = []

    # Loop over each index to extract the faces of
    for m in range(M):
        # For the face located at start or end of current index
        for iface in (0, -1):
            # Construct a fancy indexing tuple that slices everything
            ind = [
                slice(None),
            ] * (M + 1)
            # On the current index, select only the start or end
            # Add one because the first element slices over dimensions
            ind[m + 1] = iface
            # Extract these elements and add to list
            xf.append(x[tuple(ind)].reshape(M, -1))

    # Join all the elements from each of the faces
    xf = np.concatenate(xf, axis=-1)

    # Remove duplicates
    xf = np.unique(xf, axis=1)

    return xf


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


def rms(x):
    return np.sqrt(np.mean(np.array(x) ** 2))


def round_mg(n, mult=8):
    return int(mult * np.ceil((n - 1) / mult)) + 1


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
    # print(re_id.match('beans_0001').groups()[0])

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


def load_mean_line(mean_line_type):
    if not mean_line_type.endswith(".py"):
        # Attempt to load a built-in meanline
        mod = importlib.import_module(f".{mean_line_type}", package="turbigen.meanline")
    else:
        # Use as a file path
        mod_file = os.path.abspath(mean_line_type)
        mod_name = os.path.basename(mean_line_type)
        spec = importlib.util.spec_from_file_location(
            f"turbigen.meanline.{mod_name}", mod_file
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules[f"turbigen.meanline.{mod_name}"] = mod
        spec.loader.exec_module(mod)
    return mod


def load_annulus(annulus_type):
    if not annulus_type.endswith(".py"):
        # Attempt to load a built-in annulus
        mod = importlib.import_module(".annulus", package="turbigen")
        mod = getattr(mod, annulus_type)
    else:
        # Use as a file path
        mod_file = os.path.abspath(annulus_type)
        mod_name = os.path.basename(annulus_type)
        spec = importlib.util.spec_from_file_location(
            f"turbigen.annulus.{mod_name}", mod_file
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules[f"turbigen.annulus.{mod_name}"] = mod
        spec.loader.exec_module(mod)
        mod = mod.Annulus
    return mod


def load_install(install_type):
    if not install_type.endswith(".py"):
        # Attempt to load a built-in meanline
        mod = importlib.import_module(f".{install_type}", package="turbigen.install")
    else:
        # Use as a file path
        mod_file = os.path.abspath(install_type)
        mod_name = os.path.basename(install_type)
        spec = importlib.util.spec_from_file_location(
            f"turbigen.install.{mod_name}", mod_file
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules[f"turbigen.install.{mod_name}"] = mod
        spec.loader.exec_module(mod)
    return mod


def load_post(post_type):
    if not post_type.endswith(".py"):
        # Attempt to load a built-in post
        mod = importlib.import_module(f".{post_type}", package="turbigen.post")
    else:
        # Use as a file path
        mod_file = os.path.abspath(post_type)
        mod_name = os.path.basename(post_type)
        spec = importlib.util.spec_from_file_location(
            f"turbigen.post.{mod_name}", mod_file
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules[f"turbigen.post.{mod_name}"] = mod
        spec.loader.exec_module(mod)
    return mod


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


def stagnation_point_angle(grid, machine, meanline, fac_Rle=1.0):

    surfs = grid.cut_blade_surfs()

    chi_stag = []

    # Loop over rows
    for irow, surfi in enumerate(surfs):

        chi_stag.append([])

        # Loop over main/splitter
        for jbld, surfj in enumerate(surfi):

            surf = surfj.squeeze()
            _, nj = surf.shape

            spf = [surf.spf[surf.i_stag[j], j] for j in range(nj)]

            # Get coordinates of stagnation point
            xrt_stag = surf.xrt_stag

            # Get coordinates of LE center
            bldnow = machine.split[irow] if jbld else machine.bld[irow]
            xrt_cent = np.stack(
                [bldnow.get_LE_cent(spf[j], fac_Rle).squeeze() for j in range(nj)],
                axis=-1,
            )

            # Get vector between stagnation point and centre of LE
            dxrt = xrt_cent - xrt_stag

            # Multiply theta component by reference radius
            dxrrt = dxrt.copy()
            rref = 0.5 * (xrt_cent + xrt_stag)[1]
            dxrrt[2] *= rref

            # Calculate angle
            denom = np.sqrt(dxrrt[0] ** 2 + dxrrt[1] ** 2)
            chi_stag_now = np.degrees(np.arctan2(dxrrt[2], denom))

            # # If we are going radially inwards then flip sign
            # if meanline.Beta[irow * 2] < -45.0:
            #     chi_stag_now *= -1.0

            chi_stag[-1].append(np.stack((spf, chi_stag_now)))

    return chi_stag


def incidence(grid, machine, meanline, fac_Rle=1.0):

    chi_stag_all = stagnation_point_angle(grid, machine, meanline, fac_Rle)

    out = []

    # Loop over rows
    for irow, chi_stag_row in enumerate(chi_stag_all):

        out.append([])

        for jblade, chi_stag_blade in enumerate(chi_stag_row):

            spf, chi_stag = chi_stag_blade

            bldnow = machine.split[irow] if jblade else machine.bld[irow]
            chi_metal = np.array([bldnow.get_chi(spfj)[0] for spfj in spf])

            # Smooth
            nsmooth = 10
            sf = 0.5
            for _ in range(nsmooth):
                chi_avg = 0.5 * (chi_stag[:-2] + chi_stag[2:])
                chi_stag[1:-1] = sf * chi_stag[1:-1] + (1.0 - sf) * chi_avg

            incidence = chi_stag - chi_metal

            out_now = np.stack((spf, incidence, chi_stag, chi_metal))

            # Remove results in tip gap
            out_now[1:, spf > (1.0 - machine.tip[irow])] = np.nan

            out[-1].append(out_now)

    return out


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

    def mp_from_xr(xr):
        func = scipy.interpolate.NearestNDInterpolator(xr_ref.T, mp_ref)
        xru = xr.reshape(2, -1)
        mpu = func(xru.T)
        return mpu.reshape(xr.shape[1:])

    return mp_from_xr, spf_actual
