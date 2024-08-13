"""Classes for generating meshes in two dimensions."""
import numpy as np
from turbigen import util
from turbigen import clusterfunc
from scipy.interpolate import interp1d


class Point:
    def __init__(self, x, y):
        """A 0-D point in 2-D space."""
        # Check input
        util.check_scalar(x=x, y=y)
        self.xy = np.reshape((x, y), (2, 1))

    @property
    def x(self):
        return self.xy[0, 0]

    @property
    def y(self):
        return self.xy[1, 0]

    def __eq__(self, other):
        return np.isclose(self.xy, other.xy).all(axis=0)

    def plot(self, ax):
        ax.plot(*self.xy, "o")


class Curve:
    def __init__(self, x, y):
        """A 1-D curve in 2-D space."""
        # Check input
        # self.n = len(x)
        # util.check_vector((self.n,), x=x, y=y)
        # self.xy = np.stack((np.reshape(x, -1), np.reshape(y, -1)))

        self.xy = np.stack(np.broadcast_arrays(x, y))
        self.n = self.xy.shape[1]

        # Check that there are no repeats
        if (self.dxy == 0.0).all(axis=0).any():
            raise Exception("Could not create curve from repeated points.")

    def __getitem__(self, key):
        # Special case for scalar indices
        if np.isscalar(key):
            return Point(*self.xy[:, key])
        else:
            return Curve(*self.xy[:, key])

    def split_by_index(self, ind):
        curves = []
        ind = np.concatenate(
            (
                [
                    0,
                ],
                ind,
                [-1],
            )
        )
        ind[ind < 0] = ind[ind < 0] + self.n
        for i in range(len(ind) - 1):
            curves.append(self[ind[i] : (ind[i + 1] + 1)])
        return curves

    def __len__(self):
        return self.xy.shape[-1]

    @property
    def reversed(self):
        return Curve(*np.flip(self.xy, axis=1))

    @classmethod
    def from_join(cls, *args):
        """Create a Curve by concatenating other curves."""

        curves = list(args)

        # Sort the curves in an order of connectivity
        # Starting with the first curve
        curves_sorted = [
            curves.pop(0),
        ]

        # Loop over remaining curves
        while len(curves):
            if curves_sorted[-1].en == curves[0].st:
                curves_sorted.append(curves.pop(0)[1:])
            elif curves_sorted[-1].en == curves[0].en:
                curves_sorted.append(curves.pop(0).reversed[1:])
            elif curves_sorted[0].st == curves[0].en:
                curves_sorted.insert(0, curves.pop(0)[:-1])
            elif curves_sorted[0].st == curves[0].st:
                curves_sorted.insert(0, curves.pop(0).reversed[:-1])

        return Curve(*np.concatenate([c.xy for c in curves_sorted], axis=1))

    @classmethod
    def from_points(cls, *args):
        return Curve(*np.concatenate([p.xy for p in args], axis=1))

    @classmethod
    def from_cluster_single(cls, p0, p1, d0, dmax, ER=1.2):
        c01 = Curve.from_points(p0, p1)
        d0n = d0 / c01.S
        dmaxn = dmax / c01.S
        sn = clusterfunc.single.free(d0n, dmaxn, ER, mult=4)
        return c01.interpolate(sn)

    @classmethod
    def from_cluster_double(cls, p0, p1, d0, d1, dmax, ER=1.2):
        c01 = Curve.from_points(p0, p1)
        d0n = d0 / c01.S
        d1n = d1 / c01.S
        dmaxn = dmax / c01.S
        sn = clusterfunc.double.free(d0n, d1n, dmaxn, ER, mult=4)
        return c01.interpolate(sn)

    @classmethod
    def from_uniform(cls, p0, p1, N):
        c01 = Curve.from_points(p0, p1)
        sn = np.linspace(0.0, 1.0, N)
        return c01.interpolate(sn)

    def roll(self, n):
        if self.is_closed:
            xy_roll = np.roll(self.xy[:, :-1], n, axis=1)
            xy_roll = np.append(xy_roll, xy_roll[:, (0,)], axis=1)
        else:
            xy_roll = np.roll(self.xy, n, axis=1)
        return Curve(*xy_roll)

    @property
    def angle(self):
        """Angle of slope of this curve."""
        dxy = self.dxy
        return np.degrees(np.arctan2(dxy[1], dxy[0]))

    @property
    def is_closed(self):
        return self[0] == self[-1]

    @property
    def x(self):
        """Horizontal coordinate."""
        return self.xy[0]

    @property
    def y(self):
        """Vertical coordinates."""
        return self.xy[1]

    @property
    def dxy(self):
        """Stepwise vectors for each curve segment."""
        return np.diff(self.xy, axis=1)

    @property
    def dx(self):
        return self.dxy[0]

    @property
    def dy(self):
        return self.dxy[1]

    @property
    def ds(self):
        """Elements of arc length."""
        return np.sqrt(np.sum(self.dxy**2, axis=0))

    @property
    def perp_edge(self):
        """Perpendicular unit vectors, edge-centered."""
        dxy = self.dxy
        perp_edge = np.stack((-dxy[1], dxy[0]))
        perp_edge /= np.linalg.norm(perp_edge, axis=0, keepdims=True)
        return perp_edge

    @property
    def perp_node(self):
        """Perpendicular unit vectors, node-centered."""
        # Put the perpendicular vectors back to nodes
        perp_edge = self.perp_edge
        perp_node = np.concatenate(
            (
                perp_edge[:, (0,)],
                0.5 * (perp_edge[:, :-1] + perp_edge[:, 1:]),
                perp_edge[:, (-1,)],
            ),
            axis=1,
        )
        if self.is_closed:
            perp_end = np.mean(perp_edge[:, (0, -1)], axis=1)
            perp_node[:, 0] = perp_end
            perp_node[:, -1] = perp_end

        return perp_node

    @property
    def s(self):
        """Cumulative arc length along the curve"""
        return util.cumsum0(self.ds)

    @property
    def S(self):
        """Total arc length along the curve"""
        return self.s[-1]

    @property
    def sn(self):
        """Normalised cumulative arc length along the curve"""
        return self.s / self.S

    @property
    def st(self):
        """Start point of the curve."""
        return Point(*self.xy[:, 0])

    @property
    def en(self):
        """End point of the curve."""
        return Point(*self.xy[:, -1])

    def __eq__(self, other):
        return np.isclose(self.xy, other.xy).all(axis=0)

    def __contains__(self, key):
        return (self == key).any()

    def copy(self):
        return Curve(*self.xy)

    def interpolate(self, snq):
        """Interpolate new points on the Curve by normalised distance"""
        return Curve(*interp1d(self.sn, self.xy, axis=1)(snq))

    def decluster(self, f=1.0):
        snu = np.linspace(0.0, 1.0, self.n)
        return self.interpolate(util.relax(self.sn, snu, f))

    def project_to_x(self, xp):
        xy = self.xy.copy()
        xy[0] = xp
        return Curve(*xy)

    def plot(self, ax):
        ax.plot(*self.xy, ".-", lw=0.5, ms=1)

    def offset(self, L, flip=False):

        # Check input
        assert np.isscalar(L)

        # Choose direction
        if flip:
            L *= -1.0

        # Add on the distance
        xy_offset = self.xy + self.perp_node * L

        return Curve(*xy_offset)


class Block:
    def __init__(self, x, y):
        """A 2-D grid of points in 2D space."""
        # Check input
        self.ni, self.nj = np.shape(x)
        util.check_vector(self.shape, x=x, y=y)
        self.xy = np.stack((x, y))
        # Check that there are no repeats
        if (self.dxyi == 0.0).all(axis=0).any():
            raise Exception("Could not create block, repeated points in i-dirn.")
        if (self.dxyj == 0.0).all(axis=0).any():
            raise Exception("Could not create block, repeated points in j-dirn.")

    def __getitem__(self, key):
        if not len(key) == 2:
            raise Exception(f"Need two indices for a Block, got {len(key)}")
        i, j = key
        if np.isscalar(i) and np.isscalar(j):
            return Point(*self.xy[:, i, j])
        elif np.isscalar(i) or np.isscalar(j):
            return Curve(*self.xy[:, i, j])
        else:
            return Block(*self.xy[:, i, j])

    @classmethod
    def from_offset(cls, c, L, flip=False):
        """Generate a Block by offsetting a Curve by a vector of perpendicular distances."""

        # Check input
        L = np.array(L)
        assert L.ndim == 1

        # Arrange vectors so they broadcast
        L = np.array(L).reshape(1, 1, -1)
        perp_node = c.perp_node.reshape(2, -1, 1)
        xy = c.xy.reshape(2, -1, 1)

        # Choose direction
        if flip:
            L *= -1.0

        # Add on the distance
        xy_offset = xy + perp_node * L

        # Make sure closed curves remain closed
        if c.is_closed:
            xy_TE = np.mean(xy_offset[:, (0, -1)], axis=1)
            xy_offset[:, 0] = xy_TE
            xy_offset[:, -1] = xy_TE

        return cls(*xy_offset)

    @property
    def dxyi(self):
        """Stepwise vectors along i dirn."""
        return np.diff(self.xy, axis=1)

    @property
    def dxyj(self):
        """Stepwise vectors along j dirn."""
        return np.diff(self.xy, axis=2)

    @classmethod
    def from_stack(cls, curves):
        """Create a Block by stacking Curves along i."""
        xy = np.stack([c.xy for c in curves], axis=1)
        return cls(*xy)

    @classmethod
    def from_project_to_x(cls, c0, xp, dl0, AR, ER=1.2):

        # Get a block with start and projected end curves
        c1 = c0.project_to_x(xp).decluster()
        b = cls.from_stack((c0, c1))

        # Calculate clustering
        L = b.Si.mean()  # Average distance between two curves
        dj1 = b.dsj[-1, :].mean()  # Average j-spacing on the new curve
        dln1 = AR * dj1 / L
        dln0 = dl0 / L
        print(dln0, dln1)
        clu = clusterfunc.single.free(dln0, dln1, ER)
        bclu = b.interpolate_i(clu)
        return bclu

    def interpolate_i(self, sniq):
        xrq = np.full((2, len(sniq), self.nj), np.nan)
        for j in range(self.nj):
            xrq[:, :, j] = interp1d(self.sni[:, j], self.xy[:, :, j], axis=1)(sniq)
        return Block(*xrq)

    @classmethod
    def from_transfinite(cls, *args):

        curves = list(args)
        # Sort the curves by mean x and y
        ix = np.argsort([c.x.mean() for c in curves])
        iy = np.argsort([c.y.mean() for c in curves])

        # Work out which curve is which
        c_i0 = curves[ix[0]]
        c_ni = curves[ix[-1]]
        c_j0 = curves[iy[0]]
        c_nj = curves[iy[-1]]

        # Make sure the curves go in increasing x or y
        if c_i0.dy.sum() < 0.0:
            c_i0 = c_i0.reversed
        if c_ni.dy.sum() < 0.0:
            c_ni = c_ni.reversed
        if c_j0.dx.sum() < 0.0:
            c_j0 = c_j0.reversed
        if c_nj.dx.sum() < 0.0:
            c_nj = c_nj.reversed

        c_all = [c_j0, c_i0, c_nj, c_ni]
        xy_all = [c.xy for c in c_all]
        return Block(*util.interpolate_transfinite(xy_all))

    @property
    def dsi(self):
        return np.sqrt(np.sum(self.dxyi**2, axis=0))

    @property
    def dsj(self):
        return np.sqrt(np.sum(self.dxyj**2, axis=0))

    @property
    def si(self):
        """Cumulative arc length along i-dirn"""
        return util.cumsum0(self.dsi, axis=0)

    @property
    def sj(self):
        """Cumulative arc length along i-dirn"""
        return util.cumsum0(self.dsj, axis=1)

    @property
    def Si(self):
        """Total arc length along i-dirn"""
        return self.si[-1, :]

    @property
    def Sj(self):
        """Total arc length along j-dirn"""
        return self.sj[:, -1]

    @property
    def sni(self):
        """Normalised cumulative arc length along i-dirn"""
        return self.si / self.Si.reshape(1, -1)

    @property
    def snj(self):
        """Normalised cumulative arc length along j-dirn"""
        return self.sj / self.Sj.reshape(-1, 1)

    @property
    def T(self):
        return Block(*self.xy.transpose(0, 2, 1))

    @property
    def xyu(self):
        return self.xy.reshape(2, -1)

    @property
    def ij(self):
        return np.stack(
            np.meshgrid(*[np.arange(0, n) for n in self.shape], indexing="ij")
        )

    @property
    def iju(self):
        return self.ij.reshape(2, 1)

    def iju_corner(self):
        pass

    @property
    def shape(self):
        return (self.ni, self.nj)

    def plot(self, ax):
        ax.plot(*self.xy, "k-", lw=0.5)
        ax.plot(*self.T.xy, "k-", lw=0.5)


def split_by_angle(block, angles, j=-1):

    curve = block[:, j]

    isplit = [np.argmin(np.abs(curve.angle - angi)) + 1 for angi in angles]
    isplit = np.array(
        [
            0,
        ]
        + isplit
        + [
            curve.n - 1,
        ]
    )
    if not (np.diff(isplit) > 0).all():
        raise Exception(
            f"Found non-monotonic split indices {isplit} for angles={angles} "
        )
    nsplit = len(isplit) - 1
    curves = [curve[isplit[k] : (isplit[k + 1] + 1)] for k in range(nsplit)]
    if curve.is_closed:
        curve_end = Curve.from_join(curves[-1], curves[0])
        curves = curves[1:-1] + [
            curve_end,
        ]
    # Now get cell sizes at the splits
    # In the j-direction
    dsj = block.dsj[isplit[1:-1], j]
    # In the i-direction
    dsi1 = block.dsi[isplit[1:-1], j]
    dsi2 = block.dsi[isplit[1:-1] + 1, j]
    ds = np.mean(np.stack((dsj, dsi1, dsi2)), axis=0)

    return curves, isplit, ds


def find_periodic(blocks, pitch):
    pass

    # periodics = []
    # nb = len(blocks)
    # for n in range(nb):
    #     for m in range(nb):
    #         ind_match = np.where(blocks
    #


def concatenate_blocks(*args):
    """Join a sequence of Blocks, automatically reorienting as needed."""

    blocks = list(args)
