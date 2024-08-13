"""Classes for generating meshes in two dimensions."""
import numpy as np
from turbigen import util
from turbigen import clusterfunc
from scipy.interpolate import interp1d
from enum import Enum
from dataclasses import dataclass

Edge = Enum("Edge", ["i0", "ni", "j0", "nj"])


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

    def copy(self):
        return Point(*self.xy.squeeze().copy())


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

    def xy_mod(self, pitch):
        """Coordinates with modulus wrt pitch on vertical."""
        if pitch:
            return np.stack(
                (
                    self.x,
                    np.mod(self.y, pitch),
                )
            )
        else:
            return self.xy

    def __eq__(self, other):
        return np.isclose(self.xy, other.xy).all(axis=0)

    def __contains__(self, key):
        return (self == key).any()

    def copy(self):
        return Curve(*self.xy)

    def interpolate(self, snq):
        """Interpolate new points on the Curve by normalised distance"""
        if isinstance(snq, int):
            snq = np.linspace(0.0, 1.0, snq)
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

    @property
    def shape(self):
        return (self.ni,)


class Block:
    def __init__(self, x, y, label=None):
        """A 2-D grid of points in 2D space."""
        # Check input
        self.ni, self.nj = np.shape(x)
        util.check_vector(self.shape, x=x, y=y)
        self.xy = np.stack((x, y))
        self.conn = {k: [] for k in Edge}
        self.label = label
        # Check that there are no repeats
        if (self.dxyi == 0.0).all(axis=0).any():
            raise Exception("Could not create block, repeated points in i-dirn.")
        if (self.dxyj == 0.0).all(axis=0).any():
            raise Exception("Could not create block, repeated points in j-dirn.")

    def __repr__(self):
        return f"Block(label={self.label})"

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
        return Block(*self.xy.transpose(0, 2, 1).copy())

    @property
    def y(self):
        return self.xy[1]

    @property
    def x(self):
        return self.xy[0]

    @property
    def shape(self):
        return (self.ni, self.nj)

    def plot(self, ax):
        ax.plot(*self.xy, "k-", lw=0.5)
        ax.plot(*self.T.xy, "k-", lw=0.5)

    def flip(self, axis):
        return Block(*np.flip(self.xy, axis=axis + 1).copy())

    def extrude(self, zv):
        return np.stack(
            np.broadcast_arrays(
                *self.xy[..., None],
                zv.reshape(1, 1, -1),
            )
        )

    @property
    def edges(self):
        """Curves for the bounding edges of this block in i0, ni, j0, nj order."""
        return {
            Edge.i0: self[0, :],
            Edge.ni: self[-1, :],
            Edge.j0: self[:, 0],
            Edge.nj: self[:, -1],
        }


@dataclass
class Conn:
    """A periodic connection between nodes."""

    b: Block
    e: Edge
    st: int
    en: int
    flip: bool

    def __post_init__(self):
        assert self.st <= self.en

    def get_xy(self):
        xy = self.b.edges[self.e][self.st : self.en]
        if self.flip:
            xy = np.flip(xy)
        return xy


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


def concatenate_blocks(*args):
    """Join a sequence of Blocks, automatically reorienting as needed."""

    blocks = list(args)

    # Orient the blocks to have a consistent orientation
    nb = len(blocks)
    for ib in range(nb):

        b = blocks[ib]

        # First make sure that i ~ x and j ~ y
        if np.abs(b.dxyi[1]).mean() > np.abs(b.dxyi[0]).mean():
            b = b.T

        # Now make sure that i is +x
        if b.dxyi[0].mean() < 0.0:
            b = b.flip(axis=0)

        # Now make sure that j is +y
        if b.dxyj[1].mean() < 0.0:
            b = b.flip(axis=1)

        blocks[ib] = b

    # Next we need to determine the concatenation axis
    xya = np.array([b.xy.mean(axis=(1, 2)) for b in blocks])
    axis = np.argmax(np.abs(np.diff(xya, axis=0).sum(axis=0)))
    blocks = [blocks[ib] for ib in np.argsort(xya[:, axis])]

    # Remove first point from blocks other than start to avoid repeats
    if axis == 0:
        xyb = [b.xy[:, :-1, :] for b in blocks] + [blocks[-1].xy[:, (-1,), :]]
    else:
        xyb = [b.xy[:, :, :-1] for b in blocks] + [blocks[-1].xy[:, :, (-1,)]]

    xy = np.concatenate(xyb, axis=axis + 1)

    return Block(*xy)


def find_periodic(b1, b2, pitch, ax):

    conn = []

    # Loop over all combinations of edges
    for e1, c1 in b1.edges.items():
        for e2, c2 in b2.edges.items():

            ds = np.minimum(c1.ds.min(), c2.ds.min()) * 1e-3

            # Skip if the edges are the same
            if b1 is b2 and e1 == e2:
                continue

            # Compare the nodes on each edge
            i1, i2 = util.intersect_indices(c1.xy_mod(pitch), c2.xy_mod(pitch), ds)

            # Only consider connections involving multiple elements
            # (Ignore single points and len(2) repeated single points)
            if len(i1) <= 2:
                continue

            # Add to plot
            if ax:
                ax.plot(*c1[i1].xy, "ro")
                ax.plot(*c2[i2].xy, "bx")

            # Check for a flipped indexing
            flip1 = bool((np.diff(i1) < 0).any())
            flip2 = bool((np.diff(i2) < 0).any())
            flip = flip1 or flip2

            sten1 = get_st_en(i1, c1.n, flip1)
            sten2 = get_st_en(i2, c2.n, flip2)
            nseg = len(sten1)
            for iseg in range(nseg):
                conn.append(
                    (
                        Conn(b1, e1, *sten1[iseg], flip),
                        Conn(b2, e2, *sten2[iseg], flip),
                    )
                )

    return conn

    # Todo add Conn for both sides to a global list
    # Not storing as an attribute on each block;w
    #


def find_periodics(blocks, pitch=None, ax=None):
    """Locate periodic nodes and assemble their indices."""

    conn = []

    # Loop over all combinations of blocks
    for b1 in blocks:
        for b2 in blocks:
            conn.extend(find_periodic(b1, b2, pitch, ax))

    # Remove reversed repeats
    nconn = len(conn)
    for iconn in reversed(range(nconn)):
        crev = tuple(reversed(conn[iconn]))
        if crev in conn:
            conn.pop(iconn)

    return conn


def get_st_en(ind, n, flip):

    st = []
    en = []
    dind = np.diff(ind)

    gaps = np.where(dind > 1)[0]

    st.append(int(ind[0]))
    for iseg in range(len(gaps)):
        en.append(ind[gaps[iseg]])
        st.append(ind[gaps[iseg] + 1])
    en.append(int(ind[-1]))

    if flip:
        st, en = en, st
    return tuple(zip(st, en))


# x = np.array([1,2,3,5,6,7])
# print(x)
# print(get_st_en(x, 10))
# quit()


# def find_periodic(blocks, pitch):
#     pass

# periodics = []
# nb = len(blocks)
# for n in range(nb):
#     for m in range(nb):
#         ind_match = np.where(blocks
#
