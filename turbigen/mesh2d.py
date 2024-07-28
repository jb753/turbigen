"""Classes for generating meshes in two dimensions."""
import numpy as np
from turbigen import util


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


class Curve:
    def __init__(self, x, y):
        """A 1-D curve in 2-D space."""
        # Check input
        self.n = len(x)
        util.check_vector((self.n,), x=x, y=y)
        self.xy = np.stack((np.reshape(x, -1), np.reshape(y, -1)))

        # Check that there are no repeats
        if (self.dxy == 0.0).all(axis=0).any():
            raise Exception("Could not create curve from repeated points.")

    def __getitem__(self, key):
        # Special case for scalar indices
        if np.isscalar(key):
            return Point(*self.xy[:, key])
        else:
            return Curve(*self.xy[:, key])

    def __len__(self):
        return self.xy.shape[-1]

    @property
    def reversed(self):
        return Curve(*np.flip(self.xy, axis=1))

    @classmethod
    def from_join(cls, *args):
        """Create a Curve by concatenating other curves."""

        curves = list(args)
        ncurve = len(curves)

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

        return Curve(*np.concatenate([c.xy for c in curves_sorted], axis=1))

    def roll(self, n):
        if self.is_closed:
            xy_roll = np.roll(self.xy[:, :-1], n, axis=1)
            xy_roll = np.append(xy_roll, xy_roll[:, (0,)], axis=1)
        else:
            xy_roll = np.roll(self.xy, n, axis=1)
        return Curve(*xy_roll)

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

        return perp_node

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

    # def index(self, point):
    #     ind = np.isclose(self.xy, point.xy.reshape(2,1)).all(axis=0)
    #     if ind.any():
    #         return ind
    #     else:
    #         raise ValueError(f'{point} is not in {self}')


class Block:
    def __init__(self, x, y):
        """A 2-D grid of points in 2D space."""
        # Check input
        self.ni, self.nj = np.shape(x)
        util.check_vector(self.shape, x=x, y=y)
        self.xy = np.stack((x, y))

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
    def T(self):
        return Block(*self.xy.transpose(0, 2, 1))

    @property
    def shape(self):
        return (self.ni, self.nj)
