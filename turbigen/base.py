import numpy as np
from turbigen import util
import turbigen.yaml
import turbigen.average

logger = util.make_logger()


def concatenate(sd, axis=0):
    """Join a sequence of StructuredData along an axis."""
    out = sd[0].empty()
    out._data = np.concatenate([sdi._data for sdi in sd], axis=axis + 1)
    out._metadata = sd[0]._metadata
    return out


class dependent_property:
    """Decorator which returns a cached value if instance data unchanged."""

    def __init__(self, func):
        self._property_name = func.__name__
        self._func = func
        self.__doc__ = func.__doc__

    def __get__(self, instance, owner):
        del owner  # So linters do not find unused var
        if self._property_name not in instance._dependent_property_cache:
            instance._dependent_property_cache[self._property_name] = self._func(
                instance
            )
        return instance._dependent_property_cache[self._property_name]

    def __set__(self, instance, value):
        raise TypeError(f"Cannot assign to dependent property '{self._property_name}'")


class StructuredData:
    """Store array data with scalar metadata in one sliceable object."""

    _read_only = False
    _data_rows = ()

    def __init__(self, shape=(), order="C", typ=np.double):
        if not isinstance(shape, tuple):
            raise ValueError(f"Invalid input shape, got {shape}, expected a tuple")
        self._order = order
        if order == "C":
            self._data = np.full((self.nprop,) + shape, np.nan, order=order, dtype=typ)
        else:
            self._data = np.full(shape + (self.nprop,), np.nan, order=order, dtype=typ)
        self._metadata = {}
        self._dependent_property_cache = {}

    def to_dict(self):
        """Make a dictionary for this object."""
        out = {
            "class": self.__class__.__name__,
            "metadata": self._metadata.copy(),
            "data": self._data.tolist(),
            "data_rows": self._data_rows,
        }
        md = out["metadata"]
        for k in md:
            if isinstance(md[k], np.ndarray):
                md[k] = md[k].tolist()
        md.pop("abstract_state", None)
        return out

    @classmethod
    def from_dict(cls, d):
        data = np.array(d["data"])
        out = cls(data.shape)
        out._data = data
        out._metadata = d["metadata"]
        assert cls.__name__ == d["class"]
        return out

    def write(self, fname, mode="w"):
        """Save this object to a yaml file."""
        turbigen.yaml.write_yaml(self.to_dict(), fname, mode)

    def write_npz(self, fname, mode="w"):
        """Save this object to an npz file."""
        np.savez_compressed(fname, **self.to_dict())

    @classmethod
    def stack(cls, sd, axis=0):
        out = cls()
        out._data = np.stack([sdi._data for sdi in sd], axis=axis + 1)
        out._metadata = sd[0]._metadata
        return out

    def flip(self, axis):
        out = self.__class__()
        out._data = np.flip(self._data, axis=axis + 1)
        out._metadata = self._metadata
        return out

    def transpose(self, order):
        out = self.__class__()
        order1 = [
            0,
        ] + [o + 1 for o in order]
        out._data = np.transpose(self._data, order1)
        out._metadata = self._metadata
        return out

    def squeeze(self):
        out = self.__class__()
        out._data = np.squeeze(self._data)
        out._metadata = self._metadata
        return out

    def to_unstructured(self):
        """Make an unstructured view of these data."""
        # Make an empty object by calling constructor with no args
        out = self.__class__()
        # Insert unstructured version of current data and metadata
        out._data = self._data.reshape(self._data.shape[0], -1)
        out._metadata = self._metadata
        return out

    def triangulate(self):
        """Convert to a triangulated unstructured cut."""
        # Only work on 2D cuts
        assert self.ndim == 2
        #
        # Every structured quad becomes two triangles:
        #
        # i,j+1 +----+ i+1, j+1
        #       |A / |
        #       | / B|
        #   i,j +----+ i+1, j
        #
        # Determine new shape
        ni, nj = self.shape
        ntri = (ni - 1) * (nj - 1) * 2
        # Preallocate output data
        out = self.empty(shape=(ntri, 3))
        # Loop over quads
        ktri = 0
        for i in range(ni - 1):
            for j in range(nj - 1):
                data_tri_A = np.stack(
                    (
                        self._data[:, i, j],
                        self._data[:, i, j + 1],
                        self._data[:, i + 1, j + 1],
                    ),
                    axis=-1,
                )
                data_tri_B = np.stack(
                    (
                        self._data[:, i, j],
                        self._data[:, i + 1, j + 1],
                        self._data[:, i + 1, j],
                    ),
                    axis=-1,
                )
                out._data[:, ktri, :] = data_tri_A
                out._data[:, ktri + 1, :] = data_tri_B
                ktri += 2
        return out

    def __getitem__(self, key):
        # Special case for scalar indices
        if np.shape(key) == ():
            key = (key,)
        # Now prepend a slice for all properties to key
        key = (slice(None, None, None),) + key
        # Make an empty object by calling constructor with no args
        out = self.__class__()
        # Insert sliced data and all metadata
        out._data = self._data[key]
        out._metadata = self._metadata
        return out

    def _get_metadata_by_key(self, key, default=None):
        if default is None:
            return self._metadata[key]
        else:
            return self._metadata.get(key, default)

    def _set_metadata_by_key(self, key, val):
        if self._read_only:
            raise Exception(f"Cannot modify read-only {self}")
        else:
            self._metadata[key] = val
            self._dependent_property_cache.clear()

    def _lookup_index(self, key):
        if isinstance(key, tuple):
            ind = [self._data_rows.index(ki) for ki in key]
        else:
            ind = self._data_rows.index(key)
        return ind

    def _get_data_by_key(self, key):
        ind = self._lookup_index(key)
        if self._order == "C":
            return self._data[
                ind,
            ]
        else:
            return self._data[..., ind]

    def _set_data_by_key(self, key, val):
        if self._read_only:
            raise Exception(f"Cannot modify read-only {self}")
        else:
            ind = self._lookup_index(key)
            if np.shape(val) == (1,):
                if self._order == "C":
                    self._data[ind] = val[0]
                else:
                    self._data[..., ind] = val[0]
            else:
                if self._order == "C":
                    self._data[ind] = val
                else:
                    self._data[..., ind] = val
            self._dependent_property_cache.clear()

    def set_read_only(self):
        self._read_only = True
        return self

    def unset_read_only(self):
        self._read_only = False
        return self

    def copy(self):
        # Make an empty object by calling constructor with no args
        out = self.__class__()
        # Insert copies of current data and metadata
        out._data = self._data.copy()
        out._metadata = self._metadata.copy()
        return out

    def empty(self, shape=()):
        # Make an empty object by calling constructor with no args
        out = self.__class__()
        # Insert empty data and current metadata
        out._data = np.empty((self.nprop,) + shape)
        out._metadata = self._metadata
        return out

    def reshape(self, shape):
        self._data = self._data.reshape((self.nprop,) + shape)

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def nprop(self):
        return len(self._data_rows)

    @property
    def shape(self):
        return self._data.shape[1:]

    @property
    def size(self):
        return np.prod(self.shape)


class Kinematics:
    """Methods to calculate coordinates and velocities from instance attributes."""

    #
    # Independent coordinates
    #

    @property
    def x(self):
        """Axial coordinate [m]"""
        return self._get_data_by_key("x")

    @x.setter
    def x(self, value):
        self._set_data_by_key("x", value)

    @property
    def r(self):
        return self._get_data_by_key("r")

    @r.setter
    def r(self, value):
        self._set_data_by_key("r", value)

    @property
    def t(self):
        return self._get_data_by_key("t")

    @t.setter
    def t(self, value):
        self._set_data_by_key("t", value)

    @property
    def xrt(self):
        return self._get_data_by_key(("x", "r", "t"))

    @xrt.setter
    def xrt(self, value):
        return self._set_data_by_key(("x", "r", "t"), value)

    @property
    def xr(self):
        return self._get_data_by_key(("x", "r"))

    @xr.setter
    def xr(self, value):
        return self._set_data_by_key(("x", "r"), value)

    #
    # Independent velocities
    #

    @property
    def Vx(self):
        return self._get_data_by_key("Vx")

    @Vx.setter
    def Vx(self, value):
        self._set_data_by_key("Vx", value)

    @property
    def Vr(self):
        return self._get_data_by_key("Vr")

    @Vr.setter
    def Vr(self, value):
        self._set_data_by_key("Vr", value)

    @property
    def Vt(self):
        return self._get_data_by_key("Vt")

    @Vt.setter
    def Vt(self, value):
        self._set_data_by_key("Vt", value)

    @property
    def Vxrt(self):
        return self._get_data_by_key(("Vx", "Vr", "Vt"))

    @Vxrt.setter
    def Vxrt(self, value):
        self._set_data_by_key(("Vx", "Vr", "Vt"), value)

    @property
    def Omega(self):
        return self._get_data_by_key("Omega")

    @Omega.setter
    def Omega(self, Omega):
        self._set_data_by_key("Omega", Omega)

    #
    # Coordinaets
    #

    @dependent_property
    def rt(self):
        return self.r * self.t

    @dependent_property
    def xrrt(self):
        return np.concatenate((self.xr, np.expand_dims(self.rt, 0)), axis=0)

    @dependent_property
    def xyz(self):
        return np.stack((self.x, self.y, self.z))

    @dependent_property
    def yz(self):
        return np.stack((self.y, self.z))

    @dependent_property
    def y(self):
        return self.r * np.sin(self.t)

    @dependent_property
    def z(self):
        return self.r * np.cos(self.t)

    @dependent_property
    def vol(self):
        if not self.ndim == 3:
            raise Exception("Cell volume is only defined for 3D grids")

        # Put coord components on last dimension for numpy cross
        v = np.moveaxis(self.xyz, 0, -1)

        # Extract vertices A to H
        A = v[:-1, 1:, :-1, :]  # i j+1 k
        B = v[1:, 1:, :-1, :]  # i+1 j+1 k
        C = v[1:, 1:, 1:, :]  # i+1 j+1 k+1
        D = v[:-1, 1:, 1:, :]  # i j+1 k+1
        E = v[:-1, :-1, :-1, :]  # i j k
        F = v[1:, :-1, :-1, :]  # i+1 j k
        G = v[1:, :-1, 1:, :]  # i+1 j k+1
        H = v[:-1, :-1, 1:, :]  # i j k+1

        # Evaluate side vectors
        GA = A - G
        DB = B - D
        AC = C - A
        BE = E - B
        AF = F - A
        ED = D - E
        AH = H - A
        GB = B - G
        GC = C - G
        FC = C - F
        GE = E - G
        GF = F - G
        HF = F - H
        GD = D - G
        GH = H - G
        CH = H - C

        # For brevity
        def dot(a, b):
            # Dot product of [ni,nj,nk,3] along last axis
            return np.sum(a * b, axis=-1)

        cross = np.cross

        # Evaluate volume terms
        # Eqn. (14) Davies and Salmond (1985)
        V1 = dot(GA, cross(DB, AC) + cross(BE, AF) + cross(ED, AH))
        V2 = dot(GB, cross(DB, AC) + cross(GC, FC))
        V3 = dot(GE, cross(BE, AF) + cross(GF, HF))
        V4 = dot(GD, cross(ED, AH) + cross(GH, CH))
        vol = (V1 + V2 + V3 + V4) / 12.0

        return vol

        # dli = np.moveaxis(self.dli[:, :, :-1, :-1], 0, -1)
        # dlj = np.moveaxis(self.dlj[:, :-1, :, :-1], 0, -1)
        # dlk = np.moveaxis(self.dlk[:, :-1, :-1, :], 0, -1)
        # return np.sum(dlk * np.cross(dli, dlj),axis=-1)

        # # Numpy cross function assumes that the components are in last axis
        # xyz = np.moveaxis(self.xrrt, 0, -1).astype(np.float64)

        # # Vectors for cell sides
        # qi = np.diff(xyz[:, :-1, :-1, :], axis=0)
        # qj = np.diff(xyz[:-1, :, :-1, :], axis=1)
        # qk = np.diff(xyz[:-1, :-1, :, :], axis=2)

        # return np.sum(qk * np.cross(qi, qj), axis=-1)

    @dependent_property
    def Vxrt_rel(self):
        return np.stack((self.Vx, self.Vr, self.Vt_rel))

    @dependent_property
    def Vi_rel(self):
        """Velocity in grid i-direction."""

        # Edge-center vector for grid spacing
        qi_edge = np.diff(self.xrt, axis=1)

        # Multiply theta component by average radius on that face
        r_edge = 0.5 * (self.r[:-1, ...] + self.r[1:, ...])
        qi_edge[2] *= r_edge

        # Convert to node-centered
        qi_node = np.full(self.xrrt.shape, np.nan)
        qi_node[:, 0, ...] = qi_edge[:, 0, ...]
        qi_node[:, -1, ...] = qi_edge[:, -1, ...]
        qi_node[:, 1:-1, ...] = 0.5 * (qi_edge[:, :-1, ...] + qi_edge[:, 1:, ...])

        # Normalise to unit length
        qi_node /= util.vecnorm(qi_node)

        return (self.Vxrt_rel * qi_node).sum(axis=0)

    @dependent_property
    def surface_area(self):
        # if not self.ndim == 2:
        #     raise Exception("Surface area is only defined for 2D grids")
        # Numpy cross function assumes that the components are in last axis
        xyz = np.moveaxis(self.xyz, 0, -1).astype(np.float64)
        # Vectors for cell sides
        qi = np.diff(xyz[:, :-1, ...], axis=0)
        qj = np.diff(xyz[:-1, :, ...], axis=1)
        dA = np.cross(qi, qj)
        return dA

    @dependent_property
    def surface_area_xrrt(self):
        if not self.ndim == 2:
            raise Exception("Surface area is only defined for 2D grids")
        # Numpy cross function assumes that the components are in last axis
        xrrt = np.moveaxis(self.xrrt, 0, -1).astype(np.float64)
        # Vectors for cell sides
        qi = np.diff(xrrt[:, :-1, :], axis=0)
        qj = np.diff(xrrt[:-1, :, :], axis=1)
        dA = np.cross(qi, qj)
        return dA

    @dependent_property
    def dAt(self):
        if not self.ndim == 2:
            raise Exception("Surface area is only defined for 2D grids")

        # Numpy cross function assumes that the components are in last axis
        xrrt = np.moveaxis(self.xrrt, 0, -1).astype(np.float64)

        # Vectors for cell sides
        qi = np.diff(xrrt[:, :-1, :], axis=0)
        qj = np.diff(xrrt[:-1, :, :], axis=1)
        dA = np.cross(qi, qj)

        return dA[..., 2]

    @dependent_property
    def dlif(self):
        # Forward diagonal vector across i face
        # From j,k to j+1, k+1
        #
        # j+1 * *
        #      /
        # j   * *
        #    k  k+1
        dlif = self.xrt[:, :, 1:, 1:] - self.xrt[:, :, :-1, :-1]
        # Reference theta is at j,k
        dlif[2] *= self.r[:, 1:, 1:]
        return dlif

    @dependent_property
    def dlib(self):
        # Backward diagonal vector across i face
        # From j,k+1 to j+1, k ***via j,k**
        #
        # j+1 * *
        #     |\
        # j   *-*
        #    k  k+1

        # from k+1 to k, reference radius at j,k for
        dk = self.xrt[:, :, :-1, :-1] - self.xrt[:, :, :-1, 1:]
        dk[2] *= self.r[:, :-1, :-1]
        # from j to j+1, reference radius at e

        dlib = self.xrt[:, :, 1:, :-1] - self.xrt[:, :, :-1, 1:]
        return dlib

    @dependent_property
    def dljf(self):
        # Forward diagonal vector across j face
        # From i,k to i+1, k+1
        dljf = self.xrt[:, 1:, :, 1:] - self.xrt[:, :-1, :, :-1]
        dljf[2] *= self.r_face[1]
        return dljf

    @dependent_property
    def dljb(self):
        # Backward diagonal vector across i face
        # From i,k+1 to i+1,k
        dljb = self.xrt[:, 1:, :, :-1] - self.xrt[:, :-1, :, 1:]
        dljb[2] *= self.r_face[1]
        return dljb

    @dependent_property
    def dlkf(self):
        # Forward diagonal vector across k face
        # From i,j to i+1, j+1
        dlkf = self.xrt[:, 1:, 1:, :] - self.xrt[:, :-1, :-1, :]
        dlkf[2] *= self.r_face[2]
        return dlkf

    @dependent_property
    def dlkb(self):
        # Backward diagonal vector across k face
        # From i,j+1 to i+1,j
        dlkb = self.xrt[:, 1:, :-1, :] - self.xrt[:, :-1, 1:, :]
        dlkb[2] *= self.r_face[2]
        return dlkb

    @dependent_property
    def dli(self):
        return np.diff(self.xyz, axis=1)

    @dependent_property
    def dlj(self):
        return np.diff(self.xyz, axis=2)

    @dependent_property
    def dlk(self):
        return np.diff(self.xyz, axis=3)

    @dependent_property
    def dlmin_new(self):
        # Get face area magnitudes
        dAi = turbigen.util.vecnorm(self.dAi_new)
        dAj = turbigen.util.vecnorm(self.dAj_new)
        dAk = turbigen.util.vecnorm(self.dAk_new)

        # For each volume, take the minimum of the bounding length
        # scales for every coordinate direction
        vol = self.vol_new
        dli = np.minimum(vol / dAi[1:, :, :], vol / dAi[:-1, :, :])
        dlj = np.minimum(vol / dAj[:, 1:, :], vol / dAj[:, :-1, :])
        dlk = np.minimum(vol / dAk[:, :, 1:], vol / dAk[:, :, :-1])

        # Now take minimum of all directions
        dlmin = np.minimum(dli, dlj)
        dlmin = np.minimum(dlmin, dlk)

        return dlmin

    @dependent_property
    def dlmin(self):
        # Shortest side length

        # Vectors along all cell edges
        dli = turbigen.util.vecnorm(self.dli)
        dlj = turbigen.util.vecnorm(self.dlj)
        dlk = turbigen.util.vecnorm(self.dlk)

        # Shortest side length on i-faces
        dli = np.min(
            np.stack(
                (dli[:, :-1, :-1], dli[:, :-1, 1:], dli[:, 1:, :-1], dli[:, :-1, :-1])
            ),
            axis=0,
        )

        # Shortest side length on j-faces
        dlj = np.min(
            np.stack(
                (dlj[:-1, :, :-1], dlj[:-1, :, 1:], dlj[1:, :, :-1], dlj[:-1, :, :-1])
            ),
            axis=0,
        )

        # Shortest side length on k-faces
        dlk = np.min(
            np.stack(
                (dlk[:-1, :-1, :], dlk[:-1, 1:, :], dlk[1:, :-1, :], dlk[:-1, :-1, :])
            ),
            axis=0,
        )

        # Shortest side length for each volume
        dlmin = np.minimum(dli, dlj)
        dlmin = np.minimum(dlmin, dlk)
        return dlmin

    @dependent_property
    def dAi(self):
        # Vector area for i=const faces
        if not self.ndim == 3:
            raise Exception("Face area is only defined for 3D grids")

        # # Numpy cross function assumes that the components are in last axis
        # dlif = np.moveaxis(self.dlif, 0, -1)
        # dlib = np.moveaxis(self.dlib, 0, -1)
        # dAi = -np.moveaxis(np.cross(dlif, dlib), -1, 0) * 0.5

        # Define the four vertices OBAC
        #    B      A
        #     *----*
        #  ^  |    |
        #  k  *----*
        #    O      C
        #      j>

        O = self.xrt[:, :, :-1, :-1]
        A = self.xrt[:, :, 1:, 1:]
        B = self.xrt[:, :, :-1, 1:]
        C = self.xrt[:, :, 1:, :-1]

        # Form three vectors AO, BO, CO
        AO = A - O
        AO[2] *= A[1]  # Theta reference radius at A
        BO = B - O
        BO[2] *= B[1]  # Theta reference radius at B
        CO = C - O
        CO[2] *= C[1]  # Theta reference radius at C

        return 0.5 * np.cross(AO, BO - CO, axis=0)

    @dependent_property
    def dAj(self):
        # Vector area for i=const faces
        if not self.ndim == 3:
            raise Exception("Face area is only defined for 3D grids")

        # # Numpy cross function assumes that the components are in last axis
        # dljf = np.moveaxis(self.dljf, 0, -1)
        # dljb = np.moveaxis(self.dljb, 0, -1)
        # dAj = np.moveaxis(np.cross(dljf, dljb), -1, 0) * 0.5

        # Define the four vertices OBAC
        #    B      A
        #     *----*
        #  ^  |    |
        #  k  *----*
        #    O      C
        #      i>

        O = self.xrt[:, :-1, :, :-1]
        A = self.xrt[:, 1:, :, 1:]
        B = self.xrt[:, :-1, :, 1:]
        C = self.xrt[:, 1:, :, :-1]

        # Form three vectors AO, BO, CO
        AO = A - O
        AO[2] *= A[1]  # Theta reference radius at A
        BO = B - O
        BO[2] *= B[1]  # Theta reference radius at B
        CO = C - O
        CO[2] *= C[1]  # Theta reference radius at C

        return -0.5 * np.cross(AO, BO - CO, axis=0)

    @dependent_property
    def dAk(self):
        # Vector area for i=const faces
        if not self.ndim == 3:
            raise Exception("Face area is only defined for 3D grids")

        # Define the four vertices OBAC
        #    B      A
        #     *----*
        #  ^  |    |
        #  j  *----*
        #    O      C
        #      i>

        xyz = self.xyz
        O = xyz[:, :-1, :-1, :]
        A = xyz[:, 1:, 1:, :]
        B = xyz[:, :-1, 1:, :]
        C = xyz[:, 1:, :-1, :]

        # O = self.xrt[:, :-1, :-1, :]
        # A = self.xrt[:, 1:, 1:, :]
        # B = self.xrt[:, :-1, 1:, :]
        # C = self.xrt[:, 1:, :-1, :]

        # Form three vectors AO, BO, CO
        AO = A - O
        BC = B - C
        dAk = -0.5 * np.cross(AO, BC, axis=0)

        # convert to polar
        _, _, tk = self.t_face
        cost = np.cos(tk)
        sint = np.sin(tk)
        dAkx, dAky, dAkz = dAk
        dAkr = -dAky * sint - dAkz * cost
        dAkt = dAky * cost - dAkz * sint

        return np.stack((dAkx, dAkr, dAkt))

    @dependent_property
    def r_face(self):
        return turbigen.util.node_to_face3(self.r)

    @dependent_property
    def r_cell(self):
        return turbigen.util.node_to_cell(self.r)

    @dependent_property
    def t_face(self):
        return turbigen.util.node_to_face3(self.t)

    @dependent_property
    def rt_face(self):
        return turbigen.util.node_to_face3(self.rt)

    @dependent_property
    def x_face(self):
        return turbigen.util.node_to_face3(self.x)

    @dependent_property
    def dAi_new(self):
        # Vector area for i=const faces, Gauss' theorem method
        if not self.ndim == 3:
            raise Exception("Face area is only defined for 3D grids")

        # Define four vertices ABCD
        #    B      C
        #     *----*
        #  ^  |    |
        #  k  *----*
        #    A      D
        #      j>
        #
        v = self.xrrt
        A = v[:, :, :-1, :-1]
        B = v[:, :, :-1, 1:]
        C = v[:, :, 1:, 1:]
        D = v[:, :, 1:, :-1]

        return util.dA_Gauss(A, B, C, D)

    @dependent_property
    def dAi_cr(self):
        # Vector area for i=const faces, Gauss' theorem method
        if not self.ndim == 3:
            raise Exception("Face area is only defined for 3D grids")

        # Define four vertices ABCD
        #    B      C
        #     *----*
        #  ^  |    |
        #  k  *----*
        #    A      D
        #      j>
        #
        v = self.xyz
        A = v[:, :, :-1, :-1]
        B = v[:, :, :-1, 1:]
        C = v[:, :, 1:, 1:]
        D = v[:, :, 1:, :-1]

        dA = util.dA_Gauss(A, B, C, D)
        t = self.t_face[0]
        return util.cart_to_pol(dA, t)

    @dependent_property
    def dAj_new(self):
        # Vector area for j=const faces, Gauss' theorem method
        if not self.ndim == 3:
            raise Exception("Face area is only defined for 3D grids")

        # Define four vertices ABCD
        #    B      C
        #     *----*
        #  ^  |    |
        #  k  *----*
        #    A      D
        #      i>
        #
        v = self.xrrt
        A = v[:, :-1, :, :-1]
        B = v[:, :-1, :, 1:]
        C = v[:, 1:, :, 1:]
        D = v[:, 1:, :, :-1]

        return -util.dA_Gauss(A, B, C, D)

    @dependent_property
    def dAj_cr(self):
        # Vector area for j=const faces, Gauss' theorem method
        if not self.ndim == 3:
            raise Exception("Face area is only defined for 3D grids")

        # Define four vertices ABCD
        #    B      C
        #     *----*
        #  ^  |    |
        #  k  *----*
        #    A      D
        #      i>
        #
        v = self.xyz
        A = v[:, :-1, :, :-1]
        B = v[:, :-1, :, 1:]
        C = v[:, 1:, :, 1:]
        D = v[:, 1:, :, :-1]

        dA = util.dA_Gauss(A, B, C, D)
        t = self.t_face[1]
        return util.cart_to_pol(dA, t)

    @dependent_property
    def dAk_new(self):
        # Vector area for k=const faces, Gauss' theorem method
        if not self.ndim == 3:
            raise Exception("Face area is only defined for 3D grids")

        # Define four vertices ABCD
        #    B      C
        #     *----*
        #  ^  |    |
        #  k  *----*
        #    A      D
        #      i>
        #
        v = self.xrrt
        A = v[:, :-1, :-1, :]
        B = v[:, :-1, 1:, :]
        C = v[:, 1:, 1:, :]
        D = v[:, 1:, :-1, :]

        return util.dA_Gauss(A, B, C, D)

    @dependent_property
    def dAk_cr(self):
        # Vector area for k=const faces, Gauss' theorem method
        if not self.ndim == 3:
            raise Exception("Face area is only defined for 3D grids")

        # Define four vertices ABCD
        #    B      C
        #     *----*
        #  ^  |    |
        #  k  *----*
        #    A      D
        #      i>
        #
        v = self.xyz
        A = v[:, :-1, :-1, :]
        B = v[:, :-1, 1:, :]
        C = v[:, 1:, 1:, :]
        D = v[:, 1:, :-1, :]

        dA = util.dA_Gauss(A, B, C, D)
        t = self.t_face[2]
        return util.cart_to_pol(dA, t)

    @dependent_property
    def vol_new(self):
        # Volume
        if not self.ndim == 3:
            raise Exception("Face area is only defined for 3D grids")

        # Get face-centered coordinates
        xi, xj, xk = self.x_face
        ri, rj, rk = self.r_face
        rti, rtj, rtk = self.rt_face
        Fi = np.stack((xi, ri / 2.0, rti))
        Fj = np.stack((xj, rj / 2.0, rtj))
        Fk = np.stack((xk, rk / 2.0, rtk))
        dAi = self.dAi_new
        dAj = self.dAj_new
        dAk = self.dAk_new

        # Volume by Gauss' theorem
        Fisum = np.diff(np.sum(Fi * dAi, axis=0), axis=0)
        Fjsum = np.diff(np.sum(Fj * dAj, axis=0), axis=1)
        Fksum = np.diff(np.sum(Fk * dAk, axis=0), axis=2)
        vol = Fisum + Fjsum + Fksum

        return vol / 3.0

    @dependent_property
    def flux_all(self):
        return np.stack(
            (
                self.flux_mass,
                self.flux_xmom,
                self.flux_rmom,
                self.flux_rtmom,
                self.flux_energy,
            )
        )

    @dependent_property
    def spf(self):
        if self.ndim == 1:
            span = util.cum_arc_length(self.xr, axis=1)
            spf = span / np.max(span, axis=0, keepdims=True)
        else:
            span = util.cum_arc_length(self.xr, axis=2)
            spf = span / np.max(span, axis=1, keepdims=True)
        return spf

    @dependent_property
    def zeta(self):
        """Arc length along each i gridline."""
        return util.cum_arc_length(self.xyz, axis=1)

    @dependent_property
    def tri_area(self):
        if not self.shape[1] == 3:
            raise Exception("This is not a triangulated cut.")

        # Vectors for each side
        qAB = self.xrrt[..., 1] - self.xrrt[..., 0]
        qAC = self.xrrt[..., 2] - self.xrrt[..., 0]

        # Numpy cross function assumes that the components are in last axis
        qAB = np.moveaxis(qAB, 0, -1).astype(np.float64)
        qAC = np.moveaxis(qAC, 0, -1).astype(np.float64)

        return 0.5 * np.cross(qAC, qAB).transpose(1, 0)

    def get_mpl_triangulation(self):
        """Generate a matplotlib-compatible triangulation for an unstructured cut."""

        # Check we have a triangulated shape (ntri, 3)
        try:
            ntri, ndim = self.shape
            assert ndim == 3
        except Exception:
            raise Exception("This is not a triangulated cut.")

        # Reshape to a 1D array
        C = self.to_unstructured()

        # Because we store all three vertices for every triangle, many vertices are repeated
        # Matplotlib prefers without repeats
        # So find the 1D indices of unique coordiates only
        _, iunique, triangles = np.unique(
            C.xrt,
            axis=1,
            return_index=True,
            return_inverse=True,
        )

        # Only keep unique points
        C = C[(iunique,)]

        # The triangles are indices into the 1D unique data that
        # reconstruct the original (ntri, 3) data
        triangles = triangles.reshape(-1, 3)

        return C, triangles

    def repeat_pitchwise(self, N, axis=0):
        """Replicate the data in pitchwise direction."""

        # Make a list of copies of this cut with different theta
        C_all = []
        for i in range(N):
            Ci = self.copy()
            Ci.t += self.pitch * i
            C_all.append(Ci)

        # Join the copies together
        C_all = concatenate(C_all, axis=axis)

        return C_all

    @dependent_property
    def U(self):
        return self.r * self.Omega

    @dependent_property
    def V(self):
        return util.vecnorm(self.Vxrt)

    @dependent_property
    def Vm(self):
        return util.vecnorm(self.Vxrt[:2])

    @dependent_property
    def Vt_rel(self):
        return self.Vt - self.U

    @dependent_property
    def V_rel(self):
        return np.sqrt(self.Vm**2.0 + self.Vt_rel**2.0)

    #
    # Angles
    #

    @dependent_property
    def Alpha_rel(self):
        return np.degrees(np.arctan2(self.Vt_rel, self.Vm))

    @dependent_property
    def Alpha(self):
        return np.degrees(np.arctan2(self.Vt, self.Vm))

    @dependent_property
    def Beta(self):
        return np.degrees(np.arctan2(self.Vr, self.Vx))

    @dependent_property
    def tanBeta(self):
        return self.Vr / self.Vx

    @dependent_property
    def tanAlpha(self):
        return self.Vt / self.Vm

    @dependent_property
    def tanAlpha_rel(self):
        return self.Vt_rel / self.Vm

    @dependent_property
    def cosBeta(self):
        return self.Vx / self.Vm

    @dependent_property
    def cosAlpha(self):
        return self.Vm / self.V

    @dependent_property
    def cosAlpha_rel(self):
        return self.Vm / self.V_rel

    #
    # Misc
    #

    @dependent_property
    def rpm(self):
        return self.Omega / 2.0 / np.pi * 60.0


class Composites:
    """Methods for properties depending on thermodynamic and velocity fields."""

    @dependent_property
    def conserved(self):
        return np.stack((self.rho, self.rhoVx, self.rhoVr, self.rhorVt, self.rhoe))

    @dependent_property
    def rhoVx(self):
        return self.rho * self.Vx

    @dependent_property
    def rhoVr(self):
        return self.rho * self.Vr

    @dependent_property
    def rhoVt(self):
        return self.rho * self.Vt

    @dependent_property
    def rhorVt(self):
        return self.r * self.rhoVt

    @dependent_property
    def rVt(self):
        return self.r * self.Vt

    @dependent_property
    def rhoe(self):
        return self.rho * self.e

    @dependent_property
    def e(self):
        return self.u + 0.5 * self.V**2.0

    @dependent_property
    def Ma(self):
        return self.V / self.a

    @dependent_property
    def Ma_rel(self):
        return self.V_rel / self.a

    @dependent_property
    def Mam(self):
        return self.Vm / self.a

    @dependent_property
    def I(self):
        return self.h + 0.5 * self.V**2.0 - self.U * self.Vt

    @dependent_property
    def stagnation(self):
        return self.to_stagnation(self.Ma).set_read_only()

    @dependent_property
    def stagnation_rel(self):
        return self.to_stagnation(self.Ma_rel).set_read_only()

    @property
    def Po(self):
        """Stagnation pressure [Pa]."""
        return self.stagnation.P

    @property
    def To(self):
        return self.stagnation.T

    @property
    def ao(self):
        return self.stagnation.a

    @property
    def ho(self):
        # We can directly use static enthalpy and velocity
        return self.h + 0.5 * self.V**2.0

    @property
    def halfVsq(self):
        return 0.5 * self.V**2.0

    @property
    def Po_rel(self):
        return self.stagnation_rel.P

    @property
    def To_rel(self):
        return self.stagnation_rel.T

    @property
    def ho_rel(self):
        # We can directly use static enthalpy and velocity
        return self.h + 0.5 * self.V_rel**2.0

    @dependent_property
    def Vy(self):
        cost = np.cos(self.t)
        sint = np.sin(self.t)
        return self.Vr * cost - self.Vt * sint

    @dependent_property
    def Vz(self):
        cost = np.cos(self.t)
        sint = np.sin(self.t)
        return -self.Vr * sint - self.Vt * cost

    @dependent_property
    def P_rot(self):
        # Rotary static pressure
        if self.Omega.any():
            S = self.copy()
            # In rotating frame
            # Replace horel with rothalpy
            # i.e. subtract blade speed dyn head from h
            S.set_h_s(self.h - 0.5 * self.U**2, self.s)
            P = S.P
        else:
            # Just use normal static pressure in stationary frame
            P = self.P
        return P

    #
    # Fluxes
    #

    @dependent_property
    def flux_mass(self):
        # Mass fluxes in x and r dirns
        return np.stack((self.rhoVx, self.rhoVr, self.rhoVt))

    @dependent_property
    def flux_xmom(self):
        # Axial momentum fluxes in x and r dirns
        return np.stack(
            (self.rhoVx * self.Vx + self.P, self.rhoVr * self.Vx, self.rhoVt * self.Vx)
        )

    @dependent_property
    def flux_rmom(self):
        # Radial momentum fluxes in x and r dirns
        return np.stack(
            (
                self.rhoVx * self.Vr,
                self.rhoVr * self.Vr + self.P,
                self.rhoVt * self.Vr,
            )
        )

    @dependent_property
    def flux_rtmom(self):
        # Moment of angular momentum fluxes in x and r dirns
        return np.stack(
            (
                self.Vx * self.rhorVt,
                self.Vr * self.rhorVt,
                self.Vt * self.rhorVt + self.r * self.P,
            )
        )

    @dependent_property
    def flux_rothalpy(self):
        # Stagnation rothalpy fluxes in x an r dirns
        return self.flux_mass * self.I

    @dependent_property
    def flux_energy(self):
        # Stagnation entahlpy fluxes in x an r dirns
        return np.stack(
            (
                self.rhoVx * self.ho,
                self.rhoVr * self.ho,
                self.rhoVt * self.ho + self.Omega * self.r * self.P,
            )
        )

    @dependent_property
    def source_all(self):
        source_rtmom = (self.P + self.rho * self.Vt**2) / self.r
        Z = np.zeros_like(source_rtmom)
        return np.stack(
            (
                Z,  # mass
                Z,  # xmom
                Z,  # rmom
                source_rtmom,  # rtmom
                Z,  # energy
            )
        )

    @dependent_property
    def flux_entropy(self):
        # Mass fluxes in x and r dirns
        return self.flux_mass * self.s

    def meridional_slice(self, xrc):
        """Slice a block cut using a meridional curve."""

        # Get signed distance
        dist = util.signed_distance_piecewise(xrc, self.xr)

        # Get j indices above slice
        jcut = np.argmax(dist > 0, axis=1, keepdims=True) - 1

        # Preallocate
        data = self._data
        nv, ni, nj, nk = self._data.shape
        data_cut = np.empty(
            (
                nv,
                ni,
                nk,
            )
        )
        for i in range(ni):
            jnow = jcut[i]
            dist_now = dist[i, (jnow, jnow + 1), :]
            frac = -dist_now[0] / (dist_now[1] - dist_now[0])
            data_cut[:, i, :] = (
                data[:, i, jnow, :]
                + (data[:, i, jnow + 1, :] - data[:, i, jnow, :]) * frac
            )[:, 0, 0, :]

        out = self.empty(shape=(ni, nk))
        out._data = data_cut
        out._metadata = self._metadata
        return out

    def mix_out(self):
        """Mix out the cut to a scalar state, conserving mass, momentum and energy."""
        return turbigen.average.mix_out(self)

    def set_conserved(self, conserved):
        rho, *rhoVxrt, rhoe = conserved
        Vxrt = rhoVxrt / rho
        Vxrt[2] /= self.r
        self.Vxrt = Vxrt
        u = rhoe / rho - 0.5 * self.V**2
        self.set_rho_u(rho, u)

    def area_average(self):
        dA = np.linalg.norm(self.surface_area[:, :, 0, :], axis=-1, ord=2)
        A = np.sum(dA)
        conserved = np.moveaxis(self.conserved, -1, 1)
        xrt = np.moveaxis(self.xrt, -1, 1)
        conserved_av = (
            np.sum(dA * turbigen.util.node_to_face(conserved), axis=(-2, -1)) / A
        )
        xrt_av = np.sum(dA * turbigen.util.node_to_face(xrt), axis=(-2, -1)) / A

        F = self.empty((conserved_av.shape[1],))
        F.xrt = xrt_av
        F.set_conserved(conserved_av)

        return F

    def mix_out_pitchwise(self):
        """Mix out in the pitchwise direction, to a spanwise profile."""

        nj = self.shape[1]
        cuts = []
        spf = np.empty(nj - 1)
        for j in range(nj - 1):
            spf[j] = self.spf[:, j : j + 2, :].mean()
            cut_now = self[:, j : j + 2, :]
            try:
                cuts.append(cut_now.mix_out()[0])
            except Exception:
                cuts.append(cut_now.empty())
        Call = self.stack(cuts)
        return spf, Call

    @dependent_property
    def i_stag(self):
        """i-index of stagnation point."""

        if not self.ndim == 2:
            raise Exception(
                "Can only find stagnation point on 2D cuts; "
                f"this cut has shape {self.shape}"
            )

        # Use rotary static pressure to take out centrifugal pressure gradient
        P = self.P_rot

        # Extract surface distance, normalise to [-1,1] on each j-line
        z = self.zeta / np.ptp(self.zeta, axis=0) * 2.0 - 1.0

        # Find pressure maxima
        # This must be a loop over j because there can be a different number of
        # turning points at each spanwise lnocation
        _, nj = self.shape
        istag = np.full((nj,), 0, dtype=int)
        for j in range(nj):
            # Calculate gradient and curvature
            dP = np.diff(P[:, j])

            # Indices of downward zero crossings of pressure derivative
            izj = np.where(np.diff(np.sign(dP[:-2])) < 0.0)[0] + 1

            # Only keep maxima close to LE
            izj = izj[np.abs(z[izj, j]) < 0.2]

            # Now take the candiate point with maximum pressure
            try:
                istag[j] = izj[np.argsort(P[izj, j])][-1]
            except Exception:
                istag[j] = 0

        return istag

    @dependent_property
    def zeta_stag(self):
        """Surface distance along i-line with origin at stagnation point."""

        _, nj = self.shape
        zstag = np.full(
            (
                1,
                nj,
            ),
            np.nan,
        )
        istag = self.i_stag
        for j in range(nj):
            zstag[0, j] = self.zeta[istag[j], j]

        return self.zeta - zstag

    @dependent_property
    def xrt_stag(self):
        """Coordinates of the stagnation point at all j indices."""
        _, nj = self.shape
        xrt_stag = np.full(
            (
                3,
                nj,
            ),
            np.nan,
        )
        for j in range(nj):
            xrt_stag[:, j] = self.xrt[:, self.i_stag[j], j]
        return xrt_stag

    @dependent_property
    def fluxes(self):
        return np.stack(
            (
                self.rhoVx,
                self.rhoVx * self.Vx + self.P,
                self.rhoVx * self.Vr,
                self.rhoVx * self.Vt,
                self.rhoVx * self.ho,
            )
        )

    @dependent_property
    def bcond(self):
        return np.stack(
            (
                self.ho,
                self.s,
                self.tanAlpha,
                self.tanBeta,
                self.P,
            )
        )

    @dependent_property
    def drhoe_drho_P(self):
        return self.e + self.rho * self.dudrho_P

    @dependent_property
    def drhoe_dP_rho(self):
        return self.rho * self.dudP_rho

    @dependent_property
    def prim(self):
        return np.stack((self.rho, self.Vx, self.Vr, self.Vt, self.P))

    def set_prim(self, prim):
        rho, *Vxrt, P = prim
        self.set_P_rho(P, rho)
        self.Vxrt = Vxrt
        return self

    def primitive_to_conserved(self):
        """Get a matrix at every node that converts linear pertubations in
        primitive variables [rho, Vx, Vr, Vt, P]
        to perturbations in
        conserved variables [rho, rhoVx, rhoVr, rhorVt, rhoe].

        Returns
        -------
        C: (npts, 5, 5) array

        """

        if not self.ndim == 1:
            raise Exception(f"Need a flattened flow to do this")

        Z = np.zeros(self.shape)
        one = np.ones(self.shape)
        C = np.stack(
            (
                (one, self.Vx, self.Vr, self.rVt, self.drhoe_drho_P),  # d/drho
                (Z, self.rho, Z, Z, self.rhoVx),  # d/dVx
                (Z, Z, self.rho, Z, self.rhoVr),  # d/dVr
                (Z, Z, Z, self.r * self.rho, self.rhoVt),  # d/dVt
                (Z, Z, Z, Z, self.drhoe_dP_rho),  # d/dP
            )
        ).T
        return C

    def conserved_to_primitive(self):
        """Get a matrix at every node that converts linear pertubations in
        conserved variables [rho, rhoVx, rhoVr, rhorVt, rhoe].
        to perturbations in
        primitive variables [rho, Vx, Vr, Vt, P]

        Returns
        -------
        Cinv: (npts, 5, 5) array

        """
        return np.linalg.inv(self.primitive_to_conserved())

    def primitive_to_chic(self):
        """Get a matrix at every node that converts linear pertubations in
        primitive variables [rho, Vx, Vr, Vt, P]
        to perturbations in
        characteristic variables
        [dp+rho*a*dVx, dp-rho*a*dVx, rho*a*dVr, rho*a*dVt, dp - (a^2)*drho].
        [upstream acoustic, downstream acoustic, r-mom, t-mom, entropy wave]

        Returns
        -------
        B: (npts, 5, 5) array

        """

        if not self.ndim == 1:
            raise Exception(f"Need a flattened flow to do this")

        Z = np.zeros(self.shape)
        one = np.ones(self.shape)
        rhoa = self.rho * self.a
        B = np.stack(
            (
                (Z, Z, Z, Z, -(self.a**2)),  # d/rho
                (rhoa, -rhoa, Z, Z, Z),  # d/dVx
                (Z, Z, rhoa, Z, Z),  # d/dVr
                (Z, Z, Z, rhoa, Z),  # d/dVt
                (one, one, Z, Z, one),  # d/dP
            )
        ).T
        return B

    def chic_to_primitive(self):
        """Get a matrix at every node that converts linear pertubations in
        characteristic variables
        [dp+rho*a*dVx, dp-rho*a*dVx, rho*a*dVr, rho*a*dVt, dp - (a^2)*drho].
        [upstream acoustic, downstream acoustic, r-mom, t-mom, entropy wave]
        to perturbations in
        primitive variables [rho, Vx, Vr, Vt, P]

        Returns
        -------
        Binv: (npts, 5, 5) array
        """
        return np.linalg.inv(self.primitive_to_chic())

    def primitive_to_flux(self):
        """Get a matrix at every node that converts linear pertubations in
        primitive variables [rho, Vx, Vr, Vt, P]
        to perturbations in
        flux variables
        [rhoVx, rhoVx^2+P, rhoVxVr, rhoVxrVt, rhoVx*ho].

        Returns
        -------
        A: (npts, 5, 5) array

        """

        if not self.ndim == 1:
            raise Exception(f"Need a flattened flow to do this")

        Z = np.zeros(self.shape)
        one = np.ones(self.shape)
        VxVr = self.Vx * self.Vr
        VxrVt = self.Vx * self.rVt
        VxVx = self.Vx**2
        dE_drho = self.Vx * self.ho + self.rhoVx * self.dhdrho_P
        dE_dVx = self.rho * self.ho + self.rhoVx * self.Vx
        A = np.stack(
            (
                (self.Vx, VxVx, VxVr, VxrVt, dE_drho),  # d/rho
                (self.rho, 2.0 * self.rhoVx, self.rhoVr, self.rhorVt, dE_dVx),  # d/dVx
                (Z, Z, self.rhoVx, Z, self.rhoVx * self.Vr),  # d/dVr
                (Z, Z, Z, self.rhoVx * self.r, self.rhoVx * self.Vt),  # d/dVt
                (Z, one, Z, Z, self.rhoVx * self.dhdP_rho),  # d/dP
            )
        ).T
        return A

    def flux_to_primitive(self):
        """Get a matrix at every node that converts linear pertubations in
        flux variables
        [rhoVx, rhoVx^2+P, rhoVxVr, rhoVxrVt, rhoVx*ho].
        to perturbations in
        primitive variables [rho, Vx, Vr, Vt, P]

        Returns
        -------
        Ainv: (npts, 5, 5) array
        """
        return np.linalg.inv(self.primitive_to_flux())

    def primitive_to_bcond(self):
        """Get a matrix at every node that converts linear pertubations in
        primitive variables [rho, Vx, Vr, Vt, P]
        to perturbations in
        boundary condition variables
        [ho, s, tanAlpha, tanBeta, P].

        Returns
        -------
        Y: (npts, 5, 5) array

        """

        if not self.ndim == 1:
            raise Exception(f"Need a flattened flow to do this")

        Z = np.zeros(self.shape)
        one = np.ones(self.shape)

        dtanAl_dVx = -self.tanAlpha * self.Vx / self.Vm**2
        dtanAl_dVr = -self.tanAlpha * self.Vr / self.Vm**2
        dtanAl_dVt = 1.0 / self.Vm

        dtanBe_dVx = -self.Vr / self.Vx**2
        dtanBe_dVr = 1.0 / self.Vx

        Y = np.stack(
            (
                (self.dhdrho_P, self.dsdrho_P, Z, Z, Z),  # d/rho
                (self.Vx, Z, dtanAl_dVx, dtanBe_dVx, Z),  # d/dVx
                (self.Vr, Z, dtanAl_dVr, dtanBe_dVr, Z),  # d/dVr
                (self.Vt, Z, dtanAl_dVt, Z, Z),  # d/dVt
                (self.dhdP_rho, self.dsdP_rho, Z, Z, one),  # d/dP
            )
        ).T
        return Y

    def bcond_to_primitive(self):
        """Get a matrix at every node that converts linear pertubations in
        boundary condition variables
        [ho, s, tanAlpha, tanBeta, P].
        to perturbations in
        primitive variables [rho, Vx, Vr, Vt, P]

        Returns
        -------
        Yinv: (npts, 5, 5) array

        """

        return np.linalg.inv(self.primitive_to_bcond())

    def bcond_to_conserved(self, chics):
        # bcond to primitive
        Yinv = self.bcond_to_primitive()

        # primitive to chic
        B = self.primitive_to_chic()

        # Selectors for direction of chics
        # [dp+rho*a*dVx, dp-rho*a*dVx, rho*a*dVr, rho*a*dVt, dp - (a^2)*drho].
        # [upstream acoustic, downstream acoustic, r-mom, t-mom, entropy wave]
        ni = B.shape[0]
        if chics == "all":
            D = np.tile(np.eye(5), (ni, 1, 1))
        elif chics == "up":
            D = np.tile(np.diag([-1.0, 0.0, 0.0, 0.0, 0.0]), (ni, 1, 1))
        elif chics == "dn":
            D = np.tile(np.diag([0.0, -1.0, 1.0, 1.0, 1.0]), (ni, 1, 1))

        # chic to primitive
        Binv = self.chic_to_primitive()

        # primitive to conserved
        C = self.primitive_to_conserved()

        # Assemble the complete transformations
        CBinv = C @ Binv
        BYinv = B @ Yinv
        Q = CBinv @ D @ BYinv

        return Q

    def resolve_meridional(self, psi):
        """Replace axial and radial components by resolving at angle to axial dirn."""
        cospsi = util.cosd(psi)
        sinpsi = util.sind(psi)
        Vn = self.Vx * cospsi + self.Vr * sinpsi
        Vs = -self.Vx * sinpsi + self.Vr * cospsi
        self.Vx = Vn
        self.Vr = Vs


class MeanLine:
    """Encapsulate flow and geometry on a nomial mean streamsurface."""

    @property
    def A(self):
        return self._get_data_by_key("A")

    @A.setter
    def A(self, val):
        return self._set_data_by_key("A", val)

    @property
    def Nb(self):
        return self._get_data_by_key("Nb")

    @Nb.setter
    def Nb(self, val):
        return self._set_data_by_key("Nb", val)

    @classmethod
    def from_states(cls, rrms, A, Omega, Vxrt, S, Nb=None):
        """Construct a mean-line from generic state objects."""

        # Preallocate class of correct shape
        F = cls(S.shape)

        # Reference the metadata (which defines fluid properties)
        F._metadata = S._metadata

        # Set mean-line variables
        F.r = rrms
        F.A = A
        F.Vxrt = Vxrt
        F.Omega = Omega
        F.set_P_T(S.P, S.T)

        if Nb is not None:
            F.Nb = Nb

        return F

    def interpolate_guess(self, ann):
        # Get coordinates along mean-line
        npts = 100
        sg = np.linspace(0.0, ann._mctl[-1], npts)
        xg, rg = ann.evaluate_xr(sg, 0.5)

        # Get variations in thermodynamic state at inlet,exit and row boundaries
        # From the mean-line
        ro_mid = np.pad(self.rho, 1, "edge")
        u_mid = np.pad(self.u, 1, "edge")
        Vx_mid = np.pad(self.Vx, 1, "edge")
        Vr_mid = np.pad(self.Vr, 1, "edge")
        Vt_mid = np.pad(self.Vt, 1, "edge")

        # Interpolate flow properties linearly along mean-line
        rog = np.interp(sg, ann._mctl, ro_mid)
        ug = np.interp(sg, ann._mctl, u_mid)
        Vxg = np.interp(sg, ann._mctl, Vx_mid)
        Vrg = np.interp(sg, ann._mctl, Vr_mid)
        Vtg = np.interp(sg, ann._mctl, Vt_mid)

        Fg = self.empty(shape=(npts,)).set_rho_u(rog, ug)
        Fg.xr = np.stack((xg, rg))
        Fg.Vxrt = np.stack((Vxg, Vrg, Vtg))

        return Fg

    #
    # Override Omega methods to make it a vector
    #

    @staticmethod
    def _check_vectors(*args):
        """Ensure that some inputs have same shape and len multiple of 2."""
        shp = np.shape(args[0])
        assert np.mod(shp[0], 2) == 0
        assert len(shp) == 1
        for arg in args:
            assert np.shape(arg) == shp

    @property
    def rrms(self):
        return self.r

    @dependent_property
    def mdot(self):
        return self.rho * self.Vm * self.A

    @dependent_property
    def Q(self):
        return self.Vm * self.A

    @dependent_property
    def span(self):
        return self.A / 2.0 / np.pi / self.rmid

    @dependent_property
    def rmid(self):
        return (self.rtip + self.rhub) * 0.5

    @dependent_property
    def rhub(self):
        return np.sqrt(2.0 * self.rrms**2.0 - self.rtip**2.0)

    @dependent_property
    def rtip(self):
        return np.sqrt(self.A * self.cosBeta / 2.0 / np.pi + self.rrms**2.0)

    @dependent_property
    def htr(self):
        return self.rhub / self.rtip

    @dependent_property
    def Aflow(self):
        return self.A * self.cosAlpha_rel

    @dependent_property
    def ARflow(self):
        return self.Aflow[1:] / self.Aflow[:-1]

    def check(self):
        # """Assert that conserved quantities are in fact conserved"""

        logger.info("Checking mean-line conservation...")

        check_failed = False

        # Check mass is conserved
        rtol = 1e-2
        mdot = self.mdot
        if np.isnan(mdot).any():
            check_failed = True
            logger.iter("NaN mass flow rate")

        if np.ptp(mdot) > (mdot[0] * rtol):
            check_failed = True
            logger.iter(f"Mass is not conserved, mdot={mdot}")

        # Check that rothalpy is conserved in blade rows
        I = self.I
        if (self.Omega == 0.0).all():
            Itol = I[0] * rtol
        else:
            Itol = np.ptp(I) * rtol

        irow = np.where(np.abs(np.diff(I)) < Itol)[0]
        logger.debug("Checking row rothalpies")
        for iIrow, Irow in enumerate(np.array_split(I, irow)[1:]):
            logger.debug(f"Irow: {Irow}")
            if np.ptp(Irow) > Itol:
                check_failed = True
                logger.iter(
                    f"Rothalpy not conserved in row {iIrow}: I = [{Irow[0], Irow[1]}]"
                )

        # Check that stagnation enthalpy is conserved between blade rows
        ho = self.ho
        hotol = np.ptp(ho) * rtol
        igap = np.where(np.abs(np.diff(I)) < Itol)[0] + 1
        logger.debug("Checking gap enthalpies")
        for igap, hogap in enumerate(np.array_split(ho, igap)[1:-1]):
            logger.debug(f"hogap: {hogap}")
            if not np.all(np.ptp(hogap) < hotol):
                check_failed = True
                logger.iter(
                    f"Absolute stagnation enthalpy not conserved across gap {igap}: ho"
                    f" = [{hogap[0], hogap[1]}]"
                )

        return not check_failed

    def show_debug(self):
        np.set_printoptions(linewidth=np.inf, precision=4, floatmode="maxprec_equal")
        logger.iter(f"rrms: {self.rrms}")
        logger.iter(f"rhub: {self.rhub}")
        logger.iter(f"rtip: {self.rtip}")
        logger.iter(f"A: {self.A}")
        logger.iter(f"To: {self.To}")
        logger.iter(f"T: {self.T}")
        logger.iter(f"Po: {self.Po}")
        logger.iter(f"P: {self.P}")
        logger.iter(f"Vx: {self.Vx}")
        logger.iter(f"Vr: {self.Vr}")
        logger.iter(f"Vt: {self.Vt}")
        logger.iter(f"Vt_rel: {self.Vt_rel}")
        logger.iter(f"Ma: {self.Ma}")
        logger.iter(f"Ma_rel {self.Ma_rel}")
        logger.iter(f"U: {self.U}")
        logger.iter(f"Al: {self.Alpha}")
        logger.iter(f"Al_rel: {self.Alpha_rel}")
        logger.iter(f"Beta: {self.Beta}")
        logger.iter(f"Omega: {self.Omega}")
        logger.iter(f"mdot: {self.mdot}")
        logger.iter(f"ho: {self.ho}")
        logger.iter(f"rho: {self.rho}")
        logger.iter(f"s: {self.s}")

    def __str__(self):
        Pstr = np.array2string(self.Po / 1e5, precision=4)
        Tstr = np.array2string(self.To, precision=4)
        Mastr = np.array2string(self.Ma, precision=3)
        Alrstr = np.array2string(self.Alpha_rel, precision=2)
        Alstr = np.array2string(self.Alpha, precision=2)
        Vxstr = np.array2string(self.Vx, precision=1)
        Vrstr = np.array2string(self.Vr, precision=1)
        Vtstr = np.array2string(self.Vt, precision=1)
        Vtrstr = np.array2string(self.Vt_rel, precision=1)
        rpmstr = np.array2string(self.rpm, precision=0)
        mstr = np.array2string(self.mdot, precision=2)
        return f"""MeanLine(
    Po={Pstr} bar,
    To={Tstr} K,
    Ma={Mastr},
    Vx={Vxstr},
    Vr={Vrstr},
    Vt={Vtstr},
    Vt_rel={Vtrstr},
    Al={Alstr},
    Al_rel={Alrstr},
    rpm={rpmstr},
    mdot={mstr} kg/s
    )"""

    def rspf(self, spf):
        if not np.shape(spf) == ():
            spf = spf.reshape(-1, 1)
        return self.rhub * (1.0 - spf) + self.rtip * spf

    def Vt_free_vortex(self, spf, n=-1):
        return self.Vt * (self.rspf(spf) / self.rrms) ** n

    def Vt_rel_free_vortex(self, spf, n=-1):
        return self.Vt_free_vortex(spf, n) - self.Omega * self.rspf(spf)

    def Alpha_free_vortex(self, spf, n=-1):
        return np.degrees(np.arctan(self.Vt_free_vortex(spf, n) / self.Vm))

    def Alpha_rel_free_vortex(self, spf, n=-1):
        return np.degrees(np.arctan(self.Vt_rel_free_vortex(spf, n) / self.Vm))

    def _get_ref(self, key):
        """Return a variable at inlet/exit of rows, for compressor/turbine."""
        x = getattr(self, key)
        try:
            return np.where(self.ARflow[::2] > 1.0, x[::2], x[1::2])
        except (IndexError, TypeError):
            return x

    @dependent_property
    def rho_ref(self):
        return self._get_ref("rho")

    @dependent_property
    def V_ref(self):
        return self._get_ref("V_rel")

    @dependent_property
    def mu_ref(self):
        return self._get_ref("mu")

    @dependent_property
    def L_visc(self):
        return self.mu_ref / self.rho_ref / self.V_ref

    @dependent_property
    def P_ref(self):
        return self._get_ref("P")

    @dependent_property
    def RR(self):
        return self.rrms[1:] / self.rrms[:-1]

    @dependent_property
    def VmR(self):
        return self.Vm[1:] / self.Vm[:-1]

    def s_ell(self, Co):
        #

        # r"""Pitch to suction surface length using Kaufmann circulation coefficient.

        # From the mean-line data and a specified circulation coefficient,
        # calculate pitch to suction surface length ratios for each blade row.

        # The circulation coefficient :math:`C_0` is a universal measure of blade
        # loading, comparing the circulation of the blade to an idealised
        # circulation with stagnated flow on the pressure surface, and a
        # reference velocity on the suction surface.

        # .. math::

        #     C_0 = \frac{\Gamma_\mathrm{actual}}{\Gamma_\mathrm{ideal}}
        #     = \frac{s}{\ell}\frac{V_{m1}}{V_\mathrm{ref}}\left[
        #     \frac{1-\RR^2}{\phi_1} + \tan \alpha^\rel_1 - \RR \VmR \tan \alpha^\rel_2
        #     \right]

        # where :math:`\RR=r_2/r_1`, :math:`\VmR=V_{m2}/V_{m1}`, and
        # :math:`s/\ell` is the pitch to suction surface length ratio. For turbines,
        # :math:`V_\mathrm{ref}=V_2`  and for compressors :math:`V_\mathrm{ref}=V_1`.

        # :cite:`Coull2013` proposed the circulation coefficient, and
        # :cite:`Kaufmann2020` derived a generalised form for radial machines.

        # """

        centrifugal = (1.0 - self.RR[::2] ** 2.0) * (
            self.tanAlpha[::2] - self.tanAlpha_rel[::2]
        )
        tangential = (
            self.tanAlpha_rel[::2]
            - self.RR[::2] * self.VmR[::2] * self.tanAlpha_rel[1::2]
        )

        total_in = self.cosAlpha_rel[::2] * (centrifugal + tangential)
        total_out = self.cosAlpha_rel[1::2] / self.VmR[::2] * (centrifugal + tangential)
        total = np.where(self.ARflow[::2] > 1.0, total_in, total_out)

        with np.errstate(divide="ignore"):
            return np.abs(Co / total)

    #
    # Non-dimensionals
    #

    @dependent_property
    def phi(self):
        with np.errstate(divide="ignore"):
            return np.where(self.U != 0.0, self.Vm / np.abs(self.U), np.nan)

    @dependent_property
    def PR_tt(self):
        PR = self.Po[-1] / self.Po[0]
        if PR < 1.0:
            PR = 1.0 / PR
        return PR

    @dependent_property
    def PR_ts(self):
        PR = self.P[-1] / self.Po[0]
        if PR < 1.0:
            PR = 1.0 / PR
        return PR

    @dependent_property
    def eta_tt(self):
        hos = self.copy().set_P_s(self.Po, self.s[0]).h
        ho = self.ho
        eta_tt = (hos[-1] - ho[0]) / (ho[-1] - ho[0])
        if eta_tt > 1.0:
            eta_tt = 1.0 / eta_tt
        return eta_tt

    @dependent_property
    def eta_ts(self):
        hs = self.copy().set_P_s(self.P, self.s[0]).h
        ho = self.ho
        eta_ts = (hs[-1] - ho[0]) / (ho[-1] - ho[0])
        if eta_ts > 1.0:
            eta_ts = 1.0 / eta_ts
        # assert eta_ts < self.eta_tt
        return eta_ts

    @dependent_property
    def eta_poly(self):
        eta_poly = (
            self.gamma
            / (self.gamma - 1.0)
            * np.log(self.To[-1] / self.To[0])
            / np.log(self.Po[-1] / self.Po[0])
        )
        if eta_poly > 1.0:
            eta_poly = 1.0 / eta_poly
        return eta_poly

    @dependent_property
    def Yp(self):
        return (self.Po_rel[:-1:2] - self.Po_rel[1::2]) / (
            self.Po_rel[:-1:2] - self.P_ref
        )

    def get_table_limits(self, safety_factors):
        # """Return limiting property values and deltas for gas table generation."""

        # Min/max entropy
        smin = self.s.min()
        smax = self.s.max()
        Ds = smax - smin

        # Minimum pressure
        iPmin = np.argmin(self.P)
        Pmin = self.P[iPmin]
        # Perform the mean-line design
        DP = (self.Po - self.P)[iPmin]

        # Maximum temperature
        iTmax = np.argmax(self.To)
        Tmax = self.To[iTmax]
        DT = (self.To - self.T)[iTmax]

        # Apply a safety factor
        smin -= Ds * safety_factors[0]
        smax += Ds * safety_factors[1]
        Pmin -= DP * safety_factors[2]
        Tmax += DT * safety_factors[3]

        return smin, smax, Pmin, Tmax

    def eval_Cbtob(self, chord, Cbtob):
        # Change of angular momentum
        DrVt = self.rVt[1::2] - self.rVt[::2]
        span = np.minimum(self.span[1::2], self.span[::2])
        r = np.minimum(self.rmid[1::2], self.rmid[::2])
        Vrel = np.minimum(self.V_rel[1::2], self.V_rel[::2])
        rho = np.minimum(self.rho[1::2], self.rho[::2])
        mdot = self.mdot[0]

        Nb = DrVt * mdot / 2.0 / rho / Vrel**2.0 / span / chord / r / Cbtob * 2.0

        return Nb

    def set_Lieblein_DF(self, DFL):
        V1 = self.V_rel[::2]
        V2 = self.V_rel[1::2]
        DVt = np.abs(self.Vt_rel[1::2] - self.Vt_rel[::2])
        logger.debug(f"V1={V1}, V2={V2}, DVt={DVt}")
        if np.any((DFL + V2 / V1 - 1.0) < 0.0):
            raise Exception(
                f"V2/V1={V2/V1} is too low for this DFL={DFL}, need DFL + V2/V1 > 1"
            )
        s_c = 2.0 * V1 / DVt * (DFL + V2 / V1 - 1.0)
        logger.debug(f"s_c={s_c}")
        Al1 = self.Alpha_rel[::2]
        Al2 = self.Alpha_rel[1::2]
        stag = util.atand(0.5 * (util.tand(Al1) + util.tand(Al2)))
        logger.debug(f"stag={stag}")
        s_cx = s_c / util.cosd(stag)
        return s_cx

    @property
    def Co(self):
        return self._get_metadata_by_key("Co")

    @Co.setter
    def Co(self, Co):
        self._set_metadata_by_key("Co", Co)

    @property
    def mdot_choke(self):
        return self._get_metadata_by_key("mdot_choke")

    @mdot_choke.setter
    def mdot_choke(self, mdot_choke):
        self._set_metadata_by_key("mdot_choke", mdot_choke)

    @property
    def mdot_stall(self):
        return self._get_metadata_by_key("mdot_stall")

    @mdot_stall.setter
    def mdot_stall(self, mdot_stall):
        self._set_metadata_by_key("mdot_stall", mdot_stall)

    @property
    def ske(self):
        return self._get_metadata_by_key("ske")

    @ske.setter
    def ske(self, ske):
        self._set_metadata_by_key("ske", ske)

    @property
    def A_surf(self):
        return self._get_metadata_by_key("A_surf", default=np.nan)

    @A_surf.setter
    def A_surf(self, A_surf):
        self._set_metadata_by_key("A_surf", A_surf)

    @property
    def Vcubed(self):
        return self._get_metadata_by_key("Vcubed")

    @Vcubed.setter
    def Vcubed(self, Vcubed):
        self._set_metadata_by_key("Vcubed", Vcubed)

    @property
    def Sdot_surf(self):
        return self._get_metadata_by_key("Sdot_surf", default=np.nan)

    @Sdot_surf.setter
    def Sdot_surf(self, Sdot_surf):
        self._set_metadata_by_key("Sdot_surf", Sdot_surf)

    @property
    def Sdot_tip(self):
        return self._get_metadata_by_key("Sdot_tip")

    @Sdot_tip.setter
    def Sdot_tip(self, Sdot_tip):
        self._set_metadata_by_key("Sdot_tip", Sdot_tip)

    @property
    def mean_line_type(self):
        return self._get_metadata_by_key("mean_line_type")

    @mean_line_type.setter
    def mean_line_type(self, mean_line_type):
        self._set_metadata_by_key("mean_line_type", mean_line_type)

    @property
    def Lsurf(self):
        return self._get_metadata_by_key("Lsurf")

    @Lsurf.setter
    def Lsurf(self, Lsurf):
        self._set_metadata_by_key("Lsurf", Lsurf)

    @property
    def Ds_mix(self):
        return self._get_metadata_by_key("Ds_mix", default=0.0)

    @Ds_mix.setter
    def Ds_mix(self, Ds_mix):
        self._set_metadata_by_key("Ds_mix", Ds_mix)

    @property
    def tip(self):
        return self._get_metadata_by_key("tip")

    @tip.setter
    def tip(self, tip):
        self._set_metadata_by_key("tip", tip)

    @property
    def workdir(self):
        return self._get_metadata_by_key("workdir")

    @workdir.setter
    def workdir(self, workdir):
        self._set_metadata_by_key("workdir", workdir)

    def get_row(self, irow):
        ist = irow * 2
        ien = ist + 2
        return self[ist:ien]


class BaseConfig:
    _name = "Base"

    def __repr__(self):
        s = (
            self._name
            + "Config("
            + ", ".join(
                [f"{v}={getattr(self,v)}" for v in dir(self) if not v.startswith("_")]
            )
            + ")"
        )
        return s

    def __setattr__(self, key, value):
        """Validate attribute assignment."""
        # Should not be able to define new attributes
        current_value = getattr(self, key)
        if key not in dir(self):
            raise TypeError(f"Invalid {self._name}Config variable '{key}'")
        #
        # Type must match existing value
        if isinstance(current_value, type):
            if not isinstance(value, current_value):
                raise TypeError(
                    f"Invalid type={type(value)} "
                    f"for {self._name}Config variable {key}={value}, "
                    f"should be {current_value}"
                )
            super().__setattr__(key, value)
        elif not isinstance(value, type(current_value)):
            raise TypeError(
                f"Invalid type={type(value)} "
                f"for {self._name}Config variable {key}={value}, "
                f"should be {type(current_value)}"
            )
        else:
            super().__setattr__(key, value)

    def __init__(self, **kwargs):
        """Override default parameters using keyword args."""
        for k, v in kwargs.items():
            setattr(self, k, v)
