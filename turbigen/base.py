import numpy as np
from turbigen import util
import turbigen.average

logger = util.make_logger()


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

    def __init__(self, shape=()):
        if not isinstance(shape, tuple):
            raise ValueError(f"Invalid input shape, got {shape}, expected a tuple")
        self._data = np.full((self.nprop,) + shape, np.nan)
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
        util.write_yaml(self.to_dict(), fname, mode)

    @classmethod
    def stack(cls, sd, axis=0):
        out = cls()
        out._data = np.stack([sdi._data for sdi in sd], axis=axis + 1)
        out._metadata = sd[0]._metadata
        return out

    @classmethod
    def concatenate(cls, sd, axis=0):
        out = cls()
        out._data = np.concatenate([sdi._data for sdi in sd], axis=axis + 1)
        out._metadata = sd[0]._metadata
        return out

    def flip(self, axis):
        out = self.__class__()
        out._data = np.flip(self._data, axis=axis + 1)
        out._metadata = self._metadata
        return out

    def transpose(self, order):
        out = self.__class__()
        order1 = [0,] + [o+1 for o in order]
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

    # def _get_data_by_key(self, ind):
    #     return self._data[ind,]

    # def _set_data_by_key(self, ind, val):
    #     if self._read_only:
    #         raise Exception(f"Cannot modify read-only {self}")
    #     else:
    #         self._data[ind] = val
    #         self._dependent_property_cache.clear()

    def _get_metadata_by_key(self, key):
        return self._metadata[key]

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
        return self._data[
            self._lookup_index(key),
        ]

    def _set_data_by_key(self, key, val):
        if self._read_only:
            raise Exception(f"Cannot modify read-only {self}")
        else:
            if np.shape(val) == (1,):
                self._data[self._lookup_index(key)] = val[0]
            else:
                self._data[self._lookup_index(key)] = val
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
    def y(self):
        return self.r * np.sin(self.t)

    @dependent_property
    def z(self):
        return self.r * np.cos(self.t)

    @dependent_property
    def vol(self):
        if not self.ndim == 3:
            raise Exception("Cell volume is only defined for 3D grids")

        # Numpy cross function assumes that the components are in last axis
        xyz = np.moveaxis(self.xrt, 0, -1).astype(np.float64)

        # Vectors for cell sides
        qi = np.diff(xyz[:, :-1, :-1, :], axis=0)
        qj = np.diff(xyz[:-1, :, :-1, :], axis=1)
        qk = np.diff(xyz[:-1, :-1, :, :], axis=2)

        return np.sum(qk * np.cross(qi, qj), axis=-1)

    @dependent_property
    def Vxrt_rel(self):
        return np.stack((self.Vx, self.Vr, self.Vt_rel))

    @dependent_property
    def Vi_rel(self):
        """Velocity in grid i-direction."""

        # Edge-center vector for grid spacing
        qi_edge = np.diff(self.xrrt, axis=1)

        # Convert to node-centered
        qi_node = np.full(self.xrrt.shape, np.nan)
        qi_node[:, 0, ...] = qi_edge[:, 0, ...]
        qi_node[:, -1, ...] = qi_edge[:, -1, ...]
        qi_node[:, 1:-1, ...] = 0.5 * (qi_edge[:, :-1, ...] + qi_edge[:, 1:, ...])

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
    def dli(self):
        # Edge vectors along i dirn
        # Numpy cross function assumes that the components are in last axis
        return np.diff(self.xrrt.astype(np.float64), axis=1)

    @dependent_property
    def dlj(self):
        # Edge vectors along j dirn
        # Numpy cross function assumes that the components are in last axis
        return np.diff(self.xrrt.astype(np.float64), axis=2)

    @dependent_property
    def dlk(self):
        # Edge vectors along k dirn
        # Numpy cross function assumes that the components are in last axis
        return np.diff(self.xrrt.astype(np.float64), axis=3)

    @dependent_property
    def dAi(self):
        # Vector area for i=const faces
        if not self.ndim == 3:
            raise Exception("Face area is only defined for 3D grids")
        # Numpy cross function assumes that the components are in last axis
        dlj = np.moveaxis(self.dlj[:, :, :, :-1], 0, -1)
        dlk = np.moveaxis(self.dlk[:, :, :-1, :], 0, -1)
        return np.moveaxis(np.cross(dlj, dlk), -1, 0)

    @dependent_property
    def dAj(self):
        # Vector area for i=const faces
        if not self.ndim == 3:
            raise Exception("Face area is only defined for 3D grids")
        # Numpy cross function assumes that the components are in last axis
        dli = np.moveaxis(self.dli[:, :, :, :-1], 0, -1)
        dlk = np.moveaxis(self.dlk[:, :-1, :, :], 0, -1)
        return np.moveaxis(np.cross(dli, dlk), -1, 0)

    @dependent_property
    def dAk(self):
        # Vector area for i=const faces
        if not self.ndim == 3:
            raise Exception("Face area is only defined for 3D grids")
        # Numpy cross function assumes that the components are in last axis
        dli = np.moveaxis(self.dli[:, :, :-1, :], 0, -1)
        dlj = np.moveaxis(self.dlj[:, :-1, :, :], 0, -1)
        return np.moveaxis(np.cross(dli, dlj), -1, 0)

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
        return util.cum_arc_length(self.xrrt, axis=1)

    @dependent_property
    def tri_area(self):
        if not self.shape[1] == 3:
            raise Exception("This is not a triangulate cut.")

        # Vectors for each side
        qAB = self.xrrt[..., 1] - self.xrrt[..., 0]
        qAC = self.xrrt[..., 2] - self.xrrt[..., 0]

        # Numpy cross function assumes that the components are in last axis
        qAB = np.moveaxis(qAB, 0, -1).astype(np.float64)
        qAC = np.moveaxis(qAC, 0, -1).astype(np.float64)

        return 0.5 * np.cross(qAC, qAB).transpose(1, 0)

    #
    # Velocities
    #

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
        return self.rho * (self.u + 0.5 * self.V**2.0)

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
    def Po_rel(self):
        return self.stagnation_rel.P

    @property
    def To_rel(self):
        return self.stagnation_rel.T

    @property
    def ho_rel(self):
        # We can directly use static enthalpy and velocity
        return self.h + 0.5 * self.V_rel**2.0

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
                self.Vr * self.rhorVt + self.r * self.P,
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

    def mix_out(self):
        """Mix out the cut to a scalar state, conserving mass, momentum and energy."""
        return turbigen.average.mix_out(self)

    def set_conserved(self, conserved):
        rho, *rhoVxrt, rhoe = conserved
        Vxrt = rhoVxrt/rho
        Vxrt[2] /= self.r
        self.Vxrt = Vxrt
        u = rhoe/rho - 0.5*self.V**2
        self.set_rho_u(rho, u)

    def area_average(self):

        dA = np.linalg.norm(self.surface_area[:,:,0,:], axis=-1, ord=2)
        A = np.sum(dA)
        conserved = np.moveaxis(self.conserved,-1,1)
        xrt = np.moveaxis(self.xrt,-1,1)
        conserved_av = np.sum(dA*turbigen.util.node_to_face(conserved),axis=(-2,-1))/A
        xrt_av = np.sum(dA*turbigen.util.node_to_face(xrt),axis=(-2,-1))/A

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

        # Check mass is conserved
        rtol = 1e-2
        mdot = self.mdot
        if np.isnan(mdot).any():
            raise Exception("NaN mass flow rate")

        if mdot.ptp() > (mdot[0] * rtol):
            raise Exception(f"Mass is not conserved, mdot={mdot}")
        assert mdot.ptp() < (mdot[0] * rtol)

        # Check that rothalpy is conserved in blade rows
        I = self.I
        if (self.Omega == 0.0).all():
            Itol = I[0] * rtol
        else:
            Itol = I.ptp() * rtol

        irow = np.where(np.abs(np.diff(I)) < Itol)[0]
        logger.debug("Checking row rothalpies")
        for iIrow, Irow in enumerate(np.array_split(I, irow)[1:]):
            logger.debug(f"Irow: {Irow}")
            if Irow.ptp() > Itol:
                raise Exception(
                    f"Rothalpy not conserved in row {iIrow}: I = [{Irow[0], Irow[1]}]"
                )

        # Check that stagnation enthalpy is conserved between blade rows
        ho = self.ho
        hotol = ho.ptp() * rtol
        igap = np.where(np.abs(np.diff(I)) < Itol)[0] + 1
        logger.debug("Checking gap enthalpies")
        for igap, hogap in enumerate(np.array_split(ho, igap)[1:-1]):
            logger.debug(f"hogap: {hogap}")
            if not np.all(hogap.ptp() < hotol):
                raise Exception(
                    f"Absolute stagnation enthalpy not conserved across gap {igap}: ho"
                    f" = [{hogap[0], hogap[1]}]"
                )

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

        return np.abs(Co / total)

    #
    # Non-dimensionals
    #

    @dependent_property
    def phi(self):
        with np.errstate(divide="ignore"):
            return np.where(self.U != 0.0, self.Vm / self.U, np.nan)

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
    def Asurf(self):
        return self._get_metadata_by_key("Asurf")

    @Asurf.setter
    def Asurf(self, Asurf):
        self._set_metadata_by_key("Asurf", Asurf)

    @property
    def Sdot_wall(self):
        return self._get_metadata_by_key("Sdot_wall")

    @Sdot_wall.setter
    def Sdot_wall(self, Sdot_wall):
        self._set_metadata_by_key("Sdot_wall", Sdot_wall)

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
        return self._get_metadata_by_key("Ds_mix")

    @Ds_mix.setter
    def Ds_mix(self, Ds_mix):
        self._set_metadata_by_key("Ds_mix", Ds_mix)

    @property
    def blockage(self):
        return self._get_metadata_by_key("blockage")

    @blockage.setter
    def blockage(self, blockage):
        self._set_metadata_by_key("blockage", blockage)

    @property
    def tip(self):
        return self._get_metadata_by_key("tip")

    @tip.setter
    def tip(self, tip):
        self._set_metadata_by_key("tip", tip)

    @property
    def workdir(self):
        return self._get_metadata_by_key("tip")

    @workdir.setter
    def workdir(self, workdir):
        self._set_metadata_by_key("workdir", workdir)


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
