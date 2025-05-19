"""New attempt at a unified encapsulation of a flow field."""

import numpy as np
from abc import ABC, abstractmethod


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

    def __set__(self, instance, _):
        del instance  # So linters do not find unused var
        raise TypeError(f"Cannot assign to dependent property '{self._property_name}'")


class StructuredData:
    """Store array data with scalar metadata in one sliceable object."""

    _data_rows = ()
    _defaults = {}

    #
    # Initialisation
    #

    def __init__(self, *, shape=(), order="C", dtype=np.double, **kwargs):
        """Allocate the data array and accept arbitrary metadata.

        Parameters
        ----------
        shape : tuple
            Shape of a single property array.
        order : str
            Memory layout order, either 'C' (row-major) or 'F' (column-major).
        dtype : type
            Data type of the array elements.
        kwargs : dict
            Metadata to be stored in the object.

        """

        shape = tuple(shape)

        if order == "C":
            self._data = np.full((self.nvar,) + shape, np.nan, order=order, dtype=dtype)
        elif order == "F":
            self._data = np.full(shape + (self.nvar,), np.nan, order=order, dtype=dtype)
        else:
            raise ValueError(f"Invalid order '{order}'. Use 'C' or 'F'.")

        self._order = order
        self._metadata = kwargs
        self._dtype = dtype

        self._dependent_property_cache = {}

        self.__post_init__()

    def __post_init__(self):
        """Post-initialisation function to be called after all variables are set."""
        # Default no-op
        pass

    def _allocate_variable(self, name):
        """Add a variable to the data array.

        Note that you are responsible for adding get and set methods for the new variable.
        This is a low-level function that should not be used outside of the class,
        only when initialising a delegate class.

        Parameters
        ----------
        name : str
            Key  of the variable.

        """
        self._data_rows += (name,)
        if self._order == "C":
            self._data = np.append(self._data, np.full(self.shape, np.nan), axis=0)
        elif self._order == "F":
            self._data = np.append(self._data, np.full(self.shape, np.nan), axis=-1)
        else:
            raise ValueError(f"Invalid order '{self._order}'. Use 'C' or 'F'.")

    def _stack(self, *args):
        """Stack some variables into a composite variable.

        If the order is 'F', the variables are stacked along the last axis.
        If the order is 'C', the variables are stacked along the first axis.

        Parameters
        ----------
        args : tuple
            Variables to stack.

        Returns
        -------
        out : ndarray
            A composite stacked variable.

        """
        out = self.view()
        if self._order == "C":
            out._data = np.stack(args, axis=0)
        elif self._order == "F":
            out._data = np.stack(args, axis=-1)
        else:
            raise ValueError(f"Invalid order '{self._order}'.")

        return out

    #
    # numpy ndarray style functions
    #

    def view(self):
        """Get a new view of the data array.

        Returns
        -------
        out: StructuredData
            A view of the data array with the same shape and metadata.

        """
        out = self.__class__()
        out._data = self._data
        out._metadata = self._metadata
        out._order = self._order
        out._dtype = self._dtype
        return out

    def flip(self, axis):
        """Reverse indices along the specified axis.

        The metadata and data are views onto the originals.

        Parameters
        ----------
        axis : int
            Axis along which to flip the data.

        Returns
        -------
        out : StructuredData
            A new StructuredData object with flipped data.

        """
        out = self.view()
        out._data = np.flip(self._data, axis=axis + 1)
        assert out._data.base is self._data.base
        return out

    def transpose(self, axes=None):
        """Change the order of the data axes.

        The metadata is a view, and the data is a view where possible.

        Parameters
        ----------
        axes : tuple
            New order of the axes. If None, the axes order is reversed.

        Returns
        -------
        out : StructuredData
            A new StructuredData object with transposed data.

        """

        out = self.view()

        # Default to reverse
        if axes is None:
            axes = tuple(reversed(range(self.ndim)))

        # Add a leading axis for the variable
        axes1 = [
            0,
        ] + [o + 1 for o in axes]

        out._data = np.transpose(self._data, axes1)

        return out

    def squeeze(self):
        """Remove single-dimensional entries from the shape of the data.

        The data and metadata are views of the original.

        Returns
        -------
        out : StructuredData
            A new StructuredData object with squeezed data.

        """

        out = self.view()
        out._data = np.squeeze(self._data)
        return out

    def flat(self):
        """Make a flattened view of these data.

        The data and metadata are views of the original.

        Returns
        -------
        out : StructuredData shape (npoints,)
            A new StructuredData object with all points in a single dimension.

        """
        out = self.view()
        out._data = self._data.reshape(self._data.shape[0], -1)
        assert out._data.base is self._data.base
        return out

    def copy(self):
        """Make a copy of the data.

        Metadata is preserved as a view.

        Returns
        -------
        out : StructuredData
            A new StructuredData object with copied data.

        """
        out = self.view()
        out._data = self._data.copy()
        out._metadata = self._metadata.copy()
        return out

    def empty(self, shape=()):
        """Get an empty object with the same metadata.

        Parameters
        ----------
        shape : tuple
            Shape of the new array. Defaults to scalar.

        """
        out = self.view()
        out._data = np.full((self.nvar,) + shape, np.nan, dtype=self._dtype)
        return out

    def reshape(self, shape):
        """Change the shape of the data in place.

        Parameters
        ----------
        shape : tuple
            New shape of the data array. Must have the same number of elements as the original.

        """
        self._data = self._data.reshape((self.nvar,) + shape)

    def __getitem__(self, key):
        """Slice the data.

        Scalar indices are converted to tuples, so that the
        output will have a singleton dimension. This is different
        from the behaviour of numpy, where a scalar index will
        return a scalar value.

        """
        # Special case for scalar indices
        if np.shape(key) == ():
            key = (key,)
        # Now prepend a slice for all variables to key
        key = (slice(None, None, None),) + key

        out = self.view()
        out._data = self._data[key]
        return out

    #
    # Methods for accessing data and metadata
    #

    def _get_metadata_by_key(self, key):
        """Extract metadata by variable name.

        Return the value from self._defaults if the key is not found.

        Parameters
        ----------
        key : str
            Variable name.

        Returns
        -------
        val : object
            Value of the metadata variable.

        """
        return self._metadata.get(key, self._defaults.get(key))

    def _set_metadata_by_key(self, key, val):
        """Set metadata by variable name.

        Parameters
        ----------
        key : str
            Variable name.
        val : object
            Value to set for the metadata variable.

        """
        self._metadata[key] = val
        self._dependent_property_cache.clear()

    def _lookup_index(self, key):
        """Convert a variable name to an index into the data array.

        Parameters
        ----------
        key : str or tuple
            Variable name or tuple of variable names.

        Returns
        -------
        ind : int or tuple of int
            Index or indices into the data array.

        """
        if isinstance(key, tuple):
            ind = tuple([self._data_rows.index(ki) for ki in key])
        else:
            ind = self._data_rows.index(key)
        return ind

    def _get_data_by_key(self, key):
        """Extract data by variable name.

        Parameters
        ----------
        key : str
            Variable name. Must be in self._data_rows.

        Returns
        -------
        out: ndarray
            Data array for the specified variable.

        """
        ind = self._lookup_index(key)
        if self._order == "C":
            out = self._data[
                ind,
            ]
        elif self._order == "F":
            out = self._data[..., ind]
        else:
            raise ValueError(f"Invalid order '{self._order}'.")

        out = out.view()
        if not np.shape(out) == ():
            out.flags.writeable = False

        return out

    def _set_data_by_key(self, key, val):
        """Set data by variable name.

        Parameters
        ----------
        key : str
            Variable name. Must be in self._data_rows.

        """

        # Which row to set
        ind = self._lookup_index(key)

        # Special case for singleton arrays
        if np.shape(val) == (1,):
            if self._order == "C":
                self._data[ind] = val[0]
            elif self._order == "F":
                self._data[..., ind] = val[0]
        # Otherwise, we assume the data is already in the right shape
        else:
            if self._order == "C":
                self._data[ind] = np.asarray(val)
            elif self._order == "F":
                self._data[..., ind] = np.asarray(val)

        self._dependent_property_cache.clear()

    #
    # Size and shape properties
    #

    @property
    def ndim(self):
        """Number of dimensions of the data array."""
        return len(self.shape)

    @property
    def nvar(self):
        """Number of variables stored at each point."""
        return len(self._data_rows)

    @property
    def ni(self):
        """Number of points along first axis."""
        shape = self._data.shape[1:]
        if len(shape) >= 1:
            return shape[0]
        else:
            raise ValueError("ni not defined for scalar data.")

    @property
    def nj(self):
        """Number of points along second axis."""
        shape = self._data.shape[1:]
        if len(shape) >= 2:
            return shape[1]
        else:
            raise ValueError("nj not defined for 1D data.")

    @property
    def nk(self):
        """Number of points along third axis."""
        shape = self._data.shape[1:]
        if len(shape) >= 3:
            return shape[2]
        else:
            raise ValueError("nk not defined for 2D data.")

    @property
    def shape(self):
        """Shape of the points in the data array."""
        return self._data.shape[1:]

    @property
    def size(self):
        """Total number of points in the data array."""
        return np.prod(self.shape)


class BaseFluid(StructuredData, ABC):
    """Base class for representing thermodynamic state and velocity vector."""

    _data_rows = ("x", "r", "t", "rho", "rhoVx", "rhoVr", "rhorVt", "rhoe")
    _defaults = {"Omega": 0.0, "Tu0": 300.0, "Ps0": 1e5, "Ts0": 300.0, "Nb": 1}

    def __post_init__(self):
        # Ensure r is positive and finite
        self._set_data_by_key("r", 1.0)
        # Set velocity to zero
        self._set_data_by_key("rhoVx", 0.0)
        self._set_data_by_key("rhoVr", 0.0)
        self._set_data_by_key("rhorVt", 0.0)

    def _get_stagnation(self):
        """Copy the state and set h -> h + 0.5 * V^2."""
        return self.copy().set_h_s(self.ho, self.s)

    #
    # Angular velocity and number of blades
    #

    @property
    def Omega(self):
        """Relative frame angular velocity [rad/s]."""
        return self._get_metadata_by_key("Omega")

    @property
    def Nb(self):
        """Number of blades, circumferential periodicity [--]."""
        return self._get_metadata_by_key("Nb")

    #
    # Datum levels
    #

    @property
    def Tu0(self):
        """Temperature at internal energy datum [K]."""
        return self._get_metadata_by_key("Tu0")

    @property
    def Ps0(self):
        """Pressure at entropy datum [Pa]."""
        return self._get_metadata_by_key("Ps0")

    @property
    def Ts0(self):
        """Temperature at entropy datum [K]."""
        return self._get_metadata_by_key("Ts0")

    #
    # Direct access to data rows
    #

    @property
    def x(self):
        return self._get_data_by_key("x")

    @property
    def r(self):
        return self._get_data_by_key("r")

    @property
    def t(self):
        return self._get_data_by_key("t")

    @property
    def rho(self):
        return self._get_data_by_key("rho")

    @property
    def rhoVx(self):
        return self._get_data_by_key("rhoVx")

    @property
    def rhoVr(self):
        return self._get_data_by_key("rhoVr")

    @property
    def rhorVt(self):
        return self._get_data_by_key("rhorVt")

    @property
    def rhoe(self):
        return self._get_data_by_key("rhoe")

    @property
    def conserved(self):
        """Vector of all conserved variables."""
        cons = self._get_data_by_key(("rho", "rhoVx", "rhoVr", "rhorVt", "rhoe"))
        assert cons.base is self._data.base
        return cons

    #
    # Thermodynamic properties
    # (abstract)
    #

    @property
    @abstractmethod
    def cp(self):
        """Specific heat at constant pressure [J/kg/K]."""
        pass

    @property
    @abstractmethod
    def cv(self):
        """Specific heat at constant volume [J/kg/K]."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def a(self):
        """Acoustic speed [m/s]."""
        pass

    @property
    @abstractmethod
    def P(self):
        """Pressure [Pa]."""
        pass

    @property
    @abstractmethod
    def T(self):
        """Temperature [K]."""
        pass

    # @property
    # @abstractmethod
    # def s(self):
    #     """Specific entropy [J/kg/K]."""
    #     pass

    #
    # Transport properties
    # (abstract)
    #

    # @property
    # @abstractmethod
    # def mu(self):
    #     """Kinematic viscosity [m^2/s]."""
    #     raise NotImplementedError()

    # @property
    # @abstractmethod
    # def Pr(self):
    #     """Prandtl number [--]."""
    #     raise NotImplementedError()

    #
    # Derived coordinates
    #

    @dependent_property
    def y(self):
        return self.r * np.sin(self.t)

    @dependent_property
    def z(self):
        return self.r * np.cos(self.t)

    #
    # Derived velocities
    #

    @dependent_property
    def Vx(self):
        return self.rhoVx / self.rho

    @dependent_property
    def Vr(self):
        return self.rhoVr / self.rho

    @dependent_property
    def Vt(self):
        return self.rhorVt / self.r / self.rho

    @dependent_property
    def Vxrt(self):
        return self._stack(self.Vx, self.Vr, self.Vt)

    @dependent_property
    def V(self):
        return np.sqrt(self.Vx**2 + self.Vr**2 + self.Vt**2)

    @dependent_property
    def halfVsq(self):
        return 0.5 * self.V**2

    @dependent_property
    def U(self):
        return self.r * self.Omega

    @dependent_property
    def Vm(self):
        return np.sqrt(self.Vx**2 + self.Vr**2)

    @dependent_property
    def Vt_rel(self):
        return self.Vt - self.U

    @dependent_property
    def V_rel(self):
        return np.sqrt(self.Vm**2 + self.Vt_rel**2)

    @dependent_property
    def halfVsq_rel(self):
        return 0.5 * self.V_rel**2

    #
    # Derived angles
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

    #
    # Derived thermodynamic properties
    #

    @property
    @abstractmethod
    def gamma(self):
        """Ratio of specific heats [--]."""
        pass

    @property
    @abstractmethod
    def rgas(self):
        """Specific gas constant [J/kg/K]."""
        pass

    @dependent_property
    def e(self):
        """Specific total energy [J/kg]."""
        return self.rhoe / self.rho

    @dependent_property
    def u(self):
        """Specific internal energy [J/kg]."""
        return self.e - self.halfVsq

    @dependent_property
    def h(self):
        """Specific enthalpy [J/kg]."""
        return self.u + self.P / self.rho

    #
    # Derived composite
    #

    @dependent_property
    def Ma(self):
        """Mach number [--]."""
        return self.V / self.a

    @dependent_property
    def Ma_rel(self):
        """Mach number in blade relative frame [--]."""
        return self.V_rel / self.a

    @dependent_property
    def Mam(self):
        """Meridional Mach number [--]."""
        return self.Vm / self.a

    @dependent_property
    def I(self):
        """Rothalpy [J/kg]."""
        return self.h + self.halfVsq - self.U * self.Vt

    @dependent_property
    def ho(self):
        """Stagnation enthalpy [J/kg]."""
        return self.h + self.halfVsq

    @dependent_property
    def ho_rel(self):
        """Relative frame stagnation enthalpy [J/kg]."""
        return self.h + self.halfVsq_rel

    #
    # Derived miscellaneous
    #

    @dependent_property
    def rpm(self):
        """Shaft revolutions per minute [rpm]."""
        return self.Omega / 2.0 / np.pi * 60.0

    @dependent_property
    def pitch(self):
        """Angular blade pitch, circumferential period [rad]."""
        return 2.0 * np.pi / self.Nb

    def set_rho_u(self, rho, u):
        """Set density and internal energy.

        Parameters
        ----------
        rho : float
            Density [kg/m^3].
        u : float
            Internal energy [J/kg].

        """
        self._set_data_by_key("rho", rho)
        self._set_data_by_key("rhoe", rho * (u + self.halfVsq))

    #
    # Thermodynamic state setters
    # (abstract)

    # @abstractmethod
    # def set_h_s(self, h, s):
    #     """Set enthalpy and entropy.
    #
    #     Parameters
    #     ----------
    #     h : float
    #         Enthalpy [J/kg].
    #     s : float
    #         Entropy [J/kg/K].
    #
    #     """
    #     pass
    #


class PerfectFluid(BaseFluid):
    """Flow and thermodynamic properties of a perfect gas."""

    def __post_init__(self):
        """Check that the cp and gamma are set."""
        if "cp" not in self._metadata:
            raise ValueError("cp must be set for a Perfect fluid.")
        if "gamma" not in self._metadata:
            raise ValueError("gamma must be set for a Perfect fluid.")
        if self.gamma <= 1.0:
            raise ValueError("gamma must be greater than 1.0 for a Perfect fluid.")
        if self.cp <= 0.0:
            raise ValueError("cp must be positive for a Perfect fluid.")
        super().__post_init__()

    @property
    def cp(self):
        return self._get_metadata_by_key("cp")

    @property
    def gamma(self):
        return self._get_metadata_by_key("gamma")

    @dependent_property
    def cv(self):
        return self.cp / self.gamma

    @dependent_property
    def rgas(self):
        return self.cp * (self.gamma - 1.0) / self.gamma

    @dependent_property
    def P(self):
        return self.rho * (self.gamma - 1.0) * (self.u + self.cv * self.Tu0)

    @dependent_property
    def a(self):
        return np.sqrt(self.gamma * self.rgas * self.T)

    @dependent_property
    def h(self):
        return self.gamma * self.u + self.Tu0 * self.rgas

    @dependent_property
    def T(self):
        return self.u / self.cv + self.Tu0


if __name__ == "__main__":
    # Test the class
    f = PerfectFluid(cp=1000, gamma=1.4)
    f.set_rho_u(1.0, 100e3)
    print(f.rho)
    print(f.u)
    print(f.cp, f.gamma, f.rgas)
