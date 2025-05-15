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
        """Slice the data and return a new object."""
        # Special case for scalar indices
        if np.shape(key) == ():
            key = (key,)
        # Now prepend a slice for all variables to key
        key = (slice(None, None, None),) + key
        # Make an empty object by calling constructor with no args
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
        if key not in self._data_rows:
            raise KeyError(
                f"Key '{key}' not found in data rows. Should be one of {self._data_rows}."
            )
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

    @property
    def ndim(self):
        """Number of dimensions of the data array."""
        return len(self.shape)

    @property
    def nvar(self):
        """Number of variables stored at each point."""
        return len(self._data_rows)

    @property
    def shape(self):
        """Shape of the points in the data array."""
        return self._data.shape[1:]

    @property
    def size(self):
        """Total number of points in the data array."""
        return np.prod(self.shape)


class BaseState(StructuredData, ABC):
    """Base class for representing thermodynamic state and velocity vector."""

    _data_rows = ("rho", "rhoVx", "rhoVr", "rhoVt", "rhoe")
    _defaults = {"Omega": 0.0, "Tu0": 300.0, "Ps0": 1e5, "Ts0": 300.0}

    def __init__(self, *, shape=(), order="C", dtype=np.double, **kwargs):
        """Initialize the state with a shape and metadata.

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
        super().__init__(shape=shape, order=order, dtype=dtype, **kwargs)

    #
    # Angular velocity
    #

    @property
    def Omega(self):
        """Relative frame angular velocity [rad/s]."""
        return self._get_metadata_by_key("Omega")

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
    def rho(self):
        return self._get_data_by_key("rho")

    @property
    def rhoVx(self):
        return self._get_data_by_key("rhoVx")

    @property
    def rhoVr(self):
        return self._get_data_by_key("rhoVr")

    @property
    def rhoVt(self):
        return self._get_data_by_key("rhoVt")

    @property
    def rhoe(self):
        return self._get_data_by_key("rhoe")

    #
    # Absolute velocities
    #

    @dependent_property
    def Vx(self):
        return self.rhoVx / self.rho

    @dependent_property
    def Vr(self):
        return self.rhoVr / self.rho

    @dependent_property
    def Vt(self):
        return self.rhoVt / self.rho

    @property
    def Vxrt(self):
        return self._get_data_by_key(("Vx", "Vr", "Vt"))

    @Vxrt.setter
    def Vxrt(self, value):
        self._set_data_by_key(("Vx", "Vr", "Vt"), value)

    #
    # Thermodynamic properties
    #

    @property
    @abstractmethod
    def cp(self) -> float:
        """Specific heat at constant pressure [J/kg/K]."""
        pass

    @property
    @abstractmethod
    def gamma(self) -> float:
        """Ratio of specific heats [--]."""
        pass

    @property
    @abstractmethod
    def rgas(self) -> float:
        """Specific gas constant [J/kg/K]."""
        pass

    @property
    @abstractmethod
    def P(self) -> float:
        """Pressure [Pa]."""
        pass

    @property
    @abstractmethod
    def a(self):
        """Acoustic speed [m/s]."""
        pass

    @property
    @abstractmethod
    def h(self):
        """Specific enthalpy [J/kg]."""
        pass

    @property
    @abstractmethod
    def T(self):
        """Temperature [K]."""
        pass

    @property
    @abstractmethod
    def s(self):
        """Specific entropy [J/kg/K]."""
        pass

    #
    # Derived properties
    #

    @dependent_property
    def cv(self) -> float:
        """Specific heat at constant volume [J/kg/K]."""
        return self.cp / self.gamma

    #
    # Transport properties
    #

    @abstractmethod
    def mu(self):
        """Kinematic viscosity [m^2/s]."""
        raise NotImplementedError()

    @abstractmethod
    def Pr(self):
        """Prandtl number [--]."""
        raise NotImplementedError()
