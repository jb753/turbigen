import numpy as np
from turbigen import util

logger = util.make_logger()


def concatenate(sd, axis=0, touching=False):
    """Join a sequence of StructuredData along an axis."""
    out = sd[0].empty()
    data = [sdi._data for sdi in sd]
    # Cut out touching points
    if touching:
        for i in range(1, len(data)):
            data[i] = np.delete(data[i], 0, axis=axis + 1)
    out._data = np.concatenate(data, axis=axis + 1)
    out._metadata = sd[0]._metadata
    return out


def stack(sd, axis=0):
    """Join a sequence of StructuredData along a new axis."""
    out = sd[0].empty()
    ax = axis if axis < 0 else axis + 1
    out._data = np.stack([sdi._data for sdi in sd], axis=ax)
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

    def __set__(self, instance, _):
        del instance  # So linters do not find unused var
        raise TypeError(f"Cannot assign to dependent property '{self._property_name}'")


class StructuredData:
    """Store array data with scalar metadata in one sliceable object."""

    _data_rows = ()

    def __init__(self, shape=(), order="C", typ=np.double):
        if not isinstance(shape, tuple):
            raise ValueError(f"Invalid input shape, got {shape}, expected a tuple")
        self._order = order
        if order == "C":
            self._data = np.full((self.nprop,) + shape, np.nan, order=order, dtype=typ)
        elif order == "F":
            self._data = np.full(shape + (self.nprop,), np.nan, order=order, dtype=typ)
        else:
            raise ValueError(f"Invalid order {order}, expected 'C' or 'F'")
        self._metadata = {}
        self._dependent_property_cache = {}

    def flip(self, axis):
        out = self.__class__()
        out._data = np.flip(self._data, axis=axis + 1)
        out._metadata = self._metadata
        return out

    def transpose(self, order=None):
        out = self.__class__()
        if order is None:
            order = tuple(reversed(range(self.ndim)))
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

    def flatten(self):
        """Make an flattened view of these data."""
        # Make an empty object by calling constructor with no args
        out = self.__class__()
        # Insert flattened version of current data and metadata
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
            return self._data[ind,]
        else:
            return self._data[..., ind]

    def _set_data_by_key(self, key, val):
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

    def copy(self, dtype=None):
        # Make an empty object by calling constructor with no args
        out = self.__class__()
        # Insert copies of current data and metadata
        out._data = self._data.copy()
        if dtype:
            out._data = out._data.astype(dtype)
        out._metadata = self._metadata.copy()
        return out

    def empty(self, shape=()):
        # Make an empty object by calling constructor with no args
        out = self.__class__()
        # Insert empty data and current metadata
        out._data = np.zeros((self.nprop,) + shape)
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
