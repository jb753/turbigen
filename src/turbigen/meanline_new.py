"""New meanline data struct using ember base."""

import ember.block
import ember.fluid
import numpy as np

f32 = np.float32


def _make_concat_property(property_name):
    """Factory method to create a property that concatenates values from all stations.

    Args:
        property_name: Name of the property to extract from each station

    Returns:
        A property object that concatenates the named property from all stations
    """

    def getter(self):
        # Inline concatenation: extract property from all stations and concatenate
        # Handle uninitialized stations by returning NaN for those indices
        values = []
        for station in self._stations:
            try:
                value = getattr(station, property_name)
                # Convert scalar to array for concatenation
                values.append(np.atleast_1d(value))
            except (ValueError, AttributeError):
                # Station not initialized, use NaN
                values.append(np.array([np.nan], dtype=f32))

        result = np.concatenate(values, dtype=f32)

        # Make the array read-only to prevent confusion
        result.flags.writeable = False

        return result

    # Get docstring from Block's property if available
    block_property = getattr(ember.block.Block, property_name, None)
    if block_property is not None and hasattr(block_property, "__doc__"):
        getter.__doc__ = block_property.__doc__
    else:
        getter.__doc__ = f"{property_name} concatenated from all stations."

    return property(getter)


def _make_setter_method(method_name):
    """Factory method to create a setter that distributes calls to all stations.

    Args:
        method_name: Name of the setter method (e.g., 'set_Vxrt')

    Returns:
        A method that splits input arrays and calls the setter on each station
    """

    def setter(self, *args):
        """Distribute setter call to all stations with shape checking."""
        # Determine expected total length (2 stations per row)
        n_stations = self._n_row * 2

        # Check that all array arguments have compatible shapes
        for arg in args:
            if isinstance(arg, np.ndarray):
                if arg.shape[0] != n_stations:
                    raise ValueError(
                        f"{method_name} expects arrays of length {n_stations} "
                        f"(2 stations × {self._n_row} rows), got {arg.shape[0]}"
                    )

        # Call setter on each scalar station with its corresponding value
        for i, station in enumerate(self._stations):
            # Extract scalar value for this station
            station_args = []
            for arg in args:
                if isinstance(arg, np.ndarray):
                    station_args.append(arg[i])
                else:
                    station_args.append(arg)

            # Call the setter method on the station
            getattr(station, method_name)(*station_args)

        return self

    # Get docstring from Block's method if available
    block_method = getattr(ember.block.Block, method_name, None)
    if block_method is not None and hasattr(block_method, "__doc__"):
        setter.__doc__ = block_method.__doc__
    else:
        setter.__doc__ = f"{method_name} distributed to all stations."

    return setter


class MeanLine:
    """One-dimensional flow field and geometry along nominal mean line."""

    def __init__(self, n_row, fluid):
        """Allocate a meanline given working fluid and number of rows."""
        n_stations = n_row * 2
        self._stations = []
        for _ in range(n_stations):
            station = Station(shape=())
            station.set_fluid(fluid)
            self._stations.append(station)
        self._n_row = n_row

    @property
    def n_row(self):
        """Number of blade rows."""
        return self._n_row

    @property
    def shape(self):
        """Number of stations."""
        return (self._n_row, 2)

    Vx = _make_concat_property("Vx")
    set_Vxrt = _make_setter_method("set_Vxrt")

    def __getitem__(self, key):
        """Index into the meanline.
        MeanLine[i,0] returns inlet station of row i (scalar Station)
        MeanLine[i,1] returns outlet station of row i (scalar Station)
        MeanLine[i] returns both stations of row i as another MeanLine
        """
        if isinstance(key, tuple):
            # A scalar station: MeanLine[row, station]
            row_idx, station_idx = key
            if station_idx not in (0, 1):
                raise IndexError(
                    f"Station index must be 0 (inlet) or 1 (outlet), got {station_idx}"
                )
            return self._stations[row_idx * 2 + station_idx]
        elif isinstance(key, int):
            # Two stations as a view of existing MeanLine object
            row_idx = key
            out = MeanLine(n_row=1, fluid=self._stations[0].fluid)
            out._stations = self._stations[row_idx * 2 : row_idx * 2 + 2]
            return out
        raise TypeError(f"MeanLine indices must be int or tuple, got {type(key)}")


class Station(ember.block.Block):
    """A single station in a mean-line flow path."""

    _data_keys = ember.block.Block._data_keys + ("Am",)

    def __post_init__(self):
        """Initialize Station and verify it is scalar."""
        if self.shape != ():
            raise ValueError(f"Station must be a scalar, got shape {self.shape}")
        super().__post_init__()

    @property
    def Am(self):
        """Annulus area projected in meridional direction [m^2]."""
        return self._get_data_by_key("Am")

    @property
    def mdot(self):
        """Annulus mass flow rate [kg/s]"""
        return self.rho * self.Vm * self.Am

    @property
    def r_rms(self):
        """Annulus root-mean-square radius [m]."""
        return self._get_data_by_key("r")

    @property
    def cosBeta(self):
        """Cosine of pitch angle [-]."""
        return np.cos(self.Beta)

    @property
    def r_cas(self):
        """Annulus casing radius [m]."""
        return np.sqrt(self.Am * self.cosBeta / 2.0 / np.pi + self.r_rms**2.0)

    @property
    def r_hub(self):
        """Annulus hub radius [m]."""
        return np.sqrt(self.r_rms**2.0 - self.Am * self.cosBeta / 2.0 / np.pi)

    @property
    def r_mid(self):
        """Annulus mid radius [m]."""
        return 0.5 * (self.r_hub + self.r_cas)

    @property
    def span(self):
        """Annulus span [m]."""
        return self.Am / 2.0 / np.pi / self.r_mid

    @property
    def htr(self):
        """Annulus hub-to-tip ratio [--]."""
        return self.r_hub / self.r_cas

    def set_Am(self, Am):
        """Set annulus area projected in meridional direction."""
        self._set_data_by_key("Am", Am)
        return self

    def set_r(self, r):
        del r
        raise NotImplementedError("Use set_r_rms on a mean-line station.")

    def set_r_rms(self, r_rms):
        """Set annulus root-mean-square radius [m]."""
        self._set_data_by_key("r", r_rms)
        return self


if __name__ == "__main__":
    fluid = ember.fluid.PerfectFluid(cp=1005, gamma=1.4, mu=1.8e-5, Pr=1.0)

    # Test scalar Station
    station = Station(shape=())
    station.set_fluid(fluid)
    station.set_r_rms(3.0)
    station.set_Am(3.0)
    station.set_Vxrt(100.0, 0.0, 50.0)
    station.set_P_T(1e5, 300.0)
    print("Scalar Station mdot:", station.mdot)
    print("Scalar Station r_rms:", station.r_rms)

    # Test MeanLine
    ml = MeanLine(n_row=2, fluid=fluid)
    print(f"\nCreated MeanLine with {ml.n_row} rows (4 scalar stations)")

    # Test that n_row is read-only
    print("\nTesting n_row is read-only:")
    try:
        ml.n_row = 5
        print("ERROR: n_row should not be assignable!")
    except AttributeError as e:
        print(f"PASS: n_row is read-only - {e}")

    # Test indexing with scalar
    print("\nTesting scalar indexing:")
    station0 = ml[0]
    print(f"ml[0] type: {type(station0).__name__}")
    print(f"ml[0] shape: {station0.shape}")
    assert isinstance(station0, Station), "ml[0] should return a Station"
    assert station0.shape == (), f"ml[0] should have shape (), got {station0.shape}"
    print("PASS: scalar indexing works")

    # Test indexing with tuple
    print("\nTesting tuple indexing:")
    station01 = ml[0, 1]
    print(f"ml[0, 1] type: {type(station01).__name__}")
    print(f"ml[0, 1] shape: {station01.shape}")
    assert isinstance(station01, Station), "ml[0, 1] should return a Station"
    assert (
        station01.shape == ()
    ), f"ml[0, 1] should have shape (), got {station01.shape}"
    # Verify it's the same as ml[1]
    assert ml[0, 1] is ml[1]
    print("PASS: tuple indexing works")

    # Test setter method
    print("\nTesting setter method:")
    # Set geometry for all 4 stations
    r_rms = np.array([0.5, 0.55, 0.6, 0.65], dtype=f32)
    Am = np.array([1.0, 1.1, 1.2, 1.3], dtype=f32)
    P = np.array([1e5, 0.95e5, 0.9e5, 0.85e5], dtype=f32)
    T = np.array([300.0, 295.0, 290.0, 285.0], dtype=f32)

    for i in range(4):
        ml[i].set_r_rms(r_rms[i])
        ml[i].set_Am(Am[i])
        ml[i].set_P_T(P[i], T[i])

    # Use set_Vxrt to set all velocities at once
    Vx = np.array([100.0, 110.0, 120.0, 130.0], dtype=f32)
    Vr = np.array([0.0, 0.0, 0.0, 0.0], dtype=f32)
    Vt = np.array([50.0, 55.0, 60.0, 65.0], dtype=f32)
    ml.set_Vxrt(Vx, Vr, Vt)

    # Verify using concatenated property
    print(f"Set Vx: {Vx}")
    print(f"Got Vx: {ml.Vx}")
    assert np.allclose(ml.Vx, Vx)
    print("PASS: setter method works")

    print("\nAll tests passed!")
