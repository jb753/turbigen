"""New meanline data struct using ember base."""

import ember.block
import ember.fluid
import numpy as np
import turbigen.plugins
import turbigen.meanline_design_new
import inspect


f32 = np.float32


class MeanLineConfig:
    """Configuration for a MeanLine object."""

    def __init__(self, mean_line_type, n_row, design_vars):
        """Initialize the configuration."""
        self.n_row = n_row
        self.type = mean_line_type
        self.design_vars = design_vars

        # Store the forward and backward functions
        reg = turbigen.plugins.get_registry()
        self._forward = reg["mean_line_forward"][self.type]
        self._backward = reg["mean_line_backward"][self.type]

        # Allocate placeholders for nominal and actual mean lines
        self.nominal = MeanLine(n_row)
        self.actual = MeanLine(n_row)

        # Check that design_vars match forward function signature
        sig = inspect.signature(self._forward)
        params = list(sig.parameters.values())[1:]  # Skip first 'mean_line' param
        param_names = {p.name for p in params if p.default is p.empty}
        missing = param_names - set(design_vars.keys())
        if missing:
            raise ValueError(
                f"Missing required design variables for mean_line type '{self.type}': {missing}"
            )

        # Check for unexpected design_vars
        unexpected = set(design_vars.keys()) - {p.name for p in params}
        if unexpected:
            raise ValueError(
                f"Unexpected design variables for mean_line type '{self.type}': {unexpected}"
            )

    @classmethod
    def from_dict(cls, d):
        """Initialize from a dictionary."""

        turbigen.plugins.check_plugins()

        # Extract values from dictionary
        mean_line_type = d.pop("type")
        n_row = d.pop("n_row")

        # Get available types from plugin registry
        reg = turbigen.plugins.get_registry()
        all_types = set(reg["mean_line_forward"].keys())

        # Validate type
        if not mean_line_type:
            raise ValueError(
                f"mean_line configuration requires a 'type' key. Available types: {all_types}"
            )
        if mean_line_type not in all_types:
            raise ValueError(
                f"Unknown mean_line type '{mean_line_type}'. Available types: {all_types}"
            )

        # Validate n_row
        if n_row is None:
            raise ValueError("mean_line configuration requires an 'n_row' key.")
        if n_row < 1:
            raise ValueError(f"n_row must be >= 1, got {n_row}")

        # Remaining keys are design variables
        design_vars = d

        return cls(mean_line_type, n_row, design_vars)

    def to_dict(self):
        """Convert to a dictionary."""
        return {
            "type": self.type,
            "n_row": self.n_row,
            **self.design_vars,
        }

    def set_nominal(self, fluid):
        """Set the nominal mean-line flow field."""
        self.nominal.set_fluid(fluid)
        self._forward(self.nominal, **self.design_vars)

    def check_nominal(self):
        params_inv = self._backward(self.nominal)

        rtol = 1e-3

        # Compare forward and inverse params, check within a tolerance
        for k, v in self.design_vars.items():
            if k not in params_inv:
                raise Exception(
                    f"Design variable {k} not returned by inverse function."
                )
            # Allow uncalculated variables to be None
            if params_inv[k] is None:
                continue

            # Compare the value of the design variable to nominal
            if np.all(v == 0.0):
                # Absolute tolerance for zero values
                if np.allclose(v, params_inv[k], atol=0.1):
                    continue
            else:
                # Relative tolerance for non-zero values
                if np.allclose(v, params_inv[k], rtol=rtol):
                    continue

            raise Exception(
                f"Meanline inverted {k}={params_inv[k]} not same as nominal value {v}"
            )

        # Check mass is conserved
        mdot = self.nominal.mdot
        if np.isnan(mdot).any():
            raise Exception(f"NaN mass flow rate mdot={mdot}")

        if np.ptp(mdot) > (mdot[0] * rtol):
            raise Exception(f"Mass flow rate not conserved: mdot={mdot}")


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
            except (ValueError, AttributeError) as e:
                # print full traceback for debugging
                import traceback

                traceback.print_exc()
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

        # Broadcast all arguments to arrays of correct length
        broadcasted_args = [np.broadcast_to(arg, (n_stations,)) for arg in args]

        # Call setter on each scalar station with its corresponding value
        for i, station in enumerate(self._stations):
            getattr(station, method_name)(*[arg[i] for arg in broadcasted_args])

        return

    # Get docstring from Block's method
    block_method = getattr(ember.block.Block, method_name, None)
    setter.__doc__ = block_method.__doc__

    return setter


class MeanLine:
    """One-dimensional flow field and geometry along nominal mean line."""

    def __init__(self, n_row):
        """Allocate a meanline given working fluid and number of rows."""
        n_stations = n_row * 2
        self._stations = [Station(shape=()) for _ in range(n_stations)]
        self._n_row = n_row

    def set_fluid(self, fluid):
        """Set the working fluid for all stations."""
        for station in self._stations:
            station.set_fluid(fluid)
        return self

    @property
    def n_row(self):
        """Number of blade rows."""
        return self._n_row

    @property
    def shape(self):
        """Number of stations."""
        return (self._n_row, 2)

    Vx = _make_concat_property("Vx")
    s = _make_concat_property("s")
    Alpha = _make_concat_property("Alpha")
    rho = _make_concat_property("rho")
    halfVsq = _make_concat_property("halfVsq")
    U = _make_concat_property("U")
    Vm = _make_concat_property("Vm")
    ho = _make_concat_property("ho")
    Ma = _make_concat_property("Ma")
    mdot = _make_concat_property("mdot")
    htr = _make_concat_property("htr")
    eta_tt = _make_concat_property("eta_tt")
    Po = _make_concat_property("Po")
    T = _make_concat_property("T")
    Vr = _make_concat_property("Vr")
    Vt = _make_concat_property("Vt")
    h = _make_concat_property("h")
    span = _make_concat_property("span")
    rhoVx = _make_concat_property("rhoVx")
    r_mid = _make_concat_property("r_mid")
    r_hub = _make_concat_property("r_hub")
    r_cas = _make_concat_property("r_cas")
    r_rms = _make_concat_property("r_rms")
    Omega = _make_concat_property("Omega")
    Beta = _make_concat_property("Beta")
    Alpha_rel = _make_concat_property("Alpha_rel")
    Am = _make_concat_property("Am")

    set_Vxrt = _make_setter_method("set_Vxrt")
    set_Vx = _make_setter_method("set_Vx")
    set_Vr = _make_setter_method("set_Vr")
    set_Vt = _make_setter_method("set_Vt")
    set_h_s = _make_setter_method("set_h_s")
    set_r_rms = _make_setter_method("set_r_rms")
    set_Am = _make_setter_method("set_Am")
    set_Omega = _make_setter_method("set_Omega")
    set_span_htr = _make_setter_method("set_span_htr")

    def __getitem__(self, key):
        """Index into the meanline.
        MeanLine[i] returns station i
        """
        if isinstance(key, int):
            return self._stations[key]
        raise TypeError(f"MeanLine index must be int, got {type(key)}")

    def get_row(self, i_row):
        """Get the two stations for a given row index."""
        if i_row < 0 or i_row >= self._n_row:
            raise IndexError(f"Row index {i_row} out of range for n_row={self._n_row}")
        _stations = [self._stations[2 * i_row], self._stations[2 * i_row + 1]]
        out = MeanLine(n_row=1)
        out._stations = _stations
        return out


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
        # Block set_r to avoid confusion
        del r
        raise NotImplementedError("Use set_r_rms on a mean-line station.")

    def set_r_rms(self, r_rms):
        """Set annulus root-mean-square radius [m]."""
        super().set_r(r_rms)
        return self

    def set_span_htr(self, span, htr):
        """Define annulus geometry using span and hub-to-tip ratio."""
        assert (
            np.abs(self.Beta) < 1.0
        ), "Beta must be set zero before calling set_span_htr"
        r_rms = span * np.sqrt(0.5 * (1.0 + htr**2)) / (1.0 - htr)
        Am = 2.0 * np.pi * r_rms**2 * (1.0 - htr**2) / (1.0 + htr**2)
        self.set_r_rms(r_rms)
        self.set_Am(Am)
        return self

    def set_span_r_rms(self, span, r_rms):
        self.set_r_rms(r_rms)
        dr = span / np.cos(np.radians(self.Beta))
        r_mid = np.sqrt(r_rms**2 - (dr / 2.0) ** 2)
        Am = 2.0 * np.pi * r_mid * span
        self.set_Am(Am)
        return self

    def set_span_r_mid(self, span, r_mid):
        Am = 2.0 * np.pi * r_mid * span
        self.set_Am(Am)
        r_rms = np.sqrt(r_mid**2 + (span / 2.0) ** 2)
        self.set_r_rms(r_rms)
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
