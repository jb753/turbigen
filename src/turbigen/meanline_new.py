"""New meanline data struct using ember base."""

import ember.block
import ember.fluid
import numpy as np
import turbigen.plugins
import turbigen.meanline_design_new
import inspect


f32 = np.float32


# Configure pdoc to show inherited members for Station class
__pdoc__ = {
    "Station": True,
}


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

    def warn(self):
        """Print a warning if there are any suspicious values."""

        # Warn for very high flow angles
        if np.abs(self.nominal.Alpha_rel).max() > 85.0:
            logger.warning(
                """WARNING: Relative flow angles are approaching 90 degrees.
This suggests a physically-consistent but suboptimal mean-line design
and will cause problems with meshing and solving for the flow field."""
            )

        # Warn for wobbly annulus
        is_radial = np.abs(self.nominal.Beta).max() > 10.0
        is_multirow = self.nominal.n_row > 2
        if is_radial and is_multirow:
            if np.diff(np.sign(np.diff(self.r_rms))).any():
                logger.warning(
                    """WARNING: Radii do not vary monotonically.
This suggests a physically-consistent but suboptimal mean-line design
and will cause problems with meshing and solving for the flow field."""
                )


def _make_concat_property(property_name):
    """Factory method to create a property that concatenates values from all stations.

    Args:
        property_name: Name of the property to extract from each station

    Returns:
        A property object that concatenates the named property from all stations.
        For scalar properties, returns a 1D array of length n_stations.
        For non-scalar properties (e.g., conserved with shape (5,)), returns a 2D array
        of shape (n_stations, ...) by stacking along the first axis.
    """

    def getter(self):
        # Inline concatenation: extract property from all stations and concatenate
        # Handle uninitialized stations by returning NaN for those indices
        values = []
        is_scalar = None
        for station in self._stations:
            try:
                value = getattr(station, property_name)
                # Determine if property is scalar or non-scalar on first valid value
                if is_scalar is None:
                    is_scalar = np.ndim(value) == 0
                values.append(value)
            except (ValueError, AttributeError) as e:
                # print full traceback for debugging
                import traceback

                traceback.print_exc()
                # Station not initialized, use NaN
                if is_scalar is None:
                    # Default to scalar if we haven't seen a valid value yet
                    values.append(np.array([np.nan], dtype=f32))
                elif is_scalar:
                    values.append(np.nan)
                else:
                    # For non-scalar properties, we'll need to infer shape from first valid value
                    # For now, use a placeholder
                    values.append(np.array([np.nan], dtype=f32))

        # Use stack for non-scalar properties, concatenate for scalar properties
        if is_scalar:
            # Convert scalars to 1D array for concatenation
            values_1d = [np.atleast_1d(v) for v in values]
            result = np.concatenate(values_1d, dtype=f32)
        else:
            # Stack non-scalar properties along first axis
            result = np.stack(values, axis=0).astype(f32)

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
        return (self._n_row * 2,)

    @property
    def fluid(self):
        """Equation of state."""
        return self._stations[0].fluid

    def copy(self):
        """Create a deep copy of the MeanLine."""
        new_ml = MeanLine(self.n_row)
        for i in range(self.n_row * 2):
            new_ml._stations[i] = self._stations[i].copy()
        return new_ml

    @property
    def eta_tt(self):
        """Total-to-total isentropic efficiency: (ho1 - ho2) / (ho1 - ho2s)."""
        ho1 = self[0].ho
        ho2 = self[-1].ho
        ho2s = self[0].empty().set_P_s(self[-1].Po, self[0].s).h

        with np.errstate(divide="ignore", invalid="ignore"):
            eta = (ho1 - ho2) / (ho1 - ho2s)

        if np.isnan(eta):
            return np.inf

        if eta > 1.0:
            eta = 1.0 / eta

        return float(eta)

    @property
    def eta_ts(self):
        """Total-to-static isentropic efficiency: (ho1 - ho2) / (ho1 - h2s)."""
        ho1 = self[0].ho
        ho2 = self[-1].ho
        h2s = self[0].empty().set_P_s(self[-1].P, self[0].s).h

        with np.errstate(divide="ignore", invalid="ignore"):
            eta = (ho1 - ho2) / (ho1 - h2s)

        if np.isnan(eta):
            return np.inf

        if eta > 1.0:
            eta = 1.0 / eta

        return float(eta)

    # Coordinates
    x = _make_concat_property("x")
    r = _make_concat_property("r")
    t = _make_concat_property("t")
    xrt = _make_concat_property("xrt")
    xyz = _make_concat_property("xyz")
    xrrt = _make_concat_property("xrrt")
    y = _make_concat_property("y")
    z = _make_concat_property("z")

    # Conserved variables
    rho = _make_concat_property("rho")
    rhoVx = _make_concat_property("rhoVx")
    rhoVr = _make_concat_property("rhoVr")
    rhorVt = _make_concat_property("rhorVt")
    rhoe = _make_concat_property("rhoe")
    conserved = _make_concat_property("conserved")

    # Velocity components
    Vx = _make_concat_property("Vx")
    Vr = _make_concat_property("Vr")
    Vt = _make_concat_property("Vt")
    Vxrt = _make_concat_property("Vxrt")
    Vxyz = _make_concat_property("Vxyz")
    Vy = _make_concat_property("Vy")
    Vz = _make_concat_property("Vz")

    # Velocity magnitudes and derived
    V = _make_concat_property("V")
    Vm = _make_concat_property("Vm")
    U = _make_concat_property("U")
    V_rel = _make_concat_property("V_rel")
    Vt_rel = _make_concat_property("Vt_rel")
    rhoVm = _make_concat_property("rhoVm")

    # Energy
    e = _make_concat_property("e")
    u = _make_concat_property("u")
    halfVsq = _make_concat_property("halfVsq")
    halfVsq_rel = _make_concat_property("halfVsq_rel")

    # Flow angles
    Alpha = _make_concat_property("Alpha")
    Beta = _make_concat_property("Beta")
    Alpha_rel = _make_concat_property("Alpha_rel")
    tanAlpha = _make_concat_property("tanAlpha")
    tanAlpha_rel = _make_concat_property("tanAlpha_rel")
    tanBeta = _make_concat_property("tanBeta")
    sinBeta = _make_concat_property("sinBeta")

    # Thermodynamic properties
    P = _make_concat_property("P")
    T = _make_concat_property("T")
    s = _make_concat_property("s")
    h = _make_concat_property("h")
    a = _make_concat_property("a")
    cp = _make_concat_property("cp")
    cv = _make_concat_property("cv")
    gamma = _make_concat_property("gamma")
    rgas = _make_concat_property("rgas")

    # Stagnation properties
    ho = _make_concat_property("ho")
    Po = _make_concat_property("Po")
    To = _make_concat_property("To")
    rhoo = _make_concat_property("rhoo")
    uo = _make_concat_property("uo")

    # Relative frame stagnation
    ho_rel = _make_concat_property("ho_rel")
    Po_rel = _make_concat_property("Po_rel")
    To_rel = _make_concat_property("To_rel")
    rhoo_rel = _make_concat_property("rhoo_rel")
    uo_rel = _make_concat_property("uo_rel")
    I = _make_concat_property("I")

    # Non-dimensional numbers
    Ma = _make_concat_property("Ma")
    Ma_rel = _make_concat_property("Ma_rel")

    # Transport properties
    mu = _make_concat_property("mu")
    Pr = _make_concat_property("Pr")

    # Thermodynamic derivatives
    dhdP_rho = _make_concat_property("dhdP_rho")
    dhdrho_P = _make_concat_property("dhdrho_P")
    dsdP_rho = _make_concat_property("dsdP_rho")
    dsdrho_P = _make_concat_property("dsdrho_P")
    dudP_rho = _make_concat_property("dudP_rho")
    dudrho_P = _make_concat_property("dudrho_P")

    # Variable sets
    primitive = _make_concat_property("primitive")
    bcond = _make_concat_property("bcond")

    # Jacobians
    J_prim_to_cons = _make_concat_property("J_prim_to_cons")
    J_cons_to_prim = _make_concat_property("J_cons_to_prim")
    J_prim_to_chic = _make_concat_property("J_prim_to_chic")
    J_chic_to_prim = _make_concat_property("J_chic_to_prim")
    J_prim_to_flux = _make_concat_property("J_prim_to_flux")
    J_flux_to_prim = _make_concat_property("J_flux_to_prim")
    J_prim_to_bcond = _make_concat_property("J_prim_to_bcond")
    J_bcond_to_prim = _make_concat_property("J_bcond_to_prim")
    J_flux_to_cons = _make_concat_property("J_flux_to_cons")
    J_cons_to_flux = _make_concat_property("J_cons_to_flux")
    J_bcond_to_cons = _make_concat_property("J_bcond_to_cons")
    J_cons_to_bcond = _make_concat_property("J_cons_to_bcond")

    # Annulus geometry (Station-specific)
    Am = _make_concat_property("Am")
    r_rms = _make_concat_property("r_rms")
    r_mid = _make_concat_property("r_mid")
    r_hub = _make_concat_property("r_hub")
    r_cas = _make_concat_property("r_cas")
    span = _make_concat_property("span")
    htr = _make_concat_property("htr")
    mdot = _make_concat_property("mdot")

    # Rotation
    Omega = _make_concat_property("Omega")

    set_Vxrt = _make_setter_method("set_Vxrt")
    set_Vx = _make_setter_method("set_Vx")
    set_Vr = _make_setter_method("set_Vr")
    set_Vt = _make_setter_method("set_Vt")
    set_x = _make_setter_method("set_x")
    set_h_s = _make_setter_method("set_h_s")
    set_r_rms = _make_setter_method("set_r_rms")
    set_Am = _make_setter_method("set_Am")
    set_Omega = _make_setter_method("set_Omega")
    set_span_htr = _make_setter_method("set_span_htr")
    set_conserved = _make_setter_method("set_conserved")

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

    def get_ref(self, i_row):
        """Reference station at inlet/exit of rows, for compressor/turbine."""
        row = self.get_row(i_row)
        A_flow = row.Am / np.cos(np.radians(row.Beta))
        AR_flow = A_flow[1] / A_flow[0]
        return row[0] if AR_flow >= 1.0 else row[1]

    def __repr__(self):
        """Return a string representation of the MeanLine object."""
        return f"MeanLine(n_row={self.n_row}, id={id(self)})"

    def to_string(self):
        """Provide a concise string representation of MeanLine properties."""
        # Define the properties to display
        properties = [
            ("Po", self.Po / 1e5, "[bar]"),
            ("To", self.To, "[K]"),
            ("Ma", self.Ma, ""),
            ("Ma_rel", self.Ma_rel, ""),
            ("Alpha", self.Alpha, "[deg]"),
            ("Alpha_rel", self.Alpha_rel, "[deg]"),
        ]

        # Build the table
        table_str = ""
        for name, values, unit in properties:
            # Format the property row
            row = f"{name + ' ' + unit:<15}"
            for val in values:
                # Special formatting for different properties
                if name == "Po":
                    row += f"{val:>12.3f}"
                elif name == "To":
                    row += f"{val:>12.2f}"
                elif name in ["Ma", "Ma_rel"]:
                    row += f"{val:>12.3f}"
                elif name in ["Alpha", "Alpha_rel"]:
                    row += f"{val:>12.1f}"
            table_str += row + "\n"

        return table_str


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
    ml = MeanLine(n_row=2)
    ml.set_fluid(fluid)
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

    # Test non-scalar property (conserved)
    print("\nTesting non-scalar property (conserved):")
    conserved = ml.conserved
    print(f"conserved shape: {conserved.shape}")
    assert conserved.shape == (
        4,
        5,
    ), f"conserved should have shape (4, 5), got {conserved.shape}"
    # Verify each station's conserved values match
    for i in range(4):
        station_conserved = ml[i].conserved
        assert np.allclose(
            conserved[i], station_conserved
        ), f"Station {i} conserved mismatch"
    print("PASS: non-scalar property (conserved) works correctly")

    print("\nAll tests passed!")
