import logging

"""New meanline data struct using ember base."""

import ember.block
import ember.fluid
import numpy as np
import turbigen.plugins
import turbigen.meanline_design_new
import turbigen.util
import inspect

logger = logging.getLogger("turbigen")


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
        required = self.valid_design_params["required"]
        all_valid = self.valid_design_params["all"]
        missing = required - set(design_vars.keys())
        if missing:
            raise ValueError(
                f"Missing required design variables for mean_line type '{self.type}': {missing}"
            )

        # Check for unexpected design_vars
        unexpected = set(design_vars.keys()) - all_valid
        if unexpected:
            raise ValueError(
                f"Unexpected design variables for mean_line type '{self.type}': {unexpected}"
            )

    @property
    def valid_design_params(self):
        """Valid design variable names for the mean-line forward function.

        Returns a dict with keys 'required' (no default) and 'all' (all params).
        """
        sig = inspect.signature(self._forward)
        params = list(sig.parameters.values())[1:]  # Skip 'mean_line' arg
        return {
            "required": {p.name for p in params if p.default is p.empty},
            "all": {p.name for p in params},
        }

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

        rtol = 0.5e-2

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
            except (ValueError, AttributeError):
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

        return self

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

    def set_L_ref(self, L_ref):
        """Set the reference length scale."""
        for station in self._stations:
            station.set_L_ref(L_ref)
        return self

    def adjust_ref(self, L_ref):
        """Set fluid references and L_ref from the current design.

        Parameters
        ----------
        L_ref : float
            Reference length to use for non-dimensionalisation [m].

        Returns
        -------
        fluid_ref : Fluid
            The new fluid object set on all blocks.
        """

        rho_ref = self.rho.mean()
        V_ref = self.V.mean()
        Rgas_ref = self.Rgas.mean()
        P_dtm = self.P.mean()
        T_dtm = (self.T + (self.P / self.rho + self.halfVsq) / self.cv).mean()

        fluid_ref = self.fluid.change_ref(
            rho_ref=rho_ref,
            V_ref=V_ref,
            Rgas_ref=Rgas_ref,
        ).change_datum(P_dtm=P_dtm, T_dtm=T_dtm)

        self.set_L_ref(L_ref)
        self.set_fluid(fluid_ref)

        return fluid_ref

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

    @property
    def L_ref(self):
        """Reference length scale [m]."""
        return self._stations[0].L_ref

    def copy(self):
        """Create a deep copy of the MeanLine."""
        new_ml = MeanLine(self.n_row)
        for i in range(self.n_row * 2):
            new_ml._stations[i] = self._stations[i].copy()
        return new_ml

    @property
    def PR_ts(self):
        """Total-to-static pressure ratio."""
        return self.Po[0] / self.P[-1]

    @property
    def PR_tt(self):
        """Total-to-total pressure ratio."""
        return self.Po[0] / self.Po[-1]

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
    rt = _make_concat_property("rt")
    xr = _make_concat_property("xr")
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

    @property
    def halfVsq(self):
        return 0.5 * self.V**2

    @property
    def halfVsq_rel(self):
        return 0.5 * self.V_rel**2

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
    Rgas = _make_concat_property("Rgas")

    # Stagnation properties
    ho = _make_concat_property("ho")
    ao = _make_concat_property("ao")
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
    Mam = _make_concat_property("Mam")
    Max = _make_concat_property("Max")

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

    # Relative velocity components
    Vxrt_rel = _make_concat_property("Vxrt_rel")

    # Rotation-corrected pressure
    P_rot = _make_concat_property("P_rot")

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

    def to_quasi3d(self, ann, Nb, n=101, nj=11):
        """Generate a quasi-3D initial guess as a Block of shape (n, nj, 2).

        Axes:
          - axis 0 (i, streamwise):  n points from inlet to outlet
          - axis 1 (j, radial):      nj points from hub (spf=0) to tip (spf=1)
          - axis 2 (k, pitchwise):   2 points — index 0 low-theta, index 1 high-theta

        Inside each blade row, a pitchwise pressure difference is imposed between
        the two sides consistent with the meanline angular momentum change.  The
        balance on a control volume of axial length dx, height span, and pitchwise
        width pitch = 2*pi*r/Nb gives:

            (p_low - p_high) * span * dx * r = pitch * span * rho * Vm * d(rVt)

        so (r cancels with pitch = 2*pi*r/Nb):

            dp = rho * Vm * d(rVt)/dx * 2*pi / Nb

        Outside blades the two sides have equal pressure.

        A radial equilibrium correction is also applied.  Integrating
        dP/dr = rho * Vt**2 / r with Vt uniform over r gives:

            dP_rad(r) = rho * Vt**2 * ln(r / r_mid)

        where r_mid is the midspan radius at each streamwise station.

        Parameters
        ----------
        ann : AnnulusDesigner
            Annulus geometry for evaluating (x, r) coordinates.
        Nb : array-like, length n_row
            Number of blades in each row.
        n : int, optional
            Number of streamwise grid points.  Default 101.
        nj : int, optional
            Number of radial grid points (hub to tip).  Default 11.

        Returns
        -------
        ember.block.Block
            Block of shape (n, nj, 2).
        """
        nrow = self.n_row
        Nb = np.asarray(Nb, dtype=float).ravel()

        # Normalised meridional coordinates of the meanline stations (m=1..2*nrow)
        m_stations = np.arange(1, 2 * nrow + 1, dtype=float)

        # Query coordinates uniformly spanning [0, mmax]
        m_query = np.linspace(0.0, ann.mmax, n)  # (ni,)
        spf_query = np.linspace(0.0, 1.0, nj)  # (nj,)

        # --- 2D (nj, ni) coordinate grid via broadcasting --------------------
        # evaluate_xr broadcasts (1, ni) x (nj, 1) -> (2, nj, ni)
        xr_2d = ann.evaluate_xr(m_query[np.newaxis, :], spf_query[:, np.newaxis])
        x_2d = xr_2d[0].T  # (ni, nj)
        r_2d = xr_2d[1].T  # (ni, nj)

        # Midspan row (spf=0.5) for meanline interpolation and radial reference
        j_mid = nj // 2
        x_q = x_2d[:, j_mid]  # (ni,)
        r_q = r_2d[:, j_mid]  # (ni,)

        # --- interpolate meanline scalars to streamwise query points ---------
        def interp(vals):
            return np.interp(m_query, m_stations, vals.astype(float))

        P_q = interp(self.P)
        s_q = interp(self.s)
        rho_q = interp(self.rho)
        Vm_q = interp(self.Vm)
        rVt_q = interp(self.r * self.Vt)
        Vx_q = interp(self.Vx)
        Vr_q = interp(self.Vr)
        Vt_q = rVt_q / np.where(r_q > 0, r_q, 1.0)  # (ni,)

        # --- blade loading shape function ------------------------------------
        # Smooth trapezoid: quadratic ramps at LE (r1) and TE (r2) with zero
        # slope at the flat-top junctions, constant in between.
        # Front: f(0)=0, f(r1)=1, f'(r1)=0  -> f = xi*(2*r1 - xi) / r1**2
        # Back:  g(1-r2)=1, g'(1-r2)=0, g(1)=0 -> g = (1-xi)*(2*r2-(1-xi)) / r2**2
        # Normalisation: integral = 1 - r1/3 - r2/3
        r1, r2 = 0.3, 0.4
        h = 1.0 / (1.0 - r1 / 3.0 - r2 / 3.0)

        def shape_func(xi):
            front = h * xi * (2.0 * r1 - xi) / r1**2
            back = h * (1.0 - xi) * (2.0 * r2 - (1.0 - xi)) / r2**2
            return np.where(xi < r1, front, np.where(xi > 1.0 - r2, back, h))

        # --- compute blade-loading pressure difference for each row ----------
        dp_q = np.zeros(n)  # pitchwise dp at midspan, shape (ni,)

        for irow in range(nrow):
            # Station indices: inlet = 2*irow, exit = 2*irow+1
            i_in = 2 * irow
            i_out = 2 * irow + 1

            # Change in rVt across the row
            delta_rVt = self.r[i_out] * self.Vt[i_out] - self.r[i_in] * self.Vt[i_in]

            # Mask for query points inside this blade row
            m_in = m_stations[i_in]
            m_out = m_stations[i_out]
            mask = (m_query >= m_in) & (m_query <= m_out)
            if not mask.any():
                continue

            # Axial chord of the blade row at midspan
            xr_in = ann.evaluate_xr(np.array([m_in]), spf=0.5)
            xr_out = ann.evaluate_xr(np.array([m_out]), spf=0.5)
            x_chord = float(xr_out[0, 0]) - float(xr_in[0, 0])
            if abs(x_chord) < 1e-12:
                continue

            # Normalised chord coordinate xi in [0, 1] for blade query points
            xi = (x_q[mask] - float(xr_in[0, 0])) / x_chord

            # dp_mean is the uniform loading that would satisfy the integral;
            # the trapezoid shape redistributes it while preserving the integral.
            dp_mean = (
                rho_q[mask]
                * Vm_q[mask]
                * (delta_rVt / x_chord)
                * 2.0
                * np.pi
                / Nb[irow]
            )
            dp_q[mask] = dp_mean * shape_func(xi)

        # --- radial equilibrium pressure correction --------------------------
        # dP/dr = rho * Vt**2 / r  with Vt uniform over r
        # => dP_rad(r) = rho * Vt**2 * ln(r / r_mid)
        # Shape: (ni, nj)
        dP_rad = (
            rho_q[:, np.newaxis]
            * Vt_q[:, np.newaxis] ** 2
            * np.log(r_2d / r_q[:, np.newaxis])
        )

        # --- assemble (ni, nj) pressure fields -------------------------------
        # Mean pressure including radial equilibrium correction
        P_mean = P_q[:, np.newaxis] + dP_rad  # (ni, nj)
        P_lo = P_mean + 0.5 * dp_q[:, np.newaxis]  # (ni, nj)
        P_hi = P_mean - 0.5 * dp_q[:, np.newaxis]  # (ni, nj)

        # Clamp to positive pressure
        P_ref = 1e-6 * self.P.mean()
        P_lo = np.maximum(P_lo, P_ref)
        P_hi = np.maximum(P_hi, P_ref)

        # --- build shape-(n, nj, 2) arrays -----------------------------------
        # Pitchwise theta: low-theta at zero, high-theta offset by a small
        # token amount so the two sides are geometrically distinguishable.
        t_lo = np.zeros((n, nj))
        t_hi = np.full((n, nj), 1e-6)

        x_3d = np.stack([x_2d, x_2d], axis=-1)  # (ni, nj, 2)
        r_3d = np.stack([r_2d, r_2d], axis=-1)  # (ni, nj, 2)
        t_3d = np.stack([t_lo, t_hi], axis=-1)  # (ni, nj, 2)
        P_3d = np.stack([P_lo, P_hi], axis=-1)  # (ni, nj, 2)

        s_2d = np.broadcast_to(s_q[:, np.newaxis], (n, nj))
        s_3d = np.stack([s_2d, s_2d], axis=-1)  # (ni, nj, 2)

        Vx_2d = np.broadcast_to(Vx_q[:, np.newaxis], (n, nj))
        Vr_2d = np.broadcast_to(Vr_q[:, np.newaxis], (n, nj))
        Vt_2d = np.broadcast_to(Vt_q[:, np.newaxis], (n, nj))
        Vx_3d = np.stack([Vx_2d, Vx_2d], axis=-1)  # (ni, nj, 2)
        Vr_3d = np.stack([Vr_2d, Vr_2d], axis=-1)  # (ni, nj, 2)
        Vt_3d = np.stack([Vt_2d, Vt_2d], axis=-1)  # (ni, nj, 2)

        # --- build Block of shape (n, nj, 2) ---------------------------------
        b = ember.block.Block(shape=(n, nj, 2))
        b.set_fluid(self.fluid)
        b.set_x(x_3d).set_r(r_3d).set_t(t_3d)
        b.set_P_s(P_3d, s_3d)
        b.set_Vx(Vx_3d).set_Vr(Vr_3d).set_Vt(Vt_3d)

        mu_mean = float(np.mean(self.mu))
        b.set_mu_turb(np.full((n, nj, 2), mu_mean))

        return b

    def to_block(self, ann):
        """Convert mean-line stations to a 1D Block along the annulus midspan.

        Parameters
        ----------
        ann : Annulus
            Annulus geometry for evaluating (x, r) coordinates.

        Returns
        -------
        ember.block.Block
            1D block with shape (n_stations,).
        """
        n = self.n_row * 2
        xr = np.array(ann._xr_stations()).mean(0).T
        b = ember.block.Block(shape=(n,))
        b.set_fluid(self.fluid)
        b.set_xrt(xr[:, 0], xr[:, 1], np.zeros(n))
        b.set_conserved(self.conserved)
        b.set_mu_turb(np.full(n, np.mean(self.mu)))
        return b

    def __repr__(self):
        """Return a string representation of the MeanLine object."""
        return f"MeanLine(n_row={self.n_row})"

    def to_string(self):
        """Provide a concise string representation of MeanLine properties."""
        properties = [
            ("Po/bar", self.Po / 1e5, ".3f"),
            ("To/K", self.To, ".2f"),
            ("Ma", self.Ma, ".3f"),
            ("Ma_rel", self.Ma_rel, ".3f"),
            ("Alpha/deg", self.Alpha, ".1f"),
            ("Alpha_rel/deg", self.Alpha_rel, ".1f"),
        ]
        return turbigen.util.format_table("Mean line:", self.n_row, properties)


class Station(ember.block.Block):
    """A single station in a mean-line flow path."""

    _data_keys = ember.block.Block._data_keys + ("Am",)

    def __post_init__(self):
        """Initialize Station and verify it is scalar."""
        if self.shape != ():
            raise ValueError(f"Station must be a scalar, got shape {self.shape}")
        super().__post_init__()

    def set_L_ref(self, L_ref):
        """Set the reference length scale."""
        fac_L = L_ref / self.L_ref
        Am = self._get_data_by_keys(("Am",))
        self._set_data_by_keys(("Am",), Am / fac_L**2, store_init=False)
        super().set_L_ref(L_ref)
        return self

    @property
    def Am(self):
        """Annulus area projected in meridional direction [m^2]."""
        return self._get_data_by_keys(("Am",)) * self.L_ref**2

    @property
    def mdot(self):
        """Annulus mass flow rate [kg/s]"""
        return self.rho * self.Vm * self.Am

    @property
    def r_rms(self):
        """Annulus root-mean-square radius [m]."""
        return self._get_data_by_keys(("r",)) * self.L_ref

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
        self._set_data_by_keys(("Am",), Am / self.L_ref**2)
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
