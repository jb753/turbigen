"""Mean-line flow field and annulus geometry, built on the ember Block.

A :class:`MeanLine` is an :class:`ember.block.Block` of shape ``(2, n_row)``:

* axis 0 indexes the two stations of a blade row, inlet (0) and outlet (1);
* axis 1 indexes the blade rows.

Because it *is* a Block, every flow property ember defines --- ``Po``, ``Ma``,
``Ma_rel``, the Jacobians, and so on --- is available directly and vectorised
over both axes, with no forwarding layer to keep in sync.

Indexing follows numpy and returns views that share storage, so writes
propagate back to the parent:

* ``ml[0]`` and ``ml[1]`` are the inlet and outlet planes of *every* row, each of
  shape ``(n_row,)``;
* ``ml[:, i]`` --- equivalently ``ml.row(i)`` --- is row ``i``, of shape
  ``(2,)``;
* ``ml[0, i]`` and ``ml[1, i]`` are that row's inlet and outlet stations.

The station axis comes first so that :py:attr:`ember.block.Block.flat` runs in
streamwise order, ``ml.flat`` giving the ``2 * n_row`` stations from machine
inlet to machine outlet. Ember flattens column-major, and the fastest-varying
axis in memory is the one that varies fastest in the flat sequence, so putting
the station axis first is what makes streamwise flattening a writeable view
rather than a copy. That is the interface to everything indexed by station
rather than by row --- the annulus in particular, whose meridional stations run
``m = 1 .. 2 * n_row``.

Two quantities are stored beyond ember's own variables. ``Am``, the meridional
annulus area, from which the rest of the annulus geometry (span, radii,
hub-to-tip ratio) is derived. And ``Omega``, which ember holds as a scalar per
block but which must vary from row to row here; it is stored as nodal data so
that row views carry their own blade speed. Storing it as metadata could never
work, because :py:meth:`ember.block.Block.view` shares the metadata dict with
the parent.

Note that the block radius ``r`` *is* the annulus root-mean-square radius: a
mean line carries one radius per station, and that is the one it carries.
"""

import logging

import numpy as np
import ember.block
import ember.fluid

import turbigen.designer
import turbigen.plugins
import turbigen.util

logger = logging.getLogger("turbigen")

f32 = np.float32


class MeanLine(ember.block.Block):
    """One-dimensional flow field and annulus geometry along the mean line.

    See the module docstring for the layout of the ``(2, n_row)`` shape and for
    the meaning of the block radius.
    """

    _data_keys = ember.block.Block._data_keys + ("Am", "Omega")

    def __init__(self, n_row=None, shape=None):
        """Allocate a mean line.

        Parameters
        ----------
        n_row : int, optional
            Number of blade rows, giving shape ``(2, n_row)``.
        shape : tuple, optional
            Explicit shape, used by ember's ``empty``/``view`` machinery to
            build scalar and row-sized instances. Takes precedence over
            `n_row`.

        """
        if shape is None:
            if n_row is None:
                shape = ()
            elif n_row < 1:
                raise ValueError(f"n_row must be >= 1, got {n_row}")
            else:
                shape = (2, n_row)
        super().__init__(shape=shape)

    def __post_init__(self):
        """Initialise the mean-line-specific variables."""
        super().__post_init__()
        # Omega defaults to zero, mirroring ember's metadata default, and is
        # marked initialised so the relative-frame properties are readable on a
        # stationary row without an explicit set_Omega call.
        self._set_data_by_keys(("Omega",), 0.0)
        # Am has no sensible default: leave it uninitialised so that reading
        # annulus geometry before it is set raises rather than returning junk.
        self._set_data_by_keys(("Am",), np.nan, store_init=False)

    def __repr__(self):
        """Return a string representation of the MeanLine object."""
        return f"MeanLine(shape={self.shape})"

    #
    # STRUCTURE
    #

    def row(self, i_row):
        """Return blade row `i_row` as a shape-(2,) view of inlet and outlet."""
        return self[:, i_row]

    @property
    def n_row(self):
        """Number of blade rows."""
        return self.shape[1] if self.ndim == 2 else 1

    @property
    def inlet(self):
        """Machine inlet, the first station in streamwise order.

        Not to be confused with ``ml[0]``, which is the inlet station of
        every row.
        """
        return self.flat[0]

    @property
    def outlet(self):
        """Machine outlet, the last station in streamwise order.

        Not to be confused with ``ml[1]``, which is the outlet station of
        every row.
        """
        return self.flat[-1]

    #
    # ROTATION
    #
    # Omega is nodal data here, not block metadata as in ember, so that each
    # row can spin at its own speed and keep it under slicing. The property and
    # setter below shadow Block's metadata-backed versions; everything derived
    # from Omega (U, Vt_rel, Po_rel, Ma_rel, I, ...) is a plain numpy
    # expression over Omega_nd and so becomes elementwise for free.
    #

    @property
    def Omega(self):
        r"""Reference frame angular velocity :math:`\Omega` [rad/s], per station."""
        return self._get_data_by_keys(("Omega",))

    @property
    def Omega_nd(self):
        r"""Nondimensional angular velocity :math:`\Omega L_\mathrm{ref}/V_\mathrm{ref}` [--]."""
        return self.Omega * self.L_ref / self.fluid.V_ref

    def set_Omega(self, Omega):
        """Set reference frame angular velocity [rad/s], broadcast over stations."""
        self._set_data_by_keys(("Omega",), np.broadcast_to(Omega, self.shape))

    def set_Omega_row(self, Omega_row):
        """Set one angular velocity per blade row [rad/s]."""
        if self.ndim != 2:
            raise ValueError("set_Omega_row requires a full (2, n_row) mean line")
        self.set_Omega(np.asarray(Omega_row).reshape(1, -1))

    #
    # ANNULUS GEOMETRY
    #

    @property
    def Am(self):
        """Annulus area projected in the meridional direction [m^2]."""
        return self._get_data_by_keys(("Am",)) * self.L_ref**2

    def set_Am(self, Am):
        """Set annulus area projected in the meridional direction [m^2]."""
        self._set_data_by_keys(("Am",), np.asarray(Am) / self.L_ref**2)

    @property
    def cosBeta(self):
        """Cosine of the pitch angle [--]."""
        return np.cos(np.radians(self.Beta))

    @property
    def r_cas(self):
        """Annulus casing radius [m]."""
        return np.sqrt(self.Am * self.cosBeta / 2.0 / np.pi + self.r**2.0)

    @property
    def r_hub(self):
        """Annulus hub radius [m]."""
        return np.sqrt(self.r**2.0 - self.Am * self.cosBeta / 2.0 / np.pi)

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

    @property
    def mdot(self):
        """Annulus mass flow rate [kg/s]."""
        return self.rho * self.Vm * self.Am

    def set_span_htr(self, span, htr):
        """Define annulus geometry using span and hub-to-tip ratio."""
        if not np.all(np.abs(self.Beta) < 1.0):
            raise ValueError("Beta must be set zero before calling set_span_htr")
        self.set_r(span * np.sqrt(0.5 * (1.0 + htr**2)) / (1.0 - htr))
        self.set_Am(2.0 * np.pi * self.r**2 * (1.0 - htr**2) / (1.0 + htr**2))

    def set_span_r_rms(self, span, r_rms):
        """Define annulus geometry using span and root-mean-square radius."""
        self.set_r(r_rms)
        dr = span / self.cosBeta
        r_mid = np.sqrt(np.asarray(r_rms) ** 2 - (dr / 2.0) ** 2)
        self.set_Am(2.0 * np.pi * r_mid * span)

    def set_span_r_mid(self, span, r_mid):
        """Define annulus geometry using span and mid-span radius."""
        self.set_Am(2.0 * np.pi * np.asarray(r_mid) * span)
        self.set_r(np.sqrt(np.asarray(r_mid) ** 2 + (np.asarray(span) / 2.0) ** 2))

    #
    # REFERENCE SCALES
    #

    def set_L_ref(self, L_ref):
        """Set the reference length scale [m], rescaling the stored area."""
        fac_L = L_ref / self.L_ref
        Am = self._get_data_by_keys(("Am",), raise_uninit=False)
        self._set_data_by_keys(("Am",), Am / fac_L**2, store_init=False)
        super().set_L_ref(L_ref)

    def adjust_ref(self, L_ref):
        """Set fluid references and L_ref from the current design.

        Parameters
        ----------
        L_ref : float
            Reference length to use for non-dimensionalisation [m].

        Returns
        -------
        fluid_ref : Fluid
            The new fluid object set on the mean line.

        """
        fluid_ref = self.fluid.change_ref(
            rho_ref=self.rho.mean(),
            V_ref=self.V.mean(),
            Rgas_ref=self.Rgas.mean(),
        ).change_datum(
            P_dtm=self.P.mean(),
            T_dtm=(self.T + (self.P / self.rho + self.halfVsq) / self.cv).mean(),
        )

        self.set_L_ref(L_ref)
        self.set_fluid(fluid_ref)

        return fluid_ref

    #
    # OVERALL PERFORMANCE
    #

    @property
    def halfVsq(self):
        """Specific kinetic energy in the stationary frame [J/kg]."""
        return 0.5 * self.V**2

    @property
    def halfVsq_rel(self):
        """Specific kinetic energy in the rotating frame [J/kg]."""
        return 0.5 * self.V_rel**2

    @property
    def PR_ts(self):
        """Total-to-static pressure ratio."""
        return self.inlet.Po / self.outlet.P

    @property
    def PR_tt(self):
        """Total-to-total pressure ratio."""
        return self.inlet.Po / self.outlet.Po

    def _eta(self, P_ideal):
        """Isentropic efficiency for an ideal outlet at the given pressure.

        Dispatches on the sign of the stagnation enthalpy change, so
        compressors get ideal/actual work and turbines actual/ideal, rather
        than computing one and inverting it whenever it exceeds unity.
        """
        inlet, outlet = self.inlet, self.outlet

        ideal = inlet.empty()
        ideal.set_P_s(P_ideal, inlet.s)

        dho = outlet.ho - inlet.ho
        dho_ideal = ideal.h - inlet.ho

        with np.errstate(divide="ignore", invalid="ignore"):
            # Work in: the ideal work is the smaller. Work out: the larger.
            eta = dho_ideal / dho if dho > 0.0 else dho / dho_ideal

        if np.isnan(eta):
            return np.inf

        return float(eta)

    @property
    def eta_tt(self):
        """Total-to-total isentropic efficiency."""
        return self._eta(self.outlet.Po)

    @property
    def eta_ts(self):
        """Total-to-static isentropic efficiency."""
        return self._eta(self.outlet.P)

    #
    # CONVERSION
    #

    def to_block(self, ann):
        """Convert mean-line stations to a 1D Block along the annulus midspan.

        Parameters
        ----------
        ann : Annulus
            Annulus geometry for evaluating (x, r) coordinates.

        Returns
        -------
        ember.block.Block
            1D block with shape ``(2 * n_row,)`` in streamwise station order.

        """
        flat = self.flat
        n = flat.size
        xr = np.array(ann._xr_stations()).mean(0).T
        xrt = np.append(xr, np.zeros((n, 1)), axis=1)
        b = ember.block.Block(shape=(n,))
        b.set_fluid(self.fluid)
        b.set_xrt(xrt)
        b.set_conserved(flat.conserved)
        b.set_mu_turb(np.full(n, np.mean(self.mu)))
        return b

    def to_quasi3d(self, ann, Nb, n=101, nj=11):
        """Generate a quasi-3D initial guess as a Block of shape (n, nj, 2).

        Axes:
          - axis 0 (i, streamwise):  n points from inlet to outlet
          - axis 1 (j, radial):      nj points from hub (spf=0) to tip (spf=1)
          - axis 2 (k, pitchwise):   2 points -- index 0 low-theta, index 1 high-theta

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

        # Station quantities in streamwise order, matching the annulus
        ml = self.flat

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

        P_q = interp(ml.P)
        s_q = interp(ml.s)
        rho_q = interp(ml.rho)
        Vm_q = interp(ml.Vm)
        rVt_q = interp(ml.r * ml.Vt)
        Vx_q = interp(ml.Vx)
        Vr_q = interp(ml.Vr)
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
            # Station indices in streamwise order: inlet = 2*irow, outlet = +1
            i_in = 2 * irow
            i_out = 2 * irow + 1

            # Change in rVt across the row
            delta_rVt = ml.r[i_out] * ml.Vt[i_out] - ml.r[i_in] * ml.Vt[i_in]

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
        P_ref = 1e-6 * ml.P.mean()
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
        b.set_x(x_3d)
        b.set_r(r_3d)
        b.set_t(t_3d)
        b.set_P_s(P_3d, s_3d)
        b.set_Vx(Vx_3d)
        b.set_Vr(Vr_3d)
        b.set_Vt(Vt_3d)

        b.set_mu_turb(np.full((n, nj, 2), float(np.mean(self.mu))))

        return b

    def to_string(self):
        """Provide a concise string representation of MeanLine properties."""
        ml = self.flat
        properties = [
            ("Po/bar", ml.Po / 1e5, ".3f"),
            ("To/K", ml.To, ".2f"),
            ("Ma", ml.Ma, ".3f"),
            ("Ma_rel", ml.Ma_rel, ".3f"),
            ("Alpha/deg", ml.Alpha, ".1f"),
            ("Alpha_rel/deg", ml.Alpha_rel, ".1f"),
        ]
        return turbigen.util.format_table("Mean line:", self.n_row, properties)


class MeanLineConfig:
    """Configuration for a MeanLine object.

    Holds the design variables read from the config file, the designer selected
    by `type`, and the nominal and actual mean lines produced from them.
    """

    def __init__(self, mean_line_type, n_row, design_vars):
        """Initialize the configuration."""
        self.type = mean_line_type

        reg = turbigen.plugins.get_registry()
        try:
            self.designer = reg["designer"][mean_line_type]
        except KeyError:
            raise ValueError(
                f"Unknown mean_line type '{mean_line_type}'. "
                f"Available types: {sorted(reg['designer'])}"
            ) from None

        if n_row != self.designer.n_row:
            raise ValueError(
                f"mean_line type '{mean_line_type}' designs {self.designer.n_row} "
                f"row(s), but the configuration asks for n_row={n_row}."
            )
        self.n_row = n_row

        # Store the design variables with defaults filled in, so that a config
        # written back out records every value the design actually used.
        self.design_vars = turbigen.designer.resolve_defaults(
            self.designer, design_vars
        )

        # Allocate placeholders for nominal and actual mean lines
        self.nominal = MeanLine(n_row)
        self.actual = MeanLine(n_row)

    @property
    def valid_design_params(self):
        """Valid design variable names, as ``{'required': ..., 'all': ...}``."""
        params = turbigen.designer.design_params(self.designer)
        return {
            "required": {
                k for k, v in params.items() if v is turbigen.designer.REQUIRED
            },
            "all": set(params),
        }

    @classmethod
    def from_dict(cls, d):
        """Initialize from a dictionary, which is not modified."""

        turbigen.plugins.check_plugins()

        d = dict(d)
        mean_line_type = d.pop("type", None)
        n_row = d.pop("n_row", None)

        reg = turbigen.plugins.get_registry()
        all_types = sorted(reg["designer"])

        if not mean_line_type:
            raise ValueError(
                f"mean_line configuration requires a 'type' key. "
                f"Available types: {all_types}"
            )
        if mean_line_type not in reg["designer"]:
            raise ValueError(
                f"Unknown mean_line type '{mean_line_type}'. "
                f"Available types: {all_types}"
            )

        if n_row is None:
            raise ValueError("mean_line configuration requires an 'n_row' key.")
        if n_row < 1:
            raise ValueError(f"n_row must be >= 1, got {n_row}")

        # Remaining keys are design variables
        return cls(mean_line_type, n_row, d)

    def to_dict(self):
        """Convert to a dictionary, including resolved defaults."""
        return {
            "type": self.type,
            "n_row": self.n_row,
            **self.design_vars,
        }

    def set_nominal(self, fluid):
        """Set the nominal mean-line flow field."""
        self.nominal.set_fluid(fluid)
        self.designer.forward(self.nominal, **self.design_vars)

    def check_nominal(self):
        """Verify the nominal mean line reproduces its design variables."""
        turbigen.designer.check_round_trip(
            self.designer, self.nominal, self.design_vars
        )

    def warn(self):
        """Print a warning if there are any suspicious values."""

        ml = self.nominal

        # Warn for very high flow angles
        if np.abs(ml.Alpha_rel).max() > 85.0:
            logger.warning(
                """WARNING: Relative flow angles are approaching 90 degrees.
This suggests a physically-consistent but suboptimal mean-line design
and will cause problems with meshing and solving for the flow field."""
            )

        # Warn for wobbly annulus
        is_radial = np.abs(ml.Beta).max() > 10.0
        is_multirow = ml.n_row > 2
        if is_radial and is_multirow:
            if np.diff(np.sign(np.diff(ml.flat.r))).any():
                logger.warning(
                    """WARNING: Radii do not vary monotonically.
This suggests a physically-consistent but suboptimal mean-line design
and will cause problems with meshing and solving for the flow field."""
                )
