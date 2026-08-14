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

    def ref(self, i_row):
        """Return the reference station of blade row `i_row`.

        The end with the smaller meridional flow area, taken as representative
        of the row for wall spacings and Reynolds numbers. Note that this is an
        area criterion, not a velocity one: it picks the inlet of a row whose
        annulus opens out, even where the flow accelerates through it.
        """
        row = self.row(i_row)
        A_flow = row.Am / row.cosBeta
        return row[0] if A_flow[1] >= A_flow[0] else row[1]

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
        """Annulus area normal to the meridional direction [m^2].

        The true area of the annular surface the flow crosses, ``2 pi r_mid
        span``, not a projection of it: ``mdot = rho Vm Am`` exactly. At a pitch
        angle the surface is inclined to the axis, so it is larger than the
        ``pi (r_cas**2 - r_hub**2)`` an axial view of the annulus would give, by
        one over ``cos(Beta)``.
        """
        return self._get_data_by_keys(("Am",)) * self.L_ref**2

    def set_Am(self, Am):
        """Set annulus area normal to the meridional direction [m^2].

        Note that this is ``mdot / rho / Vm``. A design that writes
        ``mdot / rho / Vx`` is correct only where the flow is axial.
        """
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

    def referenced_fluid(self):
        """Return an equation of state scaled and datumed to this mean line.

        The scales come from the design itself: mean density, velocity and gas
        constant, with the datum placed near the mean thermodynamic state so
        that internal energy and kinetic energy are comparable rather than one
        being lost in the rounding of the other.

        This *returns* rather than applies, which is what lets it work on a
        frozen mean line, and it takes no reference length, which is what frees
        it from needing any geometry. The caller decides what to put it on --
        in practice the grid, since that is the object a solver iterates on. A
        mean line is only ever read dimensionally, so its own scales do not
        matter.

        Returns
        -------
        fluid : Fluid
            A new equation of state; this mean line is left unchanged.

        """
        return self.fluid.change_ref(
            rho_ref=self.rho.mean(),
            V_ref=self.V.mean(),
            Rgas_ref=self.Rgas.mean(),
        ).change_datum(
            P_dtm=self.P.mean(),
            T_dtm=(self.T + (self.P / self.rho + self.halfVsq) / self.cv).mean(),
        )

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
    # SERIALISATION
    #
    # A mean line is a result, not a config node, so it has no `type` and never
    # appears among a config's own keys. It does serialise, though: a run's
    # answer has to be readable back without repeating the CFD that produced
    # it.
    #

    STATE = ("P", "T", "Vx", "Vr", "Vt", "r", "Am", "Omega")
    """The quantities that make up a complete mean line.

    Not obvious from the data keys, which is why the pair below lives here
    rather than on whatever wants to write one. A mean line inherits twelve
    keys from :class:`ember.block.Block` and deliberately leaves four of them
    unset --- ``x``, ``t``, ``mu_turb`` and ``wdist`` all raise on read --- so
    the complete state is these eight and no others.

    Pressure and temperature rather than the conserved variables, because
    conserved energy is measured from its fluid's datum: copied into a block
    whose fluid has a different one it is silently reinterpreted, which on a
    realistic design is a hundred kelvin. Everything here is dimensional and
    so crosses a fluid boundary unchanged.
    """

    @classmethod
    def from_dict(cls, data, fluid):
        """Build a mean line from `data` and an equation of state.

        Parameters
        ----------
        data : dict
            As produced by :meth:`to_dict`.
        fluid : Fluid
            Equation of state to read the stored state against. Required, and
            an argument rather than stored, because the numbers in `data` are
            dimensional and mean nothing without one.

        """
        missing = [key for key in cls.STATE if key not in data]
        if missing:
            raise ValueError(f"Mean line state is missing {missing}.")

        values = {key: np.asarray(data[key], dtype=float) for key in cls.STATE}
        n_station = values["P"].size
        if n_station % 2:
            raise ValueError(
                f"A mean line has two stations per row, so an even number of "
                f"them; got {n_station}."
            )

        ml = cls(n_row=n_station // 2)
        ml.set_fluid(fluid)

        flat = ml.flat
        flat.set_r(values["r"])
        flat.set_P_T(values["P"], values["T"])
        flat.set_Vx(values["Vx"])
        flat.set_Vr(values["Vr"])
        flat.set_Vt(values["Vt"])
        flat.set_Am(values["Am"])
        flat.set_Omega(values["Omega"])

        return ml

    def to_dict(self):
        """Return this mean line's complete state, in streamwise order."""
        flat = self.flat
        return {
            key: np.asarray(getattr(flat, key), dtype=float).tolist()
            for key in self.STATE
        }

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
