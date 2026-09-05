"""Mean-line flow field, radius and annulus area.

This module contains the basic data structure used in :program:`turbigen` to
represent a one-dimensional flow field along the mean line of a turbomachine,
holding averaged flow variables at the inlet and outlet of each blade row and
minimal geometry. When combined with annulus and blade shape choices, a :class:`MeanLine` is sufficient to define a complete turbomachine design.

A :class:`MeanLine` is an :class:`ember.block.Block` of shape ``(2, n_row)``;
; full documentation of the base class is in :mod:`ember.block`.

Two quantities are stored beyond the base class variables. :attr:`MeanLine.Am`
is the meridional annulus area, from which the rest of the annulus geometry
(span, radii, hub-to-tip ratio) is derived. :attr:`MeanLine.Omega` is the
reference-frame angular velocity, which must vary from row to row here and is
stored as nodal
data.

Note that stored radius :attr:`ember.block.Block.r` is by convention the
root-mean-square radius on a :class:`MeanLine`. To make this explicit use the
:attr:`MeanLine.r_rms` alias and its setter, :meth:`MeanLine.set_r_rms`.

Indexing
^^^^^^^^

Mean lines are always two-dimensional with shape ``(2, n_row)`` where

* axis 0 indexes the two stations of a blade row, inlet (0) and outlet (1);
* axis 1 indexes the blade rows.

Every flow property the base class defines --- ``Po``, ``Ma``, ``Ma_rel``, the
Jacobians, and so on --- is available directly and vectorised over both axes.

Indexing follows numpy and returns views that share storage, so writes
propagate back to the parent:

* ``ml[0]`` and ``ml[1]`` are the inlet and outlet planes of every row, each of
  shape ``(n_row,)``;
* ``ml[:, i]`` is row ``i``, of shape ``(2,)``;
* ``ml[0, i]`` and ``ml[1, i]`` are that row's inlet and outlet stations.

The following properties return useful views of the mean line:

.. autosummary::

   MeanLine.flat
   MeanLine.inlet
   MeanLine.outlet

Building a mean line
^^^^^^^^^^^^^^^^^^^^^

In practice, the design flow is to call
:meth:`~turbigen.design.MeanLineDesign.allocate` to initialise a mean line of
the correct size and working fluid, and then fill in the data using setter
methods --- see :doc:`/design` for the designer's own reference. See the
:doc:`/tutorial` for a full worked example.

A handful of the base class's own setters cover most of what is
needed:

.. list-table::
   :widths: 40 60

   * - :meth:`~ember.block.Block.set_P_T`
     - Store static pressure and temperature.
   * - :meth:`~ember.block.Block.set_P_s`
     - Store static pressure and entropy.
   * - :meth:`~ember.block.Block.set_h_s`
     - Store enthalpy and entropy.
   * - :meth:`~ember.block.Block.set_Vx`
     - Store axial velocity.
   * - :meth:`~ember.block.Block.set_Vr`
     - Store radial velocity.
   * - :meth:`~ember.block.Block.set_Vt`
     - Store circumferential velocity.

The :class:`MeanLine` adds the following setters:

.. autosummary::

   MeanLine.set_Am
   MeanLine.set_Omega
   MeanLine.set_r_rms
   MeanLine.set_span_htr
   MeanLine.set_span_r_mid
   MeanLine.set_span_r_rms


Derived properties
^^^^^^^^^^^^^^^^^^

Once the mean line flow field has been filled in, either by a design routine or by averaging a CFD solution, we can access most of the derived properties provided by the base class. The more useful ones are listed below:

.. list-table::
   :widths: 40 60

   * - :attr:`~ember.block.Block.P`
     - Static pressure [Pa].
   * - :attr:`~ember.block.Block.T`
     - Static temperature [K].
   * - :attr:`~ember.block.Block.rho`
     - Mass density [kg/m^3].
   * - :attr:`~ember.block.Block.h`
     - Static enthalpy [J/kg].
   * - :attr:`~ember.block.Block.s`
     - Specific entropy [J/kg/K].
   * - :attr:`~ember.block.Block.Po`
     - Stagnation pressure [Pa].
   * - :attr:`~ember.block.Block.To`
     - Stagnation temperature [K].
   * - :attr:`~ember.block.Block.ho`
     - Stagnation enthalpy [J/kg].
   * - :attr:`~ember.block.Block.Po_rel`
     - Relative-frame stagnation pressure [Pa].
   * - :attr:`~ember.block.Block.To_rel`
     - Relative-frame stagnation temperature [K].
   * - :attr:`~ember.block.Block.ho_rel`
     - Relative-frame stagnation enthalpy [J/kg].
   * - :attr:`~ember.block.Block.Vx`
     - Axial velocity [m/s].
   * - :attr:`~ember.block.Block.Vr`
     - Radial velocity [m/s].
   * - :attr:`~ember.block.Block.Vt`
     - Tangential velocity [m/s].
   * - :attr:`~ember.block.Block.Vm`
     - Meridional velocity magnitude [m/s].
   * - :attr:`~ember.block.Block.V`
     - Absolute velocity magnitude [m/s].
   * - :attr:`~ember.block.Block.V_rel`
     - Relative velocity magnitude [m/s].
   * - :attr:`~ember.block.Block.U`
     - Blade speed [m/s].
   * - :attr:`~ember.block.Block.Alpha`
     - Absolute yaw angle [deg].
   * - :attr:`~ember.block.Block.Alpha_rel`
     - Relative-frame yaw angle [deg].
   * - :attr:`~ember.block.Block.a`
     - Acoustic speed [m/s].
   * - :attr:`~ember.block.Block.Ma`
     - Absolute Mach number [-].
   * - :attr:`~ember.block.Block.Ma_rel`
     - Relative-frame Mach number [-].

The subclass provide these additional properties, which are mean-line specific geometry, integrated flow variables, and performance metrics:

.. autosummary::

   MeanLine.Am
   MeanLine.Dho
   MeanLine.Dhos_ts
   MeanLine.Dhos_tt
   MeanLine.eta_ts
   MeanLine.eta_tt
   MeanLine.halfVsq
   MeanLine.halfVsq_rel
   MeanLine.htr
   MeanLine.mdot
   MeanLine.n_row
   MeanLine.Omega
   MeanLine.PR_ts
   MeanLine.PR_tt
   MeanLine.r_cas
   MeanLine.r_hub
   MeanLine.r_mid
   MeanLine.r_rms
   MeanLine.span

Reference scales and thermodynamic datum
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Only changes in the thermodynamic properties :attr:`~ember.block.Block.u`,
:attr:`~ember.block.Block.s`, and by extension :attr:`~ember.block.Block.ho`
and so on are physically meaningful. Therefore, the datum level is arbitrary
as discussed in :ref:`ember:datum-state`.

The base class stores data in non-dimensional form against reference scales,
according to :ref:`ember:reference-scales`. It is not neccesary to set these on
a :class:`MeanLine`, but downstream code uses
:meth:`MeanLine.get_referenced_fluid` to get a fluid scaled and datumed to the
mean line design, to give optimal numerical conditioning for a CFD solve.

"""

import logging

import ember.block
import numpy as np

import turbigen.util

logger = logging.getLogger("turbigen")


class MeanLine(ember.block.Block):
    """One-dimensional flow field and annulus geometry along the mean line.

    An :class:`ember.block.Block` of shape ``(2, n_row)``: axis 0 indexes the
    inlet and outlet stations of a blade row, axis 1 the rows. Every base class
    flow property is therefore available and vectorised over both axes. The
    :class:`MeanLine` adds annulus geometry, a per-station :attr:`Omega`, and
    overall performance metrics."""

    _data_keys = ember.block.Block._data_keys + ("Am", "Omega")

    _STATE = ("P", "T", "Vx", "Vr", "Vt", "r", "Am", "Omega")
    """The quantities that make up a complete mean line.

    Not obvious from the data keys, which is why this lives here rather than
    on whatever wants to write one. A mean line inherits twelve keys from
    :class:`ember.block.Block` and deliberately leaves four of them unset ---
    ``x``, ``t``, ``mu_turb`` and ``wdist`` all raise on read --- so the
    complete state is these eight and no others.

    Pressure and temperature rather than the conserved variables, because
    conserved energy is measured from its fluid's datum: copied into a block
    whose fluid has a different one it is silently reinterpreted, which on a
    realistic design is a hundred kelvin. Everything here is dimensional and
    so crosses a fluid boundary unchanged.

    Private: an implementation detail of :meth:`to_dict` and :meth:`from_dict`,
    not a public part of the serialised format. Callers round-trip through
    those two methods rather than this list of keys.

    A mean line is a result, not a config node, so it has no ``type`` and
    never appears among a config's own keys. It does serialise, though: a
    run's answer has to be readable back without repeating the CFD that
    produced it.
    """

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

    @property
    def _cosBeta(self):
        r"""Cosine of the pitch angle, :math:`\cos\beta` [-], array."""
        return np.cos(np.radians(self.Beta))

    def _Dhos(self, P):
        r"""Ideal stagnation enthalpy change to pressure `P`, isentropic from inlet.

        .. math::
            \Delta h_{0s}(P) = h(P,\, s_\mathrm{in}) - h_{0,\mathrm{in}}

        :math:`h(P, s_\mathrm{in})` is the enthalpy reached by an isentropic
        process from the inlet state to `P`. Backs :attr:`Dhos_ts` and
        :attr:`Dhos_tt`, which fix `P` to the outlet static and stagnation
        pressures respectively; kept as a private helper taking `P` so the
        isentropic-state construction is not duplicated between them.
        """
        inlet = self.inlet
        ideal = inlet.empty()
        ideal.set_P_s(P, inlet.s)
        return ideal.h - inlet.ho

    def _eta(self, Dhos):
        """Isentropic efficiency for a given ideal stagnation enthalpy change.

        Dispatches on the sign of :attr:`Dho`, so compressors get ideal/actual
        work and turbines actual/ideal, rather than computing one and
        inverting it whenever it exceeds unity.
        """
        Dho = self.Dho

        with np.errstate(divide="ignore", invalid="ignore"):
            # Work in: the ideal work is the smaller. Work out: the larger.
            eta = Dhos / Dho if Dho > 0.0 else Dho / Dhos

        if np.isnan(eta):
            return np.inf

        return float(eta)

    #
    # CLASSMETHODS
    #

    @classmethod
    def from_dict(cls, data, fluid):
        """Build a mean line from `data` and an equation of state.

        Parameters
        ----------
        data : dict
            As produced by :meth:`to_dict`.
        fluid : ember.fluid.Fluid
            Equation of state to read the stored state against. Required, and
            an argument rather than stored, because the numbers in `data` are
            dimensional and mean nothing without one.

        Returns
        -------
        MeanLine
            A new mean line built from `data`.

        """
        missing = [key for key in cls._STATE if key not in data]
        if missing:
            raise ValueError(f"Mean line state is missing {missing}.")

        values = {key: np.asarray(data[key], dtype=float) for key in cls._STATE}
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

    #
    # SETTERS
    #

    def set_Am(self, Am):
        """Set the annulus area normal to the meridional direction.
        See :attr:`Am`.

        Parameters
        ----------
        Am : array-like
            Annulus area [m^2]. Must broadcast to ``(2, n_row)``.

        """
        self._set_data_by_keys(("Am",), np.asarray(Am) / self.L_ref**2)

    def set_L_ref(self, L_ref):
        """Set the reference length scale, rescaling the stored area.

        Parameters
        ----------
        L_ref : float
            Reference length [m].

        """
        fac_L = L_ref / self.L_ref
        Am = self._get_data_by_keys(("Am",), raise_uninit=False)
        self._set_data_by_keys(("Am",), Am / fac_L**2, store_init=False)
        super().set_L_ref(L_ref)

    def set_Omega(self, Omega):
        r"""Set reference frame angular velocity :math:`\Omega`.

        A plain ``(n_row,)`` array sets one value per blade row: numpy
        broadcasting aligns from the trailing axis, so it is implicitly
        treated as ``(1, n_row)`` and spread across the two stations of each
        row.

        Parameters
        ----------
        Omega : array-like
            Angular velocity [rad/s]. Must broadcast to ``(2, n_row)``.

        """
        self._set_data_by_keys(("Omega",), np.broadcast_to(Omega, self.shape))

    def set_r_rms(self, r_rms):
        """Set :attr:`MeanLine.r_rms`, an alias for :meth:`ember.block.Block.set_r`.

        By mean-line convention the stored block radius *is* the
        root-mean-square radius, so this exists purely to let a caller name
        it that way rather than reaching for the base class's setter.

        Parameters
        ----------
        r_rms : array-like
            Root-mean-square radius [m]. Must broadcast to ``(2, n_row)``.

        """
        self.set_r(r_rms)

    def set_span_htr(self, span, htr):
        r"""Define annulus geometry from span and hub-to-tip ratio.

        .. math::
            r_\mathrm{rms} = \frac{H \sqrt{\tfrac{1}{2}\left(1+\mathit{HTR}^2\right)}}{1-\mathit{HTR}}

        Requires an unpitched mean line, since the annulus geometry is
        derived from :attr:`r_rms` rather than the true root-mean-square
        radius of an inclined surface.

        Parameters
        ----------
        span : array-like
            Annulus span [m]. Must broadcast to ``(2, n_row)``.
        htr : array-like
            Hub-to-tip ratio [-]. Must broadcast to ``(2, n_row)``.

        """
        if not np.all(np.abs(self.Beta) < 1.0):
            raise ValueError("Beta must be set zero before calling set_span_htr")
        self.set_r(span * np.sqrt(0.5 * (1.0 + htr**2)) / (1.0 - htr))
        self.set_Am(2.0 * np.pi * self.r_rms**2 * (1.0 - htr**2) / (1.0 + htr**2))

    def set_span_r_mid(self, span, r_mid):
        r"""Define annulus geometry from span and mid-span radius.

        .. math::
            A_m = 2\pi r_\mathrm{mid} H, \qquad
            r_\mathrm{rms} = \sqrt{r_\mathrm{mid}^2 + \left(\frac{H}{2}\right)^2}

        Parameters
        ----------
        span : array-like
            Annulus span [m]. Must broadcast to ``(2, n_row)``.
        r_mid : array-like
            Annulus mid-span radius [m]. Must broadcast to ``(2, n_row)``.

        """
        self.set_Am(2.0 * np.pi * np.asarray(r_mid) * span)
        self.set_r(np.sqrt(np.asarray(r_mid) ** 2 + (np.asarray(span) / 2.0) ** 2))

    def set_span_r_rms(self, span, r_rms):
        r"""Define annulus geometry from span and root-mean-square radius.

        .. math::
            r_\mathrm{mid} = \sqrt{r_\mathrm{rms}^2 - \left(\frac{H}{2\cos\beta}\right)^2},
            \qquad A_m = 2\pi r_\mathrm{mid} H

        Parameters
        ----------
        span : array-like
            Annulus span [m]. Must broadcast to ``(2, n_row)``.
        r_rms : array-like
            Annulus root-mean-square radius [m]. Must broadcast to
            ``(2, n_row)``.

        """
        self.set_r(r_rms)
        dr = span / self._cosBeta
        r_mid = np.sqrt(np.asarray(r_rms) ** 2 - (dr / 2.0) ** 2)
        self.set_Am(2.0 * np.pi * r_mid * span)

    #
    # GETTERS
    #

    def get_characteristic_station(self, i_row):
        """Return the station of blade row `i_row` characteristic of its flow.

        The end with the higher relative velocity --- the inlet of a
        compressor row, which decelerates the relative flow, or the outlet
        of a turbine row, which accelerates it. Used to scale quantities
        derived from the mean flow, such as the wall spacing target and the
        surface Reynolds number, that need one representative station per
        row rather than a spatially varying one.

        Parameters
        ----------
        i_row : int
            Row index.

        Returns
        -------
        MeanLine
            A scalar view: the inlet or outlet station of row `i_row`,
            whichever has the higher relative velocity.

        """
        row = self[:, i_row]
        return row[0] if row.V_rel[0] >= row.V_rel[1] else row[1]

    def get_referenced_fluid(self):
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
        fluid : ember.fluid.Fluid
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
    # OTHER PUBLIC METHODS
    #

    def to_dict(self):
        """Return this mean line's complete state, in streamwise order.

        Returns
        -------
        dict
            The eight quantities that make up a complete mean line, each a
            list of ``2 * n_row`` values, in the form :meth:`from_dict`
            reads back.

        """
        flat = self.flat
        return {
            key: np.asarray(getattr(flat, key), dtype=float).tolist()
            for key in self._STATE
        }

    def to_string(self):
        """Return a concise tabular summary of the mean line, one row per station.

        Returns
        -------
        str
            Stagnation pressure and temperature, Mach number, and flow angle
            in both frames, formatted via ``turbigen.util.format_table``.

        """
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

    #
    # PROPERTIES
    #

    @property
    def Am(self):
        r"""Annulus area normal to meridional velocity :math:`A_m` [m^2], array.

        The true area of the annular surface the flow crosses, not a
        projection of it, defined so that mass flow :attr:`mdot` is

        .. math::
            \dot{m} = \rho V_m A_m

        At a pitch angle the surface is inclined to the axis, so :math:`A_m`
        is larger than the axial view by one over :math:`\cos\beta`:

        .. math::
            A_m = \frac{\pi \left(r_\mathrm{cas}^2 - r_\mathrm{hub}^2\right)}{\cos\beta}

        """
        return self._get_data_by_keys(("Am",)) * self.L_ref**2

    @property
    def Dho(self):
        r"""Actual stagnation enthalpy change from inlet to outlet, :math:`\Delta h_0 = h_{0,\mathrm{out}} - h_{0,\mathrm{in}}` [J/kg], scalar."""
        return self.outlet.ho - self.inlet.ho

    @property
    def Dhos_ts(self):
        r"""Ideal stagnation enthalpy change to the outlet static pressure, :math:`\Delta h_{0s,\mathrm{ts}} = h(p_\mathrm{out},\, s_\mathrm{in}) - h_{0,\mathrm{in}}` [J/kg], scalar.

        The isentropic counterpart to :attr:`Dho` used by :attr:`eta_ts`: an
        ideal expansion or compression from the inlet state to the actual
        outlet *static* pressure.

        """
        return self._Dhos(self.outlet.P)

    @property
    def Dhos_tt(self):
        r"""Ideal stagnation enthalpy change to the outlet stagnation pressure, :math:`\Delta h_{0s,\mathrm{tt}} = h(p_{0,\mathrm{out}},\, s_\mathrm{in}) - h_{0,\mathrm{in}}` [J/kg], scalar.

        As :attr:`Dhos_ts`, but to the actual outlet *stagnation* pressure;
        used by :attr:`eta_tt`.

        """
        return self._Dhos(self.outlet.Po)

    @property
    def eta_ts(self):
        r"""Total-to-static isentropic efficiency :math:`\eta_\mathrm{ts}` [-], scalar.

        .. math::
            \eta_\mathrm{ts} = \begin{cases}
                \Delta h_{0s,\mathrm{ts}} / \Delta h_0 & \Delta h_0 > 0 \text{ (compressor)} \\
                \Delta h_0 / \Delta h_{0s,\mathrm{ts}} & \Delta h_0 \le 0 \text{ (turbine)}
            \end{cases}

        :attr:`Dho` over :attr:`Dhos_ts`, or the reciprocal for a turbine ---
        so the ideal enthalpy change is to a static state, and any exit
        kinetic energy not recovered is folded into :attr:`Dho` and so
        penalises :math:`\eta_\mathrm{ts}`.

        """
        return self._eta(self.Dhos_ts)

    @property
    def eta_tt(self):
        r"""Total-to-total isentropic efficiency :math:`\eta_\mathrm{tt}` [-], scalar.

        .. math::
            \eta_\mathrm{tt} = \begin{cases}
                \Delta h_{0s,\mathrm{tt}} / \Delta h_0 & \Delta h_0 > 0 \text{ (compressor)} \\
                \Delta h_0 / \Delta h_{0s,\mathrm{tt}} & \Delta h_0 \le 0 \text{ (turbine)}
            \end{cases}

        :attr:`Dho` over :attr:`Dhos_tt`, or the reciprocal for a turbine ---
        so the ideal enthalpy change is to the actual outlet *stagnation*
        pressure, and exit kinetic energy is not penalised.

        """
        return self._eta(self.Dhos_tt)

    @property
    def flat(self):
        """View of all stations in streamwise order, a :class:`MeanLine`, shape ``(2 * n_row,)``.

        A writeable view sharing storage with the parent, not a copy, running
        from machine inlet to machine outlet.

        """
        return super().flat

    @property
    def halfVsq(self):
        r"""Specific kinetic energy in the stationary frame, :math:`\tfrac{1}{2}V^2` [J/kg], array."""
        return 0.5 * self.V**2

    @property
    def halfVsq_rel(self):
        r"""Specific kinetic energy in the rotating frame, :math:`\tfrac{1}{2}\left(V^\mathrm{rel}\right)^2` [J/kg], array."""
        return 0.5 * self.V_rel**2

    @property
    def htr(self):
        r"""Annulus hub-to-tip ratio, :math:`\mathit{HTR} = r_\mathrm{hub}/r_\mathrm{cas}` [-], array."""
        return self.r_hub / self.r_cas

    @property
    def inlet(self):
        """Machine inlet, the first station in streamwise order, a scalar :class:`MeanLine`.

        Not to be confused with ``self[0]``, which is the inlet station of
        every row of shape ``(n_row,)``.
        """
        return self.flat[0]

    @property
    def mdot(self):
        r"""Annulus mass flow rate, :math:`\dot{m} = \rho V_m A_m` [kg/s], array."""
        return self.rho * self.Vm * self.Am

    @property
    def n_row(self):
        """Number of blade rows [-], scalar."""
        return self.shape[1] if self.ndim == 2 else 1

    @property
    def Omega(self):
        r"""Reference frame angular velocity :math:`\Omega` [rad/s], array.

        Note that this overrides the base class :attr:`ember.block.Block.Omega`
        to make it per-station rather than a single scalar."""
        return self._get_data_by_keys(("Omega",))

    @property
    def Omega_nd(self):
        r"""Nondimensional angular velocity, :math:`\Omega^* = \Omega L_\mathrm{ref}/V_\mathrm{ref}` [-], array.

        Shadows :attr:`ember.block.Block.Omega_nd` for the same reason as
        :attr:`Omega`: it must read the per-station value, not the
        metadata-backed scalar ember defines.

        """
        return self.Omega * self.L_ref / self.fluid.V_ref

    @property
    def outlet(self):
        """Machine outlet, the last station in streamwise order, a scalar :class:`MeanLine`.

        Not to be confused with ``ml[1]``, which is the outlet station of
        every row of shape ``(n_row,)``.
        """
        return self.flat[-1]

    @property
    def PR_ts(self):
        r"""Total-to-static pressure ratio, :math:`\mathit{PR}_\mathrm{ts} = p_{0,\mathrm{in}}/p_\mathrm{out}` [-], scalar.

        For a turbine, :math:`\mathit{PR}_\mathrm{ts} < 1`; for a compressor, :math:`\mathit{PR}_\mathrm{ts} > 1`.

        """
        return self.inlet.Po / self.outlet.P

    @property
    def PR_tt(self):
        r"""Total-to-total pressure ratio, :math:`\mathit{PR}_\mathrm{tt} = p_{0,\mathrm{in}}/p_{0,\mathrm{out}}` [-], scalar.

        For a turbine, :math:`\mathit{PR}_\mathrm{tt} < 1`; for a compressor, :math:`\mathit{PR}_\mathrm{tt} > 1`.

        """
        return self.inlet.Po / self.outlet.Po

    @property
    def r_cas(self):
        r"""Annulus casing radius :math:`r_\mathrm{cas}` [m], array.

        .. math::
            r_\mathrm{cas} = \sqrt{\frac{A_m \cos\beta}{2\pi} + r_\mathrm{rms}^2}

        """
        return np.sqrt(self.Am * self._cosBeta / 2.0 / np.pi + self.r_rms**2.0)

    @property
    def r_hub(self):
        r"""Annulus hub radius :math:`r_\mathrm{hub}` [m], array.

        .. math::
            r_\mathrm{hub} = \sqrt{r_\mathrm{rms}^2 - \frac{A_m \cos\beta}{2\pi}}

        """
        return np.sqrt(self.r_rms**2.0 - self.Am * self._cosBeta / 2.0 / np.pi)

    @property
    def r_mid(self):
        r"""Annulus mid radius, :math:`r_\mathrm{mid} = \tfrac{1}{2}(r_\mathrm{hub} + r_\mathrm{cas})` [m], array."""
        return 0.5 * (self.r_hub + self.r_cas)

    @property
    def r_rms(self):
        r"""Annulus root-mean-square radius, :math:`r_\mathrm{rms} = \sqrt{(r_\mathrm{hub}^2 + r_\mathrm{cas}^2)/2}` [m], array.

        This is an alias for :attr:`~ember.block.Block.r` to make it explicit the convention that mean-lines are defined at the root-mean-square radius of the hub and tip.

        """
        return self.r

    @property
    def span(self):
        r"""Annulus span, :math:`H = A_m/(2\pi r_\mathrm{mid})` [m], array.

        Valid for all pitch angles, because :math:`A_m` is already inclined at
        :math:`\beta` and not an axial projection. See :attr:`Am`.

        """
        return self.Am / 2.0 / np.pi / self.r_mid
