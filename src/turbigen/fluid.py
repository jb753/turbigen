r"""Calculation of thermodynamic properties for working fluids.

Turbomachinery design and analysis requires knowledge of thermodynamic
properties of the working fluid. This module contains classes that abstract
real or perfect gases, so that calculations of thermodynamic properties can
be performed using a common interface independent of the fluid equation of
state.

The general pattern of usage of these classes is as follows. First initialise a
state using :py:meth:`turbigen.fluid.PerfectState.from_properties` or
:py:meth:`turbigen.fluid.RealState.from_fluid_name` constructors. If the type
of working fluid is not known at runtime, the `copy()` method of an existing
state can be used. Second, specify a pair of two thermodynamic properties
using, for example, :py:meth:`turbigen.fluid.PerfectState.set_P_T`. Finally,
other derived properties can be read off as
attributes of the state, such as :py:attr:`turbigen.fluid.PerfectState.rho`.
The examples below illustrate functionality of the
classes; practical usages are found in the :py:mod:`turbigen.meanline` functions.

Example code
------------

Create a state for real Air, print some properties, then if this is a
stagnation state, find static pressure given a Mach number:

.. doctest::

    >>> import turbigen.fluid
    >>> So1 = turbigen.fluid.RealState.from_fluid_name('Air')
    >>> So1.set_P_T(16e5, 1600.)
    RealState(Air, P=16.000 bar, T=1600.0 K)
    >>> print('cp = %.1f kJ/kg' % So1.cp)
    cp = 1220.9 kJ/kg
    >>> print('ga = %.3f' % So1.gamma)
    ga = 1.308
    >>> S1 = So1.to_static(Ma=0.6)
    >>> print('P1 = %.3e' % S1.P)
    P1 = 1.271e+06

Initialise states for both perfect air and steam, then calculate the work
required for an isentropic compression:

.. doctest::

    >>> S_water = turbigen.fluid.RealState.from_fluid_name('water')
    >>> S_air = turbigen.fluid.PerfectState.from_properties(
    ...     cp=1005.,gamma=1.4, mu=1.8e-4
    ... )
    >>> PR = 3.
    >>> for S1 in (S_water, S_air):
    ...     S1.set_P_T(1e5, 300.)
    ...     S2 = S1.copy().set_P_s(S1.P * PR, S1.s)
    ...     print(f'wx12 = {(S2.h-S1.h)/1e3:.2f} kJ/kg')
    RealState(water, P=1.000 bar, T=300.0 K)
    wx12 = 0.20 kJ/kg
    PerfectState(P=1.000 bar, T=300.0 K)
    wx12 = 111.17 kJ/kg

Array states are also supported, by passing a shape tuple on initialisation.
Broadcasting input values, and read-only iterating and slicing work:

.. doctest::

    >>> import turbigen.fluid
    >>> S = turbigen.fluid.RealState.from_fluid_name('water', shape=(3,))
    >>> S.set_P_T(1e5, [400., 500., 800])
    RealState(water, P=[1. 1. 1.] bar, T=[400. 500. 800.] K)
    >>> print(S.rho)
    [0.54760542 0.43514008 0.27102399]
    >>> for Si in S[1:]:
    ...     print(Si)
    RealState(water, P=1.000 bar, T=500.0 K)
    RealState(water, P=1.000 bar, T=800.0 K)

Property attributes
-------------------

The state classes expose the following thermodynamic and fluid properties as
attributes:

=========== ============================================ =======
 Attribute   Property                                     Units
=========== ============================================ =======
`a`          Acoustic Speed                               m/s
`cp`         Specific heat capacity at constant pressure  J/kg/K
`cv`         Specific heat capacity at constant volume    J/kg/K
`gamma`      Ratio of specific heats
`h`          Specific enthalpy                            J/kg
`mu`         Dynamic viscosity                            kg/m/s
`P`          Pressure                                     Pa
`Pr`         Prandtl number
`rgas`       Specific gas constant                        J/kg/K
`rho`        Mass density                                 kg/m^3
`s`          Specific entropy                             J/kg/K
`T`          Temperature                                  K
`u`          Specific internal energy                     J/kg
=========== ============================================ =======


Setting methods
---------------

The state classes expose the following methods to specify the thermodynamic
state, each taking two properties as arguments:

=================== ============================================
 Method              Properties specified
=================== ============================================
`set_P_T(P,T)`      Pressure, temperature
`set_P_s(P,s)`      Pressure, entropy
`set_h_s(h,s)`      Enthalpy, entropy
`set_T_s(T,s)`      Temperature, entropy
`set_P_h(P,h)`      Enthalpy, entropy
`set_rho_u(rho,u)`  Density, internal energy
`set_rho_s(rho,s)`  Density, entropy
=================== ============================================

Static/stagnation methods
-------------------------

Consider the problem of converting between static and stagnation conditions at
a given Mach number :math:`\Ma=V/a`. If static conditions are known, it is
straightforward to use Mach number and acoustic speed to evaluate velocity and
hence enthalpy of the stagnation state; an ideal stagnation is an isentropic
process, so static and stagnation states have the same entropy. The reverse
problem, where static conditions must be found for known stagnation conditions,
in general requires iteration because acoustic speed is unknown a-priori.

The state classes have two methods to streamline these conversions:

    * `to_stagnation(Ma)` returns a State object for stagnation conditions
    * `to_static(Ma)` returns a State object for static conditions



Constructor methods
-------------------

"""

from CoolProp import CoolProp
import numpy as np
from turbigen.base import dependent_property, StructuredData
import turbigen.util


# Share tabular property data across instances in a class-level dict
_abstract_states = {"HEOS": {}, "BICUBIC&HEOS": {}}


class PerfectState(StructuredData):
    """Thermodynamic properties from perfect gas equations of state."""

    _data_rows = (
        "rho",
        "u",
    )

    # Arbitrary reference properties for entropy datum
    _Ps0_default = 1e5
    _Ts0_default = 300.0
    _Tu0_default = 300.0

    def __eq__(self, other):
        if other is None:
            return False

        return (
            self.shape == other.shape
            and self.cp == other.cp
            and self.gamma == other.gamma
            and self.mu == other.mu
            and turbigen.util._match(self.rho, other.rho)
            and turbigen.util._match(self.u, other.u)
        )

    def __repr__(self):
        try:
            return f"PerfectState(P={self.P/1e5:.3f} bar, T={self.T:.1f} K)"
        except TypeError:
            return (
                f"PerfectState(P={np.array2string(self.P/1e5, precision=3)} bar, "
                f" T={np.array2string(self.T, precision=1)} K)"
            )

    @classmethod
    def from_properties(cls, cp, gamma, mu, shape=(), Pr=0.72):
        r"""Create a state for a perfect gas with specified properties.

        Parameters
        ----------
        cp: float
            Specific heat capacity at constant pressure [J/kg/K]
        gamma: float
            Ratio of specific heats [--].
        mu: float
            Kinematic viscosity [kg/m/s].
        shape: tuple
            Specify a tuple to allocate an array of states; defaults to scalar.
        Pr: float
            Prandtl number [--], default to room-temperature air.

        """

        S = cls(shape)
        S.cp, S.gamma, S.mu = cp, gamma, mu
        S.Pr = Pr
        return S

    @property
    def Tu0(self):
        if "Tu0" not in self._metadata:
            self._set_metadata_by_key("Tu0", self._Tu0_default)
        return self._get_metadata_by_key("Tu0")

    @Tu0.setter
    def Tu0(self, Tu0):
        self._set_metadata_by_key("Tu0", Tu0)

    @property
    def Ps0(self):
        if "Ps0" not in self._metadata:
            self._set_metadata_by_key("Ps0", self._Ps0_default)
        return self._get_metadata_by_key("Ps0")

    @Ps0.setter
    def Ps0(self, Ps0):
        self._set_metadata_by_key("Ps0", Ps0)

    @property
    def Ts0(self):
        if "Ts0" not in self._metadata:
            self._set_metadata_by_key("Ts0", self._Ts0_default)
        return self._get_metadata_by_key("Ts0")

    @Ts0.setter
    def Ts0(self, Ts0):
        self._set_metadata_by_key("Ts0", Ts0)

    @property
    def cp(self):
        return self._get_metadata_by_key("cp")

    @cp.setter
    def cp(self, cp):
        self._set_metadata_by_key("cp", cp)

    @property
    def Pr(self):
        return self._get_metadata_by_key("Pr")

    @Pr.setter
    def Pr(self, Pr):
        self._set_metadata_by_key("Pr", Pr)

    @property
    def gamma(self):
        return self._get_metadata_by_key("gamma")

    @gamma.setter
    def gamma(self, gamma):
        _check_positive_finite_scalar(gamma)
        self._set_metadata_by_key("gamma", gamma)

    @property
    def mu(self):
        return self._get_metadata_by_key("mu")

    @mu.setter
    def mu(self, mu):
        self._set_metadata_by_key("mu", mu)

    @property
    def rho(self):
        return self._get_data_by_key("rho")

    @rho.setter
    def rho(self, value):
        val_array = np.array(value)
        self._set_data_by_key("rho", val_array)

    @property
    def u(self):
        return self._get_data_by_key("u")

    @u.setter
    def u(self, value):
        val_array = np.array(value)
        self._set_data_by_key("u", val_array)

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

    @property
    def is_two_phase(self):
        # Perfect gas is never liquid or two-phase
        return np.full(self.shape, False)

    @dependent_property
    def s(self):
        return self.cp * np.log(self.T / self.Ts0) - self.rgas * np.log(
            self.P / self.Ps0
        )

    @dependent_property
    def dsdrho_P(self):
        return -self.cp / self.rho

    @dependent_property
    def dsdP_rho(self):
        return self.cv / self.P

    @dependent_property
    def dhdP_rho(self):
        ga = self.gamma
        return ga / (ga - 1.0) / self.rho

    @dependent_property
    def dhdrho_P(self):
        return -self.cp * self.T / self.rho

    @dependent_property
    def dudrho_P(self):
        return -self.P / self.rho**2 / (self.gamma - 1.0)

    @dependent_property
    def dudP_rho(self):
        return 1.0 / self.rho / (self.gamma - 1.0)

    def set_Tu0(self, Tu0):
        # Set the temperature for internal energy datum u(Tu0) = 0
        P = self.P.copy()
        T = self.T.copy()
        self.Tu0 = Tu0
        return self.set_P_T(P, T)

    def set_P_T(self, P, T):
        u = self.cv * (T - self.Tu0)
        rho = P / self.rgas / T
        return self.set_rho_u(rho, u)

    def set_P_s(self, P, s):
        T = self.Ts0 * np.exp((s + self.rgas * np.log(P / self.Ps0)) / self.cp)
        self.set_P_T(P, T)
        return self

    def set_h_s(self, h, s):
        T = (h + self.cv * self.Tu0) / self.cp
        P = self.Ps0 * np.exp((self.cp * np.log(T / self.Ts0) - s) / self.rgas)
        self.set_P_T(P, T)
        # atol_s = self.cv * 1e-6
        # assert np.allclose(self.h, h)
        # assert np.allclose(self.s, s, atol=atol_s)
        return self

    def set_T_s(self, T, s):
        P = self.Ps0 * np.exp((self.cp * np.log(T / self.Ts0) - s) / self.rgas)
        self.set_P_T(P, T)
        return self

    def set_P_h(self, P, h):
        T = (h + self.cv * self.Tu0) / self.cp
        self.set_P_T(P, T)
        # print(self.h[0], h[0])
        # assert np.allclose(self.h, h)
        # assert np.allclose(self.P, P)
        return self

    def set_P_rho(self, P, rho):
        T = P / self.rgas / rho
        self.set_P_T(P, T)
        return self

    def set_rho_u(self, rho, u):
        self.rho = rho
        self.u = u
        return self

    def set_rho_s(self, rho, s):
        rhos0 = self.Ps0 / self.rgas / self.Ts0
        T = self.Ts0 * np.exp((s + self.rgas * np.log(rho / rhos0)) / self.cv)
        u = self.cv * (T - self.Tu0)
        self.set_rho_u(rho, u)
        return self

    def to_static(self, Ma):
        To_T = 1.0 + 0.5 * (self.gamma - 1.0) * Ma**2.0
        return self.copy().set_T_s(self.T / To_T, self.s)

    def to_stagnation(self, Ma):
        To_T = 1.0 + 0.5 * (self.gamma - 1.0) * Ma**2.0
        return self.copy().set_T_s(self.T * To_T, self.s)


class RealState(StructuredData):
    """Thermodynamic properties for a real fluid using table lookups."""

    _data_rows = (
        "rho",
        "u",
    )

    Tu0 = 0.0

    _backend = "HEOS"

    def __repr__(self):
        try:
            return (
                f"RealState({self.fluid_name}, P={self.P/1e5:.3f} bar,"
                f" T={self.T:.1f} K)"
            )
        except TypeError:
            return (
                f"RealState({self.fluid_name},"
                f" P={np.array2string(self.P/1e5, precision=3)} bar, "
                f" T={np.array2string(self.T, precision=1)} K)"
            )
        except (KeyError, ValueError):
            return "RealState(uninitialised)"

    def __eq__(self, other):
        if other is None:
            return False

        try:
            other_fluid_name = other.fluid_name
        except KeyError:
            other_fluid_name = None

        try:
            self_fluid_name = self.fluid_name
        except KeyError:
            self_fluid_name = None

        return (
            self.shape == other.shape
            and self_fluid_name == other_fluid_name
            and turbigen.util._match(self.rho, other.rho)
            and turbigen.util._match(self.u, other.u)
        )

    @classmethod
    def from_fluid_name(cls, fluid_name, shape=()):
        r"""Create a state for a particular real working fluid.

        Parameters
        ----------
        fluid_name: str
            Name of the fluid in the `CoolProp nomenclature
            <http://www.coolprop.org/fluid_properties/PurePseudoPure.html#list-of-fluids>`_.
        shape: tuple
            Specify a tuple to allocate an array of states; defaults to scalar.

        """

        S = cls(shape)
        S._metadata["fluid_name"] = None
        S.fluid_name = fluid_name
        return S

    @property
    def fluid_name(self):
        return self._get_metadata_by_key("fluid_name")

    @property
    def _as(self):
        return self._get_metadata_by_key("abstract_state")

    @_as.setter
    def _as(self, val):
        return self._set_metadata_by_key("abstract_state", val)

    @fluid_name.setter
    def fluid_name(self, fluid_name):
        # Look for an abstract state for this fluid in the module-level cache
        # and use if present, otherwise create a new table
        _as = _abstract_states[self._backend].get(fluid_name)
        if _as:
            self._as = _as
        else:
            self._as = _abstract_states[self._backend][
                fluid_name
            ] = CoolProp.AbstractState(self._backend, fluid_name)

        self._set_metadata_by_key("fluid_name", fluid_name)

    def set_backend(self, backend):
        self._backend = backend
        self.fluid_name = self.fluid_name

    @property
    def rho(self):
        return self._get_data_by_key("rho")

    @property
    def u(self):
        return self._get_data_by_key("u")

    def _store(self, inputs, x, y):
        """Calculate density and internal energy and store for future use."""
        # Both inputs scalar, no need to loop
        if np.shape(x) == () and np.shape(y) == ():
            self._as.update(inputs, x, y)
            self._set_data_by_key("rho", self._as.rhomass())
            self._set_data_by_key("u", self._as.umass())
        else:
            # Otherwise, broadcast and loop
            b = np.broadcast(x, y)
            if not b.shape == self.shape:
                raise Exception(
                    f"Broadcasted input shape {b.shape} is wrong, expected {self.shape}"
                )
            rho = np.zeros(b.shape)
            u = np.zeros(b.shape)
            for i, (xk, yk) in enumerate(b):
                self._as.update(inputs, xk, yk)
                rho.flat[i] = self._as.rhomass()
                u.flat[i] = self._as.umass()
            self._set_data_by_key("rho", rho)
            self._set_data_by_key("u", u)

    def _lookup_property(self, prop_func):
        """Table lookup a general property using stored density and internal energy."""
        # Scalar state, no need to loop
        if self.shape == ():
            self._as.update(CoolProp.DmassUmass_INPUTS, self.rho, self.u)
            z = prop_func()
        # Vector state, must loop
        else:
            z = np.zeros(self.shape)
            for i, (rhok, uk) in enumerate(zip(self.rho.flat, self.u.flat)):
                self._as.update(CoolProp.DmassUmass_INPUTS, rhok, uk)
                z.flat[i] = prop_func()
        return z

    def _lookup_saturated_property(self, prop_func, chi):
        """Table lookup for saturated property at current pressure."""
        # Scalar state, no need to loop
        if self.shape == ():
            self._as.update(CoolProp.PQ_INPUTS, self.P, chi)
            z = prop_func()
        # Vector state, must loop
        else:
            z = np.zeros(self.shape)
            for i, Pk in enumerate(self.P.flat):
                self._as.update(CoolProp.PQ_INPUTS, Pk, chi)
                z.flat[i] = prop_func()
        return z

    def set_P_h(self, P, h):
        self._store(CoolProp.HmassP_INPUTS, h, P)
        return self

    def set_P_rho(self, P, rho):
        self._store(CoolProp.DmassP_INPUTS, rho, P)
        return self

    def set_P_s(self, P, s):
        self._store(CoolProp.PSmass_INPUTS, P, s)
        return self

    def set_h_s(self, h, s):
        self._store(CoolProp.HmassSmass_INPUTS, h, s)
        assert np.allclose(self.h, h)
        assert np.allclose(self.s, s)
        return self

    def set_T_s(self, T, s):
        self._store(CoolProp.SmassT_INPUTS, s, T)
        return self

    def set_P_T(self, P, T):
        self._store(CoolProp.PT_INPUTS, P, T)
        assert np.allclose(self.P, P)
        assert np.allclose(self.T, T)
        return self

    def set_rho_T(self, ro, T):
        self._store(CoolProp.DmassT_INPUTS, ro, T)
        return self

    def set_rho_u(self, ro, u):
        self._set_data_by_key("rho", ro)
        self._set_data_by_key("u", u)
        return self

    def set_rho_s(self, ro, s):
        self._store(CoolProp.DmassSmass_INPUTS, ro, s)
        return self

    def set_T_chi(self, T, chi):
        self._store(CoolProp.QT_INPUTS, chi, T)
        return self

    def set_P_chi(self, P, chi):
        self._store(CoolProp.PQ_INPUTS, P, chi)
        return self

    def set_Tu0(self, Tu0):
        if Tu0:
            raise NotImplementedError("Real gas does not support arbitrary u datum")

    def to_static(self, Ma):
        # Function to solve Ma for a scalar state
        def _solve_scalar(Sstag, Mai):
            if Mai == 0.0:
                return Sstag
            if (not np.isfinite(Mai)) or np.iscomplex(Mai):
                raise ValueError(f"Invalid Ma={Mai}")

            # Use fixed point iteration
            err = np.inf
            tolMa = 1e-6
            rf = 0.5
            V = Mai * Sstag.a
            for i in range(20):
                Sstat = Sstag.copy().set_h_s(Sstag.h - 0.5 * V**2.0, Sstag.s)
                V = np.sqrt(2.0 * (Sstag.h - Sstat.h))
                Ma_now = V / Sstat.a
                err = np.abs(Mai - Ma_now)
                V = V * (1.0 - rf) + rf * (Mai * Sstat.a)
                if err < tolMa:
                    break
            if err > tolMa:
                raise Exception("did not converge")

            return Sstat

        if self.shape == () and np.shape(Ma) == ():
            return _solve_scalar(self.copy(), Ma)

        else:
            raise NotImplementedError()
            # TODO use slicing capabilities of the class here
            # # Broadcast Mach number
            # b = np.broadcast(self.h, self.P, Ma)
            # h = np.zeros(b.shape)
            # P = np.zeros(b.shape)
            # for i, (hk, Pk, Mak) in enumerate(b):
            #     Sk_stag = self.copy().set_P_h(Pk, hk)
            #     Sk_stat = _solve_scalar(Sk_stag, Mak)
            # h.flat[i] = Sk_stat.h
            # P.flat[i] = Sk_stat.P
            # return self.copy().set_P_h(P, h)

    def to_stagnation(self, Ma):
        V = self.a * Ma
        hstag = self.h + 0.5 * V**2.0
        return self.copy().set_h_s(hstag, self.s)

    @dependent_property
    def T(self):
        return self._lookup_property(self._as.T)

    @dependent_property
    def P(self):
        return self._lookup_property(self._as.p)

    @dependent_property
    def a(self):
        return self._lookup_property(self._as.speed_sound)

    @dependent_property
    def cp(self):
        return self._lookup_property(self._as.cpmass)

    @dependent_property
    def cv(self):
        return self._lookup_property(self._as.cvmass)

    @dependent_property
    def gamma(self):
        return self.cp / self.cv

    @dependent_property
    def rgas(self):
        Rgas_molar = self._lookup_property(self._as.gas_constant)
        M_molar = self._lookup_property(self._as.molar_mass)
        return Rgas_molar / M_molar

    @dependent_property
    def mu(self):
        return self._lookup_property(self._as.viscosity)

    @dependent_property
    def s(self):
        return self._lookup_property(self._as.smass)

    @dependent_property
    def h(self):
        return self._lookup_property(self._as.hmass)

    @dependent_property
    def Pr(self):
        return self._lookup_property(self._as.Prandtl)

    # Real gas stuff

    @dependent_property
    def chi(self):
        return self._lookup_property(self._as.Q)

    @dependent_property
    def hsat_vapour(self):
        return self._lookup_saturated_property(self._as.hmass, 1.0)

    @dependent_property
    def Tsat(self):
        return self._lookup_saturated_property(self._as.T, 0.5)

    @dependent_property
    def DTsuperheat(self):
        # Superheat in vapour dome extrapolated using saturated vapour cp
        cpsat = self._lookup_saturated_property(self._as.cpmass, 1.0)
        DT_sup_2phs = (self.h - self.hsat_vapour) / cpsat

        # Temperature diference w.r.t. Tsat outside vapour dome
        DT_sup_sup = self.T - self.Tsat

        # Choose the appropriate definition
        return np.where(self.is_two_phase, DT_sup_2phs, DT_sup_sup)

    @dependent_property
    def is_two_phase(self):
        return self.phase == CoolProp.iphase_twophase

    @dependent_property
    def is_liquid(self):
        return self.phase == CoolProp.iphase_liquid

    @dependent_property
    def is_gas(self):
        return self.phase == CoolProp.iphase_gas

    @dependent_property
    def phase(self):
        phase = self._lookup_property(self._as.phase)
        # Convert numpy arrays to integer data type
        try:
            phase = phase.astype(int)
        except AttributeError:
            pass
        return phase

    @dependent_property
    def is_supercritical(self):
        return self._lookup_property(self._as.phase) in (
            CoolProp.iphase_supercritical_gas,
            CoolProp.iphase_supercritical_liquid,
            CoolProp.iphase_supercritical,
        )

    @property
    def Tcrit(self):
        return self._as.T_critical()

    @property
    def Pcrit(self):
        return self._as.p_critical()


def _check_positive_finite_scalar(x):
    if np.shape(x) not in ((), (1,)):
        raise ValueError(f"{x} is not scalar")

    if not (x > 0.0 and np.isfinite(x)):
        raise ValueError(f"Bad value for {x}")
