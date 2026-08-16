"""Mean-line designers shipped with turbigen_ref.

Each designer is a :class:`turbigen_ref.designer.Designer` subclass registered under
a name that a config file can select with ``mean_line: {type: ...}``.

Note the use of ``ml.flat`` throughout. A mean line is stored as ``(2, n_row)``,
station by row, but the physics of a machine reads naturally in streamwise
order, so these designers take the flat view once and index it ``0, 1, 2, 3``
from machine inlet to machine outlet. That view shares storage with `ml`, so
writes through it land in the mean line being designed.
"""

import numpy as np

import ember.set_iterative
from turbigen_ref.designer import Designer
from turbigen_ref.plugins import register_designer


@register_designer("turbine_cascade")
class TurbineCascade(Designer):
    """A single turbine blade row at fixed inlet stagnation conditions."""

    n_row = 1

    #
    # SHARED DEFINITIONS
    #
    # Written once and used by both directions. Anything needed by forward and
    # backward alike belongs here: a formula duplicated between them is free to
    # drift, and has.
    #

    @staticmethod
    def entropy_rise(Ys, ao1, To1):
        """Entropy rise implied by a pseudo loss coefficient.

        The inverse of :meth:`Ys`.
        """
        return Ys * (0.5 * ao1**2) / To1

    def Ys(self, ml):
        """Pseudo entropy loss coefficient of a mean line.

        Non-dimensionalised on inlet stagnation conditions,
        ``Ys = (s2 - s1) * To1 / (0.5 * ao1**2)``. The inverse of
        :meth:`entropy_rise`.
        """
        inlet = ml.inlet
        return (ml.outlet.s - inlet.s) * inlet.To / (0.5 * inlet.ao**2)

    #
    # DESIGN
    #

    def forward(self, ml, span, Alpha, Ma2, Ys, htr=0.95, Po1=1e5, To1=300.0):
        """Build a turbine cascade from aerodynamic design variables.

        Parameters
        ----------
        span : (2,) array
            Annulus span at inlet and outlet [m].
        Alpha : (2,) array
            Yaw angle at inlet and outlet [deg].
        Ma2 : float
            Outlet Mach number [--].
        Ys : float
            Pseudo entropy loss coefficient [--], see :meth:`Ys`.
        htr : float
            Outlet hub-to-tip ratio [--].
        Po1, To1 : float
            Inlet stagnation pressure [Pa] and temperature [K].

        """
        span = np.asarray(span, dtype=float)
        Alpha = np.asarray(Alpha, dtype=float)
        if span.shape != (2,) or Alpha.shape != (2,):
            raise ValueError(
                f"turbine_cascade needs span and Alpha of length 2, got "
                f"{span.shape} and {Alpha.shape}."
            )

        # Inlet stagnation state, on a scratch station so the mean line itself
        # is untouched until we have something to write.
        stag = ml.inlet.empty()
        stag.set_P_T(Po1, To1)
        ho1, s1, ao1 = stag.h, stag.s, stag.a

        s2 = s1 + self.entropy_rise(Ys, ao1, To1)

        # Outlet state from stagnation enthalpy, entropy, Mach and angles
        ember.set_iterative.set_ho_s_Ma_Alpha_Beta(
            ml.outlet, ho1, s2, Ma2, Alpha[1], Beta=0.0
        )
        ml.outlet.set_span_htr(span[1], htr)

        # Conserve mass to fix the inlet state
        rhoVx1 = ml.outlet.rhoVx * span[1] / span[0]
        ember.set_iterative.set_ho_s_rhoVm_Alpha_Beta(
            ml.inlet, ho1, s1, rhoVx1, Alpha[0], Beta=0.0
        )

        # Inlet annulus shares the outlet mid radius; hub-to-tip may differ
        ml.inlet.set_span_r_mid(span[0], ml.outlet.r_mid)

    def backward(self, ml):
        """Return the design variables represented by a cascade mean line."""
        flat = ml.flat
        return {
            "span": flat.span,
            "Alpha": flat.Alpha,
            "Ma2": ml.outlet.Ma,
            "Ys": self.Ys(ml),
            "htr": ml.outlet.htr,
            "Po1": ml.inlet.Po,
            "To1": ml.inlet.To,
            # Diagnostics, not design variables
            "PR_ts": ml.PR_ts,
            "eta_ts": ml.eta_ts,
        }


@register_designer("axial_turbine")
class AxialTurbine(Designer):
    """An axial turbine stage: a stator row followed by a rotor row."""

    n_row = 2

    #
    # SHARED DEFINITIONS
    #

    @staticmethod
    def entropy_rise(Ys, ao1, To1):
        """Entropy rise at each row exit from the pseudo loss coefficients.

        The inverse of :meth:`Ys`.
        """
        return np.asarray(Ys, dtype=float) * (0.5 * ao1**2) / To1

    def Ys(self, ml):
        """Pseudo entropy loss coefficient at each row exit.

        Non-dimensionalised on inlet stagnation conditions. The inverse of
        :meth:`entropy_rise`.
        """
        inlet = ml.inlet
        # ml[1] is the outlet station of every row
        return (ml[1].s - inlet.s) * inlet.To / (0.5 * inlet.ao**2)

    def psi(self, ml):
        """Stage loading coefficient, the work done per unit blade speed squared."""
        return (ml.inlet.ho - ml.outlet.ho) / self.U(ml) ** 2

    def phi(self, ml):
        """Flow coefficient at rotor inlet."""
        return ml.flat.Vx[2] / self.U(ml)

    def U(self, ml):
        """Rotor blade speed at the mean radius."""
        return ml.flat.U[2]

    def zeta(self, ml):
        """Axial velocity at inlet and outlet, relative to rotor inlet."""
        Vx = ml.flat.Vx
        return Vx[(0, 3),] / Vx[2]

    #
    # DESIGN
    #

    def forward(
        self,
        ml,
        psi,
        phi2,
        Ma2,
        fac_Ma3_rel,
        mdot,
        Ys,
        r_rms,
        zeta=(1.0, 1.0),
        Po1=1e5,
        To1=300.0,
    ):
        """Build an axial turbine stage from aerodynamic design variables.

        Parameters
        ----------
        psi : float
            Stage loading coefficient [--].
        phi2 : float
            Flow coefficient at rotor inlet [--].
        Ma2 : float
            Stator exit Mach number [--].
        fac_Ma3_rel : float
            Rotor exit relative Mach number, as a multiple of `Ma2` [--].
        mdot : float
            Mass flow rate [kg/s].
        Ys : (2,) array
            Pseudo entropy loss coefficient at each row exit [--].
        r_rms : float
            Mean radius, constant through the stage [m].
        zeta : (2,) array
            Axial velocity at stage inlet and outlet, relative to rotor
            inlet [--].
        Po1, To1 : float
            Inlet stagnation pressure [Pa] and temperature [K].

        """
        Ys = np.asarray(Ys, dtype=float)
        zeta = np.asarray(zeta, dtype=float)

        # Reference state from the inlet stagnation conditions. This is the
        # Fluid interface, which returns the state rather than storing it --
        # not to be confused with Block.set_P_T, which mutates and returns None.
        rhoo1, uo1 = ml.fluid.set_P_T(Po1, To1)
        ao1 = ml.fluid.get_a(rhoo1, uo1)
        s1 = ml.fluid.get_s(rhoo1, uo1)
        ho1 = ml.fluid.get_h(rhoo1, uo1)

        # Entropy is fixed by the loss coefficients and does not iterate.
        # Streamwise: inlet, stator exit, rotor inlet, rotor exit.
        ds = self.entropy_rise(Ys, ao1, To1)
        s = s1 + np.array([0.0, ds[0], ds[0], ds[1]])

        flat = ml.flat
        ml.set_r(r_rms)

        def build(U, Vt1, Vt2, Vt3_rel):
            """Construct the stage for one trial set of unknowns.

            Every unknown is a velocity. Parametrising by tangential velocity
            rather than by velocity magnitude means no combination of the
            unknowns can ask for the square root of a negative number, and
            using Vt1 rather than the inlet angle keeps the residual smooth:
            an angle unknown wraps at +-90 degrees, which puts a discontinuity
            in the middle of the search.
            """
            Vx = np.array([zeta[0], 1.0, 1.0, zeta[1]]) * U * phi2
            Vt3 = Vt3_rel + U
            Vt = np.array([Vt1, Vt2, Vt2, Vt3])

            # Euler work equation across the rotor
            ho3 = ho1 + U * (Vt3 - Vt2)
            ho = np.array([ho1, ho1, ho1, ho3])
            h = ho - 0.5 * (Vx**2 + Vt**2)

            flat.set_h_s(h, s)
            flat.set_Vx(Vx)
            flat.set_Vr(0.0)
            flat.set_Vt(Vt)

            # Conservation of mass sets the annulus areas
            flat.set_Am(mdot / flat.rho / Vx)

            # Stator is stationary, rotor turns
            ml.set_Omega_row([0.0, U / r_rms])

        # Guesses. The rotor exit swirl is negative for a turbine.
        U0 = ao1 * Ma2 * 0.5
        Vx0 = U0 * phi2
        Vt2_0 = np.sqrt(max((Ma2 * ao1) ** 2 - Vx0**2, (0.5 * Vx0) ** 2))
        Vt3_rel_0 = -np.sqrt(
            max((fac_Ma3_rel * Ma2 * ao1) ** 2 - Vx0**2, (0.5 * Vx0) ** 2)
        )

        self.solve_for(
            ml,
            build,
            unknowns={
                "U": U0,
                "Vt1": 0.0,
                "Vt2": Vt2_0,
                "Vt3_rel": Vt3_rel_0,
            },
            targets={
                "psi": psi,
                "Ma2": Ma2,
                "fac_Ma3_rel": fac_Ma3_rel,
                # Repeating stage: the flow leaves as it entered
                "Alpha1": "Alpha3",
            },
            name="stage",
        )

    def backward(self, ml):
        """Return the design variables represented by a stage mean line."""
        flat = ml.flat
        h = flat.h

        return {
            # Design variables
            "psi": self.psi(ml),
            "phi2": self.phi(ml),
            "Ma2": flat.Ma[1],
            "fac_Ma3_rel": flat.Ma_rel[3] / flat.Ma[1],
            "mdot": ml.inlet.mdot,
            "Ys": self.Ys(ml),
            "r_rms": ml.inlet.r,
            "zeta": self.zeta(ml),
            "Po1": ml.inlet.Po,
            "To1": ml.inlet.To,
            # Diagnostics
            "Alpha1": ml.inlet.Alpha,
            "Alpha3": ml.outlet.Alpha,
            "Ma3_rel": flat.Ma_rel[3],
            "Lam": (h[1] - h[0]) / (h[3] - h[0]),
            "PR_ts": ml.PR_ts,
            "PR_tt": ml.PR_tt,
            "eta_tt": ml.eta_tt,
            "eta_ts": ml.eta_ts,
            "Omega": ml.outlet.Omega,
        }
