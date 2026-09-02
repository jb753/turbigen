"""An axial turbine stage: a stator row followed by a rotor row."""

from typing import ClassVar

import numpy as np

from turbigen.design import MeanLineDesign

VT1_FRACTION = 0.05
"""Fraction of the guessed blade speed to start the inlet swirl at.

Any value off zero conditions the Jacobian equally well --- 1%, 5% and 10%
all give a condition number near 6.7, against 2.5e+06 at exactly zero --- so
this is a guess rather than a tuned constant. Five percent is small enough to
be a fair guess at an axial inlet and large enough to be nowhere near zero.
"""


class AxialTurbine(MeanLineDesign):
    """An axial turbine stage."""

    type: ClassVar[str] = "axial_turbine"
    n_row: ClassVar[int] = 2

    psi: float
    """Stage loading coefficient [--]."""

    phi2: float
    """Flow coefficient at rotor inlet [--]."""

    Ma2: float
    """Stator exit Mach number [--]."""

    fac_Ma3_rel: float
    """Rotor exit relative Mach number, as a multiple of ``Ma2:`` [--]."""

    mdot: float
    """Mass flow rate [kg/s]."""

    Ys: tuple[float, ...]
    """Pseudo entropy loss coefficient at each row exit [--]."""

    r_rms: float
    """Mean radius, constant through the stage [m]."""

    zeta: tuple[float, float] = (1.0, 1.0)
    """Axial velocity at stage inlet and outlet, relative to rotor inlet [--]."""

    Po1: float = 1e5
    """Inlet stagnation pressure [Pa]."""

    To1: float = 300.0
    """Inlet stagnation temperature [K]."""

    #
    # SHARED DEFINITIONS
    #

    @staticmethod
    def entropy_rise(Ys, ao1, To1):
        """Entropy rise at each row exit from the pseudo loss coefficients.

        The inverse of :meth:`loss_coefficient`.
        """
        return np.asarray(Ys, dtype=float) * (0.5 * ao1**2) / To1

    def loss_coefficient(self, ml):
        """Pseudo entropy loss coefficient at each row exit.

        Non-dimensionalised on inlet stagnation conditions. The inverse of
        :meth:`entropy_rise`.
        """
        inlet = ml.inlet
        # ml[1] is the outlet station of every row
        return (ml[1].s - inlet.s) * inlet.To / (0.5 * inlet.ao**2)

    def blade_speed(self, ml):
        """Rotor blade speed at the mean radius [m/s]."""
        return ml.flat.U[2]

    def loading(self, ml):
        """Stage loading coefficient, work done per unit blade speed squared."""
        return (ml.inlet.ho - ml.outlet.ho) / self.blade_speed(ml) ** 2

    def flow_coefficient(self, ml):
        """Flow coefficient at rotor inlet."""
        return ml.flat.Vx[2] / self.blade_speed(ml)

    def velocity_ratios(self, ml):
        """Axial velocity at inlet and outlet, relative to rotor inlet."""
        Vx = ml.flat.Vx
        return Vx[(0, 3),] / Vx[2]

    #
    # DESIGN
    #

    def forward(self, fluid):
        ml = self.allocate(fluid)

        Ys = np.asarray(self.Ys, dtype=float)
        zeta = np.asarray(self.zeta, dtype=float)
        phi2, mdot, r_rms = self.phi2, self.mdot, self.r_rms

        # Reference state from the inlet stagnation conditions. This is the
        # Fluid interface, which returns the state rather than storing it --
        # not to be confused with Block.set_P_T, which mutates and returns None.
        rhoo1, uo1 = ml.fluid.set_P_T(self.Po1, self.To1)
        ao1 = ml.fluid.get_a(rhoo1, uo1)
        s1 = ml.fluid.get_s(rhoo1, uo1)
        ho1 = ml.fluid.get_h(rhoo1, uo1)

        # Entropy is fixed by the loss coefficients and does not iterate.
        # Streamwise: inlet, stator exit, rotor inlet, rotor exit.
        ds = self.entropy_rise(Ys, ao1, self.To1)
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
            ml.set_Omega([0.0, U / r_rms])

        # Guesses. The rotor exit swirl is negative for a turbine.
        U0 = ao1 * self.Ma2 * 0.5
        # Off zero, though an axial inlet is what this converges to. A guess
        # of exactly zero degenerates the finite-difference step: it is taken
        # relative to the value, floored at one, so Vt1 got a step of 3.5e-04
        # where the other three velocities got 0.05 to 0.10. Dividing by a
        # step 156x smaller inflates that column of the Jacobian by about as
        # much, leaving the columns spread 2800x and the matrix conditioned at
        # 2.5e+06 -- whose smallest singular value, 6.4e-06, is below the
        # noise floor of a float32 residual, so the solver was working a
        # rank-3 problem believing it had rank 4. Seeded here instead: the
        # condition number becomes 6.7, and does so for anything from 1% to
        # 10%, which is what a guess should look like.
        Vx0 = U0 * phi2
        Vt2_0 = np.sqrt(max((self.Ma2 * ao1) ** 2 - Vx0**2, (0.5 * Vx0) ** 2))
        Vt3_rel_0 = -np.sqrt(
            max((self.fac_Ma3_rel * self.Ma2 * ao1) ** 2 - Vx0**2, (0.5 * Vx0) ** 2)
        )

        self.solve_for(
            ml,
            build,
            unknowns={
                "U": U0,
                "Vt1": VT1_FRACTION * U0,
                "Vt2": Vt2_0,
                "Vt3_rel": Vt3_rel_0,
            },
            targets={
                "psi": self.psi,
                "Ma2": self.Ma2,
                "fac_Ma3_rel": self.fac_Ma3_rel,
                # Repeating stage: the flow leaves as it entered
                "Alpha1": "Alpha3",
            },
            name="stage",
        )

        return ml

    def backward(self, ml):
        flat = ml.flat
        h = flat.h

        return {
            # Design variables
            "psi": self.loading(ml),
            "phi2": self.flow_coefficient(ml),
            "Ma2": flat.Ma[1],
            "fac_Ma3_rel": flat.Ma_rel[3] / flat.Ma[1],
            "mdot": ml.inlet.mdot,
            "Ys": self.loss_coefficient(ml),
            "r_rms": ml.inlet.r,
            "zeta": self.velocity_ratios(ml),
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
