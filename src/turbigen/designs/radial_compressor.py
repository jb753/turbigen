"""A radial compressor rotor with axial inflow and a radial exit.

The design variables are the aerodynamic set of Brind (2024), "Data-Driven
Radial Compressor Design Space Mapping", J. Turbomach. 147(2): pressure ratio,
inlet flow coefficient, inlet relative Mach number, inlet hub-to-tip ratio,
rotor de Haller number and rotor exit relative yaw angle. Together with the
inlet state and mass flow these fix the mean line without iteration, and every
combination of them within sensible bounds is feasible: no radius ratio is
asked for, so none can be impossible to reach, and the inlet Mach number is
stated directly, so the inlet cannot choke.

The blade count and tip gap, the other two variables of that parameterisation,
are blade-row choices rather than mean-line ones and are set under
``blades:``.

The mean line is the rotor alone. Pressure ratio is total-to-total from rotor
inlet to rotor exit, and any diffuser is left to the outlet duct of the
annulus.
"""

from typing import ClassVar

import ember.set_iterative
import numpy as np

from turbigen.design import DesignError, MeanLineDesign


class RadialCompressor(MeanLineDesign):
    """A radial compressor rotor with axial inflow and a radial exit."""

    type: ClassVar[str] = "radial_compressor"
    n_row: ClassVar[int] = 1

    PR_tt: float
    """Total-to-total pressure ratio across the rotor [--]."""

    phi1: float
    """Inlet flow coefficient, meridional velocity over blade speed at the
    inlet root-mean-square radius [--]."""

    Ma1_rel: float
    """Inlet relative Mach number at the root-mean-square radius [--]."""

    htr1: float
    """Inlet hub-to-tip radius ratio [--]."""

    DH: float
    """Rotor de Haller number, exit over inlet relative velocity [--]."""

    Alpha2_rel: float
    """Rotor exit relative yaw angle [deg]; negative for backsweep."""

    mdot: float
    """Mass flow rate [kg/s]."""

    Ys: float
    """Pseudo entropy loss coefficient across the rotor [--], see
    :meth:`loss_coefficient`."""

    Po1: float = 1e5
    """Inlet stagnation pressure [Pa]."""

    To1: float = 300.0
    """Inlet stagnation temperature [K]."""

    #
    # SHARED DEFINITIONS
    #

    @staticmethod
    def entropy_rise(Ys, ao1, To1):
        """Entropy rise implied by a pseudo loss coefficient.

        The inverse of :meth:`loss_coefficient`.
        """
        return Ys * (0.5 * ao1**2) / To1

    def loss_coefficient(self, ml):
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

    def forward(self, fluid):
        self._check_bounds()

        ml = self.allocate(fluid)
        inlet, outlet = ml.inlet, ml.outlet

        # Inlet stagnation state, on a scratch station
        stag = inlet.empty()
        stag.set_P_T(self.Po1, self.To1)
        ho1, s1, ao1 = stag.h, stag.s, stag.a

        # Rotor exit stagnation state follows from pressure ratio and loss
        s2 = s1 + self.entropy_rise(self.Ys, ao1, self.To1)
        stag.set_P_s(self.PR_tt * self.Po1, s2)
        ho2 = stag.h

        #
        # INLET
        #

        # With no inlet swirl, V1_rel**2 = V1**2 + U1**2 = V1**2 (1 + 1/phi1**2)
        Ma1 = self.Ma1_rel / np.sqrt(1.0 + self.phi1**-2)

        # A placeholder radius for the iterative setter. The inlet is unswirled
        # so the stored angular momentum is zero, and the true radius can be
        # written over it once the area is known.
        ml.set_r(1.0)
        ember.set_iterative.set_ho_s_Ma_Alpha_Beta(inlet, ho1, s1, Ma1, 0.0, 0.0)

        # Mass flow sets the area, hub-to-tip ratio the radius
        Am1 = self.mdot / inlet.rho / inlet.Vm
        r1 = np.sqrt(Am1 / 2.0 / np.pi * (1.0 + self.htr1**2) / (1.0 - self.htr1**2))
        inlet.set_r(r1)
        inlet.set_Am(Am1)

        U1 = inlet.Vm / self.phi1
        Omega = U1 / r1
        ml.set_Omega(Omega)

        #
        # ROTOR EXIT
        #

        # Relative velocity from the de Haller number and exit angle
        V2_rel = self.DH * inlet.V_rel
        Vr2 = V2_rel * np.cos(np.radians(self.Alpha2_rel))
        Vt2_rel = V2_rel * np.sin(np.radians(self.Alpha2_rel))

        # Euler work with no inlet swirl, ho2 - ho1 = U2 Vt2 = U2 (Vt2_rel + U2),
        # is a quadratic in the exit blade speed. The work is positive, so the
        # roots have opposite signs and the positive one is the blade speed.
        Dho = ho2 - ho1
        U2 = 0.5 * (-Vt2_rel + np.sqrt(Vt2_rel**2 + 4.0 * Dho))
        r2 = U2 / Omega
        Vt2 = Vt2_rel + U2

        outlet.set_r(r2)
        outlet.set_h_s(ho2 - 0.5 * (Vr2**2 + Vt2**2), s2)
        outlet.set_Vx(0.0)
        outlet.set_Vr(Vr2)
        outlet.set_Vt(Vt2)
        outlet.set_Am(self.mdot / outlet.rho / Vr2)

        # The exit must clear the inducer casing, or there is no radial
        # passage to turn the flow through.
        rc1 = float(np.squeeze(inlet.r_cas))
        r2 = float(np.squeeze(r2))
        if not rc1 < r2:
            raise DesignError(
                f"Mean-line type {self.type!r}: inlet casing radius {rc1:.4g} m "
                f"is not inside the rotor exit radius {r2:.4g} m. Lower Ma1_rel, "
                "raise htr1 or raise the pressure ratio."
            )

        return ml

    def _check_bounds(self):
        """Refuse variables that leave no physical design."""
        problems = []
        if not self.PR_tt > 1.0:
            problems.append(f"PR_tt={self.PR_tt} must exceed 1")
        if not self.phi1 > 0.0:
            problems.append(f"phi1={self.phi1} must be positive")
        if not self.Ma1_rel > 0.0:
            problems.append(f"Ma1_rel={self.Ma1_rel} must be positive")
        if not 0.0 <= self.htr1 < 1.0:
            problems.append(f"htr1={self.htr1} must lie in [0, 1)")
        if not self.DH > 0.0:
            problems.append(f"DH={self.DH} must be positive")
        if not abs(self.Alpha2_rel) < 90.0:
            problems.append(
                f"Alpha2_rel={self.Alpha2_rel} must lie strictly between -90 and 90"
            )
        if not self.mdot > 0.0:
            problems.append(f"mdot={self.mdot} must be positive")
        if problems:
            raise DesignError(
                f"Mean-line type {self.type!r}: " + "; ".join(problems) + "."
            )

    def backward(self, ml):
        inlet, outlet = ml.inlet, ml.outlet
        r1, r2 = inlet.r, outlet.r
        U2 = outlet.U
        return {
            # Design variables
            "PR_tt": outlet.Po / inlet.Po,
            "phi1": inlet.Vm / inlet.U,
            "Ma1_rel": inlet.Ma_rel,
            "htr1": ml.htr[0],
            "DH": outlet.V_rel / inlet.V_rel,
            "Alpha2_rel": outlet.Alpha_rel,
            "mdot": inlet.mdot,
            "Ys": self.loss_coefficient(ml),
            "Po1": inlet.Po,
            "To1": inlet.To,
            # Diagnostics
            "Alpha1": inlet.Alpha,
            "Ma1": inlet.Ma,
            "Ma2": outlet.Ma,
            "Ma2_rel": outlet.Ma_rel,
            "Alpha2": outlet.Alpha,
            "Omega": inlet.Omega,
            "RR": r2 / r1,
            "r1": r1,
            "rh1": inlet.r_hub,
            "rc1": inlet.r_cas,
            "r2": r2,
            "span2": outlet.span,
            "rc1_r2": inlet.r_cas / r2,
            "psi": (outlet.ho - inlet.ho) / U2**2,
            "Ma_U2": U2 / inlet.ao,
            "eta_tt": ml.eta_tt,
            "eta_ts": ml.eta_ts,
        }
