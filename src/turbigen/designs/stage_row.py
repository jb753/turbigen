"""One blade row of a repeating axial stage, compressor or turbine.

The row is set by the stage it belongs to, through the usual stage
coefficients at constant axial velocity and mean radius,

    phi = Vx / U,    psi = Delta ho / U^2,    Lam = Delta h_rotor / Delta h_stage,

with `psi` positive for a compressor and negative for a turbine. Euler's work
equation and the definition of reaction then fix the absolute swirl at rotor
inlet and exit,

    Vt_a / U = 1 - Lam - psi / 2,    Vt_b / U = 1 - Lam + psi / 2,

for either kind of machine. A repeating stage leaves as it entered, so the
rotor turns the flow from `Vt_a` to `Vt_b` and the stator from `Vt_b` back to
`Vt_a`, whichever of the two comes first in the stage.

Only the row asked for is built. A rotor rotates, so the spanwise design twists
its blades in the relative frame as a real rotor's are. A stator does not, but
its blade speed is still fixed by the blade Mach number, which is how the stage
coefficients are recovered from a stator alone.

The axial velocity and mean radius are held constant across the row, so at
finite Mach number the span changes with density to conserve mass.
"""

from typing import ClassVar

import numpy as np

from turbigen.design import DesignError, MeanLineDesign


class StageRow(MeanLineDesign):
    """One rotor or stator row of a repeating axial stage."""

    type: ClassVar[str] = "stage_row"
    n_row: ClassVar[int] = 1

    phi: float
    """Flow coefficient, axial velocity over blade speed [--]."""

    psi: float
    """Stage loading coefficient, stagnation enthalpy rise over blade speed
    squared [--]. Positive for a compressor, negative for a turbine."""

    Lam: float
    """Stage reaction, the rotor's share of the static enthalpy change [--]."""

    Mb: float
    """Blade Mach number, mean blade speed over inlet stagnation sound speed
    [--]."""

    htr: float
    """Hub-to-tip ratio at the row inlet [--]."""

    rotor: bool
    """True for the rotor of the stage, False for the stator."""

    r_rms: float = 0.5
    """Mean radius, constant through the row [m]."""

    Ys: float = 0.0
    """Pseudo entropy loss coefficient at the row exit [--]."""

    Po1: float = 1e5
    """Inlet stagnation pressure, absolute frame [Pa]."""

    To1: float = 300.0
    """Inlet stagnation temperature, absolute frame [K]."""

    #
    # SHARED DEFINITIONS
    #

    @staticmethod
    def entropy_rise(Ys, ao1, To1):
        """Entropy rise implied by a pseudo loss coefficient."""
        return Ys * (0.5 * ao1**2) / To1

    @staticmethod
    def swirl_ratios(psi, Lam):
        """Absolute swirl over blade speed at rotor inlet and exit."""
        return np.array([1.0 - Lam - 0.5 * psi, 1.0 - Lam + 0.5 * psi])

    def blade_speed(self, ml):
        """Mean blade speed of the stage this row belongs to [m/s].

        A rotor carries it. A stator does not, so it is taken from the blade
        Mach number, the one design variable a stator cannot report.
        """
        if self.rotor:
            return float(ml.inlet.Omega * ml.inlet.r)
        return self.Mb * float(ml.inlet.ao)

    #
    # DESIGN
    #

    def forward(self, fluid):
        if not self.phi > 0.0:
            raise DesignError(f"Mean-line type {self.type!r}: phi={self.phi} <= 0.")
        if not 0.0 < self.htr < 1.0:
            raise DesignError(
                f"Mean-line type {self.type!r}: htr={self.htr} not in (0, 1)."
            )

        ml = self.allocate(fluid)
        flat = ml.flat

        rhoo1, uo1 = fluid.set_P_T(self.Po1, self.To1)
        ao1 = fluid.get_a(rhoo1, uo1)
        s1 = fluid.get_s(rhoo1, uo1)
        ho1 = fluid.get_h(rhoo1, uo1)

        U = self.Mb * ao1
        Vx = self.phi * U
        Vt_a, Vt_b = self.swirl_ratios(self.psi, self.Lam) * U
        if self.rotor:
            Vt = np.array([Vt_a, Vt_b])
            ho = np.array([ho1, ho1 + self.psi * U**2])
        else:
            Vt = np.array([Vt_b, Vt_a])
            ho = np.array([ho1, ho1])
        s = s1 + np.array([0.0, self.entropy_rise(self.Ys, ao1, self.To1)])
        h = ho - 0.5 * (Vx**2 + Vt**2)
        if np.any(h <= 0.0):
            raise DesignError(
                f"Mean-line type {self.type!r}: no static enthalpy left at "
                f"Mb={self.Mb}; reduce the blade Mach number."
            )

        flat.set_h_s(h, s)
        flat.set_Vx(Vx)
        flat.set_Vr(0.0)
        flat.set_Vt(Vt)
        ml.set_r(self.r_rms)

        # Inlet area from the hub-to-tip ratio, outlet area from mass
        A1 = 2.0 * np.pi * self.r_rms**2 * (1.0 - self.htr**2) / (1.0 + self.htr**2)
        rho = flat.rho
        flat.set_Am(A1 * rho[0] / rho)

        ml.set_Omega(U / self.r_rms if self.rotor else 0.0)
        return ml

    def backward(self, ml):
        inlet, outlet = ml.inlet, ml.outlet
        U = self.blade_speed(ml)
        Vt = ml.flat.Vt
        Vt_a, Vt_b = (Vt[0], Vt[1]) if self.rotor else (Vt[1], Vt[0])
        V_rel = ml.flat.V_rel
        return {
            # Design variables
            "phi": float(inlet.Vx) / U,
            # Euler's work equation, which a stator's swirl change mirrors
            "psi": (Vt_b - Vt_a) / U,
            "Lam": 1.0 - 0.5 * (Vt_a + Vt_b) / U,
            "Mb": U / float(inlet.ao),
            "htr": inlet.htr,
            "rotor": bool(np.any(ml.flat.Omega != 0.0)),
            "r_rms": inlet.r,
            "Ys": (outlet.s - inlet.s) * inlet.To / (0.5 * inlet.ao**2),
            "Po1": inlet.Po,
            "To1": inlet.To,
            # Diagnostics, in the frame of the row
            "Alpha_rel": ml.flat.Alpha_rel,
            "Ma_rel": ml.flat.Ma_rel,
            "DH": V_rel[1] / V_rel[0],
            "htr2": outlet.htr,
        }
