r"""A rotor-only axial fan, the design written in the tutorial.

Every equation here is one of the mean-line equations derived in
:doc:`/tutorial`, in the same order, and this file is what that page shows: it
is run by `tests/turbigen/test_examples.py` on every commit and solved by
`doc/generate_examples.py`, so the tutorial cannot narrate a design that no
longer works.

The design is explicit --- each variable follows from the ones before it, with
nothing to guess --- so `forward` constructs the mean line in one pass and has
no need of :meth:`~turbigen.design.MeanLineDesign.solve_for`.
"""

import numpy as np

from turbigen.design import MeanLineDesign


class Fan(MeanLineDesign):
    """A single rotor at fixed inlet stagnation conditions and no inlet swirl."""

    type: str = "fan"
    n_row: int = 1

    DPo: float
    """Stagnation pressure rise across the rotor [Pa]."""

    mdot: float
    """Mass flow rate [kg/s]."""

    phi: float
    """Flow coefficient, axial velocity over blade speed [--]."""

    psi: float
    """Loading coefficient, work over blade speed squared [--]."""

    htr: float
    """Inlet hub-to-tip ratio [--]."""

    eta_tt: float
    """Total-to-total isentropic efficiency [--]."""

    Po1: float = 1e5
    """Inlet stagnation pressure [Pa]."""

    To1: float = 300.0
    """Inlet stagnation temperature [K]."""

    def forward(self, fluid):
        # Inlet stagnation enthalpy and entropy
        rhoo1, uo1 = fluid.set_P_T(self.Po1, self.To1)
        ho1 = fluid.get_h(rhoo1, uo1)
        s1 = fluid.get_s(rhoo1, uo1)

        # Ideal exit stagnation enthalpy
        Po2 = self.Po1 + self.DPo
        ho2s = fluid.get_h(*fluid.set_P_s(Po2, s1))

        # Work from the definition of efficiency
        Dho = (ho2s - ho1) / self.eta_tt

        # Blade speed from the definition of loading coefficient
        U = np.sqrt(Dho / self.psi)

        # Axial velocity from the definition of flow coefficient
        Vx = self.phi * U

        # Exit swirl from the Euler work equation, no inlet swirl
        Vt2 = Dho / U

        # Exit entropy from actual work and pressure rise
        ho2 = ho1 + Dho
        s2 = fluid.get_s(*fluid.set_P_h(Po2, ho2))

        # Static states, stagnation enthalpy less KE
        Vt = np.array([0.0, Vt2])
        ho = np.array([ho1, ho2])
        s = np.array([s1, s2])
        h = ho - 0.5 * (Vx**2 + Vt**2)

        # Store the flow field
        ml = self.allocate(fluid)
        flat = ml.flat
        flat.set_h_s(h, s)
        flat.set_Vx(Vx)
        flat.set_Vr(0.0)
        flat.set_Vt(Vt)

        # Conservation of mass sets the annulus areas
        flat.set_Am(self.mdot / flat.rho / Vx)

        # Fix a constant mean radius using inlet HTR
        Am1 = flat.Am[0]
        r_rms = np.sqrt(Am1 / (2.0 * np.pi) * (1.0 + self.htr**2) / (1.0 - self.htr**2))
        ml.set_r_rms(r_rms)
        ml.set_Omega(U / r_rms)

        return ml

    def backward(self, ml):
        return {
            # Design variables
            "DPo": ml.outlet.Po - ml.inlet.Po,
            "mdot": ml.inlet.mdot,
            "phi": ml.inlet.Vx / ml.inlet.U,
            "psi": (ml.outlet.ho - ml.inlet.ho) / ml.inlet.U**2,
            "htr": ml.inlet.htr,
            "eta_tt": ml.eta_tt,
            "Po1": ml.inlet.Po,
            "To1": ml.inlet.To,
            # Optional diagnostic variables
            "PR_tt": ml.PR_tt,
            "eta_ts": ml.eta_ts,
        }
