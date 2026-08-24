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

from typing import ClassVar

import numpy as np

from turbigen.design import MeanLineDesign


class Fan(MeanLineDesign):
    """A single rotor at fixed inlet stagnation conditions and no inlet swirl."""

    type: ClassVar[str] = "fan"
    n_row: ClassVar[int] = 1

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

    etatt: float
    """Total-to-total isentropic efficiency [--].

    A guess, in that it is an input to a design whose losses are not known
    until the flow is solved. The `mean_line` iterator moves it onto what the
    CFD achieves.
    """

    Po1: float = 1e5
    """Inlet stagnation pressure [Pa]."""

    To1: float = 300.0
    """Inlet stagnation temperature [K]."""

    #
    # SHARED DEFINITIONS
    #
    # Written once and called from both directions. A formula duplicated
    # between forward and backward is free to drift.
    #

    @staticmethod
    def blade_speed(ml):
        """Blade speed at the mean radius [m/s].

        The mean radius is constant through this machine, so either station
        gives the same number; the inlet is taken because that is where the
        flow coefficient is defined.
        """
        return ml.inlet.U

    @staticmethod
    def work(ml):
        r"""Specific work input, :math:`\Delta h_0` [J/kg]."""
        return ml.outlet.ho - ml.inlet.ho

    def flow_coefficient(self, ml):
        r"""Flow coefficient :math:`\phi = V_x/U` [--]."""
        return ml.inlet.Vx / self.blade_speed(ml)

    def loading(self, ml):
        r"""Loading coefficient :math:`\psi = \Delta h_0/U^2` [--]."""
        return self.work(ml) / self.blade_speed(ml) ** 2

    #
    # DESIGN
    #

    def forward(self, fluid):
        ml = self.allocate(fluid)

        # Inlet stagnation state. This is the Fluid interface, which returns
        # the state rather than storing it -- not to be confused with
        # Block.set_P_T, which mutates and returns None.
        rhoo1, uo1 = ml.fluid.set_P_T(self.Po1, self.To1)
        ho1 = ml.fluid.get_h(rhoo1, uo1)
        s1 = ml.fluid.get_s(rhoo1, uo1)

        # Ideal exit stagnation enthalpy: the real exit pressure reached at the
        # inlet entropy. Nothing here assumes a perfect gas, so the same design
        # runs against a real-gas equation of state.
        Po2 = self.Po1 + self.DPo
        ho2s = ml.fluid.get_h(*ml.fluid.set_P_s(Po2, s1))

        # Work from the definition of efficiency
        Dho = (ho2s - ho1) / self.etatt

        # Blade speed from the definition of loading coefficient
        U = np.sqrt(Dho / self.psi)

        # Axial velocity from the definition of flow coefficient
        Vx = self.phi * U

        # Exit swirl from the Euler work equation, there being none at inlet
        Vt2 = Dho / U

        # The exit entropy follows from the pressure rise asked for and the
        # work just found: it is the loss the efficiency guess implies.
        ho2 = ho1 + Dho
        s2 = ml.fluid.get_s(*ml.fluid.set_P_h(Po2, ho2))

        # Static states, from stagnation enthalpy less kinetic energy at the
        # same entropy
        Vt = np.array([0.0, Vt2])
        ho = np.array([ho1, ho2])
        s = np.array([s1, s2])
        h = ho - 0.5 * (Vx**2 + Vt**2)

        flat = ml.flat
        flat.set_h_s(h, s)
        flat.set_Vx(Vx)
        flat.set_Vr(0.0)
        flat.set_Vt(Vt)

        # Conservation of mass sets the annulus areas
        flat.set_Am(self.mdot / flat.rho / Vx)

        # The inlet hub-to-tip ratio fixes the mean radius, which is then held
        # constant through the machine, so the exit annulus opens or closes
        # about it as its own area requires.
        Am1 = flat.Am[0]
        r_rms = np.sqrt(Am1 / (2.0 * np.pi) * (1.0 + self.htr**2) / (1.0 - self.htr**2))
        ml.set_r(r_rms)
        ml.set_Omega(U / r_rms)

        return ml

    def backward(self, ml):
        return {
            # Design variables
            "DPo": ml.outlet.Po - ml.inlet.Po,
            "mdot": ml.inlet.mdot,
            "phi": self.flow_coefficient(ml),
            "psi": self.loading(ml),
            "htr": ml.inlet.htr,
            "etatt": ml.eta_tt,
            "Po1": ml.inlet.Po,
            "To1": ml.inlet.To,
            # Diagnostics
            "PR_tt": ml.PR_tt,
            "eta_ts": ml.eta_ts,
        }
