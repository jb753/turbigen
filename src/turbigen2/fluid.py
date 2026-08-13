"""Working fluids.

A :class:`Fluid` node holds the parameters of an equation of state as they
appear in the config file. The ember object that does the thermodynamics is
built on demand by :meth:`Fluid.eos`, and is deliberately not stored: a node is
a value, and keeping a derived object on it would put it in the instance
``__dict__``, where pickling a config would drag it along.
"""

from typing import ClassVar

import ember.fluid

from turbigen2.node import Node


class Fluid(Node):
    """Base for equations of state."""

    def eos(self) -> ember.fluid._Fluid:
        """Return the ember fluid object for this equation of state."""
        raise NotImplementedError(f"{type(self).__name__} must implement eos(self)")


class PerfectFluid(Fluid):
    """A perfect gas with constant specific heats."""

    type: ClassVar[str] = "perfect"

    cp: float
    """Specific heat at constant pressure [J/kg/K]."""

    gamma: float
    """Ratio of specific heats [--]."""

    mu: float
    """Dynamic viscosity [kg/m/s]."""

    Pr: float = 0.7
    """Prandtl number [--]."""

    # The non-dimensionalisation reference scales and the entropy datum
    # (V_ref, rho_ref, Rgas_ref, P_dtm, T_dtm) are deliberately absent. They
    # are not inputs: MeanLine.adjust_ref derives them from the design once it
    # exists and applies them with change_ref/change_datum. Declaring them here
    # would put five values in every config file that a user cannot usefully
    # set and that are overwritten before they are ever read.

    def eos(self) -> ember.fluid.PerfectFluid:
        return ember.fluid.PerfectFluid(
            cp=self.cp,
            gamma=self.gamma,
            mu=self.mu,
            Pr=self.Pr,
        )
