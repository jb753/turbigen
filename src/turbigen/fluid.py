"""Working fluids.

A :class:`Fluid` node holds the parameters of an equation of state as they
appear in the config file. The ember object that does the thermodynamics is
built on demand by :meth:`Fluid.eos`, and is deliberately not stored: a node is
a value, and keeping a derived object on it would put it in the instance
``__dict__``, where pickling a config would drag it along.
"""

from typing import ClassVar

import ember.fluid

from turbigen.node import Node


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
    # are not inputs: MeanLine.referenced_fluid derives them from the design
    # once it exists, and the mesher applies them to the grid, which is the
    # object whose conditioning matters. Declaring them here would put five
    # values in every config file that a user cannot usefully set and that are
    # replaced before they are ever read -- which is what the package this
    # replaces does, defaulting all five to 1.0 and overwriting every one.

    def eos(self) -> ember.fluid.PerfectFluid:
        return ember.fluid.PerfectFluid(
            cp=self.cp,
            gamma=self.gamma,
            mu=self.mu,
            Pr=self.Pr,
        )


class RealFluid(Fluid):
    """A real gas defined by a fitted entropy surface."""

    type: ClassVar[str] = "real"

    alpha: tuple[tuple[float, ...], ...]
    """Legendre coefficients of the compressibility factor Z(rho, u) [--]."""

    beta: tuple[float, ...]
    """Legendre coefficients of s/R along the reference isochor [--]."""

    rho_lim: tuple[float, float]
    """Density bounds of the fit box [kg/m^3]."""

    u_lim: tuple[float, float]
    """Internal energy bounds of the fit box [J/kg], on the datum the
    coefficients were fitted against."""

    rho_isochor: float
    """Density of the isochor the entropy integral starts from [kg/m^3]."""

    Rgas: float
    """Specific gas constant [J/kg/K]."""

    mu: float
    """Dynamic viscosity [kg/m/s]."""

    Pr: float = 0.7
    """Prandtl number [--]."""

    P_dtm: float | None = None
    """Datum pressure where u = s = 0 [Pa]. Must lie in the fit box.

    Omit it and ember places the datum at the centre of the box, which is
    inside it by construction."""

    T_dtm: float | None = None
    """Datum temperature where u = s = 0 [K]. As :attr:`P_dtm`."""

    # The three reference scales are absent for the reason given on
    # PerfectFluid above, and the datum is optional rather than absent, which
    # is the one place a real gas departs from that argument. A fitted surface
    # exists only inside its box and the datum has to lie in there, so unlike
    # a perfect gas -- whose datum is free, and is therefore left entirely to
    # MeanLine.referenced_fluid -- a real gas needs one that is constructible
    # before there is any design to derive it from. ember defaults it to the
    # centre of the fit box, which the coefficients here already determine, so
    # a config only carries the datum when it wants a different one. The
    # conditioning pass still moves it afterwards either way.

    def eos(self) -> ember.fluid.RealFluid:
        # The coefficients go across as the tuples the node holds; ember's
        # constructor is what turns them into arrays, and it is the only place
        # that should, so the node stays a value that pickles as one.
        return ember.fluid.RealFluid(
            alpha=self.alpha,
            beta=self.beta,
            rho_lim=self.rho_lim,
            u_lim=self.u_lim,
            rho_isochor=self.rho_isochor,
            Rgas=self.Rgas,
            mu=self.mu,
            Pr=self.Pr,
            P_dtm=self.P_dtm,
            T_dtm=self.T_dtm,
        )
