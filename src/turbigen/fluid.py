"""Working fluids for equation-of-state handling.

:program:`turbigen` does not evaluate thermodynamic properties itself: it
outsources equation of state handling to the dependency
:class:`ember.fluid.Fluid`. The :ref:`fluid: <config-fluid>` key in the input
file contains everything needed to instantiate one of these equations of
state, via :meth:`Fluid.eos`.

Once built, the fluid is passed to
:meth:`~turbigen.design.MeanLineDesign.design` and onwards to the CFD solver.
The whole interface is comprehensively documented in :mod:`ember.fluid`. It is
good practice to always use the fluid interface rather than taking any
perfect-gas shortcuts, to allow the same design to be run with a real gas
later.

.. _fluid-builtin:

Built-in equations of state
^^^^^^^^^^^^^^^^^^^^^^^^^^^

:class:`PerfectFluid` (``perfect``) is a perfect gas with constant specific
heats, taking ``cp``, ``gamma``, ``mu`` and ``Pr``.

:class:`RealFluid` (``real``) is a real gas defined by Legendre-polynomial fits
of the compressibility factor, entropy, viscosity and conductivity over a box
in density and internal energy. It carries the fit coefficients and the box
bounds, which are produced by fitting a reference equation of state such as
CoolProp over the expected operating range.

The full field list for each is in the :ref:`configuration reference
<config-fluid>`, generated from the classes below.


.. _fluid-scales:

Reference scales and datum
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The non-dimensionalisation reference scales and the entropy datum are not
config inputs. They are not properties of the fluid but of the design:
:meth:`turbigen.meanline.MeanLine.get_referenced_fluid` derives them once the
mean line exists, and the mesher applies them to the grid, whose numerical
conditioning is what they are for. See :ref:`ember:reference-scales` and
:ref:`ember:datum-state`.

Since only changes in internal energy and entropy are physical, the datum
level is arbitrary. A real gas needs one inside its fit box to be
constructible at all, and ember places it at the centre of the box, which is
in range by construction. A design that wants a better-conditioned datum for
its own pass moves it with :meth:`~ember.fluid.Fluid.change_datum`.


.. _fluid-custom:

Writing an equation of state
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Subclass :class:`Fluid`, set a ``type``, declare the parameters as dataclass
fields, and implement :meth:`~Fluid.eos` to return an :class:`ember.fluid.Fluid`
built from them. Like any node, it is picked up from a ``turbigen_plugins``
directory beside the input file and need not be installed.

"""

from typing import ClassVar

import ember.fluid

from turbigen.node import Node


class Fluid(Node):
    """Specify the equation of state of the working fluid.

    The :doc:`/fluid` page covers the built-in equations of state and how a
    fluid is used in a design.
    """

    def eos(self) -> ember.fluid.Fluid:
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
    # are not inputs: MeanLine.get_referenced_fluid derives them from the design
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

    delta: tuple[tuple[float, ...], ...]
    """Legendre coefficients of the viscosity surface [--], normalised by
    :attr:`mu_c`."""

    gamma: tuple[tuple[float, ...], ...]
    """Legendre coefficients of the conductivity surface [--], normalised by
    :attr:`kappa_c`."""

    rho_lim: tuple[float, float]
    """Density bounds of the fit box [kg/m^3]."""

    u_lim: tuple[float, float]
    """Internal energy bounds of the fit box [J/kg], on the datum the
    coefficients were fitted against."""

    Rgas: float
    """Specific gas constant [J/kg/K]."""

    mu_c: float
    """Dynamic viscosity at the centre of the fit box [kg/m/s], the scale the
    :attr:`delta` surface is normalised by."""

    kappa_c: float
    """Thermal conductivity at the centre of the fit box [W/m/K], the scale
    the :attr:`gamma` surface is normalised by."""

    scale_visc: float = 1.0
    """Factor multiplying the viscosity, for sweeping Reynolds number without
    touching the fit [--]."""

    # The reference scales and the entropy datum are absent for the reason
    # given on PerfectFluid above: they are properties of the design, not the
    # fluid, and MeanLine.get_referenced_fluid sets the datum that matters
    # before the grid ever sees it. ember defaults the datum to the centre of
    # the fit box, which is always in range, and a design that needs a
    # better-conditioned one for its own pass moves it with change_datum.

    def eos(self) -> ember.fluid.RealFluid:
        # The coefficients go across as the tuples the node holds; ember's
        # constructor is what turns them into arrays, and it is the only place
        # that should, so the node stays a value that pickles as one.
        return ember.fluid.RealFluid(
            alpha=self.alpha,
            beta=self.beta,
            delta=self.delta,
            gamma=self.gamma,
            rho_lim=self.rho_lim,
            u_lim=self.u_lim,
            Rgas=self.Rgas,
            mu_c=self.mu_c,
            kappa_c=self.kappa_c,
            scale_visc=self.scale_visc,
        )
