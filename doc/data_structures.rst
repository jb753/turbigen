
Data structures
===============

This page documents the internal data structures used in :program:`turbigen`.
It is intended as a reference for users extending the program using custom
plugins or developers modifying the source code.


.. _state:

State
-----

Both perfect and real working fluids are represented by a :class:`State`
class, which has a common interface for setting and reading thermodynamic
properties. The interface allows the same mean-line design code to work
with any working fluid. :class:`State` does not store velocity information
and hence makes no distinction between static and stagnation states, the
handling of which is left to the calling code.

Setter methods
^^^^^^^^^^^^^^

The following methods are used to set the thermodynamic state of the fluid
to a new value. The object is updated in-place; a copy can be explicitly
created using :meth:`State.copy`. By the two-property rule, the setters all
take two arguments to uniquely specify the thermodynamic state. The
following methods are available:

.. list-table::
   :widths: 50 25 25
   :header-rows: 1

   * - Method
     - Arguments
     -
   * - ``State.set_h_s(h, s)``
     - Enthalpy
     - Entropy
   * - ``State.set_P_h(P, h)``
     - Pressure
     - Enthalpy
   * - ``State.set_P_rho(P, rho)``
     - Pressure
     - Density
   * - ``State.set_P_s(P, s)``
     - Pressure
     - Entropy
   * - ``State.set_P_T(P, T)``
     - Pressure
     - Temperature
   * - ``State.set_rho_u(rho, u)``
     - Density
     - Internal energy

Property attributes
^^^^^^^^^^^^^^^^^^^

Thermodynamic and transport properties of the fluid are accessed as attributes of the
:class:`State` object. The following properties are available:

.. list-table::
   :widths: 25 55 20
   :header-rows: 1

   * - Property
     - Description
     - Units

   * - ``State.a``
     - Acoustic speed
     - m/s
   * - ``State.cp``
     - Specific heat at constant pressure
     - J/kg/K
   * - ``State.cv``
     - Specific heat at constant volume
     - J/kg/K
   * - ``State.gamma``
     - Ratio of specific heats
     - --
   * - ``State.h``
     - Specific enthalpy
     - J/kg
   * - ``State.mu``
     - Kinematic viscosity
     - m^2/s
   * - ``State.P``
     - Pressure
     - Pa
   * - ``State.Pr``
     - Prandtl number
     - --
   * - ``State.rgas``
     - Specific gas constant
     - J/kg/K
   * - ``State.rho``
     - Density
     - kg/m^3
   * - ``State.s``
     - Specific entropy
     - J/kg/K
   * - ``State.T``
     - Temperature
     - K
   * - ``State.u``
     - Specific internal energy
     - J/kg

.. _flowfield:

Flow fields
-----------

Augmenting a thermodynamic state with velocity and coordinate data
allows the :class:`FlowField` class to represent a flow field.
Composite properties such as stagnation pressure and Mach number
can then be computed from the thermodynamic state and velocity vector.
Setting an angular velocity allows evaluation of quantities in a rotating
frame. Circumferential periodicity is represented by a number of blades.

Setter methods
^^^^^^^^^^^^^^

The :class:`FlowField` class has the same thermodynamic setter methods as
the :class:`State` class. Velocity and coordinate data are set directly
by assigning to the corresponding attributes.

Property attributes
^^^^^^^^^^^^^^^^^^^

In addition to all the pure thermodynamic properties defined in
:class:`State`, incorporating velocity and coordinate data allow the
:class:`FlowField` to provide the following other properties:

.. list-table::
   :widths: 25 60 15
   :header-rows: 1

   * - Property
     - Description
     - Units

   * - ``FlowField.Alpha``
     - Yaw angle
     - deg
   * - ``FlowField.Alpha_rel``
     - Relative frame yaw angle
     - deg
   * - ``FlowField.ao``
     - Stagnation acoustic speed
     - m/s
   * - ``FlowField.Beta``
     - Pitch angle
     - deg
   * - ``FlowField.e``
     - Specific total energy
     - J/kg
   * - ``FlowField.halfVsq``
     - Specific kinetic energy
     - J/kg
   * - ``FlowField.halfVsq_rel``
     - Relative frame specific kinetic energy
     - J/kg
   * - ``FlowField.ho``
     - Stagnation specific enthalpy
     - J/kg
   * - ``FlowField.ho_rel``
     - Relative frame stagnation specific enthalpy
     - J/kg
   * - ``FlowField.I``
     - Rothalpy
     - J/kg
   * - ``FlowField.Ma``
     - Mach number
     - --
   * - ``FlowField.Ma_rel``
     - Relative frame Mach number
     - --
   * - ``FlowField.Omega``
     - Reference frame angular velocity
     - rad/s
   * - ``FlowField.Po``
     - Stagnation pressure
     - Pa
   * - ``FlowField.Po_rel``
     - Relative frame stagnation pressure
     - Pa
   * - ``FlowField.r``
     - Radial coordinate
     - m
   * - ``FlowField.rhoe``
     - Volumetric total energy
     - J/m^3
   * - ``FlowField.rhorVt``
     - Volumetric angular momentum
     - kg/m^2/s
   * - ``FlowField.rhoVr``
     - Volumetric radial momentum
     - kg/m^2/s
   * - ``FlowField.rhoVt``
     - Volumetric angular momentum
     - kg/m^2/s
   * - ``FlowField.rhoVx``
     - Volumetric axial momentum
     - kg/m^2/s
   * - ``FlowField.rpm``
     - Reference frame revolutions per minute
     - rpm
   * - ``FlowField.t``
     - Circumferential coordinate
     - rad
   * - ``FlowField.tanAlpha``
     - Tangent of yaw angle
     - --
   * - ``FlowField.tanAlpha_rel``
     - Tangent of relative frame yaw angle
     - --
   * - ``FlowField.tanBeta``
     - Tangent of pitch angle
     - --
   * - ``FlowField.To``
     - Stagnation temperature
     - K
   * - ``FlowField.To_rel``
     - Relative frame stagnation temperature
     - K
   * - ``FlowField.U``
     - Blade speed
     - m/s
   * - ``FlowField.V``
     - Absolute velocity magnitude
     - m/s
   * - ``FlowField.V_rel``
     - Relative frame velocity magnitude
     - m/s
   * - ``FlowField.Vm``
     - Meridional velocity magnitude
     - m/s
   * - ``FlowField.Vr``
     - Radial velocity
     - m/s
   * - ``FlowField.Vt``
     - Circumferential velocity
     - m/s
   * - ``FlowField.Vt_rel``
     - Relative frame circumferential velocity
     - m/s
   * - ``FlowField.Vx``
     - Axial velocity
     - m/s
   * - ``FlowField.x``
     - Axial coordinate
     - m

.. _meanline:

Mean line
---------

The :class:`MeanLine` class encapsulates the quasi-one-dimensional geometry
and flow field of a turbomachine. In addition to thermodynamic states and
velocity vectors, it also contains a root-mean-square radii and annulus
areas. Assuming the span is perpendicular to the mean-line pitch angle,
These data are sufficient to determine hub and tip radii, and
the midspan blade angles.

Property attributes
^^^^^^^^^^^^^^^^^^^

In addition to the properties defined in
:class:`State` and
:class:`FlowField`, the :class:`MeanLine` class provides the following

.. list-table::
   :widths: 25 60 15
   :header-rows: 1

   * - Property
     - Description
     - Units

   * - ``MeanLine.A``
     - Annulus area
     - m^2
   * - ``MeanLine.eta_poly``
     - Total-to-total polytropic efficiency
     - --
   * - ``MeanLine.eta_ts``
     - Total-to-static isentropic efficiency
     - --
   * - ``MeanLine.eta_tt``
     - Total-to-total isentropic efficiency
     - --
   * - ``MeanLine.htr``
     - Hub-to-tip radius ratio
     - --
   * - ``MeanLine.mdot``
     - Mass flow rate
     - kg/s
   * - ``MeanLine.Nb``
     - Number of blades
     - --
   * - ``MeanLine.PR_ts``
     - Total-to-static pressure ratio
     - --
   * - ``MeanLine.PR_tt``
     - Total-to-total pressure ratio
     - --
   * - ``MeanLine.rhub``
     - Hub radius
     - m
   * - ``MeanLine.rmid``
     - Midspan radius
     - m
   * - ``MeanLine.rrms``
     - Root-mean-square radius
     - m
   * - ``MeanLine.rtip``
     - Tip radius
     - m
   * - ``MeanLine.span``
     - Span
     - m
