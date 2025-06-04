
Data Structures
===============

This page documents the internal data structures used in :program:`turbigen`.
It is intended as a reference for users extending the program using custom
plugins or developers modifying the source code.


Working fluids
-------------

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
   * - ``State.dhdP_rho``
     - Derivative of enthalpy with respect to pressure at constant density.
     -
   * - ``State.dhdrho_P``
     - Derivative of enthalpy with respect to density at constant pressure.
     -
   * - ``State.dsdP_rho``
     - Derivative of entropy with respect to pressure at constant density.
     -
   * - ``State.dsdrho_P``
     - Derivative of entropy with respect to density at constant pressure.
     -
   * - ``State.dudP_rho``
     - Derivative of internal energy with respect to pressure at constant density.
     -
   * - ``State.dudrho_P``
     - Derivative of internal energy with respect to density at constant pressure.
     -
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

The :class:`FlowField` class has extra setter methods as well as those
defined in :class:`State`. Omitting a coordinate or velocity argument
will leave the corresponding values unchanged.

.. list-table::
   :widths: 65 35
   :header-rows: 1

   * - Method
     - Arguments
   * - ``FlowField.set_conserved(conserved)``
     - Vector of conserved variables
   * - ``FlowField.set_Nb(Nb)``
     - Number of blades
   * - ``FlowField.set_Omega(Omega)``
     - Set reference frame angular velocity
   * - ``FlowField.set_V_Alpha_Beta(V, Alpha, Beta)``
     - Velocity magnitude and angles
   * - ``FlowField.set_Vxrt(Vx, Vr, Vt)``
     - Polar velocity vector
   * - ``FlowField.set_Vxyz(Vx, Vy, Vz)``
     - Cartesian velocity vector
   * - ``FlowField.set_xrt(x, r, t)``
     - Set polar coordinates
   * - ``FlowField.set_xyz(x, y, z)``
     - Set Cartesian coordinates

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
   * - ``FlowField.conserved``
     - Vector of conserved variables
     -
   * - ``FlowField.drhoe_dP_rho``
     - Derivative of volumetric total energy with respect to pressure at constant density
     -
   * - ``FlowField.drhoe_drho_P``
     - Derivative of volumetric total energy with respect to density at constant pressure
     -
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
   * - ``FlowField.Nb``
     - Number of blades, circumferential periodicity
     - --
   * - ``FlowField.Omega``
     - Reference frame angular velocity
     - rad/s
   * - ``FlowField.pitch``
     - Angular blade pitch, circumferential period
     - rad
   * - ``FlowField.Po``
     - Stagnation pressure
     - Pa
   * - ``FlowField.Po_rel``
     - Relative frame stagnation pressure
     - Pa
   * - ``FlowField.r``
     - Radial coordinate
     - m
   * - ``FlowField.rpm``
     - Reference frame revolutions per minute
     - rpm
   * - ``FlowField.t``
     - Circumferential coordinate
     - rad
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
