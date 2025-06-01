Nomenclature
============

General conventions for variable names:

* Symbol or numeric subscripts are not separated by underscores: `Vx`, `Alpha1`.
* Accents and descriptive subscripts are separated by underscores: `Vt2_rel`, `mdot_stall`.
* Stagnation quantities are denoted by a subscript letter oh: `Po`, `ho_rel`.
* Depending on context, concatenation of symbols is either: a product, such as `rhoVx`; or a stack along the first array dimension, like `xrt`.

The yaw angle, `Alpha` or :math:`\alpha`, is defined as the angle between the flow direction and
the meridional direction:

.. math::
    \alpha = \arctan\left(\frac{V_\theta}{V_m}\right)

where :math:`V_\theta` is the tangential and :math:`V_m` the meridional
components of flow velocity. In the rotating frame, the relative yaw angle,
`Alpha_rel` or :math:`\alpha^\mathrm{rel}`, is defined as:

.. math::
    \alpha^\mathrm{rel} = \arctan\left(\frac{V_\theta^\mathrm{rel}}{V_m}\right)

The pitch angle, `Beta`, is defined as the angle between
the meridional flow direction and the axial direction:

.. math::
    \beta = \arctan\left(\frac{V_r}{V_x}\right)

where :math:`V_r` is the radial velocity component. Throughout the code, angles
have units of degrees.


=========== ============================================ =======
 Symbol      Quantity                                     Units
=========== ============================================ =======
`a`          Acoustic Speed                               m/s
`cp`         Specific heat capacity at constant pressure  J/kg/K
`cv`         Specific heat capacity at constant volume    J/kg/K
`gamma`      Ratio of specific heats
`h`          Specific enthalpy                            J/kg
`mu`         Dynamic viscosity                            kg/m/s
`P`          Pressure                                     Pa
`Pr`         Prandtl number
`rgas`       Specific gas constant                        J/kg/K
`rho`        Mass density                                 kg/m^3
`s`          Specific entropy                             J/kg/K
`T`          Temperature                                  K
`u`          Specific internal energy                     J/kg
`x`          Axial coordinate                             m
`r`          Radial coordinate                            m
`t`          Tangential coordinate                        rad
`m`          Meridional coordinate                        m
`Omega`      Shaft angular velocity                       rad/s
`y`          Cartesian vertical coordinate                m
`z`          Cartesian horizontal coordinate              m
`spf`        Span fraction coordinate
`V`          Velocity                                     m/s
`mdot`       Mass flow rate                               kg/s
`Alpha`      Yaw angle                                    deg
`Beta`       Pitch angle                                  deg
`U`          Blade speed                                  m/s
`e`          Specific total energy                        J/kg
`I`          Rothalpy                                     J/kg
`Ma`         Mach number
`A`          Area                                         m^2
`zeta`       Surface distance coordinate                  m
=========== ============================================ =======
