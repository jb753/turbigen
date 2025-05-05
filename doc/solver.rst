Base class for flow solvers.

:program:`turbigen` is CFD-solver agnostic, in that all all pre- and
post-processing is done by native code.

Each CFD solver accepts different configuration options. Solver options and
their default values are listed below; override the defaults using the
`solver` section of the configuration file.

Adding a new solver only requires
functions to save CFD input files, execute the
solver, and read back the flow solution into :program:`turbigen`'s internal
data structures. See the :ref:`solver-custom` section for more details.



ember
-----

:program:`ember` is the native flow solver built into :program:`turbigen`.

.. list-table:: Parameters
   :widths: 20 15 15 50
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``smooth4``
     - ``float``
     - ``0.01``
     - Fourth-order smoothing factor.
   * - ``smooth2_adapt``
     - ``float``
     - ``1.0``
     - Second-order smoothing factor, adaptive on pressure.
   * - ``smooth2_const``
     - ``float``
     - ``0.0``
     - Second-order smoothing factor, constant throughout the flow.
   * - ``smooth_ratio_min``
     - ``float``
     - ``0.1``
     - Largest reduction in smoothing on a non-isotropic grid. Unity disables directional scaling, lower values clip the local smoothing factor to be sf_local >= sf * smooth_ratio_min.
   * - ``CFL``
     - ``float``
     - ``0.65``
     - Courant--Friedrichs--Lewy number, time step normalised by local wave speed and cell size. Reduced values are more stable but slower to converge.
   * - ``n_step``
     - ``int``
     - ``5000``
     - Number of time steps to run for.
   * - ``n_step_mix``
     - ``int``
     - ``5``
     - Number of time steps between mixing plane updates.
   * - ``n_step_dt``
     - ``int``
     - ``10``
     - Number of time steps between updates of the local time step.
   * - ``n_step_log``
     - ``int``
     - ``500``
     - Number of time steps between log prints.
   * - ``n_step_avg``
     - ``int``
     - ``1``
     - Number of time steps to average over.
   * - ``n_step_ramp``
     - ``int``
     - ``250``
     - Number of time steps to ramp smoothing and damping.
   * - ``n_loss``
     - ``int``
     - ``5``
     - Number of time steps between viscous force updates.
   * - ``nstep_damp``
     - ``int``
     - ``-1``
     - Number of steps to apply damping.
   * - ``damping_factor``
     - ``float``
     - ``25.0``
     - Negative feedback to damp down high residuals. Lower values are more stable.
   * - ``Pr_turb``
     - ``float``
     - ``1.0``
     - Turbulent Prandtl number.
   * - ``xllim_pitch``
     - ``float``
     - ``0.03``
     - Maximum mixing length as a fraction of the pitch.
   * - ``precision``
     - ``int``
     - ``1``
     - Precision of the solver. 1: single, 2: double.
   * - ``i_scheme``
     - ``int``
     - ``1``
     - Which time-stepping scheme to use. 0: scree, 1: super.
   * - ``i_loss``
     - ``int``
     - ``1``
     - Viscous loss model. 0: inviscid, 1: viscous.
   * - ``K_exit``
     - ``float``
     - ``0.5``
     - Relaxation factor for outlet forcing.
   * - ``K_inlet``
     - ``float``
     - ``0.5``
     - Relaxation factor for inlet forcing.
   * - ``K_mix``
     - ``float``
     - ``0.1``
     - Relaxation factor for mixing plane forcing.
   * - ``sf_mix``
     - ``float``
     - ``0.01``
     - Smoothing factor for uniform enthalpy and entropy downstream of mixing plane.
   * - ``print_conv``
     - ``bool``
     - ``True``
     - Print convergence history in the log.
   * - ``fmgrid``
     - ``float``
     - ``0.2``
     - Factor scaling the multigrid residual.
   * - ``multigrid``
     - ``tuple``
     - ``(2, 2, 2)``
     - Number of cells forming each multigrid level. (2, 2, 2) gives coarse cells of side length 2, 4, and 8 fine cells.
   * - ``area_avg_Pout``
     - ``bool``
     - ``True``
     - Force area-averaged outlet pressure to target, otherwise use uniform outlet pressure.

.. _solver-custom:

Custom solvers
--------------

To add a new solver, create a new class that inherits from :class:`turbigen.solvers.base.BaseSolver`.  and implement the following methods:

- :meth:`run`: Run the solver on the given grid and machine geometry.
- :meth:`robust`: Create a copy of the config with more robust settings.
- :meth:`restart`: Create a copy of the config with settings to restart from converged solution.

:class:`turbigen.solvers.base.BaseSolver` is a dataclass, so has an automatic constructor and
useful built-in methods. The configuration file `solver` section
is fed into the constructor as keyword arguments and becomes attributes of
the instance.
