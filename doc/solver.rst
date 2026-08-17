Flow solvers
============

Once the machine geometry is generated, and the fluid domain discretised
using a suitable mesh, the next step is to predict the flow field and hence
aerodynamic performance using a computational fluid dynamics (CFD) solver.

:program:`turbigen` is solver-agnostic: either the built in CFD solver or
an interface to an external code can be used for flow field prediction;
pre- and post-processing is done in common, native Python code. Adding a new
external solver only requires implementing functions to save input files,
execute the solver, and then read back the output file into
:program:`turbigen`'s internal data structures. See the
:ref:`solver-custom` section for more details.

Select which solver to use by setting the `type` key under the `solver` section
of the configuration file. Valid choices are:

- `ember`: :ref:`Enhanced Multi-Block solvER<solver-ember>`
- `ts3`: :ref:`solver-ts3`
- `ts4`: :ref:`solver-ts4`

Each CFD solver accepts different configuration options. Solver options and
their default values are listed below; override the defaults using the
`solver` section of the configuration file. For example, to use the
built-in :program:`ember` solver, with a reduced damping factor:

.. code-block:: yaml

    solver:
      type: ember
      n_step: 1000  # Case-dependent
      n_step_avg: 250  # Typically ~1/4 of n_step
      damping_factor: 10.  # Override default value of 25.0


Every solver accepts a `soft_start` flag, which runs a precursor simulation
with more robust settings before running the actual simulation. Enabling
this option should be the first step when a simulation fails to converge.
If a design does not run with `soft_start` enabled, this suggests the mesh
is poor quality, or the aerodynamic design itself is not feasible.



.. _solver-ember:

ember
-----

:program:`ember` is the 'Enhanced MultiBlock flow solvER' built into :program:`turbigen`. It is a hybrid Python--Fortran reimplementation of the classic :cite:t:`Denton1992,Denton2017` algorithms
for compressible turbomachinery flows, with a few enhancements.

To use this solver, add the following to your configuration file:

.. code-block:: yaml

    solver:
      type: emb
      n_step: 2000  # Case-dependent
      n_step_avg: 500  # Typically ~0.25 n_step

Configuration options
~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 10 10 10 70
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``CFL``
     - ``float``
     - ``0.65``
     - Courant--Friedrichs--Lewy number, time step normalised by local wave speed and cell size.
   * - ``K_exit``
     - ``float``
     - ``0.5``
     - Relaxation factor for outlet boundary.
   * - ``K_inlet``
     - ``float``
     - ``0.5``
     - Relaxation factor for inlet boundary.
   * - ``K_mix``
     - ``float``
     - ``0.1``
     - Relaxation factor for mixing plane.
   * - ``Pr_turb``
     - ``float``
     - ``1.0``
     - Turbulent Prandtl number.
   * - ``area_avg_Pout``
     - ``bool``
     - ``True``
     - Force area-averaged outlet pressure to target, otherwise use uniform outlet pressure.
   * - ``damping_factor``
     - ``float``
     - ``25.0``
     - Negative feedback to damp down high residuals. Smaller values are more stable.
   * - ``fmgrid``
     - ``float``
     - ``0.2``
     - Factor scaling the multigrid residual.
   * - ``i_loss``
     - ``int``
     - ``1``
     - Viscous loss model. 0: inviscid, 1: viscous.
   * - ``i_scheme``
     - ``int``
     - ``1``
     - Which time-stepping scheme to use. 0: scree, 1: super.
   * - ``multigrid``
     - ``tuple``
     - ``(2, 2, 2)``
     - Number of cells forming each multigrid level. `(2, 2, 2)` gives coarse cells of side length 2, 4, and 8 fine cells.
   * - ``n_loss``
     - ``int``
     - ``5``
     - Number of time steps between viscous force updates.
   * - ``n_step``
     - ``int``
     - ``5000``
     - Total number of time steps to run.
   * - ``n_step_avg``
     - ``int``
     - ``1``
     - Number of final time steps to average over.
   * - ``n_step_dt``
     - ``int``
     - ``10``
     - Number of time steps between updates of the local time step.
   * - ``n_step_log``
     - ``int``
     - ``500``
     - Number of time steps between log prints.
   * - ``n_step_mix``
     - ``int``
     - ``5``
     - Number of time steps between mixing plane updates.
   * - ``n_step_ramp``
     - ``int``
     - ``250``
     - Number of inital time steps to ramp smoothing and damping down.
   * - ``nstep_damp``
     - ``int``
     - ``-1``
     - Number of steps to apply damping, -1 for all steps.
   * - ``precision``
     - ``int``
     - ``1``
     - Precision of the solver. 1: single, 2: double.
   * - ``print_conv``
     - ``bool``
     - ``True``
     - Print convergence history in the log.
   * - ``sf_mix``
     - ``float``
     - ``0.01``
     - Smoothing factor for uniform enthalpy and entropy downstream of mixing plane.
   * - ``smooth2_adapt``
     - ``float``
     - ``1.0``
     - Second-order smoothing factor, adaptive on pressure.
   * - ``smooth2_const``
     - ``float``
     - ``0.0``
     - Second-order smoothing factor, constant throughout the flow.
   * - ``smooth4``
     - ``float``
     - ``0.01``
     - Fourth-order smoothing factor.
   * - ``smooth_ratio_min``
     - ``float``
     - ``0.1``
     - Largest directional reduction in smoothing on a non-isotropic grid. Unity disables directional scaling.
   * - ``xllim_pitch``
     - ``float``
     - ``0.03``
     - Maximum mixing length as a fraction of row pitch.

.. _solver-ts3:

Turbostream 3
-------------

Turbostream 3 is a multi-block structured, GPU-accelerated Reynolds-averaged
Navier--Stokes code developed by :cite:t:`Brandvik2011`.

To use this solver, add the following to your configuration file:

.. code-block:: yaml

    solver:
      type: ts3
      nstep: 10000  # Case-dependent
      nstep_avg: 2500  # Typically ~0.25 nstep

Configuration options
~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 10 10 10 70
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``Lref_xllim``
     - ``str``
     - ``pitch``
     - Mixing length characteristic dimension, "pitch" or "span".
   * - ``atol_eta``
     - ``float``
     - ``0.005``
     - Absolute tolerance on drift in isentropic efficiency.
   * - ``cfl``
     - ``float``
     - ``0.4``
     - Courant--Friedrichs--Lewy number, reduce for more stability.
   * - ``dampin``
     - ``float``
     - ``25.0``
     - Negative feedback factor, reduce for more stability.
   * - ``environment_script``
     - ``Path``
     - ``/usr/local...``
     - Setup environment shell script to be sourced before running.
   * - ``facsecin``
     - ``float``
     - ``0.005``
     - Fourth-order smoothing feedback factor, increase for more stability.
   * - ``fmgrid``
     - ``float``
     - ``0.2``
     - Multigrid factor, reduce for more stability.
   * - ``ilos``
     - ``int``
     - ``2``
     - Viscous model, 0 for inviscid, 1 for mixing-length, 2 for Spalart-Allmaras.
   * - ``nchange``
     - ``int``
     - ``2000``
     - At start of simulation, ramp smoothing and damping over this many time steps.
   * - ``nstep``
     - ``int``
     - ``10000``
     - Number of time steps.
   * - ``nstep_avg``
     - ``int``
     - ``5000``
     - Average over the last `nstep_avg` steps of the calculation.
   * - ``nstep_soft``
     - ``int``
     - ``0``
     - Number of steps for soft start precursor simulation.
   * - ``rfin``
     - ``float``
     - ``0.5``
     - Inlet relaxation factor, reduce for low-Mach flows.
   * - ``rfmix``
     - ``float``
     - ``0.0``
     - Mixing plane relaxation factor.
   * - ``rtol_mdot``
     - ``float``
     - ``0.01``
     - Relative tolerance on mass flow conservation error and drift.
   * - ``sa_ch2``
     - ``float``
     - ``0.6``
     - Convert the configuration to a dictionary.
   * - ``sa_helicity_option``
     - ``int``
     - ``0``
     - Spalart--Allmaras turbulence model helicity correction.
   * - ``sfin``
     - ``float``
     - ``0.5``
     - Proportion of second-order smoothing, increase for more stability.
   * - ``tvr``
     - ``float``
     - ``10.0``
     - Initial guess of turbulent viscosity ratio.
   * - ``xllim``
     - ``float``
     - ``0.03``
     - Mixing length limit as a fraction of characteristic dimension.

.. _solver-ts4:

Turbostream 4
-------------

Turbostream 4 is an unstructured, GPU-accelerated Reynolds-averaged
Navier--Stokes code developed by Turbostream Ltd.

To use this solver, add the following to your configuration file:

.. code-block:: yaml

    solver:
      type: ts4
      nstep: 10000  # Case-dependent
      nstep_avg: 2500  # Typically ~0.25 nstep


.. _solver-ts4-tables:

Real gas tables generation
~~~~~~~~~~~~~~~~~~~~~~~~~~

For real gas simulations, working fluid property tables must be
pre-generated before the calculation. This can be done using the
:meth:`turbigen.tables.make_tables` function following the example script below:

.. code-block:: python

    from turbigen.tables import make_tables

    fluid_name = "water"  # Fluid name in CoolProp
    smin = 7308.0  # Minimum entropy in J/kg/K
    smax = 7600.0  # Maximum entropy in J/kg/K
    Pmin = 37746.0  # Minimum pressure in Pa
    Tmax = 550.0  # Maximum temperature in K
    ni = 200  # Number of interpolation points in each direction
    new_npz_path = "water_new.npz"  # Path to save the new tables

    make_tables(fluid_name, smin, smax, Pmin, Tmax, ni, new_npz_path)


The enthalpy and entropy datums are those used by CoolProp, so in general

.. math::

    h &= c_p (T - T_\mathrm{ref}) \\
    s &= c_p \ln \left( \frac{T}{T_\mathrm{ref}} \right) - R \ln \left( \frac{P}{P_\mathrm{ref}} \right)

This means that the correct numerical values for the entropy limits are not
immediately obvious. :program:`turbigen` will print out numerical values for
the limits calculated from the nominal mean-line design. These should be,
however, extended by some margin of safety. It is vital that the limits
of the tables are wide enough to cover fluid property values over the
entire flow field, including local features like the suction peak, shock
waves and boundary layers.

Finally, in the configuration file, specify the path to the tables:

.. code-block:: yaml

    solver:
      type: ts4
      tables_path: water_new.npz

Some notes on real gas calculations:

- The real gas working fluid is less stable than the ideal gas, so take
  care with mesh generation and avoid racy solver settings.
- There is no handling of phase changes in the tables, so the fluid must
  remain a single phase for accurate results.
- Of order 1000 points may be required in each direction to get
  discretisation-independent results.

Configuration options
~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 10 10 10 70
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``area_avg_pout``
     - ``bool``
     - ``True``
     - Allow non-uniform outlet pressure, force area-average to target.
   * - ``cfl``
     - ``float``
     - ``25.0``
     - Courant--Friedrichs--Lewy number setting the time step.
   * - ``cfl_ramp_nstep``
     - ``int``
     - ``500``
     - Ramp the CFL number up over this many initial time steps.
   * - ``cfl_ramp_st``
     - ``int``
     - ``1.0``
     - Starting value for CFL ramp.
   * - ``custom_pipeline``
     - ``str``
     - `` ``
     - Specify a custom pipeline to convert Turbostream 3 to 4 input file. Should run using pvpython and take two command-line arguments like `pvpython custom_pipeline.py input_ts3.hdf5 input_ts4`
   * - ``environment_script``
     - ``Path``
     - ``/usr/local...``
     - Setup shell script to be sourced before running.
   * - ``implicit_scheme``
     - ``int``
     - ``1``
     - 1: implicit, 0: explicit time marching.
   * - ``interpolation_update``
     - ``int``
     - ``1``
     - Explicit with a slow CFL ramp.
   * - ``nstep``
     - ``int``
     - ``5000``
     - Number of time steps for the calculation.
   * - ``nstep_avg``
     - ``int``
     - ``1000``
     - Number of final time steps to average over.
   * - ``nstep_soft``
     - ``int``
     - ``5000``
     - Number of time steps for soft start.
   * - ``nstep_ts3``
     - ``int``
     - ``0``
     - Number of steps to run a Turbostream 3 initial guess
   * - ``outlet_tag``
     - ``str``
     - ``Outlet``
     - String to identify the outlet boundary condition in the TS4 input file. Only requires changing for custom pipelines.
   * - ``tables_path``
     - ``str``
     - `` ``
     - Path to gas tables npz for real working fluids. See :ref:`solver-ts4-tables`.
   * - ``viscous_model``
     - ``int``
     - ``2``
     - Turbulence model, 0: inviscid, 1: laminar, 2:  Spalart-Allmaras, 3: k-omega.

.. _solver-custom:

Custom solvers
--------------

To add a new solver, create a new Python module in the user-defined plugin
directory, say `./plug/my_solver.py` and set `plugdir: ./plug` in the
configuration file. Write a new class that inherits from
:class:`turbigen.solvers.base.BaseSolver`  and implement the following
methods:

- :meth:`run`: Run the solver on the given grid and machine geometry.
- :meth:`robust`: Create a copy of the config with more robust settings.
- :meth:`restart`: Create a copy of the config with settings to restart from converged solution.

:class:`turbigen.solvers.base.BaseSolver` is a dataclass, so has an automatic constructor and
useful built-in methods. The configuration file `solver` section
is fed into the constructor as keyword arguments and becomes attributes of
the instance.
