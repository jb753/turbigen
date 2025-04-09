Configuration file format
=========================

The working example below illustrates the general layout of a :program:`turbigen` configuration file. More specimens are available in the :file:`examples` directory.

.. literalinclude:: ../examples/turbine_cascade.yaml
   :language: yaml

Input data is specified in a text-based `YAML
<https://en.wikipedia.org/wiki/YAML>`_ format. Comments are prefixed with `#`,
and indentation is significant. A configuration file
may contain the following top-level keys:

* :ref:`cnf-workdir`
* :ref:`cnf-inlet`
* :ref:`ml_conf`
* :ref:`cnf-annulus`
* :ref:`cnf-blades`
* :ref:`cnf-mesh`
* :ref:`cnf-solver`
* :ref:`cnf-operating_point`
* :ref:`cnf-iterate`
* :ref:`cnf-job`

The following sections give more detail on each of the keys.

.. _cnf-workdir:

workdir
--------

Specifies relative or absolute path to a working directory for the run. If
present, `*` is replaced with a new numbered directory on each invocation of
the program. Examples:

.. code-block:: yaml

   workdir: runs/axial_turbine

   workdir: /home/james/turbigen/runs/radial

   workdir: design_*
   # produces design_0000, design_0001, ... each time turbigen is run


.. _cnf-inlet:

inlet
-----

The thermodynamic state at inlet to the turbomachine is set by fixing stagnation pressure,
`Po`, and stagnation temperature `To`. For perfect gases, the fluid properties `cp`, `mu`,
and `gamma` can be directly prescribed. For real gases, only the name of the
working fluid in `CoolProp nomenclature
<http://www.coolprop.org/fluid_properties/PurePseudoPure.html#list-of-fluids>`_
is required, and the fluid properties are obtained from tabulations. Examples:

.. code-block:: yaml

   # Perfect gas
   inlet:
     Po: 1e5  # Stagnation pressure [Pa]
     To: 300.  # Stagnation temperature [K]
     cp: 1005.  # Specific heat capacity at constant pressure [J/kg/K]
     mu: 1.8e-5  # Working fluid dynamic viscosity [kg/m/s]
     gamma: 1.4  # Working fluid ratio of specific heats [-]

   # Real gas
   inlet:
     Po: 1e5  # Stagnation pressure [Pa]
     To: 400.  # Stagnation temperature [K]
     fluid_name: water  # Name of working fluid

For more information on how :program:`turbigen` abstracts the working fluid, see :py:mod:`turbigen.fluid`

.. _ml_conf:

mean_line
---------

Under the `mean_line` key, a `type` key is required to specify the architecture
or topology of turbomachine to design. This can either be: one the built-in
mean-lines listed in in :py:mod:`turbigen.meanline`, or a file path to a
user-written mean-line module that provides the correct functions. Also
specified are the aerodynamic or geometric parameters needed to solve for the
flow along the mean-line for the current turbomachine type. The design
variables are fed directly into the `forward` mean-line design functions in
:py:mod:`turbigen.meanline` (see the :ref:`Mean line page <ml>` for more
detail) The configuration file might look like:

.. code-block:: yaml

   mean_line:
     type: axial_turbine  # Single-stage axial turbine
     Alpha1: 0.  # Inlet yaw angle [deg]
     phi: 0.8  # Flow coefficient
     psi: 1.6  # Stage loading coefficient
     Lam: 0.5  # Degree of reaction
     Ma2: 0.6  # Vane exit Mach number
     eta: 0.92  # Guess of polytropic efficiency
     loss_split: 0.4  # Fraction of entropy rise in stator
     htr: 0.9  # Hub-to-tip radius ratio
     Omega: 314.159  # Shaft angular velocity [rad/s]


   mean_line:
     type: radial_impeller  # Radial compressor with vaneless diffuser
     Alpha1: 0.0  # Inlet yaw angle [deg]
     Alpharel2: -50.0  # Rotor exit yaw angle [deg]
     DHdiff: 0.75  # Diffuser de Haller
     DHimp: 0.9  # Rotor de Haller
     Marel1: 0.6  # Inlet relative Mach number
     PR_tt: 2.0  # Total-to-total pressure ratio
     eta_tt: 0.91  # Total-to-total isentropic efficiency
     htr1: 0.5  # Inlet hub-to-tip radius ratio
     loss_split: 0.8  # Fraction of entropy rise in rotor
     mdot: 5.0  # Mass flow rate [kg/s]
     phi1: 0.6  # Inlet flow coefficient

.. _cnf-annulus:

annulus
-------

With mean radii and areas fixed by the mean-line design, meridional aspect
ratios set the length of the machine. For a turbomachine with `N` rows: there
are `N+1` aspect ratios defining the spacing between rows and the inlet and
exit boundaries set in `AR_gap`;  and `N` aspect ratios defining the blade
chords in `AR_chord`. :program:`turbigen` generates a smooth curvature
continuous annulus between these defining points.
For example, in a single-stage axial cases:

.. code-block:: yaml

   annulus:
     AR_gap: [0.6, 2., 0.7]
     AR_chord: [1.6, 1.6]

There are two special values that the meridional aspect ratios can take. For
fully radial flow with a pitch angle of :math:`\beta=\pm 90 ^\circ`, blade
chords are already fixed by the inlet and exit radii from mean-line design, so
must be set to `.nan` in the configuration file. To optimise axial length to
minimise integrated curvature over the hub and casing, set a negative aspect
ratio.

For a radial compressor, we want to optimise the rotor length to minimise
curvature, but the radial extent of the stator and vaneless space is already
determined by the mean-line design. This would be specified as:

.. code-block:: yaml

   annulus:
     AR_chord: [-1.0, .nan]
     AR_gap: [0.5,  .nan, 0.5]

.. _cnf-blades:

blades
------

Under the `blades` key is a list of data for each blade row, then a `section`
key specifies geometry for each section. The section data requires a span
fraction `spf` and vectors of thickness and camber parameters, `q_thick` and
`qstar_camber`:

.. code-block:: yaml

    # Span fraction for this section
    spf: 0.5
    # Vector of thickness parameters
    q_thick: [0.05, 0.12, 0.3, 0.02, 0.02, 0.18]
    # Vector of camber parameters
    qstar_camber: [0., 0., 1.0, 1.0, 0.0]

`q_thick` is a vector with 6 elements:
    * Leading-edge radius
    * Maximum thickness
    * Location of maximum thickness
    * Curvature at maximum thickness
    * Trailing edge thickness
    * Trailing edge wedge angle tangent

`qstar_camber` is a vector with 5 elements:
    * Leading-edge recamber with respect to flow angle
    * Trailing-edge recamber with respect to flow angle
    * Camber line slope at leading edge
    * Camber line slope at trailing edge
    * Camber line curvature at mid-chord

All lengths are normalised by the meridional chord.

The number of blades can be set via three choices: directly using `Nb`, a
non-dimensional circulation using `Co`, or the Lieblein diffusion factor `DFL`.
Thickness and camber vectors are interpolated between each of the specified
span fractions.

Tip gaps can be modelled by adding a `tip` key to the row configuration. The
gap is normalised by the mean of the inlet and exit span.

A complete specification of the blade geometry is:

.. code-block:: yaml

   blades:
     # First row
     - Nb: 65   # Set number of blades directly
       # Set blade geometry on two sections
       sections:
         - spf: 0.2  # Hub section
           q_thick: [0.05, 0.12, 0.3, 0.02, 0.02, 0.18]
           qstar_camber: [0., 0., 1.0, 1.0, 0.0]
         - spf: 0.8  # Tip section
           q_thick: [0.05, 0.12, 0.3, 0.02, 0.02, 0.18]
           qstar_camber: [0., 0., 1.0, 1.0, 0.0]
     # Second row
     - Co: 0.7  # Set number of blades using circulation coeff
       tip: 0.01  # 1% of span tip gap
       # Prismatic blade section
       sections:
         - spf: 0.5  # Midspan
           q_thick: [0.05, 0.12, 0.3, 0.02, 0.02, 0.18]
           qstar_camber: [0., 0., 1.0, 1.0, 0.0]


.. _cnf-mesh:

mesh
----

To generate a computational mesh, the topology must be specified in a `type`
key; currently `h` and `oh` are the only supported values. Following the type,
settings to control, for example, mesh resolution and clustering can optionally
be specified. The possible settings and their default values are listed in
:ref:`meshing`.

H-meshing requires less configuration and is more robust to consistently
automate across a design space. With some extra settings, this could look like:

.. code-block:: yaml

   mesh:
     type: h
     yplus: 60.0
     # Optional settings
     dm_LE: 0.0001
     nchord_relax: 0.5
     resolution_factor: 0.5
     # ...

OH-meshing, at present, requires SSH access to the proprietary AutoGrid software on
a remote machine, and is more complicated to set up and debug. More information
is given in :ref:`oh-meshing`.

.. code-block:: yaml

   mesh:
     type: oh
     yplus: 60.0
     remote_host: gp-111 # Hostname of AutoGrid machine
     # Optional settings
     nk_omesh: 33
     untwist_outlet: true
     # ...

.. _cnf-solver:

solver
------

The `solver` key can be omitted entirely to skip running CFD, and just
post-process the initial guess solution, which can facilitate debugging.
If present, the `solver` section requires a `type` key to indicate
which CFD code to use, and then solver-specific configurations settings may be
given, with default values used otherwise. :ref:`solvers` gives more detail.
Sample configurations for Turbostream 3 and 4 are:

.. code-block:: yaml

   # Turbostream 3
   solver:
     type: ts3
     nstep: 5000
     nstep_avg: 1000
     # Optional settings
     dampin: 10.
     # ...

   # Turbostream 4
   solver:
     type: ts4
     nstep: 3000
     nstep_avg: 500
     # Optional settings
     cfl_ramp_nstep: 2000
     # ...

.. _cnf-operating_point:

operating_point
---------------

To set the mass flow rate to the design value using a PID controller on exit static pressure, specify the PID constants under an `operating_point` key using:

.. code-block:: yaml

   operating_point:
     mdot_pid: [0.5, 0.1, 0.0]

The constants are non-dimensionalised with the nominal duty, so values of order
one are a good first guess. Smaller values are more robust at the expense of
longer convergence times.

To run the CFD off-design, keys `mass_adjust` and `rpm_adjust` can be added to
change the mass flow rate and shaft speed respectively, relative to the nominal
values. For example, to throttle to 90% of the design mass flow at 110% of the
design shaft speed:

.. code-block:: yaml

   operating_point:
     mdot_pid: [0.5, 0.1, 0.0]
     mass_adjust: -0.1  # 10% below design mass flow
     rpm_adjust: 0.1  # 10% above design rpm

.. _cnf-iterate:

iterate
-------

:program:`turbigen` has the capability to iterate on unknown values of mean-line loss,
deviation, and incidence until they match CFD results. Each of the three can be configured individually under the `iterate` key.

To match mean-line loss to CFD results, include a `mean_line` key.
Under `match_tolerance`, for each design variable that should be
adjusted, is an absolute tolerance on the difference between nominal mean-line
and mixed-out CFD values. Every iteration, the change in the design variables
is multiplied by a `relaxation_factor`, which may be reduced if the iterations
are unstable. If all variables are within tolerance, the iteration terminates.

The example below is for a radial compressor, where mean-line design requires
guesses of total-to-total efficiency, `eta_tt`, and rotor entropy rise
fraction, `loss_split`:

.. code-block:: yaml

   iterate:
     mean_line:
       match_tolerance:
         eta_tt: 0.005  # Efficiency to within 0.5%
         loss_split: 0.05  # Loss fraction to within 5%
       relaxation_factor: 0.5  # Change is half difference CFD-nominal

To match mean-line flow angle to the mixed-out CFD value, include a `deviation` key. The trailing-edge recamber of all rows is iterated simultaneously. An example configuration is:

.. code-block:: yaml

    iterate:
      deviation:
        clip: 5.0  # Maximum recamber in one step
        relaxation_factor: 0.8  # Multiplier on changes to metal angle
        tolerance: 0.5  # Absolute tolerance for termination in degrees


To locate the stagnation point on the nose of the blades, include an `incidence` key. This will recamber the leading edges of all blade sections defined for all rows simultaneously. The configuration is:

.. code-block:: yaml

    iterate:
      incidence:
        clip: 5.0   # Maximum recamber in one step
        relaxation_factor: 0.2  # Multiplier on changes to metal angle
        target: 0.0  # Desired stagnation point angle wrt camber line
        tolerance: 5.0  # Absolute tolerance for termination in degrees
        rtol_mdot: 0.05  # Recamber when mdot this close to nominal

In the event of instability, `clip` and `relaxation_factor` can be
reduced for all three adjustments.

A complete iteration configuration looks something like this:

.. code-block:: yaml

    iterate:
      deviation:
        clip: 5.0
        relaxation_factor: 2.0
        tolerance: 1.0
      incidence:
        clip: 2.0
        relaxation_factor: 0.2
        target: 0.0
        tolerance: 5.0
      mean_line:
        match_tolerance:
          eta_tt: 0.005
          loss_split: 0.05
        relaxation_factor: 0.5


.. _cnf-job:

job
---

Automations are in place for running :program:`turbigen` as a SLURM queue job
using the `sbatch` command. Some parameters are hardcoded assuming the
University of Cambridge Wilkes3 cluster and may require modifying for different
systems. Submitting jobs requires the following
configuration:


.. code-block:: yaml

    job:
      account: NAME-SL2-GPU   # SLURM account to charge
      hours: 4  # Integer number of hours for job time limit
      tasks: 2  # Number of GPUs
