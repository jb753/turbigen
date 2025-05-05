Mean-line
=========

The first step in turbomachinery design is a one-dimensional analysis along
a representative 'mean-line', a simplified model of the true
three-dimensional flow. We consider an axisymmetric streamsurface at an
intermediate position between hub and casing, with stations at the
inlet and exit of each blade row.

The inputs to the mean-line design are the inlet condition and the machine duty.
Specifying some aerodynamic design variables and applying conservation of
mass, momentum and energy, we can calculate the outputs of
annulus areas, mean radii and flow angles at each station.

The mean-line design process is different for each machine architecture:
compressor/turbine, axial/radial, and so on. :program:`turbigen` provides
the built-in architectures listed below, and also allows considerable flexibility
in defining your own :ref:`ml-custom`.



Turbine cascade
---------------

A single-row, stationary turbine cascade. The geometry is defined by spans,
flow angles and an exit Mach number. By default, the cascade is approximately linear,
with a hub-to-tip ratio close to unity, no radius change, and no pitch
angle. An annular cascade can be defined by specifying a radius ratio and
pitch angles.

To use this architecture, add the following snippet in your configuration file:

.. code-block:: yaml

    mean_line:
      type: turbine_cascade
      # Inlet and outlet spans [m], (2,) vector
      span:
      # Inlet and outlet yaw angles [deg], (2,) vector
      Alpha:
      # Exit Mach number [--]
      Ma2:
      # Energy loss coefficient [--]
      Yh:
      # Inlet hub-to-tip radius ratio [--]
      htr: 0.99
      # Outlet to inlet radius ratio [--]
      RR: 1.0
      # Inlet and outlet pitch angles [deg], (2,) vector
      Beta: (0.0, 0.0)

Axial turbine
-------------

A repeating-stage axial turbine. Duty is set by a mass flow rate and a constant mean radius.
Vane exit Mach number is set directly, while the rotor exit relative Mach
number is set by a scaling factor off the vane value. This allows compressibility effects to be predominantly controlled by `Ma2`; the degree of reaction is controlled by `fac_Ma3_rel`, with 50% reaction corresponding approximately to unity.
The default is constant axial velocity, but this can be controlled by `zeta`.
Pressure ratio and shaft speed are dependent variables under this parameterisation.

To use this architecture, add the following snippet in your configuration file:

.. code-block:: yaml

    mean_line:
      type: axial_turbine
      # Mass flow rate [kg/s]
      mdot:
      # Root-mean-square radius [m]
      rrms:
      # Stage loading coefficient [--]
      psi:
      # Rotor inlet flow coefficient [--]
      phi2:
      # Vane exit Mach number [--]
      Ma2:
      # Rotor exit relative Mach factor [--]
      fac_Ma3_rel:
      # Entropy loss coefficients [--], (2,) vector
      Ys:
      # Axial velocity ratios [--], (2,) vector
      zeta: (1.0, 1.0)

.. _ml-custom:

Custom architectures
--------------------

Custom architectures are defined by subclassing the :class:`MeanLineDesigner`.
First, set the user plugin directory in the configuration file, e.g. to a new folder called `plug` in the current directory by adding the following line:

.. code-block:: yaml

    plugdir: ./plug

Then, create a new Python file in the `plug` directory, e.g.
`custom.py`, and define a new class that inherits from
:class:`MeanLineDesigner` like this:

.. code-block:: python

    # File: ./plug/custom.py

    import turbigen.meanline

    class MyCustomMeanLine(turbigen.meanline.MeanLineDesigner):

        # Your design variables are arguments to the forward method
        @staticmethod
        def forward(So1, phi, psi, Ma1):
            '''Use design variables to calculate flow field.

            Parameters
            ----------
            So1: Fluid
                The working fluid and its thermodynamic state at inlet.
            phi: float
                Flow coefficient at inlet.
            psi: float
                Stage loading coefficient.
            Ma1: float
                Inlet Mach number.
            ... your chosen design variables ...

            Returns
            -------
            rrms: (2*nrow,) array
                Mean radii at all stations [m].
            A: (2*nrow,) array
                Annulus areas at all stations [m^2].
            Omega: (2*nrow,) array
                Shaft angular velocities at all stations [rad/s].
            Vxrt: (3, 2*nrow) array
                Velocity components at all stations [m/s].
            S: (2*nrow,) list of Fluid
                Static states for all stations.

            '''

            # Your code here...
            raise NotImplementedError("Implement the forward method")

            # Manipulate thermodynamic states by copying the inlet
            # add setting new property values, say
            V1 = Ma1 * So1.a  # Approx, should iterate this
            h1 = So1.h - 0.5 * V1**2
            S1 = So1.copy().set_h_s(h1 So1.s)

            # Collect the static states
            S = [S1, S2]

            return rrms, A, Omega, Vxrt, S

        @staticmethod
        def backward(mean_line):
            '''Calculate design variables from flow field.

            Parameters
            ----------
            mean_line: MeanLine
                Flow field along the mean line.

            Returns
            -------
            out : dict
                Dictionary of design variables, keyed by arguments
                to the `forward` method.

            '''

            # The mean_line object has all the flow field data
            # and calculates most composite quantities like
            # velocity components, stagnation enthalpy, for you

            # Blade speed at station 1 (first row inlet)
            U = mean_line.U[0]

            return {
                # Inlet flow coefficient
                'phi': mean_line.Vm[0] / U,
                # Stage loading coefficient
                'psi': (mean_line.ho[-1] - mean_line.ho[0]) / U**2,
                # Mach number at inlet
                'Ma1': mean_line.Ma[0],
                # Your design variables here...
                # ...
                # Other keys are printed to the log file and saved to the
                # output configuration file
                'eta_tt': mean_line.eta_tt,
                'Alpha1': mean_line.Alpha[0],
                'DH': mean_line.V_rel[1] / mean_line.V_rel[0],
            }

You will need to implement two static methods: `forward()` and `backward()`.

The `forward()` function takes as arguments an inlet state object and
some duty, geometric, or aerodynamic design variables. This function
returns all information required to lay out the mean-line: radii, annulus
areas, angular velocities, velocity components, and thermodynamic states at
the inlet and exit of each blade row.

The entries in the `mean_line` part of the configuration file are fed into
`forward()` as keyword arguments. For example, if the configuration file
contains:

.. code-block:: yaml

    mean_line:
        type: my_custom_mean_line
        phi: 0.8
        psi: 1.6
        Ma1: 0.5

Then, within :program:`turbigen`, the `type` key identifies
the `MyCustomMeanLine` class and calls its `forward()` method like:

.. code-block:: python

    MyCustomMeanLine.forward(
        So1,  # Inlet state calculated elsewhere, positional arg
        phi=0.8, # Keys from `mean_line` config unpacked as kwargs
        psi=1.6,
        Ma1=0.5,
    )

Some notes on implementing the `forward()` method:

* Retain generality of the working fluid by using the set property methods
  of the `Fluid` class, as in the example above. This is preferable to
  hard-coding calculations assuming a specific equation of state such as
  ideal gas.
* Specify aerodynamic design variables instead of geometric ones, e.g.
  flow coefficient and Mach number instead of shaft
  angular velocity and radius ratio. Constraining geometry can lead to
  feasible designs only over a narrow range of duty, with many infeasible designs with no solution to the mean-line equations. It is more
  straighforward to map out a design space by varying independent variables
  within their natural aerodynamic bounds.
* Controlling Mach number prevents choking when moving around the
  design space. It is imperative to limit deceleration through compressors
  to avoid flow separation, so specifying a relative velocity ratio or de
  Haller number is a good idea. Letting meridional velocity float can lead
  to wide variations in span and hence unfavourable high-curvature annulus
  lines, so controlling the change in meridional velocity through the
  machine is advisable.
* Loss is best handled by guessing a vector of entropy rise at each
  station, which does not depend on the frame of reference. The values can then be
  updated using CFD results.
* Iteration is required to solve for density in compressible cases. It is
  often easiest to guess a value for the blade speed, then iterate to converge
  on a blade speed that satisfies the requested duty. Matching a
  total-to-static pressure ratio requires iteration because the exit
  dynamic head is not known a priori.

The `backward()` function takes a mean-line flow field as its only
argument, and calculates a dictionary of the arguments to `forward()`.
Given a suitably averaged CFD solution, `backward()` is a post-processing
step that allows comparison of the three-dimensional simulated flow field
to the one-dimensional design intent. Also, feeding the output of
`forward()` straight into `backward()` acts as a check that the mean-line
design is consistent with the requested inputs. `backward()` can also be
used to post-process other quantities of interest from the mixed-out CFD
flow field. Extra keys are printed the log file and saved to the output configuration
file; only keys that are also design variables will be fed back
into `forward()` for the consistency check.
