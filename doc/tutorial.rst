Tutorial
=========

This tutorial will walk through the process of writing a new user-defined
mean-line solver and using it to design and simulate a compressor.
We first need to install :program:`turbigen`:

.. code-block:: console

   $ curl -LsSf https://astral.sh/uv/install.sh | sh
   $ source $HOME/.local/bin/env
   $ uv tool install turbigen

Problem statement
^^^^^^^^^^^^^^^^^

Suppose we wish to design a rotor-only axial fan. We shall assume
a constant axial velocity. The inlet state is specified
as fixed values of :math:`T_{01}` and :math:`p_{01}` with no inlet swirl,
:math:`\alpha_1=0`. We can then parametrise the aerodynamics of the stage using the
following design variables (many other choices are possible):

* Total pressure rise, :math:`\Delta p_0`
* Mass flow rate, :math:`\dot{m}`
* Flow coefficient, :math:`\phi=V_x/U`
* Loading coefficient, :math:`\psi= \Delta h_0/U^2`
* Hub-to-tip ratio, :math:`\mathit{HTR}=r_\mathrm{hub}/r_\mathrm{cas}`
* Total-to-total isentropic efficiency guess, :math:`\eta_\mathrm{tt}`


To proceed with the annulus and blade shape design, :program:`turbigen` requires the following quantities to be calculated from the above information. At the inlet and exit of the blade row:

* Mean radii, :math:`r_\mathrm{rms}`
* Annulus areas, :math:`A`
* Absolute frame velocity vectors, :math:`(V_x, V_r, V_\theta)`
* Static thermodynamic states, :math:`(P, T)` for example
* Rotor shaft speed, :math:`\Omega`

:math:`r_\mathrm{rms}`, :math:`A`, :math:`\Omega` and the thermodynamic states
should all be arrays of length 2. The velocity vectors should be of shape (3,2).

.. _tut-ml-algo:

Mean-line design equations
^^^^^^^^^^^^^^^^^^^^^^^^^^

We need to combine conservation of mass, momentum, and energy with
definitions of our design variables to solve for the flow in the turbomachine.
This will yield a set of equations that we will later implement numerically in
the `forward` function.


The specified total pressure rise and guess of total-to-total efficiency allow calculation of the compressor work :math:`\Delta h_0 = h_{02}-h_{01}`

.. math::
   :label: eqn-eta

   \eta_\mathrm{tt} = \frac{h_{02s}-h_{01}}{h_{02}-h_{01}} \quad\Rightarrow\quad \Delta h_0 = \frac{1}{\eta_\mathrm{tt}}\left[h(p_{01}+\Delta p_0, s_1) - h_{01}\right]

where :math:`h_{02s}=h(p_{01}+\Delta p_0, s_1)` is the ideal exit stagnation enthalpy for isentropic compression, i.e. enthalpy evaluated at the real exit stagnation pressure and inlet entropy.

The blade speed :math:`U` is then given from the definition of loading coefficient

.. math::
   :label: eqn-psi

   \psi = \frac{\Delta h_0}{U^2} \quad\Rightarrow\quad U = \sqrt{\frac{\Delta h_0}{\psi}}

The definition of flow coefficient yields the axial velocity :math:`V_x`

.. math::
   :label: eqn-phi

   \phi = \frac{V_x}{U} \quad\Rightarrow\quad V_x = \phi U

Assuming no inlet swirl, :math:`V_{\theta 1}=0` and the Euler work equation yields the rotor exit circumferential velocity

.. math::
   :label: eqn-Vt

   \Delta h_0 = U\left(V_{\theta 2}-V_{\theta 1}\right)\quad\Rightarrow\quad V_{\theta 2}=\frac{\Delta h_0}{U}

We now have stagnation thermodynamic states and velocity vectors at inlet and
exit of the rotor. If we knew the equation of state of the working fluid, we
could evaluate static thermodynamic states and quantities such as density.

Conservation of mass gives the annulus area

.. math::
   :label: eqn-A

   \dot{m} = \rho A V_x \quad\Rightarrow\quad A = \frac{\dot{m}}{\rho V_x}

and further specifying a hub-to-tip ratio fixes the mean radius

.. math::

   A = \pi\left(r_\mathrm{cas}^2 - r_\mathrm{hub}^2\right)\,,\
   r_\mathrm{rms} = \sqrt{\frac{1}{2}\left(r_\mathrm{cas}^2 + r_\mathrm{hub}^2\right)}
   \,,\ \mathit{HTR}=\frac{r_\mathrm{hub}}{r_\mathrm{cas}}

.. math::
   :label: eqn-rrms

   \Rightarrow r_\mathrm{rms} = \sqrt{\frac{A}{2\pi}\frac{1+\mathit{HTR}^2}{1-\mathit{HTR}^2}}

Finally, the shaft angular velocity is simply

.. math::
   :label: eqn-Omega

   \Omega = U/r_\mathrm{rms}

Setting up skeleton files
^^^^^^^^^^^^^^^^^^^^^^^^^

To integrate our new mean-line design into :program:`turbigen`, we need to
write a subclass with two static methods: a `forward` function which takes our
design variables as inputs and returns a flow field along the mean line; and a
`backward` function that recalculates the design variables from an input flow
field. The :ref:`ml-custom` section describes the general process in more detail.

Start by creating a new directory to store our custom plugins, for example

.. code-block:: console

   $ mkdir ./plugins

and create a new file called `fan.py` inside this directory. We will code up the
equations from the previous section in this file. The structure looks like this:

.. code-block:: python
   :caption: ./plugins/fan.py

   import turbigen.meanline

   class Fan(turbigen.meanline.MeanLineDesigner):

      @staticmethod
      def forward(So1, DPo, mdot, phi, psi, htr, etatt):
         '''Use design variables to calculate flow field.'''

         raise NotImplementedError("Implement the forward method")

         return rrms, A, Omega, Vxrt, S

      @staticmethod
      def backward(mean_line):
         '''Calculate design variables from flow field.'''

         raise NotImplementedError("Implement the backward method")

         # Dictionary of design variables
         return {}


We also need a minimal configuration file to test our mean-line functions.
Create a new `config.yaml` with the following content:

.. code-block:: yaml
   :caption: ./config.yaml

   workdir: runs/fan  # Store output files here
   plugdir: ./plugins  # Directory containing our custom mean line

   # Perfect gas inlet state
   inlet:
       Po: 1e5
       To: 300.
       cp: 1005.
       mu: 1.8e-5
       gamma: 1.4

   # Mean-line design
   mean_line:
       type: fan  # Path to the mean-line module we are writing
       # Our chosen design variables (args to forward)
       DPo: 2000.
       mdot: 5.
       phi: 0.5
       psi: 0.4
       htr: 0.8
       etatt: 0.9

The file structure should now look like this:

.. code-block:: console

   $ tree
   .
   ├── config.yaml
   └── plugins
       └── fan.py

At this point, running the config.yaml file through :program:`turbigen`
generates a `NotImplementedError` because the body of the `forward` function is
missing.

.. code-block:: console

   $ turbigen config.yaml
   *** TURBIGEN v2.3.0 ***
   Starting at 2025-05-29T12:21:26
   Working directory: /home/jb753/python/turbigen-dev/runs/fan
   Importing plugins from /home/jb753/python/turbigen-dev/plugins
   Loaded plugin: /home/jb753/python/turbigen-dev/plugins/fan.py
   Inlet: PerfectState(P=1.000 bar, T=300.0 K)
   Error encountered, quitting...
   Traceback (most recent call last):
   ...
   NotImplementedError: Implement the forward method


Implementing forward
^^^^^^^^^^^^^^^^^^^^

We can now start to add the :ref:`tut-ml-algo` to the `forward` function inside
`fan.py`.

The first task is to calculate the ideal exit enthalpy :math:`h_{02s}=h(p_{01}+\Delta p_0, s_1)`
in Eqn. :eq:`eqn-eta`. Mean-line design functions should be written to make
no assumptions about the working fluid equation of state --- this is accomplished
using the fluid modelling abstractions of the :ref:`state` class. We take a
copy of the inlet state, and set its pressure and entropy to the required
values.

.. code-block:: python

   @staticmethod
   def forward(So1, DPo, mdot, phi, psi, htr, etatt):
       """Calculate mean-line from inlet and design variables."""

       # Get the ideal exit state
       So2s = So1.copy()  # Duplicate the inlet state
       So2s.set_P_s(So1.P + DPo, So1.s)  # Set pressure and entropy

       # ...

We can now calculate the compressor work by reading off
enthalpy values from our two state objects `So1` and `So2s`.

.. code-block:: python

       # Work from defn efficiency Eqn. (1)
       Dho = (So2s.h-So1.h)/etatt

Proceeding straightforwardly to calculate blade speed and velocity vectors

.. code-block:: python

       # Blade speed from defn psi Eqn. (2)
       U = np.sqrt(Dho/psi)

       # Axial velocity from defn phi Eqn. (3)
       Vx = phi*U

       # Circumferential velocity from Euler Eqn. (4)
       Vt2 = Dho/U

       # Assemble velocity vectors
       # shape (3 directions, 2 stations)
       Vxrt = np.stack(
           (
               (Vx, Vx),  # Constant axial velocity
               (0., 0.),  # No radial velocity
               (0., Vt2),  # Zero inlet swirl
           )
       )

Next, we need to calculate the static thermodynamic states. As we know
stagnation states and velocity vectors everywhere, this is most straightforward
to do by evaluating the static enthalpy :math:`h=h_0-\frac{1}{2}V^2`. The
static and stagnation states have the same entropy. In code, this looks like:

.. code-block:: python

       # Outlet stagnation state from known total rises
       So2 = So1.copy().set_P_h(So1.P + DPo, So1.h + Dho)

       # Assemble both stagnation states into a vector state
       So = So1.stack((So1,So2))

       # Get static states using velocity magnitude and same entropy
       Vmag = np.sqrt(np.sum(Vxrt**2,axis=0))
       h = So.h - 0.5*Vmag**2  # Static enthalpy
       S = So.copy().set_h_s(h , So.s)

Now that the static states are known, the density can be used in the
conservation of mass equation to continue with evaluating areas, the RMS
radius, and the shaft angular velocity. The completed function is:

.. code-block:: python
   :caption: ./plugins/fan.py

   def forward(So1, DPo, mdot, phi, psi, htr, etatt):
       """Calculate mean-line from inlet and design variables."""

       # Get the ideal exit state
       So2s = So1.copy()  # Duplicate the inlet state
       So2s.set_P_s(So1.P + DPo, So1.s)  # Set pressure and entropy

       # Work from defn efficiency Eqn. (1)
       Dho = (So2s.h-So1.h)/etatt

       # Blade speed from defn psi Eqn. (2)
       U = np.sqrt(Dho/psi)

       # Axial velocity from defn phi Eqn. (3)
       Vx = phi*U

       # Circumferential velocity from Euler Eqn. (4)
       Vt2 = Dho/U

       # Assemble velocity vectors
       # shape (3 directions, 2 stations)
       Vxrt = np.stack(
           (
               (Vx, Vx),  # Constant axial velocity
               (0., 0.),  # No radial velocity
               (0., Vt2),  # Zero inlet swirl
           )
       )

       # Outlet stagnation state from known total rises
       So2 = So1.copy().set_P_h(So1.P + DPo, So1.h + Dho)

       # Assemble both stagnation states into a vector state
       So = So1.stack((So1,So2))

       # Get static states using velocity magnitude and same entropy
       Vmag = np.sqrt(np.sum(Vxrt**2,axis=0))
       h = So.h - 0.5*Vmag**2  # Static enthalpy
       S = So.copy().set_h_s(h , So.s)

       # Conservation of mass for annulus area, Eqn. (5)
       A = mdot/S.rho/Vx

       # Mean radius from HTR Eqn. (6)
       rrms = np.sqrt(A[0] / np.pi / 2.0 * (1.0 + htr**2) / (1.0 - htr**2))
       rrms = np.ones((2,)) * rrms  # Make rrms constant across stations

       # Shaft angular velocity
       Omega = U / rrms

       # Return mean-line data
       return rrms, A, Omega, Vxrt, S


This concludes the `forward` function --- all the required quantities have been
evaluated and can be returned for further processing.
If we run :program:`turbigen` on the `config.yaml` file now, it will complete
mean-line design successfully using the `forward` function, but raise an
Exception because the `backward` function is incomplete:

.. code-block:: console

   *** TURBIGEN v2.3.0 ***
   Starting at 2025-05-29T12:43:46
   Working directory: /home/jb753/python/turbigen-dev/tut2/runs/fan
   Importing plugins from /home/jb753/python/turbigen-dev/tut2/plugins
   Loaded plugin: /home/jb753/python/turbigen-dev/tut2/plugins/fan.py
   Inlet: PerfectState(P=1.000 bar, T=300.0 K)
   MeanLine(
      Po=[1.   1.02] bar,
      To=[300.     301.8913] K,
      Ma=[0.099 0.127],
      Vx=[34.5 34.5] m/s,
      Vr=[0. 0.] m/s,
      Vt=[ 0.  27.6] m/s,
      Vt_rel=[-68.9 -41.4] m/s,
      Al=[ 0.   38.66] deg,
      Al_rel=[-63.43 -50.19] deg,
      rpm=[2182. 2193.],
      mdot=[5. 5.] kg/s
      )
   Error encountered, quitting...
   Traceback (most recent call last):
   ...
   NotImplementedError: Implement the backward method

Implementing backward
^^^^^^^^^^^^^^^^^^^^^

The `backward` function serves as a verification check that the mean-line
matches the design intent, and also to extract design variables from a
mixed-out CFD solution. We add the design variables as keys in the output
dictionary, using the attributes of the :ref:`meanline` class to calculate them. Many
useful quantities are already available in the :ref:`meanline` object, such as efficiency.

.. code-block:: python
   :caption: ./plugins/fan.py

      @staticmethod
      def backward(mean_line):
         '''Calculate design variables from flow field.'''

         return {
               "DPo": mean_line.Po[-1] - mean_line.Po[0],
               "mdot": mean_line.mdot[0],
               "phi": mean_line.Vx[0] / mean_line.U[0],
               "psi": (mean_line.ho[-1] - mean_line.ho[0]) / (mean_line.U[0]) ** 2,
               "etatt": mean_line.eta_tt,
               "htr": mean_line.rhub[0] / mean_line.rtip[0],
         }

Running :program:`turbigen` on the `config.yaml` file now will complete the
mean-line design using `forward`, check it using `backward` and then halt
because further information is needed to proceed with the design.

Running CFD
^^^^^^^^^^^

To create blade shapes and run a
computational fluid dynamics simulation, we add extra options to the
`config.yaml`:

.. code-block:: yaml
   :caption: ./config.yaml

   workdir: runs/fan  # Store output files here
   plugdir: ./plugins  # Directory containing our custom mean line

   # Perfect gas inlet state
   inlet:
      Po: 1e5
      To: 300.
      cp: 1005.
      mu: 1.8e-5
      gamma: 1.4

   # Mean-line design
   mean_line:
      type: fan  # Path to the mean-line module we are writing
      # Our chosen design variables (args to forward)
      DPo: 2000.
      mdot: 5.
      phi: 0.5
      psi: 0.4
      htr: 0.8
      etatt: 0.9

   #
   # ADD THE BELOW
   #

   # Annulus configuration
   annulus:
      AR_gap: [1.0, 1.0]  # Span to inlet/exit boundary distance
      AR_chord: 3.  # Span to chord

   # Blade shapes
   blades:
      - spf: 0.5  # Define one section at midspan
        thick: [0.02, 0.05, 0.3, 0.2, 0.0, 0.1]
        camber: [0., 4., 0.0]

   # Lieblein to set number of blades
   nblade:
      - DFL: 0.45

   # Mesh generation
   mesh:
      type: h  # Mesh topology
      yplus: 30.0  # Non-dimensional wall distance
      resolution_factor: 0.5  # Use a coarse mesh

   # Built-in Enhanced Multiblock solvER
   solver:
      type: ember
      n_step: 8000
      n_step_avg: 2000

   # Control mass flow using a PID on exit pressure
   operating_point:
      throttle: true

If we now run :program:`turbigen` on our `config.yaml` using the shell command,
we can quickly obtain a CFD solution for our newly designed fan.

.. code-block:: console

   $ turbigen config.yaml
   *** TURBIGEN v2.3.0 ***
   Starting at 2025-05-29T16:26:06
   Working directory: /home/jb753/python/turbigen-dev/tut2/runs/fan
   Importing plugins from /home/jb753/python/turbigen-dev/tut2/plugins
   Loaded plugin: /home/jb753/python/turbigen-dev/tut2/plugins/fan.py
   Inlet: PerfectState(P=1.000 bar, T=300.0 K)
   MeanLine(
      Po=[1.   1.02] bar,
      To=[300.     301.8913] K,
      Ma=[0.099 0.127],
      Vx=[34.5 34.5] m/s,
      Vr=[0. 0.] m/s,
      Vt=[ 0.  27.6] m/s,
      Vt_rel=[-68.9 -41.4] m/s,
      Al=[ 0.   38.66] deg,
      Al_rel=[-63.43 -50.19] deg,
      rpm=[2182. 2182.],
      mdot=[5. 5.] kg/s
      )
   Designing annulus...
   FixedAR(nrow=1, x=[0.01105477], r=[0.29992128], AR=[2.99850036])
   Designing blades...
   Nblade: [55]
   Tip gaps: [0.]
   Re_surf=[1.99e+05]
   Generating mesh...
   Making an H-mesh...
   ncell/1e6=0.1
   Applying 2D guess...
   Setting operating point...
   Exit PID constants=(1.0, 0.5, 0.0)
   Initialising ember...
   Patitioning onto 1 processors...
   Starting the main time-stepping loop...
   500: tpnps=3.174e-07, remaining=4m15s
   block 0: 3.60e-05 6.73e-03 3.62e-03 1.42e-03 3.14e+00
   ...
   7999: tpnps=3.189e-07, remaining=0m0s
   block 0: 2.28e-07 1.40e-04 2.92e-05 1.22e-05 1.81e-02
   Elapsed time 4.56 min
   Average tpnps=3.176e-07
   mdot_in/out=5/5, err=0.0%
   Post-processing...
   Variable  Nominal    Actual    Err_abs  Err_rel/%
   -------------------------------------------------
        DPo    2e+03  1.63e+03        367       18.4
      etatt      0.9      0.92    -0.0201      -2.24
        htr      0.8       0.8  -0.000129    -0.0162
       mdot        5         5   -0.00436    -0.0873
        phi      0.5       0.5  -0.000259    -0.0519
        psi      0.4      0.32     0.0801         20
   Efficiency/%: eta_tt=92.0, eta_ts=35.7

Creating and running designs with different velocity triangles is as simple as
changing a line or two in the mean-line section of `config.yaml`. This allows
us to explore a new design space very quickly.

Iterating the design
^^^^^^^^^^^^^^^^^^^^

The table at the end of the program output compares the nominal mean-line
design variables to actual values calculated using cuts from the
three-dimensional CFD solution (the cuts are mixed out at constant area).
Inspecting the output for our new fan, we can identify several problems:

.. code-block:: console

   Variable  Nominal    Actual    Err_rel/%
   ----------------------------------------
        DPo    2e+03  1.63e+03         18.4  # Pressure rise too low
      etatt      0.9      0.92        -2.24  # Loss guess too low
        htr      0.8       0.8      -0.0162
       mdot        5         5      -0.0873
        phi      0.5       0.5      -0.0519
        psi      0.4      0.32           20  # Not enough loading

The root cause of the lack of pressure rise is that we have not allowed for
deviation in designing the blade shapes, hence the flow is underturned.
Assuming a guess of efficiency was necessary to complete mean-line design, but
its value should be updated so that the annulus areas are compatible with the
intended velocity triangles.

Although it is not evident from the table, the inlet flow is not precisely
aligned with the inlet metal angle, leading to unwanted accelerations around
the leading edge. We should locate the stagnation point on the nose of the
aerofoil to yield the smoothest pressure distributions.

:program:`turbigen` has the capability to correct for all these issues. Adding
an `iterate` key to the `config.yaml` will cause the program to repeatedly run
the CFD, updating the efficiency guess and recambering the leading and trailing
edges as needed. We can also cut the number of time steps, as each CFD
simulation is restarted from the previous flow field, so convergence can take
place over multiple iterations in parallel with geometry adjustment.


.. code-block:: yaml
   :caption: ./config.yaml

   # ...

   # Reduce number of time steps to speed up convergence
   solver:
     type: ember
     n_step: 4000
     n_step_avg: 2000

   # Add new section for iterative corrections
   iterate:
     # Make the mean-line loss match CFD within 0.5% effy
     mean_line:
       tolerance:
         etatt: 0.005
     # Correct for deviation using trailing-edge recamber
     deviation:
     # Correct for incidence using leading-edge recamber
     incidence:

Running the extended input file gives:

.. code-block:: console

   $ turbigen config.yaml
   *** TURBIGEN v2.3.0 ***
   Starting at 2025-05-29T21:02:49
   Working directory: /home/jb753/python/turbigen-dev/tut2/runs/fan
   Importing plugins from /home/jb753/python/turbigen-dev/tut2/plugins
   Loaded plugin: /home/jb753/python/turbigen-dev/tut2/plugins/fan.py
   Iterating for max 20 iterations...
   Min Dev[0] DDev[0] Inc[0] DInc[0] etatt Detatt
   ------------------------------------------------
   2.45  -2.95       2    112       2 0.926  0.013
   2.56  -1.08    1.08   7.07   0.354 0.932  0.009
   2.55  -0.34   0.343   4.23   0.212 0.933  0.005
   2.55  -0.07   0.076   32.9    1.64 0.933  0.002
   2.55  0.499   -0.49   18.1   0.905 0.933  0.001
   Finished iterating, converged=True.
   Variable  Nominal  Actual    Err_abs  Err_rel/%
   -----------------------------------------------
        DPo    2e+03   2e+03      -1.74    -0.0868
      etatt    0.932   0.933   -0.00125     -0.134
        htr      0.8     0.8  -0.000128     -0.016
       mdot        5       5   -0.00479    -0.0958
        phi      0.5     0.5  -0.000301    -0.0602
        psi      0.4   0.399   0.000598      0.149
   Efficiency/%: eta_tt=93.3, eta_ts=40.9

The corrections applied, `DInc`, `DDev`, and `Detatt`, decrease with each
iteration indicating stable convergence. When the iteration terminates, the
mixed-out CFD solution corresponds closely to the design intent. A new
configuration file has been written out in the working directory
`runs/fan/config.yaml` for the converged solution. Inspecting this file:

.. code-block:: yaml
   :caption: ./runs/fan/config.yaml

   # ...
   - camber:
      - - 5.113164958868248
         - 6.99784007331111
         - 0.0
   # ...


Under the `camber` key that defines the camber line, we see that 5.1
degrees of leading-edge recamber was required to align the stagnation point,
and the deviation was 7 degrees. The efficiency has also been updated to 93.3%.

Extensions
^^^^^^^^^^

This tutorial has demonstrated some of the functionality of
:program:`turbigen`. Within the current choice of parameterisation, any change to
the design is just an edit to the `config.yaml`.

* Increase the number of blades by changing `DFL`
* Increase the grid density under `mesh`
* Control camber and thickness distributions by changing `thick` and `camber`
* Specify blade sections at multiple spanwise locations
* Change the aspect ratio `AR_chord`
* With a compatible CFD solver, change the working fluid to a real gas under `inlet`

To change the mean-line design, edit the `forward` and `backward` functions in
`fan.py`. For example: relax the assumption of constant axial velocity by
adding a velocity ratio as one of the arguments to forward, replace
specification of loading coefficient with a de Haller number, or specify an
inlet Mach number instead of mass flow rate.

To add a stator, extend `forward` to take additional design variables and
perform the necessary calculations. The output data should be at the inlet and
exit of both blade rows, e.g. `A` an array of length 4, the velocity vectors
should be of shape (3,4).
