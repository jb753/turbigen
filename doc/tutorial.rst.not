Tutorial
========

This tutorial will walk through the process of writing a new user-defined
mean-line solver and using it to run CFD on the Cambridge Wilkes3 cluster.

Problem statement
^^^^^^^^^^^^^^^^^

Suppose we wish to design a rotor-only axial fan. We shall assume
a constant axial velocity. The inlet state is specified
as fixed values of :math:`T_{01}` and :math:`p_{01}` with no inlet swirl,
:math:`\alpha_1=0`. We could then parametrise the aerodynamics of the stage using the
following design variables (but other choices are possible):

* Total pressure rise, :math:`\Delta p_0` [Pa]
* Mass flow rate, :math:`\dot{m}` [kg/s]
* Flow coefficient, :math:`\phi=V_x/U`
* Loading coefficient, :math:`\psi= \Delta h_0/U^2`
* Hub-to-tip ratio, :math:`\mathit{HTR}=r_\mathrm{hub}/r_\mathrm{cas}`
* Total-to-total isentropic efficiency guess, :math:`\eta_\mathrm{tt}`


To proceed with the annulus and blade shape design, :program:`turbigen` requires the following quantities to be calculated from the above information. At the inlet and exit of the blade row:

* Root-mean-square radii, :math:`r_\mathrm{rms}` [m]
* Annulus areas, :math:`A` [m^2]
* Absolute frame velocity vectors, :math:`(V_x, V_r, V_\theta)` [m/s]
* Static thermodynamic states, :math:`(P, T)` for example [Pa, K]
* Rotor shaft speed, :math:`\Omega` [rad/s]

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

We need a minimal configuration file to test our mean-line functions.
Create a `config.yaml` with the following content:

.. code-block:: yaml
   :caption: config.yaml

   # All files relating to the case are held in a working directory
   workdir: tutorial
   plugdir: tutorial

   # Perfect gas inlet state
   inlet:
      Po: 1e5
      To: 300.
      cp: 1005.
      mu: 1.8e-5
      gamma: 1.4

   # Mean-line design
   mean_line:
      type: fan  # Name of the mean-line we are writing
      # Our chosen design variables (args to forward)
      DPo: 2000.
      mdot: 5.
      phi: 0.5
      psi: 0.4
      etatt: 0.9
      htr: 0.8

   # Hub and casing lines
   annulus:
      AR_gap: [1.0, 1.0]  # Span to inlet/exit boundary distance
      AR_chord: 3.  # Span to chord
      nozzle_ratio: 0.9

   # Aerofoil sections
   blades:
      - spf: 0.5
         q_thick: [0.05, 0.12, 0.3, 0.02, 0.02, 0.18]
         q_camber: [0., 0., 1.0]

   # Mesh generation
   mesh:
      type: h
      yplus: 30.0
      resolution_factor: 0.5

   # Set number of blades using Lieblein
   nblade:
      - DFL: 0.45

   # CFD solver
   solver:
      nstep: 5000
      nstep_avg: 1000

Most of the file is not relevant to the mean-line design, but we will
eventually need to specify all this data to run a CFD simulation.

If we attempt to run this file through :program:`turbigen`, we will get an error:

.. code-block:: console

   $ turbigen config.yaml
   ...
   ValueError: No MeanLineDesigner named fan.
   Available subclasses are:
   turbine_cascade
   axial_turbine

This is because we have not yet implemented the mean-line design function.
Create a new file under the `tutorial` directory called `fan.py` and add the
following code:

.. code-block:: python
   :caption: tutorial/fan.py

   from turbigen.meanline import MeanLineDesigner
   import numpy as np

   class Fan(MeanLineDesigner):
      """Mean-line designer for an axial fan rotor."""

      @staticmethod
      def forward(So1, DPo, mdot, phi, psi, htr, etatt):
         """Calculate mean-line from inlet and design variables.

         Parameters
         ----------
         So1: float
               Stagnation state at inlet.
         DPo: float
               Stagnation pressure rise across the fan [Pa].
         mdot: float
               Mass flow rate through the fan [kg/s].
         phi: float
               Flow coefficient.
         psi: float
               Loading coefficient.
         htr: float
               Hub-to-tip ratio.
         etatt: float
               Total-to-total isentropic efficiency.

         Returns
         -------
         rrms: (2,) array
               Mean radii of the fan at inlet and exit [m].
         A: (2,) array
               Annulus areas at inlet and exit [m^2].
         Omega: float
               Shaft angular velocity [rad/s].
         Vxrt: (3, 2) array
               Velocity vectors at inlet and exit [m/s].
         S: (2) array of thermodynamic states
               Static states at inlet and exit.

         """

         # Insert code to calculate rrms, A, Omega, Vxrt, states
         # ...
         raise NotImplementedError

         # Return assembled mean-line object
         return (
               rrms,  # Mean radii
               A,  # Annulus areas
               Omega,  # Shaft angular velocity
               Vxrt,  # Velocity vectors
               S,  # Thermodynamic states
         )

      @staticmethod
      def backward(mean_line):
         """Reverse a cascade mean-line to design variables.

         Parameters
         ----------
         mean_line: MeanLine
               A mean-line object specifying the flow in a cascade.

         Returns
         -------
         out : dict
               Dictionary of aerodynamic design variables.
               The fields have the same meanings as in :func:`forward`.

         """

         # The output should be a dictionary keyed by the args to forward
         return {
               # 'DPo': ...,
               # 'mdot': ...,
               # 'phi': ...,
               # 'psi': ...,
               # 'htr': ...,
               # 'etatt': ...,
         }

To integrate our new mean-line design into :program:`turbigen`, we have two
static methods to write on a new subclass: a `forward` function which takes our design variables as
inputs and returns the geometry, velocity vectors, and thermodynamic states; and a
`backward` function that recalculates the design variables from an input
:py:class:`turbigen.meanline.MeanLine` object. Now that we know what input and
output data are required, and have the equations for our algorithm, we can
start writing the functions.

`So1` is a fluid object that encapsualtes the inlet stagnation thermodynamic state. All
thermodynamic properties can be accessed as attributes, and there are functions
to manipulate the state to new values, described fully in :py:mod:`turbigen.fluid` .

At this point, running the config.yaml file through :program:`turbigen`
generates a `NotImplementedError` because the body of the `forward` function is
missing.



Implementing the algorithm
^^^^^^^^^^^^^^^^^^^^^^^^^^

We can now start to add the :ref:`tut-ml-algo` to the `forward` function inside
`fan.py`.

The first task is to calculate the ideal exit enthalpy :math:`h_{02s}=h(p_{01}+\Delta p_0, s_1)`
in Eqn. :eq:`eqn-eta`. Mean-line design functions should be written to make
no assumptions about the working fluid equation of state --- this is accomplished
using the fluid modelling abstractions in :py:mod:`turbigen.fluid`. We take a
copy of the inlet state, and set its pressure and entropy to the required
values.

.. code-block:: python
   :caption: fan.py

   def forward(So1, DPo, mdot, phi, psi, htr, etatt):
       """Caluclate mean-line from inlet and design variables."""

       # Get the ideal exit state
       So2s = So1.copy()  # Duplicate the inlet state
       So2s.set_P_s(So1.P + DPo, So1.s)  # Set pressure and entropy

       # ...

We can now calculate the compressor work by reading off
enthalpy values from our two state objects `So1` and `So2s`.

.. code-block:: python
   :caption: fan.py

       # ...

       # Work from defn efficiency Eqn. (1)
       Dho = (So2s.h-So1.h)/etatt

       # ...

Proceeding straightforwardly to calculate blade speed and velocity vectors

.. code-block:: python
   :caption: fan.py

       # ...

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

       # ...

Next, we need to calculate the static thermodynamic states. As we know
stagnation states and velocity vectors everywhere, this is most straightforward
to do by evaluating the static enthalpy :math:`h=h_0-\frac{1}{2}V^2`. The
static and stagnation states have the same entropy. In code, this looks like:

.. code-block:: python
   :caption: fan.py

       # ...

       # Outlet stagnation state from known total rises
       So2 = So1.copy().set_P_h(So1.P + DPo, So1.h + Dho)

       # Assemble both stagnation states into a vector state
       So = So1.stack((So1,So2))

       # Get static states using velocity magnitude and same entropy
       Vmag = np.sqrt(np.sum(Vxrt**2,axis=0))
       h = So.h - 0.5*Vmag**2  # Static enthalpy
       S = So.copy().set_h_s(h , So.s)

       # ...

Now that the static states are known, the density can be used in the
conservation of mass equation to continue with evaluating areas, the RMS
radius, and the shaft angular velocity. The completed function is:

.. code-block:: python
   :caption: fan.py

   def forward(So1, DPo, mdot, phi, psi, htr, etatt):
       """Caluclate mean-line from inlet and design variables."""

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
       rrms = np.sqrt(A/np.pi/2.*(1.+htr**2)/(1.-htr**2))

       # Shaft angular velocity
       Omega = U / rrms

       # Return assembled mean-line object
       return turbigen.flowfield.make_mean_line(
           rrms,  # Mean radii
           A,  # Annulus areas
           Omega,  # Shaft angular velocity
           Vxrt, # Velocity vectors
           S  # Thermodynamic states
       )

This concludes the `forward` function --- all the required quantities have been
evaluated and can be returned for further processing.

Inverse function
^^^^^^^^^^^^^^^^

If we run :program:`turbigen` on the `config.yaml` file now, it will complete
mean-line design successfully using the `forward` function, but raise an
Exception because the `inverse` function is incomplete.

The `inverse` function serves as a verification check that the mean-line
matches the design intent, and also to extract design variables from a
mixed-out CFD solution. We add the design variables as keys in the output
dictionary, using the attributes of the :py:class:`turbigen.meanline.MeanLine`
class to calculate them:

.. code-block:: python
   :caption: fan.py

   def inverse(ml):
       """Calculate design variables from a mean-line object."""

       So1 = ml.stagnation[0]
       So2s = So1.copy().set_P_s(ml.Po[-1], ml.s[0])
       ho2s = So2s.h

       return {
           "So1": So1,
           'DPo': ml.Po[-1] - ml.Po[0],
           'mdot': ml.mdot[0],
           'phi': ml.Vx[0]/ml.U[0],
           'psi': (ml.ho[-1]-ml.ho[0])/(ml.U[0])**2,
           'etatt': (ho2s-So1.h)/(ml.ho[-1]-ml.ho[0]),
           'htr': ml.rhub[0]/ml.rtip[0]
       }

Running CFD
^^^^^^^^^^^

We have now finished the mean-line design. To create blade shapes and run a
computational fluid dynamics simulation, we can add some extra code to the
`config.yaml`. These options are described fully in :doc:`config`.

.. code-block:: yaml
   :caption: config.yaml

   # All files relating to the case are held in a working directory
   workdir: runs/fan

   # Perfect gas inlet state
   inlet:
       Po: 1e5
       To: 300.
       cp: 1005.
       mu: 1.8e-5
       gamma: 1.4

   # Mean-line design
   mean_line:
       type: fan.py  # Path to the mean-line module we are writing
       # Our chosen design variables (args to forward)
       DPo: 2000.
       mdot: 5.
       phi: 0.5
       psi: 0.4
       htr: 0.8
       etatt: 0.9

   # ADD annulus configuration
   annulus:
     AR_gap: [1.0, 1.0]  # Span to inlet/exit boundary distance
     AR_chord: 3.  # Span to chord
     nozzle_ratio: 0.9  # Exit nozzle contraction

   # ADD blade shapes
   blades:
     - DFL: 0.45  # Set number of blades using Lieblein
       sections:  # One blade section at midheight
         - spf: 0.5
           q_thick: [0.05, 0.12, 0.3, 0.02, 0.02, 0.18]
           qstar_camber: [0., 0., 1.0, 1.0, 0.0]

   # ADD mesh generation
   mesh:
     type: h  # Mesh topology
     yplus: 30.0  # Non-dimensional wall distance
     resolution_factor: 0.5  # Use a coarse mesh

   # ADD CFD solver
   solver:
     type: ts3  # Use Turbostream 3
     nstep: 20000
     nstep_avg: 5000
     ilos: 1  # Mixing-length turbulence model
     fmgrid: 0.4  # Full multigrid

   # ADD control mass flow using a PID on exit pressure
   operating_point:
     mdot_pid: [0.5, 0.1, 0.0]

If we now run :program:`turbigen` on our `config.yaml` using the shell command,
we can quickly obtain a CFD solution for our newly designed fan.

.. code-block:: console

    $ turbigen config.yaml
    TURBIGEN v1.5.1
    Starting at 2024-01-29T13:57:10
    Working directory: ...
    Inlet: PerfectState(P=1.000 bar, T=300.0 K)
    Designing a fan.py...
    MeanLine(
        Po=[1.   1.02] bar,
        To=[300.     301.8913] K,
        Ma=[0.099 0.127],
        Vx=[34.5 34.5],
        Vr=[0. 0.],
        Vt=[ 0.  27.6],
        Vt_rel=[-68.9 -41.4],
        Al=[ 0.   38.66],
        Al_rel=[-63.43 -50.19],
        rpm=[2182. 2193.],
        mdot=[5. 5.] kg/s
        )
    Checking mean-line conservation...
    Checking mean-line inversion...
    Annulus(
        xmid=[-7.5253e-07  2.2099e-02],
        rmid=[0.2999 0.2983],
        span=[0.0666 0.0663]
        )
    Re_surf/10^5=[2.]
    Nblade=[54], s_cm=[1.57], tip=[0.]
    Generating an H-mesh...
    Mesh Npts/10^6=0.11
    Applying boundary conditions...
    Setting intial guess...
    Calculating wall distance...
    Running solver ts3...
    Using 1 GPUs on 1 nodes, 1 per node.
    Checking convergence over last 5000 steps...
    mdot drift = 0.0%, mdot error = -0.3%, eta_drift = -0.0%
    Post-processing...
    Mixed-out CFD result:
    MeanLine(
        Po=[0.9997 1.0115] bar,
        To=[300.0521 301.2397] K,
        Ma=[0.099 0.113],
        Vx=[34.5 34.6],
        Vr=[ 0.4 -0.8],
        Vt=[ 0.8 18.2],
        Vt_rel=[-68.1 -50.7],
        Al=[ 1.35 27.72],
        Al_rel=[-63.17 -55.69],
        rpm=[2182. 2193.],
        mdot=[5. 5.] kg/s
        )
    Design variable Nom    CFD
    ------------------------------
    DPo             2000.0 1172.7
    etatt           0.9000 0.8433
    htr             0.8000 0.8000
    mdot            5.0000 4.9961
    phi             0.5000 0.4997
    psi             0.4000 0.2511
    eta_tt=0.843, eta_ts=0.203
    Elapsed time 1.40 min.


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

    Design variable Nom    CFD
    ------------------------------
    DPo             2000.0 1172.7  # Pressure rise too low
    etatt           0.9000 0.8433  # Guessed efficiency too high
    htr             0.8000 0.8000
    mdot            5.0000 4.9961
    phi             0.5000 0.4997
    psi             0.4000 0.2511  # Loading too low

The root cause of the lack of pressure rise is that we have not allowed for
deviation in designing the blade shapes, hence the flow is underturned.
Assuming a guess of efficiency was neccesary to complete mean-line design, but
its value should be updated so that the annulus areas are compatible with the
intended velocity triangles.

Although it is not evident from the table, the inlet flow is not precisely
aligned with the inlet metal angle, leading to unwanted accelerations around
the leading edge. We should locate the stagnation point on the nose of the
aerofoil to yield the smoothest pressure distributions.

:program:`turbigen` has the capability to correct for all these issues. Adding
an `iterate` key to the `config.yaml` will cause the program to repeatedly run
the CFD, updating the efficiency guess and recambering the leading and trailing
edges as needed:


.. code-block:: yaml
   :caption: config.yaml

   # ...

   # ADD new section for iterative corrections
   iterate:
   mean_line:  # Correct efficiency guess
     match_tolerance:
       etatt: 0.01  # Efficiency to within 1%
     relaxation_factor: 0.5  # Change is half CFD minus nominal
   deviation:  # Correct exit flow angles by TE recamber
     clip: 5.0  # Maximum recamber in one step
     relaxation_factor: 0.8  # Multiplier on changes to metal angle
     tolerance: 1.0  # Absolute tolerance for termination in degrees
   incidence:  # Correct incidence by LE recamber
     clip: 5.0   # Maximum recamber in one step
     relaxation_factor: 0.2  # Multiplier on changes to metal angle
     tolerance: 2.0  # Absolute tolerance for termination in degrees

Running the extended input file gives:

.. code-block:: console

    $ turbigen config.yaml
    TURBIGEN v1.5.1
    Starting at 2024-01-29T09:58:58
    Working directory: ...
    Iterating for max_iter=20 iterations
    Min   Inc   DInc  Dev   DDev  etatt Detatt
    -------------------------------------------
    1.401 6.271 1.254 -5.49 4.398 0.844 -0.002
    1.401 4.064 0.812 -2.56 2.051 0.893 0.0232
    1.401 2.502 0.500 -1.24 0.996 0.907 0.0182
    1.401 1.491 0.298 -0.60 0.482 0.911 0.0115
    1.401 0.885 0.177 -0.32 0.258 0.913 0.0067
    1.401 0.497 0.099 -0.11 0.093 0.915 0.0041
    Design variable Nom    CFD
    ------------------------------
    DPo             2000.0 1932.2
    etatt           0.9111 0.9153
    htr             0.8000 0.8000
    mdot            5.0000 4.9941
    phi             0.5000 0.4995
    psi             0.4000 0.3832
    eta_tt=0.915, eta_ts=0.383
    Iteration finished in 8.4 min.

The corrections applied, `DInc`, `DDev`, and `Detatt`, decrease with each
iteration indicating stable convergence. When the iteration terminates, the
mixed-out CFD solution corresponds closely to the design intent. A new
configuration file has been written out in the working directory
`runs/fan/config_conv.yaml` for the converged solution. Inspecting this file:

.. code-block:: yaml
   :caption: config_conv.yaml

   # ...
   qstar_camber:
      - 3.142689298065078
      - 8.280434755893827
      - 1.0
      - 1.0
      - 0.0
   # ...

Under the `qstar_camber` key that defines the camber line, we see that 3.1
degrees of recamber was required to align the stagnation point, and the
deviation was 8.3 degrees. The efficiency has also been updated to 91.5%.

Extensions
^^^^^^^^^^

This tutorial has demonstrated some of the functionality of
:program:`turbigen`. With the current choice of parameterisation, any change to
the design is just an edit to the `config.yaml`, as described in :doc:`config`.

* Increase the number of blades by changing `DFL`
* Increase the grid density under `mesh`
* Control camber and thickness distributions by changing `qthick` and `qstar_camber`
* Specify blade sections at multiple spanwise locations
* Change the aspect ratio `AR_chord`
* With a compatible CFD solver, change the working fluid to a real gas under `inlet`

To change the mean-line design, edit the `forward` and `inverse` functions in
`fan.py`. For example: relax the assumption of constant axial velocity by
adding a velocity ratio as one of the arguments to forward, replace
specification of loading coefficient with a de Haller number, or specify an
inlet Mach number instead of mass flow rate.

To add a stator, extend `forward` to take additional design variables and
perform the necessary calculations. The output data should be at the inlet and
exit of both blade rows, e.g. `A` an array of length 4, the velocity vectors
should be of shape (3,4).
