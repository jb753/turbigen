Tutorials
=========

Writing a mean-line
-------------------

This tutorial will walk through the process of writing a new user-defined
mean-line solver to integrate into the :program:`turbigen` design system.
We first need to download and install the code:

.. code-block:: console

    $ git clone https://gitlab.developers.cam.ac.uk/jb753/turbigen.git
    $ cd turbigen
    $ source setup.sh

Problem statement
^^^^^^^^^^^^^^^^^

Suppose we wish to design a rotor-only, low-speed axial fan. We shall assume
incompressible flow and a constant axial velocity. The inlet state is specified
as fixed values of :math:`T_{01}` and :math:`p_{01}` with no inlet swirl,
:math:`\alpha_1=0`. We can then parametrise the aerodynamics of the stage using the
following design variables:

* Total pressure rise, :math:`\Delta p_0`
* Mass flow rate, :math:`\dot{m}`
* Flow coefficient, :math:`\phi=V_x/U`
* Loading coefficient, :math:`\psi=c_p \Delta T_0/U^2`
* Hub-to-tip ratio, :math:`\mathit{HTR}=r_\mathrm{hub}/r_\mathrm{cas}`
* Total-to-total isentropic efficiency guess, :math:`\eta_\mathrm{tt}`


To proceed with the annulus and blade shape design, :program:`turbigen` requires the following quantities to be calculated from the above information. At the inlet and exit of the blade row:

* Mean radii, :math:`r_\mathrm{rms}`
* Annulus areas, :math:`A`
* Absolute frame velocity vectors, :math:`(V_x, V_r, V_\theta)`
* Static thermodynamic states, :math:`(P, T)` for example
* Rotor shaft speed, :math:`\Omega`

.. _tut-ml-algo:

Mean-line design equations
^^^^^^^^^^^^^^^^^^^^^^^^^^

We need to combine conservation of mass, momentum, and energy with
definitions for our design variables to solve for the flow in the turbomachine.
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

We have two functions to write: a `forward` function which takes our design
variables as inputs and returns a :py:class:`turbigen.meanline.MeanLine`
object; and an `inverse` function that recalculates the design variables from
an input :py:class:`turbigen.meanline.MeanLine` object. Now that we know what
input and output data are required, we can start writing these functions. In a
new file called `fan.py`, copy and paste these definitions:

.. code-block:: python
   :caption: fan.py

   import turbigen.flowfield
   import numpy as np

   def forward(So1, DPo, mdot, phi, psi, htr, etatt):
       """Caluclate mean-line from inlet and design variables."""

       # Insert code to calculate rrms, A, Omega, Vxrt, states
       # ...
       raise NotImplementedError

       # Return assembled mean-line object
       return turbigen.flowfield.make_mean_line(
           rrms,  # Mean radii
           A,  # Annulus areas
           Omega,  # Shaft angular velocity
           Vxrt, # Velocity vectors
           S  # Thermodynamic states
       )

   def inverse(ml):
       """Calculate design variables from a mean-line object."""

       # The output should be a dictionary keyed by the args to forward
       return {
           'So1': ml.stagnation[0],
           # 'DPo': ...,
           # 'mdot': ...,
           # 'phi': ...,
           # 'psi': ...,
           # 'htr': ...,
           # 'etatt': ...,
       }

`So1` is a fluid object that encapsualtes the inlet stagnation thermodynamic state. All
thermodynamic properties can be accessed as attributes, and there are functions
to manipulate the state to new values, described fully in :py:mod:`turbigen.fluid` .

We also need a minimal configuration file to test our mean-line functions.
Create a new `config.yaml` with the following content:

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
       DPo: 200.
       mdot: 5.
       phi: 0.5
       psi: 0.4
       etatt: 0.8

At this point, running the config.yaml file through :program:`turbigen` by using the shell command

.. code-block:: console

    $ turbigen config.yaml

generates a `NotImplementedError` because the body of the `forward` function is missing.

Implementing the algorithm
^^^^^^^^^^^^^^^^^^^^^^^^^^

We can now start to add the :ref:`tut-ml-algo` to the `forward` function inside
`fan.py`.

The first task is to calculate the idea exit enthalpy :math:`h_{02s}`
from Eqn. :eq:`eqn-eta`. Mean-line design functions should be written to make
no assumptions about the working fluid equation of state --- this is accomplished
using the fluid modelling abstractions in :py:mod:`turbigen.fluid`. We take a
copy of the inlet state, and set its pressure and entropy to the required
values.

.. code-block:: python
   :caption: fan.py

   # ...

   def forward(So1, DPo, mdot, phi, psi, htr, etatt):
       """Caluclate mean-line from inlet and design variables."""

       # Get the ideal exit state
       So2s = So1.copy()  # Duplicate the inlet state
       So2s.set_P_s(So1.P + DPo, So1.s)  # Set pressure and entropy

       # ...

We can now calculate the compressor work and velocity vectors by reading off
enthalpy values from our two state objects `So1` and `So2s`.

.. code-block:: python
   :caption: fan.py

   # ...

   def forward(So1, DPo, mdot, phi, psi, htr, etatt):
       """Caluclate mean-line from inlet and design variables."""

       # Get the ideal exit state
       So2s = So1.copy()  # Duplicate the inlet state
       So2s.set_P_s(So1.P + DPo, So1.s)  # Set pressure and entropy

       # Work from defn efficiency Eqn. (1)
       Dho = (So2s.h-So1.h)/etatt

       # ...

Proceeding straightforwardly to calculate blade speed and velocity vectors

.. code-block:: python
   :caption: fan.py

   # ...

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

Next, the
# Annulus area from cons mass Euler Eqn. (5)
# Shaft angular velocity Eqn. (6)
