========
Tutorial
========

This tutorial walks through writing a new mean-line design and using it to
design and simulate a compressor. Everything shown below is a real file in the
repository, run on every commit, so nothing here can quietly stop working:

* the design itself is :download:`examples/turbigen_plugins/fan.py
  <../examples/turbigen_plugins/fan.py>`
* the configuration is :download:`examples/fan.yaml <../examples/fan.yaml>`

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

To proceed with the annulus and blade shape design, :program:`turbigen`
requires a *mean line*: the flow at the inlet and outlet of every blade row.
For each of those stations we must supply

* a thermodynamic state, :math:`(h, s)` for example
* an absolute-frame velocity vector, :math:`(V_x, V_r, V_\theta)`
* an annulus area, :math:`A_m`, and a mean radius, :math:`r`
* the angular velocity of the frame the row turns in, :math:`\Omega`

.. _tut-ml-algo:

Mean-line design equations
^^^^^^^^^^^^^^^^^^^^^^^^^^

We need to combine conservation of mass, momentum, and energy with
definitions of our design variables to solve for the flow in the turbomachine.
This will yield a set of equations that we will later implement numerically in
the `forward` method.

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
exit of the rotor. Static states follow from subtracting the kinetic energy at
constant entropy, :math:`h = h_0 - \tfrac{1}{2}V^2`, and with an equation of
state those give a density.

Conservation of mass then gives the annulus area

.. math::
   :label: eqn-A

   \dot{m} = \rho A_m V_x \quad\Rightarrow\quad A_m = \frac{\dot{m}}{\rho V_x}

and further specifying a hub-to-tip ratio fixes the mean radius

.. math::

   A_m = \pi\left(r_\mathrm{cas}^2 - r_\mathrm{hub}^2\right)\,,\
   r_\mathrm{rms} = \sqrt{\frac{1}{2}\left(r_\mathrm{cas}^2 + r_\mathrm{hub}^2\right)}
   \,,\ \mathit{HTR}=\frac{r_\mathrm{hub}}{r_\mathrm{cas}}

.. math::
   :label: eqn-rrms

   \Rightarrow r_\mathrm{rms} = \sqrt{\frac{A_m}{2\pi}\frac{1+\mathit{HTR}^2}{1-\mathit{HTR}^2}}

Finally, the shaft angular velocity is simply

.. math::
   :label: eqn-Omega

   \Omega = U/r_\mathrm{rms}

Setting up the files
^^^^^^^^^^^^^^^^^^^^

A mean-line design is a subclass of
:class:`~turbigen.design.MeanLineDesign` with two methods: `forward`, which
turns design variables into a mean line, and `backward`, which recovers the
design variables from a mean line. The :ref:`ml-custom` section describes the
general process in more detail.

:program:`turbigen` finds user-written designs by walking up from the
configuration file looking for a directory called `turbigen_plugins`, in the
way that :program:`git` looks for a `.git` directory. So create one beside the
configuration file and put `fan.py` inside it:

.. code-block:: console

   $ mkdir turbigen_plugins

The skeleton of the design looks like this. The `type` string is the name a
configuration file will use to ask for it, and `n_row` says how many blade
rows it describes:

.. code-block:: python
   :caption: turbigen_plugins/fan.py

   from typing import ClassVar

   import numpy as np

   from turbigen.design import MeanLineDesign


   class Fan(MeanLineDesign):
       """A single rotor at fixed inlet stagnation conditions and no inlet swirl."""

       type: ClassVar[str] = "fan"
       n_row: ClassVar[int] = 1

       def forward(self, fluid):
           """Return a mean line built from this design's variables."""
           raise NotImplementedError("Implement the forward method")

       def backward(self, ml):
           """Return the design variables represented by mean line `ml`."""
           raise NotImplementedError("Implement the backward method")

Design variables
^^^^^^^^^^^^^^^^

The design variables are not arguments to `forward`; they are *fields on the
class*. This is what lets them carry their own documentation and defaults, and
what allows a configuration file to be checked against the design before
anything is run. Each of the quantities from the problem statement becomes one
field:

.. literalinclude:: ../examples/turbigen_plugins/fan.py
   :language: python
   :start-at: DPo: float
   :end-before: # SHARED DEFINITIONS
   :dedent: 4

Implementing forward
^^^^^^^^^^^^^^^^^^^^

We can now code up the :ref:`tut-ml-algo`.

:meth:`~turbigen.design.MeanLineDesign.allocate` gives an empty mean line of
the right size, carrying the working fluid from the configuration file. The
first task is the ideal exit enthalpy :math:`h_{02s}=h(p_{01}+\Delta p_0, s_1)`
in Eqn. :eq:`eqn-eta`. Note that nothing here names a perfect gas: a design
should make no assumption about the equation of state, and asking the fluid for
enthalpy at a pressure and an entropy is how that is done.

.. literalinclude:: ../examples/turbigen_plugins/fan.py
   :language: python
   :start-at: ml = self.allocate(fluid)
   :end-at: ho2s = ml.fluid.get_h(*ml.fluid.set_P_s(Po2, s1))
   :dedent: 8

The work, blade speed and velocities then follow directly from the definitions,
Eqns. :eq:`eqn-eta` to :eq:`eqn-Vt`:

.. literalinclude:: ../examples/turbigen_plugins/fan.py
   :language: python
   :start-at: # Work from the definition of efficiency
   :end-at: Vt2 = Dho / U
   :dedent: 8

Next the thermodynamic states. The exit entropy is not a free choice: it is
whatever the pressure rise we asked for and the work we just found imply. With
entropy and stagnation enthalpy known at both stations, the static states come
from subtracting the kinetic energy. `ml.flat` is the mean line seen as a
single streamwise list of stations, which is how the physics of a machine reads
even though the storage is row by row.

.. literalinclude:: ../examples/turbigen_plugins/fan.py
   :language: python
   :start-at: # The exit entropy follows
   :end-at: flat.set_Vt(Vt)
   :dedent: 8

The static states give a density, so conservation of mass now fixes the annulus
areas, Eqn. :eq:`eqn-A`, and the hub-to-tip ratio fixes the mean radius,
Eqn. :eq:`eqn-rrms`, from which the shaft speed follows by Eqn. :eq:`eqn-Omega`:

.. literalinclude:: ../examples/turbigen_plugins/fan.py
   :language: python
   :start-at: # Conservation of mass sets the annulus areas
   :end-at: ml.set_Omega(U / r_rms)
   :dedent: 8

That is every quantity the mean line needs, so `forward` returns it. In full:

.. literalinclude:: ../examples/turbigen_plugins/fan.py
   :language: python
   :caption: turbigen_plugins/fan.py
   :pyobject: Fan.forward
   :dedent: 4

Implementing backward
^^^^^^^^^^^^^^^^^^^^^

`backward` is the single definition of what each design variable *means*. It
serves as a check that the mean line `forward` built really is the design that
was asked for --- :program:`turbigen` runs the round trip every time --- and it
is also how design variables are read back out of a mixed-out CFD solution, so
that the nominal design can be compared against what was achieved.

Anything used by both directions is worth writing once and calling twice. A
formula duplicated between `forward` and `backward` is free to drift:

.. literalinclude:: ../examples/turbigen_plugins/fan.py
   :language: python
   :start-at: def blade_speed(ml)
   :end-before: # DESIGN
   :dedent: 4

`backward` then returns a dictionary keyed by the field names, using
:ref:`meanline` attributes to evaluate them. Keys beyond the design variables
are reported alongside them but are not part of the round trip, which is a
convenient place to put derived quantities such as efficiency:

.. literalinclude:: ../examples/turbigen_plugins/fan.py
   :language: python
   :caption: turbigen_plugins/fan.py
   :pyobject: Fan.backward
   :dedent: 4

The configuration file
^^^^^^^^^^^^^^^^^^^^^^

With the design written, a configuration file can ask for it. The `fluid`
section states the working fluid, and `mean_line` names the design by its
`type` string and sets its fields:

.. literalinclude:: ../examples/fan.yaml
   :language: yaml
   :caption: fan.yaml
   :start-at: fluid:
   :end-before: # Span-to-meridional-chord ratios

At this point ``turbigen design fan.yaml`` will run `forward`, check it with
`backward`, and print the mean line --- everything the design describes, and
nothing more.

To grow blades and a mesh, add the sections that describe them. The annulus is
specified by *aspect ratio* rather than by a chord in metres, which is the
number a designer carries between machines: the chord follows from the span the
mean line chose. The number of blades comes from a circulation coefficient
rather than being stated outright.

.. literalinclude:: ../examples/fan.yaml
   :language: yaml
   :caption: fan.yaml
   :start-at: annulus:
   :end-before: # Hold the design mass flow

Because mass flow is one of our design variables, we want the simulation to
pass the mass flow we designed for rather than whatever falls out of a
prescribed back pressure. Throttling the exit does that: the solver moves the
outlet pressure until the flow is the design value.

.. literalinclude:: ../examples/fan.yaml
   :language: yaml
   :caption: fan.yaml
   :start-at: operating_point:
   :end-at: mdot_adjust: 0.0

Finally the solver settings:

.. literalinclude:: ../examples/fan.yaml
   :language: yaml
   :caption: fan.yaml
   :start-at: solver:

Running CFD
^^^^^^^^^^^

``turbigen run fan.yaml`` now designs the fan, meshes it, solves it once and
reports. The input file, the log and the plots below are the real output of
that command on the files shown above:

.. turbigen-example:: fan

Creating and running designs with different velocity triangles is as simple as
changing a line or two in the `mean_line` section. This allows us to explore a
new design space very quickly.

Iterating the design
^^^^^^^^^^^^^^^^^^^^

The table at the end of the log compares the nominal mean-line design variables
against actual values calculated from the three-dimensional CFD solution, using
cuts that are mixed out at constant area. Reading it, several things are wrong.

The mass flow, flow coefficient and hub-to-tip ratio are all met to within a
fraction of a percent: the throttle held the mass flow, and the annulus is the
one that was drawn. But the pressure rise and the loading are both far short of
what was asked for. The root cause is deviation: the blades were drawn to turn
the flow to the design angle exactly, with no allowance for the flow leaving a
little short of the metal angle, so the rotor does less work than intended and
raises less pressure. The efficiency guess is out too --- it had to be a guess,
since the losses are not known until the flow is solved, but the annulus areas
depend on it.

There is a fourth problem the table does not show. The inlet flow is not
precisely aligned with the leading-edge metal angle, so the flow accelerates
sharply around the nose rather than dividing cleanly on it.

:program:`turbigen` can correct all of these, by re-running the CFD and
adjusting the design between runs. Each corrector is one entry in the `iterate`
list: it names one mismatch, measures it, and moves one design variable to
close it.

.. literalinclude:: ../examples/fan.yaml
   :language: yaml
   :caption: fan.yaml
   :start-at: iterate:
   :end-before: # Step counts

`deviation` recambers the trailing edge until the flow leaves at the design
angle, `incidence` recambers the leading edge until the stagnation point sits
on the nose, and `mean_line` relaxes the `etatt` field of our design onto the
efficiency the solution actually achieved --- named by the same string
`backward` returns it under, which is what ties the two together.

Note that these are listed under `run` above at no cost: ``turbigen run``
solves once and reports every mismatch it can measure, but changes nothing.
Acting on them is a different verb:

.. code-block:: console

   $ turbigen iterate fan.yaml

This repeats the design-mesh-solve cycle, applying the corrections each time,
until every one of them is inside its tolerance. Each simulation restarts from
the previous flow field, so convergence of the flow happens in parallel with
adjustment of the geometry, and no single run has to converge from scratch.
Each iteration is written to its own `iter_NNNN` directory, and the corrections
applied are printed as a table, so a design that is failing to settle can be
seen doing so. Expect this fan to take of the order of ten iterations: the
recamber is deliberately limited per step, because a corrector that took the
whole measured error at once would overshoot and ring.

When it finishes, the working directory holds a configuration file for the
converged design. The recamber angles it records are the deviation and
incidence that were corrected for, and `etatt` is the efficiency the machine
turned out to have rather than the one we guessed.

Extensions
^^^^^^^^^^

This tutorial has demonstrated some of the functionality of
:program:`turbigen`. Within the current choice of parameterisation, any change
to the design is just an edit to `fan.yaml`:

* Change the number of blades by changing the circulation coefficient `Co`,
  or state a count outright with ``count: {type: Nb, Nb: 55}``
* Increase the grid density with `resolution_factor` under `mesh`
* Reshape the blade through the `camber` and `thickness` sections
* Specify blade sections at several spanwise locations
* Change the aspect ratio `AR_row`
* Move off the design mass flow with `mdot_adjust`, which is how a
  characteristic is walked out
* Change the working fluid to a real gas under `fluid`

To change the mean-line design itself, edit `forward` and `backward` in
`fan.py`. For example: relax the assumption of constant axial velocity by
adding a velocity ratio as a field, replace the loading coefficient with a de
Haller number, or specify an inlet Mach number instead of a mass flow rate.

To add a stator, raise `n_row` to 2 and extend `forward` to fill in four
stations rather than two. `ml.flat` will then be a list of four, from machine
inlet to machine outlet, and `ml.set_Omega_row` sets a different speed for each
row.
