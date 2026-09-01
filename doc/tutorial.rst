========
Tutorial
========

This tutorial walks through writing a new mean-line class in order to design a compressor. Running actual CFD is out of scope for now.

We first need to install :program:`turbigen`, which can be achieved on Linux or
macOS with something like these shell commands:

.. code-block:: console

   $ curl -LsSf https://astral.sh/uv/install.sh | sh
   $ source $HOME/.local/bin/env
   $ uv tool install turbigen

Usually, we run :program:`turbigen` by passing an input YAML file containing
all the data required to construct a turbomachine. Your mean-line design
algorithm, being code not data, must be written seperately in Python.
:program:`turbigen` finds user-written designs by walking up from the input
file looking for a directory called `turbigen_plugins`. The below commands
create a directory called `tutorial`, change into it, create the
`turbigen_plugins` directory, and create an empty input file and a Python file
for the design:

.. code-block:: console

   $ mkdir tutorial
   $ cd tutorial
   $ mkdir turbigen_plugins
   $ touch input.yaml turbigen_plugins/fan.py

The file structure should then look like this:

.. code-block:: console

   $ tree
   .
   ├── input.yaml
   └── turbigen_plugins
       └── fan.py

   2 directories, 2 files

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

To proceed with the annulus and blade shape construction, :program:`turbigen`
requires a *mean line design*: a nominal averaged flow field at the inlet and
outlet of every blade row. For each of those stations we must supply

* a thermodynamic state, :math:`(h, s)` for example
* an absolute-frame velocity vector, :math:`(V_x, V_r, V_\theta)`
* an annulus area, :math:`A_m`, and a mean radius, :math:`r_\mathrm{rms}`
* the angular velocity of the blade frame, :math:`\Omega`

.. _tut-ml-algo:

Mean-line design equations
^^^^^^^^^^^^^^^^^^^^^^^^^^

We need to combine conservation of mass, momentum, and energy with
definitions of our design variables to solve for the flow in the turbomachine.
This will yield a set of equations that we will later implement numerically in
a method on our new class.

The specified total pressure rise and guess of total-to-total efficiency allow calculation of the compressor work :math:`\Delta h_0 = h_{02}-h_{01}`

.. math::
   :label: eqn-eta

   \eta_\mathrm{tt} = \frac{h_{02s}-h_{01}}{h_{02}-h_{01}} \quad\Rightarrow\quad \Delta h_0 = \frac{1}{\eta_\mathrm{tt}}\left[h(p_{01}+\Delta p_0, s_1) - h_{01}\right]

where :math:`h_{02s}=h(p_{01}+\Delta p_0, s_1)` is the ideal exit stagnation enthalpy for isentropic compression, i.e. enthalpy evaluated at the (lossy) exit stagnation pressure and (lossless) inlet entropy.

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
exit of the rotor. Static enthalpy follows from subtracting the kinetic energy at
constant entropy, :math:`h = h_0 - \tfrac{1}{2}V^2`; entropy does not depend on the frame of reference. Now, passing :math:`(h, s)`
to the equation of state will yield density :math:`\rho` or any other static thermodynamic
property as needed.

Conservation of mass then gives the annulus area at each station. Subscript :math:`m` denotes the meridional projection with no circumferential component, as opposed to the flow area which is normal to the velocity vector (and thus dependent on flow angle).

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
:class:`~turbigen.design.MeanLineDesign` which has two methods: :meth:`~turbigen.design.MeanLineDesign.forward`, which
turns design variables into a flow field :class:`~turbigen.meanline.MeanLine`, and :meth:`~turbigen.design.MeanLineDesign.backward`
, which recovers the
design variables from a :class:`~turbigen.meanline.MeanLine` built by :meth:`~turbigen.design.MeanLineDesign.forward` or averaged from a CFD solution.
The :ref:`design-contract` sets out in full what such a subclass declares and
what it inherits.

Copy the below skeleton of the class into `turbigen_plugins/fan.py`. The `type`
string is the name a input file will use to ask for the new turbomachine, and
`n_row` says how many blade rows it describes:

.. literalinclude:: ../tutorial/step1/turbigen_plugins/fan.py
   :language: python
   :caption: turbigen_plugins/fan.py

We can now copy into `input.yaml` a skeleton configuration that asks for this
design, but in order for the mean-line design to run we also need to specify the
working fluid. The :ref:`fluid: <config-fluid>` section names a fluid by its `type` and sets any
properties needed to specify that type.

.. literalinclude:: ../tutorial/step1/input.yaml
   :language: yaml
   :caption: input.yaml

If we ask :program:`turbigen` to run the design for `input.yaml` at this point, it will load the input data but stop because `forward` is not implemented.

.. program-output:: turbigen design input.yaml
   :cwd: ../tutorial/step1
   :returncode: 1
   :prompt:
   :ellipsis: 1, -2

Design variables
^^^^^^^^^^^^^^^^

We now need to declare the design variables we need to take from the input file,
by adding them as fields to the class.

.. literalinclude:: ../tutorial/step2/turbigen_plugins/fan.py
   :language: python
   :caption: turbigen_plugins/fan.py
   :start-at: class Fan
   :end-before: def forward
   :prepend: # ...
   :append: # ...

Optionally, a default can be set for a field, which will be used if the input file does not specify a value; the input file must include all fields without defaults. Hence, running the design now will produce a different error:

.. program-output:: turbigen design input.yaml
   :cwd: ../tutorial/step2
   :returncode: 1
   :prompt:
   :ellipsis: 1, -2

Note that `To1` and `Po1` are not mentioned in the error message, because we specified their defaults in the class (although the input file can override those defaults if needed). Finally, before moving to implementing the algorithm, add values for the missing design variables to `input.yaml`:

.. literalinclude:: ../tutorial/step3/input.yaml
   :language: yaml
   :caption: input.yaml
   :start-at: mean_line:
   :prepend: # ...


.. _tut-forward:

Forward method
^^^^^^^^^^^^^^

We can now code up the :ref:`tut-ml-algo`. The `forward` method takes a `fluid`
object as an argument, and can access the design variables as attributes of
`self`. The method must return a :class:`~turbigen.meanline.MeanLine` object
with the flow field filled in.

The `Fan` class specifies inlet stagnation pressure and
temperature, but enthalpy and entropy are more
convinient to work with. `fluid.set_P_T` returns the density and internal energy
pair for a pressure and a temperature, which are then passed
to `fluid.get_h` and `fluid.get_s` to evaluate stagnation enthalpy and
entropy (noting that entropy does not depend on the frame of reference):

.. literalinclude:: ../tutorial/step3/turbigen_plugins/fan.py
   :language: python
   :start-at: def forward(self, fluid):
   :end-at: s1 = fluid.get_s(rhoo1, uo1)
   :dedent: 4

At no point have we used perfect gas relations --- a design class should make no
assumptions about the equation of state.

To evaluate Eqn. :eq:`eqn-eta` we straightforwardly calculate exit stagnation pressure from the specified pressure rise, and pass it together with inlet entropy through `fluid.set_P_s` and  `fluid.get_h` to evaluate the ideal exit enthalpy. Then rearrange and use the definition of efficiency to find the work done:

.. literalinclude:: ../tutorial/step3/turbigen_plugins/fan.py
   :language: python
   :start-at: # Ideal exit stagnation enthalpy
   :end-at: Dho = (ho2s - ho1) / self.eta_tt
   :dedent: 4

The blade speed and velocities then follow directly from the definitions
Eqns. :eq:`eqn-psi` to :eq:`eqn-Vt`:

.. literalinclude:: ../tutorial/step3/turbigen_plugins/fan.py
   :language: python
   :start-at: # Blade speed from the definition of loading coefficient
   :end-at: Vt2 = Dho / U
   :dedent: 4

The exit entropy is set by the actual work and pressure rise. Then
with entropy, stagnation enthalpy, and velocity known at both stations, the
static states come from subtracting kinetic energy.

.. literalinclude:: ../tutorial/step3/turbigen_plugins/fan.py
   :language: python
   :start-at: # Exit entropy from actual work and pressure rise
   :end-at: h = ho - 0.5 * (Vx**2 + Vt**2)
   :dedent: 4

We can now store the flow field in a :class:`~turbigen.meanline.MeanLine`
object, a blank instance of which is created by calling
:meth:`~turbigen.design.MeanLineDesign.allocate`. The indexing convention for mean
line stations is that `ml.shape == (2, n_row)`, where the first index is the
station (0 for inlet, 1 for exit) and the second index is the row number. It is
often more convenient to work with a one-dimensional view of shape `(2*n_row,)`, which
can be obtained through the  `ml.flat` property. All data flows go through setter
methods on the :class:`~turbigen.meanline.MeanLine` object that ensure
the flow field is consistent and valid. See :doc:`/meanline` for the full
reference of what is stored, how it is indexed, and which setters are
available. In our case:

.. literalinclude:: ../tutorial/step3/turbigen_plugins/fan.py
   :language: python
   :start-at: # Store the flow field
   :end-at: flat.set_Vt(Vt)
   :dedent: 4

Now that the flow field is stored, we can use attributes on the :class:`~turbigen.meanline.MeanLine` object to evaluate derived quantities automatically. For example, we need density to set the annulus area via conservation of mass, Eqn. :eq:`eqn-A`, which we can access as `ml.flat.rho`. The hub-to-tip ratio then fixes the mean radius,
Eqn. :eq:`eqn-rrms`, from which the shaft speed follows by Eqn. :eq:`eqn-Omega`:

.. literalinclude:: ../tutorial/step3/turbigen_plugins/fan.py
   :language: python
   :start-at: # Conservation of mass sets the annulus areas
   :end-at: ml.set_Omega(U / r_rms)
   :dedent: 4

That is every quantity the mean line needs, so `forward` can now return it. In full:

.. literalinclude:: ../tutorial/step3/turbigen_plugins/fan.py
   :language: python
   :caption: turbigen_plugins/fan.py
   :pyobject: Fan.forward
   :dedent: 4

If we now run :program:`turbigen` on `input.yaml`, it will load the input data, build the mean line, and then fall over because `backward` is not implemented:

.. program-output:: turbigen design input.yaml
   :cwd: ../tutorial/step3
   :returncode: 1
   :prompt:
   :ellipsis: 1, -2

.. _tut-backward:

Backward method
^^^^^^^^^^^^^^^

`backward` is the encapsulation of what each design variable means. It is a
check that the mean line `forward` built really is the design that was asked
for and it is also how design variables are calculated from a mixed-out CFD
solution to compare to the nominal design.

`backward` returns a dictionary keyed by the field names; extra keys beyond the
design variables are reported for information only but not checked for
consistency, as described under :ref:`design-process`. The :class:`~turbigen.meanline.MeanLine` has many attributes that
contain useful derived properties for this purpose --- see the
:doc:`derived properties </meanline>` tables. The properties `ml.inlet`
and `ml.exit` index into the machine inlet and exit (across all rows).

.. literalinclude:: ../tutorial/step4/turbigen_plugins/fan.py
   :language: python
   :caption: turbigen_plugins/fan.py
   :pyobject: Fan.backward
   :dedent: 4

With both methods written, the design runs to completion. :program:`turbigen`
calls `forward` to build the mean line, passes it back through `backward` to
check that the design variables it asked for are the ones it got, and prints
the result --- the :ref:`design-process` in full:

.. program-output:: turbigen design input.yaml
   :cwd: ../tutorial/step4
   :returncode: 0
   :prompt:
   :ellipsis: 1, 3

Annulus and blade shapes
^^^^^^^^^^^^^^^^^^^^^^^^

With the mean line built, we can now construct the annulus lines and blades by
specifying some more input data.

To use the built-in annulus requires only a few fairly self-explanatory
lines in the input file:

.. literalinclude:: ../tutorial/step5/input.yaml
   :language: yaml
   :caption: input.yaml
   :start-at: annulus:
   :end-before: blades:
   :prepend:
      # ...
   :append:
      # ...

Blade specification is more involved. We need to choose how many blades in each
row; the Lieblein diffusion factor is a good starting point for compressors. We
can construct the blade sections at any number spanwise positions by choosing
camber and thickness distribution, with the camber and thickness parameters
smoothly interpolated across the span.

.. literalinclude:: ../tutorial/step5/input.yaml
   :language: yaml
   :caption: input.yaml
   :start-at: blades:
   :prepend:
      # ...

Now passing the input file to :program:`turbigen` will now build the mean line
and the annulus and blade shapes to give us some more printout, including the calculated meridional chord, number of blades, tip gap, and pitch-to-chord ratio:

.. program-output:: turbigen design input.yaml
   :cwd: ../tutorial/step5
   :returncode: 0
   :prompt:
   :ellipsis: 1, 3

Plotting
^^^^^^^^

So far, our invocation of :program:`turbigen` has been in :ref:`design <usage-design>` mode, which
only prints to the console. We can save more detailed plots to disk by
running :ref:`report <usage-report>` mode, which will write a `post.pdf` report with plots of the
design, and if CFD were run the post-processed flow field as well. Report mode also saves an `output.yaml` file which is the same as the input file but with any defaults filled in, and a transcript of the console output in `log_turbigen.txt`. So if we run:

.. program-output:: turbigen report input.yaml
   :cwd: ../tutorial/step5
   :returncode: 0
   :prompt:
   :ellipsis: 1, -7

The flow-field plots are skipped because no CFD has been run; the annulus,
blade section and velocity triangle plots are drawn from the geometry alone. All
of the output is written alongside the input file:

.. program-output:: ls
   :cwd: ../tutorial/step5
   :prompt:


Opening `post.pdf` shows the mean-line velocity triangles, the meridional annulus view, the blade-to-blade
sections of each row:

.. figure:: /_static/tut_step5_triangle.svg
   :width: 100%

.. figure:: /_static/tut_step5_annulus.svg
   :width: 90%

.. figure:: /_static/tut_step5_sections.svg
   :width: 90%

Extensions
^^^^^^^^^^

This tutorial has demonstrated some of the functionality of
:program:`turbigen`. We have now designed a new turbomachine and had a look at the resulting geometry.

Within the current choice of mean-line parameterisation,
any change to the design is just an edit to `input.yaml`:

* Change the number of blades by changing the diffusion factor
  :ref:`DFL <config-blades-count-dfl>`, or fix a
  :ref:`count <config-blades-count>` directly with ``count: {type: Nb, Nb: 55}``
* Reshape the blade through the :ref:`camber <config-blades-sections-camber>`
  and :ref:`thickness <config-blades-sections-thickness>` sections
* Specify blade sections at several spanwise locations
* Change the aspect ratio `AR_row`
* Change the working fluid to a :ref:`real <config-fluid-real>` gas under
  :ref:`fluid: <config-fluid>`, evaluated from a
  thermodynamically consistent fitted equation of state :cite:`Wheeler2024`

To change the mean-line design itself, edit `forward` and `backward` in
`fan.py`. For example: relax the assumption of constant axial velocity by
adding a velocity ratio as a field, replace the loading coefficient with a de
Haller number, or specify an inlet Mach number instead of a mass flow rate. The
last two cannot be built in one pass like the design variables here, because
the flow field has to be known before either can be evaluated; see
:ref:`design-implicit` for how to solve for them.

To add a stator, raise `n_row` to 2 and extend `forward` to fill in four
stations rather than two. `ml[0]` will then index the rotor, and `ml[1]` the
stator. We need to call `ml[0].set_Omega` to make only the rotor spin.
`ml.flat` will now be of shape `(4,)`.
