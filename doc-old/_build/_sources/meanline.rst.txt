.. _ml:

Mean-line
=========

The first step in turbomachinery design is a one-dimensional analysis along the
'mean-line'. We consider an axisymmetric streamsurface at an intermediate mean
radius, for the inlet and exit of each blade row, as a simplified model of the
real three-dimensional flow. Given an inlet condition and machine duty,
mean-line design fixes the annulus radii and flow angles.

:program:`turbigen` abstracts mean-line design in the following
way. Each type of machine (e.g. 'axial turbine', 'radial compressor') has
`forward` and `inverse` functions, stored in a module under :file:`turbigen/meanline/`.

The `forward()` function takes as arguments: an inlet state object from
:py:mod:`turbigen.fluid`; and some duty, geometric, or aerodynamic choices
(e.g. mass flow rate, radius ratio, flow coefficient). The `forward()` function
returns a :py:class:`turbigen.meanline.MeanLine` object that encapsulates all
required information about the mean-line design: flow properties, velocities,
annulus areas and hub and shroud radii.

As described in the :ref:`Configuration page<ml_conf>`, the entries in the `mean_line` section of a :program:`turbigen` YAML file are fed directly into the chosen
`forward()` function as keyword arguments. For example, if the configuration file contains:

.. code-block:: yaml

   mean_line:
     type: axial_turbine
     Alpha1: 0.
     phi: 0.8
     psi: 1.6
     Lam: 0.5
     # ... more keys

Then within the code, a :py:class:`turbigen.meanline.MeanLine` object will be created using something like:

.. code-block:: python

    import turbigen.meanline.axial_turbine

    ml = turbigen.meanline.axial_turbine.forward(
        So1,  # Inlet state calculated elsewhere, positional arg
        Alpha1=0.,  # Keys from `mean_line` config unpacked as kwargs
        phi=0.8,
        psi=1.6,
        Lam=0.5,
        # ... more args
    )

The `inverse()` function takes a :py:class:`turbigen.meanline.MeanLine` object as
its only argument, and calculates from the flow field a dictionary of the
same parameters that are input to `forward()`. Given a suitably averaged CFD
solution, `inverse()` is a post-processing step that allows comparison of the
three-dimensional simulated flow field to the one-dimensional design intent.
Also, feeding the output of `forward()` straight into `inverse()` performs a
check that the mean-line design is consistent with the requested input
parameters.

The mean-line parts of :program:`turbigen` are flexible and extensible. Any
parameterisation can be used for mean-line design, once a `forward()` function
is written to take the chosen design variabes as arguments and perform the
calculations. Any turbomachine type can be programmed and represented as a
:py:class:`turbigen.meanline.MeanLine` object.

The following turbomachine architectures are currently implemented

* Cascade
* Axial turbine
* Radial compressor

.. autoclass:: turbigen.meanline.MeanLine
   :members:

Cascade
-------
.. automodule:: turbigen.meanline.turbine_cascade
   :members:

Axial turbine
-------------
.. automodule:: turbigen.meanline.axial_turbine
   :members:

Radial compressor
-----------------
.. automodule:: turbigen.meanline.radial_compressor
   :members:
