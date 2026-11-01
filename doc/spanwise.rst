.. _spanwise:

Spanwise flow
=============

.. automodule:: turbigen.spanwise

A spline design gives a profile at all four stations of a two-row machine,
here a fan rotor and outlet guide vane whose blades each have sections at span
fractions 0.2, 0.5 and 0.8. Each station takes one value per section of its
row, a free vortex at uniform meridional velocity where nothing is written:

.. code-block:: yaml

   spanwise:
     type: velocity_spline
     stations:
       - {}                         # rotor inlet
       - {Vm: [0.95, 1.05, 0.98]}   # rotor outlet
       - {Vm: [0.95, 1.05, 0.98]}   # vane inlet
       - {}                         # vane outlet

Every knot is a leaf of the config, so ``spanwise.stations[1].rVt[2]`` can be
swept by a batch or set from the command line like any other. Adding a section
to a blade adds a knot to both its stations, so their lists must grow with it.

Iterating a spanwise design
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Three iterators under :ref:`iterate: <config-iterate>` move the knots of a
``velocity_spline`` design:

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - ``type:``
     - Knots it moves
     - Until
   * - ``spanwise_inflow``
     - ``Vm`` at every station and ``rVt`` at every row inlet, of the
       ``rows`` it names or all of them
     - the profiles match the pitch-averaged CFD
   * - ``spanwise_work``
     - ``rVt`` at one rotor's outlet
     - the stagnation enthalpy rise along each stream surface matches a work
       coefficient distribution ``psi``, one value per section
   * - ``spanwise_swirl``
     - ``rVt`` at one stator's outlet
     - the swirl leaving matches a distribution ``rVt``, one value per section

The first copies a measured state in. The other two set the trailing edge, so
neither can be used with ``deviation``, which recambers every row's: the outlet
knots carry the mean deviation, and the mean work or swirl is corrected along
with its distribution. Only the shape of a ``psi`` or ``rVt`` target is used
--- its mass average is removed --- since the mean is the mean line's to set.

.. code-block:: yaml

   iterate:
     correct:
       - type: spanwise_inflow
       - type: spanwise_work
         i_row: 0
         psi: [0.03, 0.0, -0.03]
       - type: spanwise_swirl
         i_row: 1
         rVt: [0.0, 0.0, 0.0]

Each is measured on the cut planes the mean line is mixed out on, a little
into the gap beside each blade edge, pitch-averaged by mass and fitted onto the
section knots by least squares, over the span between the outermost sections.
The endwall bands outside them still count towards the mean: an outlet
iterator holds the mean of its knots to the mean measured over the whole span,
so the row as a whole meets the mean line. The ``spanwise_targets``
post-processor draws each measured profile against the target it is driven to.
