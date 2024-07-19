Post-processing
===============

:program:`turbigen` can generate plots of the designed geometry and CFD-predicted flow field, or write out such data to files, on completion
of a run.
The plots or data files appear in a subdirectory of the working directory named `post`.
Post-processing is configured using the `post_process` key in
a YAML input file, which may list one or more of the modules under :py:mod:`turbigen.post`, and provide arguments as subkeys.  For example, to write out surface
coordinates for CAD, and plot the first row chordwise pressure distribution at three spanwise locations:


.. code-block:: yaml

   post_process:
     write_surf:  # Needs no further arguments
     plot_pressure_distributions:
       # Nested list of span fractions for each row
       row_spf: [[0.1, 0.5, 0.9],]



.. automodule:: doc.dummy_post
   :autosummary:
   :members:
   :imported-members:
