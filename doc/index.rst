
.. image:: turbigen-logo.svg
   :width: 30%
   :align: center


:program:`turbigen` is a general turbomachinery design system developed by `Dr. James Brind <https://jamesbrind.uk/>`_ of the `Whittle Laboratory <https://whittle.eng.cam.ac.uk/>`_, University of Cambridge.

This documentation contains
usage instructions, descriptions of the theory involved, and listings of configuration options.

Publications using :program:`turbigen`:

- Brind, J., "Data-driven radial compressor design space mapping". *J. Turbomach.* :cite:`Brind2024`.

- Torres-Gomez, A., Brind, J., and Pullan, G. "Cryogenic Radial Turbine Design for High-Efficiency Hydrogen Liquefaction Plants". *ASME Turbo Expo 2025* :cite:`TorresGomez2025`.


User manual
===========
.. Two trees, for one depth each. The schema reference is a section per
   top-level key, so the input file format is worth three levels: a reader
   looking for `mesh:` finds it in the sidebar rather than by scrolling. The
   pages below it are worth two, because `usage` grows a "Positional
   Arguments" and a "Named Arguments" heading for every command it documents,
   and fourteen of those in the sidebar bury everything else.

.. toctree::
   :maxdepth: 3

   tutorial
   config

.. toctree::
   :maxdepth: 2

   usage
   meanline
   design
   changelog
   license
   Source code repository <https://github.com/jb753/turbigen>
   references
