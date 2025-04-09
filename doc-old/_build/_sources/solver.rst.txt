.. _solvers:

Solvers
=======

turbigen is CFD-solver agnostic, in that all all pre- and post-processing is
done by native code. Adding a new solver only requires routines to save the
internal grid data to a CFD input file, execute the solver, and read back the
flow solution.

Each CFD solver accepts different configuration options. Solver options and their
default values are listed below; override the defaults using the `solver`
section of the `config.yaml` file.

Turbostream 3
-------------

Turbostream 3 is a multi-block structured, GPU-accelerated Reynolds-averaged
Navier--Stokes code developed by :cite:`Brandvik2011`.

.. autoclass:: turbigen.solvers.ts3.Config
   :members:

Turbostream 4
-------------

Turbostream 4 is an unstructured, GPU-accelerated Reynolds-averaged
Navier--Stokes code.

.. autoclass:: turbigen.solvers.ts4.Config
   :members:
