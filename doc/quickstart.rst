Quick-start guide
=================

This page describes how to install and run :program:`turbigen`.
Lines prefixed with `$` are to be executed at the Linux terminal.

.. warning::
   :program:`turbigen` is only tested on Linux --- expect issues if running on Windows.


Prerequisites
^^^^^^^^^^^^^

The program requires a working Python installation and a Fortran compiler. To
keep the installation separate from your system Python modules, it is
recommended to use a Python virtual environment and/or another Python package
manager like `conda`. Your distribution will provide a Fortran compiler, e.g.
`sudo apt install gfortran` in Debian.

Once your environment is ready, install :program:`turbigen` and its
dependencies using:

.. code-block:: console

   $ pip install turbigen


Basic usage
^^^^^^^^^^^

To run a case, use,

.. code-block:: console

    $ turbigen INPUT_YAML

where `INPUT_YAML` is a yaml configuration file. Several specimen configuration files are provided in the :doc:`examples/index` directory.

Test case
^^^^^^^^^

As a test to verify the installation has completed sucessfully, run the configuration :file:`examples/cascade_test.yaml`. This should design and mesh a turbine cascade, but not run the CFD, producing the following output:

.. program-output:: turbigen ../examples/cascade_test.yaml
