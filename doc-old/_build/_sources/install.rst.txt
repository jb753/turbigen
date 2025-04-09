Installation
============

This page describes how to install :program:`turbigen`.
Lines prefixed with `$` are to be run at the Linux terminal.

.. warning::
   :program:`turbigen` is only tested on Linux --- expect issues if running on Windows.


Prerequisites
^^^^^^^^^^^^^

The program requires a working Python installation, version 3.9 or greater.
Check your Python version by running,

.. code-block:: console

   $ python3 -V
   Python 3.11.2

If your Python version is below 3.9, you need to install a more recent version.
You will also need the `virtualenv` module to make an installation separate
from your system Python modules; this may require an extra package, e.g. `sudo
apt install python3-virtualenv` in Debian. Check for the `virtualenv` module by
running,

.. code-block:: console

   $ python3 -c 'import virtualenv; print("virtualenv installed")'
   virtualenv installed

Obtain the code
^^^^^^^^^^^^^^^

The code is available to download from the University of Cambridge GitLab
instance.

.. code-block:: console

   $ git clone https://gitlab.developers.cam.ac.uk/jb753/turbigen.git
   $ cd turbigen
   $ source setup.sh

This will create a Python virtualenv using your system interpreter, to hold
:program:`turbigen` and it's dependencies in isolation from your operating
system Python installation.

Basic usage
^^^^^^^^^^^

.. role:: bash(code)
   :language: console

.. note::

   Each time you start a session of turbigen in a new terminal, you must setup
   the environment by running :bash:`$ source setup.sh` again. Failing to
   do so will result in errors like :bash:`turbigen: command not found`.

To run a case, use,

.. code-block:: console

    $ turbigen INPUT_FILE

where `INPUT_FILE` is a yaml configuration file. The manual contains a list of all :ref:`usage`. Several specimen configuration files are provided in the `examples` directory.

Test case
^^^^^^^^^

As a test to verify the installation has completed sucessfully, run the configuration :file:`examples/cascade_test.yaml`. This should design and mesh a turbine cascade, but not run the CFD, producing the following output:

.. program-output:: turbigen ../examples/cascade_test.yaml
