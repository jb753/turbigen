Quick-start guide
=================

This page describes how to install and run :program:`turbigen`.
Lines prefixed with `$` are to be executed at the Linux terminal.

.. warning::
   :program:`turbigen` is only compatible with Linux -- Windows is not supported.


Prerequisites
^^^^^^^^^^^^^

The program requires Python version 3.11 or greater.
To keep the program separate from your system Python modules to
avoid interference, we recommend the `uv` `package manager <https://docs.astral.sh/uv/getting-started/installation/>`_. To install :program:`turbigen` using it, run the below commands.

.. code-block:: console

   $ curl -LsSf https://astral.sh/uv/install.sh | sh
   $ source $HOME/.local/bin/env
   $ uv tool install turbigen

These commands will install `uv`, then use it to install :program:`turbigen` into an
isolated environment with a compatible Python interpreter. The executable is placed on your `$PATH` so it can always be found without explicitly activating a virtual environment.

To later upgrade
your installation to a newer version, run:

.. code-block:: console

   $ uv tool upgrade turbigen


Basic usage
^^^^^^^^^^^

To run a case, use,

.. code-block:: console

    $ turbigen INPUT_YAML

where `INPUT_YAML` is a yaml configuration file. Specimen configuration files
are provided in the :doc:`examples/index` directory.
Results will be logged to the screen and saved to a file in the working
directory specified in the input file.

Command-line flags can
also be used to temporarily change the behaviour of the program and override some settings in the input file. To see a list of these,
run:

.. code-block:: console

    $ turbigen --help
