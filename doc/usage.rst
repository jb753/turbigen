.. _usage:

Command-line interface
======================

Four of the commands are the same pipeline stopped at different points, and
each implies the stages before it:

* `design` builds the mean line and the geometry;
* `run` meshes and solves for the flow field;
* `iterate` updates the geometry and repeats a run until the CFD-predicted flow field matches the nominal design intent;
* `chic` sweeps a geometry along its mass flow characteristic.

There are two more commands:

* `report` recreates plots from a finished run;
* `batch` writes a set of configuration files over a design space.

Examples
^^^^^^^^

.. code-block:: console

   # Construct a design and print to console
   $ turbigen design input.yaml

   # Construct a design and write plots to disk
   $ turbigen report input.yaml

   # The same, with one design variable changed
   $ turbigen design input.yaml -s mean_line.psi=1.8

   # Design, run CFD, and post process one case
   $ turbigen run input.yaml

   # Iterate a geometry until it matches design intent
   $ turbigen iterate input.yaml

   # Sweep a characteristic
   $ turbigen chic input.yaml

   # Write configurations over a design space
   $ turbigen batch input.yaml

A command only asks of a configuration what its own stage needs, and says so
when it is missing: `run`, `iterate` and `chic` want a
:ref:`solver: <config-solver>` key, `chic` a :ref:`chic: <config-chic>` key as
well, and `batch` a :ref:`batch: <config-batch>` key naming the design
variables to vary. `design` and `report` need neither, which is why a
configuration describing only a mean line still has two commands that work on
it. The whole file is described in :ref:`config`.

Output files
^^^^^^^^^^^^

The `design` command only prints to console, but all other commands write files
and require a working directory. By default, all output is written in the same
directory as the input file; with the `-o DIR` switch specified
:program:`turbigen` will create `DIR` if it does not exist, copy the input file
there, and write all output there. The commands produce the following files:

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - File
     - What it is
   * - `input.yaml`
     - the original input file copied into the newly created working directory
       if requested by `-o DIR`
   * - `output.yaml`
     - the achieved design together with the configuration that produced it,
       which may differ from the input if `-s` or `iterate` were used
   * - `post.pdf`
     - post-processed figures
   * - `restart.npz`
     - CFD-predicted flow field, to start a new run from a converged solution
   * - `conv.cnv`
     - the CFD convergence history
   * - `log_turbigen.txt`
     - a transcript, same as the console output

`iterate` creates `iter_NNNN` directories, `chic` creates `chic_NNNN` and
`batch` creates `batch_NNNN`, one directory per iteration, operating point or
sample. Each is a complete case directory in its own right, so can be resolved
or use as an input for a new run.

Three notes on the output files:

* `output.yaml` is refused as an input file name, to prevent results being
  overwritten.
* After `iterate` or `chic`, the final run's `output.yaml`, `restart.npz`,
  `conv.cnv` and `post.pdf` are promoted to the top level working
  directory, whereas the intermediate iterations stay in their own directories.
* The full three-dimensional mesh is never written, only stored in memory.

turbigen
^^^^^^^^

.. argparse::
   :module: turbigen.cli
   :func: _make_parser
   :prog: turbigen
   :nosubcommands:

Every command below takes one or more configuration files, and accepts `--set`
and `--verbose`.

.. _usage-design:

design
^^^^^^

.. argparse::
   :module: turbigen.cli
   :func: _make_parser
   :prog: turbigen
   :path: design

.. _usage-report:

report
^^^^^^

.. argparse::
   :module: turbigen.cli
   :func: _make_parser
   :prog: turbigen
   :path: report

.. _usage-run:

run
^^^

.. argparse::
   :module: turbigen.cli
   :func: _make_parser
   :prog: turbigen
   :path: run

.. _usage-iterate:

iterate
^^^^^^^

.. argparse::
   :module: turbigen.cli
   :func: _make_parser
   :prog: turbigen
   :path: iterate

.. _usage-chic:

chic
^^^^

.. argparse::
   :module: turbigen.cli
   :func: _make_parser
   :prog: turbigen
   :path: chic

.. _usage-batch:

batch
^^^^^

.. argparse::
   :module: turbigen.cli
   :func: _make_parser
   :prog: turbigen
   :path: batch
