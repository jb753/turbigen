.. _config:

Input file format
=================

Top-level keys
^^^^^^^^^^^^^^

Every :program:`turbigen` command takes an input YAML file describing a
turbomachine. The file holds one top-level key for each phase of the design
pipeline. Only ``fluid:`` and ``mean_line:`` are required; every other key is
optional and :program:`turbigen` will simply halt if it does not have the data
to proceed any further. Every key and subkey is described in full the
:ref:`config-reference` below. The following table lists which key is read by
which command.

.. list-table::
   :header-rows: 1
   :widths: 25 75
   :class: config-keys

   * - Key
     - Read by
   * - :ref:`fluid: <config-fluid>`, :ref:`mean_line: <config-mean_line>`,
       :ref:`annulus: <config-annulus>`, :ref:`blades: <config-blades>`
     - every command, listed in :ref:`usage`
   * - :ref:`mesh: <config-mesh>`, :ref:`solver: <config-solver>`,
       :ref:`operating_point: <config-operating_point>`,
       :ref:`inlet_profile: <config-inlet_profile>`
     - commands that run CFD: :ref:`run <usage-run>`,
       :ref:`iterate <usage-iterate>` and :ref:`chic <usage-chic>`
   * - :ref:`iterate: <config-iterate>`, :ref:`database: <config-database>`
     - iterative geometry updates in :ref:`iterate <usage-iterate>`
   * - :ref:`chic: <config-chic>`
     - running a characteristic in :ref:`chic <usage-chic>`
   * - :ref:`batch: <config-batch>`
     - configuration sweeping in :ref:`batch <usage-batch>`
   * - :ref:`post_process: <config-post_process>`,
       :ref:`metrics: <config-metrics>`
     - all commands excluding :ref:`design <usage-design>` and
       :ref:`batch <usage-batch>`
   * - :ref:`job: <config-job>`
     - any command given ``--queue``, described in :ref:`usage`

A key of the file is written with the trailing colon it carries there; a
command typed at the shell is not. That tells apart the three names that are
both, so ``iterate:`` is what the file holds and :ref:`iterate <usage-iterate>`
is what runs it.

Two more top-level keys are not phases of the design: ``include:``, described
below, and ``result:``, which a finished run writes. Under ``result:`` sit the
mixed-out mean line the CFD achieved, the error each iterator last measured,
and, when the config asks for them under :ref:`metrics: <config-metrics>`, any
quantities measured from the solved field --- a surface integral, a plane
average, a loss breakdown. Each :ref:`metrics: <config-metrics>` entry is
evaluated once the march is over and its value recorded under ``result:
metrics:``, so a run archived today can be mined later. Metric designs of your
own go in a ``turbigen_plugins`` directory, like any other.

Duplicate keys or subkeys are refused.

The whole file is checked early before anything is designed, so a typo will
show up at the earliest opportunity.

A typical configuration file looks something like this:

.. literalinclude:: ../examples/turbine_cascade.yaml
   :language: yaml


Includes
^^^^^^^^

A top-level ``include:`` names other files whose keys are merged underneath
this one's, so a site's ``job:`` settings or a standard ``solver:`` can be
written once and shared across many cases:

.. code-block:: yaml

   include:
     - common.yaml

   solver:
     n_step: 2000

The merging rules are: later beats earlier, the including file beats everything
it includes, and mappings merge exactly one level deep. So the ``solver:`` above
keeps everything `common.yaml` set and changes ``n_step:`` alone. Lists replace
whole, since merging ``blades:`` by index is never the row you meant.

File names are resolved relative to the file that names them, never to the
directory you ran from, so a case directory and its fragments can be copied
anywhere together. Two files in one ``include:`` list that both set the same
top-level key is an error.

Overriding values from the command line
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Any value in the file can be replaced from the command line with `-s`, which is
repeatable:

.. code-block:: console

   $ turbigen design input.yaml -s mean_line.psi=1.8
   $ turbigen design input.yaml -s mean_line.Ys=0.03 -s blades[0].count.Co=0.8

The value after the `=` is read as YAML, so lists and mappings work as well as
numbers. Keys are joined with dots, and entries in a list are indexed with
brackets --- `blades[0].sections[1].dchi_TE` --- or with a bare number between
dots, `blades.0.sections.1.dchi_TE`, if that is easier to quote in a shell. A
path names a key without giving it a value, so it carries no colon.

.. _config-reference:

Schema reference
^^^^^^^^^^^^^^^^

What follows is generated from the code that reads the file, so it lists every
key and what it may be written with. A key with a default may be left out, and
takes that default; a key with no default has to be written. A key that is a
link holds further keys of its own, listed in the section it points to.

Designs of your own, found in a `turbigen_plugins` directory, do not appear
here: this is what :program:`turbigen` ships with.

.. turbigen-config::
