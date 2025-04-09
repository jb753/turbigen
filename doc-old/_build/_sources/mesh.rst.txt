.. _meshing:

Meshing
=======

In order to run a computational fluid dynamics simulation, the domain must be
discretised into a 'mesh' of cells. `turbigen` supports structured grids
in polar coordinates of any multi-block topology.

H-meshing
---------

The simplest topology is an H-mesh. Each blade row is a single block, indexed
in the streamwise, spanwise, and pitchwise directions. No special setup is
required for H-meshing as it is performed using native :program:`turbigen` code.

Configuration options
^^^^^^^^^^^^^^^^^^^^^

.. automodule:: turbigen.hmesh
   :members:

.. _oh-meshing:

OH-meshing
----------

Placing an O-mesh around the blade gives better resolution of the boundary
layers, and removes H-mesh grid distortions at the leading edge and with highly
staggered blades. Additional H blocks placed around the O-mesh discretise the
remainder of the fluid domain, yielding an OH-mesh for the entire passage.

Setting up
^^^^^^^^^^

:program:`turbigen` accomplishes OH-meshing by scripting the commercial
software :program:`AutoGrid`. In general, the machine we want to run the CFD on is not the same one as for the meshing. So we need to start a server process on a remote
machine with access to an :program:`AutoGrid` license. On the *machine with
AutoGrid*, install :program:`turbigen` and run:

.. code-block:: console

    $ turbigen-autogrid-server

The server will stay running indefinitely, waiting for jobs. If the server
finds a job it will perform the meshing. Meanwhile, the turbigen process on the
HPC will wait for the meshing to finish and copy back the results for
preprocessing.

We will be connecting to the AutoGrid machine frequently. So, on the *CFD machine*, add an entry in `~/.ssh/config` to allow connection
reuse:

.. code-block:: console

    Host autogrid-box
        ControlMaster auto
        ControlPath ~/.ssh/control-%C
        ControlPersist yes
        ServerAliveInterval 240

where `autogrid-box` is the hostname of the AutoGrid machine. If you do not
have SSH keys set up for password-free access to the AutoGrid machine, you will
have to do that first by, *on the CFD machine*, running some combination of:

.. code-block:: console

    # Make an ssh private/public key pair
    $ ssh-keygen

    # Copy to the remote machine
    $ ssh-copy-id autogrid-box

    # Load keys into agent to save your passphrase
    $ eval `ssh-agent` && ssh-add

Before moving on, check you can connect from the CFD machine to the AutoGrid
machine without being prompted for a password by running,

.. code-block:: console

    $ ssh autogrid-box

Configuration options
^^^^^^^^^^^^^^^^^^^^^

.. automodule:: turbigen.ohmesh
   :members:
