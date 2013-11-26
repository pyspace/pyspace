.. _tutorials:

Tutorials
--------------------------------

.. toctree::
    :maxdepth: 2

    tutorial_git
    tutorial_installation
    yaml.rst
    first_operation
    tutorial_interface_weka
    tutorial_interface_to_mmlf
    tutorial_analysis_gui
    tutorial_node_chain_operation
    tutorial_node_chain
    tutorial_node_chain_online.rst
    tutorial_work_with_the_eegserver.rst
    data_handling

.. _node_chain:

Processing with node chains: Tutorials and HowTos
+++++++++++++++++++++++++++++++++++++++++++++++++

The main way of starting node chains is with
:mod:`~pySPACE.run.launch`.
This is done by the
:class:`~pySPACE.missions.operations.node_chain.NodeChainOperation`.
As a starting point you should
have a look at the :ref:`node chain tutorial<tutorial_node_chain>`,
which tells you what you need at least to process some data.
To get a good understanding of the structure of the configuration files in the software,
have a look at the
:ref:`YAML description<yaml>`.

The usual way to run the node chains is to use it via pySPACE. To get
an idea how this works, you should take a look
at the :ref:`NodeChainOperation tutorial<tutorial_node_chain_operation>`.
The configuration in either case is based on a configuration file (in the
:ref:`YAML <yaml>` format) specifying which nodes shall be processed in which order and
with which parameters. For starting the software with pySPACE
you will also need a specification file
for the corresponding :class:`~pySPACE.missions.operations.node_chain.NodeChainOperation`.

After going through the aforementioned tutorials, you roughly know how to deal with data
that has already been recorded, the so called **offline data processing**.
Nevertheless, there is the live part,
that was specifically developed for **online data processing**,
which is done using the corresponding
:mod:`scripts <pySPACE.run.scripts.node_chain_scripts>`
or :mod:`~pySPACE.run.launch_live`.
