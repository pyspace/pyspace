.. _getting_started:

Getting started
---------------

Before getting started, you should always follow the instructions in:
:ref:`t_install`.

For the execution of pySPACE it is helpful to take a look at the following questions:

 0. How do I specify configuration files?
    The software comes with a lot of examples but before diving into,
    your should read :ref:`yaml` thoroughly.
    The standard interfacing is with the help of this configuration files,
    which come with an intuitive format and avoid complicated Python
    programming. Only for advanced usage and for implementing or modifying
    algorithms, this is needed.

 1. Do I have to modify the :ref:`configuration file<conf>`? A standard configuration
    which should work for most cases
    is provided during :ref:`t_install` in the pySPACEcenter. Take a look if you need to modify, e.g.,
    storage, spec_dir or what is appended to your PYTHONPATH during execution.

 2. Take a look at your pySPACEcenter. You already find a suggestion for your data organization
    and some examples there.

 3. Does the storage provide the necessary data? The storage (i.e., the place for input and output
    of pySPACE) is by default located in your pySPACEcenter.

 4. Which :mod:`backend<pySPACE.environments.backends>` will I use? The most common are the *serial* and the *mcore* (=multicore) backend.
    By default, the :class:`~pySPACE.environments.backends.serial.SerialBackend` is used.

 5. Did I specify the :ref:`operation specification file <operation_spec>` 
    or :ref:`operation chain specification file <operation_chain_spec>`
    that should be used? For an :mod:`operation<pySPACE.missions.operations>` 
    you need to specify a yaml-file. In case you want to execute a 
    :mod:`node chain operation<pySPACE.missions.operations.node_chain>`, you might want to
    specify the node chain in a separate yaml-file (which is only optional). For more information
    please take a look at the :ref:`example_specs`, some of them you can also find in your pySPACEcenter.


How do I then execute pySPACE? Mostly this is done using the command-line interface as described below.
A GUI frontend is currently under construction, you may already have a look at it in the :mod:`run <pySPACE.run>`
module.
An alternative mode of execution is the :mod:`launch live<pySPACE.run.launch_live>` mode which is not a good 
starting point with pySPACE, and should be considered when you feel confident in using the framework and want to
process your data online.

.. _CLI:

Command-Line Interface
^^^^^^^^^^^^^^^^^^^^^^

The software can be started from the command line via the script :mod:`launch.py<pySPACE.run.launch>`
in ``pySPACE/run/``. You'll also find a link to this file in your pySPACEcenter which you can use for execution, too.

So, for starting you have to run ``python launch.py`` and make sure that you are in the ``pySPACE/run/`` folder or
are executing a link which points to this file. Before doing this, you can have a look at the options by typing:

.. code-block:: bash

   python launch.py --help
   
You will see that there are a lot of options and usually you have to chose some of these. The main options are:
 * the :mod:`backend<pySPACE.environments.backends>`:

        :``--serial``: serial execution with the
                       :class:`~pySPACE.environments.backends.serial.SerialBackend`
        :``--mcore``:  use all cores of a PC with the
                       :class:`~pySPACE.environments.backends.multicore.MulticoreBackend`
        :``--mpi``:    distribute jobs on cluster with the
                       :class:`~pySPACE.environments.backends.mpi_backend.MpiBackend`
        :``--loadl``:  Submit jobs to the IBM LoadLeveler client with the
                       :class:`~pySPACE.environments.backends.ll_backend.LoadLevelerBackend`

 * the operation ``-o operation_file_name`` (or operation_chain ``--operation_chain chain_file_name``)
 * and: if you are not using the ``config.yaml``, the config-file:
   ``-c my_conf.yaml``

So a proper call of :mod:`launch.py<pySPACE.run.launch>` would be for instance:

.. code-block:: bash

   python launch.py --serial -o my_op.yaml

Or with a config file:

.. code-block:: bash

   python launch.py --serial -c my_conf.yaml -o my_op.yaml

The Role of the Configuration File
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As an alternative you can always specify your sources in the later
mentioned main :ref:`configuration file<conf>`,
and then you should be able to run the software from everywhere.

To get an idea of all the possible configuration parameters and their effects,
have a look at: :ref:`conf`.

.. note:: You can manually specify the location of the configuration directory
    in your bash file using

    .. code-block:: bash

        export PYSPACE_CONF_DIR=<myconfdir>

So, let us see what happens during execution:

.. code-block:: bash

   python launch.py --mcore --config user.yaml --operation_chain example.yaml

uses the :class:`~pySPACE.environments.backends.multicore.MulticoreBackend` with the
configuration specified in ``PYSPACE_CONF_DIR/user.yaml`` and starts
the operation chain specified in ``operation_chains/example.yaml``, where the path is
relative to the specification folder, which is set in the configuration file.

As outlined in the chapter  :ref:`storage`,
the results of an operation are written to 
``$storage/operation_results/TIME_STAMP``,
and the results of a operation chain to ``storage/operation_chain_results/TIME_STAMP``,
where  ``$storage/`` is the directory specified in your configuration file
and ``TIME_STAMP`` is a time stamp of the time when the operation was started. 
In the case of an operation, the result is directly written to this directory.
In the case of a operation chain, all intermediate results
(i.e. outputs of an operation that act as input for the subsequent operation) 
and the final result are stored into subdirectories of this directory,
which are again time stamped.


Usage Within Python
^^^^^^^^^^^^^^^^^^^

The software can also be used directly from within other Python applications.
The same operation chain as called from the command line in the subsection before
could be executed by the following sequence of Python statements::

  import pySPACE
  # Load configuration from file "user.yaml"
  pySPACE.load_configuration("user.yaml")
  # Create a MulticoreBackend
  backend = pySPACE.create_backend("mcore")
  # Create operation chain object for operation chain specified in "example.yaml"
  operation_chain = pySPACE.create_operation_chain("example.yaml")
  # Run the create operation chain on the backend object
  pySPACE.run_operation_chain(backend, operation_chain)

More information concerning the interface are available
in the :ref:`API documentation <api>`.

.. note:: It is recommended to use the command line interface.