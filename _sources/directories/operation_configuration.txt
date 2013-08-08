.. _spec:

Specification Files
-------------------

All specification files rely on the usage of :ref:`YAML<yaml>`.
Here some examples are given.

.. _operation_chain_spec:

Operation Chains
################

For defining an operation chain you need to specify with :ref:`YAML<yaml>`
a dictionary with three entries:

    :input_path:  path to the summary of input datasets relative to the :ref:`storage`

    :runs:          number of repetitions for each operation to handle random effects
                    
                    (*optional, default: 1*)
                    
    :operations:    list of operation specification files, being executed in the
                    given order
                    
                    The specification files should be in the ``operations`` subfolder
                    of your specs folder.

Example of an operation chain specification file
""""""""""""""""""""""""""""""""""""""""""""""""

.. literalinclude:: ../examples/specs/operation_chains/example.yaml
    :language: yaml


.. _operation_spec:

Operations
##########

For defining an :mod:`operation<pySPACE.missions.operations>` you need to
specify a dictionary with :ref:`YAML<yaml>` syntax with these entries:

    :type:          name of the operation type you want to use 
                    (e.g. node_chain, merge, mmlf, statistic, shuffle)
    :input:         path to the summary of input datasets relative to the data folder

                    This parameter is irrelevant for 
                    :mod:`operations<pySPACE.missions.operation>` 
                    in an operation chain.

    :backend:       overwrites the default :mod:`backend<pySPACE.environments.backends>`
                    of the command line call (e.g. "mcore", "serial")

    :runs:          number of repetitions for this operation to handle random effects

                    This parameter is irrelevant for operations in an operation chain.

                    (*optional, default: 1*)

    :. . .:         Each operation has its own additional parameters.
                    For the details have a look at the documentation of your
                    specific :mod:`operation type<pySPACE.missions.operations>`
                    or at the corresponding example file in the pySPACE
                    specification folder in the subfolder ``operations``.


Examples of operation specification files
"""""""""""""""""""""""""""""""""""""""""

Example of a :class:`~pySPACE.missions.operations.node_chain.NodeChainOperation`:

.. literalinclude:: ../examples/specs/operations/examples/node_chain.yaml
    :language: yaml

Example of a :mod:`weka classification operation<pySPACE.missions.operations.weka_classification>`:

.. literalinclude:: ../examples/specs/operations/examples/weka_classification_operation.yaml
    :language: yaml

.. _conf:

The Configuration File
######################

The pySPACE configuration file is mainly used in the
:ref:`command-line interface<getting_started>`.

Here all the relevant general parameters for the execution of 
pySPACE are specified.
The most important parameters are the ``storage`` and the ``spec_dir``.
If you want to debug you may want to change the logging levels.
Note, that when using the command-line interface, it is good to activate
the default serial backend.

To find out all the defaults and possibilities have a look at the
the default configuration file:

.. literalinclude:: ../examples/conf/example.yaml
    :language: yaml

After using ``python setup.py`` this file should be located at:
``~/pySPACEcenter/config.yaml``.

.. note:: By default, the configuration file is searched for in the folder
          ``~/pySPACEcenter/``,
          but you can manually specify the location of the configuration
          directory in your bash or the bash configuration file
          (bash-profile, bashrc, ...) using

           .. code-block:: bash

                export PYSPACE_CONF_DIR=<myconfdir>

          In your IDE you would have to add this variable to your
          environment variables.