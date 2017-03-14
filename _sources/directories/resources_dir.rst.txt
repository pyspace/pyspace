.. _storage:

The Data Directory (storage)
----------------------------

The data (summaries and datasets, input and output) are stored in the
``storage``.
Summaries used as input for operations can be located in arbitrary subdirectories
of the directory.
Results of operation chains are stored in subdirectories of the directory
"operation_chain_results", results of operations in subdirectories of  "operation_results".
These subdirectories are named based on the execution time. 

.. seealso::

    * :ref:`data_handling`
    * :ref:`conf`

.. _operation_chain_spec_example:

An OperationChain Specification File
""""""""""""""""""""""""""""""""""""

When executing an operation chain as in the following example or an operation
you get a result, which we will explain in the following.

.. literalinclude:: ../examples/specs/operation_chains/example.yaml
    :language: yaml

Structure of the Storage Directory
"""""""""""""""""""""""""""""""""""""

Lets say, we have only a single directory "example_data"
in the storage directory containing one or a summary of datasets
(each in an extra folder).
Now we execute the operation chain specified
in the operation chain specification file given
:ref:`above <operation_chain_spec_example>`. When we start the operation chain,
the subdirectory "operation_chain_results" is created
(if not already existing)
and in this directory
a further subdirectory to which all results of the operation chain are written.
If we start the operation chain at 11:35:02pm at 2009-07-22, this subdirectory
would be called "2009_7_22_11_35_2". 
The operation chain would immediately start the first operation
and a subdirectory for the results  of this operation would be created,
lets say 2009_7_22_11_35_3. 
The directory structure would at that moment be as follows:

.. code-block:: guess

  storage
                  /example_data
                  /operation_chain_results
                                   /2009_7_22_11_35_2
                                                     /2009_7_22_11_35_3

Once the first operation of the operation chain is finished,
the result summary (consisting of several datasets) of this operation
is written to ``storage\operation_chain_results\2009_7_22_11_35_2\2009_7_22_11_35_3``.
Lets say the the result summary consists of three result datasets named
dataset_1, dataset_2, and dataset_3.
Then the directory structure would look as follows:

.. code-block:: guess

  storage
                  /example_data
                  /operation_chain_results
                                   /2009_7_22_11_35_2
                                                     /2009_7_22_11_35_3
                                                                       /dataset_1
                                                                                     /metadata.yaml
                                                                                     /input_metadata.yaml
                                                                                     ...
                                                                       /dataset_2
                                                                                     ...
                                                                       /dataset_3
                                                                                     ...
                                                                       /operation.log
                                                                       /source_operation.yaml
                

Each result dataset stores its meta data in a file named "metadata.yaml"
and the meta data of the dataset on which it is based in a file named
"input_metadata.yaml" The operations stores its specification file
it is based on in a file named "source_operation.yaml".

The second operation of the operation chain would use the data of the result summary
"2009_7_22_11_35_2/2009_7_22_11_35_3" as input and create a directory 
for its own result, lets say "2009_7_22_11_40_17".
After its completion, it would also write the result to this directory.
After completion of this operation, the operation chain would be finished
and the directory structure might look as follows:

.. code-block:: guess

    storage
        /example_data
        /operation_chain_results
            /2009_7_22_11_35_2
                /2009_7_22_11_35_3
                    /dataset_1
                        /metadata.yaml
                        /input_metadata.yaml
                        ...
                    ...
                /2009_7_22_11_40_17
                    /dataset_1
                        /metadata.yaml
                        /input_metadata.yaml
                        ...
                    ...

If we leave the resources directory unchanged and execute an arbitrary operation,
the resources directory might afterwards look as follows:

.. code-block:: guess

    storage
            /example_data
            /operation_chain_results
                    /2009_7_22_11_35_2
                            /2009_7_22_11_35_3
                                    ...
                            /2009_7_22_11_40_17
                                    ...
            /operation_results
                    /2009_7_22_11_52_12_7
                            /dataset_1
                                    ...
                            /dataset_2
                                    ...
                            ...
                            /operation.log
                            /source_operation.yaml


