""" Large parameterized processing task, divided into smaller parts with different parameters

Each operation consists of a set of processes.
Processes need to be independent so
that potentially they can be executed concurrently on different machines. 
For instance, an operation might consist of comparing the performance of
a NodeChain on an EEG dataset for different parameter values of the NodeChain
nodes (based on 10 runs of ten-fold crossvalidation). Then, a process
would be to do one run of ten-fold crossvalidation of the NodeChain on the test
:mod:`dataset <pySPACE.resources_dataset_defs>`
for one specific parameter setting. An operation can take several
input datasets that need to contain some meta information.

Naming Conventions
------------------

The mapping between operations and the *type* parameter in their specification
file has been automatized.
For each operation there is a separate module with the *same name* as the *type*.
Furthermore, the type should not use any capital letters and be written with
underscores, if several subnames are required as in *node_chain*.
The corresponding class name of the operation should be the default
transformation form under_score to CamelCase. Here it is important to mention
that abbreviations are not written with capital letters.

Documentation Conventions
-------------------------

The user documentation of each operation has to be at the docstring of the
module. It is structured using sections and subsections.

It is important to define the sections **Specification file Parameters**
and **Exemplary Call**.
An exhaustive example can be found at:
:mod:`~pySPACE.missions.operations.concatenate`.

Programming your own Operation
------------------------------

Each operation module consists of an operation, named as mentioned above,
and a process.
The operation defines the mapping between the specification and
the elemental tasks, called processes.

Each operation inherits from :class:`~pySPACE.missions.operations.base.Operation`
and each process from :class:`~pySPACE.missions.operations.base.Process`.
The `__init__` method should always call its corresponding base part.
Processes should be generated, by using and defining a *classmethod* called:
`_createProcesses`.
Each operation has to define the *classmethod* `create` and the method
`consolidate`.
A process itself basically just needs to define its `__call__` and
`__init__` method. The latter is needed to define the parameter mappings,
since there is currently no standard definition of parameters of a process.
"""