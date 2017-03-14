.. _specs_dir:

The Specs Directory
-------------------

The pySPACE specs directory contains all kinds of specifications for pySPACE.
For instance it contains operation_chains and operation specification files and
node chain specification files.
The structure of the specs directory is as follows:

.. code-block:: guess

  Specs_Dir
               /node_chains
               /operation_chains
               /operations

The default ``$Specs_Dir`` can be found at ``~/pySPACEcenter/specs/``.
If one starts pySPACE with ``python launch.py --operation_chain example.yaml ...``,
the operation_chain specification file 'example.yaml' is looked up in $Specs_Dir/operation_chains.
Similarly, ``python launch.py --operation example_operation.yaml ...``
will look for the file ``example_operation.yaml`` in ``$Specs_Dir/operations``.
The directory ``node_chains`` contains specifications of data processing flows
in the form of a concatenation of nodes
that can be used within an operation  of type
:class:`node_chain <pySPACE.missions.operations.node_chain.NodeChainOperation>`.
For instance, the node_chain used in the
:class:`~pySPACE.missions.operations.node_chain.NodeChainOperation`
found at ``$Specs_Dir/operations/examples/node_chain.yaml``:

.. literalinclude:: ../examples/specs/operations/examples/node_chain.yaml
    :language: yaml

might look as follows:

.. literalinclude:: ../examples/specs/node_chains/example_flow.yaml
    :language: yaml

:ref:`Operation chain <operation_chain_spec>`, :ref:`operation<operation_spec>`
and :ref:`node chain <yaml_node_chain>` specifications are described in extra sections.

The directory ``weka_templates`` contains templates for command line calls of Weka.
These templates are parametrized and the respective instantiation
of the template depends on the parameters of the operation. 
For instance, the template for Weka classification operations look as follows:

.. literalinclude:: ../examples/specs/operations/weka_templates/classification
    :language: yaml

The operation sets most of the parameters automatically, 
for instance %(run_number)s is automatically replaces 
with the respective run_number. 
%(classifier)s is replaced by the respective classifier that should be used, 
for instance '-W weka.classifiers.bayes.NaiveBayes' for the Naive Bayes classifier. 
An operation  specification file for an operation
using such a Weka template might look as follows:

.. literalinclude:: ../examples/specs/operations/weka_classification/default_classification.yaml
    :language: yaml

Here, "template: classification" controls which weka template is used, 
"classifier: 'naive_bayes'" determines the classifier that is inserted into 
the templates parameter %(classifier)s, and "ir_class_index" controls 
on which class information retrieval metrics like precision and recall are based.

The mapping from the name 'naive_bayes' to actual text '-W weka.classifiers.bayes.NaiveBayes' 
that is inserted into the template is handled by the file "abbreviations.yaml". 
The following abbreviations for Weka are currently defined:

.. literalinclude:: ../examples/specs/operations/weka_templates/abbreviations.yaml
    :language: yaml

The usage of the ``windower`` folder is described in:

    - :ref:`tutorial_node_chain_operation` and
    - :ref:`tutorial_node_chain_online`.

It specifies one connection between :mod:`datasets <pySPACE.resources.dataset_def>`
and :class:`~pySPACE.missions.operations.node_chain.NodeChainOperation`.

