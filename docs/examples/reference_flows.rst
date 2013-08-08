.. _reference_flows:

Reference node chains for EEG processing
----------------------------------------

In order to have a common basis for computation, certain reference processing chains
have been created. You can find the
corresponding files in the documentation under */examples/specs/node_chains/*.
The revision and improvement of the node chains listed below is part of the
ongoing research. The documentation is updated automatically.


P300 Reference Node Chain
^^^^^^^^^^^^^^^^^^^^^^^^^

This is the node chain for the P300 detection:

.. literalinclude:: specs/node_chains/ref_P300_flow.yaml
    :language: yaml


LRP Reference Node Chain
^^^^^^^^^^^^^^^^^^^^^^^^

This is the node chain for LRP detection:

.. literalinclude:: specs/node_chains/ref_LRP_flow.yaml
    :language: yaml