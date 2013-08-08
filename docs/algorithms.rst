.. _missions:

Missions for the User
=====================

An overview on the different algorithms categories.

General
-------

.. currentmodule:: pySPACE
.. autosummary::

    missions
    missions.nodes
    missions.operations
    missions.support

Nodes
-----



.. autosummary::

    missions.nodes.source
    missions.nodes.preprocessing
    missions.nodes.spatial_filtering
    missions.nodes.spatial_filtering.sensor_selection
    missions.nodes.feature_generation
    missions.nodes.splitter
    missions.nodes.postprocessing
    missions.nodes.classification
    missions.nodes.meta
    missions.nodes.meta.parameter_optimization
    missions.nodes.sink
    missions.nodes.visualization

Together with the documentation generation, there automatically comes a
:ref:`list of all available nodes <node_list>` and corresponding name mappings.

Operations
----------

.. autosummary::

    missions.operations.node_chain
    missions.operations.weka_filter
    missions.operations.weka_classification
    missions.operations.mmlf
    missions.operations.merge
    missions.operations.shuffle
    missions.operations.concatenate
    missions.operations.statistic