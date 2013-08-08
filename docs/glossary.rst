.. _glossary:

Glossary
========

.. glossary::
    :sorted:

    pySPACE
        Name of the Software. **SPACE** stands for **Signal Processing And
        Classification Environment** and the **py** comes from
        Python, parallelization and YAML

    dataset
        Set of data of one type of one recordings in a fixed processing stage.
        Formerly called *collection*.

    summary
        Set of datasets of same type with some common meaning.
        It is described with the directory path, where datasets and
        additional information are stored.
        In contrast to other namings of aggregation, summary normally asks for
        some compression or reduction of information.
        This can be here seen in the aspect of having only folder names or
        links to folders and that a summary is just a chosen subset of the
        total amoutn of available data.
        A special summary is the
        :class:`~pySPACE.resources.dataset_defs.performance_result.PerformanceResultSummary`,
        which can largely
        compress the result to one exemplary folder, a zip file, few metadata,
        and most importantly a short and a complete summary of all performance
        results in a csv tabular.
        Formerly called *bundle*.

    node
        Elementary signal processing unit

    node chain
        Concatenation of nodes to define a linear processing chain.
        Formerly called *flow*. This came from the usage in MDP.
        *Flow* is a very general description for any data processing.
        In pySPACE also *backends*, *operations*, *operation chains*
        and even *nodes* handle data flows.
        A *node chain* can be seen as a special data flow.
        It specifies several processing steps of the data.

    operation
        Large parameterized processing task, which
        is divided into smaller parts with different parameters.

    operation chain
        Concatenation of operations to be processed consecutively.
        Formerly called *campaign*.

    process
        Independent part of an operation, to be processed in parallel
        with all processes in the operation.

    backend
        Distributor of processes.

    channel
        Identifier of one sensor.
        The name comes from EEG processing but can be simply used like *sensor*.

    csv
        There are three usages of comma separated values files. A
        :class:`~pySPACE.resources.dataset_defs.performance_result.PerformanceResultSummary`
        is a csv-table containing all performance results of your study.
        Furthermore, each line can correspond to
        a time point in a
        :class:`~pySPACE.resources.dataset_defs.stream.StreamDataset`,
        which is emitting
        :class:`~pySPACE.resources.data_types.time_series.TimeSeries` objects
        or a
        :class:`~pySPACE.resources.data_types.feature_vector.FeatureVector`
        in a
        :class:`~pySPACE.resources.dataset_defs.feature_vector.FeatureVectorDataset`.
        For creating the needed ``metadata.yaml`` for the latter,
        you can use the script:
        :mod:`pySPACE.run.scipts.md_creator`.

        Loading, storing and manipulation
        of csv-files is supported by the software.

    YAML
        (YAML Ain't Markup Language)
        Intuitive format for writing specification files. It is the basis.
        Further reading: :ref:`YAML<yaml>`.