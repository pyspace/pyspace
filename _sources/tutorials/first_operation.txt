.. _first_operation:

Processing Benchmark Data - A First Usage Example
----------------------------------------------------

In this tutorial, we  process the example data, which comes with the software.
Our first step is to get some performance results and the next step is
to compare some algorithms.

Before we start
^^^^^^^^^^^^^^^

First of all you need to :ref:`download` and :ref:`install<t_install>`
the software including the run of ``setup.py`` such that we can assume,
that you have the default configuration in ``~/pySPACEcenter/``.

Resources
^^^^^^^^^

We are going to process some simple example benchmark data.
You can have a look at it in ``~/pySPACEcenter/storage/example_summary/``.
When browsing through the data, you may have a look at one metadata file,
e.g.:

.. literalinclude:: ../examples/storage/example_summary/Titanic/metadata.yaml
    :language: yaml

This data already comes with the whole structure for processing with pySPACE.

First Processing
^^^^^^^^^^^^^^^^

After having a look at the data, we now want to apply a classification
algorithm. This is done by applying an *operation*:

.. literalinclude:: ../examples/specs/operations/examples/classification.yaml
    :language: yaml


You can start it in the command line directly in the pySPACE center by
invoking::

    python launch.py --operation examples/classification.yaml

Alternatively, you could change your current directory to ``pySPACE/run``
beforehand if you did not use the setup.py to create the required links.
Now you should get some information on your setup and finally a progress bar.
The result can now be found at
``~/pySPACEcenter/storage/operation_results/CURRENT_DATE_TIME``,
where the time tag in the folder name corresponds to the start time of
your algorithm.
You may have a look at the ``short_result.csv``.
If you want to browse the result tabular,
start :mod:`~pySPACE.run.gui.performance_results_analysis`.

For having a faster execution using all cores of your PC,
simply change the command to::

    python launch.py --mcore --operation examples/classification.yaml

If you now want to compare different algorithms you can execute the
following operation:

.. literalinclude:: ../examples/specs/operations/examples/bench.yaml
    :language: yaml

This is done with the command::

    python launch.py --mcore --operation examples/bench.yaml

When now browsing through the results as described above,
you see a lot of more possible parameters.

Which parameter combination was the best?
Complexity of 0.1 and GausianFeatureNormalization?

Using now :ref:`node_list` you can check the available algorithms and their
parameters and play around with the specification files and change it.
They should be found at ``~/pySPACEcenter/specs/operations/examples/``.

