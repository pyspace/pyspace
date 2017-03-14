.. _tutorial_node_chain_operation:

Process EEG data - Usage of node chains
----------------------------------------------------

In this tutorial we process electroencephalographic (EEG) data.

In this tutorial we use EEG data in the format of Brain Products recording software, 
that was recorded during EEG experiments.
Our goal is the statistical analysis of the data 
to find out which methods works good.

Specifically we want to detect a movement preparation 
(or lateralised readiness potential, LRP) in the data. 

Windowing
^^^^^^^^^
A good idea and the basic step is to "window" the prerecorded data in a preparation step. 
This means that we cut out slices of the EEG data stream 
which can be used more easily in further processing steps. 
The slices are given by markers that are contained in the given datasets. 
Usually they are added to the data during the recording process 
to mark important sections, e.g. the occurrence of designated events 
during the experiments.

To use the automatic windowing of the data,
we have to provide it with some information. 

The data should be placed in the collections directory, see :ref:`storage` .

Furthermore, we need the spec and the windower file.
For the windowing, we will use a *operation*.
Another opportunity would be to use an *operation chain*.
See :ref:`processing` for further information about this topic.

Here is an example of the example_lrp_windower.yaml operation spec file 
(placed in ``docs/examples/specs/operations`` or in the your pySPACEcenter
in the folder ``specs/operations``, the designated directory for operations),
which we will use for the windowing:

.. literalinclude:: ../examples/specs/operations/example_lrp_windower.yaml
    :language: yaml

Spec files specify what should be done and which data should be used.

.. code-block:: yaml

    type: node_chain
    node_chain_templates : ["example_lrp_windower.yaml"]
 
means, that we would like to use a node chain that is defined
in the file example_lrp_windower.yaml. Note this:

.. code-block:: yaml

    parameter_ranges:
        __DTvsST__ : [Noop, Detrending, Standardization]

That means, that we test all these different methods on the data.

For further information about spec files see :ref:`specs_dir`.

The part

.. code-block:: yaml

    input_path: "eeg/LRP_Demo"

    runs : 1

specifies which dataset we want to use and that we want to do a single run.

You can find the specification in the file 
``docs/examples/specs/node_chains/example_offline_windower.yaml``, as stated above:

.. literalinclude:: ../examples/specs/node_chains/example_lrp_windower.yaml
    :language: yaml

Furthermore, for the windower we need a windower specification file as stated in:

.. code-block:: yaml

    - 
        node: EEG_Source
        parameters : 
            windower_spec_file : "example_lrp_window_spec.yaml"

Here you can see the respective window spec file:

.. literalinclude:: ../examples/specs/node_chains/windower/example_lrp_window_spec.yaml
    :language: yaml

The windower spec file describes which slices of the data (aka *windows*) 
are important and should be cut out.
The specifications of this file are normally forwarded to
:class:`~pySPACE.missions.support.windower.MarkerWindower`,
which is finally doing the segmentation of the data.
Check out the corresponding documentation to find out more about the
parameters of the windower file.

In the *window_defs* section, the respective windows are specified.

Hence we want to detect movement *preparation* the data has to have some
markers related to executed movements. In this example the marker "S 16" means that
there has been some (e.g. physical) movement detected. The preparation
happened before the movement so the resulting window is defined like this:

.. code-block:: yaml

     s16: 
         classname : LRP
         markername : "S 16"
         startoffsetms : -1280
         endoffsetms : 0
         jitter : 0
         excludedefs : []

means, that a window should be cut out 
where we have the marker with name "S 16" in the data, 
beginning at 1280ms *before* the marker and ending at the marker position (0ms). 

The markernames can be found in the *.vmrk* file of the dataset.

Finally, you can start the operation by invoking::

    python launch.py --mcore --configuration your_configuration_file.yaml --operation example_lrp_windower.yaml

For the details on this command, see: :ref:`CLI`.

Processing the data
^^^^^^^^^^^^^^^^^^^
After the windowing operation, we can evaluate different methods
and parameters for the analysis of the data.
We do that by applying an *operation chain*.

Below is an example operation chain file:

.. literalinclude:: ../examples/specs/operation_chains/example_lrp_detection.yaml
    :language: yaml

This file references three other *operation* files.
Very important for our case is the section :

.. code-block:: yaml

    - 
        example_lrp_detection.yaml


This references the lrp specific node chain,
which contains the lrp specific preprocessing etc.

This file is shown here:
 
.. literalinclude:: ../examples/specs/operations/example_lrp_detection.yaml
    :language: yaml

In this case, it simply references a node chain, which is shown below:

.. literalinclude:: ../examples/specs/node_chains/example_lrp_detection.yaml
    :language: yaml

One more important thing is the *example_lrp_libsvm*:

.. literalinclude:: ../examples/specs/operations/weka_classification/example_lrp_libsvm.yaml
    :language: yaml

You can see, that we evaluate different complexities and weights.

You can start by invoking::

    python launch.py --mcore --configuration your_configuration_file.yaml --operation example_offline_windower.yaml

Hopefully, you will see something like::

    Running operation example_lrp_detection.yaml of the operation chain (1/3)
    Operation progress: 100% |######################################| Time: 00:05:49
    Running operation weka_classification/default_libsvm.yaml of the operation chain (2/3)
    Operation progress: 100% |######################################| Time: 00:22:32
    Running operation analysis.yaml of the operation chain (3/3)

The results are contained in the operation_chain_results directory of the collections directory.


