.. _data_handling:

HOWTO: Data Handling
--------------------

When starting with pySPACE the first question, which often arises is:
"What do I have to do to get my data in this framework and
how can I get the right output?"
The documentation on this issue is distributed and so we will give an overview
with the needed links.

Internal data structures in pySPACE are defined in the
:mod:`~pySPACE.resources` package.
Here, we distinguish between samples and datasets.
Samples are grouped to datasets and datasets can be grouped to summaries.

.. note:: Summaries are no defined data structure in pySPACE,
          because they are nothing more than the address to a folder name
          with one subfolder for each contained dataset
          (as depicted in :ref:`storage`).

Data Handling for Benchmarking Applications
###########################################

The Summary
+++++++++++

The easiest way of getting your data into pySPACE
is to transform it to a summary.
When using the ``setup.py`` in the installation process,
an example summary will be saved on your hard disk.

For doing so, you first create a folder with the summary name in
your storage folder.

.. seealso::

    * :ref:`storage`
    * :ref:`conf`

.. note:: A summary should only include datasets of the same kind
          (e.g., streaming data or feature vector data) and no mixing.
          Otherwise the processing will not be compatible.

The Dataset
+++++++++++

As a next step you need at least one subfolder for the dataset you want to
process with pySPACE.
This folder then needs two components:

    * the data file(s),
    * and a file called ``metadata.yaml``, which contains all meta information
      including the type and storage format of the data.

The loading procedure of datasets is defined in the
:mod:`~pySPACE.resources.dataset_defs`.
For reading new data, there are three main types of dataset definitions for:

    * :mod:`stream data <pySPACE.resources.dataset_defs.stream>`,
    * :mod:`time_series data <pySPACE.resources.dataset_defs.time_series>`,
    * and :mod:`feature_vector data <pySPACE.resources.dataset_defs.feature_vector>`.

The type name is in the metadata file is written with underscore and directly
corresponds to the module name.
The respective class name is the same but written in camel case notation and with
``Dataset`` added to the name.
The first two types deliver samples of
:class:`times series objects <pySPACE.resources.data_types.time_series>`
and the last one
:class:`feature vector objects <pySPACE.resources.data_types.feature_vector>`.

As a next step, you need to check, if your data format is supported by the
corresponding dataset type.
Therefore, you should check the respective documentation.
All types support the csv format.
For feature vector data, the arff format is supported, too.
For streaming data BrainProducts eeg format, the EDF2 file format and
the EEGLAB set format are supported.
When processing streaming data with a
:mod:`node chain operation <pySPACE.missions.operations.node_chain>`
you will also need an additional windower file,
specifying how the data stream is segmented into time series objects.
This is documented in the respective source node.
If your storage format is supported, you just have to add the used
``storage_format`` parameter into your meta data file, as documented in
the dataset definition.

Case of not supported storage format
************************************

If your storage format is not supported, there are two possibilities.
You can use an external tool, which converts your data to a compatible format,
or you can integrate the code for loading your format into the
dataset definition.

Defining New Types of Data
##########################

If you cannot use the existing data types, extra effort is needed and it is
probably a good idea to search the discussion with the software developers.

    * A new :mod:`dataset definitions <pySPACE.resources.dataset_defs>`
      need to be implemented
    * and new :mod:`~pySPACE.resources.data_types` respectively.
    * A :mod:`~pySPACE.missions.nodes.source` node will be required
      for getting the data into a
      :mod:`node chain operation <pySPACE.missions.operations.node_chain>`
    * and a :mod:`~pySPACE.missions.nodes.sink` node
      for getting the data format out of a
      :mod:`node chain operation <pySPACE.missions.operations.node_chain>`.
    * :mod:`~pySPACE.missions.nodes` or
      :mod:`~pySPACE.missions.operations`.need to be implemented
      or modified, to process this data.

A good example would be to integrate picture or video data into the framework.
This data could be handled as feature vector or time series data, too, but
a special format might be a better choice.



Data Handling for Direct Processing in Applications
###################################################

For the application case, the aforementioned hard disc usage is infeasible and
data needs to be directly forwarded to the node chain which shall process
the data probably using :mod:`~pySPACE.run.launch_live`.
Therefore, an iterator is needed, which produces objects of pySPACE
:mod:`~pySPACE.resources.data_types`. To achieve this the
:class:`~pySPACE.missions.nodes.source.external_generator_source.ExternalGeneratorSourceNode`
is used.

For demonstration purposes this functionality is implemented in the
:mod:`~pySPACE.tools.live` tools. It contains a C++ based streaming software
which can access EEG acquisition devices manufactured by BrainProducts
(requires proprietary driver for Windows, see
:ref:`eegmanager tutorial<tutorial_work_with_the_eegserver>`).
The data is sent via TCP/IP and can be unpacked and formatted accordingly by the
:class:`client side <pySPACE.tools.live.eeg_stream.EEGClient>`. The received data is
then handed over to the windower inside the
:class:`~pySPACE.environments.live.eeg_stream_manager.LiveEegStreamManager`.
The created windows are then fed into the current node chain using the
:class:`~pySPACE.missions.nodes.source.external_generator_source.ExternalGeneratorSourceNode`.

To have your own data processed in pySPACE live you have to replace the
:class:`~pySPACE.environments.live.eeg_stream_manager.LiveEegStreamManager`
to fit to your custom
protocol or medium, your raw-data gets transmitted over. Currently this involves
replacing the use of this class by hand - but in future releases a
modular architecture is intended when handling different kinds of live-data.