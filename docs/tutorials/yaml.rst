.. _yaml:

Using YAML Syntax in pySPACE
------------------------------------


This chapter will give an overview on YAML ("YAML Ain't Markup Language")
and it's usage in the framework.
The focus will be on a general description
and the use in node chain specification files,
but there will also be hints to
configuration and  other specification files.
A standard example of a YAML specification file for a node chain
can be found at the end of this tutorial.

General Information on YAML
###########################

A complete description on YAML 
in general can be found at: http://www.yaml.org,
where you should have a look at `the specification website <http://www.yaml.org/spec/1.2/spec.html>`_.
There you can find the general description:
`YAML™ (rhymes with “camel”) is a human-friendly, cross language, 
Unicode based data serialization language designed 
around the common native data types of agile programming languages. 
It is broadly useful for programming needs 
ranging from configuration files to Internet messaging 
to object persistence to data auditing.`

The usage of comments in YAML is the same as in Python with a hash ("#")
at the beginning of each comment line.
For the entries in your YAML file you have always to keep in mind,
that if you do not define the type (called *untagged* in YAML framework),
the YAML loading program tries to detect it automatically
and if it can not detect it, the entry will be tagged as a string.
Strings can also be defined by using double or normal quotation marks.
YAML can detect for example:

    * integers (e.g. ``42``),
    * floats (e.g. ``42.42``),
    * null (``None`` in Python, Syntax: use ``Null``, ``NULL`` or nothing),
    * string (e.g. ``"fourtytwo"`` or ``'fourtytwo'``),
    * lists (e.g. node chain specification file),
    * dictionaries (e.g. operation specification file),
    * booleans (``True`` and ``False``),
    * and much more, which is yet not relevant for this software framework.

For the framework it is mainly important to know, how to define something as a string.
If you are further interested in explicit tagging and new data types,
just check the `the specification website <http://www.yaml.org/spec/1.2/spec.html>`_.

.. warning:: YAML does not support all float formats. So check beforehand, if
             you try special formats. Python also has problems with special
             formats

Application in the Framework
############################

YAML is mainly used in this framework,
because it

    * delivers a unified way of defining configuration and specification files,
    * is easy to read,
    * can be easily loaded into Python, and
    * the loading already detects standard data types even
      if they are used in nested structures, like dictionaries of lists.

In pySPACE configuration files, in operation specification files
and for dataset descriptions
the main structure is a *dictionary* (mapping),
which is simply defined by ``key: value`` pairs at each line.
The empty space after the colon is important.
Additional empty spaces between key and colon can be used for cosmetic reasons.
If the value is another substructure 
you may use a new line instead of the empty space,
but then you have to indent everything belonging to the definition of the value,
to show, that it belongs to this key.
For the definition of dictionaries you can also use braces (``{key1: value1, key2: value2}``).
Some of the values may be *lists*. They can be defined in two ways:

    * use a dash followed by a space ("- ") or new line with indentation or
    * use squared brackets (``[item1, item2]``) as in Python.
    
Single-line versions are mainly used if you have
lists of lists or dictionaries of dictionaries, but they do not have to be used.
These data types are called *collections* in YAML.

.. todo:: Documentation on datasets needed!

.. _eval:

Eval(uate) syntax for Python code to ease specification file writing
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Finally, it is important to mention that we made two small syntax modifications.

When loading YAML specification files with specification of
:class:`~pySPACE.missions.nodes`
you may want to use Python commands to define your parameter values.
Therefore, a string of the form ``eval(command)`` results in a new value,
which is the result of the command, and which is then send to the
initialization of the node.
Normally, this is only applicable to the values of the keys in the parameters
dictionary of the node specification.
This syntax can be even used to specify the node name.

The same syntax possibility is included in the framework,
when using the key *parameter_ranges* in :mod:`operations <pySPACE.missions.operations>`.
Here you can use for example ``eval(range(10))`` instead of
``[0,1,3,4,5,6,7,8,9,10]``.

.. note:: In contrast to the the normal Python syntax, the command must not be
          marked as string.

.. _yaml_node_chain:

Usage of YAML in node chain specification files
+++++++++++++++++++++++++++++++++++++++++++++++

Since a signal processing node chain is mainly
a linear concatenation of elemental processing steps (called *nodes*),
the corresponding specification file is a list,
specifying the elemental processing steps in the form of nodes.

A node itself is always a dictionary of the keys *node*, containing the node name,
and *parameters* (optional), which value is another dictionary,
defining the variables for the node.
Normally these variables are directly given to the initialization function of the node
via a method called ``node_from_yaml`` of the general :class:`~pySPACE.missions.nodes.base_node.BaseNode`.
Since variables may be lists or dictionaries,
you may get a lot of indentations, but this is not the normal use-case.

For constructing your node chain you first have to choose your :mod:`nodes <pySPACE.missions.nodes>`.
To find out the possibilities of a node just check its documentation.
There you find an example and (normally) an exhaustive description of all
mandatory, recommended and optional parameters.
Finally the specifications of your chosen nodes are concatenated as a list
in your file and you are finished.


Example Node Chain File
..................................

.. literalinclude:: /examples/specs/node_chains/example_flow.yaml
    :language: yaml


.. _metadata_yaml:

Metadata.yaml
+++++++++++++

This file is responsible for defining dataset properties.
It can be always found in the data folder, associated with the dataset.
It is a dictionary written in YAML syntax.
If you get a dataset as a result of processing with pySPACE,
this file is stored automatically and some information about previous
processing is added.
For further processing, the resulting dataset can be immediately used.

From the programming perspective, the dictionary is loaded and forwarded
as *dataset_md* to the respective datasets.
It is even possible to access this data with the
:func:`~pySPACE.missions.nodes.base_node.BaseNode.get_metadata` method in a node.

If you want to define your own metadata.yaml,
to enable pySPACE to read your data, there are three categories of parameters
you can define:

    :mandatory parameters:
        This is always the *type* of the data and the *storage_format*
        and potentially additional required information for loading the data.
        For the additional loading information, check the documentation of the
        respective dataset which corresponds to your chosen type.

        There is a direct mapping between the type variable, the respective
        `dataset class <pySPACE.resources.dataset_defs>`,
        and the used :mod:`data type <pySPACE.resources.data_types>`.
        The type is written in lower case with underscores.
        The data type is the same but with camel case and for respective
        dataset, *Dataset* is added to the class name.
        The respective module names are the same as the *type*.

        Currently implemented types are:

            - :class:`stream <pySPACE.resources.dataset_defs.stream.StreamDataset>`
            - :class:`time_series <pySPACE.resources.dataset_defs.time_series.TimeSeriesDataset>`
              and
            - :class:`feature_vector <pySPACE.resources.dataset_defs.feature_vector.FeatureVectorDataset>`.

        The *storage_format* might consist of two components:
        ``[general_format, value_format]``.
        Currently, for the *value_format* only *real* is supported
        but it is used as a placeholder, to also for example
        support symbolic or text data in future.
        The *general_format* is the important part.
        For the stream type, only this parameter is used
        and for the other types it is possible, to for example use
        ``storage_format: csv`` in the *metadata.yaml*.
        By default, the framework stores data in the Python specific
        *pickle* format.
        the most commonly used other format is *csv*.

        An example for additional loading information is the *data_pattern*.
        It contains the placeholders *_run* for the run number,
        *_sp* for the split number, and *_tt* for the distinction between
        training and test data.
        These parameters are later on replaced in the pattern
        to get the needed file names.
        Hence, these keywords should not occur in the name of the dataset as
        for example in *data/my_special_running_tt_train_data*.

    :optional parameter:
        Optional parameters are either determined automatically
        or set with a sufficient default.

        The really important ones are:

            :run: This parameter defines the number of total processing
                repetitions, which were applied to the original data.

                If you define a new dataset, the correct value is *1*,
                which is also the default.

                The respective dataset will then contain
                a separate file for each repetition as specified in the
                aforementioned *data_pattern*:
                *_run* is replaced by an underscore
                and the respective *run_number*.
                The repetitions are sometimes needed, to account for
                randomness effects.
                For each repetition, the random seed is fixed, using
                the number of the current repetition (*run_number*).
                This is needed to get reproducible results and
                for example to get the same splitting
                into training and test data when processing the same
                dataset with different parametrization.

                (*optional, default: 1*)

            :split: This parameter defines the total number of splits,
                which were created in previous processing of a dataset
                using a cross-validation scheme.
                This parameter is handled in the same way as the run number
                with the *data_pattern*:
                *_sp* is replaced by an underscore and the respective index
                of the splitting.

                (*optional, default: 1*)

            :train_test: Defines if the data is already split into training
                and testing data. This holds true for the MNIST dataset of
                handwritten digits but not for the data in our
                example_summary, provided in the default *pySPACE_center*.

                By default this parameter is set to *false* and all data
                is assumed to be testing data.
                This can be for example changed with nodes for the
                :mod:`~pySPACE.missions.nodes.splitter` module.

                If the parameter is set to *true*, the loading procedure
                needs to know, which part is used for training and which one
                for testing. This is usually done by using the
                aforementioned *data_pattern* with the place holder *_tt*
                for the strings *_train* and *_test* to define training
                and testing data.

                For example the dataset pattern
                *data/MINST_1_vs_2_tt.csv*

                (*optional, default: false*)

        For special dataset dependent parameters and loading parameters
        refer to the documentation of the respective dataset (e.g,
        electrode positions from an EEG recording).
        Furthermore, ``parameter_setting`` of previous processing
        might be specified. This can be later on used for comparisons
        in an evaluation.

    :additional information:

        There can be a lot of additional information specified for the dataset
        which are not used by the software but which can provide useful
        information about the data and previous processing steps.
        This can be the class names, a dataset description,
        the source url, the data of creation, the author, or the detailed
        the specifications of previous processing
        which was applied to the data.




Example of a FeatureVectorDataset metadata.yaml
...............................................

.. code-block:: yaml

  type: feature_vector
  author: Max Mustermann
  date: '2009_4_5'
  node_chain_file_name: example_flow.yaml
  input_collection_name: input_collection_example
  classes_names: [Standard, Target]
  feature_names: [feature1, feature2, feature3]
  num_features: 3
  parameter_setting: {__LOWER_CUTOFF__: 0.1, __UPPER_CUTOFF__: 4.0}
  runs: 10
  splits: 5
  storage_format: [arff, real]
  data_pattern: data_run/features_sp_tt.arff
