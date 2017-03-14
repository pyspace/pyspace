.. _tutorial_interface_weka:

Feature Selection, Classification using WEKA
--------------------------------------------------------
In this tutorial you will see how the software is working together
with the popular data mining tool WEKA (see http://www.cs.waikato.ac.nz/ml/weka/). 
WEKA contains a lot of methods for data pre-processing 
e.g. filtering and for classification. 
They can be used within the framework through specific operation types:
Weka Filter Operation and Weka Classification Operation respectively. 
The tutorial consists of two parts. 
In the first section you will see how a feature selection is performed 
and in the second section how a classification is performed using WEKA with pySPACE.
Since you should have WEKA when you're doing this tutorial, 
we will use as example-files the data that comes with WEKA.

Feature Selection with WEKA
^^^^^^^^^^^^^^^^^^^^^^^^^^^
A feature selection is a Weka Filter Operation in pySPACE.
The main characteristics of this operation type is the transformation 
of one FeaturesVectorDataset summary into another.
More precisely WEKA will apply a filter to all arff-files 
(for more information about arff file format see http://www.cs.waikato.ac.nz/~ml/weka/arff.html) 
in the summary. In the case of a feature selection the goal is
to reduce the number of attributes.

First we have to provide the data that we want to use in the experiment in 
:ref:`storage`. To get a probable evaluation of the feature selection method
it is strongly recommended to divide the amount of data into training and test set. 
We use here the *segment-challenge* data set to train the feature selector 
and the *segment-test* data set for testing. 
In order that the datasets will be found correctly, :ref:`storage`
should look something like this:

.. code-block:: guess

   storage
           /tutorial_data
                   /{arff_tutorial_dataset}
                           /metadata.yaml
                           /data_run0
                                   features_sp0_test.arff
                                   features_sp0_train.arff

where the *metadata.yaml* might be

.. code-block:: yaml

   type: feature_vector
   author: Max Mustermann
   date: '2009_4_5'
   classes_names: [brickface,sky,foliage,cement,window,path,grass]
   feature_names: [region-centroid-col, region-centroid-row, region-pixel-count,
   short-line-density-5, short-line-density-2, vedge-mean, vegde-sd, hedge-mean, 
   hedge-sd, intensity-mean, rawred-mean, rawblue-mean, rawgreen-mean, exred-mean, 
   exblue-mean, exgreen-mean, value-mean, saturation-mean, hue-mean]
   num_features: 19
   runs: 1
   splits: 1
   storage_format: [arff, numeric]
   data_pattern: data_run/features_sp_tt.arff
   train_test: true

and *features_sp0_train.arff* is the renamed segment-challenge.arff file 
from WEKA and *features_sp0_test.arff* the renamed segment-test.arff file 
respectively. Additionally to the renaming you have to change the relation name 
from both arff files to the name of your collection, e.g. "{arff_tutorial_collection}".

The second step is to specify the operation. 
In the subdirectory *operation* of :ref:`specs_dir` you'll find the subdir *weka_filtering*. 
Here you can store your operation spec file or just modify the example one:

.. literalinclude:: ../examples/specs/operations/examples/weka_feature_selection.yaml
    :language: yaml

Third, start e.g. by invoking::

   python launch.py --mcore --configuration your_configuration_file.yaml --operation examples/weka_feature_selection.yaml

How to configure `your_configuration_file.yaml` please look in section :ref:`getting_started`. 
Important here is especially the correct WEKA class path.

Now four processes, one for each feature selector, are executed and
and the results are stored in four result directories.
In each result directory you'll find, beside the new *metadata.yaml*
for further processing, again the two arff files which now contain only **ten** attributes. 
In addition there is a *score.txt* file for every split that shows the rating 
of features for the specific feature selector. 
So you might check which attributes the different evaluators chose. 
To make later analysis easier there are also the input *metadata.yaml*,
the *operation spec file* and the used *template* to start WEKA stored.

Classification with WEKA
^^^^^^^^^^^^^^^^^^^^^^^^
The Weka Classification Operation is used to train and test a classifier with WEKA.
The operation needs a FeatureVectorDataset summary as input
and produces a Weka PerformanceResultSummary as output,
which is immediately transformed to the software format.

The first step, again, is to provide the data for this operation. 
This might be very easy since we can use the same data as for the feature selection.

As second step the operation spec file has to be specified. 
The designated location for it is the subdirectory *weka_classification* 
in the *operation* directory of :ref:`specs_dir`. 
Here an example of this file:

.. literalinclude:: ../examples/specs/operations/examples/weka_classification_operation.yaml
    :language: yaml

By using this spec file for the operation the Libsvm, a classifier 
that not directly comes with WEKA, is applied. 
You can get it on http://www.csie.ntu.edu.tw/~cjlin/libsvm/. 
As parameter only the complexity of the linear Libsvm will be changed. 
The weights for the classes will be by default 1 for all classes 
so you can ignore this parameters. 
WEKA is calculating the performance metrics like precision and recall 
with respect to the first class, here *brickface*. Finally start ::

   python launch.py --mcore --configuration your_configuration_file.yaml --operation examples/weka_classification_operation.yaml

and consider to add the libsvm.jar path
in *your_configuration_file.yaml* to the WEKA class paths.

In the *operation_result* directory you'll find, 
beside the new *metadata.yaml*, a *results.csv* file
where all classification results are stored. 
You might want to visualize the results. 
Therefore another Operation is used: the Analysis Operation.

.. literalinclude:: ../examples/specs/operations/examples/weka_analysis.yaml
    :language: yaml

In this example the *DATAFLOW* parameter refers to the feature selector that was used.

You can now start another operation like the Weka Filter Operation 
or the Weka Classification Operation before, 
or you do all together in an operation chain.
The spec file for this chain might look like:

.. literalinclude:: ../examples/specs/operation_chains/examples/weka.yaml
    :language: yaml

and you can start it with the command::

   python launch.py --mcore --configuration your_configuration_file.yaml --operation_chain examples/weka.yaml

Two of the graphics from the result directory you can see here:

.. image:: ../graphics/chi_squared_F_measure_Complexity.png
   :width: 500

.. image:: ../graphics/relief_F_measure_Complexity.png
   :width: 500

The first graphic refers to the chi-squared feature selector 
and the second to the relief feature selector. 
It is shown that with different feature sets 
the optimal complexity parameter of the SVM might change. 
In addition in this case the relief feature selector chooses attributes 
that result in a higher performance for the *brickface* class. 