A regression tutorial in pySPACE
--------------------------------

Introduction
============


As with any machine learning software, pySPACE is also capable of performing regression on
selected data sets. In this introductory tutorial, we will be looking at exactly how some
simple regression tasks can be run through the pySPACE suite.

Contents of the tutorial

	1) `Preparing the data`_
	2) `Building the node chain`_


Preparing the data
==================

For the pupose of this tutorial, we will use a dataset downloaded from the
`UCI repository <https://archive.ics.uci.edu/ml/datasets.html>`_ and namely the
`Wine Quality Dataset <https://archive.ics.uci.edu/ml/datasets/Wine+Quality>`_.
For this dataset, the taks is rather simple: based on the 11 physicochemical input
features and on the accopamnying quality score, train a regression algorithm that
will be able to predict the quality of a new wine sample.

Before we can perform the regression, we must prepare the dataset that we just
downloaded such that it can be processed by pySPACE. We thus go to the
`pySPACEcenter` directory(which was initialized when the software was first
installed) and make a new folder inside the `storage` folder. Let's call our
new folder `winedata` and make two subfolders inside this folder namely `red`
and `white`. Your directory structure should now look like this:

.. code-block:: bash

	pySPACEcenter\
		...\
		storage\
			...\
			winedata\
				red\
				white\

At this point, you are ready tow download the datasets. Go to the
`Wine Quality data directory <https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/>`_
and save the two `*.csv` files each as `data.csv` in the corresponding directory.
Thus, the original `winequality-red.csv` is now `red\data.csv` and likewise
for the white wine. Youe directory structure should now look like:

.. code-block:: bash

	pySPACEcenter\
		...\
		storage\
			...\
			winedata\
				red\
					data.csv
				white\
					data.csv

The final step in preparing the data-set is to generate the `metadata.yaml`
files. This can be done either by hand or through the use of an automatic script
namely :mod:`~pySPACE.run.scripts.md_creator`. If you want to use the automatic
script, first go to you data directory(either the `red` or the `white` one) and
then type:

.. code-block:: bash

	$ python <PATH_TO_PYSPACE>/pySPACE/run/scripts/md_creator.py

The script will then ask you to a couple of questions regarding the format of your data and namely

.. code-block:: bash

	Running meta data creator ... 

	Please enter the name of the file. --> data.csv
	Please enter the storage_format of the data.
	 one of arff, csv (csv with header), csvUnnamed (csv without header)--> csv
	Please enter csv delimiter/separator. 
	(e.g. ','  , ' ' , ';' or '	' for a tab, default:',')-->;
	Please enter all rows that can be ignored, separated by comma or range.
	 eg. [1, 2, 3] or [1-3] --> 
	Please enter all columns that can be ignored, separated by comma or range.
	The first row gets number 1.eg. [1, 2, 3] or [1-3] --> 
	Please enter the column that contains the label. -1 for last column 
	 --> -1

	Meta data file metadata.yaml written. 


At this point, there is a new metadata file in your `red`/`white` directory
which should read

.. code-block:: yaml

	author: <YOUR_NAME>
	date: <TODAY>
	type: feature_vector
	ignored_rows: []
	storage_format: [csv, real]
	file_name: data.csv
	label_column: -1
	ignored_columns: []

The final version of your directory structure(for the purposes of this tutorial)
should therefore be:

.. code-block:: bash

	pySPACEcenter\
		...\
		storage\
			...\
			winedata\
				red\
					data.csv
					metadata.yaml
				white\
					data.csv
					metadata.yaml

Building the node chain
=======================

Now that we have nicely organized data, we can start doing something with it.
The following example is based on nodes that are direct implementations of
`scikit-learn <http://scikit-learn.org/stable/>`_. Therefore, in order to run
the following node chain, you need to
`install scikit-learn <http://scikit-learn.org/stable/install.html>`_.

We plan to do the following to our dataset:

1) Preprocessing(so that the data is nicely formatted,
    not too high-dimensional and normalized)
2) Ridge Regression(extending this to another regression model
    is a matter of changing a couple of lines in the definition of the node chain)
3) Analyze the results(and implicitly see how well our
    algorithms have performed)


In order to do all of the above, we need to define a `node_chain` under
a `YAML` format. For that, go to the `pySPACEcenter/specs/operations/`
directory and open your favorite text editor. If you already want to give
the file a name, save this new file under `winedata.yaml`. The first lines
of the file should say what the file represents i.e. a `node_chain` and where
to look for the input data(relative to the `pySPACEcenter/storage/` directory).
If you have followed the above steps for saving your input, the first lines of
your `winedata.yaml` file should read:

.. code-block:: yaml

	type: node_chain
	runs: 1
	input_path: "winedata"

Next up is the content of the node chain itself. Whenever a node chain is
defined, it must start with a :mod:`SourceNode <pySPACE.missions.nodes.source>`.
In our case, we will be using the :class:`~pySPACE.missions.nodes.source.feature_vector_source`,
since we want to cast our data into `~pySPACE.resources.data_types.feature_vector`.
Your `winedata.yaml` file should now look like:

.. code-block:: yaml

	type: node_chain
	runs: 1
	input_path: "winedata"
	node_chain: 
	    -
	        node: FeatureVectorSourceNode


Now that the node chain has a source node, we can start the preprocessing.
Since for the purpose of this tutorial we want to keep things simple, we
will just implement two failsafe methods(in case the initial data contains
`int` or `NaN` values) through
:class:`~pySPACE.missions.nodes.type_manipulation.float_conversion.Int2FloatNode`
and :class:`~pySPACE.missions.nodes.type_manipulation.float_conversion.NaN2NumberNode`.
We will then split our data set into test and training data using
:class:`~pySPACE.missions.nodes.splitter.traintest_splitter.TrainTestSplitterNode`
and normalize the values using
:class:`~pySPACE.missions.nodes.postprocessing.feature_normalization.OutlierFeatureNormalizationNode`.
Translating this into `YAML` directives yields:

.. code-block:: yaml

	type: node_chain
	runs: 1
	input_path: "winedata"
	node_chain: 
	    -   
	        node: FeatureVectorSourceNode
	    -
	        node: NaN2Number
	    -
	        node: Int2Float
	    -       
	        node : TrainTestSplitter
	        parameters :
	            train_ratio : 0.7
	            random : False
	    -
	        node: OutlierFeatureNormalization

Good. Now our data is well behaved and we can perform regression on it. For this
purpose, we will pick a regressor node from the `sklearn` suite and namely
`KNeighborsRegressor <http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html#sklearn.neighbors.KNeighborsRegressor>`_.
While this node might not be the optimal choice, it is definitely a well
behaved and well understood choice and therefore suitable for the purposes
of this tutorial.

Now that is done, we just have to add a sink node at the end of our node chain.
Since we want this sink node to check the performance of our node, we will be
using :class:`~pySPACE.missions.nodes.sink.classification_performance_sink.PerformanceSinkNode`.
The final version of your `winedata.yaml` file should look like:


.. code-block:: yaml

	type: node_chain
	runs: 1
	input_path:
	node_chain: 
	    -   
	        node: FeatureVectorSourceNode
	    -
	        node: NaN2Number
	    -
	        node: Int2Float
	    -       
	        node : TrainTestSplitter
	        parameters :
	            train_ratio : 0.7
	            random : False
	    -
	        node: OutlierFeatureNormalization
	    -
	        node: KNeighborsRegressorSklearnNode
	    -
	        node: PerformanceSinkNode
	        parameters:     
	            evaluation_type: "regression"

Congratulations! You have just finished writing your first regression node-chain
in pySPACE! In order to run the code, go to your `pySPACEcenter` and type the
following command in the terminal

.. code-block:: bash

	$ python launch.py --operation winedata.yaml
