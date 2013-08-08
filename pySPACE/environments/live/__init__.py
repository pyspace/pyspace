""" Modules for the online usage of pySPACE

    Contains the modules that are actually needed
    if pySPACE is used in any online application.

    It wires all required parts together and configures
    them according to given configuration and parameterization
    files.

    The pyspace live mode of operation is structured in several
    different epochs.

* In the *optional* **prewindowing** phase, pyspace live connects to a server that provides the data, performs preprocessing operations and stores the preprocessed data.
    The preprocessing operations have to be provided in a flow, and they should not be trainable.
    The trainable modules can only be used in the train phase.

* In the **training** phase, pyspace live loads stored data and uses it to train the trainable nodes of a flow.
    If the origin of the data is the prewindowing phase, this phase is called prewindowed train.

* In the *optional* **adaptation** phase, the threshold of the classifier is adapted to the actual task

* In the **prediction** phase, the trained flows are used to predict the class of new data, this

An introductory unit test is found in unit test directory.

"""


