""" Source for :class:`~pySPACE.resources.data_types.prediction_vector.PredictionVector`
data from `*.csv` and `*.pickle` files

.. note::
    Nearly a total copy of the :mod:`~pySPACE.missions.nodes.source.time_series_source`.
    The important part of the code can be found in the corresponding
    metadata.yaml .
"""

from pySPACE.missions.nodes.base_node import BaseNode
from pySPACE.tools.memoize_generator import MemoizeGenerator


class PredictionVectorSourceNode(BaseNode):
    """ Source for samples of type
    :class:`~pySPACE.resources.data_types.prediction_vector.PredictionVector`

    This node reads
    :class:`~pySPACE.resources.data_types.prediction_vector.PredictionVector`
    elements
    accumulated in a :mod:`~pySPACE.resources.dataset_defs.prediction_vector` and
    passes them into the :mod:`~pySPACE.environments.chains.node_chain`.
    As described in :mod:`~pySPACE.resources.dataset_defs.prediction_vector` it is important,
    that the storage format is correct specified in the metadata.yaml.
    If the dataset has been constructed by pySPACE, this is done automatically.

    .. note::
        This node is still in an experimental phase as of 17 Dec 2014.
        Further testing and development are required before releasing the node.

    **Parameters**

    **Exemplary Call**

    .. code-block:: yaml

        -
            node : PredictionVectorSource

    :Author: Andrei Ignat (andrei_cristian.ignat@dfki.de)
    :Created: 2014/17/12
    """
    input_types = ["PredictionVector"]

    def __init__(self, **kwargs):
        super(PredictionVectorSourceNode, self).__init__(**kwargs)

    def set_input_dataset(self, dataset):
        """ Sets the dataset from which this node reads the data """
        self.set_permanent_attributes(dataset=dataset)

    def register_input_node(self, node):
        """ Register the given node as input """
        raise Exception("No nodes can be registered as inputs for source nodes")

    def use_next_split(self):
        """
        Use the next split of the data into training and test data.
        Returns True if more splits are available, otherwise False.

        This method is useful for benchmarking
        """
        return False

    def train_sweep(self, use_test_data):
        """
        Performs the actual training of the node.
        .. note:: Source nodes cannot be trained
        """
        raise Exception("Source nodes cannot be trained")

    def request_data_for_training(self, use_test_data):
        """
        Returns the time windows that can be used for training of subsequent nodes
        """
        # TODO:Is all this really necessary?
        if not use_test_data:
            # If the input dataset consists only of one single run,
            # we use this as input for all runs to be conducted (i.e. we
            # rely on later randomization of the order). Otherwise
            # we use the data for this run number
            if self.dataset.meta_data["runs"] > 1:
                key = (self.run_number, self.current_split, "train")
            else:
                key = (0, self.current_split, "train")
            # Check if there is training data for the current split and run
            if key in self.dataset.data.keys():
                self._log("Accessing input dataset's training prediction vectors.")
                self.data_for_training = MemoizeGenerator(self.dataset.get_data(*key).__iter__(),
                                                          caching=self.caching)
            else:
                # Returns an iterator that iterates over an empty sequence
                # (i.e. an iterator that is immediately exhausted), since
                # this node does not provide any data that is explicitly
                # dedicated for training
                self._log("No training data available.")
                self.data_for_training = MemoizeGenerator((x for x in [].__iter__()),
                                                          caching=self.caching)
        else:
            # Return the test data as there is no additional data that
            # was dedicated for training
            return self.request_data_for_testing()

        # Return a fresh copy of the generator
        return self.data_for_training.fresh()

    def request_data_for_testing(self):
        """
        Returns the data that can be used for testing of subsequent nodes


        """
        # TODO:Is all this really necessary?
        # If we haven't read the data for testing yet
        if self.data_for_testing == None:
            self._log("Accessing input dataset's test prediction vectors.")
            # If the input dataset consists only of one single run,
            # we use this as input for all runs to be conducted (i.e. we
            # rely on later randomization of the order). Otherwise
            # we use the data for this run number
            if self.dataset.meta_data["runs"] > 1:
                key = (self.run_number, self.current_split, "test")
            else:
                key = (0, self.current_split, "test")

            test_data_generator = self.dataset.get_data(*key).__iter__()

            self.data_for_testing = MemoizeGenerator(test_data_generator,
                                                     caching=self.caching)

        # Return a fresh copy of the generator
        return self.data_for_testing.fresh()


    def get_metadata(self, key):
        """ Return the value corresponding to the given key from the dataset meta data of this source node. """
        return self.dataset.meta_data.get(key)
