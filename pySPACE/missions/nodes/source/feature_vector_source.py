""" Source for :class:`~pySPACE.resources.data_types.feature_vector.FeatureVector` elements e.g. from arff or csv files

.. note::    Nearly a total copy of the :mod:`~pySPACE.missions.nodes.source.time_series_source`.
            The important part of the code can be found in the corresponding
            metadata.yaml .
"""

from pySPACE.missions.nodes.base_node import BaseNode
from pySPACE.tools.memoize_generator import MemoizeGenerator


class FeatureVectorSourceNode(BaseNode):
    """ Source for samples of type :class:`~pySPACE.resources.data_types.feature_vector.FeatureVector`

    This node reads :class:`~pySPACE.resources.data_types.feature_vector.FeatureVector`
    elements
    accumulated in a :mod:`~pySPACE.resources.dataset_defs.feature_vector` and
    passes them into the :mod:`~pySPACE.environments.chains.node_chain`.
    As described in :mod:`~pySPACE.resources.dataset_defs.feature_vector` it is important,
    that the storage format is correct specified in the metadata.yaml.
    If the dataset has been constructed by pySPACE, this is done automatically.

    **Parameters**

    **Exemplary Call**
    
    .. code-block:: yaml

        - 
            node : Feature_Vector_Source

    :Author: Jan Hendrik Metzen (jhm@informatik.uni-bremen.de)
    :Created: 2008/11/25
    """
    input_types = ["FeatureVector"]

    def __init__(self, **kwargs):
        super(FeatureVectorSourceNode, self).__init__(**kwargs)

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

        .. todo:: to document
        """
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
                self._log("Accessing input dataset's training feature vector windows.")
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

        .. todo:: to document
        """
        # If we haven't read the data for testing yet
        if self.data_for_testing == None:
            self._log("Accessing input dataset's test feature vector windows.")
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
    
    
    def getMetadata(self, key):
        """ Return the value corresponding to the given key from the dataset meta data of this source node. """
        return self.dataset.meta_data.get(key)


_NODE_MAPPING = {"Feature_Vector_Source": FeatureVectorSourceNode,
                "Labeled_Feature_Vector_Source": FeatureVectorSourceNode,}
