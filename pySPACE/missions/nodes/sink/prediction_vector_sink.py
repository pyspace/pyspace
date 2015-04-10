""" Collect prediction vectors """

import copy

from pySPACE.missions.nodes.base_node import BaseNode
from pySPACE.resources.dataset_defs.prediction_vector import PredictionVectorDataset


class PredictionVectorSinkNode(BaseNode):
    """ Collect all :class:`~pySPACE.resources.data_types.prediction_vector.PredictionVectorDataset` elements
    that are passed through it in a collection of type
    :mod:`~pySPACE.resources.dataset_defs.prediction_vector`.

    .. note::
        The code is heavily based on its counterpart for
        :class:`~pySPACE.resources.data_types.feature_vector.FeatureVector`
        elements that can be found in :mod:`~pySPACE.missions.nodes.sink.feature_vector_sink.FeatureVectorSinkNode`


    **Exemplary Call**

    .. code-block:: yaml

        -
            node: PredictionVectorSink

    :input:  PredictionVector
    :output: PredictionVectorDataset
    :Author: Andrei Ignat (andrei_cristian.ignat@dfki.de)
    :Created: 2014/10/15
    """
    input_types = ["PredictionVector"]

    def __init__(self, **kwargs):
        super(PredictionVectorSinkNode, self).__init__(**kwargs)
        self.set_permanent_attributes(prediction_vector_collection=None)


    def reset(self):
        """ Reset the state of the object to the clean state it had after its
        initialization
        """
        tmp = self.permanent_state
        # TODO: just a hack to get it working quickly...
        tmp["prediction_vector_collection"] = self.prediction_vector_collection
        self.__dict__ = copy.copy(tmp)
        self.permanent_state = tmp

    def is_trainable(self):
        """ Returns whether this node is trainable.

        Since we want to sink the training examples as well, this
        function wil return True
        """
        return True

    def is_supervised(self):
        """ Returns whether this node requires supervised training """
        return True

    def _train(self, data, label):
        # We do nothing
        pass

    def _create_result_sets(self):
        """ Instantiate the :class:`~pySPACE.resources.data_types.prediction_vector.PredictionVectorDataset`
        """
        self.prediction_vector_collection = PredictionVectorDataset()

    def process_current_split(self):
        """ Compute the results of this sink node for the current split
        of the data into train and test data
        """
        # Compute the prediction vectors for the data used for training
        for prediction_vector, label in self.input_node.request_data_for_training(False):
            if prediction_vector.tag != "Discard":
                if self.prediction_vector_collection is None:
                    # create the dataset if it does not already exist
                    self._create_result_sets()

                self.prediction_vector_collection.add_sample(prediction_vector,
                                                             label,
                                                             train=True,
                                                             split=self.current_split,
                                                             run=self.run_number)

        # Compute the prediction vectors for the data used for testing
        for prediction_vector, label in self.input_node.request_data_for_testing():
            # If Prediction Vectors need to be discarded, that is done here
            if prediction_vector.tag != "Discard":
                # Do lazy initialization of the class
                # (maybe there were no training examples)
                if self.prediction_vector_collection is None:
                    self._create_result_sets()
                # Add sample
                self.prediction_vector_collection.add_sample(prediction_vector,
                                                             label,
                                                             train=False,
                                                             split=self.current_split,
                                                             run=self.run_number)

    def get_result_dataset(self):
        """ Return the result dataset """
        return self.prediction_vector_collection