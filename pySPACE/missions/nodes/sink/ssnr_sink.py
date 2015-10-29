""" Sink-Node for the Signal-to-Signal-Plus-Noise Ratio. """

from copy import copy, deepcopy

import numpy

from pySPACE.missions.nodes.base_node import BaseNode
from pySPACE.missions.nodes.spatial_filtering.xdawn import SSNR
from pySPACE.resources.dataset_defs.metric import BinaryClassificationDataset


class SSNRSinkNode(BaseNode):
    """ Sink node that collects SSNR metrics in actual and virtual sensor space.

    This sink node computes metrics based on the signal to signal-plus-noise
    ratio (SSNR). It computes the SSNR both in actual sensor space (ssnr_as)
    and virtual sensor space (ssnr_vs). For virtual sensor space, it computes
    additionally the SSNR on unseen test data (if available), meaning that the
    mapping to virtual sensor space is computed on training data and the
    SSNR is computed on test data that was not used for learning this mapping.

    **Parameters**
        :erp_class_label: Label of the class for which an ERP should be
            evoked. For instance "Target" for a P300 oddball paradigm.

        :retained_channels: The number of channels that are retained after
            xDAWN-based spatial filtering. The SSNR is computed in the
            virtual sensor space consisting of these retained channels.
            This quantity is only relevant for the SSNR metrics that are
            computed in the virtual sensor space. If this quantity is not
            defined, all channels are retained.

            (*optional, default: None*)

    **Exemplary Call**

    .. code-block:: yaml

        -
            node : SSNR_Sink
            parameters :
                  erp_class_label : "Target"
                  retained_channels : 8

    :Author: Jan Hendrik Metzen (jhm@informatik.uni-bremen.de)
    :Created: 2011/11/14

    .. todo:: It would be desirable to compute the SSNR within a node chain and
              write it along with other metrics in a result file in a later sink
              node. This could work the following way: Write the SSNR metrics in
              BaseData type, and read these in the
              :class:`~pySPACE.missions.nodes.sink.classification_performance_sink.PerformanceSinkNode`.
              A further sink node should be added that stores only metrics
              contained in BaseData type.
    """
    input_types = ["TimeSeries"]

    def __init__(self, erp_class_label, retained_channels=None):
        super(SSNRSinkNode, self).__init__()

        # We reuse the ClassificationCollection (maybe this should be renamed to
        # MetricsDataset?)
        self.set_permanent_attributes(
            # Object for handling SSNR related calculations
            ssnr=SSNR(erp_class_label, retained_channels),
            # Result collection
            ssnr_collection=BinaryClassificationDataset(),
            erp_class_label=erp_class_label)

    def reset(self):
        """ Reset the node to the clean state it had after its initialization
        """
        # We have to create a temporary reference since we remove
        # the self.permanent_state reference in the next step by overwriting
        # self.__dict__
        tmp = self.permanent_state
        # reset should not delete classification collection
        # if you want to delete the collection just do it explicitly.
        tmp["ssnr_collection"] = self.ssnr_collection
        self.__dict__ = copy(tmp)
        self.permanent_state = tmp

    def is_trainable(self):
        """ Return whether this node is trainable."""
        return True

    def is_supervised(self):
        """ Return whether this node requires supervised training. """
        return True

    def _train(self, data, label):
        # Add training example *data* along with its label to SSNR class
        self.ssnr.add_example(data, label)

    def _stop_training(self):
        self.ssnr.stop_training()

    def _execute(self, data):
        return data  # Return data unchanged

    def process_current_split(self):
        # Training of node on training data
        for data, label in self.input_node.request_data_for_training(False):
            self.train(data, label)

        # Compute performance metrics SSNR_AS and SSNR_vs on the training
        # data
        performance = {"ssnr_as" : self.ssnr.ssnr_as(),
                       "ssnr_vs" : self.ssnr.ssnr_vs()}

        # Collect test data (if any)
        X_test = None
        D_test = None
        for data, label in self.input_node.request_data_for_testing():
            if label == self.erp_class_label:
                D = numpy.diag(numpy.ones(data.shape[0]))
            else:
                D = numpy.zeros((data.shape[0], data.shape[0]))

            if X_test is None:
                X_test = deepcopy(data)
                D_test = D
            else:
                X_test = numpy.vstack((X_test, data))
                D_test = numpy.vstack((D_test, D))

        # If there was separate test data:
        # compute metrics that require test data
        if X_test is not None:
            performance["ssnr_vs_test"] = self.ssnr.ssnr_vs_test(X_test, D_test)

        # Add SSNR-based metrics computed in this split to result collection
        self.ssnr_collection.add_split(performance, train=False,
                                       split=self.current_split,
                                       run=self.run_number)

    def get_result_dataset(self):
        """ Return the result collection with the results of this node. """
        return self.ssnr_collection


_NODE_MAPPING = {"SSNR_Sink": SSNRSinkNode}
