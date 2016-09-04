""" Sink-Node for the Signal-to-Signal-Plus-Noise Ratio. """

from copy import copy, deepcopy
import warnings

import numpy
from scipy.linalg import qr

from pySPACE.missions.nodes.base_node import BaseNode
from pySPACE.resources.dataset_defs.metric import BinaryClassificationDataset


class SSNR(object):
    """ Helper-class that encapsulates SSNR related computations.

    Use as follows: add training examples one-by-one along with their labels
    using the method add_example. Once all training data has been added, metrics
    values can be computed using ssnr_as, ssnr_vs, and ssnr_vs_test

    """

    def __init__(self, erp_class_label, retained_channels=None):
        self.retained_channels = retained_channels
        self.erp_class_label = erp_class_label

        self.X = None # The data matrix (will be constructed iteratively)
        self.D = None # The Toeplitz matrix (will be constructed iteratively)

    def add_example(self, data, label):
        """ Add the example *data* for class *label*. """
        if self.retained_channels is None:
            self.retained_channels = data.shape[1]
        else:
            self.retained_channels = min(self.retained_channels, data.shape[1])

        # Iteratively construct Toeplitz matrix D and data matrix X
        if label == self.erp_class_label:
            D = numpy.diag(numpy.ones(data.shape[0]))
        else:
            D = numpy.zeros((data.shape[0], data.shape[0]))

        if self.X is None:
            self.X = deepcopy(data)
            self.D = D
        else:
            self.X = numpy.vstack((self.X, data))
            self.D = numpy.vstack((self.D, D))

    def ssnr_as(self, selected_electrodes=None):
        """ SSNR for given electrode selection in actual sensor space.

        If no electrode selection is given, the SSNR of all electrodes is
        computed.
        """
        if selected_electrodes is None:
            selected_electrodes = range(self.X.shape[1])

        self.Sigma_1, self.Sigma_X = self._compute_Sigma(self.X, self.D)

        filters = numpy.zeros(shape=(self.X.shape[1], self.X.shape[1]))
        for electrode_index in selected_electrodes:
            filters[electrode_index, electrode_index] = 1

        # Return the SSNR that these filters would obtain on training data
        return self._ssnr(filters, self.Sigma_1, self.Sigma_X)

    def ssnr_vs(self, selected_electrodes=None):
        """ SSNR for given electrode selection in virtual sensor space.

        If no electrode selection is given, the SSNR of all electrodes is
        computed.
        """
        if selected_electrodes is None:
            selected_electrodes = range(self.X.shape[1])

        self.Sigma_1, self.Sigma_X = self._compute_Sigma(self.X, self.D)

        # Determine spatial filter using xDAWN that would be obtained if
        # only the selected electrodes would be available
        partial_filters = \
            self._compute_xDAWN_filters(self.X[:, selected_electrodes],
                                        self.D)
        # Expand partial filters to a filter for all electrodes (by setting
        # weights of inactive electrodes to 0)
        filters = numpy.zeros((self.X.shape[1], self.retained_channels))
        # Iterate over electrodes with non-zero weights
        for index, electrode_index in enumerate(selected_electrodes):
            # Iterate over non-zero spatial filters
            for j in range(min(filters.shape[1], partial_filters.shape[1])):
                filters[electrode_index, j] = partial_filters[index, j]

        # Return the SSNR that these filters would obtain on training data
        return self._ssnr(filters, self.Sigma_1, self.Sigma_X)

    def ssnr_vs_test(self, X_test, D_test, selected_electrodes=None):
        """ SSNR for given electrode selection in virtual sensor space.

        Note that the training of the xDAWN spatial filter for mapping to
        virtual sensor space and the computation of the SSNR in this virtual
        sensor space are done on different data sets.

        If no electrode selection is given, the SSNR of all electrodes is
        computed.
        """
        if selected_electrodes is None:
            selected_electrodes = range(self.X.shape[1])

        # Determine spatial filter using xDAWN that would be obtained if
        # only the selected electrodes would be available
        partial_filters = \
            self._compute_xDAWN_filters(self.X[:, selected_electrodes],
                                        self.D)
        # Expand partial filters to a filter for all electrodes (by setting
        # weights of inactive electrodes to 0)
        filters = numpy.zeros((self.X.shape[1], self.retained_channels))
        # Iterate over electrodes with non-zero weights
        for index, electrode_index in enumerate(selected_electrodes):
            # Iterate over non-zero spatial filters
            for j in range(min(filters.shape[1], partial_filters.shape[1])):
                filters[electrode_index, j] = partial_filters[index, j]

        # Return the SSNR that these filters would obtain on test data
        Sigma_1_test, Sigma_X_test = self._compute_Sigma(X_test, D_test)
        return self._ssnr(filters, Sigma_1_test, Sigma_X_test)

    def _compute_Sigma(self, X, D):
        if D is None:
            warnings.warn("No data given for sigma computation.")
        elif not(1 in D):
            warnings.warn("No ERP data (%s) provided." % self.erp_class_label)
        # Estimate of the signal for class 1 (the erp_class_label class)
        A_1 = numpy.dot(numpy.dot(numpy.linalg.inv(numpy.dot(D.T, D)), D.T), X)
        # Estimate of Sigma 1 and Sigma X
        Sigma_1 = numpy.dot(numpy.dot(numpy.dot(A_1.T, D.T), D), A_1)
        Sigma_X = numpy.dot(X.T, X)

        return Sigma_1, Sigma_X

    def _ssnr(self, v, Sigma_1, Sigma_X):
        # Compute SSNR after filtering  with v.
        a = numpy.trace(numpy.dot(numpy.dot(v.T, Sigma_1), v))
        b = numpy.trace(numpy.dot(numpy.dot(v.T, Sigma_X), v))
        return a / b

    def _compute_xDAWN_filters(self, X, D):
        # Compute xDAWN spatial filters

        # QR decompositions of X and D
        if map(int, __import__("scipy").__version__.split('.')) >= [0,9,0]:
            # NOTE: mode='economy'required since otherwise the memory
            # consumption is excessive
            Qx, Rx = qr(X, overwrite_a=True, mode='economic')
            Qd, Rd = qr(D, overwrite_a=True, mode='economic')
        else:
            # NOTE: econ=True required since otherwise the memory consumption
            # is excessive
            Qx, Rx = qr(X, overwrite_a=True, econ=True)
            Qd, Rd = qr(D, overwrite_a=True, econ=True)

        # Singular value decomposition of Qd.T Qx
        # NOTE: full_matrices=True required since otherwise we do not get
        #       num_channels filters.
        Phi, Lambda, Psi = numpy.linalg.svd(numpy.dot(Qd.T, Qx),
                                           full_matrices=True)
        Psi = Psi.T

        # Construct the spatial filters
        for i in range(Psi.shape[1]):
            # Construct spatial filter with index i as Rx^-1*Psi_i
            ui = numpy.dot(numpy.linalg.inv(Rx), Psi[:,i])
            if i == 0:
                filters = numpy.atleast_2d(ui).T
            else:
                filters = numpy.hstack((filters, numpy.atleast_2d(ui).T))

        return filters


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