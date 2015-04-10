""" Original and variants of the Common Spatial Pattern algorithm

The Common Spatial Pattern algorithm is a supervised method for
spatial filtering. It is summarized e.g. in
http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=4408441

"""

import os
from collections import defaultdict
import cPickle
import numpy
import scipy
import logging

from pySPACE.missions.nodes.spatial_filtering.spatial_filtering import SpatialFilteringNode
from pySPACE.resources.data_types.time_series import TimeSeries
from pySPACE.resources.dataset_defs.stream import StreamDataset

from pySPACE.tools.filesystem import  create_directory

class CSPNode(SpatialFilteringNode):
    """ Common Spatial Pattern filtering
    
    A node that implements the supervised common spatial pattern
    algorithm for spatial filtering as described in
    http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=4408441
    
    The ordering of classes in this implementation is done in alphabetic order.
    This means when having the classes 'Target' and 'Standard',
    that the first channel is the best for 'Standard' and the last one the best
    for 'Targets', since the first one is the one with the highest eigenvalue.
    
    To unify this spatial filter with other spatial filters for the final filter
    we arranged the channels in alternating order, meaning first, last, second,
    last but one, third, last but two, ... .
    This is done, such that the ordering of filters is according to importance
    of the filter, which is important, when deciding, which channels to choose.
    Nevertheless the channel names still correspond to the ordering of the
    eigenvalues.

    **Parameters**
        :retained_channels: Determines how many of the CSP pseudo channels
            are retained. Default is None which means "all channels".

            (*optional, default: None*)

        :relevant_indices: Determines on which time indices the CSP filters
            are trained. Default is None which means "all time index".

            (*optional, default: None*)

        :spatio_temporal: If this parameter is true, each sample
            is considered as a separate channel.
            This means that a time window with 64 channels and 25 observations 
            is transformed into a time window with 64*25 channels with one 
            observation.
            After filtering, the data is  retransformed into a 64x25 time window.

            (*optional, default: False*)

        :load_path: An absolute path from which the CSP patterns can be
            loaded. If not specified, these patterns are learned from the training data.

            (*optional, default: None*)

        :visualize_pattern: If value is true, a visualization of the learned CSP
            is stored. Therefore the parameter store needs to be set to true also.

            (*optional, default: False*)

    **Exemplary Call**
    
    .. code-block:: yaml
    
        -
            node : CSP
            parameters:
                retained_channels : 32
                relevant_indices : [-4,-3,-2,-1]
                viszualize_pattern : True
                store : True

    :Author: Jan Hendrik Metzen (jhm@informatik.uni-bremen.de)
    :Created: 2008/12/09
    """
    def __init__(self, retained_channels=None, relevant_indices=None, 
                 spatio_temporal=False, load_path=None,
                 visualize_pattern=False, **kwargs):
        # Must be set before constructor of superclass is set
        self.trainable = (load_path == None)
        
        super(CSPNode, self).__init__(**kwargs)
        
        # Make sure number of channels is even
        if not(retained_channels in [None, 'None']) and not (retained_channels % 2 == 0):
            retained_channels -= 1
            self._log("CSP node should only retain an even number of channels!", level=logging.CRITICAL)

        filters = None
        # Load patterns from file if requested
        if load_path != None:
            filters_file = open(load_path, 'r')
            filters = cPickle.load(filters_file)

        if spatio_temporal and relevant_indices is not None:
            import warnings
            warnings.warn("Relevant indices are ignored in spatio-temporal"
                          "filtering.")
            relevant_indices = None

        self.set_permanent_attributes(
            # The number of channels is only known after
            # observing the first data point
            number_of_channels = None,

            # A mapping from class label to empirical covariance under this
            # condition
            # The covariance matrices cannot be created until the size of
            # the data (i.e. the number of channels is known
            covariances = dict(),
            
            # Count the number of samples for each condition to
            # normalize the covariance matrices later on
            samples_per_condition = dict(), 
            
            # The relevant indices that are used during CSP training
            relevant_indices = relevant_indices,
            
            # Should we do spatio-temporal filtering?
            spatio_temporal = spatio_temporal, 
            
            # After training is finished, this attribute will contain
            # the common spatial patterns that are used to project
            # the data onto a lower dimensional subspace
            filters = filters,
            
            # The number of channels that will be retained
            retained_channels = retained_channels,
            
            # Remember the new order of CSP-pseudo channels
            new_order = None,
            
            trainable = self.trainable,
            
            # Determines whether the CSP filters are shown after training
            visualize_pattern = visualize_pattern,
            
            # Remember the data's channel names later on
            channel_names = None,
            new_channel_names = None
        )
    
    def is_trainable(self):
        """ Returns whether this node is trainable. """
        return self.trainable
    
    def is_supervised(self):
        """ Returns whether this node requires supervised training """
        return self.trainable

    def _train(self, data, label):
        """
        Add the given data point along with its class label 
        to the training set, i.e. update the class' covariance matrix
        """
        if self.spatio_temporal:
            data = data.reshape(1, data.shape[0] * data.shape[1])

        # If this is the first data sample we obtain
        if self.number_of_channels in [None, 'None']:
            # Count the number of channels
            self.number_of_channels = data.shape[1]
            self.channel_names = data.channel_names

        # If we encounter this label for the first time
        if label not in self.covariances.keys():
            # Create the class covariance matrices lazily
            self.covariances[label] = numpy.zeros((self.number_of_channels,
                                                      self.number_of_channels)) 
        
        # Just use the relevant indices for CSP training
        # (for instance the last values of the time window are the
        #  most interesting ones for LRP prediction)
        if self.relevant_indices != None:# For each time index
            data = data[self.relevant_indices,:]

        # Add the contribution of this data sample to the 
        # respective covariance matrix:
        # cov = const * sum X_i*X_i^t
        X=data.view(numpy.ndarray)
        self.covariances[label] += numpy.dot(X.T, X) 
        
        # Count the number of samples per class
        self.samples_per_condition[label] = \
                     self.samples_per_condition.get(label, 0) + 1
    
    def _stop_training(self, debug=False):
        """
        Finish the training, i.e. solve the generalized eigenvalue problem
        Sigma_1*x = lambda*Sigma_2*x where Sigma_1 and Sigma_2 are the class
        covariance matrices and lambda and x are a eigenvalue, eigenvector pair.
        """
        sum_of_covariances = numpy.zeros((self.number_of_channels,
                                             self.number_of_channels)) 
        # Normalize the empirical covariance matrices  (i.e. divide by the 
        # number of samples per class) and compute the sum of all (two)
        # covariance matrices
        class_labels = self.covariances.keys()
        class_labels.sort()
        for label in class_labels:
            self.covariances[label] /= self.samples_per_condition[label]
            sum_of_covariances += self.covariances[label]
        
        # Solve the generalized eigenvalue problem to obtain the CSPs
        # NOTE:  It doesn't matter which of the two covariance matrix
        #        is passed as first argument, we simply pick the first...
        # numpy does not support generalized eigenvalue decomposition
        (eigenvalues, unordered_filters) = \
             scipy.linalg.eig(self.covariances[class_labels[0]],
                                 sum_of_covariances)

        # Sort filters according to eigenvalues
        preordered_filters = unordered_filters[:, numpy.argsort(-eigenvalues)]
        # We resort everything according to importance 
        self.new_order=[]
        for i in range(self.number_of_channels/2):
            self.new_order.append(i)
            self.new_order.append(self.number_of_channels-1-i)
        self.filters = preordered_filters[:, self.new_order]

    def _execute(self, data):
        """ Apply the learned spatial filters to the given data point. """
        # We must have computed the common spatial patterns, before
        # we can project the data onto the CSP subspace
        assert(self.filters != None)

        # If retained_channels not specified, retain all
        if self.retained_channels in [None, 'None']:
            self.retained_channels = self.number_of_channels

        if self.channel_names is None:
            self.channel_names = data.channel_names

        if len(self.channel_names)<self.retained_channels:
            self.retained_channels = len(self.channel_names)
            self._log("To many channels chosen for the retained channels! Replaced by maximum number.",level=logging.CRITICAL)

        if self.new_order is None:
            # We resort everything according to importance 
            self.new_order=[]
            for i in range(self.number_of_channels/2):
                self.new_order.append(i)
                self.new_order.append(self.number_of_channels-1-i)

        if self.spatio_temporal:
            orig_shape = data.shape
            data = data.reshape(1, data.shape[0] * data.shape[1])
        # Project the data using the learned CSP
        projected_data = numpy.dot(data, 
                                      self.filters[:, :self.retained_channels])
        if self.spatio_temporal:
            projected_data = data.reshape(orig_shape[0], orig_shape[1])
        
        if self.new_channel_names is None:
            self.new_channel_names = ["csp%03d" % i for i in self.new_order]
        
        return TimeSeries(projected_data, self.new_channel_names[:self.retained_channels],
                          data.sampling_frequency, data.start_time,
                          data.end_time, data.name, data.marker_name)

    def store_state(self, result_dir, index=None): 
        """ Stores this node in the given directory *result_dir* """
        if self.store or self.visualize_pattern:
            node_dir = os.path.join(result_dir, self.__class__.__name__)
            create_directory(node_dir)
        if self.store:
            # This node only stores the learned CSP patterns
            name = "%s_sp%s.pickle" % ("patterns", self.current_split)
            result_file = open(os.path.join(node_dir, name), "wb")
            result_file.write(cPickle.dumps(self.filters, 
                                            protocol=cPickle.HIGHEST_PROTOCOL))
            result_file.close()
            # Store spatial filter plots if desired
        if self.visualize_pattern:
            CSPNode._store_spatial_filter_plots(self.filters,
                                                self.channel_names,
                                                node_dir)

    @staticmethod
    def _store_spatial_filter_plots(filters, channel_names, result_dir):
        """ Store spatial filter plots in the *result_dir*"""
        # TODO: Move to common spatial filter superclass 
        import pylab
        # For all filters
        for filter_index in range(filters.shape[1]):
            # Clear figure
            pylab.gcf().clear()
            # The i-th filter 
            z = filters[:, filter_index] * 1.0
            # Plot the filter
            CSPNode._plot_spatial_values(pylab.gca(), z, channel_names,
                                         title='Filter %s' % filter_index)
            pylab.savefig("%s%sfilter%02d.png" % (result_dir, os.sep, filter_index))

    @staticmethod
    def _plot_spatial_values(ax, spatial_values, channel_names, title=""):
        # TODO: Move to common spatial filter superclass 
        import pylab
        
        ec_2d = StreamDataset.project2d(StreamDataset.ec)
        
        # Define x and y coordinates of electrodes in the order of the channels
        # of data
        x = numpy.array([ec_2d[key][0] for key in channel_names])
        y = numpy.array([ec_2d[key][1] for key in channel_names])
        
        # define grid.
        xi = numpy.linspace(-150, 150, 100)
        yi = numpy.linspace(-125, 125, 100)
        
        # grid the data.
        zi = pylab.griddata(x, y, spatial_values, xi, yi)
        
        # contour the gridded data, plotting dots at the nonuniform data points.
        ax.contour(xi, yi, zi, 15, linewidths=0.5, colors='k')
        CS = ax.contourf(xi, yi, zi, 15, cmap=pylab.cm.jet)
        
        cb = pylab.colorbar(CS, ax=ax)
        
        # plot data points.
        pylab.scatter(x, y, marker='o', c='b', s=5)
        # Add channel labels
        for label, position in ec_2d.iteritems():
            if label in channel_names:
                ax.text(position[0], position[1], label, fontsize='x-small')
                            
        ax.set_xlim(-125, 125)
        ax.set_ylim(-100, 100)
        
        ax.text(0, 80, title, fontweight='bold', horizontalalignment='center', 
                verticalalignment='center')
