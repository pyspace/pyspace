""" Independent Component Analysis variants """

import os
import cPickle
from copy import deepcopy
import numpy

try:
    import mdp
    from mdp.nodes import FastICANode
    import_error = False
except ImportError, e:
    import_error = e

from pySPACE.missions.nodes.spatial_filtering.spatial_filtering import SpatialFilteringNode
from pySPACE.resources.data_types.time_series import TimeSeries

from pySPACE.tools.filesystem import  create_directory

import logging

try:
    class FastICANodeWrapper(FastICANode):
        """The only reason for this node is to deal with the fact
        that the ICANode super class does not accept the output_dim kwarg
        """
        def __init__(self, trainable=True,*args, **kwargs):
            if "output_dim" in kwargs:
                kwargs.pop("output_dim")
            if trainable is False:
                self._trainable = False
            super(FastICANodeWrapper, self).__init__(*args, **kwargs)

        def is_training(self):
            """ Mapping to *self._training variable* """
            return self._training
except:
    pass


class ICAWrapperNode(SpatialFilteringNode): #, FastICANodeWrapper):
    """ Wrapper around the Independent Component Analysis filtering of mdp
    
    This Node implements the unsupervised independent component
    analysis algorithm for spatial filtering.

    **Parameters**
        :retained_channels: Determines how many of the ICA pseudo channels
            are retained. Default is None which means "all channels".
            
            (*optional, default: None*)

        :load_path: An absolute path from which the ICA filter 
            is loaded.
            If not specified, this matrix is learned from the training data.

            (*optional, default: None*)

    **Exemplary Call**
    
    .. code-block:: yaml
    
        -
            node : ICA
            parameters:
                retained_channels : 42
    """
    def __init__(self, retained_channels=None, load_path = None, **kwargs):
        if import_error:
            raise ImportError(import_error)
        # Must be set before constructor of superclass is set
        self.trainable = (load_path == None) 
        if "output_dim" in kwargs:
            kwargs.pop("output_dim")
        super(ICAWrapperNode, self).__init__(**kwargs)
        # Load filters from file if requested
        wrapped_node=None
        if load_path != None:
            filters_file = open(load_path, 'r')
            filters, white, whitened = cPickle.load(filters_file)
            wrapped_node = FastICANodeWrapper(trainable=False)
            wrapped_node.filters=filters
            wrapped_node.white = white
            wrapped_node.whitened = whitened
            wrapped_node._training = False
            wrapped_node._train_phase = -1
            wrapped_node._train_phase_started = False
            self.set_permanent_attributes(filters=filters, white=white,
                                          whitened=whitened)

        self.set_permanent_attributes(# The number of channels that will be retained
                                      retained_channels=retained_channels,
                                      # Determine whether this node is trainable
                                      trainable=(load_path == None),
                                      output_dim=retained_channels,
                                      new_channel_names = None,
                                      channel_names = None,
                                      wrapped_node=wrapped_node)

    def is_trainable(self):
        """ Returns whether this node is trainable. """
        return self.trainable
        
    def is_supervised(self):
        """ Returns whether this node requires supervised training. """
        return False
    
    def train(self, data, label=None):
        super(ICAWrapperNode, self).train(data)
    
    def _train(self, data, label = None):
        """ Uses *data* to learn a decomposition into independent components."""
        # We simply ignore the class label since we 
        # are doing unsupervised learning
        if self.channel_names is None:
            self.channel_names = data.channel_names
        if self.wrapped_node is None:
            self.wrapped_node = FastICANode()
        self.wrapped_node.train(data)
       

    def _execute(self, data):
        """ Execute learned transformation on *data*.
        
        Changes the base of the space in which the data is located so
        that the dimensions correspond to independent components
        """
        
        # If this is the first data sample we obtain
        if self.retained_channels == None:
            # Count the number of channels
            self.set_permanent_attributes(retained_channels = data.shape[1])
        if self.channel_names is None:
            self.channel_names = data.channel_names
        if len(self.channel_names)<self.retained_channels:
            self.retained_channels = len(self.channel_names)
            self._log("To many channels chosen for the retained channels! Replaced by maximum number.",level=logging.CRITICAL)
        if not(self.output_dim==self.retained_channels):
            # overwrite internal output_dim variable, since it is set wrong
            self._output_dim = self.retained_channels

        projected_data = self.wrapped_node.execute(data.view(numpy.ndarray)) #super(ICAWrapperNode, self)._execute(data)
        
        # Select the channels that should be retained
        # Note: We have to create a new array since otherwise the removed  
        #       channels remains in memory
        projected_data = numpy.array(projected_data[:, :self.retained_channels])
        
        if self.new_channel_names is None:
            self.new_channel_names = ["ica%03d" % i 
                                for i in range(projected_data.shape[1])]
        return TimeSeries(projected_data, self.new_channel_names,
                          data.sampling_frequency, data.start_time,
                          data.end_time, data.name, data.marker_name)
    
    def store_state(self, result_dir, index=None):
        """ Stores this node in the given directory *result_dir*. """
        if self.store:
            node_dir = os.path.join(result_dir, self.__class__.__name__)
            create_directory(node_dir)
            # This node only stores the learned eigenvector and eigenvalues
            name = "%s_sp%s.pickle" % ("filters", self.current_split)
            result_file = open(os.path.join(node_dir, name), "wb")
            result_file.write(cPickle.dumps((self.wrapped_node.filters, self.wrapped_node.white, self.wrapped_node.whitened),
                                             protocol=2))
            result_file.close()


    def get_filter(self):
        return self.get_projmatrix()


_NODE_MAPPING = {"ICA": ICAWrapperNode}
