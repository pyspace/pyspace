""" Fisher's Discriminant Analysis and variants for spatial filtering """

import os
import cPickle

try:
    import mdp
except:
    pass

import numpy

from pySPACE.missions.nodes.spatial_filtering.spatial_filtering import SpatialFilteringNode
from pySPACE.resources.data_types.time_series import TimeSeries

from pySPACE.tools.filesystem import  create_directory

import logging

class FDAFilterNode(SpatialFilteringNode):
    """ Reuse the  implementation of Fisher's Discriminant Analysis provided by mdp
    
    This node implements the supervised fisher's discriminant
    analysis algorithm for spatial filtering.

    **Parameters**
        :retained_channels: Determines how many of the FDA pseudo channels
            are retained. Default is None which means "all channels".

            (*optional, default: None*)

        :load_path: An absolute path from which the FDA filter 
            is loaded.
            If not specified, this matrix is learned from the training data.

            (*optional, default: None*)

    **Exemplary Call**
    
    .. code-block:: yaml
    
        -
            node : FDAFilter
            parameters:
                retained_channels : 42
    
    :Author: Jan Hendrik Metzen (jhm@informatik.uni-bremen.de)
    :Created: 2010/02/17
    """
    def __init__(self, retained_channels=None, load_path=None, **kwargs):
        
        # Must be set before constructor of superclass is set
        self.trainable = (load_path == None)        
        
        super(FDAFilterNode, self).__init__(**kwargs)
        
        # Load patterns from file if requested
        filters = None
        if load_path != None:
            filters_file = open(load_path, 'r')
            filters = cPickle.load(filters_file)
        
        self.set_permanent_attributes(
            trainable = self.trainable,
            # The number of channels that will be retained
            retained_channels=retained_channels,
            
            # Gather all data instances passed during training
            data=None,
            
            # Remember the classes of the data 
            labels=None,
                        
            # After training is finished, this node will contain
            # a projection matrix that is used to project
            # the data onto a lower dimensional subspace
            filters=filters,
            
            new_channel_names = None,
            channel_names = None
        )
    
    def is_trainable(self):
        """ Returns whether this node is trainable. """
        return self.trainable
    
    def is_supervised(self):
        """ Returns whether this node requires supervised training """
        return self.trainable

    def _train(self, data, label):
        """ Remember *data* and *label* for later learning of filters."""
        if self.channel_names is None:
            self.channel_names = data.channel_names
        # Simply gather all data and do the actual training in _stop_training
        if self.data == None:
            self.data = 1.0 * data
            self.labels = [label for i in range(data.shape[0])]
            self.channel_names = data.channel_names
        else:
            self.data = 1.0 * numpy.vstack([self.data, data])
            self.labels.extend([label for i in range(data.shape[0])])

    def _stop_training(self, debug=False):
        # Uses collected data to learn a transformation matrix using LDA
        fda_node = mdp.nodes.FDANode()
        fda_node.train(self.data, numpy.array(self.labels))
        fda_node.stop_training()
        fda_node.train(self.data, numpy.array(self.labels))
        fda_node.stop_training()
        self.filters = fda_node.v

    def _execute(self, data):
        """ Execute learned transformation on *data*."""
        # We must have computed the projection matrix
        assert(self.filters != None)
        
        if self.retained_channels==None:
            self.retained_channels = data.shape[1]
        if self.channel_names is None:
            self.channel_names = data.channel_names

        if len(self.channel_names)<self.retained_channels:
            self.retained_channels = len(self.channel_names)
            self._log("To many channels chosen for the retained channels! Replaced by maximum number.",level=logging.CRITICAL)
        # Project the data using the learned FDA
        projected_data = numpy.dot(data,
                                      self.filters[:, :self.retained_channels])
        if self.new_channel_names is None:
            self.new_channel_names = ["fda%03d" % i 
                                for i in range(self.retained_channels)]

        return TimeSeries(projected_data, self.new_channel_names,
                          data.sampling_frequency, data.start_time,
                          data.end_time, data.name, data.marker_name)

    def store_state(self, result_dir, index=None): 
        """ Stores the projection in the given directory *result_dir* """
        if self.store:
            node_dir = os.path.join(result_dir, self.__class__.__name__)
            create_directory(node_dir)
            name = "%s_sp%s.pickle" % ("projection", self.current_split)
            result_file = open(os.path.join(node_dir, name), "wb")
            result_file.write(cPickle.dumps(self.projection, protocol=2))
            result_file.close()


_NODE_MAPPING = {"FDAFilter": FDAFilterNode,
                "FDA": FDAFilterNode}
