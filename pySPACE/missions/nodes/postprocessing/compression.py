""" Compress the data

Used to decrease the input data via compression algorithms. 

"""

import numpy

from pySPACE.missions.nodes.base_node import BaseNode
from pySPACE.resources.data_types.feature_vector import FeatureVector

from pySPACE.missions.nodes.postprocessing.feature_normalization import InconsistentFeatureVectorsException

class RandomFeatureCompressionNode(BaseNode):
    """Compress the data using a number of random vectors
    
    Uses fixed random vector to compress the data into a smaller set of channels.

    **Parameters**
        :retained_channels:
            Number of pseudo output channels

            (*optional, default: 16*)
            
    **Exemplary Call**
    
    .. code-block:: yaml
    
        -
            node : RandomFeatureCompression

    :Authors: Hendrik Woehrle (hendrik.woehrle@dfki.de)
    :Created: 2012/03/16
    """
    def __init__(self, compression=.5, **kwargs):
        super(RandomFeatureCompressionNode, self).__init__(**kwargs)
        
        self.set_permanent_attributes(compression=compression,
                                      feature_names = [],
                                      w = None)

    def _execute(self, data):
        if self.compression < 0.9:
            data_array=data.view(numpy.ndarray)
            if self.feature_names == []:
                self.feature_names = data.feature_names
            elif self.feature_names != data.feature_names:
                raise InconsistentFeatureVectorsException("Two feature vectors used during training do not contain the same features!")
            
            if self.w == None:
                dim_after_compression = data_array.shape[1]*self.compression
                self.w = numpy.random.randn(data_array.shape[1],dim_after_compression)

            compressed_data = numpy.dot(data_array,self.w)
        
            return FeatureVector([(compressed_data[0,:])])
        else:
            return data
