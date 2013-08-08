""" Basic methods for spatial filtering

Spatial filtering means to combine information from mostly similar sensors,
distributed in space.
The main principle is to define a linear filter in a training phase,
which tries to combine the sensor information to erase noise
and compress the relevant information.
"""

import numpy
from pySPACE.missions.nodes.base_node import BaseNode
from pySPACE.resources.data_types.time_series import TimeSeries

class SpatialFilteringNode(BaseNode):
    """ Base class for spatial filters and simple channel reduction
    
    This class is superclass for nodes that implement spatial filtering.
    It contains functionality that is common to all spatial filters.
    
    It can also be used directly as node which retains just the first
    N channels. This is typically only reasonable if the channel ordering
    is somehow meaningful.
    
    This class shall unify processing steps like:

        - execution of the linear transformation on the data
        - ranking of sensor channels dependent on the weights in the filter (done)
        - initialization of the nodes
        - and visualization, if the sensors are EEG electrodes.
        
    .. todo::   Also make storing and loading of *self.filter* variable and
                execution consistent and part of this node.
    
    .. todo:: Implement the mentioned functionality in this documentation
    
    **Parameters**
      :retained_channels: The number of channels that are retained.
           If this quantity is not defined, all channels are retained.

           (*optional, default: None*)

    **Exemplary Call**
    
    .. code-block:: yaml
    
        -  
            node : SpatialFiltering
            parameters :
                  retained_channels : 8

    :Author: Jan Hendrik Metzen (jhm@informatik.uni-bremen.de)
    :Created: 2011/11/22
    """
    def __init__(self, retained_channels=None,  **kwargs):
        super(SpatialFilteringNode, self).__init__(**kwargs)

        self.set_permanent_attributes(
            retained_channels = retained_channels,
            filters = None,
            channel_names = None,
            filter_channel_names = None)
    
    def _execute(self, data):
        if self.retained_channels is None:
            self.retained_channels = len(data.channel_names)
        if self.channel_names is None:
            self.channel_names = data.channel_names
        if self.filters is None:
            return TimeSeries(data[:, :self.retained_channels],
                              data.channel_names[:self.retained_channels],
                              data.sampling_frequency, data.start_time,
                              data.end_time, data.name, data.marker_name)
        else:
            data_array=data.view(numpy.ndarray)
            projected_data = numpy.dot(data_array, self.filters[:, :self.retained_channels])
            if self.filter_channel_names is None:
                filter_channel_names = None
            else:
                filter_channel_names = self.filter_channel_names[:self.retained_channels]
            return TimeSeries(projected_data,
                              filter_channel_names,
                              data.sampling_frequency, data.start_time,
                              data.end_time, data.name, data.marker_name)

    def get_sensor_ranking(self):
        """ Special Code for the spatial filter
        
        Take a maximum number of ranking channels and add the channel weights 
        
        Channels with the highest weight are the most important.
        
        Should work for xDAWN, CSP, FDA, PCA, ICA
        """
        filters = self.get_filters()
        ranking_matrix = numpy.absolute(filters[:, :self.retained_channels]).mean(axis=1)
        ranking=[]
        for i in range(len(self.channel_names)):
            ranking.append((self.channel_names[i],ranking_matrix[i]))
        ranking.sort()
        return sorted(ranking, key=lambda t: t[1])

    def get_filters(self):
        return self.filters
