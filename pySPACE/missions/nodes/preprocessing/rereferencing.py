""" Change the reference of an EEG signal

.. todo:: use different version from keeping the average values
"""

import numpy
import warnings

from pySPACE.missions.nodes.base_node import BaseNode
from pySPACE.resources.data_types.time_series import TimeSeries

class InvalidWindowException(Exception):
    pass

class AverageReferenceNode(BaseNode):
    """ Rereference EEG signal against the average of a selected set of electrodes
    
    This node computes for every time step separately the average of a selected 
    set of electrodes (*avg_channels*) and subtracts this average from each 
    channel. It thus implements a kind of average rereferencing.
    
    **Parameters**
        :avg_channels:
             the channels over which the average is computed

             (*optional, default: all available channels*)

        :keep_average:
             Whether the average should be added as separate channel.
             
             .. todo:: the result would be no time series. Check this!

             (*optional, default: False*)

        :inverse:
             Determine whether *avg_channels* are the channels over which
             the average is computed (inverse=False) or the channels
             that are ignored when calculating the average.

             (*optional, default: False*)
        
        :old_ref:
             This is the old reference channel name usually used during recording 
             as a reference. After re-referencing and if keep_average is set to 
             true, this name will be used for the appended channel. If keep_average 
             is true, but old_ref is not specified, name of the appended channel 
             will be "avg".

    **Exemplary call**
    
    .. code-block:: yaml
    
        -
            node : Average_Reference
            parameters : 
                avg_channels : ["EMG1","EMG2"] 
                keep_average : False
                inverse : True
                old_ref : "Fcz"

    :Author: Jan Hendrik Metzen (jhm@informatik.uni-bremen.de)
    :Created: 2009/09/28
    :Revised: 2013/03/25 Foad Ghaderi (foad.ghaderi@dfki.de)
    :For more details see: http://sccn.ucsd.edu/wiki/Chapter_04:_Preprocessing_Tools
    """
    def __init__(self, avg_channels = None, keep_average = False, old_ref = None,
                 inverse=False, **kwargs):
        super(AverageReferenceNode, self).__init__(*kwargs)
        
        self.set_permanent_attributes(avg_channels = avg_channels,
                                      keep_average =  keep_average,
                                      old_ref = old_ref,
                                      inverse=inverse)

    def _execute(self, data):
        # First check if all channels actually appear in the data

        # Determine the indices of the channels that are the basis for the 
        # average reference.
        if not self.inverse:
            if self.avg_channels == None:
                self.avg_channels = data.channel_names
            channel_indices = [data.channel_names.index(channel_name) 
                                for channel_name in self.avg_channels]
        else:
            channel_indices = [data.channel_names.index(channel_name)
                               for channel_name in data.channel_names
                               if channel_name not in self.avg_channels]

        not_found_channels = \
            [channel_name for channel_name in self.avg_channels 
                     if channel_name not in data.channel_names]
        if not not_found_channels == []:
            warnings.warn("Couldn't find selected channel(s): %s. Ignoring." % 
                            not_found_channels, Warning)
                    
        if self.old_ref is None:
            self.old_ref = 'avg'
        
        # Compute the actual data of the reference channel. This is the sum of all 
        # channels divided by (the number of channels +1).
        ref_chen = -numpy.sum(data[:, channel_indices], axis=1)/(data.shape[1]+1)
        ref_chen = numpy.atleast_2d(ref_chen).T
        # Reference all electrodes against average
        avg_referenced_data = data + ref_chen
        
        # Add average as new channel to the signal if enabled
        if self.keep_average:
            avg_referenced_data = numpy.hstack((avg_referenced_data, ref_chen))
            channel_names = data.channel_names + [self.old_ref]
#        for i in range(channels_names.shape[0]):
#            print 'channel %d: %s\n', i, channel_names[i]
            
        # Create new time series object and return it
        result_time_series = TimeSeries(avg_referenced_data, channel_names,
                                        data.sampling_frequency, data.start_time,
                                        data.end_time, data.name,
                                        data.marker_name)
        
        return result_time_series


_NODE_MAPPING = {"Average_Reference": AverageReferenceNode}
    
