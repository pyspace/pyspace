""" Change the reference of an EEG signal
"""

import numpy
import warnings

from pySPACE.missions.nodes.base_node import BaseNode
from pySPACE.resources.data_types.time_series import TimeSeries
from pySPACE.resources.dataset_defs.stream import StreamDataset 

class InvalidWindowException(Exception):
    pass

class LaplacianReferenceNode(BaseNode):
    """ Apply the Laplacian spatial filter 
    
    It derives from the need of improving the spatial resolution of EEG.
    The signal recorded at each electrode is a combination of the brain activity
    immediately underneath it and of brain activity of neighboring areas.
    
    The idea is to filter from each electrode the contribution coming from
    its neighbors. It can be applied using the nearest neighboring electrodes
    (small Laplacian: 4 channels) or nearest and next nearest neighboring 
    electrodes (big Laplacian: 8 channels).
    
    The number of electrodes of *the returned time series is reduced*: 
    each electrode that has less than 4 (or 8 when the big Laplacian is applied)
    neighbors is excluded.
    
    **References**
    
        ======== ====================================================================================
                 main source: original article
        ======== ====================================================================================
        author   Hjorth, Bo
        title    An on-line transformation of EEG scalp potentials into orthogonal source derivations
        journal  Electroencephalography and Clinical Neurophysiology
        year     1975
        volume   39
        number   5
        pages    526--530
        doi      10.1016/0013-4694(75)90056-5
        ======== ====================================================================================
    
    **Parameters**
    
        :l_type:
            type of Laplacian applied, e.g. 'small' or 'big'
            
            (*optional, default: 'small'*)
        
        :selected_channels:
            A list of channel names for which the filter should be applied.
            If None, all channels are considered.
        
            (*optional, default: None*)

        
    **Exemplary Call**
    
    .. code-block:: yaml
    
        -
            node: LaplacianReference
            parameters:
                l_type: 'big'
     
     
    :Author: Laura Manca (laura.manca89@gmail.com)
    :Created: 20013/09/24           
    """
    
    def __init__(self, selected_channels=None, l_type='small', **kwargs):
        super(LaplacianReferenceNode, self).__init__(**kwargs)
        
        self.set_permanent_attributes(selected_channels=selected_channels,
                                      l_type=l_type,
                                      dist_max=0.28,
                                      dist=None
                                      )
    
    def calc_distance_matrix(self, data, distFunc=lambda deltaPoint: \
                             numpy.sqrt(sum(deltaPoint[d]**2 \
                             for d in xrange(len(deltaPoint))))):
        """ Compute the distance matrix from the dictionary StreamDataset.ec
        
        StreamDataset.ec maps the coordinates of each electrode and to the 
        respective electrode name.
        """
        # Rearrange the coordinates according to the order in data
        nDimPoints = numpy.zeros((len(self.selected_channels),3))
        for ind, name in enumerate(self.selected_channels):
            nDimPoints[ind,:] = StreamDataset.ec[name][:]
        
        # compute the distances between all selected channels
        dim = len(nDimPoints[0])   # dim=3 (coordinates)
        delta = [None]*dim
        for d in xrange(dim):
            position = nDimPoints[:,d] #all x,y or z values
            delta[d] = position - numpy.reshape(position,(len(position),1))
        # matrix of distances from one electrode to any other
        self.dist = distFunc(delta)   
        return self.dist
    
    
    def compute_laplacian(self,data):
        """Compute the Laplacian
        
        .. math::
             
             \\text{filtered data}_{i} =
             \\text{data}_{i}*\\text{number of neighbours} -
             \\sum_{i} \\text{neighbours of data}_{i}
             \\text{with } i = EEG channel
                
        The channels that are at the borders or close to Ref are excluded  
        """
        idx = numpy.argsort(self.dist)
        #compute the Laplacian
        filt_data = data * self.l_type - data[:,idx[:,1]] - \
                    data[:,idx[:,2]] - data[:,idx[:,3]] - data[:,idx[:,4]]
        
        if self.l_type == 8:
            filt_data = filt_data - data[:,idx[:,5]] - data[:,idx[:,6]] - \
                            data[:,idx[:,7]] - data[:,idx[:,8]]
        #remove unbalanced channels (either borders or electrodes close to Ref)
        if self.l_type == 4:
            nearest = idx[ : , 0 : (self.l_type + 1)]
            unbalanced = []
            balanced = []
            balanced_ch_names = []
            for ch in range(len(nearest)):
                x = [data.channel_names[ch] for i in \
                     self.dist[ch,nearest[ch,:]]if i > self.dist_max]
                if x != []:
                    unbalanced.append(x[:1])
                else:
                    balanced.append(ch)
                    balanced_ch_names.append(filt_data.channel_names[ch])
        elif self.l_type == 8:
            nearest = idx[ : , 0 : (self.l_type + 1)]
            unbalanced = []
            balanced = []
            balanced_ch_names = []
            for ch in range(len(nearest)):
                x = [data.channel_names[ch] for i in \
                     self.dist[ch,nearest[ch,:]] if i > self.dist_max]
                if x != []:
                    unbalanced.append(x[:1])
                else:
                    balanced.append(ch)
                    balanced_ch_names.append(filt_data.channel_names[ch])
        
        # list of channels left after the Laplacian filter being applied 
        data_noborder = filt_data[:,balanced]   
        data_noborder.channel_names = balanced_ch_names 
         
        filtered_time_series = TimeSeries(data_noborder,
                                          data_noborder.channel_names,
                                          data.sampling_frequency,
                                          data.start_time,
                                          data.end_time,
                                          data.name,
                                          data.marker_name,
                                          data.tag)
        
        self._log("These channels are unbalanced (border or close to reference) "
                  "they will be removed from the data: %s" % str(unbalanced))
        return filtered_time_series
    
    def _execute(self, data):
        
        if self.selected_channels == None:
            self.selected_channels = data.channel_names
        # set dist_max according to the kind of chosen filter (big or small)
        if self.l_type == 'small':
            self.l_type = 4
            self.dist_max = 0.28
        elif self.l_type == 'big':
            self.l_type = 8
            self.dist_max = 0.4
        # check if the distance matrix has been already computed,
        # if not compute it
        if self.dist is None:
            self.calc_distance_matrix(data)
        # compute the Laplacian    
        filtered_time_series = self.compute_laplacian(data)
        return filtered_time_series

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

             (*optional, default: False*)

        :inverse:
             Determine whether *avg_channels* are the channels over which
             the average is computed (inverse=False) or the channels
             that are ignored when calculating the average.

             (*optional, default: False*)
        
        :old_ref:
             This is the old reference channel name usually used during 
             recording as a reference. After re-referencing and if keep_average 
             is set to true, this name will be used for the appended channel. 
             If keep_average is true, but old_ref is not specified, name of the 
             appended channel will be "avg".

        .. todo:: use different version from keeping the average values

    **Exemplary call**
    
    .. code-block:: yaml
    
        -
            node : Average_Reference
            parameters : 
                avg_channels : ["C3","C4"]
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
            result_time_series = TimeSeries(avg_referenced_data, 
                                            channel_names,
                                            data.sampling_frequency, 
                                            data.start_time, data.end_time,
                                            data.name, data.marker_name)
        else:
            result_time_series = TimeSeries.replace_data(data, 
                                                            avg_referenced_data)
        
        return result_time_series


_NODE_MAPPING = {"Average_Reference": AverageReferenceNode,
                "Laplacian_Reference": LaplacianReferenceNode}
    
