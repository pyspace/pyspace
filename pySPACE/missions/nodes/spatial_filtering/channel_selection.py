""" Select a subset of concrete specified channels """
import warnings
import numpy

from pySPACE.missions.nodes.base_node import BaseNode
from pySPACE.resources.data_types.time_series import TimeSeries


class ChannelNameSelectorNode(BaseNode):
    """ Project onto a subset of channels based on their name
    
    This node reduces the number of channels of the signal by projecting onto
    a subset of channels that are selected explicitly by the user via their 
    name.

    **Parameters**
        :selected_channels: List of the names of the selected channels that will be 
                            retained (if inverse = False), otherwise the 
                            number of channels that are removed

        :inverse: Determines whether *selected_channels* are the channels
                  that are retained (inverse=False) or the channels 
                  that are removed (inverse=True).

                  (*optional, default: False*)

        :load_path: Advanced: If a load_path is passed, the channel names can
                        be read from a file. The path may contain the keywords
                        __RUN__ and __SPLIT__. These will be replaced by the
                        corresponding values at runtime (the Keyword
                        __INPUT_DATASET__ can also be used). If this option
                        is used, do NOT specify any channels in the
                        selected_channels parameter. Instead, pass an integer
                        n. The first n channels from the file will then be
                        used. Pass None to use all channels from the file.
                        
                        (*optional, default: None*)
    
    .. todo:: Integrate same functionality as in :class:`~pySPACE.missions.nodes.feature_selection.feature_filter.FeatureNameFilterNode`.
    
    **Exemplary Call**
    
    .. code-block:: yaml
    
        -
            node : ChannelNameSelector
            parameters:
                selected_channels : ["TP7", "TP8"]
                inverse : True
    
    :Author: Jan Hendrik Metzen (jhm@informatik.uni-bremen.de)
    :Created: 2009/03/18 
    """
    def __init__(self, selected_channels, inverse = False,
                 **kwargs):
        super(ChannelNameSelectorNode, self).__init__(**kwargs)
        
        self.set_permanent_attributes(selected_channel_names = selected_channels,
                                      inverse = inverse,
                                      retained_channel_indices = None)

    def compute_retained_channel_indices(self,data):
        """Determine the indices of the selected channels by their names"""
        # First check if all selected channels actually appear in the data
        not_found_channels = \
            [channel_name for channel_name in self.selected_channel_names 
                     if channel_name not in data.channel_names]
        if not not_found_channels == []:
            warnings.warn("Couldn't find selected channel(s): %s. Ignoring." % 
                            not_found_channels, Warning)

        if not self.inverse:
            self.selected_channel_names = \
                [channel_name for channel_name in data.channel_names 
                             if channel_name in self.selected_channel_names]
            self.retained_channel_indices = \
                [data.channel_names.index(channel_name) 
                    for channel_name in self.selected_channel_names]
        else:
            self.selected_channel_names = [channel_name for channel_name in data.channel_names 
                                                 if channel_name not in self.selected_channel_names] 
            self.retained_channel_indices = \
                    [data.channel_names.index(channel_name) 
                                 for channel_name in self.selected_channel_names]
            self.inverse = False

    def project_data(self,data):
        """ Project the data set on to the channels that will be retained """
        # Note: We have to create a new array since otherwise the removed  
        #       channels remains in memory
        projected_data = numpy.array(data.view(numpy.ndarray)[:, self.retained_channel_indices])

        # Create new TimeSeries object
        projected_time_series = TimeSeries(projected_data,
                                           self.selected_channel_names,
                                           data.sampling_frequency,
                                           data.start_time,
                                           data.end_time,
                                           data.name, 
                                           data.marker_name)

        return projected_time_series

    def _execute(self, data, n = None):
        # check if the channel names are computed, if not, compute them
        if self.retained_channel_indices == None:
            # if a load path is given, the channels stored in the path will
            # override the selected_channels from YAML.
            if self.load_path is not None:
                nr_of_channels = self.selected_channel_names
                selected_channels = \
                    __import__("yaml").load(open(self.load_path).read())
                if nr_of_channels not in [None, 'None']: #crop
                    self.selected_channel_names= \
                        selected_channels[:nr_of_channels]
                else: # use all
                    self.selected_channel_names = selected_channels
                        
            self.compute_retained_channel_indices(data)
                 
        # Create new TimeSeries object        
        projected_time_series = self.project_data(data)
        
        return projected_time_series


_NODE_MAPPING = {"Channel_Name_Selector": ChannelNameSelectorNode,
                "CNS": ChannelNameSelectorNode,
                }
