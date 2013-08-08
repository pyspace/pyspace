""" Build the difference of channels based on different criteria """
import numpy

from pySPACE.missions.nodes.base_node import BaseNode
from pySPACE.resources.data_types.time_series import TimeSeries

class HemisphereDifferenceNode(BaseNode):
    """ Build new EEG channels using the difference between left and right hemisphere of the brain
    
    The new channel names correspond to the difference.
    Channels in the middle area are totally ignored.
    Up to the standard 128 channels can be used.
    If some channels do not exist, they are ignored.

    **Parameters**

    **Exemplary Call**
    
    .. code-block:: yaml
    
        -
            node : HemisphereDifference

    :Author: Mario Krell (Mario.Krell@dfki.de)
    :Created: 2009/12/15
    """
    def __init__(self,
                 **kwargs):
        super(HemisphereDifferenceNode, self).__init__(**kwargs)
        
        # This is the dictionary giving the corresponding right channels to the
        # keys representing the channels of the left hemisphere.
        # The dictionary can be used with up to 128 electrodes.
        dual=dict(Fp1="Fp2", 
        AFp1="AFp2", 
        AF7="AF8", AF3="AF4",
        F9="F10", AFF1h="AFF2h", AFF5h="AFF6h",
        F7="F8", F5="F6", F3="F4", F1="F2", 
        FFT9h="FFT10h", FFT7h="FFT8h", FFC5h="FFC6h", FFC3h="FFC4h", FFC1h="FFC2h", 
        FT9="FT10", FT7="FT8",         FC5="FC6", FC3="FC4", FT1="FC2", 
        FTT9h="FTT10h", FTT7h="FTT8h", FCC5h="FCC6h", FCC3h="FCC4h", FTT1h="FCC2h", 
        T7="T8",                       C5="C6", C3="C4", C1="C2",
        TTP7h="TTP7h",                 CCP5h="CCP6h", CCP3h="CCP4h", CCP1h="CCP2h",
        TP9="TP10", TP7="TP8",         CP5="CP6", CP3="CP4", CP1="CP2",
        TPP9h="TPP10h", TPP7h="TPP8h", CPP5h="CPP6h", CPP3h="CPP4h", CPP1h="CPP2h", 
        P9="P10",
        P7="P8", P5="P6", P3="P4", P1="P2",
        PPO9h="PPO10h", PPO5h="PPO6h", PPO1h="PPO2h",
        PO7="PO8", PO3="PO4",
        POO9h="POO10h", POO1="POO2",
        O1="O2",
        I1="I2", OL1h="OL2h"
        )
        dual_list = dual.keys()
        self.set_permanent_attributes(dual = dual, dual_list = dual_list)

    def _execute(self, data):
        # Determine the new list of channel_names
        self.selected_channel_names = [channel_name+"-"+self.dual[channel_name] for channel_name in self.dual_list
                                                 if channel_name in data.channel_names and self.dual[channel_name] in data.channel_names]
        # Initialize the new array
        difference_data = numpy.zeros((len(data),len(self.selected_channel_names)))
        current_index = 0
        
        # Do the same check as in the determination of the channel names...
        for channel_name in self.dual_list:
            if channel_name in data.channel_names and self.dual[channel_name] in data.channel_names:
                first_channel_index = data.channel_names.index(channel_name)
                second_channel_index = data.channel_names.index(self.dual[channel_name])
                # ...and build the difference of the corresponding channels.
                difference_data[:,current_index] = data[:,first_channel_index] - data[:,second_channel_index]
                current_index +=1
        # Create new TimeSeries object        
        difference_time_series = TimeSeries(difference_data,
                                           self.selected_channel_names,
                                           data.sampling_frequency,
                                           data.start_time, data.end_time,
                                           data.name, data.marker_name)
        return difference_time_series


_NODE_MAPPING = {"Hemisphere_Difference": HemisphereDifferenceNode}
