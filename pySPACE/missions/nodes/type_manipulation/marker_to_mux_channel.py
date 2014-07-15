""" Extract markers of time series object and convert them to a normal data channel
"""

import numpy
import copy

from pySPACE.missions.nodes.base_node import BaseNode


class MarkerToMuxChannelNode(BaseNode):
    """Extract markers of time series object and convert them to a normal data channel.

    Uses the marker_name attribute of the time series object.
    This attribute is created by the :mod:`~pySPACE.missions.support.windower.MarkerWindower`.

    **Exemplary Call**

    .. code-block:: yaml

        -
            node : Marker_To_Mux_Channel

    :Authors: Hendrik Woehrle (hendrik.woehrle@dfki.de)
    :Created: 2012/04/25
    """
    input_types = ["TimeSeries"]

    def __init__(self,**kwargs):
        super(MarkerToMuxChannelNode, self).__init__(*kwargs)

    def _execute(self, data):
        """ Perform conversion
        """
        num_time_points = data.shape[0]

        marker_channel = numpy.zeros((num_time_points,1))

        if hasattr(data,"marker_name") and data.marker_name != None:
            for (marker_name, times) in data.marker_name.iteritems():
                if marker_name.startswith("R"):
                    marker_name_letter_num = 255
                    marker_num = marker_name.partition("R")[2]
                else:
                    marker_name_letter_num = 0
                    marker_num = marker_name.partition("S")[2]
                for time in times:
                    marker_index = numpy.floor(time/1000. * data.sampling_frequency)
                    marker_channel[marker_index] = int(marker_num) + marker_name_letter_num

        muxed_data = numpy.hstack((data,marker_channel))
        channel_names = copy.deepcopy(data.channel_names)
        channel_names.append("Marker")

        mux_marker_data = data.replace_data(data,muxed_data,channel_names=channel_names)

        self.output_dim = mux_marker_data.shape[1]

        return mux_marker_data

    def get_output_type(self, input_type, as_string=True):
        if as_string:
            return "TimeSeries"
        else:
            return self.string_to_class("TimeSeries")

_NODE_MAPPING = {"Marker_To_Mux_Channel": MarkerToMuxChannelNode}
