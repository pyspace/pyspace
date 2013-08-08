""" Clip all values to a certain range of values """

import numpy

from pySPACE.missions.nodes.base_node import BaseNode
from pySPACE.resources.data_types.time_series import TimeSeries

class ClipNode(BaseNode):
    """ Clip all values to a certain range of values

    Clip all values to a certain range of values.
    The values above and below that range are set to the maximum/minimum value.

    **Parameters**

        :min_threshold:
            All values below this threshold are set to min_threshold.

            (*optional, default: -numpy.inf*)

        :max_threshold:
            All values above this threshold are set to max_threshold.

            (*optional, default: numpy.inf*)

    **Exemplary Call**

    .. code-block:: yaml

        -
            node : Clip
            parameters :
                min_threshold : -250
                max_threshold : 250

    :Authors: Hendrik Woehrle (hendrik.woehrle@dfki.de)
    :Created: 2013/03/08
    """
    def __init__(self,
                 min_threshold = None,
                 max_threshold = None,
                 **kwargs):
        super(ClipNode, self).__init__(**kwargs)

        if type(min_threshold) == str:
            factor = eval(min_threshold)

        if type(max_threshold) == str:
            factor = eval(max_threshold)

        self.set_permanent_attributes(min_threshold = min_threshold,
                                      max_threshold = max_threshold)

    def _execute(self, data):
        """
        Apply the scaling to the given data x
        and return a new time series.
        """
        x = data.view(numpy.ndarray)

        x.clip(self.min_threshold, self.max_threshold, out = x)

        result_time_series = TimeSeries.replace_data(data, x)

        return result_time_series

