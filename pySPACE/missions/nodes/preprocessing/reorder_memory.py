""" Optimize the underlying memory structure

This module contains nodes that reorder the data in the underlying memory for 
performance reasons.

"""

import numpy

from pySPACE.missions.nodes.base_node import BaseNode
from pySPACE.resources.data_types.time_series import TimeSeries


class ReorderMemoryNode(BaseNode):
    """ Reorder the memory of the :class:`~pySPACE.resources.data_types.time_series.TimeSeries`
        
    This node reorders the memory from c-style coalescing to Fortran-style 
    coalescing.

    .. todo:: Explain, why this should increase performance and how the
              'underlying node' is specified!

    **Parameters**
        :convert_type:
            Convert the type of time series entries to float.

            (*optional, default: True*)
    
    **Exemplary call**
    
    .. code-block:: yaml
    
        -
            node : ReorderMemory
            parameters :
                convert_type : True

    """
    def __init__(self,
                 convert_type = True,
                 **kwargs):
            
        super(ReorderMemoryNode, self).__init__(**kwargs)
        
        self.set_permanent_attributes(convert_type = convert_type)

    def _execute(self, data):
        """ Reorder the memory. """

        # exchange data of time series object to correctly ordered data
        buffer = numpy.array(data, order='F')

        if self.convert_type and numpy.dtype('float64') != buffer.dtype:
            buffer = buffer.astype(numpy.float)
        
        data = TimeSeries.replace_data(data,buffer)
        
        return data

