""" Differentiate :class:`~pySPACE.resources.data_types.time_series.TimeSeries` channel wise

:Author: Marc Tabie (mtabie@informatik.uni-bremen.de)
:Created: 2010/01/20
"""
import numpy
from pySPACE.missions.nodes.base_node import BaseNode
from pySPACE.resources.data_types.time_series import TimeSeries

class Simple2DifferentiationNode(BaseNode):
   """ Calculate difference to previous time point to generate new :class:`~pySPACE.resources.data_types.time_series.TimeSeries`
   
    
    **Parameters**
        :datapoints: The indices of the data points that are used as features,
            If None, all data points are used.

            (*optional, default: None*)


    **Exemplary Call**
    
    .. code-block:: yaml
    
        -
            node : diff2

   """
   # Simple copy of the TimeDifferenceFeatureNode with slight change at the end.
   # More complex would be a short time regression between more than two points.
   # Yet the possibility to make it a "real" differentiation is in comment.
   def __init__(self, 
                datapoints = None,
                moving_window_length = 1,
                keep_number_of_samples = False, 
                **kwargs):
       super(Simple2DifferentiationNode, self).__init__(**kwargs)
       self.set_permanent_attributes(datapoints = datapoints,
                                     moving_window_length = moving_window_length,
                                     keep_number_of_samples = keep_number_of_samples) 

   def _execute(self, x):
       """
       f' = (f(x+h)-f(x))
       """
       if self.datapoints == None:
           self.datapoints = len(x)
       
       #create new channel names
       new_names = []
       for channel in range(len(x.channel_names)):
           new_names.append("%s'" %  (x.channel_names[channel]))
       #Derive the f' d2 from data x
       timeSeries = []
       for datapoint in range(self.datapoints):
           temp = []
           if((datapoint+1)<self.datapoints):
               for channel in range(len(x.channel_names)):
                   temp.append(x[datapoint+1][channel]-x[datapoint][channel])#*8*sampling_frequency
               timeSeries.append(temp)
       #padding with zero's if the original length of the time series have to remain equal.
       if self.keep_number_of_samples:
           temp = []
           for i in range(len(x.channel_names)):
               temp.append(0)
           timeSeries.append(temp)
       #Create a new time_series with the new data and channel names
       result_time_series = TimeSeries.replace_data(x, numpy.array(timeSeries))
       result_time_series.channel_names = new_names
       #if necessary adjust the length of the time series
       if not self.keep_number_of_samples:
           result_time_series.end_time -= 1
       
       return result_time_series


class Simple5DifferentiationNode(BaseNode):
   """ Calculate smoothed derivative using 5 time points

    Method taken from
    http://www.holoborodko.com/pavel/numerical-methods/numerical-derivative/smooth-low-noise-differentiators/

    .. math::

        f'(x)=\\frac{2(f(x+h)-f(x-h))-(f(x+2h)-f(x-2h))}{8h}

    Further smoothing functions are available, but seemingly not necessary,
    because we have already a smoothing of the signal when doing the subsampling.
    Dividing by 8h is omitted, because it is just multiplication
    with the same scalar each time.
    
    **Parameters**
        :datapoints: The indices of the data points that are used as features,
            If None, all data points are used.

            (*optional, default: None*)


    **Exemplary Call**
    
    .. code-block:: yaml
    
        -
            node : diff5
   """
   # Simple copy of the TimeDifferenceFeatureNode with slight change at the end.
   # More complex would be a short time regression between more than two points.
   # Yet the possibility to make it a "real" differentiation is in comment.
   def __init__(self, 
                datapoints = None,
                moving_window_length = 1,
                keep_number_of_samples = False, 
                **kwargs):
       super(Simple5DifferentiationNode, self).__init__(**kwargs)
       self.set_permanent_attributes(datapoints = datapoints,
                                     moving_window_length = moving_window_length,
                                     keep_number_of_samples = keep_number_of_samples) 

   def _execute(self, x):
       """ Calculate derivative

       .. math::

            f'(x)=\\frac{2(f(x+h)-f(x-h))-(f(x+2h)-f(x-2h))}{8h}
       """
       if self.datapoints == None:
           self.datapoints = len(x)
       #create new channel names
       new_names = []
       for channel in range(len(x.channel_names)):
           new_names.append("%s'" %  (x.channel_names[channel]))
       #Derive the f' d5 from data x
       timeSeries = []
       for datapoint in range(self.datapoints):
           temp = []
           if((datapoint+4)<self.datapoints):
               for channel in range(len(x.channel_names)):
                   temp.append(x[datapoint][channel]-2*x[datapoint+1][channel]+2*x[datapoint+3][channel]-x[datapoint+4][channel])#*8*sampling_frequency
               timeSeries.append(temp)
       #padding with zero's if the original length of the time series have to remain equal.
       if self.keep_number_of_samples:
           for i in range(4):
               temp = []
               for i in range(len(x.channel_names)):
                   temp.append(0)
               timeSeries.append(temp)
       #Create a new time_series with the new data and channel names
       result_time_series = TimeSeries.replace_data(x, numpy.array(timeSeries))
       result_time_series.channel_names = new_names
       #if necessary adjust the length of the time series
       if not self.keep_number_of_samples:
           result_time_series.end_time -= 4
       
       return result_time_series


_NODE_MAPPING = {"diff2": Simple2DifferentiationNode,
                "diff5": Simple5DifferentiationNode}

