""" Multiply a signal with a window """
import warnings

import numpy

# These imports are all for loading the functions.yaml file for the abbreviations of functions for benchmarking.
# Kept here local for debugging and because the file is just used in this node.
# Maybe this will change.
try:
    import os
    import yaml
    import pySPACE
except:
    pass

from pySPACE.missions.nodes.base_node import BaseNode
from pySPACE.resources.data_types.time_series import TimeSeries


class InvalidWindowException(Exception):
    pass


class WindowFuncNode(BaseNode):
    """ Multiply the :class:`~pySPACE.resources.data_types.time_series.TimeSeries` with a window
    
    If the window has trailing zeros, the time series is chopped.
    
    **Parameters**
        :window_function_str:
            This string has to be either the name of a function specified in 
            functions.yaml or a lambda expression that evaluates to a valid 
            window function. Such a window function has to be of the form 
            lambda n: lambda x: something
            where n is the number of samples (the length of the window 
            function) and x is the respective value.
        
        :reduce_window:
            If True, zeros at the beginning or ending are chopped.
            
            (*optional, default: False*)
    
    **Exemplary call**
    
    .. code-block:: yaml
    
        -
            node : Windowing
            parameters : 
                window_function_str : "hanning" # loaded from functions.yaml

    :Author: Jan Hendrik Metzen (jhm@informatik.uni-bremen.de)
    :Created: 2008/09/01
    :Revised: 2009/09/15 (Mario Krell)
    """
    def __init__(self, window_function_str, reduce_window = False, **kwargs):
                
        super(WindowFuncNode, self).__init__(**kwargs)
        # Do the import of the abbreviations of the functions.
        if not window_function_str.startswith("lambda"):
            try:

                functions_file = open(os.path.join(pySPACE.configuration.spec_dir,
                                                   'functions.yaml'), 'r')
                functions = yaml.load(functions_file)
                functions_file.close()
            except AttributeError:
                # Running outside of pySPACE, not able to load functions from YAML file
                # TODO: Fix that
                functions = {}
                warnings.warn("Function in spec folder could not be loaded. Please fix that!")
            try:
                window_function_str = functions[window_function_str]
            except KeyError:
                # window_function_str is not the key for a function in functions.yaml,
                # we assume that it is the window function itself
                pass
        
        self.set_permanent_attributes(window_function_str = window_function_str,
                                      reduce_window = reduce_window,
                                      num_of_samples = None,
                                      window_array = None)

    def create_window_array(self):
        """ Create a permanent array for the windowing of the data"""
        # the window is given as a lambda expression where the first variable 
        # is the length of the window (num of samples of the data) and the 
        # second one is for the time axis
        
        # resolve first variable
        window_function = eval(self.window_function_str)(self.num_of_samples)
        # resolve second variable for final window creation
        self.window_array = numpy.array([window_function(i) for i in \
                                                   range(self.num_of_samples)])
        
        # Check if there are zeros at the beginning or end of window
        # If yes, skip these ranges (i.e. shorten the time series window)
        self.window_not_equal_zero = numpy.where(self.window_array != 0)[0]
        self.window_has_zeros = (len(self.window_not_equal_zero) != \
                                                           self.num_of_samples)
        
        # A window with only zeros does not make sense
        if len(self.window_not_equal_zero) == 0:
            raise InvalidWindowException("The window does contain only zeros!\n"+
                "Function_str: %s\nn_samples: %d" % (self.window_function_str, 
                                                     self.num_of_samples))
    
    def _execute(self, data):
        """ Apply the windowing to the given data and return the result """        
        #Create a window of the correct length for the given data
        if self.num_of_samples is None:
            self.num_of_samples = data.shape[0]
            self.create_window_array()
             
        data_array=data.view(numpy.ndarray)
        #Do the actual windowing
        # TODO: check if windowed_data = (self.window_array.T * data) works also???
        windowed_data = (self.window_array * data_array.T).T
        
        # Skip trailing zeros
        if self.window_has_zeros and self.reduce_window:
            windowed_data = windowed_data[
                range(self.window_not_equal_zero[0],
                      self.window_not_equal_zero[-1] + 1), :]
        
            result_time_series = TimeSeries.replace_data(data, windowed_data)
            
            # Adjust start and end time when chopping was done
            result_time_series.start_time = data.start_time + \
                self.window_not_equal_zero[0] * 1000.0 / data.sampling_frequency
            result_time_series.end_time = \
                data.end_time - (data.shape[0] - self.window_not_equal_zero[-1]
                                 - 1) * 1000.0 / data.sampling_frequency
        else:
            result_time_series = TimeSeries.replace_data(data, windowed_data)
                    
        return result_time_series


class ScaleNode(BaseNode):
    """ Scale all value by a constant factor

    Scales (i.e. multiplies) all values with a given factor.

    **Parameters**

        :factor:
            The factor

    **Exemplary Call**

    .. code-block:: yaml

        -
            node : Scale
            parameters :
                factor : 2

    :Authors: Hendrik Woehrle (hendrik.woehrle@dfki.de)
    :Created: 2013/03/08
    """
    def __init__(self,
                 factor=1,
                 **kwargs):
        super(ScaleNode, self).__init__(**kwargs)

        if type(factor) == str:
            factor = eval(factor)

        self.set_permanent_attributes(factor=factor)

    def _execute(self, data):
        """
        Apply the scaling to the given data x
        and return a new time series.
        """
        x = data.view(numpy.ndarray).astype(numpy.double)
        x = x * self.factor

        result_time_series = TimeSeries.replace_data(data, x)

        return result_time_series

_NODE_MAPPING = {"Windowing": WindowFuncNode,
                }