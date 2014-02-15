#!/usr/bin/env python
""" Unittests that test window functions nodes

:Author: Jan Hendrik Metzen (jhm@informatik.uni-bremen.de)
:Created: 2008/09/01
"""


import unittest

import numpy

if __name__ == '__main__':
    import sys
    import os
    # The root of the code
    file_path = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(file_path[:file_path.rfind('pySPACE')-1])

from pySPACE.missions.nodes.preprocessing import window_func
from pySPACE.resources.data_types.time_series import TimeSeries

class WindowFuncTestCase(unittest.TestCase):
    
    def setUp(self):
        self.test_data = numpy.zeros((128, 3))
        self.test_data[:,1] = numpy.ones(128)
        self.test_data[:,2] = numpy.random.random(128)
        
        self.test_time_series = TimeSeries(self.test_data, ["A","B", "C"], 64,
                                           start_time = 0, end_time = 2000)
           
    def test_zero_window(self):
        """ Test that the window function [0 0 ... 0 0] raises an InvalidWindowException """
        window_function_str = "lambda n: lambda x: 0"
        node = window_func.WindowFuncNode(window_function_str = window_function_str)
        
        self.assertRaises(window_func.InvalidWindowException,
                          node.execute, self.test_time_series)
        
    def test_one_window(self):
        """ Test that the window function [1 1 ... 1 1] does not change the time series """
        window_function_str = "lambda n: lambda x: 1"
        node = window_func.WindowFuncNode(window_function_str = window_function_str)
        
        windowed_time_series = node.execute(self.test_time_series)
        
        self.assert_(numpy.all(windowed_time_series.view(numpy.ndarray) == self.test_time_series.view(numpy.ndarray)))
        
    
    def test_convolving(self):
        """ Test that convolving one with a window returns the window """
        window_function_str = """lambda n: lambda x: (1 - __import__("numpy").cos((x + 1) * __import__("numpy").pi/n))/2"""
        window_function = eval(window_function_str)
        window = numpy.array([window_function(self.test_time_series.shape[0])(i) 
                                for i in range(self.test_time_series.shape[0])])
        
        node = window_func.WindowFuncNode(window_function_str = window_function_str)
        
        windowed_time_series = node.execute(self.test_time_series)
        
        self.assert_(numpy.all(windowed_time_series.view(numpy.ndarray)[:,1] == window))
        
    def test_chopping(self):
        """ Test that the window function with trailing zeros chops the time series window """
        # Chopping at the start
        window_function_str = """lambda n: lambda x: 0 if x < 2 else 1"""
        node = window_func.WindowFuncNode(window_function_str = window_function_str,
                                                 reduce_window = True)
        
        windowed_time_series = node.execute(self.test_time_series)
        
        self.assert_(windowed_time_series.shape[0] + 2 == self.test_time_series.shape[0])
        self.assert_(numpy.all(windowed_time_series.view(numpy.ndarray) == self.test_time_series.view(numpy.ndarray)[2:,:]))
        
        # Chopping at the end
        window_function_str = """lambda n: lambda x: 0 if x >= n - 2 else 1"""
        node = window_func.WindowFuncNode(window_function_str = window_function_str,
                                                 reduce_window = True)
        
        windowed_time_series = node.execute(self.test_time_series)
        
        self.assert_(windowed_time_series.shape[0] + 2 == self.test_time_series.shape[0])
        self.assert_(numpy.all(windowed_time_series.view(numpy.ndarray) == self.test_time_series.view(numpy.ndarray)[:-2,:]))
        
if __name__ == '__main__':  
    suite = unittest.TestLoader().loadTestsFromName('test_window_func')
    
    unittest.TextTestRunner(verbosity=2).run(suite)