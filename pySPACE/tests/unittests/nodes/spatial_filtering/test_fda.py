#!/usr/bin/env python
"""
This module contains unittests that test splitter nodes

:Author: Jan Hendrik Metzen (jhm@informatik.uni-bremen.de)
:Created: 2008/12/18
"""

import unittest
import numpy

if __name__ == '__main__':
    import sys
    import os
    # The root of the code
    file_path = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(file_path[:file_path.rfind('pySPACE')-1])

from pySPACE.resources.data_types.time_series import TimeSeries
from pySPACE.missions.nodes.spatial_filtering.fda import FDAFilterNode

class FDAFilterTestCase(unittest.TestCase):
    """Test for FDAFilterNode"""
    
    def setUp(self):
        samples = 5000
        ranges = [-3.5, 3.5, -3.5, 3.5]
        
        numpy.random.seed(0)
        true_data = numpy.zeros((samples, 2))
        true_data[:,0] = numpy.random.normal(loc=0.0, scale=1.0, size=(samples,))
        true_data[:,1] = numpy.random.normal(loc=0.0, scale=0.5, size=(samples,))
        self.classes = [-1 if x < 0 else 1 for x in true_data[:,1]]
        
        mixed_data = numpy.zeros((samples,2))
        for i in range(samples):
            mixed_data[i,0] = 0.6*true_data[i,0] + 0.4*true_data[i,1] + 1.0
            mixed_data[i,1] = 0.4*true_data[i,0] - 0.6*true_data[i,1] + 1.5
        
        self.data = numpy.zeros(mixed_data.shape)
        self.data[:,0] = mixed_data[:,0] - numpy.mean(mixed_data[:,1])
        self.data[:,1] = mixed_data[:,1] - numpy.mean(mixed_data[:,1])
        
        self.data = [TimeSeries(data, channel_names = [("test_channel_%s" % j) for j in range(2)],
                            sampling_frequency = 10) for data in self.data]
        
        
    def test_fda(self):
        """
        Tests that FDA produces the expected transformation matrix on the data
        """
        fda_node = FDAFilterNode(retained_channels=2)
        for i,data in enumerate(self.data):
            fda_node.train(data, self.classes[i])
        fda_node.stop_training()
        
        self.assert_(numpy.allclose(fda_node.filters,
                                   numpy.array([[1.56207903, -1.15805762], 
                                                [-2.32599494, -0.79980837]])),
                    "FDA transformation matrix is wrong!")
        
        transformed_data = []
        for i,data in enumerate(self.data):
            ts = TimeSeries(input_array = data, 
                            channel_names = [("test_channel_%s" % j) for j in range(2)],
                            sampling_frequency = 10, start_time = 0, end_time = 1)
            transformed_data.append(fda_node.execute(ts).view(numpy.ndarray))
        self.assert_(numpy.allclose(transformed_data[0:2],
                                   [numpy.array([[ -0.45549484, -1.20700466]]),
                                    numpy.array([[ -1.52271298,  0.16829469]])]),
                     "FDA-transformed data does not match expectation!")
        
#        self.plot(self.data, transformed_data)
        
    def plot(self, original_data, transformed_data):    
        import pylab
        pylab.figure(1)
        pylab.subplot(1,2,1)
        pylab.gcf().subplots_adjust(left=0.04, bottom=0.04, right=0.96, top= 0.96)
        pylab.scatter(original_data[:,0], original_data[:,1], 
                      color = ['r' if c==1 else 'b' for c in self.classes],
                      marker = 'o')
        pylab.title("Original")
        
        pylab.subplot(1,2,2)
        pylab.scatter(transformed_data[:,0], transformed_data[:,1],
                      color = ['r' if c==1 else 'b' for c in self.classes], marker = 'o')
        pylab.title("Transformed")
        
        pylab.show()
        
        
#        self.assert_(found, "One data point is never used for testing in cv splitting") 
        
            
    
if __name__ == '__main__':
    
    suite = unittest.TestLoader().loadTestsFromName('test_fda')
    
    unittest.TextTestRunner(verbosity=2).run(suite)
    