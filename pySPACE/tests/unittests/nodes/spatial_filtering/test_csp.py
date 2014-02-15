#!/usr/bin/env python
"""
This module contains unittests that test splitter nodes

.. todo:: Implement tests for TCSPNode

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
from pySPACE.missions.nodes.spatial_filtering.csp import CSPNode
        
class CSPNodeCase(unittest.TestCase):
    """Test for CSPNode"""
    
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
    
    def test_csp(self):
        """
        Tests that CSP produces the expected transformation matrix on the data
        """
        csp_node = CSPNode(retained_channels=2)
        for i in range(self.data.shape[0]):
            ts = TimeSeries(input_array = self.data[i:i+1,:], 
                            channel_names = [("test_channel_%s" % j) for j in range(2)],
                            sampling_frequency = 10, start_time = 0, end_time = 1)
            csp_node.train(ts, self.classes[i])
        csp_node.stop_training()
        
        self.assert_(numpy.allclose(csp_node.filters,
                                       numpy.array([[-0.75319083, -0.35237094],
                                                       [1.,  -1.]])),
                    "CSP transformation matrix is wrong! Got:%s, expected:%s"%(
                    str(csp_node.filters),
                    str(numpy.array([[-0.75319083, -0.35237094],[1.,  -1.]]))
                    ))

        transformed_data = numpy.zeros(self.data.shape)
        for i in range(self.data.shape[0]):
            ts = TimeSeries(input_array = self.data[i:i+1,:], 
                            channel_names = [("test_channel_%s" % j) for j in range(2)],
                            sampling_frequency = 10, start_time = 0, end_time = 1)
            transformed_data[i,:] = csp_node.execute(ts)
        
        self.assert_(numpy.allclose(transformed_data[0:2,:],
                                       numpy.array([[0.14525655, -0.83028934],
                                                    [0.68796176, -0.23672793]])),
                    "CSP-transformed data (%s) does not match expectation (%s)!"%(
                    str(transformed_data[0:2,:]),str(numpy.array([[0.14525655, -0.83028934],
                                                    [0.68796176, -0.23672793]]))
                    ))
        
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

    suite = unittest.TestLoader().loadTestsFromName('test_csp')
    
    unittest.TextTestRunner(verbosity=2).run(suite)
    