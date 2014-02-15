#!/usr/bin/env python
""" Unittests which test filtering nodes

:Author: Jan Hendrik Metzen (jhm@informatik.uni-bremen.de)
:Created: 2008/08/22
"""


import unittest
import logging

logger = logging.getLogger('TestLogger')


if __name__ == '__main__':
    import sys
    import os

    # The root of the code
    file_path = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(file_path[:file_path.rfind('pySPACE')-1])

from pySPACE.tests.utils.data.test_data_generation import TestTimeSeriesGenerator
from pySPACE.missions.nodes.preprocessing.reorder_memory import ReorderMemoryNode

test_ts_generator = TestTimeSeriesGenerator()

class ReorderMemoryTestCase(unittest.TestCase):
    """
    Test for ReorderMemoryNode
    
    """
    
    def setUp(self):
        
        self.node = ReorderMemoryNode() 
       
        time_points = 10
        channels = 2
        import pySPACE.tests.utils.data.test_data_generation as test_helpers
        counter = test_helpers.Counter()
        self.data = test_ts_generator.generate_test_data(channels,time_points,counter,channel_order=True)
        
    def testReordering(self):
        
        data = self.node.execute(self.data)
        
        logger.log(logging.INFO, "striding before reordering: " + str(self.data.strides))
        logger.log(logging.INFO, "striding after reordering: " + str(self.data.strides))
        
        self.assertTrue(True)
        
        
if __name__ == '__main__':  
    suite = unittest.TestLoader().loadTestsFromName('test_reorder_memory')
    
    unittest.TextTestRunner(verbosity=2).run(suite)
    