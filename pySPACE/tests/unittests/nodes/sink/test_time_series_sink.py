"""
This module contains unit tests that test storing and loading using the sink and source nodes

:Author: Jan Hendrik Metzen (jhm@informatik.uni-bremen.de)
:Created: 2008/12/18
"""

import unittest
import os
import shutil
import numpy

if __name__ == '__main__':
    import sys
    import os
    # The root of the code
    file_path = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(file_path[:file_path.rfind('pySPACE')-1])


try:
    from pySPACE.tests.utils.data.test_data_generation import SimpleTimeSeriesSourceNode
except:
    from pySPACE.missions.nodes.source.test_source_nodes import SimpleTimeSeriesSourceNode
from pySPACE.missions.nodes.source.time_series_source import TimeSeriesSourceNode
from pySPACE.missions.nodes.sink.time_series_sink import TimeSeriesSinkNode

from pySPACE.resources.dataset_defs.base import BaseDataset
        
class TimeSeriesSinkTestCase(unittest.TestCase):
    """ test both TimeSeries sink and source """
        
    def test_time_series_storing(self):

        if os.path.exists('tmp') is False :
            os.makedirs('tmp')
        
        source = SimpleTimeSeriesSourceNode()
        sink = TimeSeriesSinkNode()
        sink.register_input_node(source)
        sink.set_run_number(0)
        sink.process_current_split()
        result_collection = sink.get_result_dataset()
        result_collection.store('tmp')
        #sink.store_results("test_time_series_storing.tmp")
        
        reloaded_collection = BaseDataset.load('tmp')
        
        reloader = TimeSeriesSourceNode()
        reloader.set_input_dataset(reloaded_collection)
        #set_permanent_attributes(time_series_file = "test_time_series_storing.tmp")
        
        orig_data = list(source.request_data_for_testing()) 
        restored_data = list(reloader.request_data_for_testing())
        
        # Check that the two list have the same length
        self.assertEqual(len(orig_data), len(restored_data),
                         "Numbers of time series before storing and after reloading are not equal!")
        
        # Check that there is a one-to-one correspondence
        for orig_datapoint, orig_label in orig_data:
            found = False
            for restored_datapoint, restored_label in restored_data:
                found |= (orig_datapoint.view(numpy.ndarray) == restored_datapoint.view(numpy.ndarray)).all() \
                            and (orig_label == restored_label)
                if found: break
            self.assert_(found, 
                         "One of the original time series cannot not be found after reloading")
        
        shutil.rmtree('tmp') # Cleaning up... 


if __name__ == '__main__':
  
    suite = unittest.TestLoader().loadTestsFromName('test_sink_source')
    unittest.TextTestRunner(verbosity=2).run(suite)  
    
  