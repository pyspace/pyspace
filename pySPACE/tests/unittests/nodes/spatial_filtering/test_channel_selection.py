#!/usr/bin/env python
"""
This module contains unittests that test the channel selection module

:Author: Jan Hendrik Metzen (jhm@informatik.uni-bremen.de)
:Created: 2009/09/09
"""


import unittest

import numpy

if __name__ == '__main__':
    import sys
    import os
    # The root of the code
    file_path = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(file_path[:file_path.rfind('pySPACE')-1])

from pySPACE.missions.nodes.spatial_filtering import channel_selection
from pySPACE.tests.utils.data.test_data_generation import Sine
from pySPACE.tests.utils.data.test_data_generation import TestTimeSeriesGenerator

test_ts_generator = TestTimeSeriesGenerator()
test_sine = Sine()

class ChannelNameSelectorTestCase(unittest.TestCase):
    """Test for ChannelNameSelecterNode"""
    
    def setUp(self):
        self.time_series = test_ts_generator.generate_test_data(8, 1000, test_sine, 100.0)        
        
    def test_channel_selection(self):
        selected_channels = ["test_channel_1", "test_channel_2"]
        channel_selection_node = channel_selection.ChannelNameSelectorNode(selected_channels=selected_channels)
        
        projected_time_series = channel_selection_node.execute(self.time_series)
        
        self.assert_(id(self.time_series) != id(projected_time_series),
                     "No new object has been created")
        
        # Check that all selected channels have been retained
        for channel_name in selected_channels:
            self.assert_(channel_name in projected_time_series.channel_names,
                         "Some selected channels were not retained")
        
        # Check that only selected channels are retained
        for channel_name in projected_time_series.channel_names:
            self.assert_(channel_name in selected_channels,
                         "Some channels were not removed")
            
        # Check that the values of the channels are unchanged
        for channel_name in selected_channels:
            self.assert_(numpy.all(self.time_series.get_channel(channel_name)
                                    == projected_time_series.get_channel(channel_name)),
                        "Channel values changed during channel selection")
            
    def test_inverse_channel_selection(self):
        selected_channels = ["test_channel_1", "test_channel_2"]
        channel_selection_node = channel_selection.ChannelNameSelectorNode(selected_channels=selected_channels,
                                                                           inverse=True)
        
        projected_time_series = channel_selection_node.execute(self.time_series)
        
        # Check that all selected channels have been removed
        for channel_name in selected_channels:
            self.assert_(channel_name not in projected_time_series.channel_names,
                         "Some channels selected for removal were retained")
        
        # Check that only non-selected channels are removed
        for channel_name in self.time_series.channel_names:
            if channel_name not in selected_channels:
                self.assert_(channel_name in projected_time_series.channel_names,
                             "Some channels were removed even though they were not selected for removal")

      
if __name__ == '__main__':  
    suite = unittest.TestLoader().loadTestsFromName('test_channel_selection')
    
    unittest.TextTestRunner(verbosity=2).run(suite)      
