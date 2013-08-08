""" Unit tests for Time Series data type

:Author: Titiruck Nuntapramote (titiruck.nuntapramote@dfki.de)
:Created: 2011/04/23
"""

import unittest
if __name__ == '__main__':
    import sys
    import os
    # The root of the code
    file_path = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(file_path[:file_path.rfind('pySPACE')-1])
    
from pySPACE.resources.data_types.time_series import TimeSeries # as ts
import numpy as np


class TimeSeriesTestCase(unittest.TestCase):
  """ Test for TimeSeries data type """ 
  def setUp(self):
      """ Define some TimeSeries objects """
      # no tag
      self.x1 = TimeSeries([[1,2,3,4,5,6]], ['a','b','c','d','e','f'], 120,
                        marker_name='S4', name='Name_text ending with Standard',
                        start_time=12004.0, end_time=13004.0)
      
      # no -
      self.x2 = TimeSeries([1,2,3,4,5,6], ['a','b','c','d','e','f'], 12,
                        marker_name='S4',
                        start_time=12004.0, end_time=13004.0,
                        tag = 'Tag of x2', name='Name_text ending with Standard')
      
      # no name, tag
      self.x3 = TimeSeries([1,2,3,4,5,6], ['a','b','c','d','e','f'], 12,
                        marker_name='S4', start_time=12004.0, end_time=13004.0)
                        
      # no name, start_time, end_time, tag
      self.x4 = TimeSeries([1,2,3,4,5,6], ['a','b','c','d','e','f'], 12,marker_name='S4')
      
      # no start_time, end_time, name, marker_name, tag         
      self.x5 = TimeSeries([1,2], ['a','b'], 25, start_time = 12004.0)
      
      # no name, start_time, end_time
      self.x6 = TimeSeries([1,2,3,4,5,6], ['a','b','c','d','e','f'], 12,
                        tag = 'Tag of x6')

  
  def test_generate_tag(self):
    self.assertEqual(TimeSeries._generate_tag(self.x1), 
                     'Epoch Start: 12004ms; End: 13004ms; Class: Standard')
    self.assertEqual(TimeSeries._generate_tag(self.x3), 
                     'Epoch Start: 12004ms; End: 13004ms; Class: na')
    self.assertEqual(TimeSeries._generate_tag(self.x4),None)
    self.assertEqual(TimeSeries._generate_tag(self.x5), 
                     'Epoch Start: 12004ms; End: nams; Class: na')
    
  # replace with new data and inherit history, key, tag, specs from the old 
  def test_repalce_data(self):
    data = TimeSeries.replace_data(self.x2, [10,11,12,13,14,15],
                channel_names=['m','n','o','p','q','r'],
                sampling_frequency=30,
                start_time=1200.0)
    self.assertFalse((data.view(np.ndarray)-[10,11,12,13,14,15]).any())
    self.assertEqual(data.channel_names, ['m','n','o','p','q','r'])
    self.assertEqual(data.sampling_frequency, 30)
    self.assertEqual(data.start_time, 1200)
    self.assertEqual(data.end_time, 13004)
    self.assertEqual(data.name, 'Name_text ending with Standard')
    self.assertEqual(data.tag,'Tag of x2')
  
  def test_get_channel(self):
    self.assertEqual(self.x1.channel_names, self.x1.get_channel_names())
    self.assertEqual(self.x6.channel_names, self.x6.get_channel_names())
  
  def test_ms_to_samples(self):
    self.assertEqual(self.x1._ms_to_samples(12),12/1000.0*120)
    self.assertEqual(self.x2._ms_to_samples(25),25/1000.0*12)
  
  def test_samples_to_ms(self):
    self.assertEqual(self.x3._samples_to_ms(34),34/12.0*1000)
    self.assertEqual(self.x5._samples_to_ms(10),10/25.0*1000)
    
    
if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromName('test_time_series')
    unittest.TextTestRunner(verbosity=2).run(suite)
