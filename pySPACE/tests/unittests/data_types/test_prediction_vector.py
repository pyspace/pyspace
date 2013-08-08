""" Unit tests for PredictionVector data type

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
    
from pySPACE.resources.data_types.prediction_vector import PredictionVector


class PredictionVectorTestCase(unittest.TestCase):
  """Test for PredictionVector data type"""
  
  def test_PredictionVector(self):
      
    # Exception should be raised if both input_array and prediction not provided
    self.assertRaises(TypeError, PredictionVector.__new__)
      
    p2 = PredictionVector([[1,2,3,4,5,6]])
    self.assertEqual(p2.prediction, [1,2,3,4,5,6])
    
    p3 = PredictionVector(prediction=1)
    self.assertEqual(p3.prediction,1)
    
    p4 = PredictionVector([[1,2]],prediction=[1,2])
    
    
if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromName('test_prediction_vector')
    unittest.TextTestRunner(verbosity=2).run(suite)
