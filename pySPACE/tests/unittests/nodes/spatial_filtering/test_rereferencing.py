""" Unittests for the LaplacianReferenceNode

:Author: Laura Manca (Laura.Manca89@gmail.com)
:Created: 20013/09/24
"""


import unittest


if __name__ == '__main__':
    import sys
    import os
    # The root of the code
    file_path = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(file_path[:file_path.rfind('pySPACE')-1])

from pySPACE.tests.utils.data.test_data_generation import TestTimeSeriesGenerator
from pySPACE.missions.nodes.spatial_filtering import rereferencing

test_ts_generator = TestTimeSeriesGenerator()
        
class LaplacianReferenceTestCase(unittest.TestCase):
    """
    Test for LaplacianReferenceNode
    
    :Author: Laura Manca (Laura.Manca89@gmail.com)
    :Created: 2013/09/24
    """
    def setUp(self):
        self.test_data = test_ts_generator.generate_test_data(
            channels=13,
            time_points=1000,
            sampling_frequency=200,
            channel_names=('FC5', 'FC3', 'FC1', 'FCC5h', 'FCC3h', 'C5', 'C3', 
                           'C1', 'CCP5h','CCP3h', 'CP5', 'CP3', 'CP1')) 
            
    def test_shape_distance_matrix(self):
        """Check that the distance matrix has the correct shape """
        selected_channels = self.test_data.channel_names
        compute_distance_node = rereferencing.LaplacianReferenceNode(
            selected_channels=selected_channels)
        dist = compute_distance_node.calc_distance_matrix(self.test_data)
        self.assertEqual(dist.shape, (13, 13))
    
    def test_left_channels_small(self):
        """Check the channels left after the small Laplacian is applied """
        Laplacian_small = rereferencing.LaplacianReferenceNode()
        filtered_time_series = Laplacian_small(self.test_data)
        self.assertEqual(filtered_time_series.shape, (1000, 5))
        self.assertEqual(filtered_time_series.channel_names, 
                         ['FCC5h', 'FCC3h', 'C3', 'CCP5h', 'CCP3h'])
    
    def test_left_channels_big(self):
        """Check the channels left after the big Laplacian is applied """
        Laplacian_big = rereferencing.LaplacianReferenceNode(l_type='big')
        filtered_time_series = Laplacian_big(self.test_data)
        self.assertEqual(filtered_time_series.shape, (1000, 1))
        self.assertEqual(filtered_time_series.channel_names, ['C3'])
        
         

if __name__ == '__main__':
    
    suite = unittest.TestLoader().loadTestsFromName('test_rereferencing')
    unittest.TextTestRunner(verbosity=2).run(suite)
