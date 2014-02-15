#!/usr/bin/env python
"""
This module contains unittests which test the time domain Feature Extraction node

.. todo:: Implement unittests for LocalStraightLineFeatureNode,
.. todo:: Implement unittests for LocalPolynomialFeatureNode

:Author: Jan Hendrik Metzen (jhm@informatik.uni-bremen.de)
:Created: 2008/08/26
"""

import numpy
import random
import unittest

if __name__ == '__main__':
    import sys
    import os
    # The root of the code
    file_path = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(file_path[:file_path.rfind('pySPACE')-1])

from pySPACE.missions.nodes.feature_generation.time_domain_features import *
from pySPACE.tests.utils.data.test_data_generation import Sine
from pySPACE.tests.utils.data.test_data_generation import TestTimeSeriesGenerator
from pySPACE.resources.data_types.time_series import TimeSeries

test_ts_generator = TestTimeSeriesGenerator()
test_sine = Sine()


class TimeDomainFeaturesTestCase(unittest.TestCase):
    """ unittest for TimeDomainFeaturesNode """
    def setUp(self):
        self.time_series = test_ts_generator.generate_test_data(
            channels=8,
            time_points=1000,
            function=test_sine,
            sampling_frequency=100.0)
        self.x1 = TimeSeries([[1,2,3],[6,5,3]], ['a', 'b', 'c'], 120)

    def test_td_features(self):
        #Choose which values are used as features
        datapoints = random.sample(range(self.time_series.shape[0]), 5)
        #Create feature extractor and compute features
        td_feature_node = TimeDomainFeaturesNode(datapoints = datapoints)
        features = td_feature_node.execute(self.time_series).view(numpy.ndarray)
        # Check that every extracted feature is in the chosen positions of
        # the time series
        for feature in features[0]:
            self.assert_(round(feature, 3) in map(lambda x: round(x, 3),
                list(self.time_series.view(numpy.ndarray)[datapoints, :].flatten())))
            
        #Check that no features has been missed
        self.assertEqual(len(features[0]), len(datapoints) * self.time_series.shape[1])

    def test_ordering(self):
        """ Test if values are in the expected ordering afterwards

        First ordering in time and then in channels is expected.
        """
        expected = [1.0, 6.0, 2.0, 5.0, 3.0, 3.0]
        node = TimeDomainFeaturesNode()
        features = node.execute(self.x1)
        feature_names = features.feature_names
        features = features.view(numpy.ndarray)
        self.assertEqual(len(features[0]), 6)
        self.assertTrue(feature_names[0].startswith('TD_a'))
        self.assertTrue(feature_names[1].startswith('TD_a'))
        self.assertTrue(feature_names[2].startswith('TD_b'))
        self.assertTrue(feature_names[3].startswith('TD_b'))
        self.assertTrue(feature_names[4].startswith('TD_c'))
        self.assertTrue(feature_names[5].startswith('TD_c'))
        for f in range(len(features[0])):
            self.assertEqual(features[0][f], expected[f])


class TimeDomainDifferenceFeatureTestCase(unittest.TestCase):
    def setUp(self):
        self.x1 = TimeSeries([[1,2,3],[6,5,3]], ['a', 'b', 'c'], 120)
        
    def test_tdd_feature(self):
        tdd_node = TimeDomainDifferenceFeatureNode()
        features = tdd_node.execute(self.x1).view(numpy.ndarray)
        expected = [5.0,1.0,3.0,-1.0,-2.0, \
                    1.0,-1.0,-1.0,3.0,2.0, \
                    -3.0,-2.0,0.0,2.0,1.0]
        
        self.assertEqual(len(features[0]), 15)
        for f in range(len(features[0])):
            self.assertEqual(features[0][f], expected[f])


class SimpleDifferentiationFeature(unittest.TestCase):
    def setUp(self):
        self.channel_names = ['a','b','c','d','e','f']
        self.x1 = TimeSeries([[1,2,3,4,5,6],[6,5,3,1,7,7]], self.channel_names, 120)
    
    def test_sd_feature(self):
        sd_node = SimpleDifferentiationFeatureNode()
        features = sd_node.execute(self.x1)
        for f in range(features.shape[1]):
            channel = features.feature_names[f][4]
            index = self.channel_names.index(channel)
            self.assertEqual(features.view(numpy.ndarray)[0][f],self.x1.view(numpy.ndarray)[1][index]-self.x1.view(numpy.ndarray)[0][index])
        
#class LocalStraightLineFeature(unittest.TestCase):
#    pass
#
#class LocalPolynomialFeature(unittest.TestCase):
#    pass


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromName('test_time_domain_features')  
    unittest.TextTestRunner(verbosity=2).run(suite)