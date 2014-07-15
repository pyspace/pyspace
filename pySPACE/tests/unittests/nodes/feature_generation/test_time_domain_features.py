#!/usr/bin/python

"""
This module contains unittests which test the time domain
Feature Extraction node

:Author: Jan Hendrik Metzen (jhm@informatik.uni-bremen.de),
    Andrei Ignat (Andrei_Cristian.Ignat@dfki.de)
:Created: 2008/08/26
:Revised: 2014/05/28
"""

import numpy
import random
import unittest

if __name__ == '__main__':
    import sys
    import os
    # The root of the code
    file_path = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(file_path[:file_path.rfind('pySPACE') - 1])

from pySPACE.missions.nodes.feature_generation.time_domain_features import *
from pySPACE.tests.utils.data.test_data_generation import Sine
from pySPACE.tests.utils.data.test_data_generation import TestTimeSeriesGenerator
from pySPACE.resources.data_types.time_series import TimeSeries
import pySPACE.tests.generic_unittest as gen_test

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
        self.x1 = TimeSeries([[1, 2, 3], [6, 5, 3]], ['a', 'b', 'c'], 120)

    def test_td_features(self):
        # Choose which values are used as features
        datapoints = random.sample(range(self.time_series.shape[0]), 5)
        # Create feature extractor and compute features
        td_feature_node = TimeDomainFeaturesNode(datapoints=datapoints)
        features = td_feature_node.execute(
            self.time_series).view(numpy.ndarray)
        # Check that every extracted feature is in the chosen positions of
        # the time series
        for feature in features[0]:
            self.assert_(
                round(feature, 3) in map(lambda x: round(x, 3),
                                         list(self.time_series.view(numpy.ndarray)[datapoints, :].flatten())))

        # Check that no features has been missed
        self.assertEqual(
            len(features[0]),
            len(datapoints) * self.time_series.shape[1])

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
        self.x1 = TimeSeries([[1, 2, 3], [6, 5, 3]], ['a', 'b', 'c'], 120)

    def test_tdd_feature(self):
        tdd_node = TimeDomainDifferenceFeatureNode()
        features = tdd_node.execute(self.x1).view(numpy.ndarray)
        expected = [5.0, 1.0, 3.0, -1.0, -2.0,
                    1.0, -1.0, -1.0, 3.0, 2.0,
                    -3.0, -2.0, 0.0, 2.0, 1.0]

        self.assertEqual(len(features[0]), 15)
        for f in range(len(features[0])):
            self.assertEqual(features[0][f], expected[f])


class SimpleDifferentiationFeature(unittest.TestCase):

    def setUp(self):
        self.channel_names = ['a', 'b', 'c', 'd', 'e', 'f']
        self.x1 = TimeSeries(
            [[1, 2, 3, 4, 5, 6], [6, 5, 3, 1, 7, 7]], self.channel_names, 120)

    def test_sd_feature(self):
        sd_node = SimpleDifferentiationFeatureNode()
        features = sd_node.execute(self.x1)
        for f in range(features.shape[1]):
            channel = features.feature_names[f][4]
            index = self.channel_names.index(channel)
            self.assertEqual(
                features.view(
                    numpy.ndarray)[0][f],
                self.x1.view(
                    numpy.ndarray)[1][index] -
                self.x1.view(
                    numpy.ndarray)[0][index])


class LocalStraightLineFeature(unittest.TestCase):

    """
    This test checks the results of a linear fit on a TimeSeries
    """

    def setUp(self):
        # initiate the two channels
        self.channel_names = ['a', 'b']
        array = []
        # fill in the data points according to a pre set equation
        for counter in range(100):
            array.append([4 * counter + 1, 4.36 * counter - 23.4])
        self.initial_data = TimeSeries(array, self.channel_names, 100)

    def test_linear_fit(self):
        # run the linear fit
        linear = LocalStraightLineFeatureNode(
            segment_width=1000,
            stepsize=1000)
        features = linear.execute(self.initial_data)
        result = [[1., 4., -23.4, 4.36]]
        # check if the results of the fit are the same as the original equation
        self.assertEqual(numpy.allclose(features.get_data(), result), True)


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromName(
        'test_time_domain_features')

    # Test the generic initialization of the class methods
    suite.addTest(gen_test.ParametrizedTestCase.parametrize(
        current_testcase=gen_test.GenericTestCase,
        node=TimeDomainFeaturesNode))
    suite.addTest(gen_test.ParametrizedTestCase.parametrize(
        current_testcase=gen_test.GenericTestCase,
        node=TimeDomainDifferenceFeatureNode))
    suite.addTest(gen_test.ParametrizedTestCase.parametrize(
        current_testcase=gen_test.GenericTestCase,
        node=SimpleDifferentiationFeatureNode))
    suite.addTest(gen_test.ParametrizedTestCase.parametrize(
        current_testcase=gen_test.GenericTestCase,
        node=LocalStraightLineFeatureNode))
    suite.addTest(gen_test.ParametrizedTestCase.parametrize(
        current_testcase=gen_test.GenericTestCase,
        node=CustomChannelWiseFeatureNode))

    # in parallel to the above implementation of the LocalStraightLineFeature,
    # we implement the exact same test but this time by using the
    # InputOutputTestCase

    # initiate the two channels
    channel_names = ['a', 'b']
    array = []
    # fill in the data points according to a pre set equation
    for counter in range(100):
        array.append([4 * counter + 1, 4.36 * counter - 23.4])
    initial_data = TimeSeries(array, channel_names, 100)
    suite.addTest(gen_test.ParametrizedTestCase.parametrize(
        current_testcase=gen_test.InputOutputTestCase,
        node=LocalStraightLineFeatureNode,
        input=[[[initial_data]]],
        output=FeatureVector([4., 1., -23.4, 4.36],
                             feature_names=['LSFSlope_a_0.000sec_1.000sec',
                                            'LSFOffset_a_0.000sec_1.000sec',
                                            'LSFOffset_b_0.000sec_1.000sec',
                                            'LSFSlope_b_0.000sec_1.000sec'])
    ))

    unittest.TextTestRunner(verbosity=2).run(suite)
