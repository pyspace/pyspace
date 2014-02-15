#!/usr/bin/env python
""" Unittests that test normalization nodes

:Author: Jan Hendrik Metzen (jhm@informatik.uni-bremen.de)
:Created: 2009/01/06
"""


import unittest

import numpy
import pylab

if __name__ == '__main__':
    import sys
    import os
    # The root of the code
    file_path = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(file_path[:file_path.rfind('pySPACE')-1])

from pySPACE.tests.utils.data.test_data_generation import TestTimeSeriesGenerator
from pySPACE.tests.utils.data.test_data_generation import Sine
from pySPACE.missions.nodes.preprocessing import normalization


test_sine = Sine()
test_ts_generator = TestTimeSeriesGenerator()

class DetrendingTestCase(unittest.TestCase):
    """ Test for Detrending Node"""

    def setUp(self):
        self.time_series = test_ts_generator.generate_test_data(8, 1000, test_sine, 100.0)
        self.time_series2 = test_ts_generator.generate_test_data(8, 1000, test_sine, 100.0)

        # INFO: pylab.detrend_mean is a default detrend_method and will be used when given no param as well
        # INFO: another method should be used when given param
        detrending_node = normalization.DetrendingNode()
        self.detrended_time_series = detrending_node._execute(self.time_series)

    def test_detrending(self):

        # The object should be different!
        self.assertNotEqual(id(self.time_series), id(self.detrended_time_series))

        # Check that the mean has actually been set to 0
        for channel_index in range(self.time_series.shape[1]):
            #self.assertTrue(numpy.mean(self.time_series[:,channel_index]) < numpy.mean(self.detrended_time_series[:,channel_index]))
            self.assertAlmostEqual(numpy.mean(self.detrended_time_series[:,channel_index]), 0.0)

    def test_selected_channelnames(self):
        # selected_channelnames param is deprecated.
        # check that exception is raised
        try:
            detrending_node2 = normalization.DetrendingNode(detrend_method = pylab.detrend_mean, selected_channels='test_channel_4')
        except Exception, e:
            self.assertEqual(type(e), DeprecationWarning)


class LocalStandardizationTestCase(unittest.TestCase):
    """
    Test for LocalStandardizationNode

    :Author: Titiruck Nuntapramote (titiruck.nuntapramote@dfki.de)
    :Created: 2012/03/30
    """
    def setUp(self):
        self.time_series = test_ts_generator.generate_test_data(
            8, 1000, test_sine, 100.0)
        lcstd = normalization.LocalStandardizationNode()
        self.local_time_series = lcstd._execute(self.time_series)

    def test_localstandardization(self):
        for channel_index in range(self.time_series.shape[1]):
            #self.assertTrue(numpy.mean(self.time_series[:,channel_index]) < numpy.mean(self.local_time_series[:,channel_index]))
            self.assertAlmostEqual(numpy.mean(self.local_time_series[:,channel_index]), 0.0)
            self.assertAlmostEqual(numpy.std(self.local_time_series[:,channel_index]), 1.0)


class MaximumStandardizationTestCase(unittest.TestCase):
    """
    Test for MaximumStandardizationNode

    :Author: Titiruck Nuntapramote (titiruck.nuntapramote@dfki.de)
    :Created: 2012/03/30
    """
    def setUp(self):
        self.time_series = test_ts_generator.generate_test_data(
            8, 1000, test_sine, 100.0)

        maxstd = normalization.MaximumStandardizationNode()
        self.max_time_series = maxstd._execute(self.time_series)

    def test_maximumstandardization(self):
        for channel_index in range(self.time_series.shape[1]):
            #self.assertTrue(numpy.mean(self.time_series[:,channel_index]) < numpy.mean(self.max_time_series[:,channel_index]))
            self.assertAlmostEqual(
                numpy.mean(self.max_time_series[:, channel_index]), 0.0)
            self.assertAlmostEqual(
                numpy.max(self.max_time_series[:, channel_index]), 1.0)


class MemoryStandardizationTestCase(unittest.TestCase):
    """
    Test for MemoryStandardizationNode

    :Author: Titiruck Nuntapramote (titiruck.nuntapramote@dfki.de)
    :Created: 2012/04/12
    """
    def setUp(self):
        self.time_series = test_ts_generator.generate_test_data(
            8, 1000, test_sine, 100.0)

        #order = 0 - works the same as LocalStandardizationNode
        memory_node = normalization.MemoryStandardizationNode()
        self.memory_ts = memory_node._execute(self.time_series)

        memory_node2 = normalization.MemoryStandardizationNode(order = 2)
        self.memory_ts2 = memory_node2._execute(self.time_series)
        self.memory_ts3 = memory_node2._execute(self.memory_ts2)
        self.memory_ts4 = memory_node2._execute(self.memory_ts3)

    def test_memorystandardization(self):

        # test for order = 0
        for channel_index in range(self.time_series.shape[1]):
            self.assertAlmostEqual(numpy.mean(self.memory_ts[:,channel_index]), 0.0)
            self.assertAlmostEqual(numpy.std(self.memory_ts[:,channel_index]), 1.0)

        # test for order = 2
        for channel_index in range(self.time_series.shape[1]):
            mean = numpy.mean(self.memory_ts2[:, channel_index])
            std = numpy.std(self.memory_ts2[:, channel_index])
            self.assertAlmostEqual(numpy.mean(self.memory_ts3[:,channel_index]), \
                                   (numpy.mean(self.memory_ts3[:, channel_index]-mean))/std)

            self.assertAlmostEqual(numpy.mean(self.memory_ts4[:,channel_index]), \
                                   (numpy.mean(self.memory_ts4[:, channel_index]-mean))/std)


class DevariancingTestCase(unittest.TestCase):
    """ Test for Devariancing Node """
    def setUp(self):
        self.time_series = test_ts_generator.generate_test_data(
            8, 1000, test_sine, 100.0)
        devariancing_node = \
            normalization.DevariancingNode(devariance_method=numpy.std)
        devariancing_node.train(self.time_series)
        devariancing_node.stop_training()
        self.devarianced_time_series = \
            devariancing_node.execute(self.time_series)

    def test_devariancing(self):

        # The object should be different!
        self.assertNotEqual(id(self.time_series),
                            id(self.devarianced_time_series))

        # Check that std has been set to 1
        for channel_index in range(self.time_series.shape[1]):
            self.assertAlmostEqual(numpy.std(self.devarianced_time_series[:,channel_index]), 1.0)


class SubsetNormalizationTestCase(unittest.TestCase):
    """
    Test for SubsetNormalizationNode

    :Author: Titiruck Nuntapramote (titiruck.nuntapramote@dfki.de)
    :Created: 2012/03/30
    """
    def setUp(self):
        self.time_series = test_ts_generator.generate_test_data(
            8, 1000, test_sine, 100.0)
        subset_node = normalization.SubsetNormalizationNode(subset=range(5),
                                                            devariance=True)
        self.subset_ts = subset_node._execute(self.time_series)

    def test_memorystandardization(self):
        for channel_index in range(self.time_series.shape[1]):
            self.assertAlmostEqual(numpy.mean(self.subset_ts[:,channel_index]), \
                                   numpy.mean(self.subset_ts[:, channel_index]-numpy.mean(self.subset_ts[range(5)])))


class EuclideanNormalizationTestCase(unittest.TestCase):
    """
    Test for EuclideanStandardizationNode

    :Author: Titiruck Nuntapramote (titiruck.nuntapramote@dfki.de)
    :Created: 2012/03/30
    """
    def setUp(self):
        self.time_series = test_ts_generator.generate_test_data(8, 1000, test_sine, 100.0)

        eustd = normalization.EuclideanNormalizationNode()
        self.eu_time_series = eustd._execute(self.time_series)

    def test_euclideanstandardization(self):
        for channel_index in range(self.time_series.shape[1]):
            self.assertAlmostEqual(numpy.linalg.norm(self.eu_time_series[:,channel_index]), 1.0)
            self.assertNotAlmostEqual(numpy.mean(self.eu_time_series[:,channel_index]), 0.0)


if __name__ == '__main__':

    suite = unittest.TestLoader().loadTestsFromName('test_normalization')
    unittest.TextTestRunner(verbosity=2).run(suite)
