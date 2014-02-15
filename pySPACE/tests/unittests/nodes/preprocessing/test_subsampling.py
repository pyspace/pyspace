#!/usr/bin/env python
""" Unittests which test subsampling nodes

.. todo:: Implement tests for FftResamplingNode

:Author: Jan Hendrik Metzen (jhm@informatik.uni-bremen.de)
:Created: 2008/08/22
"""


import unittest
import numpy
import time
import warnings

import logging
logger = logging.getLogger("TestLogger")

if __name__ == '__main__':
    import sys
    import os
    # The root of the code
    file_path = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(file_path[:file_path.rfind('pySPACE')-1])

    # configure logger
    # (TestLogger is not configured, since if main is called,
    # we have a single test)
    logger.setLevel(logging.DEBUG)
    loggingFileHandler = logging.FileHandler("unittest_log.txt")
    loggingStreamHandler = logging.StreamHandler()
    loggingStreamHandler.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(pathname)s:%(lineno)d - %(message)s")
    loggingFileHandler.setFormatter(formatter)
    loggingStreamHandler.setFormatter(formatter)

    logger.addHandler(loggingFileHandler)
    logger.addHandler(loggingStreamHandler)

    # load tests
    suite = unittest.TestLoader().loadTestsFromName('test_subsampling')

    # run tests
    unittest.TextTestRunner(verbosity=2).run(suite)


from pySPACE.missions.nodes.preprocessing import subsampling
import pySPACE.tests.utils.data.test_data_generation as test_data_generation

time_series_generator = test_data_generation.TestTimeSeriesGenerator()

class SubsamplingTestCase(unittest.TestCase):
    """ Test for SubsamplingNode """

    def setUp(self):
        self.time_points = 1000
        self.freq = 1000.
        self.channels = 1
        counter = test_data_generation.Counter()
        self.time_series = \
            time_series_generator.generate_test_data(self.channels,
                                                     self.time_points,
                                                     counter,
                                                     self.freq,
                                                     channel_order=True)


    def test_subsampling(self):
        target_frequency = 10.0

        subsampling_node = subsampling.SubsamplingNode(target_frequency = target_frequency)

        subsampled_time_series = subsampling_node.execute(self.time_series)

        self.assertEqual(subsampled_time_series.shape[0], int(round(self.time_points * target_frequency / self.time_series.sampling_frequency)))
        self.assertEqual(subsampled_time_series.sampling_frequency, target_frequency)
        self.assertEqual(len(subsampled_time_series), numpy.ceil(self.time_points * target_frequency / self.freq))

        self.assert_(id(self.time_series) != id(subsampled_time_series)) # The object should be different!

        ##TODO: What is a sensible test for the frequency content criteria??


class DecimationNodeTestCase(unittest.TestCase):
    """ Test for the DecimationFIRNode """
    def setUp(self):
        self.time_points = 5000
        self.source_frequency = 5000.
        self.channels = 64
        counter = test_data_generation.Counter()
        self.time_series = \
            time_series_generator.generate_test_data(channels=self.channels,
                                                    time_points=self.time_points,
                                                    function=counter,
                                                    sampling_frequency=self.source_frequency,
                                                    channel_order=True)

        self.target_frequency = 25.0

    def tearDown(self):
        self.filter_node = None
        self.data = None

    # a helper method that performs various tests on given subsampling nodes
    def perform_simple_tests(self, node_to_test,
                            time_series = None,
                            source_frequency = None,
                            target_frequency = None,
                            time_points = None):

        if time_series == None:
            time_series = self.time_series

        if not source_frequency:
            source_frequency = self.source_frequency

        if not target_frequency:
            target_frequency = self.target_frequency

        if not time_points:
            time_points = self.time_points

        data = time_series.copy()
        data2 = time_series.copy()

        # first filtering, the node is initialized in this step
        subsampled_time_series = node_to_test.execute(data2)

        start = time.clock()
        subsampled_time_series = node_to_test.execute(data2)
        stop = time.clock()
        logger.log(logging.INFO, "time for subsampling: " + str(stop-start))

        logger.log(logging.INFO, "time series shape: " + str(time_series.shape))
        logger.log(logging.INFO, "subsampled_time_series.shape" + str(subsampled_time_series.shape))
        a = int(round(time_points * target_frequency / time_series.sampling_frequency))
        self.assertEqual(subsampled_time_series.shape[0], int(round(time_points * target_frequency / time_series.sampling_frequency)))
        self.assertEqual(subsampled_time_series.sampling_frequency, target_frequency)
        self.assertEqual(len(subsampled_time_series), numpy.ceil(time_points * target_frequency / time_series.sampling_frequency))

        self.assert_(id(time_series) != id(subsampled_time_series)) # The object should be different!


    def test_decimation(self):
        node_to_test = subsampling.DecimationIIRNode(target_frequency = self.target_frequency)

        self.perform_simple_tests(node_to_test)

    def test_decimation_fir(self):
        node_to_test = subsampling.DecimationFIRNode(target_frequency = self.target_frequency)

        self.perform_simple_tests(node_to_test)

    def perform_compare_tests(self,standard_node,testee_node):

        data = self.time_series.copy()
        data2 = self.time_series.copy()

        # initialize & set custom kernel
        subsampled_time_series = standard_node.execute(data)

        # execute node
        start = time.time()
        subsampled_time_series = standard_node.execute(data)
        stop = time.time()

        #logger.log(logging.INFO, "time for subsampling: " + str(stop-start))

        # create a reference node with custom kernel
        subsampled_time_series2 = testee_node.execute(data2)

        start = time.time()
        subsampled_time_series2 = testee_node.execute(data2)
        stop = time.time()

        logger.log(logging.INFO, "time for subsampling: " + str(stop-start))

        self.assert_(id(self.time_series) != id(subsampled_time_series)) # The object should be different!

        numpy.set_printoptions(threshold=numpy.nan)

        #print subsampled_time_series
        #print "*"*50
        #print subsampled_time_series2
        #print "*"*50
        #print subsampled_time_series-subsampled_time_series2

        self.assertTrue(numpy.allclose(subsampled_time_series,subsampled_time_series2))

        self.assertEqual(subsampled_time_series.shape[0], numpy.ceil(self.time_points * self.target_frequency / self.time_series.sampling_frequency))
        self.assertEqual(subsampled_time_series.sampling_frequency, self.target_frequency)




class DownsamplingNodeTestCase(unittest.TestCase):
    """ Test for the downsampling node """
    def setUp(self):
        self.time_points = 5000
        self.freq = 5000.
        self.channels = 1
        counter = test_data_generation.Counter()
        self.time_series = time_series_generator.generate_test_data(self.channels,self.time_points,function=counter,sampling_frequency=self.freq,channel_order=True)

        self.target_frequency = 25

        self.downsampling_factor = self.freq / self.target_frequency

        self.node = subsampling.DownsamplingNode(self.target_frequency)

    def tearDown(self):
        self.filter_node = None
        self.data = None

    def testSimpleDownsampling(self):
        new_data = self.node.execute(self.time_series)

        self.assertEqual(new_data.shape[0],self.time_points/self.downsampling_factor)
        self.assertEqual(new_data.shape[1],self.channels)
        self.assertEqual(new_data.sampling_frequency,self.target_frequency)



