#!/usr/bin/python

"""
A module that tests the
:mod:`~pyspace.pySPACE.missions.nodes.splitter.all_train_splitter`
node

:Author:  Andrei Ignat (Andrei_Cristian.Ignat@dfki.de)
:Created: 2014/06/04
"""

if __name__ == '__main__':
    import sys
    import os
    # The root of the code
    file_path = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(file_path[:file_path.rfind('pySPACE') - 1])

import unittest
from pySPACE.missions.nodes.splitter.all_train_splitter import *
from pySPACE.resources.data_types.time_series import TimeSeries
import pySPACE.tests.generic_unittest as gen_test
from pySPACE.missions.nodes.source.external_generator_source import *


class AllTrainSplitterTestCase(unittest.TestCase):

    """
    The test itself is an embarrassingly simple one since the only thing
    that the node should do is classify all the data points as training
    data points. Therefore, the scenario of the test is to create an input
    node, feed it some data points and then just check if all the data points
    were classified as training data.
    """

    def setUp(self):
        # set up the channels
        self.channel_names = ['Target', 'Standard']
        self.points = []
        # fill in the data points according to a given equation
        for counter in range(100):
            self.points.append((2 * counter, 13 * counter))
        initial_data = TimeSeries(self.points, self.channel_names, 100)
        # since the node was built for online analysis and splitting,
        # we must fool it by giving it the input under the form of a node
        # and not just a e.g. TimeSeries object
        self.input_node = ExternalGeneratorSourceNode()
        self.input_node.set_generator(initial_data)

    def test_split(self):
        splitter = AllTrainSplitterNode()
        splitter.set_permanent_attributes(input_node=self.input_node)
        # check if the test data set is empty
        self.assertEqual(list(splitter.request_data_for_testing()), [])
        # check if the train data set contains all the data points
        self.assertEqual(
            list(splitter.request_data_for_training(True)),
            self.points)


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromName('test_all_train_splitter')

    # check the generic unittests
    suite.addTest(gen_test.ParametrizedTestCase.parametrize(
        current_testcase=gen_test.GenericTestCase, node=AllTrainSplitterNode))

    unittest.TextTestRunner(verbosity=2).run(suite)
