#!/usr/bin/python

"""
A module that tests the
:mod:`~pySPACE.home.aignat.pyspace.pySPACE.missions.nodes.splitter.traintest_splitter`
node

:Author:  Andrei Ignat (Andrei_Cristian.Ignat@dfki.de)
:Created: 2014/06/03
"""

if __name__ == '__main__':
    import sys
    import os
    # The root of the code
    file_path = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(file_path[:file_path.rfind('pySPACE') - 1])

import unittest
from pySPACE.missions.nodes.splitter.traintest_splitter import *
from pySPACE.resources.data_types.time_series import TimeSeries
import pySPACE.tests.generic_unittest as gen_test
from pySPACE.missions.nodes.source.external_generator_source import *


class TrainTestSplitterTestCase(unittest.TestCase):

    def setUp(self):
        # set up the channels
        self.channel_names = ['Target', 'Standard']
        self.points = []
        # fill in the data points according to a given equation
        for cntr in range(100):
            self.points.append((2 * cntr, 13 * cntr))
        initial_data = TimeSeries(self.points, self.channel_names, 100)
        # since the node was built for online analysis and splitting,
        # we must fool it by giving it the input under the form of a node
        # and not just a e.g. TimeSeries object
        self.input_node = ExternalGeneratorSourceNode()
        self.input_node.set_generator(initial_data)

    def test_random_split(self):
        splitter = TrainTestSplitterNode(train_ratio=0.3, random=True)
        splitter.set_permanent_attributes(input_node=self.input_node)
        splitter._create_split()
        # we check if the split has the correct length
        self.assertEqual(len(splitter.train_data), 30)
        # and then we check if the split was done in a random way
        self.assertNotEqual(splitter.train_data, self.points[:30])

    def test_reverse_split(self):
        splitter = TrainTestSplitterNode(
            train_ratio=0.3,
            random=False,
            reverse=True)
        splitter.set_permanent_attributes(input_node=self.input_node)
        splitter._create_split()
        # we check if the split has the correct length
        self.assertEqual(len(splitter.train_data), 30)
        # and then we check if the split was done in a random way
        self.assertEqual(splitter.train_data, self.points[:30])


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromName('test_traintest_splitter')

    # Test the generic initialization of the class methods
    suite.addTest(gen_test.ParametrizedTestCase.parametrize(
        current_testcase=gen_test.GenericTestCase, node=TrainTestSplitterNode))

    unittest.TextTestRunner(verbosity=2).run(suite)
