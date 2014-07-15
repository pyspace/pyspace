#!/usr/bin/python

"""
A module that tests the
:mod:`~pyspace.pySPACE.missions.nodes.spatial_filtering_xdawn`
node

:Author:  Andrei Ignat (Andrei_Cristian.Ignat@dfki.de)
:Created: 2014/06/05
"""

if __name__ == '__main__':
    import sys
    import os
    # The root of the code
    file_path = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(file_path[:file_path.rfind('pySPACE') - 1])

import unittest
from pySPACE.missions.nodes.spatial_filtering.xdawn import *
from pySPACE.resources.data_types.time_series import TimeSeries
import pySPACE.tests.generic_unittest as gen_test
import numpy


class XDAWNTestCase(unittest.TestCase):

    def setUp(self):
        ts_t_1 = TimeSeries([[1.5, -1], [1.5, -1], [1.5, -1], [1.5, -1]],
                            channel_names=["C3", "C4"], sampling_frequency=0.5,
                            start_time=0.0, end_time=3.0)
        ts_s_1 = TimeSeries([[-1, 1.5], [-1, 1.5], [-1, 1.5], [-1, 1.5]],
                            channel_names=["C3", "C4"], sampling_frequency=0.5,
                            start_time=0.0, end_time=3.0)
        ts_t_2 = TimeSeries([[0, 0], [0, 0], [0, 0], [0, 0]],
                            channel_names=["C3", "C4"], sampling_frequency=0.5,
                            start_time=0.0, end_time=3.0)
        ts_s_2 = TimeSeries([[0, 0], [0, 0], [0, 0], [0, 0]],
                            channel_names=["C3", "C4"], sampling_frequency=0.5,
                            start_time=0.0, end_time=3.0)
        self.target = [ts_t_1, ts_t_2]
        self.standard = [ts_s_1, ts_s_2]

    def test_input(self):
        # we test the erp_class_label attribute as well the way
        # inputs are stacked
        x_dawn = XDAWNNode(erp_class_label='Target', visualize_pattern=False)

        for elem in self.standard:
            x_dawn.train(elem, 'Standard')

        for elem in self.target:
            x_dawn.train(elem, 'Target')

        # now we want to check if the input has been successfully categorized
        # in the X and D matrices
        X = [[-1., 1.5], [-1., 1.5], [-1., 1.5], [-1., 1.5],
             [0., 0.], [0., 0.], [0., 0.], [0., 0.],
             [1.5, -1.], [1.5, -1.], [1.5, -1.], [1.5, -1.],
             [0., 0.], [0., 0.], [0., 0.], [0., 0.]]
        D = [[0., 0., 0., 0.], [0., 0., 0., 0.],
             [0., 0., 0., 0.], [0., 0., 0., 0.],
             [0., 0., 0., 0.], [0., 0., 0., 0.],
             [0., 0., 0., 0.], [0., 0., 0., 0.],
             [1., 0., 0., 0.], [0., 1., 0., 0.],
             [0., 0., 1., 0.], [0., 0., 0., 1.],
             [1., 0., 0., 0.], [0., 1., 0., 0.],
             [0., 0., 1., 0.], [0., 0., 0., 1.]]

        # since the data has not been processed yet, we want it to be
        # absolutely the same as the input
        self.assertTrue(numpy.allclose(x_dawn.X, X, atol=0))
        self.assertTrue(numpy.allclose(x_dawn.D, D, atol=0))


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromName('test_xdawn')

    # check the generic unittests
    suite.addTest(gen_test.ParametrizedTestCase.parametrize(
        current_testcase=gen_test.GenericTestCase, node=XDAWNNode))

    unittest.TextTestRunner(verbosity=2).run(suite)
