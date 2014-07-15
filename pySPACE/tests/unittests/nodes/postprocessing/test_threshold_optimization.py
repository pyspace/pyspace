#!/usr/bin/python

"""
This module contains unittests which test the time domain
Feature Extraction node

:Author: Andrei Ignat (Andrei_Cristian.Ignat@dfki.de)
:Created: 2014/06/16
"""

import unittest

if __name__ == '__main__':
    import sys
    import os
    # The root of the code
    file_path = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(file_path[:file_path.rfind('pySPACE') - 1])

from pySPACE.missions.nodes.postprocessing.threshold_optimization import ThresholdOptimizationNode
import pySPACE.tests.generic_unittest as gen_test
from pySPACE.resources.data_types.prediction_vector import PredictionVector

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromName(
        'test_threshold_optimization')

    # Test the generic initialization of the class methods
    suite.addTest(gen_test.ParametrizedTestCase.parametrize(
        current_testcase=gen_test.GenericTestCase,
        node=ThresholdOptimizationNode))
    outputs = [PredictionVector(prediction=-8. / 3, label="Standard"),
               PredictionVector(prediction=-5. / 3, label="Standard"),
               PredictionVector(prediction=-2. / 3, label="Standard"),
               PredictionVector(prediction=1 / 3., label="Target"),
               PredictionVector(prediction=4. / 3, label="Target")]
    suite.addTest(gen_test.ParametrizedTestCase.parametrize(
        current_testcase=gen_test.OutputTestCase,
        node=ThresholdOptimizationNode,
        output=outputs
    ))

    unittest.TextTestRunner(verbosity=2).run(suite)
