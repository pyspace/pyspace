#!/usr/bin/python

"""
A module that tests the
:mod:`~pyspace.pySPACE.missions.nodes.postprocessing`
node

:Author:  Andrei Ignat (Andrei_Cristian.Ignat@dfki.de)
:Created: 2014/06/06
"""


if __name__ == '__main__':
    import sys
    import os
    # The root of the code
    file_path = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(file_path[:file_path.rfind('pySPACE') - 1])

import unittest
from pySPACE.missions.nodes.postprocessing.feature_normalization import *
from pySPACE.resources.data_types.feature_vector import FeatureVector
import pySPACE.tests.generic_unittest as gen_test
import numpy


class GaussianFeatureNormalizationTestCase(unittest.TestCase):

    def setUp(self):
        """ initializes the GaussianFeatureNormalizationNode """
        self.node = GaussianFeatureNormalizationNode()

    def test_no_change(self):
        """ checks what the node does to already Gaussian data

        **Principle**
            1) generate data points for the FeatureVector which already follow
            the Gaussian distribution
            2) train the node using these data points
            3) stop the training in order to compute the multiplication
            variable and the translation variable

            In theory, since the data is already Gaussian generated:
                - the multiplication factor should be 1
                - the translation factor should be 0
        """
        data_points = numpy.random.normal(loc=0., scale=1.0, size=(10000, 3))
        f_names = ["TD_S1_0sec", "TD_S2_1sec", "TD_S3_1sec"]
        for point in data_points:
            self.node.train(
                FeatureVector(point,
                              feature_names=f_names))
        self.node.stop_training()
        # Since we are dealing with randomly generated data, fluctuations
        # are inherent and as such, we allow for a relative tolerance
        # margin of 5%
        self.assertTrue(
            numpy.allclose(
                self.node.mult,
                [1., 1., 1.], atol=3.e-2))
        self.assertTrue(
            numpy.allclose(
                self.node.translation,
                [0., 0., 0.], atol=3.e-2))

    def test_gaussian_normalization(self):
        # TODO: Find a test for the execution of the node
        pass


class HistogramFeatureNormalizationTestCase(unittest.TestCase):

    def setUp(self):
        self.node = HistogramFeatureNormalizationNode()

    def test_no_change(self):
        """ checks the effect on an already normalized set

        **Principle**
            The default data set which we import here is already normalized
            and as such, the multiplication and translation factors should
            have no effect on that specific data set.
        """
        from pySPACE.tests.utils.data.test_default_data import all_inputs
        for example in all_inputs["FeatureVector"]:
            self.node.train(example[0])

        self.node.stop_training()
        # Since this data set is no longer randomly generated, we lower
        # the tolerance of the difference between the arrays to rtol=1.e-5
        self.assertTrue(numpy.allclose(self.node.mult, [1., 1.], rtol=1.e-5))
        self.assertTrue(
            numpy.allclose(
                self.node.translation,
                [0., 0.], rtol=1.e-5))

    def test_histogram_normalization(self):
        # TODO: Find a test for the execution of the node
        pass


class EuclideanFeatureNormalizationTestCase(unittest.TestCase):

    def setUp(self):
        self.node = EuclideanFeatureNormalizationNode()

    def test_normalization(self):
        """ compares the FeatureVector result with a manually computed one

        **Principle**
            Try to see for int and float data points whether the normalized
            arrays are the same as the ones obtained by running the nodes
        """
        data_points = [numpy.arange(start=1, stop=1000, dtype=numpy.int64),
                       numpy.arange(start=1, stop=1000, dtype=numpy.longdouble)]

        for point in data_points:
            theoretical = numpy.divide(
                point,
                numpy.sqrt(numpy.sum(point ** 2)))
            result = self.node.execute(
                FeatureVector(point, feature_names=point.astype(str)))
            self.assertTrue(numpy.allclose(result.view(numpy.ndarray)[0, :],
                                           theoretical, atol=0.))
            self.setUp()


class InfinityNormFeatureVisualizationTestCase(unittest.TestCase):

    def setUp(self):
        self.node = InfinityNormFeatureNormalizationNode()

    def test_normalization(self):
        """ compares the FeatureVector result with a manually computed one

        **Principle**
            Each data point is divided by the maximum data point among the
            features
        """
        data_points = [numpy.arange(start=1, stop=1000, dtype=numpy.int64),
                       numpy.arange(start=1, stop=1000, dtype=numpy.longdouble)]

        for point in data_points:
            # since the InfinityNormFeatureNormalizationNode also forces the
            # type to be float, our computational of the theoretical result
            # will also be done with a forced float
            theoretical = numpy.divide(point.astype(numpy.longdouble), 999)
            result = self.node.execute(
                FeatureVector(point, feature_names=point.astype(str)))
            self.assertTrue(numpy.allclose(result.view(numpy.ndarray)[0, :],
                                           theoretical, atol=0.))
            self.setUp()


class OutlierFeatureNormalizationTestCase(unittest.TestCase):

    def test_no_outliers(self):
        """ runs the OutlierFeatureNormalizationNode with no outliers

        **Principle**
            No point in the training data set will be considered to be an
            outlier in the present case. In other words, this test only checks
            whether the resulting FeatureVector contains data points only
            within the [0,1] range
        """
        node = OutlierFeatureNormalizationNode(outlier_percentage=0)

        # we train the node with well behaved data
        for i in range(0, 101):
            point = FeatureVector([float(i), 2. * i, 14. * i - 25],
                                  feature_names=['a', 'b', 'c'])
            node.train(point)

        node.stop_training()
        # we check if the multiplication term corresponds to the
        # largest factor in that specific feature
        self.assertTrue(
            numpy.allclose(
                node.mult,
                [1. / 100, 1. / 200, 1. / 1400]))
        # and now we check whether the translation is done properly
        self.assertTrue(numpy.allclose(node.translation, [0., 0., -25.]))

    def test_with_outliers(self):
        """ test how the class reacts when part of the data is excluded

        **Principle**
            In this test, we exclude half of the elements in each feature of
            the feature vector. The test itself therefore lies in seeing
            whether the middle half of the dataset is normalized between
            0 and 1.

            This happens if the following conditions are met:

            - the translation factor excludes(in this particular case) the
              first and last quarter of the data set

            - the multiplication factor normalizes the array by considering
              the maximum value to be half of the initial maximum value

            The example is best understood by delving into the source code.
        """
        # to make the code more easy to understand, we define the following
        # variables
        max_i = 100.
        percentage = 50.
        float_percentage = percentage / 100.

        node = OutlierFeatureNormalizationNode(outlier_percentage=percentage)

        # we train the node with well behaved data
        for i in range(0, int(max_i) + 1):
            point = FeatureVector([float(i), 2. * i, 14. * i - 25],
                                  feature_names=['a', 'b', 'c'])
            node.train(point)

        node.stop_training()
        # we check if the multiplication term corresponds to the
        # specific percentage of the largest factor in that specific
        # feature

        theoretical_mult = [1. / (max_i * (1 - float_percentage)),
                            1. / (2 * max_i * (1 - float_percentage)),
                            1. / (14 * max_i * (1 - float_percentage))]
        theoretical_trans = [max_i * float_percentage / 2,
                             2 * max_i * float_percentage / 2,
                             14 * max_i * float_percentage / 2 - 25.]

        self.assertTrue(numpy.allclose(node.mult,
                                       theoretical_mult))

        # and now we check whether the translation is done properly
        self.assertTrue(numpy.allclose(node.translation,
                                       theoretical_trans))

        # just for illustration purposes, we copy/paste the code and change
        # the initial parameters
        max_i = 3500.
        percentage = 67.
        float_percentage = percentage / 100.

        node = OutlierFeatureNormalizationNode(outlier_percentage=percentage)

        for i in range(0, int(max_i) + 1):
            point = FeatureVector([float(i), 2. * i, 14. * i - 25],
                                  feature_names=['a', 'b', 'c'])
            node.train(point)

        node.stop_training()
        theoretical_mult = [1. / (max_i * (1 - float_percentage)),
                            1. / (2 * max_i * (1 - float_percentage)),
                            1. / (14 * max_i * (1 - float_percentage))]
        self.assertTrue(numpy.allclose(node.mult,
                                       theoretical_mult))

        theoretical_trans = [max_i * float_percentage / 2,
                             2 * max_i * float_percentage / 2,
                             14 * max_i * float_percentage / 2 - 25.]
        self.assertTrue(numpy.allclose(node.translation,
                                       theoretical_trans))

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromName(
        'test_feature_normalization')

    # check the generic unittests
    suite.addTest(gen_test.ParametrizedTestCase.parametrize(
        current_testcase=gen_test.GenericTestCase,
        node=OutlierFeatureNormalizationNode))
    suite.addTest(gen_test.ParametrizedTestCase.parametrize(
        current_testcase=gen_test.GenericTestCase,
        node=GaussianFeatureNormalizationNode))
    suite.addTest(gen_test.ParametrizedTestCase.parametrize(
        current_testcase=gen_test.GenericTestCase,
        node=HistogramFeatureNormalizationNode))
    suite.addTest(gen_test.ParametrizedTestCase.parametrize(
        current_testcase=gen_test.GenericTestCase,
        node=EuclideanFeatureNormalizationNode))

    unittest.TextTestRunner(verbosity=2).run(suite)
