#!/usr/bin/python

""" Unit tests for FeatureVector data type

:Author: Titiruck Nuntapramote (titiruck.nuntapramote@dfki.de)
:Created: 2011/04/23
"""


import unittest
if __name__ == '__main__':
    import sys
    import os
    # The root of the code
    file_path = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(file_path[:file_path.rfind('pySPACE')-1])

from pySPACE.resources.data_types.feature_vector import FeatureVector
import numpy


class FeatureVectorTestCase(unittest.TestCase):
    """Test for FeatureVector data type"""
    def setUp(self):
        """ Define some feature vectors"""
        # no tag
        self.f1 = FeatureVector([1,2,3,4,5,6],['a','b','c','d','e','f'])
        # no -
        self.f2 = FeatureVector([1,2,3,4,5,6],['a','b','c','d','e','f'], tag = 'Tag of f2')
        # no tag
        self.f3 = FeatureVector([1,2], ['a','b'])
        # no feature_names
        self.f4 = FeatureVector([1,2])

    def test_get_feature_names(self):
        self.assertEqual(self.f1.feature_names, self.f1.get_feature_names())
        self.assertEqual(self.f2.feature_names, self.f2.get_feature_names())
        self.assertEqual(self.f3.feature_names, self.f3.get_feature_names())
        self.assertEqual(self.f4.get_feature_names(),
                         ["feature_0_0.000sec","feature_1_0.000sec"])

    def test_set_feature_names(self):
        self.f1.set_feature_names(['m','n','o','p','q'])
        self.assertEqual(self.f1.feature_names, ['m','n','o','p','q'])

        self.f4.set_feature_names(['a','b'])
        self.assertEqual(self.f4.feature_names, ['a','b'])

    def test_replace_data(self):
        data = FeatureVector.replace_data(self.f2,[10,20,30,40,50,60])
        self.assertFalse((data.view(numpy.ndarray)-[10,20,30,40,50,60]).any())
        self.assertEqual(data.feature_names, ['a','b','c','d','e','f'])
        self.assertEqual(data.tag, 'Tag of f2')

        data2 = FeatureVector.replace_data(self.f1, [4,5,6,7,8,9],
                                           feature_names=['m','n','o','p','q','r'])
        self.assertFalse((data2.view(numpy.ndarray)-[4,5,6,7,8,9]).any())
        self.assertEqual(data2.feature_names, ['m','n','o','p','q','r'])
        self.assertEqual(data2.tag, None)

    def test_equal_vectors(self):
        # this is a very simple test that is meant to check if the equality
        # between two FeatureVectors is correctly assessed

        self.assertEqual(self.f1, self.f2)
        self.assertEqual([[self.f1], [self.f2]], [[self.f2], [self.f1]])
        self.assertEqual([self.f1, self.f2], [self.f2, self.f1])

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromName('test_feature_vector')
    unittest.TextTestRunner(verbosity=2).run(suite)
