#!/usr/bin/python

""" Unittests for feature_filter.py """

if __name__ == '__main__':
    import sys
    import os
    # The root of the code
    file_path = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(file_path[:file_path.rfind('pySPACE') - 1])

import unittest

from pySPACE.resources.data_types.feature_vector import FeatureVector
from pySPACE.missions.nodes.feature_selection.feature_filter import FeatureNameFilterNode as TestNode


class FeatureNameFilterTestCase(unittest.TestCase):

    """ Some simple tests, if filtering works as expected """

    def setUp(self):
        """ Define basic needed FeatureVector instances """
        self.x = FeatureVector(
            [[0, 1, 2, 3, 4, 5]],
            ["a", "b", "ab", "cb", "c4", "abc"])
        self.a = FeatureVector(
            [[0, 2, 5]],
            ["a", "ab", "abc"])
        self.na = FeatureVector(
            [[1, 3, 4]],
            ["b", "cb", "c4"])
        self.a4 = FeatureVector(
            [[0, 2, 4, 5]],
            ["a", "ab", "c4", "abc"])

    def test_filter_a(self):
        """ Delete or use all features containing 'a' """
        ex_node = TestNode(exclude_patterns=["a"])
        assert(ex_node._execute(self.x) == self.na), "Exclude pattern failed!"
        inc_node = TestNode(exclude_patterns="All", include_patterns=["a"])
        assert(inc_node._execute(self.x) == self.a), "Include pattern failed!"

    def test_filter_a_and_c4_a(self):
        """ include a and c4 """
        inc_node = TestNode(
            exclude_patterns="All",
            include_patterns=["a"],
            include_names=["c4"])
        assert(inc_node._execute(self.x) == self.a4), "Include pattern failed!"
