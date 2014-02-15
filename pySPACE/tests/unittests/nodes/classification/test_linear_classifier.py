#!/usr/bin/env python
"""
This module contains unittests which test linear classifier nodes.

:Author: Jan Hendrik Metzen (jhm@informatik.uni-bremen.de)
:Created: 2008/08/26
"""
import numpy

import unittest

if __name__ == '__main__':
    import sys
    import os
    # The root of the code
    file_path = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(file_path[:file_path.rfind('pySPACE')-1])

from pySPACE.missions.nodes.classification.linear_classifier import FDAClassifierNode
from pySPACE.resources.data_types.feature_vector import FeatureVector

from pySPACE.missions.nodes.classification.linear_classifier import NaiveBayesClassifierNode

class NaiveBayesClassifierTestCase(unittest.TestCase):
    """ unittest for NBClassifierNode """
    
    def test_nb(self):
        nb_node = NaiveBayesClassifierNode(output_dim=1)
        nb_node.train(numpy.array([[6,180,12]]), 'male')
        nb_node.train(numpy.array([[5.92,190,11]]), 'male')
        nb_node.train(numpy.array([[5.58,170,12]]), 'male')
        nb_node.train(numpy.array([[5.92,165,10]]), 'male')

        nb_node.train(numpy.array([[5,100,6]]), 'female')
        nb_node.train(numpy.array([[5.5,150,8]]), 'female')
        nb_node.train(numpy.array([[5.42,130,7]]), 'female')
        nb_node.train(numpy.array([[5.75,150,9]]), 'female')

        nb_node.stop_training()
        
        classification_female = nb_node._execute(numpy.array( [[6,130,8] ] ))
        self.assert_( classification_female.label =='female' )
        

class FDAClassifierTestCase(unittest.TestCase):
    """ unittest for FDAClassifierNode """
    def setUp(self):
        """ Generate separable training data """
        numpy.random.seed(0) # to avoid the very unlikely case that the data is not linearly separable
        self.x_a = numpy.random.multivariate_normal(numpy.array([0.0,0.0]),
                                                    numpy.array([[1.0,1.0],[1.0,0.0]]),
                                                    200)
        self.x_b = numpy.random.multivariate_normal(numpy.array([10.0,10.0]),
                                                    numpy.array([[1.0,1.0],[1.0,0.0]]),
                                                    200)
    
    def test_fda(self):
        """ Train FDA and test on training data """
        fda_node = FDAClassifierNode() #(generalized) Fisher Discriminant Analysis
        for x in self.x_b:
            fda_node.train(FeatureVector(x), 'b')
        for x in self.x_a:
            fda_node.train(FeatureVector(x), 'a')
        fda_node.stop_training()
        
        # for calling execute we need FeatureVectors since meta data is handled there
        self.x_a = [FeatureVector(numpy.atleast_2d(elem)) for elem in self.x_a]
        self.x_b = [FeatureVector(numpy.atleast_2d(elem)) for elem in self.x_b]
        classification_a = [fda_node.execute(fv).label for fv in self.x_a]
        classification_b = [fda_node.execute(fv).label for fv in self.x_b]
        
        self.assert_(numpy.alltrue(map(lambda x: x=='a',classification_a)))
        self.assert_(numpy.alltrue(map(lambda x: x=='b',classification_b)))

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromName('test_linear_classifier')   
    unittest.TextTestRunner(verbosity=2).run(suite)
