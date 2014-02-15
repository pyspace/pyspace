#!/usr/bin/env python
"""
This module contains unittests which test FeatureVector2TimeSeriesNode.

:Author: Titiruck Nuntapramote (titiruck.nuntapramote@dfki.de)
:Created: 2011/10/18
"""

import numpy as np
import unittest
import random

if __name__ == '__main__':
    import sys
    import os
    # The root of the code
    file_path = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(file_path[:file_path.rfind('pySPACE')-1])

from pySPACE.resources.data_types.time_series import TimeSeries
from pySPACE.missions.nodes.feature_generation.time_domain_features import TimeDomainFeaturesNode
from pySPACE.missions.nodes.type_manipulation.type_conversion import *
from pySPACE.tests.utils.data.test_data_generation import *
from pySPACE.resources.data_types.prediction_vector import PredictionVector

class FeatureVector2TimeSeriesTestCase(unittest.TestCase):
    """
    Test whether 2 TimeSeries are equal after converting
    using FeatureVector2TimeSeriesNode.
    
    :Author: Titiruck Nuntapramote (titiruck.nuntapramote@dfki.de)
    :Created: 2011/11/04
    
    """  
    def setUp(self):
        ch_names = [("ch%s" % i) for i in range(10)]
    
        self.ts1 = TestTimeSeriesGenerator().generate_test_data(channels=10,
                                                           channel_names = ch_names,
                                                            time_points=1000,
                                                            function=Sine(),
                                                            sampling_frequency=50.0)
        fv1 = TimeDomainFeaturesNode()._execute(self.ts1)
        self.ts2 = FeatureVector2TimeSeriesNode()._execute(fv1)
                
        ch_names2 = [("ch%s" % i) for i in range(15)]
        self.ts3 = TestTimeSeriesGenerator().generate_test_data(channels=15,
                                                           channel_names = ch_names2,
                                                            time_points=500,
                                                            function=GaussianNoise(),
                                                            sampling_frequency=20.0)
        fv2 = TimeDomainFeaturesNode()._execute(self.ts3) 
        self.ts4 = FeatureVector2TimeSeriesNode()._execute(fv2)
    
    def test_sampling_frequency(self):
        self.assertEqual((self.ts1).sampling_frequency,(self.ts2).sampling_frequency)
        self.assertEqual((self.ts3).sampling_frequency,(self.ts4).sampling_frequency)

    def test_channel_names(self):
        self.assertEqual(set(self.ts1.channel_names), set(self.ts2.channel_names))
        self.assertEqual(set(self.ts3.channel_names), set(self.ts4.channel_names))
    
    # Compare value to each list in TimeSeries.
    def cmp_value(self,l1, l2):
        diff = np.array(l1, dtype='f4') - np.array(l2, dtype='f4')
        if not(diff.any()): return True
        return False
    
    # Compare value in the whole TimeSeries
    def cmp_TimeSeries(self,t1,t2):
        t1=t1.view(numpy.ndarray)
        t2=t2.view(numpy.ndarray)
        for i in range(len(t1)):
            val = self.cmp_value(t1[i][0],t2[i][0])
            if not(val):
                return False
        return True

    def test_TimeSeries(self):
        self.assertTrue(self.cmp_TimeSeries(self.ts1,self.ts2))
        self.assertTrue(self.cmp_TimeSeries(self.ts3,self.ts4))
        self.assertFalse(self.cmp_TimeSeries(self.ts1,self.ts4))
        self.assertFalse(self.cmp_TimeSeries(self.ts3,self.ts1))
        
class Prediction2FeaturesTestCase(unittest.TestCase):
    """ Test Prediction2FeaturesNode
    
    :Author: Titiruck Nuntapramote (titiruck.nuntapramote@dfki.de)
    :Created: 2011/11/16
    """
    
    def setUp(self):
        
        ran1 = random.randint(2,100)
        ran2 = random.randint(2,100)
        
        list1 = range(ran1)
        list2 = range(ran2)
            
        self.pv1 = PredictionVector(prediction = 1) 
        self.pv2 = PredictionVector(prediction = 2) 
        self.pv3 = PredictionVector(prediction = list1)
        self.pv4 = PredictionVector(prediction = list2)
        
        self.fv1 = Prediction2FeaturesNode()._execute(self.pv1)
        self.fv2 = Prediction2FeaturesNode(name='test')._execute(self.pv2)
        self.fv3 = Prediction2FeaturesNode()._execute(self.pv3)
        self.fv4 = Prediction2FeaturesNode(name='test')._execute(self.pv4)
    
    def test_array(self):
        self.assertEqual(self.fv1.view(numpy.ndarray), self.pv1.view(numpy.ndarray), "fv1 is incorrect")
        self.assertEqual(self.fv2.view(numpy.ndarray), self.pv2.view(numpy.ndarray), "fv2 is incorrect")
        
        diff = np.array(self.fv3) - np.array(self.pv3)
        self.assertTrue(not(diff.all()), "fv3 is incorrect")
        
        diff2 = np.array(self.fv4) - np.array(self.pv4) 
        self.assertTrue(not(diff2.all()), "fv4 is incorrect")
        
    def test_name(self):
        self.assertEqual(self.fv1.feature_names, ["prediction"], "fv1 is incorrect")
        self.assertEqual(self.fv2.feature_names, ["testprediction"], "fv1 is incorrect")
        
        #compare size
        self.assertEqual(self.fv3.shape[1], len(self.fv3.feature_names), "len(fv3.feature_names) is incorrect")
        self.assertEqual(self.fv4.shape[1], len(self.fv4.feature_names), "len(fv4.feature_names) is incorrect")
        
        msg = ''
        
        equal = True    
        for i in range(len(self.fv3.feature_names)):
            if self.fv3.feature_names[i] != "prediction_"+str(i):
                equal = False
                msg = str(i)
                break       
        self.assertTrue(equal, "fv3.feature_names is incorrect at" + msg)
        
        equal2 = True
        for i in range(len(self.fv4.feature_names)):
            if self.fv4.feature_names[i] != "testprediction_"+str(i):
                equal = False
                msg = str(i)
                break
        self.assertTrue(equal2, "fv4.feature_names is incorrect at" + msg)
        
class Features2PredictionTestCase(unittest.TestCase):
    """ Test Features2PredictionNode
    
    :Author: Titiruck Nuntapramote (titiruck.nuntapramote@dfki.de)
    :Created: 2011/11/24
    """
    
    def setUp(self):
        
        ran1 = random.randint(2,100)
        ran2 = random.randint(2,100)
        
        list1 = range(-10, ran1)
        list2 = range(-5, ran2)
            
        self.fv1 = FeatureVector(np.array([list1]))
        self.fv2 = FeatureVector(np.array([list2]))
        
        self.class_labels = ['standard', 'target']
        self.pv1 = Features2PredictionNode(self.class_labels)._execute(self.fv1)
        self.pv2 = Features2PredictionNode(self.class_labels)._execute(self.fv2)
        
        self.classification = lambda x: self.class_labels[0] if x <= 0 \
                                        else self.class_labels[1]
        
        self.pv1_label = map(self.classification, self.pv1.view(numpy.ndarray)[0,:])
        self.pv2_label = map(self.classification, self.pv2.view(numpy.ndarray)[0,:])
        
    def test_array(self):
        
        diff = np.array(self.fv1) - np.array(self.pv1)
        self.assertTrue(not(diff.all()), "pv1 is incorrect")
        
        diff2 = np.array(self.fv2) - np.array(self.pv2)
        self.assertTrue(not(diff2.all()), "pv2 is incorrect")
        
    def test_label(self):
        self.assertEqual(self.pv1_label, self.pv1.label, "pv1 label is incorrect")
        self.assertEqual(self.pv2_label, self.pv2.label, "pv1 label is incorrect")

class Feature2MonoTimeSeriesTestCase(unittest.TestCase):
    """ Some simple tests, if conversion works as expected """
    
    def setUp(self):
        """ Define basic needed FeatureVector instances """
        self.x =FeatureVector([[0,1,2,3,4,5]],["a","b","ab","cb","c4","abc"])
        self.y =FeatureVector([[0,1,2,3,4,5]],["a_7ms","b_7ms","ab_7ms","cb_7ms","c4_7ms","abc_7ms"])
        self.tx = TimeSeries([[0,1,2,3,4,5]],channel_names=["a","b","ab","cb","c4","abc"],sampling_frequency=1)
        self.ty = TimeSeries([[0,1,2,3,4,5]],channel_names=["a_7ms","b_7ms","ab_7ms","cb_7ms","c4_7ms","abc_7ms"],sampling_frequency=1)

    def test_conversion(self):
        """ Two simple conversion tests """
        assert(Feature2MonoTimeSeriesNode()._execute(self.x)==self.tx), "Transformation to TimeSeries failed"
        assert(Feature2MonoTimeSeriesNode()._execute(self.y)==self.ty), "Transformation to TimeSeries failed!"

class MonoTimeSeries2FeatureTestCase(Feature2MonoTimeSeriesTestCase):
    def test_conversion(self):
        """ Two simple conversion tests"""
        assert(MonoTimeSeries2FeatureNode()._execute(self.tx)==self.x), "Transformation to FeatureVector failed"
        assert(MonoTimeSeries2FeatureNode()._execute(self.ty)==self.y), "Transformation to FeatureVector failed!"

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromName('test_type_conversion')
    unittest.TextTestRunner(verbosity=2).run(suite)

