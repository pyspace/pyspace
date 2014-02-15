#!/usr/bin/env python
"""
This module contains unittests that test splitter nodes

:Author: Jan Hendrik Metzen (jhm@informatik.uni-bremen.de)
:Created: 2008/12/18
"""

import unittest

if __name__ == '__main__':
    import sys
    import os
    # The root of the code
    file_path = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(file_path[:file_path.rfind('pySPACE')-1])

from pySPACE.missions.nodes.splitter.cv_splitter import CrossValidationSplitterNode
try:
    from pySPACE.tests.utils.data.test_data_generation import SimpleTimeSeriesSourceNode
except:
    from pySPACE.missions.nodes.source.test_source_nodes import SimpleTimeSeriesSourceNode
import numpy
        
class CrossValidationSplitterTestCase(unittest.TestCase):
    
    def setUp(self):
        self.source = SimpleTimeSeriesSourceNode()
        
        self.cv_splitter = CrossValidationSplitterNode(splits = 10)
        self.cv_splitter.register_input_node(self.source)
        
        
    def test_cv_coverage_by_testdata(self):
        """
        Tests that all data points are contained at least once in a test set
        """
        all_testdata = []        
        # For every split of the dataset
        while True: # As long as more splits are available
            # Append all test data of the current split
            all_testdata.extend(self.cv_splitter.request_data_for_testing())
            
            # If no more splits are available
            if not self.cv_splitter.use_next_split():
                break
            
        # Check that every data point from the source was once in a test set
        for orig_data, orig_label in self.source.time_series:
            found = False
            for test_data, test_label in all_testdata:
                found |= (orig_data.view(numpy.ndarray) == test_data.view(numpy.ndarray)).all() \
                            and (orig_label == test_label)
                if found: break
            self.assert_(found,
                         "One data point is never used for testing in cv splitting") 
        
    def test_cv_coverage_by_split(self):
        """
        Tests that each split during crossvalidation covers the whole data set
        """ 

        # For every split of the dataset
        while True: # As long as more splits are available
            split_data = []
            # Append all data of the current split
            split_data.extend(self.cv_splitter.request_data_for_training(False))
            split_data.extend(self.cv_splitter.request_data_for_testing())
            
            # Check that every data point from the source was once in a test set
            for orig_datapoint, orig_label in self.source.time_series:
                found = False
                for split_datapoint, split_label in split_data:
                    found |= (orig_datapoint.view(numpy.ndarray) == split_datapoint.view(numpy.ndarray)).all() \
                                and (orig_label == split_label)
                    if found: break
                
                self.assert_(found, 
                             "One data point is neither used for training nor for testing in one cv split")
            
            # If no more splits are available
            if not self.cv_splitter.use_next_split():
                break
    
    def test_cv_train_test_seperation(self):
        """ Test that no data point is contained in train and test set """
        #For every split of the dataset
        while True: # As long as more splits are available
            # Check that no data point in the training is used for testing
            train_data = list(self.cv_splitter.request_data_for_training(False))
            test_data = list(self.cv_splitter.request_data_for_testing())
            for training_datapoint, train_label in train_data:
                doublet = False
                for test_datapoint, test_label in test_data:
                    doublet |= (training_datapoint.view(numpy.ndarray) == test_datapoint.view(numpy.ndarray)).all() \
                                    and (train_label == test_label)
                    if doublet: 
                        break
                self.assert_(not doublet, 
                             "In one split of the cv splitter, a sample is used for training and testing")
            
            # If no more splits are available
            if not self.cv_splitter.use_next_split():
                break
    
    def test_cv_no_iterated_splitters(self):
        """ Splitter cannot be applied to a node chain, that has already been split """
        second_cv_splitter = CrossValidationSplitterNode(splits = 10)
        second_cv_splitter.register_input_node(self.cv_splitter)
        
        #check that the proper Exception is raised
        #catch the exception and then do assertEqual
        try:
            second_cv_splitter.request_data_for_training(use_test_data=False)
            self.assert_(False,"Concatenation of several splitters should not be possible!")
        except Exception as e:
            #cv_splitter just use raise Exception(msg)but Exception.message has been deprecated
            #possible solution is to define own Exception subclass
            self.assertEqual(str(e), "No iterated splitting of data sets allowed\n " +
                            "(Calling a splitter on a data set that is " +
                            "already split)",
                            "Concatenation of several splitters should not be possible!")
       
    def test_cv_dependance_on_run_number(self):
        """
        Tests that the splitting of the data by a cv splitter node 
        is randomized by the run number
        """
        second_cv_splitter = CrossValidationSplitterNode(splits = 10)
        second_cv_splitter.register_input_node(self.source)
        # Test whether the two splitter give different results for two
        # arbitrary run numbers (say 7 and 8)
        second_cv_splitter.set_run_number(7)
        self.cv_splitter.set_run_number(8)
        
        train_data1 = list(self.cv_splitter.request_data_for_training(False))
        train_data2 = list(second_cv_splitter.request_data_for_training(False))
        
        # Check that there is a  data point in the training set generated by
        # the first splitter that is not in the set of the second splitter
        # NOTE: The small chance that they produce the same split for the 
        #       specific numbers 7 and 8 but not for all run numbers is neglected...   
        one_not_contained = False
        for datapoint1, label1 in train_data1:
            this_contained = False
            for datapoint2, label2 in train_data2:
                this_contained |= (datapoint1.view(numpy.ndarray) == datapoint2.view(numpy.ndarray)).all() and (label1 == label2)
                if this_contained: break
            one_not_contained |= (not this_contained)
            if one_not_contained: break
                
        self.assert_(one_not_contained,
                     "CV Splitter generated the same split for two run numbers")
     
    def test_cv_reproducibility(self):
        """
        Tests that the splitting of the data by a cv splitter node 
        is deterministic given the run_number
        """
        second_cv_splitter = CrossValidationSplitterNode(splits = 10)
        second_cv_splitter.register_input_node(self.source)
        # Test whether the two splitter give the same results for an
        # arbitrary run number (say 7)
        second_cv_splitter.set_run_number(7)
        self.cv_splitter.set_run_number(7)
        
        train_data1 = list(self.cv_splitter.request_data_for_training(False))
        train_data2 = list(second_cv_splitter.request_data_for_training(False))
        
        # Check that all data points in the training set generated by
        # the first splitter are also in the set of the second splitter
        all_contained = True
        for datapoint1, label1 in train_data1:
            this_contained = False
            for datapoint2, label2 in train_data2:
                this_contained |= (datapoint1.view(numpy.ndarray) == datapoint2.view(numpy.ndarray)).all() and (label1 == label2)
                if this_contained: break
            all_contained &= this_contained
            if not all_contained: break     
        
        self.assert_(all_contained,
                     "CV Splitter generated different splits for the same run numbers")
            
    
if __name__ == '__main__':
    
    suite = unittest.TestLoader().loadTestsFromName('test_cv_splitter')
    
    unittest.TextTestRunner(verbosity=2).run(suite)