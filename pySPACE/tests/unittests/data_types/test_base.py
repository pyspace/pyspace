#!/usr/bin/env python
"""Unit test for BaseData type

    This unit test creates TimeSeries objects and FeatureVector,
    tries to change and inherit meta information and runs separate
    tests for key, tag, specs and inheritance.
 
    .. todo::  test pickling?
    
    :Author: Sirko Straube (sirko.straube@dfki.de), David Feess
    :Last Revision: 2012/04/02
"""

if __name__ == '__main__':
    import sys
    import os
    # The root of the code
    file_path = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(file_path[:file_path.rfind('pySPACE')-1])
         
from pySPACE.resources.data_types.time_series import TimeSeries
from pySPACE.resources.data_types.feature_vector import FeatureVector
import unittest, numpy


class BaseDataTestCase(unittest.TestCase):
    """Test BaseData data type"""
    def setUp(self):
        """Create some example data """
        # Create some TimeSeries:
        self.x1 = TimeSeries([1,2,3,4,5,6], ['a','b','c','d','e','f'], 12,
                        marker_name='S4', name='Name_text ending with Standard',
                        start_time=1000.0, end_time=1004.0)
        
        self.x1.specs={'Nice_Parameter': 1, 'Less_Nice_Param': '2'}
        self.x1.generate_meta() #automatically generate key and tag
                        
        self.x2 = TimeSeries([1,2,3,4,5,6], ['a','b','c','d','e','f'], 12,
                        marker_name='S4', start_time=2000.0, end_time=2004.0, 
                        name='Name_text ending with Standard')
        
        #manually generate key and tag
        import uuid
        self.x2_key=uuid.uuid4()
        self.x2.key=self.x2_key
        self.x2.tag='Tag of x2'
        self.x2.specs={'Nice_Parameter': 1, 'Less_Nice_Param': '2'}
                         
        self.x3 = TimeSeries([1,2,3,4,5,6], ['a','b','c','d','e','f'], 12,
                        marker_name='S4', start_time=3000.0, end_time=3004.0)
        
        self.x3.specs={'Nice_Parameter': 1, 'Less_Nice_Param': '2'}
        self.x3.generate_meta()
        
        self.x4 = TimeSeries([1,2,3,4,5,6], ['a','b','c','d','e','f'], 12,marker_name='S4')
        
        self.x4.specs={'Nice_Parameter': 1, 'Less_Nice_Param': '2'}
        
        self.x5 = TimeSeries([1,2], ['a','b'], 12)
        self.x5.inherit_meta_from(self.x2)
        
        self.x6 = TimeSeries([1,2,3,4,5,6], ['a','b','c','d','e','f'], 12)
        
        self.x6.specs={'Nice_Parameter': 11, 'Less_Nice_Param': '21'}
        self.x6.generate_meta()
        #safe information
        self.x6_key=self.x6.key
        
        self.x6.inherit_meta_from(self.x2)
        
        self.some_nice_dict = {'guido': 4127, 'irv': 4127, 'jack': 4098}
        
        self.x6.add_to_history(self.x5, self.some_nice_dict)
        
        # Create some FeatureVectors:
        self.f1 = FeatureVector([1,2,3,4,5,6],['a','b','c','d','e','f'])
        
        self.f1.specs={'NiceParam':1,'LessNiceParam':2}
        
        self.f2 = FeatureVector([1,2,3,4,5,6],['a','b','c','d','e','f'], tag = 'Tag of f2')
        
        self.f2.specs={'NiceParam':1,'LessNiceParam':2}
        
        self.f3 = FeatureVector([1,2], ['a','b'])
        self.f3.inherit_meta_from(self.x2)
        self.f3.add_to_history(self.x5)
        
    def testTag(self):
        """Test tag behavior"""
        # Generate from Meta Data
        self.assertEqual(self.x1.tag,
            'Epoch Start: 1000ms; End: 1004ms; Class: Standard')
        # Tag passed, use that!
        self.assertEqual(self.x2.tag, 'Tag of x2')
        self.assertEqual(self.f2.tag, 'Tag of f2')            
        # No tag and only partial meta passed
        self.assertEqual(self.x3.tag,
            'Epoch Start: 3000ms; End: 3004ms; Class: na')
        # No Tag and no meta passed, Tag remains None
        self.assertEqual(self.x4.tag, None)
        self.assertEqual(self.f1.tag, None)
        
    
    def testKey(self):
        """Test key behavior"""
        import uuid
        self.assertEqual(type(self.x1.key),uuid.UUID)
        # If Key passed, use that!
        self.assertEqual(self.x2.key, self.x2_key)
        
    
    def testInheritAndAddStuff(self):
        """test inheritance of meta data from other objects"""
        # Inherit
        self.assertEqual(self.x5.tag, self.x2.tag)
        self.assertEqual(self.x5.key, self.x2.key)
        
        self.assertEqual(self.f3.tag, self.x2.tag)
        self.assertEqual(self.f3.key, self.x2.key)
        
        #Inherit
        
        #suppress warning of BaseData type and cast data back to numpy
        hist_x6=self.x6.history[0].view(numpy.ndarray)
        data_x5=self.x5.view(numpy.ndarray)
        
        # history
        self.assertEqual((hist_x6==data_x5).all(),True)
        self.assertEqual(self.x6.history[0].key,self.x5.key)
        self.assertEqual(self.x6.history[0].tag,self.x5.tag)
        self.assertEqual(self.x6.history[0].specs['node_specs'],self.some_nice_dict)
        
        hist_f3=self.f3.history[0].view(numpy.ndarray)
        
        self.assertEqual((hist_f3==data_x5).all(),True)
        self.assertEqual(self.f3.history[0].key,self.x5.key)
        self.assertEqual(self.f3.history[0].tag,self.x5.tag)
        
        #if key (and tag) were already set, these original values
        #have to be kept
        # 
        self.assertEqual(self.x6.key, self.x6_key)
        self.assertEqual(self.x6.tag, self.x2.tag)
        
        self.x6.inherit_meta_from(self.f3) #should not change tag and key
        
        self.assertEqual(self.x6.key, self.x6_key)
        self.assertEqual(self.x6.tag, self.x2.tag)
        
        #testing multiple histories
        x7 = TimeSeries([1,2,3,4,5,6], ['a','b','c','d','e','f'], 12,marker_name='S4')
        x7.add_to_history(self.x1)
        x7.add_to_history(self.x2)
        x7.add_to_history(self.x3)
        x7.add_to_history(self.x4)
        x7.add_to_history(self.x5)
        x7.add_to_history(self.x6)
        x7.add_to_history(self.x1)
        
        self.assertEqual(len(x7.history),7)
        self.assertEqual(x7.history[0].key,x7.history[6].key)
        self.assertEqual(x7.history[5].history,[])        
    
    def testSpecs(self):
        """Test specs behavior"""
        # so far, there's not much going on with specs...
        # same problem as in testkey
        # timeseries doesn't set spec
        self.assertEqual(self.x1.specs, 
                         {'Nice_Parameter': 1, 'Less_Nice_Param': '2'})
        # Inherit
        self.assertEqual(self.x5.specs,self.x2.specs)    
    
if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromName('test_base_data')
    unittest.TextTestRunner(verbosity=2).run(suite)
