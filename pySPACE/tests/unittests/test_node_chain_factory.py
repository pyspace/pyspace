#!/usr/bin/env python
""" Unit tests for the NodeChainFactory

:Author: Jan Hendrik Metzen (jhm@informatik.uni-bremen.de)
:Created: 2008/11/04
"""


import unittest
import sys
import os

if __name__ == '__main__':
    # The root of the code
    file_path = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(file_path[:file_path.rfind('pySPACE')-1])
      
import pylab

from pySPACE.environments.chains.node_chain import NodeChain, NodeChainFactory
from pySPACE.missions.nodes.preprocessing.normalization import DetrendingNode
from pySPACE.missions.nodes.preprocessing.subsampling import SubsamplingNode
from pySPACE.missions.nodes.spatial_filtering.csp import CSPNode
from pySPACE.missions.nodes.source.time_series_source import TimeSeriesSourceNode
import pySPACE.missions.support.windower as windower
from pySPACE.missions.support.windower import WindowFactory
                                          

class NodeChainFactoryTestCase(unittest.TestCase):
    
    def setUp(self):
        pass
           
    def test_dataflow_from_yaml(self):
        simpleYAMLInput ="""
-
    node : Time_Series_Source
-
    node : Detrending
    parameters : 
        detrend_method : "eval(__import__('pylab').detrend_mean)"
- 
    node : Subsampling
    parameters : 
        target_frequency : 100.0
- 
    node : CSP
    parameters : 
         retained_channels : 4
""" 

        flow = NodeChainFactory.flow_from_yaml(NodeChain,
                                          simpleYAMLInput)
        self.assert_(isinstance(flow, NodeChain) and len(flow) == 4)
        self.assert_(isinstance(flow[0], TimeSeriesSourceNode) and
                     isinstance(flow[1], DetrendingNode) and
                     isinstance(flow[2], SubsamplingNode) and
                     isinstance(flow[3], CSPNode))
        self.assert_(flow[1].detrend_method == pylab.detrend_mean)
        self.assert_(flow[2].target_frequency == 100.0)
        self.assert_(flow[3].retained_channels == 4)
        
    def test_window_definitions_from_yaml(self):
        simpleYAMLInput = \
"""
skip_ranges : 
         - {start : 0, end: 15000}
window_defs :
     s16: 
         classname : A
         markername : "S 16"
         startoffsetms : -1400
         endoffsetms : -120
         jitter : 0
         excludedefs : []
     null: 
         classname : B
         markername : "null"
         startoffsetms : -1280
         endoffsetms : 0
         jitter : 0
         excludedefs : [all]
exclude_defs: 
      all:
        markernames : ["S  1", "S  2", "S  8", "S 16", "S 24", "S 32"]
        preexcludems : 2000
        postexcludems : 2000
""" 

        windows = WindowFactory.window_definitions_from_yaml(simpleYAMLInput)
        self.assert_(len(windows) == 2 
                     and isinstance(windows[0], windower.LabeledWindowDef)
                     and isinstance(windows[1], windower.LabeledWindowDef))
        self.assert_(set([windows[0].classname, windows[1].classname])
                      == set(['A', 'B']))
        self.assert_(set([windows[0].markername, windows[1].markername])
                      == set(['S 16', 'null']))
        self.assert_(set([windows[0].startoffsetms, windows[1].startoffsetms]) 
                      == set([-1400, -1280]))
        self.assert_(set([windows[0].endoffsetms, windows[1].endoffsetms])
                      == set([-120, 0]))
        self.assert_(windows[0].skipfirstms == 15000)
        self.assert_(windows[0].excludedefs == [] or windows[1].excludedefs == [])
        window = windows[0] if windows[0].excludedefs != [] else windows[1]
        self.assert_(len(window.excludedefs) == 6
                     and isinstance(window.excludedefs[0],windower.ExcludeDef))
        self.assert_(window.excludedefs[0].preexcludems == 2000)           
        self.assert_(window.excludedefs[0].postexcludems == 2000)
        
if __name__ == '__main__':  
    
    suite = unittest.TestLoader().loadTestsFromName('test_node_chain_factory')
    
    unittest.TextTestRunner(verbosity=2).run(suite)