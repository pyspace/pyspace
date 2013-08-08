""" Reduce filters with the help of name filters """

from pySPACE.missions.nodes.base_node import BaseNode
import logging
from pySPACE.resources.data_types.feature_vector import FeatureVector
import numpy

class FeatureNameFilterNode(BaseNode):
    """ Filter feature vectors by name patterns or indices
    
    .. todo:: Introduce regular expressions
    
    **Parameters**
    
        One of the following three filters should be specified.
        
        :filter_indices:
            If you now the indices of features you want to delete, use this
            parameter. Numbering begins with zero.
            
            (*optional, default: []*)
        
        :exclude_patterns: List of exclude patterns
            
            Each entry is checked, if it is included in one of the feature names.
            If this is the case it is deleted. In the other case it is kept.
            
            A special functionality comes by using *All* instead of a list.
            This means to exclude everything (except the include patterns).
        
            (*optional, default: []*)
        
        :exclude_names:
            Feature names to be excluded
            
            (*optional, default: []*)
        
        :include_patterns: Skip excluded feature name fulfilling the one of these rules
        
            (*optional, default: []*)
        
        :include_names:
            Feature names to be included (more priority than exclusion)
            
            (*optional, default: []*)
    
    **Priority Ranking** 

        1. include_names
        2. exclude_names
        3. include_patterns
        4. exclude_patterns
        5. filter_indices
    
    **Exemplary Call**

    .. code-block:: yaml

        -
            node : FeatureNameFilterNode
            parameters :
                  filter_indices : [-1]
                  exclude: [EMG, EOG]
                  include: [EMGCLassifier, Pulse]
    
    :input:    FeatureVector
    :output:   FeatureVector
    :Author: Mario Krell (mario.krell@dfki.de)
    :Date: 2012/08/24
    """
    def __init__(self,exclude_names=[],include_names=[],
                 exclude_patterns=[],include_patterns=[],
                 filter_indices=[],**kwargs):
        super(FeatureNameFilterNode, self).__init__(**kwargs)
        if not (len(exclude_names)>0 or len(filter_indices)>0 or len(exclude_patterns)>0):
            self._log("No filter specified.", level=logging.CRITICAL)
        if not (type(exclude_patterns) == list or exclude_patterns=="All"):
            self._log("Wrong format for exclude_patterns list (%s). Parameter ignored."%str(exclude_patterns),level=logging.CRITICAL)
            exclude_patterns=[]
        if not type(include_patterns)==list:
            self._log("Wrong format for include_patterns list (%s). Parameter ignored."%str(include_patterns),level=logging.CRITICAL)
            include_patterns=[]
        if not type(exclude_names) == list:
            self._log("Wrong format for exclude_names list (%s). Parameter ignored."%str(exclude_names),level=logging.CRITICAL)
            exclude_names=[]
        if not type(include_names)==list:
            self._log("Wrong format for include_names list (%s). Parameter ignored."%str(include_names),level=logging.CRITICAL)
            include_names=[]
        if not type(filter_indices) == list:
            self._log("Wrong format for filter_indices list (%s). Parameter ignored."%str(filter_indices),level=logging.CRITICAL)
            filter_indices=[]
#
#        :retained_feature_indices:
#            indices of the relevant features
#
#        :feature_names:
#            list of finally accepted feature names
        
        self.set_permanent_attributes(exclude_patterns=exclude_patterns,
                                      include_patterns=include_patterns,
                                      exclude_names=exclude_names,
                                      include_names=include_names,
                                      filter_indices=filter_indices,
                                      retained_indices=None,
                                      feature_names=None,)
    def _execute(self,data):
        """ Construct filter at first call and apply it on every vector """
        if self.retained_indices is None:
            self.build_feature_selector(data=data)
        if self.feature_names is None:
            self.feature_names=[data.feature_names[i] for i in self.retained_indices]
        data=data.view(numpy.ndarray)
        return FeatureVector(data[:,self.retained_indices],self.feature_names)

    def build_feature_selector(self,data):
        """ Define the *retained_channel_indices* for final projection """
        data_feature_names=data.feature_names
        delete_feature_names=set()
        # handle filter indices
        for index in self.filter_indices:
            delete_feature_names.add(data_feature_names[index])
        if self.exclude_patterns == "All":
            delete_feature_names=set(data.feature_names)
        # handle exclude patterns
        for pattern in self.exclude_patterns:
            for feature_name in data_feature_names:
                if pattern in feature_name:
                     delete_feature_names.add(feature_name)
        # handle include patterns
        for pattern in self.include_patterns:
            for feature_name in data_feature_names:
                if pattern in feature_name:
                     delete_feature_names.discard(feature_name)
        # handle exclude names
        for feature_name in self.exclude_names:
            delete_feature_names.add(feature_name)
        # handle include names
        for feature_name in self.include_names:
            delete_feature_names.discard(feature_name)
            if not feature_name in data_feature_names:
                self._log("Could not find feature: %s."%feature_name, level=logging.CRITICAL)
        #construct relevant_parameters
        self.retained_indices=[]
        self.feature_names=[]
        for index, feature_name in enumerate(data_feature_names):
            if not feature_name in delete_feature_names:
                self.retained_indices.append(index)
                self.feature_names.append(feature_name)


