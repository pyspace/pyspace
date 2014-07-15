""" Randomly select a number of features 

.. todo:: generalize or put together with other node
"""

import os
import cPickle
import random

from pySPACE.missions.nodes.base_node import BaseNode
from pySPACE.resources.data_types.feature_vector import FeatureVector

from pySPACE.tools.filesystem import  create_directory

class RandomFeatureSelectionNode(BaseNode):
    """ Randomly select a given number of features
    
    **Parameters**
    
      :num_retained_features: 
            The number of features that should be retained by the node. This
            information must be specified if selected_features_path is not 
            specified.
            
         (*optional, default: None*)
         
      :selected_features_path:
            An absolute path from which the selected features are loaded. If 
            not specified, these features are learned from the training data.
            In this case, num_retained_features must be specified.

         (*optional, default: None*)
         
    **Exemplary Call**
    
    .. code-block:: yaml
    
        -
            node : RandomFeatureSelection
            parameters :
                  num_retained_features : 1
              
    :Author: Jan Hendrik Metzen (jhm@informatik.uni-bremen.de)
    :Created: 2009/02/03
    """
    def __init__(self, num_retained_features=None, selected_features_path = None, 
                 **kwargs):   
        super(RandomFeatureSelectionNode, self).__init__(**kwargs)
        
        retained_feature_indices = None
        # Load patterns from file if requested
        if selected_features_path != None:
            features_file = open(selected_features_path, 'r')
            retained_feature_indices = cPickle.load(features_file)
            if num_retained_features is not None:
                if len(retained_feature_indices) > num_retained_features:
                    retained_feature_indices = retained_feature_indices[0:num_retained_features]
                elif len(retained_feature_indices) < num_retained_features:
                    import warnings
                    warnings.warn("Only %s features available, cannot retain "
                                  "%s features!" % (len(retained_feature_indices),
                                                    num_retained_features))
            features_file.close()
        
        self.set_permanent_attributes(
              retained_feature_indices = retained_feature_indices,
              num_retained_features = num_retained_features,  
              feature_names = None)

    def _execute(self, feature_vector):
        """ Projects the feature vector onto the retained features """
        if self.retained_feature_indices == None:
            if self.num_retained_features > feature_vector.shape[1]:
                self._log("Too large 'num_retained_features' (%s)!" %
                          self.num_retained_features)
                self.set_permanent_attributes(
                    num_retained_features=feature_vector.shape[1])
            # The indices of the features that will be retained
            self.retained_feature_indices = random.sample(range(feature_vector.shape[1]),
                                                          self.num_retained_features)
            
            self.feature_names = feature_vector.feature_names 

        # Project the features onto the selected subspace
        proj_features = feature_vector[:,self.retained_feature_indices]
        # Update the feature_names list
        feature_names = [feature_vector.feature_names[index] 
                                    for index in self.retained_feature_indices]
        # Create feature vector instance
        projected_feature_vector = FeatureVector(proj_features,
                                                 feature_names) 
    
        return projected_feature_vector
    
    def store_state(self, result_dir, index=None): 
        """ Stores this node in the given directory *result_dir* """
        if self.store:
            node_dir = os.path.join(result_dir, self.__class__.__name__)
            create_directory(node_dir)
            
            # This node only stores the order of the selected features' indices
            name = "%s_sp%s.pickle" % ("selected_features", self.current_split)
            result_file = open(os.path.join(node_dir, name), "wb")
            result_file.write(cPickle.dumps(self.retained_feature_indices, 
                                            protocol=2))
            result_file.close()
            
            # Store feature names
            name = "feature_names_sp%s.txt" % self.current_split
            result_file = open(os.path.join(node_dir, name), "w")
            result_file.write("%s" % self.feature_names)
            result_file.close()

