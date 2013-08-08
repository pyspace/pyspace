""" Feature selection based on the RELIEF algorithm """

import os
import random
import warnings
import cPickle
from collections import defaultdict
import numpy
import heapq

from pySPACE.missions.nodes.base_node import BaseNode
from pySPACE.resources.data_types.feature_vector import FeatureVector

from pySPACE.tools.filesystem import  create_directory

class ReliefFeatureSelectionNode(BaseNode):
    """ Feature selection based on the RELIEF algorithm
    
    Feature selection based on the RELIEF algorithm. A feature is preferred
    if instances of the same class (hits) are comparatively close to each other
    compared to instances of the other class (misses) in the feature dimension.
    Please refer to "Estimating Attributes: Analysis and Extensions of RELIEF" 
    by Kononenko for more information.
    
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
      :k: 
            The number of nearest neighbors that are considered when computing
            the closest hits and misses. Defaults to 1.
            
         (*optional, default: 1*)
              
    **Exemplary Call**
    
    .. code-block:: yaml
    
        -
            node : ReliefFeatureSelection
            parameters :
                  num_retained_features : 100
                  k : 10
              
    :Author: Jan Hendrik Metzen (jhm@informatik.uni-bremen.de)
    :Created: 2010/07/12
    
    """
    def __init__(self, num_retained_features=None, selected_features_path=None, 
                 k=1, **kwargs): 
        # Must be set before constructor of superclass is set
        self.trainable = (selected_features_path == None)    
        super(ReliefFeatureSelectionNode, self).__init__(**kwargs)
        
        retained_feature_indices = None
        # Load patterns from file if requested
        if selected_features_path != None:
            features_file = open(selected_features_path, 'r')
            retained_feature_indices = cPickle.load(features_file)
            if num_retained_features is not None:
                if len(retained_feature_indices) > num_retained_features:
                    retained_feature_indices = retained_feature_indices[0:num_retained_features]
                elif len(retained_feature_indices) < num_retained_features:
                    warnings.warn("Only %s features available, cannot retain "
                                  "%s features!" % (len(retained_feature_indices),
                                                    num_retained_features))
            features_file.close()
        
        self.set_permanent_attributes(
                          retained_feature_indices = retained_feature_indices,
                          num_retained_features = num_retained_features,
                          k = k,
                          class_data = defaultdict(list),
                          feature_names = None)
                
    def is_trainable(self):
        """ Returns whether this node is trainable """
        return self.trainable
    
    def is_supervised(self):
        """ Returns whether this node requires supervised training """
        return True

    def _train(self, data, label):
        """ Add given data point along with its label to the training set. """
        assert isinstance(data, FeatureVector), "Relief feature selection " \
                  "requires that its input node outputs feature vectors!"
                
        # Gather feature vectors and labels in two lists
        self.class_data[label].append(data[0])
        
        if self.feature_names == None:
            self.feature_names = data.feature_names
               
    
    def _stop_training(self, debug=False):
        """ Called automatically at the end of training
        
        Computes a ranking of features and stores
        a list of the indices of those feature that should be retained        
        """ 
        assert (len(self.class_data.keys()) == 2),\
                     "Relief supports only 2-class problems!"
                     
        # Generate data structures for searching class conditioned approximate 
        # nearest neighbors
        for class_label, instances in self.class_data.iteritems():
            # K needs to be decreased if not enough data is available
            self.k = min(self.k, len(instances)-1)
            
        # Mapping from class label to the other class' label
        other_class = {self.class_data.keys()[0]: self.class_data.keys()[1],
                       self.class_data.keys()[1]: self.class_data.keys()[0]}

        # Compute the features' weights (its quality).
        # The quality of a feature is the better the smaller the 
        # instance's distance in this feature dimension is to the
        # closest hit and the larger the distance to the closest miss
        weights = numpy.zeros(self.class_data.values()[0][0].shape)
        instance_counter = 0
        for class_label, instances in self.class_data.iteritems():
            instance_counter += len(instances)
            # For all instances of the given class
            for instance in instances:
                other_class_label = other_class[class_label]
                # Compute k - nearest neighbors within the same class  
                for hit in self._search_k_nearest_neighbors(instance, class_label,
                                                            self.k + 1)[1:]:
                    # Subtract distance from instance to hit from the weights
                    # (normalized by the number of neighbors)
                    weights -= numpy.absolute(instance - hit)/self.k
                # Compute k - nearest neighbors in the other class
                for miss in self._search_k_nearest_neighbors(instance, 
                                                             other_class_label,
                                                             self.k):  
                    # Add distance from instance to miss to the weights
                    # (normalized by the number of neighbors)
                    weights += numpy.absolute(instance - miss)/self.k                    
        
        weights = weights / instance_counter   
        self.retained_feature_indices = \
                        numpy.argsort(-weights)[:self.num_retained_features]
        
        self.class_data = defaultdict(list) # Get rid of stored training data
                
    def _execute(self, feature_vector):
        """ Projects the feature vector onto the retained features """
        # Project the features onto the selected subspace
        proj_features = feature_vector[:,self.retained_feature_indices]
        # Update the feature_names list
        feature_names = [feature_vector.feature_names[index] 
                                    for index in self.retained_feature_indices]
        # Create feature vector instance
        projected_feature_vector = FeatureVector(proj_features,
                                                 feature_names) 
    
        return projected_feature_vector
    
    def _search_k_nearest_neighbors(self, instance, class_label, k):
        # Search the k training examples of class class_label that are most
        # similar to instance
        heap = []
        for example in self.class_data[class_label]:
            # Sort based on distance of example to instance (1-norm)
            distance = numpy.linalg.norm(instance - example, ord=2)
            heapq.heappush(heap, (distance, random.random(), example)) # break ties randomly     
        assert len(heap) >= k
    
        return [heapq.heappop(heap)[2] for i in range(k)]
        
    
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

