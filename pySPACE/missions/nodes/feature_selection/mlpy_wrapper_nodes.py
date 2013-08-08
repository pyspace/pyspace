""" Use feature selection methods implemented in MLPY """

import os
import cPickle

import numpy

from pySPACE.missions.nodes.base_node import BaseNode
from pySPACE.resources.data_types.feature_vector import FeatureVector

from pySPACE.tools.filesystem import  create_directory

class MLPYFeatureSelectionWrapper(BaseNode):
    """ Wrapper node that allows to use MLPY algorithms
    
    This module contains a node that wraps the feature selection
    algorithms contained in the MLPY package (https://mlpy.fbk.eu/)
    
    MLPY provides several feature weighting methods that are wrappers around
    classifier, e.g.:
    
      * Support Vector Machines (SVMs)
      * Spectral Regression Discriminant Analysis (SRDA)
      * Fisher Discriminant Analysis (FDA)
      * Penalized Discriminant Analysis (PDA)
      * Diagonal Linear Discriminant Analysis (DLDA)
    
    Furthermore, MLPY has also some filter-like feature weighting methods:
      
      * Iterative Relief (I-RELIEF)
      * Discrete Wavelet Transform (DWT)
    
    Based on these feature weighting algorithms, the ranking method contained
    in MLPY can be used to create a ranking of the utility of different features.
    The following ranking algorithms can be used by MLPY:
      
      * Recursive Feature Elimination (rfe, onerfe, erfe, bisrfe, sqrtrfe)
      * Recursive Forward Selection (rfs) 
    
    The wrapper node expects as argument
      
      * how many of the features it should retain
      * which feature weighting method it should use
      * which feature ranking method it should use
    
    **Parameters**
    
      :num_retained_features: 
            The number of features that should be retained by the node. This
            information must be specified if selected_features_path is not 
            specified.
            
            (*optional, default: None*)
         
      :ranking_method: 
          A string that specifies the feature elimination method used 
          internally by MLPY's feature ranking method. Can be 'rfe', 'onerfe',
          'erfe', 'bisrfe', 'sqrtrfe', or 'rfs'
          
          (*optional, default: None*)
          
      :weighting_method: 
          An object instance that provides an weights(x,y) method that 
          takes as input:
          
             * x : training data (#sample x #feature) 2D numpy array float
             * y : classes       (two classes, -1 and 1)
             
          and returns a float in [0,1]. This float is the weight of the 
          respective feature. A set of classes that can be used are contained in 
          MLPY, for instance mlpy.svm.
          
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
                  num_retained_features : 100
                  ranking_method : "bisrfe"
                  weighting_method: "eval(__import__('mlpy').svm)"

    :Author: Jan Hendrik Metzen (jhm@informatik.uni-bremen.de)
    :Created: 2009/02/03

    """
    def __init__(self, num_retained_features=None, ranking_method=None,
                 weighting_method=None, selected_features_path = None, 
                 **kwargs):
        # Must be set before constructor of superclass is set
        self.trainable = (selected_features_path == None)      
    
        super(MLPYFeatureSelectionWrapper, self).__init__(**kwargs)
        
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
              ranking_method = ranking_method,
              weighting_method = weighting_method,
              num_retained_features = num_retained_features,  
              training_data = [],
              training_labels = [],
              feature_names = None)
                
    def is_trainable(self):
        """ Returns whether this node is trainable """
        return self.trainable
    
    def is_supervised(self):
        """ Returns whether this node requires supervised training """
        return True

    def _train(self, data, label):
        """
        Add the given data point along with its class label 
        to the training set
        """
        assert(isinstance(data, FeatureVector))
        
        # Gather feature vectors and labels in two lists
        self.training_data.append(data[0])
        self.training_labels.append(label)
        
        if self.feature_names == None:
            self.feature_names = data.feature_names
               
    
    def _stop_training(self, debug=False):
        """ Called automatically at the end of training
        
        Computes a ranking of features and stores
        a list of the indices of those feature that should be retained        
        """
        # Lazy import of mlpy (not available on all platforms/machines ?
        import mlpy
        
        # Check that we have exactly two different classes
        occuring_labels = set(self.training_labels)
        assert(len(occuring_labels) == 2)
        # Randomly map the class labels to -1 and 1
        # MLPY cannot deal with other kind of labels
        label_enumeration = dict(zip(occuring_labels, [-1,1]))
        
        # Create two dimensional array containing the 
        # respective feature vectors 
        data = numpy.array(self.training_data)
        # Create an one dimensional array containing the class labels
        labels = numpy.array([label_enumeration[label]
                                    for label in self.training_labels])
        
        # Instantiate feature weighter and ranker  
        feature_weighter = self.weighting_method()
        feature_ranking = mlpy.ranking(method=self.ranking_method)
        
        # Compute the ranks of features
        # For instance, rank = [2,3,1,0] would mean that the feature with
        # index 2 is the most distinctive while the feature with index 0 is
        # the worst.
        ranks = feature_ranking.compute(data, labels, feature_weighter)
        
        # The indices of the features that sill be retained
        self.retained_feature_indices = ranks[0:self.num_retained_features] 
        
    
    def _execute(self, feature_vector):
        """ 
        Projects the feature vector *features* onto the subspace of features
        that should be retained        
        """
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
            # TODO: Use HDF format?
            result_file.write(cPickle.dumps(self.retained_feature_indices, 
                                            protocol=2))
            result_file.close()
            
            # Store feature names
            name = "feature_names_sp%s.txt" % self.current_split
            result_file = open(os.path.join(node_dir, name), "w")
            result_file.write("%s" %  self.feature_names)
            result_file.close()

_NODE_MAPPING = {"MLPY_FS_Wrapper": MLPYFeatureSelectionWrapper}

# script to test and compare the performance speed of different feature selection methods
if __name__ == "__main__":
    import timeit

    s = """
    import numpy
    import random
    import mlpy

    num_features = %s
    num_observations = %s

    x = numpy.random.normal(size = (num_observations, num_features))
    y = numpy.array([random.choice([-1,1]) for i in range(num_observations)])     # class labels

    myrank = mlpy.Ranking(method=%s)     # initialize ranking class

    # Feature Ranking
    w = mlpy.%s()                   # initialize irelief class

    myrank.compute(x, y, w)     # compute feature ranking
    """
    num_features = 100
    num_observations = 100
    elim_method = "rfe"
    weight_method = "pda"
    for weight_method in ["Svm", "Srda", "Fda", "Pda", "Dwt", "Dlda", "Irelief"]:
    #for elim_method in ["onerfe", "rfe", "bisrfe", "sqrtrfe", "erfe", "rfs"]:
    #for num_features in [10, 30,50,70,100, 1000]:
    #for num_observations in [10, 30,50,70,100, 1000]:
        s_inst = s % (num_features, num_observations, elim_method, weight_method)
        print "%s -> %s " % ((num_features, num_observations, elim_method, weight_method),
            timeit.Timer(s_inst).timeit(25))
