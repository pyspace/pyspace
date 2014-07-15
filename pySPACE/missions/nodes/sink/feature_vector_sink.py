""" Collect feature vectors """

import copy

from pySPACE.missions.nodes.base_node import BaseNode
from pySPACE.resources.dataset_defs.feature_vector import FeatureVectorDataset


class FeatureVectorSinkNode(BaseNode):
    """ Collect all :class:`~pySPACE.resources.data_types.feature_vector.FeatureVector` elements
    that are passed through it in a collection of type :mod:`~pySPACE.resources.dataset_defs.feature_vector`.
    
    **Parameters**

    **Exemplary Call**

    .. code-block:: yaml

        - 
            node: FeatureVectorSink

    :input:  FeatureVector
    :output: FeatureVectorDataset
    :Author: Jan Hendrik Metzen (jhm@informatik.uni-bremen.de)
    :Created: 2008/09/02
    
    """
    input_types = ["FeatureVector"]

    def __init__(self, classes_names=[], num_features=None, **kwargs):
        super(FeatureVectorSinkNode, self).__init__(**kwargs)
        
        self.set_permanent_attributes(
            classes_names=classes_names,
            num_features=num_features,
            feature_vector_collection=None,  # This will be created lazily
            )

    def reset(self):
        """
        Reset the state of the object to the clean state it had after its
        initialization
        """
        # We have to create a temporary reference since we remove 
        # the self.permanent_state reference in the next step by overwriting
        # self.__dict__
        tmp = self.permanent_state
        # TODO: just a hack to get it working quickly...
        tmp["feature_vector_collection"] = self.feature_vector_collection 
        self.__dict__ = copy.copy(tmp)
        self.permanent_state = tmp   
    
       
    def is_trainable(self):
        """ Returns whether this node is trainable. """
        # Though this node is not really trainable, it returns true in order
        # to get trained. The reason is that during this training phase, 
        # it stores all training samples into an ARFF file,
        #  which can then later be used in Weka
        return True
    
    def is_supervised(self):
        """ Returns whether this node requires supervised training """
        return True

    def _train(self, data, label):
        # We do nothing
        pass
    
    def _create_result_sets(self, num_features, feature_names = None):
        """ 
        Sets some object members that could not set during
        __init__ since the depend on the dimensionality of the
        data (i.e. the number of features)
        """
        # Create the labeled samples sets lazily
        self.num_features = num_features
    
        if feature_names == None:
            feature_names = [("f%s" % i) for i in range(self.num_features)]
        
        self.feature_vector_collection = \
                FeatureVectorDataset(classes_names=self.classes_names,
                                        feature_names=feature_names,
                                        num_features =self.num_features)

    def process_current_split(self):
        """ 
        Compute the results of this sink node for the current split of the data
        into train and test data
        """     
        # Compute the feature vectors for the data used for training
        for feature_vector, label in self.input_node.request_data_for_training(False):
            # Do lazy initialization of the class 
            if self.feature_vector_collection == None:
                feature_names = feature_vector.feature_names \
                                    if hasattr(feature_vector, 
                                               "feature_names") else None
                self._create_result_sets(feature_vector.size,
                                         feature_names)
            
            # Add sample
            self.feature_vector_collection.add_sample(feature_vector,
                                                      label = label,
                                                      train = True,
                                                      split = self.current_split,
                                                      run = self.run_number)
            
        # Compute the feature vectors for the data used for testing
        for feature_vector, label in self.input_node.request_data_for_testing():
            # Do lazy initialization of the class 
            # (maybe there were no training examples)
            if self.feature_vector_collection == None:
                feature_names = feature_vector.feature_names \
                                    if hasattr(feature_vector, 
                                               "feature_names") else None 
                self._create_result_sets(feature_vector.size,
                                         feature_names)
            # Add sample
            self.feature_vector_collection.add_sample(feature_vector,
                                                      label,
                                                      train = False,
                                                      split = self.current_split,
                                                      run = self.run_number)
    
    def get_result_dataset(self):
        """ Return the result """
        return self.feature_vector_collection


_NODE_MAPPING = {"Labeled_Feature_Vector_Sink": FeatureVectorSinkNode,
               "Feature_Vector_Sink": FeatureVectorSinkNode}

