""" Contains nodes that classify randomly """

import random

from pySPACE.missions.nodes.base_node import BaseNode
# the output is a prediction vector
from pySPACE.resources.data_types.prediction_vector import PredictionVector

class RandomClassifierNode(BaseNode):
    """ Assign data randomly with probability 0.5 to the classes
    
    **Parameters**
    
    **Exemplary Call**
    
    .. code-block:: yaml
    
        -
            node : Random_Classifier
    
    :Author: Jan Hendrik Metzen (jhm@informatik.uni-bremen.de)
    :Created: 2009/07/03
    :Last change: 2010/08/13 by Mario Krell
    
    """
    
    def __init__(self, *args, **kwargs):
        super(RandomClassifierNode, self).__init__(*args, **kwargs)
        
        self.set_permanent_attributes(labels = [])
        
    def is_trainable(self):
        """ Returns whether this node is trainable. """
        return True
    
    def is_supervised(self):
        """ Returns whether this node requires supervised training """
        return True
    
    def _execute(self, data):
        """ Executes the classifier on the given data vector x"""
        # Classify randomly
        label = random.choice(self.labels)
        return PredictionVector(label=label,
                                prediction=self.labels.index(label),
                                predictor=self)
    
    def _train(self, data, class_label):
        """ Trains the classifier on the given data
        
        It is assumed that the class_label parameter
        contains information about the true class the data belongs to
        """
        # Remember the labels
        if class_label not in self.labels: 
            self.labels.append(class_label)


_NODE_MAPPING = {"Random_Classifier": RandomClassifierNode}

