""" Operate on classifiers """

import copy
import numpy

from pySPACE.missions.nodes.base_node import BaseNode
# the output is a set of predictions
from pySPACE.resources.data_types.prediction_vector import PredictionVector

class SplitClassifierLayerNode(BaseNode):
    """ Split the overrepresented class in the training set for multiple training.
    
    The node trains several classifiers with this splits such that both classes
    are nearly equally distributed.
    
    Output is a set of predictions in a prediction vector.
    An ensemble node should follow to combine these classifications.
    
    **Parameter**
    
    :classifier:
        Classifier to be split.
        Notation is as usual as in the YAML file.
        
        Maybe this will be changed later on.
    
    **Exemplary Call**
    
    .. code-block:: yaml
    
        - 
            node : Split_Classifier
            parameters :
                classifier: 
                        -
                            node : LibSVM_Classifier
                            parameters :
                                complexity : 1
                                weight : [1,3]
                                debug : False
                                store : False
                                class_labels : ['Standard', 'Target']
    
    :Author: Mari Krell (mario.krell@dfki.de)
    
    """
    input_types=["FeatureVector"]
    def __init__(self, classifier, store = False, *args, **kwargs):

        super(SplitClassifierLayerNode, self).__init__(store, *args, **kwargs)
        
        self.caching = False #Why?
        self.classifier = classifier[0]
        # self.weight=self.classifier['parameters']['weight'] #addon
        # self.classifier['parameters']['weight']=[1,1] #addon
        #############################################
        self.permanent_state = copy.deepcopy(self.__dict__)
        self.set_permanent_attributes(samples=None, labels=None, classes=[])
    
    def is_trainable(self):
        """ Returns whether this node is trainable. """
        return True
    
    def is_supervised(self):
        """ Returns whether this node requires supervised training """
        return True

    def reset(self):
        """ Reset the state to the clean state it had after its initialization """
        nodes = self.nodes
        for node in nodes:
            node.reset()
        self.samples=None
        self.labels=None
        self.nodes = []
        # resetting of Meta node is important for different splits
        super(SplitClassifierLayerNode, self).reset()
    
    def _execute(self, data):
        """Process the data through the internal nodes."""
        feature_names = []
        result_array = None
        label = []
        prediction = []
        predictor  = []
        for node_index, node in enumerate(self.nodes):
            node_result = node.execute(data)
            label.append(node_result.label)
            prediction.append(node_result.prediction)
            predictor.append(node_result.predictor)
        return PredictionVector(label = label, prediction = prediction, predictor = predictor)

    def _train(self, data, class_label):
        """ collects the data for training
        
        It is assumed that the class_label parameter
        contains information about the true class the data belongs to
        """
        self._train_phase_started = True
        if self.samples == None:
            self.samples = []
        if self.labels == None:
            self.labels = []
            self.num_retained_features = len(data[0,:])
            
        if class_label not in self.classes:
            self.classes.append(class_label)
        # Collect the data
        self.samples.append(data)
        self.labels.append(class_label)

    def _stop_training(self, debug=False):
        n0=self.labels.count(self.classes[0])
        n1=self.labels.count(self.classes[1])
        if n0>n1:
            # n[0] is divided in packages of size n[1]
            num = n0 / n1
            self.nodes=[]
            # initialization of the necessary classifier nodes
            for j in range(num):
                self.nodes.append(BaseNode.node_from_yaml(self.classifier))
            # self.classifier[0]['parameters']['weight']=self.weight#addon
            # self.nodes.append(BaseNode.node_from_yaml(self.classifier[0]))#addon
            k = 0
            for i in range(len(self.samples)):
                if self.labels[i] == self.classes[1]:
                    # underrepresented class is sent to all classifiers
                    for classifier in self.nodes:
                        classifier.train(self.samples[i],self.labels[i])
                else:
                    # feed into k-th classifier
                    self.nodes[k].train(self.samples[i],self.labels[i])
                    k = (k+1)%num
                    # self.nodes[num].train(self.samples[i],self.labels[i])#addon
        else:
            # n[1] is divided in packages of size n[0]
            num = n1 / n0
            self.nodes=[]
            # initialization of the necessary classifier nodes
            for j in range(num):
                self.nodes.append(BaseNode.node_from_yaml(self.classifier))
            k = 0
            for i in range(len(self.samples)):
                if self.labels[i] == self.classes[0]:
                    # underrepresented class is sent to all classifiers
                    for classifier in self.nodes:
                        classifier.train(self.samples[i],self.labels[i])
                else:
                    # feed into k-th classifier
                    self.nodes[k].train(self.samples[i],self.labels[i])
                    k = (k+1)%num
        for classifier in self.nodes:
            classifier.stop_training(debug)
        self.num_retained_features = "differs maybe" # self.nodes[0].num_retained_features # This should be calculated more exactly.
        self.complexity =  "differs" #self.nodes[0].complexity

    def get_output_type(self, input_type, as_string=True):
        """ overwritten method from BaseNode

        returns PredictionVector(as string or class) since this
        is the only possible output of the current node
        """
        if as_string:
            return "PredictionVector"
        else:
            return PredictionVector


class SVMComplexityLayerNode(SplitClassifierLayerNode):
    """ Calculate the minimal complexity, where the soft margin is inactive
    
    This node uses nested intervals and a tolerance variable is used to define
    when the accuracy is high enough and the slack variables are small enough.
    This was necessary because the libsvm classifier gives no exact solution
    and the slack variables may be never zero.
    
    Output is the prediction of the given classifier with the given complexity
    multiplied by the found complexity.
    Wrapper around a classifier.
    The result should be analyzed with the classification performance sink node.
    
    **Parameter**
    
    :classifier:
        SVM Classifier to be analysed.
        Notation is as usual as in the YAML file.
        
        Maybe this will be changed later on.
    
    **Exemplary Call**
    
    .. code-block:: yaml
    
        - 
            node : Get_Complexity
            parameters :
                classifier: 
                        -
                            node : LibSVM_Classifier
                            parameters :
                                complexity : 1
                                weight : [1,3]
                                debug : False
                                store : False
                                class_labels : ['Standard', 'Target']
    
    :Author: Mario Krell (mario.krell@dfki.de)
    
    """
    def __init__(self, classifier, store = False,eps=0.001,*args, **kwargs):
        self.trainable=True
        super(SVMComplexityLayerNode, self).__init__(classifier,store, *args, **kwargs)
        self.set_permanent_attributes(complexity=1,eps=eps,old_C=self.classifier["parameters"]["complexity"],nodes=None)
        
    def reset(self):
        """ Reset the state to the clean state it had after its initialization """
        nodes = self.nodes
        for node in nodes:
            node.reset()
        self.nodes = nodes
        # resetting of Meta node is important for different splits
        super(SVMComplexityLayerNode, self).reset()
        
    def _execute(self, data):
        """Process the data through the internal nodes."""
        result = self.nodes[0].execute(data)
        return result
                              
    def _train(self, data, class_label):
        """ It is assumed that the class_label parameter
        contains information about the true class the data belongs to
        """
        self._train_phase_started = True
        # init of node
        if self.nodes==None:
            self.nodes=[BaseNode.node_from_yaml(self.classifier)]
            self.nodes[0].complexity=self.complexity
        self.nodes[0].train(data,class_label)
    
    def _stop_training(self, debug=False):
        # init
        self.nodes[0].stop_training(debug)
        if ((numpy.array(self.nodes[0].t) <= self.eps).all()):
            Cmax=self.complexity
            while ((numpy.array(self.nodes[0].t) <= self.eps).all()):
                Cmax = Cmax / 10.0
                self.nodes[0].complexity = Cmax
                self.nodes[0]._stop_training(debug)
            Cmin = Cmax
            Cmax = Cmax * 10.0
        else:
            Cmin=self.complexity
            while not((numpy.array(self.nodes[0].t) <= self.eps).all()):
                Cmin = Cmin * 10.0
                self.nodes[0].complexity = Cmin * 10.0
                self.nodes[0]._stop_training(debug)
            Cmax = Cmin
            Cmin = Cmax / 10.0
            
        # Nested intervals principle
        while (Cmax-Cmin)>self.eps:
            self.nodes[0].complexity = 0.5 * (Cmax + Cmin)
            self.nodes[0]._stop_training(debug)
            if ((numpy.array(self.nodes[0].t) <= self.eps).all()):
                Cmax = 0.5 * (Cmax + Cmin)
            else:
                Cmin = 0.5 * (Cmax + Cmin)
        self.complexity = Cmax
        self.max_C = Cmax
        self.nodes[0].complexity = self.old_C * self.complexity
        if not self.old_C == 1:
            self.nodes[0]._stop_training(debug)
        self.num_retained_features = self.nodes[0].num_retained_features
        self.nodes[0].classifier_information["__Num_Retained_Features__"] = \
                                                    self.num_retained_features
        self.nodes[0].classifier_information["__Max_Complexity__"] = Cmax


_NODE_MAPPING = {"Split_Classifier": SplitClassifierLayerNode,
                "Get_Complexity":SVMComplexityLayerNode}
