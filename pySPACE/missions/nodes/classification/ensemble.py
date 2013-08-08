""" Ensemble classifiers

http://en.wikipedia.org/wiki/Ensemble_learning

Current implementations use gating for training ensemble methods.

Each gating function expects as input a special kind of 
:class:`~pySPACE.resources.data_types.prediction_vector.PredictionVector`: Each
component in the vector should correspond to the classification of one node chain
of the ensembles (i.e. the dimensionality should be equal to the cardinality
of the ensemble and each value of the vector should be one of the prediction
scores and you should get a list of labels).

This can be created using the
:class:`~pySPACE.missions.nodes.meta.same_input_layer.ClassificationFlowsLoader`
or the
:class:`~pySPACE.missions.nodes.meta.same_input_layer.SameInputLayerNode`.
"""

from collections import defaultdict
import heapq
import numpy

from pySPACE.missions.nodes.base_node import BaseNode
# the output is of Gating Functions is a prediction vector
from pySPACE.resources.data_types.prediction_vector import PredictionVector
import logging


class ProbVotingGatingNode(BaseNode):
    """ Add up prediction values for labels to find out most probable label 
    
    **Parameters**

        :enforce_absolute_values:
            Switch to map the prediction values to their absolute value.
        
            (*optional, default:False*)

    **Exemplary Call**
    
    .. code-block:: yaml
    
        -
            node : ProbVotingGating

    :Author: Mario M. Krell (mario.krell@dfki.de)
    :Created: 2012/10/01
    """
    def __init__(self, enforce_absolute_values=False, **kwargs):
        super(ProbVotingGatingNode, self).__init__(**kwargs)
        
        self.set_permanent_attributes(enforce_absolute_values=enforce_absolute_values)
    
    def _execute(self,data):
        """ Label with highest sum of prediction values wins """
        pred = defaultdict(float)

        for i,label in enumerate(data.label):
            if self.enforce_absolute_values:
                pred[label] += abs(data.prediction[i])
            else:
                pred[label] += data.prediction[i]
        res = sorted(pred.items(), key=lambda t: t[1])
        best = res[-1]
        return PredictionVector(prediction=best[1],
                                label=best[0], predictor=self)


class LabelVotingGatingNode(ProbVotingGatingNode):
    """ Gating function to classify based on the majority vote
    
    This gating function counts how often each class occurs in the feature
    vectors. It assigns the instance to the class that got the most votes.
    It does not require training.
    If there is no clear vote, the base class is used.

    **Parameters**

    see: base node documentation

    **Exemplary Call**
    
    .. code-block:: yaml
    
        -
            node : LabelVotingGating

    :Author: Mario M. Krell (mario.krell@dfki.de)
    :Created: 2012/10/01
    """
#    def __init__(self, **kwargs):
#        super(VotingGatingNode, self).__init__(**kwargs)

    def _execute(self, data):
        """ Executes the classifier on the given data vector *data* """
        prediction_value = numpy.mean([prediction for prediction in data.prediction])
        
        votes_counter = defaultdict(int)
        for label in data.label:
            votes_counter[label] += 1
        voting = sorted((votes, label) for label, votes in votes_counter.iteritems())
        max_label = [label for votes, label in voting if votes == voting[-1][0]]
        if len(max_label) == 1:
            majority_vote = voting[-1][1]
            return PredictionVector(prediction=prediction_value,
                                    label=majority_vote, predictor=self)
        else:
            relevant_indices = [index for index,label in enumerate(data.label) if label in max_label]
            new_data = PredictionVector(prediction = [data.prediction[i] for i in relevant_indices], 
                                label = [data.label[i] for i in relevant_indices], 
                                predictor =[data.predictor[i] for i in relevant_indices])
            return super(LabelVotingGatingNode, self)._execute(new_data)

class PrecisionWeightedGatingNode(BaseNode):
    """ Gating function to classify based on weighted majority vote
    
    This gating function computes weights for the ensemble's classification results based on
    training data. These weights are set based on the relative
    precision (compared to the other classification results) on the predicted class.
    If more than *required_vote_ratio* of the sum of weighted votes are for class
    1, than this node classifies as class 1 from *class_labels*, else as
    class 2 from *class_labels*.

    **Parameters**
    
        :class_labels:
            Determines the order of the two classes.
            This is important, when you want that the prediction
            value is negative for the first class and
            positive for the other one.
            Here it is used to define the relevant class
            for the voting.
    
        :required_vote_ratio:
            Determines the value the weighted sum of votes has to exceed
            to classify for the first class.
            The acceptable range is from zero to one,
            where zero means, classification is always class one
            and one means, classification is class two if and only if
            all the votes are for class one. 
    
            (*optional, default: 0.5*)
    
    
    **Exemplary Call**
    
    .. code-block:: yaml
    
        -
            node : Precision_Weighted_Gating_Function
            parameters :
                class_labels : ["Target","Standard"]
                required_vote_ratio : 0.25

    :Author: Jan Hendrik Metzen (jhm@informatik.uni-bremen.de)
    :Created: 2010/05/21
    """
    def __init__(self, class_labels, required_vote_ratio=0.5, **kwargs):
        super(PrecisionWeightedGatingNode, self).__init__(**kwargs)
        
        assert len(class_labels) == 2, \
               "%s can only be used for binary classification tasks!" % self.__class__.__name__
        
        self.set_permanent_attributes(class_labels = class_labels,
                                      required_vote_ratio = required_vote_ratio,
                                      classification_counter = None,
                                      correct_classification_counter = None,
                                      weights = None)
   
    def is_trainable(self):
        """ Returns whether this node is trainable. """
        return True
    
    def is_supervised(self):
        """ Returns whether this node requires supervised training """
        return True

    def _train(self, data, class_label):
        if self.classification_counter == None:
            self.classification_counter = [defaultdict(int) for i in range(len(data.label))]
            self.correct_classification_counter = [defaultdict(int) for i in range(len(data.label))]
            
        
        for index1 in range(len(data.label)):
            # Count how often each classifier (index1) classifies instances to the
            # two classes 
            self.classification_counter[index1][data.label[index1].strip()] +=1
            if data.label[index1].strip() == class_label:
                # Count the how often each classifier (index1) classifies instances
                # of the two classes correctly
                self.correct_classification_counter[index1][data.label[index1].strip()] +=1
        
    def _stop_training(self, debug=False):
        # Initialization
        precisions = [dict() for i in range(len(self.correct_classification_counter))]
        acc_precision = defaultdict(float)
        self.weights = [dict() for i in range(len(self.correct_classification_counter))]
        
        for class_label in self.class_labels:
            for i in range(len(self.correct_classification_counter)):
                # Compute the precision of classifier "i" on class "class_label"
                if self.classification_counter[i][class_label] > 0:
                    precision = float(self.correct_classification_counter[i][class_label])\
                                / self.classification_counter[i][class_label]
                else:
                    precision = 0
                precisions[i][class_label] = precision
                # Compute the accumulated precision
                acc_precision[class_label] += precision
                
            # Set weights to their relative contribution to the
            # accumulated precision.
            # Note: This is not a very well-founded way of computing weights
            for i in range(len(self.correct_classification_counter)):
                if not acc_precision[class_label] == 0:
                    self.weights[i][class_label] =  precisions[i][class_label] / acc_precision[class_label]
                else:
                    self._log("ZeroDevision problem occurred. Check Classifiers for class %s."%class_label, level = logging.CRITICAL)
                    self.weights[i][class_label] =  1

        super(PrecisionWeightedGatingNode, self)._stop_training()
    
    def _execute(self, data):
        """ Executes the classifier on the given data vector *data* """
        # Count weighted votes for the two classes       
        votes_counter = defaultdict(int)
        for index, prediction in enumerate(data.label):
            votes_counter[prediction] += self.weights[index][prediction.strip()]
        
        # Compute ratio of votes that voted for class 1
        vote_ratio = \
            float(votes_counter[self.class_labels[0]]) / sum(votes_counter.values())
        
        # If this ratio is above the threshold "self.required_vote_ratio",
        # classify instance as class 1 else as class 2
        vote = self.class_labels[0] if vote_ratio >= self.required_vote_ratio \
                        else self.class_labels[1]
        return PredictionVector(prediction = vote_ratio, label = vote, predictor = self)
    
class ChampionGatingNode(BaseNode):
    """ Gating function to classify with the classifier that performs best on training data
    
    This gating function evaluates the ensemble classifiers on the training data.
    It picks the classifier that maximizes the F-Measure on the *relevant_class*
    and uses this one to classify instances from the test data.

    **Parameters**
    
        :relevant_class:
            Determines the class being relevant for the F-measure calculation.
    
            (*optional, default: first occurring class in training phase*)
    
    
    **Exemplary Call**
    
    .. code-block:: yaml
    
        -
            node : Champion_Gating_Function
            parameters :
                relevant_class : "Target"

    :Author: Jan Hendrik Metzen (jhm@informatik.uni-bremen.de)
    :Created: 2010/05/21
    """
    def __init__(self, relevant_class=None, **kwargs):
        super(ChampionGatingNode, self).__init__(**kwargs)
        
        self.set_permanent_attributes(relevant_class = relevant_class,
                                      confusion_matrix = None,
                                      chosen_index =  None)

    def is_trainable(self):
        """ Returns whether this node is trainable. """
        return True
    
    def is_supervised(self):
        """ Returns whether this node requires supervised training """
        return True

    def _train(self, data, label):
        if self.relevant_class==None:
            self.relevantclass=label
        if self.confusion_matrix == None:
            self.confusion_matrix = [defaultdict(float) for i in range(len(data.label))]
                    
        for index in range(len(data.label)):
            if data.label[index].strip() == self.relevant_class:
                if label == self.relevant_class:
                    self.confusion_matrix[index]["tp"] +=1
                else:
                    self.confusion_matrix[index]["fp"] +=1
            else:
                if label == data.label[index].strip():
                    self.confusion_matrix[index]["tn"] +=1
                else:
                    self.confusion_matrix[index]["fn"] +=1
                    
    def _stop_training(self):
        # Compute the  f-measures on the chosen class
        f_measures = []
        for index in range(len(self.confusion_matrix)):
            if (self.confusion_matrix[index]["tp"] + self.confusion_matrix[index]["fp"]) > 0:
                precision = self.confusion_matrix[index]["tp"] / (self.confusion_matrix[index]["tp"] + self.confusion_matrix[index]["fp"])
            else:
                precision = 0.0
            recall = self.confusion_matrix[index]["tp"] / (self.confusion_matrix[index]["tp"] + self.confusion_matrix[index]["fn"])
            if precision + recall > 0:
                f_measures.append(2*precision*recall/(precision + recall))
            else:
                f_measures.append(0.0)
        
        # Choose classifier that maximizes F-Measure
        self.chosen_index = f_measures.index(max(f_measures))
        
    
    def _execute(self, data):
        """ Executes the classifier on the given data vector *data* """
        
        return PredictionVector(prediction = data.prediction[self.chosen_index], label = data.label[self.chosen_index],
            predictor = data.predictor[self.chosen_index])
    
class RidgeRegressionGatingNode(BaseNode):
    """ Gating function using ridge regression to learn weighting
    
    This method performs ridge regression solving the linear least
    squares solution with Tikhonov regularization:
    weights = (A^TA + Tau^T Tau)^-1 * A^T b
    where A is the feature matrix, b is the class vector and Tau is
    the Tikhonov regularization matrix.
    It classifies as class 1 from *class_labels* if the dot product of 
    weights and data is larger than the the *classification_threshold* else
    as class 2 from *class_labels*.    
    
    The regularization matrix is diag(regularization_coefficient**0.5).

    .. todo:: Implement the usage of prediction values

    **Parameters**
    
        :class_labels:
            Determines the order of the two classes.
            This is important, when you want that the prediction
            value is negative for the first class and
            positive for the other one.
            Here it is used to define the relevant class
            where the resulting voting value has to exceed the threshold.
    
            (*optional, default:["Standard","Target"]*)
    
        :use_labels:
            Should determine whether the labels are mapped to 
            -1 and 1 or if the prediction value is used.
            NOT yet implemented!
    
            (*optional, default:True*)
    
        :regularization_coefficient:
            Necessary parameter of the Tikhanov regularization.
            As a default this is not active.
    
            (*optional, default:0.0*)
    
        :classification_threshold:
            Threshold which has to be exceeded by regression,
            such that the sample is classified with the second class. 
    
            (*optional, default:0.0*)
    
    
    **Exemplary Call**
    
    .. code-block:: yaml
    
        -
            node : Ridge_Regression_Gating_Function
            parameters :
                class_labels : ["Target","Standard"]
                regularization_coefficien : 0.0
                classification_threshold : 0.2

    :Author: Jan Hendrik Metzen (jhm@informatik.uni-bremen.de)
    :Created: 2010/05/21
    """
    def __init__(self, class_labels=["Standard","Target"], use_labels=True,
                 regularization_coefficient=0.0, 
                 classification_threshold=0.0, **kwargs):
        super(RidgeRegressionGatingNode, self).__init__(**kwargs)
        
        self.set_permanent_attributes(class_labels = class_labels,
                                      use_labels = use_labels,
                                      regularization_coefficient = regularization_coefficient,
                                      classification_threshold = classification_threshold,
                                      A = None,
                                      b = None,
                                      weights= None)
   
    def is_trainable(self):
        """ Returns whether this node is trainable. """
        return True
    
    def is_supervised(self):
        """ Returns whether this node requires supervised training """
        return True

    def _train(self, data, class_label):
        # Collect data and corresponding class labels in two lists
        if self.A == None:
            self.A = []
            self.b = []
            
        if self.use_labels:
            assert (hasattr(data.label, "__len__"))
            # TODO: Check mapping
            self.A.append(map(lambda x: self.class_labels.index(x.strip())*2-1,
                              data.label))
        else:
            self.A.append(data.prediction)
        self.b.append(self.class_labels.index(class_label)*2-1)

    def _stop_training(self, debug=False):
        # This method performs ridge regression solving the linear least
        # squares solution with  Tikhonov regularization:
        # weights = (A^TA + Tau^T Tau)^-1 * A^T b
        # where Tau is the Tikhonov regularization matrix
        assert len(self.class_labels) == 2, \
               "%s can only be used for binary classification tasks!" % self.__class__.__name__
        A = numpy.array(self.A)
        b = numpy.array(self.b)
        
        tau = numpy.diag([self.regularization_coefficient 
                                        for i in range(A.shape[1])])
        
        try:
            self.weights = numpy.dot(numpy.linalg.inv(numpy.dot(A.T, A) + tau),
                                        numpy.dot(A.T, b))
        except numpy.linalg.LinAlgError:
            raise numpy.linalg.LinAlgError("Singular matrix. Choose a larger "
                                              "regularization coefficient!") 
            
        super(RidgeRegressionGatingNode, self)._stop_training()
    
    def _execute(self, data):
        """ Executes the classifier on the given data vector *data* 
        
        Classifies as class 1 if the dot product of weights and data
        is larger than the the classification threshold else as class 2.       
        .. todo:: Check mapping"""
        data = map(lambda x: self.class_labels.index(x.strip())*2-1, data.label)
        value = numpy.dot(self.weights, data)
        vote = self.class_labels[1] if value >  self.classification_threshold else self.class_labels[0] 
        
        return PredictionVector(prediction = value, label = vote, predictor = self)
        
        
class KNNGatingNode(BaseNode):
    """ Gating function based on k-Nearest-Neighbors

    **Parameters**
        :n: Number of considered neighbors
        
            (*optional, default: 1*)

    **Exemplary Call**
    
    .. code-block:: yaml
    
        -
            node : :KNN_Gating_Function
            parameters :
                n : 1

    :Author: Jan Hendrik Metzen (jhm@informatik.uni-bremen.de)
    :Created: 2010/05/21
    """
    def __init__(self, n=1, **kwargs):
        super(KNNGatingNode, self).__init__(**kwargs)
        
        self.set_permanent_attributes(n = n,
                                      training_examples = [])

    def is_trainable(self):
        """ Returns whether this node is trainable. """
        return True
    
    def is_supervised(self):
        """ Returns whether this node requires supervised training """
        return True

    def _train(self, data, label):
        self.training_examples.append((data, label))
    
    def _execute(self, data):
        """ Executes the classifier on the given data vector *data* """
        distance_fct = lambda x,y: sum((numpy.array(x) != numpy.array(y)))
        label_distance = ((label, distance_fct(training_data.label, data.label)) 
                           for training_data, label in self.training_examples)
        n_smallest_labels = map(lambda x: x[0], 
                               heapq.nsmallest(self.n, label_distance, key=lambda x: x[1]))

        votes_counter = defaultdict(int)
        for label in n_smallest_labels:
            votes_counter[label] += 1

        voting = sorted((votes, label) for label, votes in votes_counter.iteritems())
        majority_vote = voting[-1][1]
        
        return PredictionVector(label = majority_vote, predictor = self)


_NODE_MAPPING = {"Voting_Gating_Function": LabelVotingGatingNode,
                "Precision_Weighted_Gating_Function" : PrecisionWeightedGatingNode,
                "Champion_Gating_Function" : ChampionGatingNode,
                "Ridge_Regression_Gating_Function": RidgeRegressionGatingNode,
                "KNN_Gating_Function" : KNNGatingNode}

