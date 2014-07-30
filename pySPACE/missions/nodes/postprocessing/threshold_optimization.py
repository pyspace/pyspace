""" Optimize classification thresholds """

import logging
from operator import itemgetter
from bisect import insort

import scipy
import numpy

import copy

from pySPACE.missions.nodes.base_node import BaseNode
from pySPACE.resources.data_types.prediction_vector import PredictionVector
from pySPACE.resources.dataset_defs.metric import BinaryClassificationDataset as ClassificationCollection


class ThresholdOptimizationNode(BaseNode):
    """ Optimize the classification threshold for a specified metric
    
    This node changes the classification threshold (i.e. the mapping from
    real valued classifier prediction onto class label) by choosing a threshold
    that is optimal for a given metric on the training data. This may be useful
    in situations when a classifier tries to optimize a different metric than 
    the one one is interested. However, it is always preferable to use a 
    classifier that optimizes for the right target metric since this node can 
    only correct the threshold but not the hyperplane.
    
    If store is set to true, a graphic is stored in the persistency directory 
    that shows the mapping of threshold onto F-Measure on training and test 
    data.
    
    **Parameters**
    
        :metric:
            A string that determines the metric for which the threshold is 
            optimized. The string must be a valid Python expression that evaluates
            to a float. Within this string, the quantities {TP} (true positive),
            {FP} (false positives), {TN} (true negatives), and {FN} (false negatives)
            can be used to compute the the metric. For instance, the string
            "({TP}+{TN})/({TP}+{TN}+{FP}+{FN})" would correspond to the accuracy.
            Some standard metrics (F-Measure, Accuracy) are predefined, i.e.
            it suffices to give the names of these metrics as parameter, the
            corresponding Python expression is determined automatically.
            
            For details and inspiration have a look at :ref:`metric <metrics>`
            in the
            :class:`~pySPACE.resources.dataset_defs.metric.BinaryClassificationDataset`.

            .. warning:: If your metric is not existing, the algorithm will get
                         zero instead and will get problems optimizing.
                         This is due to the fact, that default values for
                         metrics are zero.
            
            (*optional, default: "Balanced_accuracy"*)
     
     
        :class_labels:
            Determines the order of classes, i.e. the mapping of class labels
            onto integers. The first element of the list will be mapped onto 0,
            the second onto 1.
    
            (*recommended, default: ['Standard', 'Target']*)
            
        :preserve_score:
            If True, only the class labels are changed according to the new
            threshold. If False, the classifier prediction score is also adjusted
            by adding the new threshold, i.e. 
            
            .. math:: score_{new} = score_{old} - (threshold_{new} - threshold_{old})
            
            (*optional, default: False*)
    
        :classifier_threshold:
            Old decision threshold of the classifier.
            For SVMs this is zero. For bayesian classifier or after probability
            fits this is 0.5.
            
            (*optional, default: 0.0*)
    
        :recalibrate:
            If the distribution in the incremental learning is expected to be
            significantly different from the training session,
            a new threshold is calculated using only the new examples and not
            considering the old ones.
            
            If the parameter is active, *retrain* is also active!
            
            (*optional, default: False*)
    
        :weight:
            Parameter for weighted metrics
            
            If you want to use it, have a look at :ref:`metric <metrics>`
            and the :mod:`pySPACE.missions.nodes.sink.classification_performance_sink.PerformanceSinkNode`
            
            (*optional, default: 0.5*)
    
        :inverse_metric:
            For some metrics one has to optimize for a low value and not a high.
            This is done by multiplication with -1 in the formula or by setting
            this parameter to True, if you use some predefined metrics, which
            requires minimization.
        
    
    **Exemplary Call**
    
    .. code-block:: yaml
    
        -
            node : Threshold_Optimization
            parameters :
                 metric : "-{FP} - 5*{FN}"
                 class_labels : ['Standard', 'Target']
    
    :Author: Jan Hendrik Metzen (jhm@informatik.uni-bremen.de)
    :Created: 2010/11/25
    
    """
    input_types=["PredictionVector"]
    def __init__(self, metric="Balanced_accuracy", 
                 class_labels=None, preserve_score = False,
                 classifier_threshold = 0.0,
                 recalibrate = False,
                 weight = 0.5,
                 inverse_metric=False,
                 **kwargs):

        super(ThresholdOptimizationNode, self).__init__(**kwargs)
        if metric.startswith("k_"):
            metric=metric[2:]
            self._log(message="Soft metrics are not supported by this node! Switching to hard variant.", level=logging.CRITICAL)
        if metric.startswith("soft_"):
            metric=metric[5:]
            self._log(message="Soft metrics are not supported by this node! Switching to hard variant.", level=logging.CRITICAL)
        if metric.startswith("pol_"):
            metric=metric[4:]
            self._log(message="Soft metrics are not supported by this node! Switching to hard variant.", level=logging.CRITICAL)
        if metric == "AUC":
            metric = "Balanced_accuracy"
            self._log(message="AUC is no relevant metric for this node! Balanced_accuracy taken.", level=logging.CRITICAL)
        # Some hard coded standard metrics
        if metric == "F_measure":
            metric = "0 if {TP} == 0 else 2*{TP}**2/(2*{TP}**2 + {TP}*{FP} + {TP}*{FN})"
        elif metric == "F_measure_standard":
            metric = "0 if {TN} == 0 else 2*{TN}**2/(2*{TN}**2 + {TN}*{FN} + {TN}*{FP})"
        elif metric == "Accuracy":
            metric = "({TP}+{TN})/({TP}+{TN}+{FP}+{FN})"
        elif metric == "Balanced_accuracy":
            #metric = "(0.5*{TP}/({TP}+{FN}) + 0.5*{TN}/({TN}+{FP}))"
            pass
            
        if recalibrate:
            self.retrainable = True

        self.set_permanent_attributes(metric=metric,
                                      metric_fct=None,
                                      classes=class_labels,
                                      preserve_score=preserve_score,
                                      classifier_threshold=classifier_threshold,
                                      weight=weight,
                                      recalibrate=recalibrate,
                                      orientation_up=True,
                                      threshold=0,
                                      instances=[], # list sorted by prediction score
                                      example=None, # classification vector input example
                                      classifier_information={},  # information from the example+own classification information
                                      inverse_metric=inverse_metric)
        
    def balanced_accuracy(self,TP, FP, TN, FN):
        if (TP+FN) == 0 or (TN+FP) == 0:
            return 0.5
        return (0.5*TP/(TP+FN) + 0.5*TN/(TN+FP))

    def is_trainable(self):
        """ Returns whether this node is trainable """
        return True
    
    def is_supervised(self):
        """ Returns whether this node requires supervised training """
        return True

    def _train(self, data, class_label):
        """ Collect training data and class labels """
                
        if self.classes is None:
            try:
                self.set_permanent_attributes(classes=data.predictor.classes)
            except:
                self.set_permanent_attributes(classes=['Standard', 'Target'])
                self._log("No class labels given. Using default: ['Standard', 'Target'].\
                        If you get errors, this was the wrong choice.",level=logging.CRITICAL)
            
        if type(data.label).__name__ == 'str':
            prediction_label = self.classes.index(data.label)
        elif type(data.label).__name__ == 'list' and len(data.label) == 1:
            prediction_label = self.classes.index(data.label[0])
        else:
            raise Exception("The ThresholdOptimizationNode can only handle a "
                            "string or a list with a string as its only element "
                            "as input. Got: %s with type: %s"%(str(data.label),type(data.label)))
        if not class_label in self.classes and "REST" in self.classes:
            class_label = "REST"
        # Insert new (score, predicted_label, actual_label) tuple into list of
        # instances that is sorted by ascending prediction score
        insort(self.instances, (data.prediction, prediction_label, 
                                self.classes.index(class_label)))

        # copying of important classifier parameters to give it to the sink node
        if self.example is None:
            self.example = data
            try:
                self.classifier_information=copy.deepcopy(data.predictor.classifier_information)
            except:
                pass

    def _stop_training(self, debug=False):
        """ Call the optimization algorithm """
        self.calculate_threshold()
        
    def calculate_threshold(self):
        """ Optimize the threshold for the given scores, labels and metric. 
        
        .. note:: 
            This method requires O(n) time (n being the number of training 
            instances). There should be an asymptotically more efficient
            implementation that is better suited for fast incremental learning.
        """
        # Create metric function lazily since it cannot be pickled
        if not hasattr(self,"metric_fct") or self.metric_fct is None:
            self.metric_fct = self._get_metric_fct()
        # Split 3-tuples in instance heap into the three components
        # predictions, predicted labels, and actual label
        predictions = map(itemgetter(0), self.instances)
        prediction_labels = map(itemgetter(1), self.instances)
        labels = map(itemgetter(2), self.instances)
        
        if prediction_labels[0] == 0:
            self.orientation_up = True
        else:
            self.orientation_up = False
        # Determine orientation of hyperplane
        if self.orientation_up:
            TP = labels.count(1)
            FP = labels.count(0)
            TN = 0
            FN = 0
        else:
            TP = 0
            FP = 0
            TN = labels.count(0)
            FN = labels.count(1)

        if self.store:
            self.predictions_train = [[], []]
        
        # Determine the threshold for which the given metric is maximized
        metric_values = []
        for label, prediction_value, in zip (labels, predictions):
            if label == 0 and self.orientation_up:
                TN += 1
                FP -= 1
            elif label == 0 and not self.orientation_up:
                TN -= 1
                FP += 1
            elif label == 1 and self.orientation_up:
                FN += 1
                TP -= 1
            elif label == 1 and not self.orientation_up:
                FN -= 1
                TP += 1    
            assert (TP >= 0 and FP >= 0 and TN >= 0 and FN >=0), \
                    "TP: %s FP: %s TN: %s FN: %s" % (TP, FP, TN, FN)
            metric_values.append(self.metric_fct(TP, FP, TN, FN))

            if self.store:
                self.predictions_train[0].append(prediction_value)
                self.predictions_train[1].append(metric_values[-1])
        # Fit a polynomial of degree 2 to the threshold that maximizes the 
        # metric and its two neighbors. The peak of this polynomial is then 
        # used as threshold of classification
        max_index = metric_values.index(max(metric_values))
        if max_index in [0, len(metric_values)-1]: # pathologic cases
            self.threshold = predictions[max_index]
        else:
            polycoeffs = scipy.polyfit(predictions[max_index-1:max_index+2], 
                                       metric_values[max_index-1:max_index+2], 
                                       2)
            self.threshold = -polycoeffs[1]/(2*polycoeffs[0])

    def start_retraining(self):
        """ Start retraining phase of this node """
        if self.recalibrate:
            # We remove all old training data since we expect that the
            # distributions have shifted and thus, the old data does not help to
            # model the new distributions
            self.set_permanent_attributes(instances=[])
    
    def _inc_train(self, data, class_label):
        """ Provide training data for retraining """
        result = self._train(data, class_label)
        # Recalculate threshold
        self.calculate_threshold()
        
        return result
    
    def _execute(self, data):
        """ Shift the data with the new offset """
        if self.orientation_up:
            predicted_label = \
                self.classes[1] if data.prediction > self.threshold \
                else self.classes[0]
        else:
            predicted_label = \
                self.classes[1] if data.prediction < self.threshold \
                else self.classes[0]
                                        
#        print "data.prediction ", data.prediction
#        print "self.threshold ", self.threshold 
#        print "self.classifier_threshold ", self.classifier_threshold                                        
        if self.preserve_score:
            prediction_score = data.prediction
        else:
            prediction_score = data.prediction - \
                                    (self.threshold - self.classifier_threshold)
        return PredictionVector(label=predicted_label,
                                prediction=prediction_score,
                                predictor=self)

    def _get_metric_fct(self):
        if self.metric == 'Mutual_information':
            metric_fct = lambda TP, FP, TN, FN: ClassificationCollection.mutual_information(TN, FN, TP, FP)
        elif self.metric == 'Normalized_mutual_information':
            metric_fct = lambda TP, FP, TN, FN: ClassificationCollection.normalized_mutual_information(TN, FN, TP, FP)
        elif self.metric == "Balanced_accuracy":
            metric_fct = self.balanced_accuracy
        elif '{TP}' in self.metric or '{FP}' in self.metric or '{TN}' in self.metric or '{FN}' in self.metric:
            metric_fct = lambda TP, FP, TN, FN: eval(self.metric.format(TP=float(TP),
                                                                        FP=float(FP), 
                                                                        TN=float(TN),
                                                                        FN=float(FN)))
        elif self.inverse_metric:
            metric_fct = lambda TP, FP, TN, FN: \
                (-1.0)*ClassificationCollection.calculate_confusion_metrics(
                    {"True_negatives": TN,
                     "True_positives": TP,
                     "False_positives": FP,
                     "False_negatives": FN},
                    weight=self.weight,)[self.metric]
        else: 
            metric_fct = lambda TP, FP, TN, FN: \
                ClassificationCollection.calculate_confusion_metrics(
                    {"True_negatives": TN,
                     "True_positives": TP,
                     "False_positives": FP,
                     "False_negatives": FN},
                    weight=self.weight,)[self.metric]
        return metric_fct 

    def store_state(self, result_dir, index=None): 
        """ Stores this node in the given directory *result_dir* 
        
        .. todo:: Documentation! What is stored? And how?
        """
        if self.store:
            try:
                # Create metric function lazily since it cannot be pickled
                metric_fct = self._get_metric_fct()
                
                # Determine curve on test data
                # TODO: Code duplication (mostly already in train)
                predictions_test = []
                labels_test = []
                for data, label in self.input_node.request_data_for_testing():
                    predictions_test.append(data.prediction)
                    labels_test.append(self.classes.index(label)) 
                
                sort_index = numpy.argsort(predictions_test)
                labels_test = numpy.array(labels_test)[sort_index]
                predictions_test = numpy.array(predictions_test)[sort_index]
                
                # Determine orientation of hyperplane
                if self.orientation_up:
                    TP = list(labels_test).count(1)
                    FP = list(labels_test).count(0)
                    TN = 0
                    FN = 0
                else:
                    TP = 0
                    FP = 0
                    TN = list(labels_test).count(0)
                    FN = list(labels_test).count(1)
                
                self.predictions_test = [[], []]
                for label, prediction_value, in zip(labels_test, predictions_test):
                    if label == 0 and self.orientation_up:
                        TN += 1
                        FP -= 1
                    elif label == 0 and not self.orientation_up:
                        TN -= 1
                        FP += 1
                    elif label == 1 and self.orientation_up:
                        FN += 1
                        TP -= 1
                    elif label == 1 and not self.orientation_up:
                        FN -= 1
                        TP += 1    
                    assert (TP >= 0 and FP >= 0 and TN >= 0 and FN >= 0), \
                        "TP: %s FP: %s TN: %s FN: %s" % (TP, FP, TN, FN)
                    metric_value = metric_fct(TP, FP, TN, FN)
                    
                    self.predictions_test[0].append(prediction_value)
                    self.predictions_test[1].append(metric_value)
                    
                ### Plot ##
                import pylab
                pylab.close()
                fig_width_pt = 307.28987*2 # Get this from LaTeX using \showthe\columnwidth
                inches_per_pt = 1.0/72.27               # Convert pt to inches
                fig_width = fig_width_pt*inches_per_pt  # width in inches
                fig_height =fig_width * 0.5     # height in inches
                fig_size = [fig_width,fig_height]
                params = {'axes.labelsize': 10,
                          'text.fontsize': 8,
                          'legend.fontsize': 8,
                          'xtick.labelsize': 10,
                          'ytick.labelsize': 10}
                pylab.rcParams.update(params)
                fig = pylab.figure(0, dpi=400, figsize=fig_size)
                
                xmin = min(min(self.predictions_train[0]),
                           min(self.predictions_test[0]))
                xmax = max(max(self.predictions_train[0]),
                           max(self.predictions_test[0]))
                ymin = min(min(self.predictions_train[1]),
                           min(self.predictions_test[1]))
                ymax = max(max(self.predictions_train[1]),
                           max(self.predictions_test[1]))
                
                pylab.plot(self.predictions_train[0], self.predictions_train[1],
                           'b', label='Training data')
                pylab.plot(self.predictions_test[0], self.predictions_test[1],
                           'g', label='Unseen test data')
                pylab.plot([self.classifier_threshold, self.classifier_threshold],
                           [ymin, ymax], 'r', label='Original Threshold', lw=5)
                pylab.plot([self.threshold, self.threshold],
                           [ymin, ymax], 'c', label='Optimized Threshold', lw=5)
                pylab.legend(loc = 0)
                pylab.xlim((xmin, xmax))
                pylab.ylim((ymin, ymax))
                pylab.xlabel("Threshold value")
                pylab.ylabel("Metric: %s" % self.metric)
                
                # Store plot
                from pySPACE.tools.filesystem import  create_directory
                import os
                node_dir = os.path.join(result_dir, self.__class__.__name__)
                create_directory(node_dir)
                
                pylab.savefig(node_dir + os.sep + "threshold_metric.pdf")
            except:
                self._log("To many channels chosen for the retained channels! "
                      "Replaced by maximum number.", level=logging.WARNING)
            
        super(ThresholdOptimizationNode,self).store_state(result_dir)


_NODE_MAPPING = {"Threshold_Optimization": ThresholdOptimizationNode}
