# This Python file uses the following encoding: utf-8
# The upper line is needed for one comment in this module.
""" Calculate performance measures from classification results and store them

All performance sink nodes interface to the
:mod:`~pySPACE.resources.dataset_defs.metric` datasets, where the final metric values .
are calculated.

These results can be put together using the
:class:`~pySPACE.resources.dataset_defs.performance_result.PerformanceResultSummary`.

"""
import os
import copy
import warnings
import cPickle
import numpy

import timeit

from pySPACE.missions.nodes.base_node import BaseNode
from pySPACE.tools.filesystem import create_directory
from pySPACE.resources.dataset_defs.metric import metricdict, \
                                            BinaryClassificationDataset, \
                                            MultinomialClassificationDataset,\
                                            RegressionDataset

import logging


class PerformanceSinkNode(BaseNode):
    """ Calculate performance measures from standard prediction vectors and store them
    
    It takes all classification vectors that are passed on to it
    from a continuous classifier, calculates the performance measures and
    stores them. The results can be later on collected and merged
    into one tabular with the
    :class:`~pySPACE.missions.operations.node_chain.NodeChainOperation`.
    This one can be read manually or it can be
    visualized with a gui.

    .. note:: FeatureVectorSinkNode was the initial model of this node.
    
    **Parameters**
        :evaluation_type:
            Define type of incoming results to be processed.
            Currently ``binary``
            (:class:`~pySPACE.resources.dataset_defs.metric.BinaryClassificationDataset`)
            and ``multinomial``
            (:class:`~pySPACE.resources.dataset_defs.metric.MultinomialClassificationDataset`)
            classification (also denoted as ``multiclass'' classification) and ``regression`` (even for n-dimensional output)
            (:class:`~pySPACE.resources.dataset_defs.metric.RegressionDataset`)
            metrics can be calculated.

            For the multinomial and regression case
            several parameters are not yet important.
            These are:

                * ir_class
                * save_roc_points
                * calc_AUC
                * calc_soft_metrics
                * calc_loss
                * sum_up_splits

            .. warning:: Multinomial classification and regression have not yet
                         been used often enough with pySPACE and require
                         additional testing.

            (*optional, default: "binary"*)

        :ir_class:
            The class name (as string) for which IR statistics are to be output.
            
            (*recommended, default: 'Target'*)

        :sec_class:
            For binary classification the second class (not the *ir_class*)
            can be specified. Normally it is detected by default and not
            required, except for one_vs_REST scenarios,
            where it can not be determined.

            (*optional, default: None*)

        :save_individual_classifications:
            If True, for every processed split a pickle file will be generated
            that contains the numerical classification result (cresult) for every
            individual window along with the estimated class label (c_est), the
            true class label (c_true) and the number of features used (nr_feat).
            The result is a list whose elements correspond to a single window and
            have the following shape:
            ::
            
                [ [c_est, cresult, nr_feat], c_true ]

            (*optional, default: False*)

        :save_roc_points:
            If True, for every processed split a pickle file will be generated
            that contains a list of tuples (=points) increasing by FP rate, that
            can be used to plot a Receiver Operator Curve (ROC) and a list, that
            contains the actually used point in the ROC space together with (0|0)
            and (1|1). The result has the following shape:
            ::
            
                ( [(fp_rate_1,tp_rate_1), ... ,(fp_rate_n,tp_rate_n)],
                  [(0.0,0.0), (fp_rate, tp_rate), (1.0,1.0)])

            For comparing ROC curves, you can use the analysis GUI
            (*performance_results_analysis.py*).

            (*optional, default: False*)

        :weight:
            weight is the weight for the weighted accuracy. For many scenarios
            a relevant performance measure is a combination of
            True-Positive-Rate (TPR) and True-Negative-Rate (TNR), where one of the
            two might be of higher importance than the other, and thus gets a
            higher weight. Essentially, the weighted accuracy is
            calculated by
            
            .. math:: \\text{Weighted\_accuracy} = weight \\cdot TPR + (1 - weight) \\cdot TNR
            
            If this parameter is not set, the value equals the balanced accuracy.
            In the case of `multinomial` classification, this parameter
            has to be a dictionary.
            
            (*optional, default: 0.5*)
            
        :measure_times:
            measure the average and maximum time that is needed for the processing 
            of the data between the last sink node in the node chain and this node.
            
            (*optional, default: True*)

        :calc_soft_metrics:
            integrate uncertainty of classifier into metric
            prediction value is projected to interval [-1,1]
            
            (*optional, default: False*)

        :calc_train:
            Switch for calculating metrics on the training data
            
            (*optional, default: True*)

        :calc_AUC:
            Calculate the AUC metric
            
            (*optional, default: True*)

        :calc_loss:
            Integrates the calculated losses into the final csv-file.
            (L1, L2), (LDA, SVM, RMM), (restricted, unrestricted) 
            and (equal weighted, *balanced*) losses are
            calculated in all combinations, resulting in 24 entries.
            
            (*optional, default: True*)

            :loss_restriction:
                Maximum value of the single loss values.
                Everything above is reduced to the maximum.

                (*optional, default: 2*)

        :sum_up_splits:
            If you use a CV-Splitter in your node chain, the performance sink adds up
            the basic metrics and calculates confusion matrix metrics with these 
            values. The other metrics are averaged.
            So a lot of more testing examples are relevant for the calculation.
            
            (*optional, default: False*)
            
        :dataset_pattern:
            If the __Dataset__ is of the form "X_Y_Z", then this pattern can be
            specified with this parameter. The different values X, Y, Z will then
            appear in corresponding columns in the results.csv. Example: If the
            datasets are of the form "NJ89_20111128_3", and one passes the
            dataset_pattern "subject_date_setNr", then the results.csv will have
            the columns __Subject__, __Date__ and __SetNr__ with the corresponding
            values parsed (note the added underscores and capitalized first letter).
            
            (*optional, default: None*)
            
        :decision_boundary:
            If your decision boundary is not at zero you should specify this for
            the calculation of metrics depending on the prediction values.
            Probabilistic classifiers often have a boundary at 0.5. 
            
            (*optional, default: 0.0*)
        
        :save_trace:
            Generates a table which contains a confusion matrix over time/samples.
            There are two types of traces: short traces and long traces. 
            The short traces contain only the information, if a classification
            was a TP, FN, FP or TN. The long traces furthermore contain
            loss values and are saved as a dictionary.
            To save only short traces (for, e.g. performance reasons),
            set save_trace to ``short``.
            To save long and short traces, set save_trace to True.
            The encoding in trace is:
            :TP: 0
            :FN: 1
            :FP: 2
            :TN: 3

            (*optional, default: False*)
            
    **Exemplary Call**

    .. code-block:: yaml

        -  
            node : Classification_Performance_Sink
            parameters :
                ir_class : "Target"
                weight : 0.5

    :input:  PredictionVector
    :output: ClassificationDataset
    :Author: Mario Krell (mario.krell@dfki.de)
    :Created: 2012/08/02
    """
    input_types = ["PredictionVector"]

    def __init__(self, classes_names=[], ir_class="Target", sec_class=None,
                 save_individual_classifications=False, save_roc_points=False,
                 weight=0.5, measure_times=True, calc_soft_metrics=False,
                 sum_up_splits=False, dataset_pattern=None, calc_AUC=True,
                 calc_loss=True, calc_train=True, save_trace=False,
                 decision_boundary=None, loss_restriction=2,
                 evaluation_type="binary",
                 **kwargs):
        super(PerformanceSinkNode, self).__init__(**kwargs)
        if save_roc_points:
            calc_AUC = True
        if evaluation_type in ["multinomial", "multiclass"]:
            evaluation_type = "multinomial"
            save_trace = False
            save_roc_points = False
            calc_AUC = False
            calc_loss = False
            calc_soft_metrics = False
            sum_up_splits = False
            cc = MultinomialClassificationDataset(dataset_pattern=
                                                  dataset_pattern)
        elif evaluation_type == "binary":
            cc = BinaryClassificationDataset(dataset_pattern=dataset_pattern)
        elif evaluation_type == "regression":
            save_trace = False
            save_roc_points = False
            calc_AUC = False
            calc_loss = False
            calc_soft_metrics = False
            sum_up_splits = False
            cc = RegressionDataset(dataset_pattern=dataset_pattern)

        store = \
            save_individual_classifications or \
            save_roc_points or \
            self.store or \
            save_trace

        self.set_permanent_attributes(
            ir_class=ir_class.strip(),
            classification_dataset=cc,
            classes_names=classes_names,
            # determined later on for checks in binary classification
            sec_class=sec_class,
            weight=weight,
            save_individual_classifications=save_individual_classifications,
            save_roc_points=save_roc_points,
            measure_times=measure_times,
            calc_soft_metrics=calc_soft_metrics,
            example=None,
            sum_up_splits=sum_up_splits,
            calc_AUC=calc_AUC,
            calc_loss=calc_loss,
            decision_boundary=decision_boundary,
            loss_restriction=loss_restriction,
            calc_train=calc_train,
            save_trace=save_trace,
            store=store,
            evaluation_type=evaluation_type,
            invert_classification=False)

    def reset(self):
        """ classification_dataset has to be kept over all splits """
        # We have to create a temporary reference since we remove 
        # the self.permanent_state reference in the next step by overwriting
        # self.__dict__
        tmp = self.permanent_state
        # reset should not delete classification dataset
        # if you want to delete the dataset just do it explicitly.
        tmp["classification_dataset"] = self.classification_dataset
        self.__dict__ = copy.copy(tmp)
        self.permanent_state = tmp

    def is_trainable(self):
        """ Return whether this node is trainable. 

        .. todo:: Check if return should be False and edit documentation
        """
        # Though this node is not really trainable, it returns true in order
        # to request the training data from previous notes.
        return True

    def is_supervised(self):
        """ Return whether this node requires supervised training. """
        return True

    def _train(self, data, label):
        # We do nothing
        pass

    def process_current_split(self):
        """ Main processing part on test and training data of current split
        
        Performance metrics are calculated for training and test data separately.
        Metrics on training data help to detect errors in classifier construction
        and to compare in how far it behaves the same way as on testing data.
        
        The function only collects the data, measures execution times 
        and calls functions to update confusion matrices.
        """
        ################
        ### TRAINING ###
        ################
        self._log("Processing training data",level=logging.INFO)
        self.train_classification_outcome = []
        self.training_time = 0
        if self.measure_times:
            start_time_stamp = timeit.default_timer()
        for classification_vector, label in self.input_node.request_data_for_training(False):
            if self.calc_train:
                self.set_helper_parameters(classification_vector,label)
                self.train_classification_outcome.append((classification_vector, label))
        if self.measure_times:
            stop_time_stamp = timeit.default_timer()
            self.training_time = stop_time_stamp - start_time_stamp



        if self.calc_train and self.evaluation_type == "binary" and not self.train_classification_outcome==[]:
            if self.decision_boundary is None and self.train_classification_outcome[0][0].predictor.node_name in \
                    ["PlattsSigmoidFitNode",
                     "LinearFitNode",
                     "SigmoidTransformationNode"]:
                self.decision_boundary=0.5
            elif self.decision_boundary is None:
                self.decision_boundary=0
            train_result = BinaryClassificationDataset.calculate_metrics(
                classification_results=self.train_classification_outcome,
                calc_soft_metrics=self.calc_soft_metrics,
                invert_classification=self.invert_classification,
                ir_class=self.ir_class, sec_class=self.sec_class,
                loss_restriction=self.loss_restriction,
                time_periods=[],
                calc_AUC=self.calc_AUC,calc_loss=self.calc_loss,
                weight=self.weight,save_roc_points=self.save_roc_points,
                decision_boundary=self.decision_boundary)
            try:
                train_metrics, self.train_R = train_result
            except:
                train_metrics = train_result
        elif self.calc_train and self.evaluation_type == "multinomial":
            train_metrics = MultinomialClassificationDataset.calculate_metrics(
                        classification_results=self.train_classification_outcome,
                        weight=self.weight, classes=self.classes_names)
        elif self.calc_train and self.evaluation_type == "regression":
            train_metrics = RegressionDataset.calculate_metrics(
                regression_results=self.train_classification_outcome,
                weight=self.weight)
        elif not self.train_classification_outcome:
            train_metrics = metricdict()
        ###############
        ### TESTING ###
        ###############
        self._log("Processing testing data",level=logging.INFO)
        # for saving the actual numerical classification results
        self.classification_outcome = []
        # class\guess ir sec
        # ir_class:   TP FN
        # sec_class:  FP TN
        
        # initialization to measure execution speed
        self.time_periods = []
        if self.measure_times:
            self.time_periods = []
            start_time_stamp = timeit.default_timer()
        
        self.example = None
        for classification_vector, label in \
                                   self.input_node.request_data_for_testing():
            if self.measure_times:
                stop_time_stamp = timeit.default_timer()
                self.time_periods.append(stop_time_stamp - start_time_stamp)
            self.set_helper_parameters(classification_vector,label)
            self.classification_outcome.append((classification_vector, label))
            # re-initialization of time before next item is requested
            if self.measure_times:
                start_time_stamp = timeit.default_timer()

        if self.decision_boundary is None and \
            len(self.classification_outcome) > 0 and \
            self.classification_outcome[0][0].predictor.node_name in \
                ["PlattsSigmoidFitNode",
                 "LinearFitNode",
                 "SigmoidTransformationNode"]:
            self.decision_boundary=0.5
        elif self.decision_boundary is None:
            self.decision_boundary=0
        if self.evaluation_type == "binary":
            result = BinaryClassificationDataset.calculate_metrics(
                        classification_results=self.classification_outcome,
                        calc_soft_metrics=self.calc_soft_metrics,
                        invert_classification=self.invert_classification,
                        ir_class=self.ir_class, sec_class=self.sec_class,
                        loss_restriction=self.loss_restriction,
                        time_periods=self.time_periods,
                        calc_AUC=self.calc_AUC,calc_loss=self.calc_loss,
                        weight=self.weight,save_roc_points=self.save_roc_points,
                        decision_boundary=self.decision_boundary)
            try:
                metrics, self.R = result
            except:
                metrics = result
        elif self.evaluation_type=="multinomial":
            metrics = MultinomialClassificationDataset.calculate_metrics(
                        classification_results=self.classification_outcome,
                        weight=self.weight)
        elif self.evaluation_type=="regression":
            metrics = RegressionDataset.calculate_metrics(
                regression_results=self.classification_outcome,
                weight=self.weight)
        # add the training time if training was done
        if self.measure_times:
            metrics["Training_time"] = self.training_time
        try:
            classifier_information = self.classification_outcome[0][0].\
                predictor.classifier_information
        except:
            classifier_information=dict()

        # add the training metrics
        if self.calc_train:
            skip_keys = classifier_information.keys()
            for key,value in train_metrics.items():
                if not key in skip_keys:
                    metrics["train_"+key] = value
        self.classification_dataset.add_split(metrics,
                                              train=False,
                                              split=self.current_split,
                                              run=self.run_number)
        if self.save_trace:
            self.trace, self.long_trace=self.calculate_classification_trace(
                        classification_results=self.classification_outcome,
                        calc_soft_metrics=self.calc_soft_metrics,
                        ir_class=self.ir_class, 
                        sec_class=self.sec_class,
                        loss_restriction=self.loss_restriction,
                        calc_loss=self.calc_loss,
                        decision_boundary=self.decision_boundary,
                        save_trace=self.save_trace)
        self._log("Metrics added to dataset",level=logging.INFO)

    def set_helper_parameters(self, classification_vector, label):
        """ Fetch some node parameters from the classification vector """
            # get an example for a classification vector for further analysis
        if self.example is None:
            self.example = classification_vector
            if not self.evaluation_type == "binary":
                return
            try:
                self.decision_boundary = \
                    classification_vector.predictor.classifier_information["decision_boundary"]
            except:
                pass
            if self.decision_boundary is None and classification_vector.predictor.node_name in \
                            ["PlattsSigmoidFitNode",
                             "LinearFitNode",
                             "SigmoidTransformationNode"]:
                self.decision_boundary = 0.5
            elif self.decision_boundary is None:
                self.decision_boundary = 0
            if (self.example.prediction > self.decision_boundary and
                    self.example.label == self.ir_class) or \
                (self.example.prediction <= self.decision_boundary and not
                    self.example.label == self.ir_class):
                self.invert_classification = False
            elif self.evaluation_type == "binary":
                self.invert_classification = True
                warnings.warn(
                    "Your ir_class did not get the higher value " +
                    "from the classifier.\n " +
                    "Label %s, got value %f.\n" % (self.example.label,
                    self.example.prediction) +
                    "You should adjust that  and " +
                    "maybe switch the given class_labels or add " +
                    "preserve_score: False to the " +
                    "Threshold_Optimization node! " +
                    "Furthermore you should check the parameter" +
                    ": decision_boundary!")
        if self.evaluation_type == "binary":
            if self.sec_class is None:
                p_label = classification_vector.label.strip()
            if self.sec_class is None and not (p_label == self.ir_class):
                self.sec_class = p_label

    @classmethod
    def calculate_classification_trace(cls,classification_results,
                                       calc_soft_metrics=False,
                                       ir_class="Target", sec_class=None,
                                       loss_restriction=2.0,
                                       calc_loss=False,
                                       decision_boundary=0.0,
                                       save_trace=True):
        """ Calculate the classification trace, i.e. TN,TP,FN,FP for every sample
        
        The trace entries are encoded for size reasons as short (trace) or 
        in a comprehensive version as dicts (long_trace)
        
        The encoding in trace is:
        
            :TP: 0
            :FN: 1
            :FP: 2
            :TN: 3
        
        :Author: Hendrik Woehrle (hendrik.woehrle@dfki.de), Mario Krell (mario.krell@dfki.de)
        :Returns: trace, long_trace
        """
        trace = []
        long_trace = []
        
        for prediction_vector,label in classification_results:
            if sec_class is None and not label == ir_class:
                sec_class = label
            confusion_matrix = metricdict(float)
            BinaryClassificationDataset.update_confusion_matrix(prediction_vector,
                                        label,calc_soft_metrics=calc_soft_metrics,
                                        ir_class=ir_class, sec_class=sec_class,
                                        confusion_matrix=confusion_matrix,
                                        decision_boundary=decision_boundary)

            # 
            if calc_loss and not save_trace == "short":
                BinaryClassificationDataset.update_loss_values(classification_vector=prediction_vector,
                                        label=label,
                                        ir_class=ir_class, sec_class=sec_class,
                                        loss_dict=confusion_matrix,
                                        loss_restriction=loss_restriction)
            if not save_trace == "short":
                long_trace.append(confusion_matrix)
            if confusion_matrix["True_positives"] == 1:
                trace.append(0)
            elif confusion_matrix["False_negatives"] == 1:
                trace.append(1)
            elif confusion_matrix["False_positives"] == 1:
                trace.append(2)
            elif confusion_matrix["True_negatives"] == 1:
                trace.append(3)
            else:
                raise ValueError("At least one element in the confusion matrix should be 1")
        return trace, long_trace

    def get_result_dataset(self):
        """ Return the result dataset """
        if not self.sum_up_splits:
            return self.classification_dataset
        else:
            self.classification_dataset.merge_splits()
            return self.classification_dataset

    def store_state(self, result_dir, index=None):
        """ Stores additional information (classification_outcome, roc_points) in the *result_dir* """
        if self.store:
            node_dir = os.path.join(result_dir, self.__class__.__name__)
            create_directory(node_dir)
            
            if self.save_individual_classifications:
                name = 'classification_outcome_sp%s.pickle' % self.current_split
                result_file = open(os.path.join(node_dir, name), "wb")
                # predictor is a reference to the actual classification node
                # object. This can not be pickled! Therefore, replace the
                # predictor attribute by the classification node's node_specs
                for single_classification in self.classification_outcome:
                    single_classification[0].predictor = \
                            single_classification[0].predictor.node_specs
                result_file.write(cPickle.dumps(self.classification_outcome, protocol=2))
                result_file.close()
            if self.save_roc_points:
                name = 'roc_points_sp%s.pickle' % self.current_split
                result_file = open(os.path.join(node_dir, name), "wb")
                result_file.write(cPickle.dumps(self.R, protocol=2))
                result_file.close()
            if self.save_trace:
                name = 'trace_sp%s.pickle' % self.current_split
                result_file = open(os.path.join(node_dir, name), "wb")
                result_file.write(cPickle.dumps(self.trace, protocol=2))
                result_file.close()
                if len(self.long_trace) > 0: 
                    name = 'long_trace_sp%s.pickle' % self.current_split
                    result_file = open(os.path.join(node_dir, name), "wb")
                    result_file.write(cPickle.dumps(self.long_trace, protocol=2))
                    result_file.close()


class LeaveOneOutSinkNode(PerformanceSinkNode):
    """ Request the leave one out metrics from the input node 
    
    **Parameters**
    
    see: :class:`PerformanceSinkNode`
    
    **Exemplary Call**

    .. code-block:: yaml

        -  
            node : LOO_Sink
            parameters :
                ir_class : "Target"
    """
    def process_current_split(self):
        """ Get training results and input node metrics """
        ### TRAINING ### # code copy till main part
        self.train_classification_outcome = []
        if self.measure_times:
            start_time_stamp = timeit.default_timer()
        for classification_vector, label in \
                                   self.input_node.request_data_for_training(False):
            if self.calc_train:
                self.set_helper_parameters(classification_vector,label)
                self.train_classification_outcome.append((classification_vector, label))
        if self.measure_times:
            stop_time_stamp = timeit.default_timer()
            self.training_time = stop_time_stamp - start_time_stamp

        if self.calc_train:
            train_result = BinaryClassificationDataset.calculate_metrics(
                classification_results=self.train_classification_outcome,
                calc_soft_metrics=self.calc_soft_metrics,
                invert_classification=self.invert_classification,
                ir_class=self.ir_class, sec_class=self.sec_class,
                loss_restriction=self.loss_restriction,
                time_periods=[],
                calc_AUC=self.calc_AUC, calc_loss=self.calc_loss,
                weight=self.weight, save_roc_points=self.save_roc_points,
                decision_boundary=self.decision_boundary)
            try:
                train_metrics,self.train_R = train_result
            except:
                train_metrics = train_result
        ######################### Main Part #########################
        try: 
            metrics = copy.deepcopy(self.train_classification_outcome[0][0].predictor.loo_metrics)
        except AttributeError:
            warnings.warn("Input node does not provide LOO metrics.")
            metrics = metricdict(float)
        #############################################################
        # add the training time #Code copy from here
        if self.measure_times:
            metrics["Training_time"] = self.training_time
        # add the training metrics
        if self.calc_train:
            try:
                classifier_information = \
                    self.train_classification_outcome[0][0].predictor.\
                        classifier_information
            except:
                classifier_information=dict()
            skip_keys=classifier_information.keys()
            for key,value in train_metrics.items():
                if not key in skip_keys:
                    metrics["train_"+key] = value
        self.classification_dataset.add_split(metrics,
                                              train=False,
                                              split=self.current_split,
                                              run=self.run_number)


class SlidingWindowSinkNode(PerformanceSinkNode):
    """ Calculate and store performance measures from classifications of sliding windows
    
    This node inherits most of its functionality from *PerformanceSinkNode*.
    Thus, for parameter description of super class parameters see documentation
    of *PerformanceSinkNode*.
    
    Additionally the following functionality is provided:
    
    1) The definition of uncertain areas, which are excluded in the metrics 
    calculation process, are possible. This is useful for sliding window 
    classification, i.e. if the true label is not known in each sliding step. 
    2) It is possible to label the test data only now. For that an epoch signal 
    (e.g. a movement marker window) must be specified. 
    3) Instead of excluding sliding windows from classifier evaluation, the
    'true' label function shape (a step function, which is zero for the 
    negative class and one for the positive class) can be somehow fit in the 
    uncertain range. At the moment there is only one way for doing this:
    
        * from_right_count_negatives: Find the point where prediction of the 
                                   negative class starts by searching backwards 
                                   in time. There can be specified how many 
                                   'outliers' are ignored, i.e. how stable the 
                                   prediction has to be.
    
    **Parameters**
    
        :uncertain_area:
            A list of tuples of the lower and the upper time value (in ms) for which
            no metrics calculation is done. The values should be given with respect
            to the last window of an epoch, i.e. sliding window series (which has 
            time value zero). 
            If additionally *determine_labels* is specified then the first tuple of 
            *uncertain_area* describes the bounds in which the label-change-point is
            determined. The lower bound should be the earliest time point when the 
            detection makes sense; the upper bound should be the earliest time point
            when there MUST BE a member of the positive class.
            
            (*optional, default: None*)
            
        :sliding_step:
            The time (in ms) between two consecutive windows.
            
            (*optional, default: 50*)
        
        :determine_labels:
            If specified the label-change-point (index where the class label changes
            from negative to positive class) is determined for every epoch. This is
            done via counting the occurrence of negative classified sliding windows
            from the index point where the positive class is sure 
            (uncertain_area[1]) to the index point where the negative class is sure
            (uncertain_area[0]) If *determine_labels* instances were found in 
            consecutively windows the label-change-point is has been reached. 
            If *determine_labels* > 1, the methods accounts for outliers.
            
            .. note:: Using this option makes it hard to figure out to which true
                     class errors pertain (since it is somehow arbitrary). You 
                     should be careful which metric you analyze for performance
                     evaluation (different class instance costs can't be modeled).
                     
        :epoch_signal:
            The class name (label) of the event that marks the end of an epoch,
            e.g. the movement. This can be used when null_marker windows (of an 
            unknown class) and a signal window which marks the event were cut out.
            With respect to this event the former windows will be relabeled 
            according to *classes_names*.
            
            (*optional, default: None*)
        
        :epoch_eval:
            If True, evaluation is done per epoch, i.e. per movement. Performance
            metrics are averaged across epochs for every split. This option might
            be necessary if the epochs have variable length, i.e. the class
            distribution alters in every epoch.
            
            (*optional, default: False*)
            
        :save_score_plot:
            If True a plot is stored which shows the average prediction value
            against the time point of classification.
            
            (*optional, default: False*)
        
        :save_trial_plot:
            If True a plot is stored which shows developing of the prediction
            scores for each single trial.
            
            (*optional, default: False*)
            
        :save_time_plot:
            If True a plot is stored which shows the predicted labels for all 
            trials across time.
            
            (*optional, default: False*)
            
        :sort: 
            If True the data has to be sorted according to the time (encoded in the
            tag attribute. Be aware that this only makes sense for data sets with
            unique time tags.
            
            (*optional, default: False*)
        
        :unused_win_defs:
            List of window definition names which shall not be used for evaluation.
             
            (*optional, default: []*)
            
    **Exemplary Call**

    .. code-block:: yaml

        -  
            node : Sliding_Window_Performance_Sink
            parameters :
                ir_class : "LRP"
                classes_names : ['NoLRP','LRP']
                uncertain_area : ['(-600,-350)']
                calc_soft_metrics : True
                save_score_plot : True

    :input:  PredictionVector
    :output: ClassificationCollection
    :Author: Anett Seeland (anett.seeland@dfki.de)
    :Created: 2011/01/23
    
    """
    def __init__(self,
                 uncertain_area=None, sliding_step=50, save_score_plot=False, 
                 save_trial_plot=False, save_time_plot=False, 
                 determine_labels=None, epoch_eval=False, epoch_signal=None, 
                 sort=False, unused_win_defs=[], **kwargs):
        if epoch_eval:
            kwargs["save_roc_points"] = False
            kwargs["calc_AUC"] = False
            
        super(SlidingWindowSinkNode,self).__init__(**kwargs)
        
        self.set_permanent_attributes(uncertain_area=uncertain_area,
                                      sliding_step=sliding_step,
                                      determine_labels=determine_labels,
                                      epoch_signal=epoch_signal,
                                      epoch_eval=epoch_eval,
                                      save_score_plot=save_score_plot,
                                      save_trial_plot=save_trial_plot,
                                      save_time_plot=save_time_plot,
                                      sort=sort,
                                      unused_win_defs=unused_win_defs)
        if self.store == False:
            self.store = save_score_plot or save_trial_plot or save_time_plot
        
    def process_current_split(self):
        """ Compute for the current split of training and test data performance
        one sliding windows.
        """
        ### TRAINING ###
        # Code from classificationSinkNode #
        self._log("Processing training data", level=logging.INFO)
        self.train_classification_outcome = []
        self.training_time = 0
        if self.measure_times:
            start_time_stamp = timeit.default_timer()
        for classification_vector, label in self.input_node.request_data_for_training(False):
            if classification_vector.specs["wdef_name"] in self.unused_win_defs:
                continue
            if self.calc_train:
                self.set_helper_parameters(classification_vector,label)
                self.train_classification_outcome.append((classification_vector,
                                                          label))
        if self.measure_times:
            stop_time_stamp = timeit.default_timer()
            self.training_time = stop_time_stamp - start_time_stamp
        
        # we assume that in the training case no sliding windows are used, i.e.,
        # the windows have a known true label
        if self.calc_train and self.evaluation_type == "binary" and not self.train_classification_outcome==[]:
            if self.decision_boundary is None and \
                    self.train_classification_outcome[0][0].predictor.node_name in [
                        "PlattsSigmoidFitNode", "LinearFitNode", "SigmoidTransformationNode"]:
                self.decision_boundary = 0.5
            elif self.decision_boundary is None:
                self.decision_boundary = 0
            train_result = BinaryClassificationDataset.calculate_metrics(
                classification_results=self.train_classification_outcome,
                calc_soft_metrics=self.calc_soft_metrics,
                invert_classification=self.invert_classification,
                ir_class=self.ir_class, sec_class=self.sec_class,
                loss_restriction=self.loss_restriction,
                time_periods=[],
                calc_AUC=self.calc_AUC,calc_loss=self.calc_loss,
                weight=self.weight,save_roc_points=self.save_roc_points,
                decision_boundary=self.decision_boundary)
            try:
                train_metrics,self.train_R = train_result
            except:
                train_metrics = train_result
        elif self.calc_train and self.evaluation_type == "multinomial":
            train_metrics = MultinomialClassificationDataset.calculate_metrics(
                       classification_results=self.train_classification_outcome,
                       weight=self.weight)
        elif self.calc_train and self.evaluation_type == "regression":
            train_metrics = RegressionDataset.calculate_metrics(
                           regression_results=self.train_classification_outcome,
                           weight=self.weight)
        elif not self.train_classification_outcome:
            train_metrics = metricdict()
        
        # TESTING
        self._log("Processing testing data",level=logging.INFO)
        # for saving the actual numerical classification results
        self.classification_outcome = [] 
        # class\guess ir sec
        # ir_class:   TP FN
        # sec_class:  FP TN
        
        # initialization to measure execution speed
        self.time_periods = []
        if self.measure_times:
            start_time_stamp = timeit.default_timer()
            
        for classification_vector, label in \
                self.input_node.request_data_for_testing():
            if self.measure_times:
                stop_time_stamp = timeit.default_timer()
                self.time_periods.append(stop_time_stamp - start_time_stamp)
            
            # parse 'tag': 'Epoch Start: 395772ms; End: 396772ms; Class: Target'
            classification_vector.specs['start_time']= \
                float(classification_vector.tag.split(';')[0].split(':')[1].strip('ms'))
            classification_vector.specs['end_time']= \
                float(classification_vector.tag.split(';')[1].split(':')[1].strip('ms'))
            self.set_helper_parameters(classification_vector,label)
            self.classification_outcome.append((classification_vector,label))
            
            if self.measure_times:
                start_time_stamp = timeit.default_timer()
        
        if self.sort:
            # sort classification vectors in time
            self.classification_outcome.sort(key=lambda tupel:tupel[0].specs['start_time'])
        
        if self.decision_boundary is None and len(self.classification_outcome) \
                > 0 and self.classification_outcome[0][0].predictor.node_name \
                                    in ["PlattsSigmoidFitNode", "LinearFitNode",
                                                   "SigmoidTransformationNode"]:
            self.decision_boundary = 0.5
        elif self.decision_boundary is None:
            self.decision_boundary = 0
        
        self.data_time = dict()

        if self.epoch_signal is not None:
            marker = 0
            self.data_time[marker] = []
            # split according to signal
            for classification_vector, label in self.classification_outcome:
                if label == self.epoch_signal:
                    marker += 1
                    self.data_time[marker] = []
                else:
                    self.data_time[marker].append((classification_vector, label))
            del self.data_time[marker]
        else:
            # split windows according to the time
            last_window_end_time = 0.0
            marker = -1          
            for classification_vector, label in self.classification_outcome:
                if classification_vector.specs['start_time'] > \
                        last_window_end_time or \
                        classification_vector.specs['end_time'] < \
                        last_window_end_time:
                    marker += 1
                    self.data_time[marker] = [(classification_vector, label)]
                elif classification_vector.specs['end_time'] == \
                                       last_window_end_time + self.sliding_step:
                    self.data_time[marker].append((classification_vector,
                                                   label))
                elif "bis-2000" in classification_vector.specs['wdef_name']:
                    marker += 1
                    self.data_time[marker] = [(classification_vector, label)]
                else:
                    # TODO: overlapping epochs - what shall we do???
                    # may be store it with marker = -1 and handle it afterwards
                    self._log("Error: Overlapping epochs in Sink detected!",
                    level=logging.ERROR)
                    #raise Exception("Overlapping epochs in Sink detected!")
                last_window_end_time = classification_vector.specs['end_time']
        
        # delete uncertain classification outcomes or relabel data in 
        # self.classification_outcome and calculate the confusion matrix
        self.classification_outcome = []
        self.label_change_points = []
        performance = None
        for k in self.data_time.keys():
            if self.determine_labels:
                # calculate uncertain indices
                nr_sliding_windows = len(self.data_time[k])
                if self.uncertain_area!=None:
                    bound_indices = range(nr_sliding_windows - \
                           abs(self.uncertain_area[0][0])/self.sliding_step - 1,
                           nr_sliding_windows-abs(self.uncertain_area[0][1])/ \
                           self.sliding_step)
                    if len(self.uncertain_area)>1:
                        uncertain_indices = []
                        for t in self.uncertain_area[1:]:
                            uncertain_indices.extend(range(nr_sliding_windows - \
                                                abs(t[0])/self.sliding_step - 1,
                                                nr_sliding_windows-abs(t[1])/ \
                                                self.sliding_step))
                    else:
                        uncertain_indices = []
                else: # if not specified, assume unbound
                    bound_indices = range(nr_sliding_windows)
                    uncertain_indices = []
                label_change_point = self.from_right_count_negatives(
                        self.data_time[k], self.determine_labels, bound_indices)
                self.label_change_points.append(label_change_point)
                for index, (classification_vector, label) \
                                                in enumerate(self.data_time[k]):
                    if index not in uncertain_indices:
                        if index < label_change_point: # assume neg class
                            self.classification_outcome.append(
                                 (classification_vector, self.classes_names[0]))
                        else: # assume that last elem in trial has correct label
                            self.classification_outcome.append(
                                                     (classification_vector, 
                                                      self.data_time[k][-1][1]))
            else:  
                # calculate uncertain indices
                if self.uncertain_area!=None:
                    nr_sliding_windows = len(self.data_time[0])
                    uncertain_indices = []
                    for t in self.uncertain_area:
                        uncertain_indices.extend(range(nr_sliding_windows - \
                              abs(t[0])/self.sliding_step - 1,
                              nr_sliding_windows-abs(t[1]) / self.sliding_step))
                else:
                    uncertain_indices = []

                for index, (classification_vector, label) \
                                              in enumerate(self.data_time[k]):
                    if index not in uncertain_indices:
                        if self.epoch_signal:
                            if index < uncertain_indices[0]: # negative class
                                new_label = self.classes_names[0]
                            else: # assume last elem in trial has correct label
                                new_label = self.data_time[k][-1][1]
                        self.classification_outcome.append(
                                             (classification_vector, new_label))
            if self.epoch_eval:
                result = self.get_result_metrics()
                if performance == None:
                    performance = result
                else: # combine with old performance
                    new_performance = result
                    performance = self.combine_perf_dict(performance, 
                                                           new_performance, k+1)
                self.classification_outcome = []
        if not self.epoch_eval:
            result = self.get_result_metrics()
            try:
                performance, self.R = result
            except:
                performance = result
        # add the training time
        if self.measure_times:
            performance["Training_time"]  = self.training_time
        try:
            classifier_information = self.classification_outcome[0][0].\
                predictor.classifier_information
        except:
            classifier_information = dict()
        # add the training metrics
        if self.calc_train:
            skip_keys = classifier_information.keys()
            for key, value in train_metrics.items():
                if not key in skip_keys:
                    performance["train_"+key] = value         
        if self.determine_labels:
            performance["~~Avg_Label_Change_Index~~"] = \
                                            numpy.mean(self.label_change_points)
        self.classification_dataset.add_split(performance, train=False,
                                              split=self.current_split,
                                              run=self.run_number)
        if self.save_trace:
            self.trace, self.long_trace=self.calculate_classification_trace(
                             classification_results=self.classification_outcome,
                             calc_soft_metrics=self.calc_soft_metrics,
                             ir_class=self.ir_class, sec_class=self.sec_class,
                             loss_restriction=self.loss_restriction,
                             calc_loss=self.calc_loss,
                             decision_boundary=self.decision_boundary,
                             save_trace=self.save_trace)
        self._log("Metrics added to dataset",level=logging.INFO)

    def get_result_metrics(self):
        """ Calculate metrics based on evaluation type """
        if self.evaluation_type == 'binary':
            result = BinaryClassificationDataset.calculate_metrics(
                             classification_results=self.classification_outcome,
                             calc_soft_metrics=self.calc_soft_metrics,
                             invert_classification=self.invert_classification,
                             ir_class=self.ir_class, sec_class=self.sec_class,
                             loss_restriction=self.loss_restriction,
                             time_periods=self.time_periods, weight=self.weight, 
                             calc_AUC=self.calc_AUC, calc_loss=self.calc_loss,
                             save_roc_points=self.save_roc_points,
                             decision_boundary=self.decision_boundary)
        elif self.evaluation_type == 'multinomial':
            result = MultinomialClassificationDataset.calculate_metrics(
                             classification_results=self.classification_outcome,
                             weight=self.weight)
        elif self.evaluation_type=="regression":
            result = RegressionDataset.calculate_metrics(
                                 regression_results=self.classification_outcome,
                                 weight=self.weight)
        return result

    def from_right_count_negatives(self, y, target_number, bounds):
        """Go through the bounded y (reverse) and find the index i, where 
        target_number values have been consecutively the negative class.
        Return i+target_number as critical index point (labels change)"""
        countNegatives = 0
        countTotal = 0
        for index in range(len(y)-1,-1,-1):
            if index not in bounds:
                continue
            countTotal+=1
            if y[index][0].label==self.classes_names[0]:
                countNegatives+=1
                if countNegatives==target_number:
                    if countTotal==target_number:
                        return bounds[-1]
                    else:
                        return index+target_number
            else:
                countNegatives=0
        return bounds[0]+countNegatives

    def combine_perf_dict(self, old_dict, new_dict, weight):
        """ Combine the values of the dicts by a weighting (iterative) average
           
           .. math:: \\frac{weight-1}{weight} \\cdot \\text{old\_dict} + \\frac{1}{weight} \\cdot \\text{new\_dict}
           
           """
        return_dict = dict()
        for key in old_dict.keys():
            try:
                return_dict[key] = (weight-1.0)/weight * old_dict[key] + \
                               1.0/weight * new_dict[key]
            except TypeError: # for Strings (like parameters)
                # they should not be different among epochs
                if old_dict[key] == new_dict[key]:
                    return_dict[key] = old_dict[key]
        return return_dict    

    def store_state(self, result_dir, index=None):
        """ Stores additional information in the given directory *result_dir* """
        if self.store:
            node_dir = os.path.join(result_dir, self.__class__.__name__)
            create_directory(node_dir)
            
            super(SlidingWindowSinkNode,self).store_state(result_dir)
            
            if self.save_score_plot or self.save_trial_plot or self.save_time_plot:
                import matplotlib.pyplot as plt
                data = [[pv.prediction for pv, _ in self.data_time[k]] for \
                       k in self.data_time.keys()]
            
            if self.save_time_plot:
                label_data = [numpy.array([(pv.label, 
                               float(pv.tag.split(';')[1].split(':')[1][:-2])) \
                                     for pv, _ in self.data_time[k]]) \
                        for k in self.data_time.keys()]
                fig = plt.figure()
                ax1 = plt.subplot(111)
                ax1.yaxis.grid(True, linestyle='-', which='major', color='grey',
                               alpha=0.5)
                ax1.xaxis.grid(True, linestyle='-', which='major', color='grey',
                               alpha=0.5)
                for trial in label_data:
                    ind = numpy.arange(-1.0*(len(trial)-1)*self.sliding_step,
                                                                         50, 50)
                    x = [ind[i] for i, (label, start_time) in enumerate(trial) \
                         if label == self.ir_class]
                    y = [start_time] * len(x)
                    if x == []:
                      plt.plot(ind[0],start_time,'ro')
                    else:
                      plt.plot(x, y,'bo')
                plt.xlabel("Time (ms)")
                plt.ylabel("Start time of trial (s)")
                name = 'trail-time_plot_sp%s.pdf' % self.current_split
                plt.savefig(os.path.join(node_dir,name),dpi=None,facecolor='w',
                            edgecolor='w',orientation='portrait',papertype=None,
                            format=None,transparent=False)  
                plt.close(fig)                
                
            if self.save_score_plot:
                max_time = max([len(trial) for trial in data])
                data_across_time = [[] for time_step in range(max_time)]
                for trial in data:
                    for i, elem in enumerate(trial[::-1]):
                        data_across_time[i].append(elem)
                means = [numpy.mean(time_step) for time_step in data_across_time]
                stds = [numpy.std(time_step) for time_step in data_across_time]
                ind = numpy.arange(-1.0*(max_time-1)*self.sliding_step, 50, 50)
                
                fig = plt.figure()
                ax1 = plt.subplot(111)
                ax1.yaxis.grid(True, linestyle='-', which='major', color='grey',
                               alpha=0.5)
                ax1.xaxis.grid(True, linestyle='-', which='major', color='grey',
                               alpha=0.5)
                width = 30
                plt.bar(ind, means[::-1], width, color='r', yerr=stds[::-1], 
                                                                 ecolor='black')
                ax1.set_xlim(ind[0]-width,ind[-1]+width)
                ax1.set_ylim(min(means)-max(stds),max(means)+max(stds))
                
                t = result_dir.split('/')[-2].strip('{}').split("}{")
                title = "SVM score average: "+str(t[0])+" "+str(t[1:])
                plt.title(title, size=10)
                plt.ylabel('Average SVM prediction value')
                plt.xlabel('Time point of classification [ms]')
                
                params = str(t[0])
                for i in range(2,len(t)):
                    params += str(t[i])
                name = 'average_score_plot_sp%s_%s.pdf' % (self.current_split, 
                                                                         params)
                plt.savefig(os.path.join(node_dir, name), dpi=None, format=None,
                            facecolor='w', edgecolor='w', papertype=None,
                            orientation='portrait', transparent=False)
                plt.close(fig)
                    
            if self.save_trial_plot:
                fig = plt.figure()
                num_trials = len(self.data_time.keys())
                num_rows = num_cols = int(round(numpy.sqrt(num_trials)+0.5))
                fig.set_size_inches((10*num_cols, 6*num_rows))
                ymin = numpy.inf
                ymax = -1.0*numpy.inf
                for trial in data:
                    ymin = min([ymin,min(trial)])
                    ymax = max([ymax,max(trial)])
                for i in range(num_trials):
                    ind = numpy.arange(-1.0*(len(data[i])-1)*self.sliding_step,
                                                                         50, 50)
                    ax = plt.subplot(num_rows, num_cols, i+1)
                    ax.yaxis.grid(True, linestyle='-', which='major', 
                                  color='grey', alpha=0.5)
                    ax.xaxis.grid(True, linestyle='-', which='major', 
                                  color='grey', alpha=0.5)
                    plt.plot(ind, data[i], 'o')
                    ax.set_xlim(ind[0],ind[-1])
                    ax.set_ylim(ymin,ymax)
                    start = self.data_time[i][0][0].tag.split(';')[1].split(':')[1]
                    end = self.data_time[i][-1][0].tag.split(';')[1].split(':')[1]
                    title = "Classification of "+self.data_time[i][-1][0].specs['wdef_name']+" from "+start+" to "+end
                    plt.title(title, size=10)
                    plt.ylabel('SVM prediction value')
                    plt.xlabel('Time point of classification [ms]')
                    if self.label_change_points != []:
                        x = (self.label_change_points[i] - len(data[i]) + 1) \
                                                            * self.sliding_step
                        # plot a line when positive class starts
                        plt.plot([x,x],[ymin,ymax])
                        x_search_low = self.uncertain_area[0][0]
                        x_search_up = self.uncertain_area[0][1]
                        # plot dashed lines to indicate bounds (where the label
                        # change point could be)
                        plt.plot([x_search_low,x_search_low],[ymin,ymax],'r--')
                        plt.plot([x_search_up, x_search_up], [ymin,ymax],'r--')
                #fig.text(0.4, 0.98, "Sliding window classifications; BA: " + \
                #         str(self.performance["Balanced_accuracy"]), size=14)
                name = 'single-trails_score_plot_sp%s.pdf' % self.current_split
                plt.savefig(os.path.join(node_dir,name),dpi=None,facecolor='w',
                            edgecolor='w',orientation='portrait',papertype=None,
                            format=None,transparent=False)
                plt.close(fig)

# Specify special node names
_NODE_MAPPING = {"Classification_Performance_Sink": PerformanceSinkNode,
                 "ClassificationSinkNode": PerformanceSinkNode,
                "Sliding_Window_Performance_Sink": SlidingWindowSinkNode,
                "LOO_Sink":LeaveOneOutSinkNode,
                "LOOSink":LeaveOneOutSinkNode,
                }
