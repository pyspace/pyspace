# This Python file uses the following encoding: utf-8
# The upper line is needed for one comment in this module.
# coding=utf-8
""" Methods to calculate and store classification results (metrics)

Several performance measures are supported.

To combine and visualize them, use the
:class:`~pySPACE.resources.dataset_defs.performance_result.PerformanceResultSummary`.

For details concerning parameters in metric calculation, have a look at
:class:`~pySPACE.missions.nodes.sink.classification_performance_sink.PerformanceSinkNode`.
"""
from collections import defaultdict
import csv
import logging
from math import cos, pi, sqrt, exp
import os
import warnings
import numpy
from pySPACE.resources.dataset_defs.base import BaseDataset

class metricdict(defaultdict):
    """ Interface to dictionaries of metrics """
    def __missing__(self,new_key):
        """ Return first occurring fitting entry and give warning, if functional metric is called without parameters """
        for key in self.keys():
            try:
                if key.startswith(new_key+"("):
                    import warnings
                    warnings.warn("Using key: '%s' instead of: '%s'."%(key,new_key))
                    return self[key]
            except:
                pass
        return super(metricdict, self).__missing__(new_key)


class BinaryClassificationDataset(BaseDataset):
    """ Handle and store binary classification performance measures

    This class derived from BaseDataset overwrites the 'store' and
    'add_split' method from the BaseDataset class so that it can
    handle and store classification performance measures to files.

    In the following there is a list of implemented metrics.
    After giving the normal name or abbreviation, the name in the final
    results file/dictionary is given.
    This is for example needed for parameter optimization algorithms.

    .. todo:: Move metrics to external rst file for better linking and
              summarize it with multinomial metrics.

    .. _metrics:

    **Metrics**

      :confusion matrix components:
        :TP - True_positives:
            correct classified examples or the *ir_class* (positive examples)

        :TN - True_negatives:
            correct classified examples or the *ir_class* (negative examples)

        :FN - False_negatives:
            wrong classified positive examples (classified as negative examples)

        :FP - False_positives:
            wrong classified negative examples (classified as positive examples)

      :confusion matrix metrics:
        :TPR - True_positive_rate:
            true positive rate, recall

            .. math:: \\frac{TP}{TP+FN}

        :PPV - IR_precision:
            positive predictive value, precision

            .. math:: \\frac{TP}{TP+FP}

        :TNR - True_negative_rate:
            true negative rate, specificity

            .. math:: \\frac{TN}{TN+FP}

        :NPV - Non_IR_precision:
            negative predictive value

            .. math:: \\frac{TN}{TN+FN}

        :FPR - False_positive_rate:
            false positive rate

            .. math:: 1-TNR = \\frac{FP}{TN+FP}

        :FNR - False_negative_rate:
            false negative rate

            .. math:: 1-TPR = \\frac{FN}{TP+FN}

        :accuracy - Percent_correct:
            rate of correct classified examples (sometimes percent correct)

            .. math:: \\frac{TP+TN}{TN+FP+FN+TP}

        :misclassification rate - Percent_incorrect:
            error rate, (sometimes percent incorrect)

            .. math:: \\frac{FP+FN}{TN+FP+FN+TP}

        :F-Measure - F_measure:
            harmonic mean of TNR and NPV

            .. math:: \\frac{2 \\cdot PPV \\cdot TPR}{PPV+TPR}=\\frac{2}{\\frac{1}{PPV}+\\frac{1}{TPR}}

        :F-neg-measure - Non_IR_F_measure:
            F-measure for negative class

            .. math:: \\frac{2 \\cdot NPV\\cdot TNR}{NPV+TNR}

        :Weighted F-measure - not implemented yet:
            .. math:: \\text{lambda } x: \\frac{(1+x^2)\\cdot PPV\\cdot TPR}{x^2 \\cdot PPV+TPR}

        :Weighted accuracy (t) - Weighted_accuracy(t):
            .. math:: t\\cdot TPR + (1-t)\\cdot TNR

        :ROC-measure:
            .. math:: \\sqrt{\\frac{TPR^2+TNR^2}{2}}

        :balanced accuracy - Balanced_accuracy:
            .. math:: \\frac{TNR + TPR}{2}

        :Gmean:
            .. math:: \\sqrt{(TPR \\cdot TNR)}

        :AUC:
            The area under the Receiver Operator Characteristic. Equal to the
            Wilcoxon test of ranks or to the probability, that a classifier will
            rank a randomly chosen positive instance higher than a randomly
            chosen negative one.

        :MCC-  Matthews_correlation_coefficient:

            .. math:: \\frac{TP*TN-FP*FN}{\\sqrt{((TP+FN)*(TP+FP)*(TN+FN)*(TN+FP))}}

        :Cohen's kappa-Kappa:
            Measures the agreement between classifier and true class
            with a correction for guessing

            .. math:: \\frac{TP+TN-\\left(
                    \\frac{P(TP+FP)}{P+N}+\\frac{N(TN+FN)}{P+N}
                \\right)}{P+N-\\left(
                    \\frac{P(TP+FP)}{P+N}+\\frac{N(TN+FN)}{P+N}
                \\right)}

    **K-metrics**
        These metrics expect classification values between zero and one.
        Instead of calculating the number of correct classifications,
        the corresponding sums of classification values are built.
        The misclassification values we get, by using one minus c-value.
        This also defines a confusion matrix, which is used to calculate the
        upper metrics.

        the notation is *k_* + normal name of metric.

    **Loss metrics**
        Some classifiers like LDA, SVM and RMM have loss terms in there model
        description. These misclassification values can be also calculated
        on test data, to evaluate the algorithm.

        The longest name used is *loss_balanced_rest_L1_SVM*
        and the shortest is *loss_L2*.

        In the LDA case, you skip the *SVM* component.
        If you want to weight the losses equally and not consider class imbalance,
        skip the *balanced* component and
        if you do not want to restrict the maximum loss, delete the *rest* component.

        The parameters *calc_loss* and *loss_restriction* can be specified.

    .. todo::   soft and pol metrics have to be checked

    **Parameters**

        :dataset_md:
            The meta data of the current input

            (*optional, default: None*)

    :Author: Mario Krell (mario.krell@dfki.de)
    :Created: 2010/04/01

    """
    def __init__(self, dataset_md = None, dataset_pattern=None):
        #: The data structure containing the actual data.
        #:
        #: The data is stored as a dictionary that maps
        #: (run, split, train/test) tuple to the actual
        #: data obtained in this split in this run for
        #: training/testing.
        self.data = dict()
        self.dataset_pattern = dataset_pattern

        self.meta_data = {"train_test": False,
                          "splits": 1,
                          "runs": 1} #: A dictionary containing some default meta data for the respective dataset

    def store(self, result_dir, s_format = "csv"):
        """ Handle meta data and meta information and save result as csv table

        This table is later on merged with the other results
        to one big result table.

        .. todo:: Try to use *PerformanceResultSummary* or *csv_analysis* methods and
                  furthermore sort the keys.
        """
        name = "results"
        if not s_format == "csv":
            self._log("The format %s is not supported! Using default."%s_format, level=logging.CRITICAL)
        for key, performance in self.data.iteritems():
            # Construct result directory
            result_path = result_dir
            final_path = os.path.join(os.sep.join(result_path.split(os.sep)[:-1]))
            if not os.path.exists(result_path):
                os.mkdir(result_path)
            key_str = "_r%s_sp%s_%s_" % key[0:] # run number, split number, if performance on training or test data
            key_str += result_path.split(os.sep)[-1] # set name used for identifier
            result_file = open(os.path.join(final_path,
                                            name + key_str + ".csv"),
                               "w") # Data saved in operation directory instead of current set directory

            performance["Key_Dataset"]=result_path.split(os.sep)[-1]
            results_writer = csv.writer(result_file)

            # if dataset_pattern given, add keys/vals
            if self.dataset_pattern is not None:
                try:
                    current_dataset = (performance["Key_Dataset"].strip("'}{")).split("}{")[0]
                    new_keys = self.dataset_pattern.split('_')
                    # make '__MyNiceKey__' from 'myNiceKey':
                    new_keys=map(lambda x: '__'+x[0].upper()+x[1:]+'__', new_keys)
                    new_vals = current_dataset.split('_')
                    for new_key_index in range(len(new_keys)):
                        performance[new_keys[new_key_index]] = new_vals[new_key_index]
                except IndexError:
                    warnings.warn("Using wrong dataset pattern '%s' on '%s'!"
                        %(self.dataset_pattern,current_dataset))
                    self.dataset_pattern = None
            results_writer.writerow(performance.keys())
            results_writer.writerow(performance.values())
            result_file.close()
        #Store meta data
        BaseDataset.store_meta_data(result_dir,self.meta_data)

    def add_split(self, performance, train, split = 0, run = 0):
        """ Add a split to this dataset

        The method expects the following parameters:

        **Parameters**

            :performance:
                dictionary of performance measures

            :train:
                If train is True, this sample has already been used for training.

            :split:
                The number of the split this sample belongs to.

                (*optional, default: 0*)

            :run:
                The run number this performance belongs to.

                (*optional, default: 0*)

        """
        if train == True:
            self.meta_data["train_test"] = True
        if split + 1 > self.meta_data["splits"]:
            self.meta_data["splits"] = split + 1
        if run + 1 > self.meta_data["runs"]:
            self.meta_data["runs"] = run + 1
        key =  (run, split, "train" if train else "test")
        performance["__Key_Run__"]=run
        performance["__Key_Fold__"]=split+1
        performance["__Run__"]="Run_"+str(run)
        performance["__Split__"]="__Split_"+str(split+1)
        self.data[key] = performance

    def merge_splits(self):
        """ Replace performances of different splits by just one performance value

        Performances of confusion matrix metrics are calculated by summing
        up the confusion matrix entries.
        The other metrics are averaged.

        This method is the preparation of the merge_performance method.
        """
        temp_data=dict()
        for key in self.data.keys():
            new_key = (key[0],0,key[2])
            if not temp_data.has_key(new_key):
                temp_data[new_key] = [self.data[key]]
            else:
                temp_data[new_key].append(self.data[key])
        del(self.data)
        self.data=dict()
        for key in temp_data.keys():
            self.data[key] = self.merge_performance(temp_data[key])

    def merge_performance(self,p_list):
        """ Replace performances of different splits by just one performance value

        Performances of confusion matrix metrics are calculated by summing
        up the confusion matrix entries.
        The other metrics are averaged.
        """
        new_p_dict = dict()
        new_p_dict["True_positives"] = 0
        new_p_dict["True_negatives"] = 0
        new_p_dict["False_positives"] = 0
        new_p_dict["False_negatives"] = 0

        new_p_dict["train_True_positives"] = 0
        new_p_dict["train_True_negatives"] = 0
        new_p_dict["train_False_positives"] = 0
        new_p_dict["train_False_negatives"] = 0

        for p in p_list:
            new_p_dict["True_positives"] += p["True_positives"]
            new_p_dict["True_negatives"] += p["True_negatives"]
            new_p_dict["False_positives"] += p["False_positives"]
            new_p_dict["False_negatives"] += p["False_negatives"]

            new_p_dict["train_True_positives"] += p["train_True_positives"]
            new_p_dict["train_True_negatives"] += p["train_True_negatives"]
            new_p_dict["train_False_positives"] += p["train_False_positives"]
            new_p_dict["train_False_negatives"] += p["train_False_negatives"]

        old_p_keys = p.keys()
        
        BinaryClassificationDataset.calculate_confusion_metrics(new_p_dict,"")
        BinaryClassificationDataset.calculate_confusion_metrics(new_p_dict,"train_")
        new_p_dict["__Key_Fold__"] = 1
        new_p_dict["__Split__"] = "Split_1"
        new_p_dict["__Key_Run__"] = p_list[0]["__Key_Run__"]
        new_p_dict["__Run__"] = p_list[0]["__Run__"]
        new_p_keys = new_p_dict.keys()
        for key in old_p_keys:
            if not key in new_p_keys:
                value_list=[]
                for p in p_list:
                    value_list.append(p[key])
                if not (type(value_list[0])==str or key =="__Key_Fold__" or key == "~~Num_Retained_Features~~"):
                    new_p_dict[key] = numpy.mean(value_list)
                elif key == "~~Num_Retained_Features~~":
                    if 'unknown' in value_list:
                        new_p_dict[key] = "unknown"
                    else:
                        new_p_dict[key] = numpy.mean(value_list)
                else:
                    self._log("Unknown performance entry:%s!"%key,level=logging.WARNING)
                    new_p_dict[key] = value_list[-1]
        return new_p_dict

    def get_average_performance(self, metric):
        """ Returns the average performance for the given metric """
        metric_values = []
        for value in self.data.itervalues():
            metric_values.append(value[metric])
        return sum(metric_values) / len(metric_values)

    def get_performance_std(self, metric):
        """ Returns the average performance for the given metric """
        metric_values = []
        for value in self.data.itervalues():
            metric_values.append(value[metric])
        return numpy.array(metric_values).std()

    def get_unified_confusion_matrix_performance(self, metric):
        """ Confusion metrics from the splits altogether """
        metric_values = []
        for value in self.data.itervalues():
            metric_values.append(value[metric])
        return sum(metric_values) / len(metric_values)

    @staticmethod
    def calculate_metrics(classification_results,calc_soft_metrics=True,
                          invert_classification=False,
                          ir_class="Target", sec_class=None,
                          loss_restriction=2.0,time_periods=[],
                          calc_AUC=True,calc_loss=True,
                          weight=0.5, save_roc_points=False,
                          decision_boundary=0.0):
        """ Calculate performance measures from the given classifications

        :Returns: metricdict and the ROC points if save_roc_point is True

        .. todo:: simplify loss metrics, mutual information and AUC
        """
        # metric initializations
        metrics = metricdict(float) #{"TP":0,"FP":0,"TN":0,"FN":0}
        # loss values are collected for each class
        # there are numerous different losses
        loss_dict = metricdict(lambda: numpy.zeros(2))
        for prediction_vector, label in classification_results:
            if sec_class is None and not label == ir_class:
                sec_class = label
            # special treatment for one vs REST evaluation
            if sec_class == "REST" and not label == ir_class:
                label = "REST"
            if not label == ir_class and not label == sec_class:
                warnings.warn("Binary metrics " \
                        "require exactly two classes. At least " \
                        "three are used:" + str(ir_class) + \
                        " (ir_class), " + str(sec_class) + \
                        " (non_ir_class), " + str(label) + \
                        " (occured true label), " + \
                        str(prediction_vector.label) + \
                        " (prediction vector label)! \n"+\
                        "Did you specify the ir_class in your sink node?\n"+\
                        "Replacing the ir_class by: " +str(label)+".")
                ir_class = label
            BinaryClassificationDataset.update_confusion_matrix(prediction_vector,
                                         label,calc_soft_metrics=calc_soft_metrics,
                                         ir_class=ir_class, sec_class=sec_class,
                                         confusion_matrix=metrics,
                                         decision_boundary=decision_boundary)
            if calc_loss:
                BinaryClassificationDataset.update_loss_values(classification_vector=prediction_vector,
                                    label=label,
                                    ir_class=ir_class, sec_class=sec_class,
                                    loss_dict=loss_dict,
                                    loss_restriction=loss_restriction)

        P = metrics["True_positives"]+metrics["False_negatives"]
        N = metrics["True_negatives"]+metrics["False_positives"]

        if calc_soft_metrics:
            prefixes = ["","soft_","pol_","k_"]
            metrics["k_True_negatives"]  = (N- metrics["k_False_positives"])
            metrics["k_False_negatives"] = (P-metrics["k_True_positives"])
        else:
            prefixes = [""]

        ### Confusion matrix metrics
        for prefix in prefixes:
            BinaryClassificationDataset.calculate_confusion_metrics(metrics, pre=prefix, P=P, N=N, weight=weight)

        ### Mutual information ###
        # Mutual information is also a confusion matrix metric but makes no
        # sense for soft metrics
        try:
            # Add mutual information between classifier output Y and the target
            metrics["Mutual_information"] = \
                    BinaryClassificationDataset.mutual_information(
                                       metrics["True_negatives"],
                                       metrics["False_negatives"],
                                       metrics["True_positives"],
                                       metrics["False_positives"])
            # Add normalized mutual information (a perfect classifier would achieve
            # metric 1)
            metrics["Normalized_mutual_information"] = \
                    BinaryClassificationDataset.normalized_mutual_information(
                                                  metrics["True_negatives"],
                                                  metrics["False_negatives"],
                                                  metrics["True_positives"],
                                                  metrics["False_positives"])
        except:
            warnings.warn("Mutual Information could not be calculated!")
            metrics["Mutual_information"] = 0
            metrics["Normalized_mutual_information"] = 0

        ## Get AUC and ROC_points
        # test if classification_outcome has prediction (float, score)
        ROC_points = None
        if len(classification_results) != 0 and calc_AUC:
            AUC, ROC_points = BinaryClassificationDataset.calculate_AUC(classification_results,
                                             ir_class=ir_class,
                                             save_roc_points=save_roc_points,
                                             performance=metrics,
                                             inverse_ordering=invert_classification)
            # If classification missed the ordering of the two classes or
            # used information in the wrong way prediction should be switched.
            # This results in the *inverse* AUC.
            if AUC < 0.5:
                AUC = 1 - AUC
                if AUC>0.6:
                    warnings.warn("AUC had to be inverted! Check this!")
            metrics["AUC"] = AUC

        ### Extract meta metrics from the predictor ###
        # set basic important predictor metrics for default
        #metrics["~~Num_Retained_Features~~"] = numpy.inf
        #metrics["~~Solver_Iterations~~"] = numpy.Inf
        #metrics["~~Classifier_Converged~~"] = True
        # Classifier information should be saved in the parameter
        # 'classifier_information'!!!
        try:
            classifier_information = classification_results[0][0].predictor.classifier_information
            for key, value in classifier_information.iteritems():
                metrics[key] = value
        except:
            pass

        ### Time metrics ###
        if len(time_periods)>0:
            # the first measured time can be inaccurate due to
            # initialization procedures performed in the first executions
            time_periods.pop(0)
            metrics["Time (average)"] = 1./1000 * sum(time_periods) / \
                                                        len(time_periods)
            metrics["Time (maximal)"] = 1./1000 * max(time_periods)

        ### Loss metrics ###
        if calc_loss:
            # initialization #
            n = (P+N)*1.0
            if n==0.0:
                n = 1.0
            if P==0:
                P=1
            if N==0:
                N=1
            # unbalanced losses
            metrics["loss_L1_SVM"] = (loss_dict["SVM_L1_loss"][0] + \
                                      loss_dict["SVM_L1_loss"][1])/n
            metrics["loss_L2_SVM"] = (loss_dict["SVM_L2_loss"][0] + \
                                      loss_dict["SVM_L2_loss"][1])/n
            metrics["loss_L1"] = (loss_dict["L1_loss"][0] + \
                                  loss_dict["L1_loss"][1])/n
            metrics["loss_L2"] = (loss_dict["L2_loss"][0] + \
                                  loss_dict["L2_loss"][1])/n
            metrics["loss_L1_RMM"] = (loss_dict["RMM_L1_loss"][0] + \
                                      loss_dict["RMM_L1_loss"][1])/n
            metrics["loss_L2_RMM"] = (loss_dict["RMM_L2_loss"][0] + \
                                      loss_dict["RMM_L2_loss"][1])/n
            metrics["loss_restr_L1_SVM"] = (loss_dict["SVM_L1_loss_restr"][0] + \
                                            loss_dict["SVM_L1_loss_restr"][1])/n
            metrics["loss_restr_L2_SVM"] = (loss_dict["SVM_L2_loss_restr"][0] + \
                                            loss_dict["SVM_L2_loss_restr"][1])/n
            metrics["loss_restr_L1"] = (loss_dict["L1_loss_restr"][0] + \
                                        loss_dict["L1_loss_restr"][1])/n
            metrics["loss_restr_L2"] = (loss_dict["L2_loss_restr"][0] + \
                                        loss_dict["L2_loss_restr"][1])/n
            metrics["loss_restr_L1_RMM"] = (loss_dict["RMM_L1_loss_restr"][0] + \
                                            loss_dict["RMM_L1_loss_restr"][1])/n
            metrics["loss_restr_L2_RMM"] = (loss_dict["RMM_L2_loss_restr"][0] + \
                                            loss_dict["RMM_L2_loss_restr"][1])/n
            # balanced losses
            metrics["loss_balanced_L1_SVM"] = (loss_dict["SVM_L1_loss"][0]/N + \
                                               loss_dict["SVM_L1_loss"][1]/P)/2
            metrics["loss_balanced_L2_SVM"] = (loss_dict["SVM_L2_loss"][0]/N + \
                                               loss_dict["SVM_L2_loss"][1]/P)/2
            metrics["loss_balanced_L1"] = (loss_dict["L1_loss"][0]/N + \
                                           loss_dict["L1_loss"][1]/P)/2
            metrics["loss_balanced_L2"] = (loss_dict["L2_loss"][0]/N + \
                                           loss_dict["L2_loss"][1]/P)/2
            metrics["loss_balanced_L1_RMM"] = (loss_dict["RMM_L1_loss"][0]/N + \
                                               loss_dict["RMM_L1_loss"][1]/P)/2
            metrics["loss_balanced_L2_RMM"] = (loss_dict["RMM_L2_loss"][0]/N + \
                                               loss_dict["RMM_L2_loss"][1]/P)/2
            metrics["loss_balanced_restr_L1_SVM"] = (loss_dict["SVM_L1_loss_restr"][0]/N + \
                                                     loss_dict["SVM_L1_loss_restr"][1]/P)/2
            metrics["loss_balanced_restr_L2_SVM"] = (loss_dict["SVM_L2_loss_restr"][0]/N + \
                                                     loss_dict["SVM_L2_loss_restr"][1]/P)/2
            metrics["loss_balanced_restr_L1"] = (loss_dict["L1_loss_restr"][0]/N + \
                                                 loss_dict["L1_loss_restr"][1]/P)/2
            metrics["loss_balanced_restr_L2"] = (loss_dict["L2_loss_restr"][0]/N + \
                                                 loss_dict["L2_loss_restr"][1]/P)/2
            metrics["loss_balanced_restr_L1_RMM"] = (loss_dict["RMM_L1_loss_restr"][0]/N + \
                                                     loss_dict["RMM_L1_loss_restr"][1]/P)/2
            metrics["loss_balanced_restr_L2_RMM"] = (loss_dict["RMM_L2_loss_restr"][0]/N + \
                                                     loss_dict["RMM_L2_loss_restr"][1]/P)/2

        if save_roc_points:
            return metrics, ROC_points
        else:
            return metrics

    @staticmethod
    def update_confusion_matrix(classification_vector, label,calc_soft_metrics=False,
                                ir_class='Target', sec_class='Standard',
                                confusion_matrix=metricdict(float),
                                decision_boundary=0.0, scaling=5):
        """ Calculate the change in the 4 basic metrics: TP, FP, TN, FN

        +--------------+----+-----+
        | class|guess  | ir | sec |
        +==============+====+=====+
        | ir_class     | TP | FN  |
        +--------------+----+-----+
        | sec_class    | FP | TN  |
        +--------------+----+-----+

        The change is directly written into the confusion matrix dictionary.

        :Returns: confusion_matrix
        """
        p_label = classification_vector.label.strip()
        label = label.strip()

        # prepare prediction in case of no mapping beforehand
        prediction = classification_vector.prediction
        if decision_boundary==0.0: # if mapping was before, this should be around 0.5
            # ir_class>0; sec_class<0
            if (p_label == ir_class and not prediction>=0) or \
                (p_label == sec_class and not prediction<=0):
                prediction*=-1.0
            if p_label == sec_class:
                prediction*=-1.0
            if not p_label == label: # negative values in case of wrong classification
                prediction*=-1.0


        # true positive
        if p_label == ir_class and p_label == label:
            confusion_matrix["True_positives"] += 1
            if calc_soft_metrics:
                confusion_matrix["soft_True_positives"] += \
                        BinaryClassificationDataset.scale(classification_vector.prediction,
                                  decision_boundary=decision_boundary)
                confusion_matrix["pol_True_positives"] += \
                        BinaryClassificationDataset.pol(classification_vector.prediction,
                                decision_boundary=decision_boundary)
                confusion_matrix["k_True_positives"] += \
                        BinaryClassificationDataset.k_sig(prediction,
                                decision_boundary=decision_boundary,
                                scaling=scaling)
        # false positive
        elif p_label == ir_class and not(p_label == label):
            confusion_matrix["False_positives"] +=1
            if calc_soft_metrics:
                confusion_matrix["soft_False_positives"] += \
                        BinaryClassificationDataset.scale(classification_vector.prediction,
                                  decision_boundary=decision_boundary)
                confusion_matrix["pol_False_positives"] += \
                        BinaryClassificationDataset.pol(classification_vector.prediction,
                                decision_boundary=decision_boundary)
                confusion_matrix["k_False_positives"] += 1- \
                         BinaryClassificationDataset.k_sig(prediction,
                                decision_boundary=decision_boundary,
                                scaling=scaling)
        # false negative
        elif p_label == sec_class and not(p_label == label):
            confusion_matrix["False_negatives"] +=1
            if calc_soft_metrics:
                confusion_matrix["soft_False_negatives"] += \
                        BinaryClassificationDataset.scale(classification_vector.prediction,
                                  decision_boundary=decision_boundary)
                confusion_matrix["pol_False_negatives"] += \
                        BinaryClassificationDataset.pol(classification_vector.prediction,
                                decision_boundary=decision_boundary)
                #prediction negative/wrong--> low value added
                confusion_matrix["k_True_positives"] += \
                        BinaryClassificationDataset.k_sig(prediction,
                                decision_boundary=decision_boundary,
                                scaling=scaling)
        # true negative
        elif p_label == sec_class and p_label == label:
            confusion_matrix["True_negatives"] +=1
            if calc_soft_metrics:
                confusion_matrix["soft_True_negatives"] += \
                        BinaryClassificationDataset.scale(classification_vector.prediction,
                                  decision_boundary=decision_boundary)
                confusion_matrix["pol_True_negatives"] += \
                        BinaryClassificationDataset.pol(classification_vector.prediction,
                                decision_boundary=decision_boundary)
                # prediction is positive --> low value subtracted--> nearly 1
                confusion_matrix["k_False_positives"] += 1- \
                         BinaryClassificationDataset.k_sig(prediction,
                                decision_boundary=decision_boundary,
                                scaling=scaling)
        else:
            raise Exception("Updating confusion matrix " \
                        "requires exactly two classes. At least " \
                        "three are used:" + str(ir_class) + \
                        " (ir_class), " + str(sec_class) + \
                        " (non_ir_class), " + str(label) + \
                        " (correct label), " + \
                        str(classification_vector.label) + \
                        " (classification)! \n"+\
                        "Did you specify the ir_class in your sink node?")
        return confusion_matrix

    @staticmethod
    def scale(value, decision_boundary=0.0):
        """ Scales the prediction output to [0,1] by simple cutting
        to show there reliability
        contribution in the prediction.
        """
        if decision_boundary==0.0:
            if value>0:
                output = value
            else:
                output = -value
            if output > 1:
                output = 1
            return output
        else: #probabilistic output assumed
            if value>decision_boundary:
                output = value
            else:
                output = 1-value
            if output > 1:
                output = 1
            return output

    @staticmethod
    def sig(value, decision_boundary=0.0):
        """ Scales the prediction output to [0,1] SMOOTH with a sinusoid function
        to show there reliability
        contribution in the prediction.

        Therefore it uses the sinusoid sigmoid function

        .. math::  0.5\\cdot (1-cos(value\\cdot \\pi))
        """
        if value>0:
            output = value
        else:
            output = -value
        if output > 1:
            output = 1
        else:
            output = 0.5*(1-cos(output*pi))
        return output

    @staticmethod
    def pol(value, decision_boundary=0.0):
        """ Scales the prediction output to [0,1] SMOOTH with a polynomial function
        to show there reliability
        contribution in the prediction.

        Therefore it uses the polynomial sigmoid function

        .. math:: value^2 (3-2 \\cdot value)
        """
        if value>0.5:
            output = value
        else:
            output = 1-value
        if output > 1:
            output = 1
        else:
            output = output**2*(3-2*output)
        return output

    @staticmethod
    def k_sig(value, decision_boundary=0.0, scaling=5):
        """ Scaling as in Keerthi 2006 for smooth target function

        "An efficient method for gradient-based adaptation of
        hyperparameters in SVM models"
        Keerthi, S. Sathiya; Sindhwani, Vikas; Chapelle, Olivier
        """
        if not decision_boundary==0.0: # no mapping needed, due to prob-fit
            return value
        else:
            return 1.0/(1+exp(-1.0*scaling*value))

    @staticmethod
    def update_loss_values(classification_vector, label,
                           ir_class="Target", sec_class="Standard",
                           loss_dict=metricdict(lambda: numpy.zeros(2)),
                           loss_restriction=2.0):
        """ Calculate classifier loss terms on test data

        Different classifiers mapping the ir_class to 1 and the other
        class to -1 try to minimize a loss term in the classification.
        For some used loss terms of least squares classifiers and SVMs
        the corresponding value is calculated as a metric to be later on used
        for optimization.
        """
        if label==ir_class:
            prediction = classification_vector.prediction
        else:
            prediction = - classification_vector.prediction
        if label ==ir_class:
            i=1
        else:
            i=0

        try:
            loss_dict["L1_loss"][i] += abs(prediction-1)
            loss_dict["L2_loss"][i] += (prediction-1)**2
            loss_dict["L1_loss_restr"][i] += min(abs(prediction-1),loss_restriction)
            loss_dict["L2_loss_restr"][i] += min(abs(prediction-1),loss_restriction)**2

            try:
                R = classification_vector.predictor.range
            except:
                R = numpy.inf

            if prediction > R:
                loss_dict["RMM_L1_loss"][i] += prediction-R
                loss_dict["RMM_L2_loss"][i] += (prediction-R)**2
                loss_dict["RMM_L1_loss_restr"][i] += min(prediction-R,loss_restriction)
                loss_dict["RMM_L2_loss_restr"][i] += min(prediction-R,loss_restriction)**2
            elif prediction > 1:
                pass
                #self.RMM_L1_loss += 0
                #self.RMM_L2_loss += 0
            else:
                loss_dict["SVM_L1_loss"][i] += 1-prediction
                loss_dict["SVM_L2_loss"][i] += (1-prediction)**2
                loss_dict["SVM_L1_loss_restr"][i] += min(1-prediction,loss_restriction)
                loss_dict["SVM_L2_loss_restr"][i] += min(1-prediction,loss_restriction)**2
                loss_dict["RMM_L1_loss"][i] += 1-prediction
                loss_dict["RMM_L2_loss"][i] += (1-prediction)**2
                loss_dict["RMM_L1_loss_restr"][i] += min(1-prediction,loss_restriction)
                loss_dict["RMM_L2_loss_restr"][i] += min(1-prediction,loss_restriction)**2
        except:
            pass

    @staticmethod
    def calculate_confusion_metrics(performance, pre="", P=None,
                                    N=None, weight=0.5):
        """ Calculate each performance metric resulting from the 4 values in the confusion matrix and return it.

        This helps to use soft metrics, generating the confusion matrix
        in a different way.

        .. warning::    Still the number of positive and negative instances
                        had to be used for the calculation of rates with soft metrics.

        :Returns: metricdict

        .. note:: If the input is a metricdict the new calculated entries are added to it.
        """
        TN = performance[pre+"True_negatives"] * 1.0
        TP = performance[pre+"True_positives"] * 1.0
        FP = performance[pre+"False_positives"] * 1.0
        FN = performance[pre+"False_negatives"] * 1.0
        if not type(performance) == metricdict:
            old_p = performance
            performance = metricdict(float)
            performance.update(old_p)
        if P is None:
            P = TP + FN
        if N is None:
            N = TN + FP

        performance[pre+"Positives"] = P
        performance[pre+"Negatives"] = N

        if TP == 0:
            TPR = 0
            PPV = 0
        else:
            # sensitivity, recall
            TPR = 1.0 * TP / P  #(TP+FN) = Num of positive examples
            # positive predictive value, precision
            PPV = 1.0 * TP / (TP + FP)
        if TN == 0:
            TNR = 0
            NPV = 0
        else:
            TNR = 1.0 * TN / N  # specificity #  Num of negative examples
            NPV = 1.0 * TN / (TN + FN)  # negative predictive value
        FPR = 1 - TNR  # 1.0*FP/(TN+FP) # 1-TNR
        FNR = 1 - TPR  # 1.0*FN/(TP+FN) # 1-TPR

        if P+N == 0:
            accuracy = 0.0
            missclassification_rate = 1.0
            warnings.warn("No examples given for performance calculation!")
        else:
            accuracy = 1.0 * (TP+TN) / (N+P) # Num of all examples
            missclassification_rate = 1.0 * (FP+FN) / \
                                             (N+P) # s.a.
        if (PPV+TPR) == 0:
            F_measure = 0
        else:
            F_measure = 2.0 * PPV * TPR / (PPV + TPR)
        if (NPV+TNR) == 0:
            F_neg_measure = 0
        else:
            F_neg_measure = 2.0 * NPV * TNR / (NPV + TNR)

        den = (TP + FN) * (TP + FP) * (TN + FN) * (TN + FP)
        if den <= 0:
            den = 1
        MCC = (TP * TN - FP * FN) / numpy.sqrt(den)

        try:
            guessing = (P * (TP + FP) + N * (TN + FN)) / (P + N)
            kappa = (TP + TN - guessing)/(P + N - guessing)
        except:
            kappa = 0

        # weighted_F_measure = lambda x: (1+x**2)*PPV*TPR/(x**2*PPV+TPR)
        performance[pre+"True_positive_rate"] = TPR
        performance[pre+"False_positive_rate"] = FPR
        performance[pre+"True_negative_rate"] = TNR
        performance[pre+"False_negative_rate"] = FNR
        performance[pre+"IR_precision"] = PPV
        performance[pre+"IR_recall"] = TPR
        performance[pre+"F_measure"] = F_measure
        performance[pre+"Non_IR_F_measure"] = F_neg_measure
        performance[pre+"Non_IR_precision"] = NPV
        performance[pre+"Percent_correct"] = accuracy*100
        performance[pre+"Percent_incorrect"] = missclassification_rate * 100
        performance[pre+"Weighted_accuracy("+str(weight)+")"] = \
            weight * TPR + (1 - weight) * TNR
        performance[pre+"ROC-measure"] = sqrt(0.5 * (TPR**2 + TNR**2))
        performance[pre+"Balanced_accuracy"] = 0.5 * (TNR + TPR)
        performance[pre+"Gmean"] = sqrt(abs(TPR * TNR))
        performance[pre+"Matthews_correlation_coefficient"] = MCC
        performance[pre+"Correct_classified"] = TP + TN
        performance[pre+"Wrong_classified"] = FP + FN
        performance[pre+"Kappa"] = kappa
        return performance

    @staticmethod
    def calculate_AUC(classification_outcome, ir_class, save_roc_points,
                      performance,inverse_ordering=False):
        """ AUC and ROC points by an algorithm from Fawcett, "An introduction to ROC analysis", 2005
        Also possible would be to calculate the Mann-Whitney-U-Statistik

        .. math:: \\sum_i^m{\\sum_j^n{S(X_i,Y_i)}} \\text{ with } S(X,Y) = 1 \\text{ if } Y < X\\text{, otherwise } 0

        """
        # need sorted list, decreasing by the prediction score
        from operator import itemgetter
        sorted_outcome = sorted(classification_outcome,
                                key=itemgetter(0), reverse=not inverse_ordering)
        P = performance["Positives"] # number of True instances
        N = performance["Negatives"] # number of False instances

        FP = 0
        TP = 0
        FP_prev = 0
        TP_prev = 0

        AUC = 0
        # first, list of roc points, second, the weka-roc-point
        R = ([],[(0.0,0.0),(performance["False_positive_rate"],
                performance["True_positive_rate"]),(1.0,1.0)])
        axis_change = True
        axis_y = False
        axis_x = False

        prediction_prev = -float("infinity")

        def _trapezoid_area(x1, x2, y1, y2):
            base = abs(x1-x2)
            height_avg = (y1+y2)/2.0
            return base * height_avg

        for classification_outcome in sorted_outcome:
            if round(classification_outcome[0],3) != prediction_prev:
                AUC += _trapezoid_area(FP, FP_prev, TP, TP_prev)
                prediction_prev = round(classification_outcome[0],3)
                if save_roc_points and axis_change:
                    R[0].append((1.0*FP_prev/N,1.0*TP_prev/P))
                    axis_change = False
                FP_prev = FP
                TP_prev = TP

            # if actual instance is a true / ir class example
            if classification_outcome[1].strip() == ir_class:
                TP += 1
                axis_y = True
                if axis_x == True:
                    axis_change = True
                    axis_x = False
            else: # instance is a false / sec class example
                FP += 1
                axis_x = True
                if axis_y == True:
                    axis_change = True
                    axis_y = False

        if save_roc_points and axis_change:
            R[0].append((1.0*FP_prev/N,1.0*TP_prev/P))

        AUC += _trapezoid_area(N, FP_prev, P, TP_prev)
        try:
            AUC = float(AUC) / (P*N) # scale from (P*N) to the unit square
            if save_roc_points:
                R[0].append((1.0*FP/N,1.0*TP/P)) # This is (1,1)
        except ZeroDivisionError:
            if P == 0:
                warnings.warn("AUC could no be computed since there are no "
                              "positive examples.")
            else:
                warnings.warn("AUC could no be computed since there are no "
                              "negative examples.")
        return AUC, R

    @staticmethod
    def mutual_information(TN, FN, TP, FP):
        """ Computes the mutual information metric I(T;Y) = H(T) - H(T|Y)

        Measures the mutual information between the classifier output Y
        and the target (the true label T), i.e. how many bits the classifier's
        output conveys about the target. H denotes the entropy function.
        """
        # Convert to float
        TN = float(TN)
        FN = float(FN)
        TP = float(TP)
        FP = float(FP)
        P = TP + FN # positive examples
        N = FP + TN # negative examples
        K = TP+FP+TN+FN # Total number of examples

        def term(y, t):
            if y: # prediction is positive
                p_y = (TP + FP) / K # ratio of positive predictions
                if p_y == 0.0:
                    p_t_y = 1 # Doesn't matter anyway since multiplied with 0
                elif t: # actually a positive
                    p_t_y = TP / (TP + FP) # ratio of true positives
                else: # actually a negative
                    p_t_y = FP / (TP + FP)# ratio of false positives
            else: # prediction is negative
                p_y = (TN + FN) / K # ratio of negative predictions
                if p_y == 0.0:
                    p_t_y = 1 # Doesn't matter anyway since multiplied with 0
                elif t: # actually a positive
                    p_t_y = FN / (TN + FN) # ratio of false negatives
                else: # actually a negative
                    p_t_y = TN / (TN + FN) # ratio of true negatives

            if t: # Actually a positive
                p_t =  P / (P + N) # ratio of positive examples
            else: # Actually a negative
                p_t =  N / (P + N) # ratio of positive examples

            if p_t == 0.0: # We don't have any examples for this class (should not happen)
                # There is no uncertainty about class and thus no information
                # gain. We return 0
                return 0.0
            elif p_t_y == 0.0:
                # We set 0*-inf = 0
                return 0.0
            else:
                return p_y*p_t_y*numpy.log2(p_t_y/p_t)

        return sum(term(y, t) for y in [True, False] for t in [True, False])

    @staticmethod
    def normalized_mutual_information(TN, FN, TP, FP):
        """ Normalized mutual information IN(T;Y) = (H(T) - H(T|Y))/H(T)

        This metric has the property that an optimal classifier will always get
        value 1 while any kind of random classifier (those on the diagonal in ROC
        space) get value 0.
        """
        return BinaryClassificationDataset.mutual_information(TN, FN, TP, FP) / \
               BinaryClassificationDataset.mutual_information(TN + FP, 0, TP + FN, 0)


class MultinomialClassificationDataset(BinaryClassificationDataset):
    """ Handle and store multiclass classification performance measures

    **Metrics**

    Balanced accuracy, accuracy and weighted accuracy are calculated as
    in the Binary case.

        :Accuracy: Number of correct classifications devided by total
                   number of classified samples

        :Balanced_accuracy: Mean of True positive rates for all classes

        :Weighted_accuracy:
            Weighted sum of True positive rates for all classes,
            using the `weight` parameter

        :Matthews_correlation_coefficient:
            Pearson’s correlation coefficient between classification
            and true label matrix.

            - Paper: Comparing two K-category assignments by a K-category correlation coefficient
            - Author: J. Gorodkin
            - Page: 369
            - Webpage: http://dx.doi.org/10.1016/j.compbiolchem.2004.09.006

        :micro/macro_average_F_measure:

            - Paper: A Study on Threshold Selection for Multi-label Classification
            - Author: Rong-En Fan and Chih-Jen Lin
            - Page: 4

    .. todo:: Integrate Mututal information, other micro/macro averages and
              other metrics.

    :Author: Mario Michael Krell
    :Created: 2012/11/02
    """
    @staticmethod
    def calculate_metrics(classification_results,
                          time_periods=[],
                          weight=None):
        """ Calculate performance measures from the given classifications """
        # metric initializations
        metrics = metricdict(float)
        classes = []
        for prediction_vector,label in classification_results:
            if not label in classes:
                classes.append(label.strip())
            if not (prediction_vector.label in classes):
                classes.append(prediction_vector.label.strip())
            MultinomialClassificationDataset.update_confusion_matrix(prediction_vector,
                                         label,confusion_matrix=metrics)
        MultinomialClassificationDataset.calculate_confusion_metrics(
                                                            performance=metrics,
                                                            classes=classes,
                                                            weight=weight)
        ### Extract meta metrics from the predictor ### (copy from BinaryClassificationSink)
        # set basic important predictor metrics for default
        #metrics["~~Num_Retained_Features~~"] = numpy.inf
        #metrics["~~Solver_Iterations~~"] = numpy.Inf
        #metrics["~~Classifier_Converged~~"] = True
        # Classifier information should be saved in the parameter
        # 'classifier_information'!!!
        try:
            classifier_information = classification_results[0][0].predictor.classifier_information
            for key, value in classifier_information.iteritems():
                metrics[key] = value
        except:
            pass

        ### Time metrics ###
        if len(time_periods)>0:
            # the first measured time can be inaccurate due to
            # initialization procedures performed in the first executions
            time_periods.pop(0)
            metrics["Time (average)"] = 1./1000 * sum(time_periods) / \
                                                        len(time_periods)
            metrics["Time (maximal)"] = 1./1000 * max(time_periods)
        return metrics

    @staticmethod
    def update_confusion_matrix(classification_vector, label,
                                confusion_matrix=metricdict(float)):
        """ Calculate the change in the confusion matrix

        +--------------+-----------+-----------+
        | class|guess  | c1        | c2        |
        +==============+===========+===========+
        | c1           | T:c1_P:c1 | T:c1_P:c2 |
        +--------------+-----------+-----------+
        | c2           | T:c2_P:c1 | T:c2_P:c2 |
        +--------------+-----------+-----------+

        The change is directly written into the confusion matrix dictionary.

        :Returns: confusion_matrix
        """
        p_label = classification_vector.label.strip()
        label = label.strip()
        metric_str="T:"+label+"_P:"+p_label
        confusion_matrix[metric_str] += 1
        return confusion_matrix

    @staticmethod
    def calculate_confusion_metrics(performance, classes, weight=None):
        """ Calculate metrics of multinomial confusion matrix """
        num_class_samples = defaultdict(float)
        num_class_predictions = defaultdict(float)
        num_samples = 0
        n = len(classes)

        if weight is None or weight == 0.5:
            weight = defaultdict(float)
            for label in classes:
                weight[label]=1.0/n

        cm = numpy.zeros((n,n))
        for i, truth in enumerate(classes):
            for j, prediction in enumerate(classes):
                metric_str = "T:" + truth + "_P:" + prediction
                num_samples += performance[metric_str]
                num_class_samples[truth] += performance[metric_str]
                num_class_predictions[prediction] += performance[metric_str]
                cm[i, j] = performance[metric_str]
        # setting number per default to one to void zero division errors
        for label in classes:
            if num_class_samples[label] == 0:
                num_class_samples[label] = 1

        b_a = 0.0
        w_a = 0.0
        acc = 0.0
        maF = 0.0 #macro F-Measure
        miF_nom = 0.0 #micro F-Measure nominator
        miF_den = 0.0 #micro F-Measure denominator

        for label in classes:
            metric_str = "T:" + label + "_P:" + label
            if not performance[metric_str] == 0:
                b_a += performance[metric_str]/(num_class_samples[label]*n)
                w_a += performance[metric_str]/(num_class_samples[label])\
                    * weight[label]
                acc += performance[metric_str]/num_samples
                maF += \
                    2 * performance[metric_str]/(n *
                    (num_class_predictions[label] + num_class_samples[label]))
                miF_nom += 2 * performance[metric_str]
            miF_den += num_class_predictions[label] + num_class_samples[label]
        performance["Balanced_accuracy"] = b_a
        performance["Accuracy"] = acc
        performance["Weighted_accuracy"] = w_a
        performance["macro_average_F_measure"] = maF
        performance["micro_average_F_measure"] = miF_nom/miF_den

        MC_nom = num_samples * numpy.trace(cm)
        f1 = num_samples**2 * 1.0
        f2 = f1
        for k in range(n):
            for l in range(n):
                MC_nom -= numpy.dot(cm[k, :], cm[:, l])
                f1 -= numpy.dot(cm[k, :], (cm.T)[:, l])
                f2 -= numpy.dot((cm.T)[k, :], cm[:, l])
        if f1 <= 0 or f2 <= 0:
            MCC = 0
        else:
            MCC = MC_nom/(numpy.sqrt(f1)*numpy.sqrt(f1))

        performance["Matthews_correlation_coefficient"] = MCC


class RegressionDataset(BinaryClassificationDataset):
    """ Calculate 1-dimensional and n-dimensional regression metrics

    Metrics for 1-dim regression were taken from:

        - Book: Data mining: practical machine learning tools and techniques
        - Authors: I. H. Witten and E. Frank
        - Page: 178
        - Publisher: Morgan Kaufmann, San Francisco
        - year: 2005

    n-dimensional metrics were variants derived by Mario Michael Krell:

    **micro**

    For the correlation coefficient, the components were treated
    like single regression results.
    For the other metrics, differences and means are taken element or
    component wise and at the final averaging stage the mean is taken
    over all components.

    **component_i_metric**

    For each dimension,
    performance values are calculated separately.

    **macro**

    The component wise metrics were averaged.


    :Author: Mario Michael Krell
    :Created: 2012/11/02

    """
    @staticmethod
    def calculate_metrics(regression_results,
                          time_periods=[],
                          weight=None):
        """ Calculate performance measures from the given classifications """
        # metric initializations
        metrics = metricdict(float)
        # transform results to distinct lists
        predicted_val = []
        actual_val = []
        for prediction_vector,label in regression_results:
            predicted_val.append(prediction_vector.prediction)
            actual_val.append(label)
        if type(actual_val[0]) == list and type(predicted_val[0]) == list:
            vector_regression = True
        elif type(predicted_val[0]) == list and len(predicted_val[0]) == 1:
            # not type(actual_val[0]) ==list
            # --> automatic parameter mapping to numbers
            for i in range(len(predicted_val)):
                predicted_val[i] = predicted_val[i][0]
        elif type(actual_val[0]) == list or type(predicted_val[0]) == list:
            raise TypeError("Prediction (%s) and "%type(predicted_val[0]) +
                            "real value/label (%s) should"%type(actual_val[0]) +
                            " have the same format (list or number/string)")
        else:
            vector_regression = False
        p = numpy.array(predicted_val).astype("float64")
        a = numpy.array(actual_val).astype("float64")
        if not vector_regression:
            metrics["mean-squared_error"] = numpy.mean((p-a)**2)
            metrics["root_mean-squared_error"] = \
                numpy.sqrt(metrics["mean-squared_error"])
            metrics["mean_absolute_error"] = numpy.mean(numpy.abs(p-a))
            metrics["relative_squared_error"] = \
                metrics["mean-squared_error"]/numpy.var(a)
            metrics["root_relative_squared_error"] = \
                numpy.sqrt(metrics["relative_squared_error"])
            metrics["relative absolute error"] = \
                metrics["mean_absolute_error"]/numpy.mean(numpy.abs(a-a.mean()))
            metrics["correlation_coefficient"] = numpy.corrcoef(a,p)[0,1]
        else:
            # treat arrays like flatten arrays!
            metrics["micro_mean-squared_error"] = numpy.mean((p-a)**2)
            metrics["micro_root_mean-squared_error"] = \
                numpy.sqrt(metrics["micro_mean-squared_error"])
            metrics["micro_mean_absolute_error"] = numpy.mean(numpy.abs(p-a))
            metrics["micro_relative_squared_error"] = \
                metrics["micro_mean-squared_error"]/numpy.var(a)
            metrics["micro_root_relative_squared_error"] = \
                numpy.sqrt(metrics["micro_relative_squared_error"])
            metrics["micro_relative absolute error"] = \
                metrics["micro_mean_absolute_error"] / \
                numpy.mean(numpy.abs(a-a.mean()))
            metrics["micro_correlation_coefficient"] = \
                numpy.corrcoef(numpy.reshape(a, a.shape[0]*a.shape[1]),
                               numpy.reshape(p, p.shape[0]*p.shape[1]))[0,1]
            pre_str = []
            metric_names=["mean-squared_error","root_mean-squared_error",
                          "mean_absolute_error", "relative_squared_error",
                          "root_relative_squared_error",
                          "relative absolute error", "correlation_coefficient"]
            # project onto one component and calculate separate performance
            for i in range(len(predicted_val[0])):
                s = "component_"+str(i)+"_"
                pre_str.append(s)
                pi = p[:,i]
                ai = a[:,i]
                metrics[s+"mean-squared_error"] = numpy.mean((pi-ai)**2)
                metrics[s+"root_mean-squared_error"] = \
                    numpy.sqrt(metrics[s+"mean-squared_error"])
                metrics[s+"mean_absolute_error"] = numpy.mean(numpy.abs(pi-ai))
                metrics[s+"relative_squared_error"] = \
                    metrics[s+"mean-squared_error"]/numpy.var(ai)
                metrics[s+"root_relative_squared_error"] = \
                    numpy.sqrt(metrics[s+"relative_squared_error"])
                metrics[s+"relative absolute error"] = \
                    metrics[s+"mean_absolute_error"] / \
                    numpy.mean(numpy.abs(ai-ai.mean()))
                metrics[s+"correlation_coefficient"] = \
                    numpy.corrcoef(ai,pi)[0,1]
            for metric in metric_names:
                l = []
                for pre in pre_str:
                    l.append(metrics[pre+metric])
                metrics["macro_"+metric] = numpy.mean(l)
        try:
            classifier_information = \
                    regression_results[0][0].predictor.classifier_information
            for key, value in classifier_information.iteritems():
                metrics[key] = value
        except:
            pass

        ### Time metrics ###
        if len(time_periods)>0:
            # the first measured time can be inaccurate due to
            # initialization procedures performed in the first executions
            time_periods.pop(0)
            metrics["Time (average)"] = \
                1./1000 * sum(time_periods) / len(time_periods)
            metrics["Time (maximal)"] = 1./1000 * max(time_periods)
        return metrics
