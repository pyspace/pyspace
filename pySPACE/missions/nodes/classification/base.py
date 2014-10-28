""" Base classes for classification """

import numpy

import os
import cPickle

import warnings
import logging
import timeit
# base class
from pySPACE.missions.nodes.base_node import BaseNode
# representation of the linear classification vector
from pySPACE.resources.data_types.feature_vector import FeatureVector
# the output is a prediction vector
from pySPACE.resources.data_types.prediction_vector import PredictionVector


class RegularizedClassifierBase(BaseNode):
    """ Basic class for regularized (kernel) classifiers with extra support in the linear case
    
    **Parameters**

        :class_labels:
            Sets the labels of the classes.
            This can be done automatically, but setting it will be better,
            if you want to have similar predictions values
            for classifiers trained on different sets.
            Otherwise this variable is built up by occurrence of labels.
            Furthermore the important class (ir_class) should get the
            second position in the list, such that it gets higher
            prediction values by the classifier.

            (*recommended, default: []*)

        :complexity:
            Complexity sets the weighting of punishment for misclassification
            in comparison to generalizing classification from the data.
            Value in the range from 0 to infinity.

            (*optional, default: 1*)

        :weight:
            Defines an array with two entries to give different complexity weight
            on the two used classes.
            Set the parameter C of class i to weight*C.

            (*optional, default: [1,1]*)

        :kernel_type:
            Defines the used kernel function.
            One of the following Strings: 'LINEAR', 'POLY','RBF', 'SIGMOID'.
            
            - LINEAR       ::
            
                                u'*v
            - POLY         ::
            
                                (gamma*u'*v + offset)^exponent
            - RBF          ::
            
                                exp(-gamma*|u-v|^2)
            - SIGMOID      ::
            
                                tanh(gamma*u'*v + offset)

            (*optional, default: 'LINEAR'*)

        :exponent:
            Defines parameter for the 'POLY'-kernel.
            Equals parameter /degree/ in libsvm-package.
            
            (*optional, default: 2*)

        :gamma:
            Defines parameter for 'POLY'-,'RBF'- and 'SIGMOID'-kernel.
            In libsvm-package it was set to 1/num_features.

            For RBF-Kernels we calculate it as described in:
            
                :Paper:
                    A practical Approach to Model Selection for Support vector
                    Machines with a Gaussian Kernel
                :Author: M. Varewyck and J.-P. Martens.
                :Formula: 15

            The quasi-optimal complexity should then be found in [0.5,2,8]
            or better to say log_2 C should be found in [-1,1,3].
            For testing a wider range, you may try: [-2,...,4].
            A less accurate version would be to use 1/(num_features*sqrt(2)).
            
            For the other kernels we set it to 1/num_features.
            
            .. warning:: 
                 For the RBF-Parameter selection the 
                 the :class:`~pySPACE.missions.nodes.postprocessing.feature_normalization.HistogramFeatureNormalizationNode`
                 should be used before.
            
            (*optional, default: None*)
            
        :offset:
            Defines parameter for 'POLY'- and 'SIGMOID'-kernel.
            Equals parameter /coef0/ in libsvm-package.
            
            (*optional, default: 0*)

        :nu:
            Defines parameter for 'nu-SVC', 'one-class SVM' and 'nu-SVR'. It
            approximates the fraction of training errors and support vectors.
            Value in the range from 0 to 1.

            (*optional, default: 0.5*)

        :epsilon:
            Defines parameter for 'epsilon-SVR'.
            Set the epsilon in loss function of epsilon-SVR.
            Equals parameter /p/ in libsvm-package.

            (*optional, default: 0.1*)

        :tolerance:
            tolerance of termination criterion, same default as in libsvm.
            
            In the SOR implementation the tolerance may be reduced to
            one tenth of the complexity, if it is higher than this value.
            Otherwise it would be no valid stopping criterion.
            
            (*optional, default: 0.001*)

        :max_time:
            Time for the construction of the classifier
            For LibSVM we restrict the number of steps but for cvxopt
            we use a signal handling to stop processes.
            This may happen, when the parameters are bad chosen or 
            the problem matrix is to large.
            
            Parameter is still in testing and implementation phase.
            
            The time is given in seconds and as a default, one hour is used.
            
            (*optional, default: 3600*)

        :keep_vectors:
            After training the training data is normally deleted,
            except this variable is set to True.
            
            (*optional, default: False*)
            
        :use_list:
            Switch to store samples as *list*. If set to *False* they are stored
            as arrays. Used for compatibility with LIBSVM. This parameter should
            not be changed by the user.

            (*optional, default False*)
        
        :multinomial:
            Accept more than two classes.
            
            (*optional, default: False*)
    
    .. note:: Not all parameter effects are implemented for all inheriting nodes.
              Kernels are available for LibSVMClassifierNode and
              partially for other nodes.
              The *tolerance* has only an effect on Liblinear, LibSVM and SOR classifier.

    :input:    FeatureVector
    :output:   PredictionVector
    :Author:   Mario Krell (mario.krell@dfki.de)
    :Created:  2012/03/28
    """
    def __init__(self, regression = False,
                 complexity = 1, weight = [1,1], kernel_type = 'LINEAR',
                 exponent = 2, gamma = None, offset = 0, nu = 0.5, epsilon = 0.1, 
                 class_labels = [], debug = False, max_time = 3600,
                 tolerance=0.001,
                 complexities_path = None, 
                 keep_vectors=False, max_steps=1,forget_oldest=False,
                 keep_label=None, use_list=False,
                 multinomial=False,
                 **kwargs):

        super(RegularizedClassifierBase, self).__init__(**kwargs)
        # type conversion
        complexity=float(complexity)
        if complexity<1e-10:
            self._log("Complexity (%.42f) is very small. Try rescaling data or check this behavior."%complexity, level = logging.WARNING)

        if self.is_retrainable():
            keep_vectors=True
        
        self.set_permanent_attributes(samples=None, labels=None,
                                      classes=class_labels,
                                      weight=weight,
                                      kernel_type=kernel_type,
                                      complexity=complexity,
                                      exponent=exponent, gamma=gamma,
                                      offset=offset, nu=nu,
                                      epsilon=epsilon, debug=debug,
                                      tolerance=tolerance,
                                      w=None, b=0, dim=None,
                                      feature_names= None,
                                      complexities_path=complexities_path,
                                      regression=regression,
                                      keep_vectors=keep_vectors,
                                      max_time=max_time,
                                      steps=0, max_steps=max_steps, 
                                      forget_oldest=forget_oldest,
                                      keep_label=keep_label,
                                      retraining_needed=False,
                                      use_list=use_list,
                                      multinomial=multinomial,
                                      classifier_information={}
                                      )

    def stop_training(self):
        """ Wrapper around stop training for measuring times"""
        if self.samples is None or len(self.samples) == 0:
            self._log("No training data given to classification node (%s), "
                      % self.__class__.__name__ + "wrong class labels "
                      + "used or your classifier is not using samples.",
                        level=logging.CRITICAL)
        start_time_stamp = timeit.default_timer()
        super(RegularizedClassifierBase, self).stop_training()
        stop_time_stamp = timeit.default_timer()
        if not self.classifier_information.has_key("Training_time(classifier)"):
            self.classifier_information["Training_time(classifier)"] = \
                stop_time_stamp - start_time_stamp
        else:
            self.classifier_information["Training_time(classifier)"] += \
                stop_time_stamp - start_time_stamp

    def is_trainable(self):
        """ Returns whether this node is trainable. """
        return True
    
    def is_supervised(self):
        """ Returns whether this node requires supervised training """
        return True

    def delete_training_data(self):
        """ Check if training data can be deleted to save memory """
        if not (self.keep_vectors or self.is_retrainable()):
            self.samples = []
            self.labels  = []

    def __getstate__(self):
        """ Return a pickable state for this object """
        odict = super(RegularizedClassifierBase, self).__getstate__()
        if self.kernel_type == 'LINEAR':
            if 'labels' in odict:
                odict['labels'] = []
            if 'samples' in odict:
                odict['samples'] = []
            if 'model' in odict: 
                del odict['model']
        else:
            if 'model' in odict: 
                del odict['model']
        return odict

    def store_state(self, result_dir, index=None): 
        """ Stores this node in the given directory *result_dir* """
        if self.store and self.kernel_type == 'LINEAR':
            node_dir = os.path.join(result_dir, self.__class__.__name__)
            from pySPACE.tools.filesystem import create_directory
            create_directory(node_dir)
            try:
                self.features
            except:
                if type(self.w) == FeatureVector:
                    self.features = self.w
                elif not self.w is None:
                    self.features = FeatureVector(self.w.T, self.feature_names)
                else:
                    self.features=None
            if not self.features is None:
                # This node stores the learned features
                name = "%s_sp%s.pickle" % ("features", self.current_split)
                result_file = open(os.path.join(node_dir, name), "wb")
                result_file.write(cPickle.dumps(self.features, protocol=2))
                result_file.close()
                name = "%s_sp%s.yaml" % ("features", self.current_split)
                result_file = open(os.path.join(node_dir, name), "wb")
                result_file.write(str(self.features))
                result_file.close()
                del self.features

    def __setstate__(self, sdict):
        """ Restore object from its pickled state""" 
        super(RegularizedClassifierBase, self).__setstate__(sdict) 
        if self.kernel_type != 'LINEAR':
            # Retraining the svm is not a semantically clean way of restoring
            # an object but its by far the most simple solution 
            self._log("Requires retraining of the classifier") 
            if self.samples != None: 
                self._stop_training()

    def get_sensor_ranking(self):
        """ Transform the classification vector to a sensor ranking

        This method will fail, if the classification vector variable
        ``self.features`` is not existing.
        This is for example the case when using nonlinear classification with
        kernels.
        """
        if not "features" in self.__dict__:
            self.features = FeatureVector(
                numpy.atleast_2d(self.w).astype(numpy.float64),
                self.feature_names)
            self._log("No features variable existing to create generic sensor "
                "ranking in %s."%self.__class__.__name__, level=logging.ERROR)
        # channel name is what comes after the first underscore
        feat_channel_names = [chnames.split('_')[1]
                        for chnames in self.features.feature_names]
        from collections import defaultdict
        ranking_dict = defaultdict(float)
        for i in range(len(self.features[0])):
            ranking_dict[feat_channel_names[i]] += abs(self.features[0][i])
        ranking = sorted(ranking_dict.items(),key=lambda t: t[1])
        return ranking

    def _train(self, data, class_label):
        """ Trains the classifier on the given data
        
        It is assumed that the class_label parameter
        contains information about the true class the data belongs to
        """
        if self.feature_names is None:
            try:
                self.feature_names = data.feature_names
            except AttributeError as e:
                warnings.warn("Use a feature generator node before a " +
                              "classification node.")
                raise e
            if self.dim is None:
                self.dim = data.shape[1]
            if self.samples is None:
                self.samples = []
            if self.labels is None:
                self.labels = []
        if class_label not in self.classes and not "REST" in self.classes and \
                not self.regression:
            warnings.warn("Please give the expected classes to the classifier! "
                          + "%s unknown. "%class_label
                          + "Therefore define the variable 'class_labels' in "
                          + "your spec file, where you use your classifier. "
                          + "For further info look at the node documentation.")
            if self.multinomial or not(len(self.classes) == 2):
                self.classes.append(class_label)
                self.set_permanent_attributes(classes=self.classes)
        # main step of appending data to the list *self.samples*
        if class_label in self.classes or self.regression:
            self.append_sample(data)
        
        if not self.regression and class_label in self.classes:
            self.labels.append(self.classes.index(class_label))
        elif not self.regression and "REST" in self.classes:
            self.labels.append(self.classes.index("REST"))
        elif self.regression:  # regression!
            try:
                self.labels.append(float(class_label))
            except ValueError:  # one-class-classification is regression-like
                self.labels.append(1)
        else:  # case, where data is irrelevant
            pass

    def train(self,data,label):
        """ Special mapping for multi-class classification """
        #one vs. REST case
        if "REST" in self.classes and not label in self.classes:
            label = "REST"
        # one vs. one case
        if not self.multinomial and len(self.classes) == 2 and \
                not label in self.classes:
            return
        start_time_stamp = timeit.default_timer()
        super(RegularizedClassifierBase, self).train(data, label)
        stop_time_stamp = timeit.default_timer()
        if not self.classifier_information.has_key("Training_time(classifier)"):
            self.classifier_information["Training_time(classifier)"] = \
                stop_time_stamp - start_time_stamp
        else:
            self.classifier_information["Training_time(classifier)"] += \
                stop_time_stamp - start_time_stamp

    def append_sample(self,sample):
        """ Some methods need a list of arrays as lists and some prefer arrays
        """
        data_array = sample.view(numpy.ndarray)
        if self.use_list:
            self.samples.append(map(float, list(data_array[0, :])))
        else:
            self.samples.append(data_array[0, :])

    def _execute(self, x):
        """ Executes the classifier on the given data vector in the linear case
        
        prediction value = <w,data>+b
        """
        if self.kernel_type == 'LINEAR':
            data = x.view(numpy.ndarray)
            # Let the SVM classify the given data: <w,data>+b
            if self.w is None:
                prediction_value = 0
                self.w = numpy.zeros(x.shape[1])
            else:
                prediction_value = float(numpy.dot(self.w.T, data[0, :]))+self.b
            # one-class multinomial handling of REST class
            if "REST" in self.classes and self.multinomial:
                if "REST" == self.classes[0]:
                    label = self.classes[1]
                elif "REST" == self.classes[1]:
                    label = self.classes[0]
                    prediction_value *= -1
            # Look up class label
            # prediction_value --> {-1,1} --> {0,1} --> Labels
            elif prediction_value > 0:
                label = self.classes[1]
            else:
                label = self.classes[0]
            
            return PredictionVector(label=label, prediction=prediction_value,
                                    predictor=self)

    def print_variables(self):
        """ Debug function for printing the classifier and the slack variables
        """
        # Precision does not work here because of the strange dtype. 
        numpy.set_printoptions(edgeitems=50, precision=4, suppress=False,
                               threshold=50)
        # ...Setting the dtype to list doesn't work either.
        print self.print_w
        print 'This is the classification vector w and b=', self.b, '.'
        print self.num_retained_features, ' out of ', self.dim, \
            ' features have been used.'
        
        print self.num_sv, " vectors of ", self.num_samples, " have been used."
        # print self.t, "are the Slack variables."
        
        if not((numpy.array(self.t) >= 0).all()):
            print "There are negative slack variables! Classification failed?"
        print "%i vectors of %i have been used for the inner margin and" \
                                    % (self.inner_margin, self.num_samples)
        numpy.set_printoptions(edgeitems=100, linewidth=75, precision=5,
                               suppress=True, threshold=1000)
        print numpy.array(self.ti), "are the inner Slack variables."
        numpy.set_printoptions(edgeitems=3, infstr='Inf', linewidth=75,
                               nanstr='NaN', precision=8, suppress=False,
                               threshold=1000)

    def kernel_func(self, u, v):
        """ Returns the kernel function applied on x and y 
        
            - POLY         ::
            
                                (gamma*u'*v + offset)^exponent
            - RBF          ::
            
                                exp(-gamma*|u-v|^2)
            - SIGMOID      ::
            
                                tanh(gamma*u'*v + offset)
        
        """
        if not self.kernel_type == "LINEAR" and self.gamma is None:
            self.calculate_gamma()

        if self.kernel_type == "LINEAR":
            return float(numpy.dot(u, v))
        elif self.kernel_type == "POLY":
            h = float(numpy.dot(u, v))
            return (self.gamma*h+self.offset)**self.exponent
        elif self.kernel_type == "RBF":
            return numpy.exp(-self.gamma*float(numpy.sum((u - v)**2)))
        elif self.kernel_type == "SIGMOID":
            h = float(numpy.dot(u, v))
            return numpy.tanh(self.gamma*h+self.offset)
        elif self.kernel_type.startswith("lambda "):
            function = eval(self.kernel_type)
            return float(function(u, v))

    def calculate_gamma(self):
        """ Calculate default gamma 
        
        This defines a parameter for 'POLY'-,'RBF'- and 'SIGMOID'-kernel.
        We calculate the parameter `gamma` as described in the base node
        description.
        """
        if (self.kernel_type == 'POLY' or self.kernel_type == 'SIGMOID') \
                and self.gamma is None:
            self.gamma = 1.0/self.dim
        elif self.kernel_type == 'RBF' and self.gamma is None and \
                not self.regression:
            a = self.labels.count(self.classes.index(self.classes[0]))
            b = self.labels.count(self.classes.index(self.classes[1]))
            if a > b:
                relevant = 1
            else:
                relevant = 0
            relevant_samples=[]
            for i, label in enumerate(self.labels):
                if label == relevant:
                    relevant_samples.append(self.samples[i])
            variance = numpy.median(numpy.var(numpy.array(self.samples),
                                              axis=0))
            self.gamma = 0.5/(variance*self.dim)
            self._log("No gamma specified. Using: %f." % self.gamma,
                      level=logging.WARNING)
        elif self.gamma is None:
            self.gamma = 0.001


class TimeoutException(Exception):
    """ Break up for to long simplex iterations """ 
    pass
