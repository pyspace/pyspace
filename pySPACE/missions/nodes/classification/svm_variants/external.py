""" Wrapper around external SVM variant implementations like LibSVM or LIBLINEAR """

import logging
import warnings

try:
#    import matplotlib as mpl
#    mpl.rcParams['text.usetex'] = True
#    mpl.rcParams['text.latex.unicode'] = True
    import matplotlib.pyplot as plt
except ImportError:
    pass

import scipy.spatial.distance
from pySPACE.missions.nodes.decorators import NoOptimizationParameter, ChoiceParameter

# import the external libraries
try: # Liblinear
    import liblinearutil
except ImportError:
    pass
try:  # Libsvm
    import svmutil
except ImportError:
    pass

# representation of the linear classification vector
from pySPACE.resources.data_types.feature_vector import FeatureVector

# the output is a prediction vector
from pySPACE.resources.data_types.prediction_vector import PredictionVector

# array handling
import numpy

# base class
from pySPACE.missions.nodes.classification.base import RegularizedClassifierBase


@NoOptimizationParameter("regression")
class LibSVMClassifierNode(RegularizedClassifierBase):
    """Classify like a Standard SVM with the LibSVM settings.
    
    This node is a wrapper around the *current* libsvm implementation of a SVM.
    
    http://www.csie.ntu.edu.tw/~cjlin/libsvm/oldfiles/
    
    **Parameters**

    Some general parameters are only documented in the
    :class:`RegularizedClassifierBase <pySPACE.missions.nodes.classification.base.RegularizedClassifierBase>`.

        :svm_type:
            Defines the used SVM type.
            One of the following Strings: 'C-SVC', 'one-class SVM',
            'epsilon-SVR', 'nu-SVR'. The last two types are for regression,
            the first for classification.

            .. warning:: For using "one-class SVM" better use the
                :class:`~pySPACE.missions.nodes.classification.one_class.LibsvmOneClassNode`.

            
            (*optional, default: 'C-SVC'*)
            
        :complexity:
            Defines parameter for 'C-SVC', 'epsilon-SVR' and 'nu-SVR'.
            Complexity sets the weighting of punishment for misclassification
            in comparison to generalizing classification from the data.
            Equals parameter /cost/ or /C/ in libsvm-package.
            Value in the range from 0 to infinity.

            (*optional, default: 1*)
            

        :str_label_function:
            A String representing a Python eval()-able function,
            that transforms the labels (list). 
            It makes only sense for numeric labels. E.g. 
            "lambda liste: [exp(-0.0001*elem**2) for elem in liste]".
            
            (*optional, default: None*)
            
        :debug:
            If *debug* is True one gets additional output 
            concerning the classification.
            
            .. note:: This makes only sense for the 'LINEAR'-*kernel_type*.
            
            (*optional, default: False*)
            
        :store:
            Parameter of super-class. If *store* is True,
            the classification vector is stored as a feature vector.
            
            .. note:: This makes only sense for the 'LINEAR'-*kernel_type*.
            
            (*optional, default: False*)
            
        :max_iterations:
            Restricts the solver inside the LibSVM to maximal
            use N iterations, where N is the product of *max_iterations*
            and the number of samples used to train the classifier.
            If omitted or set to zero the
            solver takes as much iterations it needs to
            calculate the model.
            
            .. note:: This number has to be an integer and
                    is very important if you expect the classifier
                    not to converge.
            
            .. note:: To use this feature you will need the modified libsvm
                    of the external folder in a compiled version.
                    Furthermore you should make sure,
                    that this version is imported, e.g. by adding the path
                    at the beginning of the configuration file paths.

            (*optional, default: 0*)
            
        :complexities_path:
            If a complexities_path is given, the complexity is read from a
            YAML file. This file has a dict with channel numbers as keys and
            the corresponding complexity as value. Also, a
            'features_per_channel' dict entry can be set to calculate
            channel number based on the number of features. If no 
            'features_per_channel' is given, a factor of 1 is assumed. This
            can be used to specify the number of features in the file, instead
            of the number of sensor channels. A minimal example for the file
            content could be::
            
                {32: 0.081, 62: 0.019, features_per_channel: 6}.
                
            'complexities_path' will overwrite 'complexity'.
    
            (*optional, default: 0*)

    **Exemplary Call**
    
    .. code-block:: yaml
    
        -
            node : LibSVM_Classifier
            parameters :
                svm_type : "C-SVC"
                complexity : 1
                kernel_type : "LINEAR"
                class_labels : ['Standard', 'Target']
                weight : [1,3]
                debug : True
                store : True
                max_iterations : 100
    
    :input:    FeatureVector
    :output:   PredictionVector
    :Author: Jan Hendrik Metzen (jhm@informatik.uni-bremen.de)
            & Mario Krell (Mario.krell@dfki.de)
    :Created: 2009/07/02
    :Revised: 2010/04/09
    :Last change: 2011/05/06 Mario Krell old version deleted
    
    """
    def __init__(self, svm_type='C-SVC', max_iterations=0,
                 str_label_function=None, complexities_path=None, **kwargs):

        # set default svm_type 'C-SVC' if unsupported svm_type is selected
        supported_types = ["C-SVC", "one-class SVM", "epsilon-SVR", "nu-SVR"]
        if svm_type not in supported_types:
            svm_type = 'C-SVC'
            warnings.warn("SVM-type unknown. C-SVC will be used!")

        if svm_type == 'C-SVC':
            regression = False
        else:
            regression = True
        super(LibSVMClassifierNode, self).__init__(
            regression=regression, **kwargs)
        # Check if the svm module has been correctly imported
        try:
            import svmutil
        except ImportError as e:
            self._log("svmutil.py could not be imported.")
            message = "Using the LibSVMClassifierNode requires " + \
                "the Python svm module provided by libsvm. " + \
                "For installation hints see documentation " + \
                "or http://www.csie.ntu.edu.tw/~cjlin/libsvm/." + \
                "Furthermore try to import the path to the " + \
                "external folder."
            args = e.args
            if not args:
                e.args = message
            else:
                e.args = (message,) + args
            raise

        self.set_permanent_attributes(str_label_function=str_label_function,
                                      svm_type=svm_type,
                                      max_iterations=int(max_iterations),
                                      store_all_samples=True,
                                      predictor_iterations=numpy.Inf)

    def _stop_training(self, debug=False):
        """ Finish the training, i.e. train the SVM """
        self._complete_training(debug)
        self.relabel_training_set()

    def _complete_training(self, debug=False):
        """ Iterate over the complete data to get the initial model """
        ########## read complexities file if given ##########
        if self.complexities_path is not None:
            import yaml
            complexities_file=open(self.complexities_path, 'r')
            complexities = yaml.load(complexities_file)
            # nr of channels    = nr of features (==dim) / features_per_channel
            if not 'features_per_channel' in complexities:
                complexities['features_per_channel'] = 1
            self.complexity = complexities[
                    round(self.dim/complexities['features_per_channel'])]
            self._log("Read complexity %s from file. Dimension is %s" %
                      (self.complexity, self.dim), level=logging.INFO)
            
        # not compatible with regression!
            # self._log("Instances of Class %s: %s, %s: %s" \
            #            % (self.classes[0], 
            #               self.labels.count(self.classes.index(self.classes[0])),
            #               self.classes[1], 
            #               self.labels.count(self.classes.index(self.classes[1]))))
        # instead this?:
        self._log("Performing training of SVM.")
        
        ########## Calculation of default gamma ##########
        self.calculate_gamma()

        self.num_samples = len(self.samples)

        # nr_weight is the number of elements in the array weight_label and
        # weight. Each weight[i] corresponds to weight_label[i], meaning that
        # the penalty of class weight_label[i] is scaled by a factor of 
        # weight[i]. If you do not want to change penalty for any of the 
        # classes, just set nr_weight to 0.
        
        ########## preparation of the libsvm command ##########
        # for probability output add "-b 1" to options
        options = \
            "-c %.42f -d %d -g %.42f -r %.42f -n %.42f -p %.42f -e %.20f -m %.42f" % \
            (self.complexity, self.exponent, self.gamma,
            self.offset, self.nu, self.epsilon, self.tolerance, 1000)
            # use 1000MB instead of 100MB (default)
        # options += " -b 1" un-comment this for probabilistic output!
        if self.multinomial:
            options += " -b 1"
        for i,w in enumerate(self.weight):
            options += " -w%d %.42f" % (i, w)
        if self.kernel_type == 'LINEAR':
            options += " -t 0"
        elif self.kernel_type == 'POLY':
            options += " -t 1"
        elif self.kernel_type == 'RBF':
            options += " -t 2"
        elif self.kernel_type == 'SIGMOID':
            options += " -t 3"
        else:
            self.kernel_type = 'LINEAR'
            options += " -t 0"
            warnings.warn("Kernel unknown! Precomputed Kernels are not " +
                          "yet implemented. Linear Kernel used.")
            # PRECOMPUTED: kernel values in training_set_file 
            #              (not yet implemented)

        if self.svm_type == 'C-SVC':
            options += " -s 0"
        elif self.svm_type == 'nu-SVR':
            options += " -s 1"
        elif self.svm_type == 'one-class SVM':
            options += " -s 2"
        elif self.svm_type == 'epsilon-SVR':
            options += " -s 3"
        else:
            options += " -s 0"
            self.svm_type = 'C-SVC'
            warnings.warn("SVM-type unknown. C-SVC will be used!")
        if not self.debug:
            options += " -q"
            self._log("Libsvm is now quiet!")
        
        old_libsvm_options = options
        
        if self.max_iterations != 0:
            options += " -i %d" % self.max_iterations
        try:
            param = svmutil.svm_parameter(options)
        except ValueError:
            param = svmutil.svm_parameter(old_libsvm_options)
            self._log(
                "Using max_iterations is not supported by the standard " +
                "LIBSVM. Change your Python path to our customized version!",
                level=logging.CRITICAL)

        # transform labels with *label_function*
        if self.str_label_function is not None:
            self.label_function = eval(self.str_label_function)
            self.labels = self.label_function(self.labels)
        
        # build the classifier
        # h = [map(float,list(data)) for data in self.samples]
        problem = svmutil.svm_problem(self.labels, [
            map(float, list(data)) for data in self.samples])
        model = svmutil.svm_train(problem, param)
        if not self.multinomial:
            if (self.svm_type == 'C-SVC' or self.svm_type == 'one-class SVM') \
                    and self.kernel_type == 'LINEAR':
                self.calculate_classification_vector(model)
                if self.debug:
                    # This calculation is needed for further analysis
                    self.calculate_slack_variables(model) 
                    print "LIBSVM Parameter:"
                    self.print_variables()
            else:
                # Slack variables are the same no matter which kernel is used
                # This method is mainly used to reduce the number of samples
                # being stored later on.
                if self.debug:
                    self.calculate_slack_variables(model) 
                self.model = model
        else:
            self.model = model
            # Slack variables are the same no matter which kernel is used
            # This method is mainly used to reduce the number of samples
            # being stored later on.
        
        # read number of iterations needed to solve the problem
        if self.max_iterations != 0:
            try:
                predictor_iterations = model.get_num_iterations()
                self.classifier_information["~~Solver_Iterations~~"] = \
                    predictor_iterations
                if predictor_iterations == 0 or \
                        predictor_iterations == numpy.Inf:
                    self.classifier_information["~~SVM_Converged~~"] = False
                else:
                    self.classifier_information["~~SVM_Converged~~"] = True
            except:
                warnings.warn("Could not read state of the LibSVM Solver " +
                              "from the C-Library!")

        try:
            self.classifier_information["~~offset~~"] = self.b
            self.classifier_information["~~w0~~"] = self.w[0]
            self.classifier_information["~~w1~~"] = self.w[1]
        except:
            pass
        self.delete_training_data()

    def _execute(self, x):
        """ Executes the classifier on the given data vector x.
        prediction value = <w,data>+b in the linear case."""
        data = x.view(numpy.ndarray)
        if self.svm_type == 'C-SVC':
            if self.kernel_type == 'LINEAR' and not self.multinomial:
                return super(LibSVMClassifierNode, self)._execute(x)
            else:
                # for probability output add "-b 1" as 4th parameter
                if self.multinomial:
                    try:
                        p_labs, p_acc, p_vals = svmutil.svm_predict([0], [
                            map(float, list(data[0, :]))], self.model,
                            '-q -b 1')
                    except ValueError:  # Wrong options
                        p_labs, p_acc, p_vals = svmutil.svm_predict([0], [
                            map(float, list(data[0, :]))], self.model, '-b 1')
                else:
                    try:
                        prediction_value = svmutil.svm_predict([0], [
                            map(float, list(data[0, :]))], self.model,
                            '-q')[2][0][0]
                    except ValueError:  # Wrong options
                        prediction_value = svmutil.svm_predict([0], [
                            map(float, list(data[0, :]))], self.model)[2][0][0]
                    except IndexError:
                        warnings.warn("Probably your classification failed!")
                        prediction_value = 0

                # The new version has only one output of the score.
                # The ordering can be obtained by model.labels and if it is
                # not [1,0] we have to change the sign of the score to be
                # comparable with the old libsvm AND to do the right mapping
                # back to the binary labels
                if self.model.get_labels() == [0, 1]:
                    prediction_value = -prediction_value

            # Look up class label
            # prediction_value --> {-1,1} --> {0,1} --> Labels
            if self.multinomial:
                prediction = self.classes[int(p_labs[0])]
                prediction_value = p_vals[0][int(p_labs[0])]
            else:
                if prediction_value > 0:
                    prediction = self.classes[1]
                else:
                    prediction = self.classes[0]
            
            prediction_vector = PredictionVector(label=prediction,
                                                 prediction=prediction_value,
                                                 predictor=self)
            
            return prediction_vector
        elif self.svm_type == 'one-class SVM':  # one-class! TODO: Extra Node?
            # for probability output add "-b 1" as 4th parameter
            # get prediction as mentioned above
            if not self.kernel_type == "LINEAR" and not self.multinomial:
                try:
                    prediction = svmutil.svm_predict([0], [
                        map(float, list(data[0, :]))], self.model, "-q")
                except ValueError:
                    prediction = svmutil.svm_predict([0], [
                        map(float, list(data[0, :]))], self.model)
                prediction_value = prediction[2][0][0]
                if prediction_value >= 0:
                    label = self.classes[0]
                else:
                    label = self.classes[1]
                return PredictionVector(prediction=prediction_value,
                                        predictor=self,
                                        label=label)
            else:
                result = super(LibSVMClassifierNode, self)._execute(x)
                # invert label
                result.label = self.classes[1-self.classes.index(result.label)]
                return result
        else: # regression! TODO: Extra Node?
            # for probability output add "-b 1" as 4th parameter
            try:
                prediction_value = svmutil.svm_predict([0], [
                    map(float, list(data[0, :]))], self.model, "-q")
            except ValueError:
                prediction_value = svmutil.svm_predict([0], [
                    map(float, list(data[0, :]))], self.model)
            prediction_value = prediction_value[2][0][0]
            return PredictionVector(prediction=prediction_value,
                                    predictor=self)

    def save_model(self, filename):
        svmutil.svm_save_model(filename, self.model)

    def load_model(self, filename):
        print 'load model'
        self.model = svmutil.svm_load_model(filename)
           
    def calculate_slack_variables(self, model):
        """This method calculates from the given SVM model
        the related slack variables for classification."""
        self.t = []
        self.num_sv = 0
        self.num_nsv = 0
        self.inner_margin = 0
        self.ti = []
        dropped_samples = []
        dropped_labels = []
        for i in range(self.num_samples):
            # ctype libsvm bindings
            try:
                p = svmutil.svm_predict([0], [
                    map(float, list(self.samples[i-self.num_nsv]))],
                    model, "-q")[2][0][0]
            except ValueError:
                p = svmutil.svm_predict([0], [
                    map(float, list(self.samples[i-self.num_nsv]))],
                    model)[2][0][0]
            except IndexError:
                self._log("Classification failed. " +
                          "Did you specify the parameters correctly?",
                          level=logging.ERROR)
                p = 0
            if model.get_labels() == [0,1]:
                p = -p
            p *= 2 * (self.labels[i - self.num_nsv] - 0.5)
            if p > 1:
                self.t.append(0)
                self.ti.append(0)
                dropped_samples.append(self.samples.pop(i - self.num_nsv))
                dropped_labels.append(self.labels.pop(i - self.num_nsv))
                self.num_nsv += 1
            else:
                self.t.append(1-p)
                self.num_sv += 1
                if 1-p<1e-5:
                    p = 1
                    self.ti.append(0)
                else:
                    self.ti.append(1-p)
                    self.inner_margin +=1
        # if self.store_all_samples:
        for i in range(len(dropped_samples)):
            self.samples.append(dropped_samples[i])
            self.labels.append(dropped_labels[i])
        del(dropped_samples)
        del(dropped_labels)
    
    def calculate_classification_vector(self, model):
        """ Calculate classification vector w and the offset b """
        # ctypes libsvm bindings
        # TODO get parameter maybe easier
        try:
            self.b = svmutil.svm_predict([0], [[0.0]*self.dim], model,
                                         "-q")[2][0][0]
        except ValueError:
            self.b = svmutil.svm_predict([0], [[0.0]*self.dim], model)[2][0][0]
        except IndexError:
            self._log("Classification failed. " +
                      "Did you specify the parameters correctly?",
                      level=logging.ERROR)
            self.b = 0
            self.w = numpy.zeros(self.dim)
            self.features = FeatureVector(numpy.atleast_2d(self.w).astype(
                                          numpy.float64), self.feature_names)
        if model.get_labels() == [0, 1]:
            self.b = -self.b
        self.w = numpy.zeros(self.dim)
        for i in range(self.dim):
            e = [0.0] * self.dim
            e[i] = 1.0
            try:
                self.w[i] = svmutil.svm_predict([0],[e], model, "-q")[2][0][0]
            except ValueError:
                try:
                    self.w[i] = svmutil.svm_predict([0],[e], model)[2][0][0]
                except IndexError:
                    pass
            except IndexError:
                pass
            if model.get_labels() == [0,1]:
                self.w[i] = -self.w[i]
            self.w[i] -= self.b 
        self.features = FeatureVector(numpy.atleast_2d(self.w).astype(
                                      numpy.float64), self.feature_names)
        try:
            wf = []
            for i,feature in enumerate(self.feature_names):
                if not self.w[i] == 0:
                    wf.append((self.w[i],feature))
            wf.sort()
            w = numpy.array(wf, dtype='|S200')
        except ValueError:
            self._log('w could not be converted.', level=logging.WARNING)
        except IndexError:
            self._log('There are more feature names than features. \
                      Please check your feature generation and input data.',
                      level=logging.CRITICAL)
            self.b = 0
            w = numpy.zeros(self.dim)
            self.w = w
        # only features without zero multiplier are relevant
        self.num_retained_features = len(w)
        self.classifier_information["~~Num_Retained_Features~~"] = \
            self.num_retained_features
        self.print_w = w

    def remove_no_border_points(self, retraining_required):
        """ Discard method to remove all samples from the training set that are
            not in the border of their class.

            The border is determined by a minimum distance from the center of
            the class and a maximum distance.

            :param retraining_required: flag if retraining is
                    requiered (the new point is a potential SV or a removed
                    one was a sv)
        """
        # get centers of each class
        targetSamples = [s for (s, l) in zip(self.samples, self.labels)\
                         if l == 1]  # self.classes.index("Target")]
        standardSamples = [s for (s, l) in zip(self.samples, self.labels)\
                           if l == 0]  # self.classes.index("Standard")]

        if self.training_set_ratio == "KEEP_RATIO_AS_IT_IS":
            num_target = len(targetSamples)
            num_standard = len(standardSamples)

            num_target = 1.0 * num_target / (num_target + num_standard) * \
                self.basket_size
            num_standard = self.basket_size - num_target

        # mean vector of each class (its center)
        mTarget = numpy.mean(targetSamples, axis=0)
        mStandard = numpy.mean(standardSamples, axis=0)

        # euclidean distance between the class centers
        R = scipy.spatial.distance.euclidean(mTarget, mStandard)

        if self.show_plot:
            dim = numpy.shape(self.samples)[1]
            if dim == 2:
                self.plot_class_borders(
                    mStandard, mTarget, R, self.scale_factor_small,
                    self.scale_factor_tall)

        # get distance of each point to its class center
        distances = []
        for i, (s, l) in enumerate(zip(self.samples, self.labels)):
            if l==self.classes.index("Target"):
                r_1 = scipy.spatial.distance.euclidean(s,mTarget)
                r_2 = scipy.spatial.distance.euclidean(s,mStandard)
                distances.append([i, s, l, r_1, r_2/(r_1 + r_2)])
            else:
                r_1 = scipy.spatial.distance.euclidean(s,mStandard)
                r_2 = scipy.spatial.distance.euclidean(s,mTarget)
                distances.append([i, s, l, r_1, r_2/(r_1 + r_2)])

        if self.border_handling == "USE_ONLY_BORDER_POINTS":
            # remove all points that are not in the border (in a specific
            # radius) around the center
            # does not guarantee that demanded number of samples are
            # contained in the new training set
            distances = filter(lambda x: (
                self.scale_factor_small*R < x[3] < self.scale_factor_tall*R),
                               distances)
            # sort according to weight
            distances.sort(key=lambda x: x[5])
            # pay attention to the basket size
            distances = distances[:self.basket_size]
        elif self.border_handling == "USE_DIFFERENCE":
            # take that point that differ most
            # sort by distance, then sort by weight
            distances.sort(key=lambda x: (abs(x[3] -
                                          ((self.scale_factor_tall -
                                            self.scale_factor_small)/2.0)*R) *
                                          (x[0] != len(self.samples)), x[4]))

        if self.border_handling == "USE_ONLY_BORDER_POINTS":
            # pay attention to the basket size
            distances = distances[:self.basket_size]
        elif self.training_set_ratio == "KEEP_RATIO_AS_IT_IS":
            distances_tmp = []
            for d in distances:
                if d[2] == 1 and num_target > 0:
                    num_target -= 1
                    distances_tmp.append(d)
                elif d[2] == 0 and num_standard > 0:
                    num_standard -= 1
                    distances_tmp.append(d)
            distances = distances_tmp
        elif self.training_set_ratio == "BALANCED_RATIO":
            distances_tmp = []
            num_target = 0
            num_standard = 0
            for d in distances:
                if d[2] == 1 and num_target < (self.basket_size/2):
                    num_target += 1
                    distances_tmp.append(d)
                elif d[2] == 0 and num_standard < (self.basket_size/2):
                    num_standard += 1
                    distances_tmp.append(d)
            distances = distances_tmp
        else:
            # pay attention to the basket size
            distances = distances[:self.basket_size]

        [idxs, _, _, _, _] = zip(*distances)
        retraining_required = self.remove_samples(list(
            set(numpy.arange(self.num_samples)) - set(idxs))) \
            or retraining_required

        return retraining_required

    def add_new_sample(self, data, class_label=None, default=False):
        """ Add a new sample to the training set.

            :param data:  A new sample for the training set.
            :type  data:  list of float
            :param class_label:    The label of the new sample.
            :type  class_label:    str
            :param default:  Specifies if the sample is added to the current
                             training set or to a future training set
            :param default:  bool
        """
        # use a separate knowledge base when old samples will be totally removed
        if (self.discard_type == "CDT" or self.discard_type == "INC_BATCH")\
                and default is False:
            self.future_samples.append(data)
            self.future_labels.append(class_label)

            # the sample size for the new knowledge base
            # is limited to basket size, so pop oldest
            while len(self.future_samples) > self.basket_size:
                self.future_samples.pop(0)
                self.future_labels.pop(0)
        else:
            # add new data
            self._train_sample(data, class_label)
            self.num_samples += 1

    def remove_samples(self, idxs):
        """ Remove the samples at the given indices from the training set.

            :param: idxs: Indices of the samples to remove.
            :type:  idxs: list of int
            :rtype: bool - True if a support vector was removed.
        """
        idxs.sort(reverse=True)
        for idx in idxs:
            self.samples.pop(idx)
            self.labels.pop(idx)
            if self.add_type == "UNSUPERVISED_PROB":
                self.decisions.pop(idx)
            self.num_samples -= 1
        return True

    def visualize(self):
        """ Show the training samples, SVS and the current decision function
        """
        dim = numpy.shape(self.samples)[1]
        if dim == 2:
            ax = plt.gca()
            ax.set_xlabel(r'$x_0$')
            ax.set_ylabel(r'$x_1$')

            self.plot_samples()
            self.plot_hyperplane()
        elif dim == 3:
            ax = plt.gca(projection='3d')
            ax.set_xlabel(r'$x_0$')
            ax.set_ylabel(r'$x_1$')
            ax.set_zlabel(r'$x_2$')

            self.plot_samples_3D()
            self.plot_hyperplane_3D()

        if dim == 2 or dim == 3:
            plt.draw()
            if self.save_plot is True:
                imagename = "%s/tmp%010d.png"\
                            % (self.plot_storage, self.m_counter_i)
                self.m_counter_i += 1
                plt.savefig(imagename)


@NoOptimizationParameter("use_list")
@ChoiceParameter("svm_type", choices=[0, 1, 2, 3, 4, 5, 6, 7])
class LiblinearClassifierNode(LibSVMClassifierNode):
    """ Code Integration of external linear SVM classifier program

    http://www.csie.ntu.edu.tw/~cjlin/liblinear/
    LIBLINEAR was implemented by the LIBSVM programmers.
    
    It is important to mention, that here (partially) the same modified SVM
    model is used as in the SOR variant.
    (:mod:`pySPACE.missions.nodes.classification.svm_variants.SOR`)

    **Parameters**

    Some general parameters are only documented in the
    :class:`RegularizedClassifierBase <pySPACE.missions.nodes.classification.base.RegularizedClassifierBase>`.

        :svm_type:
            :0: L2-regularized logistic regression (primal)
            :1: L2-regularized L2-loss support vector classification (dual)
            :2: L2-regularized L2-loss support vector classification (primal)
            :3: L2-regularized L1-loss support vector classification (dual)
            :4: multi-class support vector classification by Crammer and Singer
            :5: L1-regularized L2-loss support vector classification
            :6: L1-regularized logistic regression
            :7: L2-regularized logistic regression (dual)
            
            Type 3 is the standard SVM with 
            b used in the target function as component of w (offset = True)
            or b set to zero. 
            
            (*optional, default:3*)
    
        :tolerance:
            Tolerance of termination criterion, same default as in libsvm.
            
            .. todo:: Same variable name in upper class for epsilon-SVR
                      instead of tolerance.
    
            (*optional, default: 0.001*)
        
        :offset:
            If True, x is internally replaced by (x,1)
            to get an artificial offset b.
            Probably in this case b is regularized.
            Otherwise the offset b in the classifier function (w^Tx+b)
            is set to zero.
            
            (*optional, default: True*)

        :store:
            Parameter of super-class. If *store* is True,
            the classification vector is stored as a feature vector.

            (*optional, default: False*)

    **Exemplary Call**

    .. code-block:: yaml

        -
            node : lSVM
            parameters :
                class_labels : ["Target", "Standard"]

    :Author: Mario Michael Krell (mario.krell@dfki.de)
    :Created: 2012/01/19
    """

    def __init__(self,tolerance=0.001, svm_type=3, offset=True, **kwargs):
        
        if offset:
            offset = 1
        else:
            offset = -1
            
        super(LiblinearClassifierNode,self).__init__(use_list=True, **kwargs)

        # svm type is renamed such that C-SVC is still used in the super class
        # this is currently especially advantageous in the execute method
        self.set_permanent_attributes(
            tolerance=tolerance, alg_num=svm_type, offset=offset)

    def _train(self, data, class_label):
        """ Trains the classifier on the given data
        
        It is assumed that the class_label parameter
        contains information about the true class the data belongs to

        .. todo::   check in new version of liblinear, if ndarrays are accepted
                    and the method from libsvm can be used.
        """
        self._train_phase_started = True
        if self.feature_names is None:
            try:
                self.feature_names = data.feature_names
            except AttributeError as e:
                warnings.warn(
                    "Use a feature generator node before a classification node."
                    )
                raise e
            if self.dim is None:
                self.dim = data.shape[1]
            if self.samples is None:
                self.samples = []
            if self.labels is None:
                self.labels = []
        if class_label not in self.classes:
            warnings.warn(
                "Please give the expected classes to the classifier! " +
                "%s unknown. Therefore, define the variable " % class_label +
                "'class_labels' in your spec file, " +
                "where you use your classifier. " +
                "For further information refer to the node documentation.")
            self.classes.append(class_label)
            self.set_permanent_attributes(classes=self.classes)
        
        # Collect the data
        data_array=data.view(numpy.ndarray)
        self.samples.append(map(float, list(data_array[0,:])))
        self.labels.append(self.classes.index(class_label))

    def _stop_training(self, debug=False):
        """ Finish the training, i.e. train the SVM """
        self._complete_training(debug)
        self.relabel_training_set()

    def _complete_training(self, debug=False):
        """ Forward data to external training and extract classifier information
        """
        if self.str_label_function is not None:
            self.label_function = eval(self.str_label_function)
            self.labels = self.label_function()

        options = "-c %.42f  -e %.42f -s %d -B %d" % \
             (self.complexity, self.tolerance, self.alg_num, self.offset)
        for i,w in enumerate(self.weight):
            options += " -w%d %.42f" % (i, w)
        if not self.debug:
            options += " -q"
            self._log("Liblinear is now quiet!")

        import liblinearutil

        param = liblinearutil.parameter(options)
        problem = liblinearutil.problem(self.labels, self.samples)
        model = liblinearutil.train(problem, param)

        self.calculate_classification_vector(model)
        if self.debug:
            print self.print_w
            print self.b

    def calculate_classification_vector(self, model):
        """This method calculates from the given SVM model
        the related classification vector w and the offset b."""
        # ctypes liblinear bindings
        if self.offset == 1:
            self.b = model.w[self.dim]
        else:
            self.b = 0
        self.w = numpy.zeros(self.dim)
        for i in range(self.dim):
            self.w[i] = model.w[i]
        if model.get_labels() == [0,1]:
            self.w = -1*self.w
            self.b = -1*self.b
        self.features = FeatureVector(numpy.atleast_2d(self.w).astype(
                                      numpy.float64), self.feature_names)
        try:
            wf=[]
            for i,feature in enumerate(self.feature_names):
                if not self.w[i] == 0:
                    wf.append((self.w[i],feature))
            wf.sort()
            w = numpy.array(wf, dtype='|S20')
        except ValueError :
            print 'w could not be converted.'
        except IndexError :
            print 'There are more feature names than features. \
                    Please check your feature generation and input data.'
            self.b = 0
            w = numpy.zeros(self.dim)
            self.w = w
        # only features without zero multiplier are relevant
        self.num_retained_features = len(w) 
        self.classifier_information["~~Num_Retained_Features~~"] =\
            self.num_retained_features
        self.print_w = w


_NODE_MAPPING = {"LibSVM_Classifier": LibSVMClassifierNode,
                "2SVM": LibSVMClassifierNode,
                "lSVM": LiblinearClassifierNode,
                }
