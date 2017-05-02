""" Base classes for classification """

import numpy

# import matplotlib as mpl
# mpl.rcParams['text.usetex']=True
# mpl.rcParams['text.latex.unicode']=True
import matplotlib.pyplot as plt

import os
import cPickle
import logging
import math
import numpy
import os
import timeit
import warnings

# base class

from pySPACE.missions.nodes.base_node import BaseNode
# representation of the linear classification vector
from pySPACE.missions.nodes.decorators import BooleanParameter, QNormalParameter, ChoiceParameter, QUniformParameter, \
    NormalParameter, NoOptimizationParameter, LogUniformParameter, LogNormalParameter, QLogUniformParameter, \
    UniformParameter
from pySPACE.resources.data_types.feature_vector import FeatureVector
# the output is a prediction vector
from pySPACE.resources.data_types.prediction_vector import PredictionVector


@BooleanParameter("regression")
@LogUniformParameter("complexity", min_value=1e-6, max_value=1e3)
@ChoiceParameter("kernel_type", choices=["LINEAR", "POLY", "RBF", "SIGMOID"])
@QNormalParameter("offset", mu=0, sigma=1, q=1)
@UniformParameter("nu", min_value=0.01, max_value=0.99)
@LogNormalParameter("epsilon", shape=0.1 / 2, scale=0.1)
@NoOptimizationParameter("debug")
@QUniformParameter("max_time", min_value=0, max_value=3600, q=1)
@LogNormalParameter("tolerance", shape=0.001 / 2, scale=0.001)
@NoOptimizationParameter("keep_vectors")
@NoOptimizationParameter("use_list")
@NormalParameter("ratio", mu=0.5, sigma=0.5 / 2)
class RegularizedClassifierBase(BaseNode):
    """ Basic class for regularized (kernel) classifiers with extra support in
    the linear case

    This module also implements several concepts of data handling strategies
    to keep the set of training samples limited especially in an online
    learning scenario. These have been used in the *Data Selection Strategies*
    publication. This functionality is currently implemented for the
    LibSVMClassifierNode and the SorSvmNode. It requires to replace the
    *_complete_training*



    **References**

        ========= ==============================================================
        main      source: Data Selection Strategies
        ========= ==============================================================
        author    Krell, M. M. and Wilshusen, N. and Ignat, A. C., and Kim, S. K.
        title     `Comparison of Data Selection Strategies For Online Support Vector Machine Classification <http://dx.doi.org/10.5220/0005650700590067>`_
        book      Proceedings of the International Congress on Neurotechnology, Electronics and Informatics
        publisher SciTePress
        year      2015
        doi       10.5220/0005650700590067
        ========= ==============================================================

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
            Defines an array with two entries to give different complexity
            weight on the two used classes.
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

        :add_type:
            In case the classifier should be retrained, this parameter
            specifies which incoming samples should be added to the training
            set.
            One of the following strings 'ADD_ALL', 'ONLY_MISSCLASSIFIED',
            'ONLY_WITHIN_MARGIN', 'UNSUPERVISED_PROB'.

            - ADD_ALL

                Add all incoming samples.

            - ONLY_MISSCLASSIFIED

                Add only those samples that were misclassified by the current
                decision function.

                **References**

                    ========= ==================================================
                    minor
                    ========= ==================================================
                    author    Bordes, Antoine and Ertekin, Seyda and Weston,
                              Jason and Bottou, L{\'e}on
                    title     Fast Kernel Classifiers with Online and Active
                              Learning
                    journal   J. Mach. Learn. Res.
                    volume    6
                    month     dec
                    year      2005
                    issn      1532-4435
                    pages     1579--1619
                    numpages  41
                    publisher JMLR.org
                    ========= ==================================================

            - ONLY_WITHIN_MARGIN

                Add only samples that lie within the margin of
                the SVM.

                **References**

                    ========= ==================================================
                    main
                    ========= ==================================================
                    author    Bordes, Antoine and Ertekin, Seyda and Weston,
                              Jason and Bottou, L{\'e}on
                    title     Fast Kernel Classifiers with Online and Active
                              Learning
                    journal   J. Mach. Learn. Res.
                    volume    6
                    month     dec
                    year      2005
                    issn      1532-4435
                    pages     1579--1619
                    numpages  41
                    publisher JMLR.org
                    ========= ==================================================

                    ========= ==================================================
                    main
                    ========= ==================================================
                    author    Oskoei, M.A. and Gan, J.Q. and Huosheng Hu
                    booktitle Engineering in Medicine and Biology Society, 2009.
                              EMBC 2009. Annual International Conference of the
                              IEEE
                    title     Adaptive schemes applied to online SVM for BCI
                              data classification
                    year      2009
                    month     Sept
                    pages     2600-2603
                    ISSN      1557-170X
                    ========= ==================================================

            - UNSUPERVISED_PROB

                Classify the label with the current decision function and
                determine how probable this decision is. If it is most likely
                right, which means the probability exceeds a threshold, add the
                sample to the training set.

                **References**

                    ========= ==================================================
                    main
                    ========= ==================================================
                    author    Sp{\"u}ler, Martin and Rosenstiel, Wolfgang and
                              Bogdan, Martin
                    year      2012
                    isbn      978-3-642-33268-5
                    booktitle Artificial Neural Networks and Machine Learning -
                              ICANN 2012
                    volume    7552
                    series    Lecture Notes in Computer Science
                    editor    Villa, AlessandroE.P. and Duch, W\lodzis\law and
                              \'{E}rdi, P\'{e}ter and Masulli, Francesco and
                              Palm, G{\"u}nther
                    title     Adaptive SVM-Based Classification Increases
                              Performance of a MEG-Based Brain-Computer
                              Interface (BCI)
                    publisher Springer Berlin Heidelberg
                    pages     669-676
                    language  English
                    ========= ==================================================

            (*optional, default: "ADD_ALL"*)

        :discard_type:
            In case the classifier should be retrained this parameter
            specifies which samples from the training set should be discarded
            to keep the training set small.
            One of the following strings 'REMOVE_OLDEST', 'REMOVE_FARTHEST',
            'REMOVE_NO_BORDER_POINTS', 'INC', 'INC_BATCH', 'CDT',
            'DONT_DISCARD'.

            - REMOVE_OLDEST

                Remove the oldest sample from the training set.

                **References**

                    ========= ==================================================
                    main
                    ========= ==================================================
                    title     Online weighted LS-SVM for hysteretic structural
                              system identification
                    journal   Engineering Structures
                    volume    28
                    number    12
                    pages     1728 - 1735
                    year      2006
                    issn      0141-0296
                    author    He-Sheng Tang and Song-Tao Xue and Rong Chen and
                              Tadanobu Sato
                    ========= ==================================================

                    ========= ==================================================
                    minor
                    ========= ==================================================
                    author    Van Vaerenbergh, S. and Via, J. and Santamaria, I.
                    booktitle Acoustics, Speech and Signal Processing, 2006.
                              ICASSP 2006 Proceedings. 2006 IEEE International
                              Conference on
                    title     A Sliding-Window Kernel RLS Algorithm and Its
                              Application to Nonlinear Channel Identification
                    year      2006
                    month     May
                    volume    5
                    ISSN      1520-6149
                    ========= ==================================================

                    ========= ==================================================
                    minor
                    ========= ==================================================
                    author    Funaya, Hiroyuki and Nomura, Yoshihiko
                              and Ikeda, Kazushi
                    booktitle ICONIP (1)
                    date      2009-10-26
                    editor    K{\"o}ppen, Mario and Kasabov, Nikola K.
                              and Coghill, George G.
                    isbn      978-3-642-02489-4
                    keywords  dblp
                    pages     929-936
                    publisher Springer
                    series    Lecture Notes in Computer Science
                    title     A Support Vector Machine with Forgetting Factor
                              and Its Statistical Properties.
                    volume    5506
                    year      2008
                    ========= ==================================================

                    ========= ==================================================
                    minor
                    ========= ==================================================
                    title     On-Line One-Class Support Vector Machines. An
                              Application to Signal Segmentation
                    author    Gretton, A and Desobry, F
                    year      2003
                    date      2003-04
                    journal   IEEE ICASSP Vol. 2
                    pages     709--712
                    ========= ==================================================

            - INC

                Don't remove any sample, but retrain the SVM/classifier
                incrementally with each incoming sample.

                **References**

                    ========= ==================================================
                    main
                    ========= ==================================================
                    year      2012
                    isbn      978-3-642-34155-7
                    booktitle Advances in Intelligent Data Analysis XI
                    volume    7619
                    series    Lecture Notes in Computer Science
                    editor    Hollm\'{e}n, Jaakko and Klawonn, Frank
                              and Tucker, Allan
                    title     Batch-Incremental versus Instance-Incremental
                              Learning in Dynamic and Evolving Data
                    publisher Springer Berlin Heidelberg
                    author    Read, Jesse and Bifet, Albert and Pfahringer,
                              Bernhard and Holmes, Geoff
                    pages     313-323
                    ========= ==================================================

            - CDT

                Detect changes in the distribution of the data and adapt the
                classifier accordingly, by throwing old samples away and only
                take the last few for retraining.

                **References**

                    ========= ==================================================
                    main
                    ========= ==================================================
                    author    Alippi, C. and Derong Liu and Dongbin Zhao
                              and Li Bu
                    journal   Systems, Man, and Cybernetics: Systems, IEEE
                              Transactions on
                    title     Detecting and Reacting to Changes in Sensing
                              Units: The Active Classifier Case
                    year      2014
                    month     March
                    volume    44
                    number    3
                    pages     353-362
                    ISSN      2168-2216
                    ========= ==================================================

                    ========= ==================================================
                    minor
                    ========= ==================================================
                    title     Intelligence for embedded systems: a
                              methodological approach
                    author    Cesare Alippi
                    publisher Springer
                    address   Cham [u.a.]
                    year      2014
                    ISBN      978-3-319-05278-6
                    pages     211-247
                    chapter   Learning in Nonstationary and Evolving
                              Environments
                    ========= ==================================================

            - INC_BATCH

                Collect new samples until a basket size is reached. Then throw
                all old samples away. And retrain the classifier with the
                current training set.

                **References**

                    ========= ==================================================
                    main
                    ========= ==================================================
                    year      2012
                    isbn      978-3-642-34155-7
                    booktitle Advances in Intelligent Data Analysis XI
                    volume    7619
                    series    Lecture Notes in Computer Science
                    editor    Hollm\'{e}n, Jaakko and Klawonn, Frank
                              and Tucker, Allan
                    title     Batch-Incremental versus Instance-Incremental
                              Learning in Dynamic and Evolving Data
                    publisher Springer Berlin Heidelberg
                    author    Read, Jesse and Bifet, Albert
                              and Pfahringer,Bernhard and Holmes, Geoff
                    pages     313-323
                    ========= ==================================================

            - DONT_DISCARD

                Don't remove any samples from the training set.

            - REMOVE_FARTHEST

                Remove that sample that is farthest away from the hyperplane.

            - REMOVE_NO_BORDER_POINTS

                Remove all points that are not in the border of their class.

                **References**

                    ========= ==================================================
                    main
                    ========= ==================================================
                    title     Incremental SVM based on reserved set for network
                              intrusion detection
                    journal   Expert Systems with Applications
                    volume    38
                    number    6
                    pages     7698 - 7707
                    year      2011
                    issn      0957-4174
                    author    Yang Yi and Jiansheng Wu and Wei Xu
                    ========= ==================================================

            (*optional, default: "REMOVE_OLDEST"*)

        :keep_only_sv:
            Because only the support vectors determine the decision function
            remove all other samples after the SVM is trained.

            (*optional, default: False*)

        :basket_size:
            Specify the number of training samples for retraining.

            (*optional, default: infinity*)

        :relabel:
            Relabel the training set after the SVM is trained.
            If the parameter is set to *True*, the relabeling is done once.
            Otherwise, if the parameter is set to *conv*
            relabeling is repeated till convergence (with a maximum of
            10 iterations over the complete training data to ensure stopping).
            The maximum number of iterations is reset after each relabeling.

            (*optional, default: False*)

        :border_handling:
            Specify how to determine border points in case the discard_type:
            'REMOVE_ONLY_BORDER_POINTS' is selected.
            One of the following strings 'USE_ONLY_BORDER_POINTS',
            'USE_DIFFERENCE'.

            - USE_ONLY_BORDER_POINTS

                Keep only those points which distance to the center lie within
                a specified range.

            - USE_DIFFERENCE

                Use the difference from the center of the class as criterion
                to determine the border points of the class.

            (*optional, default: USE_ONLY_BORDER_POINTS*)

        :scale_factor_small:
            Factor to specify the distance of the inner border to the center
            of a class.

            This should be smaller than *scale_factor_tall*. ::

                inner border = scale_factor_small * distance between centers

            (*optional, default: 0.3*)

        :scale_factor_tall:
            Factor to specify the distance of the outer border to the center
            of a class.

            This should be greater than *scale_factor_small*. ::

                outer border = scale_factor_tall * distance between centers

            (*optional, default: 0.5*)

        :p_threshold:
            Probability threshold for unsupervised learning. Only data that is
            most likely right (p>p_threshold) classified will be added to
            training set.

            (*optional, default: 0.8*)

        :cdt_threshold:
            Specify a multiple of the amount of support vectors before the SVM
            should be retrained anyway, does not matter if something changed or
            not.

            (*optional, default: 10*)

        :training_set_ratio:
            Handle the ratio of the classes. One of the following strings:
            "DONT_HANDLE_RATIO", "KEEP_RATIO_AS_IT_IS", "BALANCED_RATIO"

            - DONT_HANDLE_RATIO

                Dont handle the ratio between the classes and dont consider
                the class labels of the samples.

            - KEEP_RATIO_AS_IT_IS

                Dont change the ratio between the classes. If a sample from one
                class is added an other sample from the same class will be
                removed from the training set.

            - BALANCED_RATIO

                Try to keep a balanced training set with just as many positive
                samples as negatives.

            (*optional, default: DONT_HANDLE_RATIO"*)

        :u_retrain:
            For the retraining, not the given label is used but it is replaced
            with the prediction of the current classifier. This option is
            interesting, where no true label can be provided and a fake label
            is used instead. It is related to the parameter *p_threshold* and
            the *relabel* parameter. The latter allows for a correction of the
            possibly wrong label and the first avoids to use to
            unsure predictions

            The *retrain* parameter has to be additionally set to *True* for
            this parameter to become really active.

            (*optional, default: False*)

        :show_plot:
            Plot the samples and the decision function.

            (*optional, default: False*)

        :save_plot:
            Save the plot of the samples and the decision function.

            (*optional, default: False*)

        :plot_storage:
            Specify a directory to store the images of the plots.
            If directory does not exists, it will be created.

            (*optional, default: "./plot_storage"*)


    .. note:: Not all parameter effects are implemented for all inheriting
              nodes. Kernels are available for LibSVMClassifierNode and
              partially for other nodes.
              The *tolerance* has only an effect on Liblinear, LibSVM and SOR
              classifier.

    :input:    FeatureVector
    :output:   PredictionVector
    :Author:   Mario Krell (mario.krell@dfki.de)
    :Created:  2012/03/28
    """
    def __init__(self, regression=False,
                 complexity=1, weight=None, kernel_type='LINEAR',
                 exponent=2, gamma=None, offset=0, nu=0.5, epsilon=0.1,
                 class_labels=None, debug=False, max_time=3600,
                 tolerance=0.001,
                 complexities_path=None,
                 keep_vectors=False, use_list=False,
                 multinomial=False,
                 add_type="ADD_ALL",
                 discard_type="REMOVE_OLDEST",
                 keep_only_sv=False,
                 basket_size=numpy.inf,
                 relabel=False,
                 border_handling="USE_ONLY_BORDER_POINTS",
                 scale_factor_small=0.3,
                 scale_factor_tall=0.5,
                 p_threshold=0.8,
                 show_plot=False,
                 save_plot=False,
                 cdt_threshold=10,
                 u_retrain=False,
                 training_set_ratio="DONT_HANDLE_RATIO",
                 plot_storage="./plot_storage",
                 ratio=0.5,
                 **kwargs):

        super(RegularizedClassifierBase, self).__init__(**kwargs)
        # type conversion
        complexity = float(complexity)
        if complexity<1e-10:
            self._log("Complexity (%.42f) is very small."+\
                      "Try rescaling data or check this behavior."\
                      % complexity, level=logging.WARNING)

        if self.is_retrainable() or basket_size != numpy.inf:
            keep_vectors=True

        if class_labels is None:
            class_labels = []

        if ratio < 0.01:
            self._log("Ratio (%.2f) is to small. Setting to 0.01" % ratio)
            ratio = 0.01
        elif ratio > 0.99:
            self._log("Ratio (%.2f) is to large. Setting to 0.99" % ratio)
            ratio = 0.99

        if weight is None:
            weight = [ratio, 1 - ratio]


        ################ Only for printing ###########################
        is_plot_active = False
        scat = None
        scatStandard = None
        scatTarget = None
        surf = None
        is_retraining = False
        is_trained = False

        circleTarget0 = None
        circleTarget1 = None

        circleStandard0 = None
        circleStandard1 = None

        m_counter_i = 0

        ################# Only to store results ########################
        if save_plot == True:
            if show_plot == False:
                plt.ioff()
            # Create storage folder if it does not exists
            try:
                import time
                plot_storage += os.path.sep + time.strftime("%d-%m-%Y_%H_%M_%S")
                os.makedirs(plot_storage)
            except OSError:
                if os.path.exists(plot_storage):
                    pass  # Path should already exists
                else:
                    raise  # Error on creation
        ################################################################

        self.set_permanent_attributes(samples=None, labels=None,
                                      future_samples=[], future_labels=[],
                                      classes=class_labels,
                                      weight=weight,
                                      kernel_type=kernel_type,
                                      complexity=complexity,
                                      exponent=exponent, gamma=gamma,
                                      offset=offset, nu=nu,
                                      epsilon=epsilon, debug=debug,
                                      tolerance=tolerance,
                                      w=None, b=0, dim=None,
                                      feature_names=None,
                                      complexities_path=complexities_path,
                                      regression=regression,
                                      keep_vectors=keep_vectors,
                                      max_time=max_time,
                                      steps=0,
                                      retraining_needed=False,
                                      use_list=use_list,
                                      multinomial=multinomial,
                                      classifier_information={},

                                      add_type=add_type,
                                      discard_type=discard_type,
                                      keep_only_sv=keep_only_sv,
                                      basket_size=basket_size,
                                      relabel=relabel,
                                      border_handling=border_handling,
                                      scale_factor_small=scale_factor_small,
                                      scale_factor_tall=scale_factor_tall,
                                      p_threshold=p_threshold,
                                      u_retrain=u_retrain,

                                      cdt_threshold=cdt_threshold,

                                      training_set_ratio=training_set_ratio,

                                      show_plot=show_plot,
                                      save_plot=save_plot,
                                      plot_storage=plot_storage,

                                      scat=scat,
                                      scatStandard=scatStandard,
                                      scatTarget=scatTarget,
                                      surf=surf,
                                      is_retraining=is_retraining,
                                      is_trained=is_trained,

                                      # parameters for circles around
                                      # first and second class
                                      circleTarget0=circleTarget0,
                                      circleTarget1=circleTarget1,
                                      circleStandard0=circleStandard0,
                                      circleStandard1=circleStandard1,

                                      m_counter_i=m_counter_i,

                                      # collection of classification scores
                                      # for probability fits
                                      decisions=[],

                                      is_plot_active=is_plot_active,
                                      )

    def stop_training(self):
        """ Wrapper around stop training for measuring times """
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
        """ Returns whether this node is trainable """
        return True
    
    def is_supervised(self):
        """ Returns whether this node requires supervised training """
        return True

    def delete_training_data(self):
        """ Check if training data can be deleted to save memory """
        if not (self.keep_vectors or self.is_retrainable()):
            self.samples = []
            self.labels = []
            self.decisions = []

    def __getstate__(self):
        """ Return a pickable state for this object """
        odict = super(RegularizedClassifierBase, self).__getstate__()
        if self.kernel_type == 'LINEAR':
            # if 'labels' in odict:
            #     odict['labels'] = []
            # if 'samples' in odict:
            #     odict['samples'] = []
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
                elif self.w is not None:
                    self.features = FeatureVector(self.w.T, self.feature_names)
                else:
                    self.features = None
            if self.features is not None:
                # This node stores the learned features
                name = "%s_sp%s.pickle" % ("features", self.current_split)
                result_file = open(os.path.join(node_dir, name), "wb")
                result_file.write(cPickle.dumps((self.features,self.b), protocol=2))
                result_file.close()
                name = "%s_sp%s.yaml" % ("features", self.current_split)
                result_file = open(os.path.join(node_dir, name), "wb")
                result_file.write(str(self.features))
                result_file.close()
                del self.features

    def __setstate__(self, sdict):
        """ Restore object from its pickled state """
        super(RegularizedClassifierBase, self).__setstate__(sdict) 
        if self.kernel_type != 'LINEAR':
            # Retraining the svm is not a semantically clean way of restoring
            # an object but its by far the most simple solution 
            self._log("Requires retraining of the classifier") 
            if self.samples is not None:
                self._complete_training()

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
        """ Add a new sample with associated label to the training set.

            In case of neither incremental learning nor the
            restriction of training samples is used,
            add the samples to the training set.
            Otherwise check whether the classifier is already trained and if so
            select an appropriate training set and retrain the classifier.
            If the classifier is not trained, train it when there are enough
            samples available.

            :param data:  A new sample for the training set.
            :type  data:  list of float
            :param class_label:    The label of the new sample.
            :type  class_label:    str
        """
        if not self.is_retrainable() and self.basket_size == numpy.inf:
            self._train_sample(data, class_label)
        else:
            # should not be relevant because first the classifier will be
            # trained if the basket size is reached but after the first training
            # only inc_train should adapt the classifier no matter how many
            # samples are in the training set
            if self.samples is not None and self.is_trained:
                self.adapt_training_set(data, class_label)
            else:
                self._train_sample(data, class_label)
                if self.show_plot or self.save_plot:
                    plt.clf()
                if len(self.samples) >= self.basket_size:
                    if not self.is_trained:
                        self._complete_training()
                        if self.discard_type == "CDT":
                            self.learn_CDT()
                        self.is_trained = True

    def _train_sample(self, data, class_label):
        """ Train the classifier on the given data sample
        
            It is assumed that the class_label parameter
            contains information about the true class the data belongs to.

            :param data:  A new sample for the training set.
            :type  data:  FeatureVector
            :param class_label:    The label of the new sample.
            :type  class_label:    str.
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
                # self.decision is by default set to empty list
                if self.add_type == "UNSUPERVISED_PROB":
                    self.decisions = []
            if self.labels is None:
                self.labels = []
            if self.discard_type == "INC_BATCH":
                self.future_samples = []
                self.future_labels = []
        if class_label not in self.classes and "REST" not in self.classes and \
                not self.regression:
            warnings.warn(
                "Please give the expected classes to the classifier! " +
                "%s unknown. " % class_label +
                "Therefore define the variable 'class_labels' in " +
                "your spec file, where you use your classifier. " +
                "For further info look at the node documentation.")
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

    def train(self, data, label):
        """ Special mapping for multi-class classification

        It enables label filtering for one vs. REST and one vs. one case.
        Furthermore, the method measures time for the training segments.
        """
        # one vs. REST case
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

    def append_sample(self, sample):
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

    def get_own_transformation(self, sample=None):
        """ Use classification function e.g. for visualization in LINEAR case
        """
        if self.kernel_type == 'LINEAR':
            return self.w, self.b, self.feature_names, "linear classifier"

    def _inc_train(self, data, class_label=None):
        """ Manipulation of training set for updating the svm """
        #######################################################################
        if not self.classifier_information.has_key("Inc_iterations"):
            self.classifier_information["Inc_iterations"] = 1
        else:
            self.classifier_information["Inc_iterations"] += 1

        if self.u_retrain:
            class_label = self._execute(data).label

        start_time_stamp = timeit.default_timer()
        #######################################################################

        self.adapt_training_set(data, class_label)

        #######################################################################
        stop_time_stamp = timeit.default_timer()
        if not self.classifier_information.has_key("Retraining_time(classifier)"):
            self.classifier_information["Retraining_time(classifier)"] = \
                stop_time_stamp - start_time_stamp
        else:
            self.classifier_information["Retraining_time(classifier)"] += \
                stop_time_stamp - start_time_stamp
        #######################################################################

    def _batch_retrain(self,data_list, label_list):
        """ Simply adding the new data to the old one an retraining """
        for i in range(label_list):
            self._train(data_list[i], label_list[i])
        # # retraining is now performed in the train method, since method
        # # needs to be retrainable to call _batch_retrain
        # self.stop_training()

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
            return numpy.tanh(self.gamma * h + self.offset)
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
            self.gamma = 1.0 / self.dim
        elif self.kernel_type == 'RBF' and self.gamma is None and \
                not self.regression:
            a = self.labels.count(self.classes.index(self.classes[0]))
            b = self.labels.count(self.classes.index(self.classes[1]))
            if a > b:
                relevant = 1
            else:
                relevant = 0
            relevant_samples = []
            for i, label in enumerate(self.labels):
                if label == relevant:
                    relevant_samples.append(self.samples[i])
            variance = numpy.median(numpy.var(numpy.array(self.samples),
                                              axis=0))
            self.gamma = 0.5/(variance*self.dim)
            self._log(
                "No parameter gamma specified for the kernel. Using: %f."\
                % self.gamma,
                level=logging.WARNING)
        elif self.gamma is None:
            self.gamma = 0.001

    def adapt_training_set(self, data, class_label=None):
        """ Select the samples that should belong to the training set and
            retrain the classifier.

            For incremental training run through four steps.

            1) Add samples to the training set according to some criteria.
            2) Discard samples from the training set according to some criteria.
            3) Retrain the classifier with the current training set.
            4) If used relabel the training set according to the current
               decision function.

            :param data:  A new sample for the training set.
            :type  data:  list of float
            :param class_label:    The label of the new sample.
            :type  class_label:    str
        """

        if (self.show_plot or self.save_plot) and self.is_plot_active == False:
            if self.show_plot:
                plt.ion()
            plt.grid(True)
            if self.show_plot:
                plt.show()
            self.is_plot_active = True

        # In case initial training is not already performed, train
        # classifier and CDT once.
        if self.is_trained == False and self.discard_type != "INC":
            self._complete_training()
            if self.discard_type == "CDT":
                self.learn_CDT()
            self.is_trained = True

        # specify flag for retraining phase
        self.is_retraining = True

        ########################################################################
        # 1) Selection of new data                                             #
        ########################################################################

        [new_data_in_training_set, retraining_required, label] =\
            self.select_new_data(data, class_label)

        ########################################################################
        # 2) Discard data                                                      #
        ########################################################################

        [new_data_in_training_set, retraining_required] =\
            self.discard_data(data, class_label,\
                              new_data_in_training_set, retraining_required,\
                              label)

        ########################################################################
        # 3) Retrain                                                           #
        ########################################################################

        self.retrain(data, class_label,\
                     new_data_in_training_set, retraining_required)

        ########################################################################
        # 4) Relabel training set                                              #
        ########################################################################

        self.relabel_training_set()

        if self.show_plot or self.save_plot:
            self.num_samples = len(self.samples)
            self.visualize()

    def select_new_data(self, data, class_label):
        """ Add the new sample to the training set if it satisfies some
            criteria.

            :param data:  A new sample for the training set.
            :type  data:  list of float
            :param class_label:    The label of the new sample.
            :type  class_label:    str
            :rtype: [flag if new data is in training set, flag if retraining is
                    required (the new point is a potential sv or a removed
                    one was a sv)]
        """
        ret_label = None

        retraining_required = False
        new_data_in_training_set = False

        if self.add_type == "ONLY_MISSCLASSIFIED":
            # get the prediction for the current data
            predictionVec = self._execute(data)
            # only append misclassified data points to training set
            if predictionVec.label != class_label:
                if self.discard_type != "INC":
                    self.add_new_sample(data, class_label)
                    ret_label = self.classes.index(class_label)
                    # no need to check if potential support vector
                    # already confirmed with criteria
                    retraining_required = True
                else:
                    new_data_in_training_set = True
        elif self.add_type == "ADD_ALL":
            # add all incomming samples
            if self.discard_type != "INC":
                self.add_new_sample(data, class_label)
                ret_label = self.classes.index(class_label)
                retraining_required =\
                    self.is_potential_support_vector(data, class_label)
            else:
                new_data_in_training_set = True
        elif self.add_type == "ONLY_WITHIN_MARGIN":
            # append only samples that are within the margin
            # (not on the other side of own border, but really between those
            # borderlines)
            predictionVec = self._execute(data)
            if abs(predictionVec.prediction) < 1.0:
                if self.discard_type != "INC":
                    self.add_new_sample(data, class_label)
                    ret_label = self.classes.index(class_label)
                    retraining_required =\
                        self.is_potential_support_vector(data, class_label)
                else:
                    new_data_in_training_set = True
        elif self.add_type == "UNSUPERVISED_PROB":
            # unsupervised classification
            # only append those samples that were most probably right
            # classified
            for i in numpy.arange(len(self.decisions), self.num_samples):
                predictionVec = self._execute(\
                                    numpy.atleast_2d(self.samples[i]))
                self.decisions.append(predictionVec.prediction)
            # get number of target and standard samples
            prior1 = sum(map(lambda x: x == 1, self.labels))
            prior0 = self.num_samples - prior1
            # get labels as list of trues and falses
            labels = map(lambda x: x == 1, self.labels)

            # calculate the label and the probability for the label of the
            # given data
            [p, label] = self.get_platt_prob(self.decisions,
                                             labels,
                                             prior1, prior0,
                                             data)

            if p > self.p_threshold:
                self.decisions.append(p)
                if self.discard_type != "INC":
                    self.add_new_sample(data, label)
                    ret_label = self.classes.index(label)
                    retraining_required =\
                        self.is_potential_support_vector(data, label)
                else:
                    new_data_in_training_set = True

        return [new_data_in_training_set, retraining_required, ret_label]

    def discard_data(self, data, class_label,\
                     new_data_in_training_set, retraining_required,
                     label=None):
        """ Discard data from training set according to some criteria.

            :param data:  A new sample for the training set.
            :type  data:  list of float
            :param class_label:    The label of the new sample.
            :type  class_label:    str
            :param new_data_in_training_set: flag if new data is in training set
            :type  new_data_in_training_set: bool
            :param retraining_required: flag if retraining is
                    requiered (the new point is a potentiell sv or a removed
                    one was a sv)
            :type  retraining_required: bool
            :rtype: [flag if new data is in training set, flag if retraining is
                    requiered (the new point is a potentiell sv or a removed
                    one was a sv)]

        """
        # Reset retraining_required flag if a new chunk is not full
        if self.discard_type == "INC_BATCH"\
            and len(self.future_samples) < self.basket_size:
            retraining_required = False

        while self.num_samples > self.basket_size\
            and (self.discard_type=="REMOVE_OLDEST"\
                 or self.discard_type=="REMOVE_FARTHEST"):
            if self.discard_type == "REMOVE_OLDEST":
                # remove the oldest sample
                idx = 0# in case "DONT_HANDLE_RATIO"
                if self.training_set_ratio == "KEEP_RATIO_AS_IT_IS":
                    # choose from the training set the oldest sample with the
                    # same label as the added sample
                    idx = next((i for i in numpy.arange(len(self.samples))\
                                if self.labels[i] == label), 0)
                elif self.training_set_ratio == "BALANCED_RATIO":
                    # try to keep the number of samples for each class equal
                    num_target = sum(l == 1 for l in self.labels)
                    num_standard = sum(l == 0 for l in self.labels)
                    if num_target != num_standard:
                        label = (num_target > num_standard)
                    idx = next((i for i in numpy.arange(len(self.samples))\
                                if self.labels[i] == label), 0)
                retraining_required = self.remove_samples([idx])\
                    or retraining_required
            elif self.discard_type == "REMOVE_FARTHEST":
                # remove the sample which distance is maximal to the
                # hyperplane
                samples = self.samples  # in case "DONT_HANDLE_RATIO"
                if self.training_set_ratio == "KEEP_RATIO_AS_IT_IS":
                    # choose only from that samples with the same label as the
                    # added sample
                    samples = []
                    idxs_label = []
                    for i in numpy.arange(len(self.samples)):
                        if self.labels[i] == label:
                            samples.append(self.samples[i])
                            idxs_label.append(i)
                elif self.training_set_ratio == "BALANCED_RATIO":
                    # try to keep the number of samples for each class equal
                    num_target = sum(l == 1 for l in self.labels)
                    num_standard = sum(l == 0 for l in self.labels)
                    if num_target != num_standard:
                        label = (num_target > num_standard)

                    samples = []
                    idxs_label = []
                    for i in numpy.arange(len(self.samples)):
                        if self.labels[i] == label:
                            samples.append(self.samples[i])
                            idxs_label.append(i)

                idx = numpy.argmax(map(\
                        lambda x: abs((self._execute(\
                            numpy.atleast_2d(x))).prediction),\
                        samples))

                if self.training_set_ratio == "KEEP_RATIO_AS_IT_IS" or\
                        self.training_set_ratio == "BALANCED_RATIO":
                    idx = idxs_label[idx]

                retraining_required = self.remove_samples([idx])\
                    or retraining_required

        # TODO: add parameter to specify possible overlap
        #       like x times basket size?
        if self.discard_type == "INC_BATCH"\
                and len(self.future_samples) == self.basket_size:
            # remove all old samples
            self.remove_samples(list(numpy.arange(self.num_samples)))
            # and add all samples from the future knowledge base
            for (d, c_l) in zip(self.future_samples, self.future_labels):
                self.add_new_sample(d, c_l, True)
            # The whole training set changes so retraining is required
            retraining_required = True

        if self.discard_type == "REMOVE_NO_BORDER_POINTS":
            if len(self.samples) < self.basket_size:
                # Don't retrain if basket size is not reached
                retraining_required = False
            elif len(self.samples) == self.basket_size:
                # Retrain if basket size is reached
                retraining_required = True
            if len(self.samples) > self.basket_size:
                # Discard useless data for next iterations
                self.remove_no_border_points(retraining_required)
                retraining_required = False

        if self.discard_type == "CDT":
            # test if a change occurred
            changeDetected = self.change_detection_test(data, class_label)
            # if a change is detected remove old samples
            if changeDetected or (numpy.floor_divide(
                    len(self.future_samples),
                    self.num_samples) > self.cdt_threshold):
                self.remove_samples(numpy.arange(len(self.samples)))
                # if a change is detected or many new samples arrived add
                # current samples to training set
                for (s, l) in zip(self.future_samples, self.future_labels):
                    self.add_new_sample(s, l, True)
                retraining_required = True
            else:
                retraining_required = False

        return [new_data_in_training_set, retraining_required]

    def retrain(self, data, class_label,
                new_data_in_training_set, retraining_required):
        """ Start retraining procedure if the training set changed.

            :param data:  A new sample for the training set.
            :type  data:  list of float
            :param class_label:    The label of the new sample.
            :type  class_label:    str
            :param new_data_in_training_set: flag if new data is in training set
            :type  new_data_in_training_set: bool
            :param retraining_required: flag if retraining is
                    required (the new point is a potential sv or a removed
                    one was a sv)
        """
        if self.classifier_information.has_key("Inc_iterations") and\
            self.classifier_information["Inc_iterations"] == 1:
            self.classifier_information["Retrain_counter"] = 0

        if self.discard_type == "INC" and new_data_in_training_set is True:
            # Incremental training
            self.incremental_training(data, class_label)

            if not self.classifier_information.has_key("Retrain_counter"):
                self.classifier_information["Retrain_counter"] = 1
            else:
                self.classifier_information["Retrain_counter"] += 1
        else:
            if retraining_required:
                # retrain the svm
                self.retrain_SVM()

                if not self.classifier_information.has_key("Retrain_counter"):
                    self.classifier_information["Retrain_counter"] = 1
                else:
                    self.classifier_information["Retrain_counter"] += 1

                if (self.keep_only_sv or
                        # approaches, where data is removed later on
                        self.discard_type == "CDT" or
                        self.discard_type == "INC_BATCH"):
                    # only keep the sv to save memory
                    self.remove_non_support_vectors()

    def relabel_training_set(self):
        """ Relabel the training set according to the current decision function.
        """
        iterations = 1
        while self.relabel:
            changed = False
            # relabel all training samples according to
            # current decision function
            for i in numpy.arange(len(self.samples)):
                predictionVec = self._execute(numpy.atleast_2d(self.samples[i]))
                if self.labels[i] != self.classes.index(predictionVec.label):
                    changed = True
                    self.labels[i] = self.classes.index(predictionVec.label)
                    # only relevant for SOR classification (outsourcing?)
                    if "version" in self.__dict__:
                        for i in range(self.num_samples):
                            if self.version == "matrix":
                                self.M[-1, i] *= -1
                                self.M[i, -1] *= -1
                            # modified from *calculate_weigts_and_class_factors*
                            if self.version in ["samples", "matrix"]:
                                self.bi[i] *= -1
                                self.ci[i] = self.complexity * \
                                    self.weight[self.labels[i]]
                if i < len(self.decisions):
                    self.decisions[i] = predictionVec.prediction
                else:
                    self.decisions.append(predictionVec.prediction)
            if not changed:
                break
            else: # Retrain the svm with the relabeled training set
                self.retrain_SVM()
            if not self.relabel == "conv" or iterations >= 10:
                break
            iterations += 1

    def is_potential_support_vector(self, data, class_label=None):
        """ Check whether the given data could become a support vector

            This is when the data is within, on or on the other side of the
            margin.

            :param data:  A new sample for the training set.
            :type  data:  list of float
            :param class_label:    The label of the new sample.
            :type  class_label:    str
        """
        predictionVec = self._execute(data)

        if class_label is not None:
            if self.classes.index(class_label) == 1:
                return predictionVec.prediction <= 1.0
            else:
                return predictionVec.prediction >= -1.0
        else:
            return True

    def remove_no_border_points(self, retraining_required):
        """ Discard method to remove all samples from the training set that are
            not in the border of their class.

            The border is determined by a minimum distance from the center of
            the class and a maximum distance.

            :param retraining_required: flag if retraining is
                    required (the new point is a potential sv or a removed
                    one was a sv)
        """
        raise NotImplementedError(
            "The node %s does not implement a border point handling." \
            % self.__class__.__name__)

    def add_new_sample(self, data, class_label=None, default=False):
        """ Add a new sample to the training set

            :param data:  A new sample for the training set.
            :type  data:  list of float
            :param class_label:    The label of the new sample.
            :type  class_label:    str
            :param default:  Specifies if the sample is added to the current
                             training set or to a future training set
            :param default:  bool
        """
        raise NotImplementedError(
            "The node %s does not implement a add sample routine." \
            % self.__class__.__name__)

    def remove_samples(self, idxs):
        """ Remove the samples at the given indices from the training set

            :param: idxs: Indices of the samples to remove.
            :type:  idxs: list of int
            :rtype: bool - True if a support vector was removed.
        """
        raise NotImplementedError(
            "The node %s does not implement a remove sample routine." \
            % self.__class__.__name__)

    def remove_non_support_vectors(self):
        """ Remove all samples that are no support vectors """
        raise NotImplementedError(
            "The node %s does not implement a remove SVs routine." \
            % self.__class__.__name__)

    def retrain_SVM(self):
        """ Retrain the svm with the current training set """
        # start retraining process
        self._complete_training()

        self.future_samples = []
        self.future_labels = []

        if self.discard_type == "CDT":
            self.learn_CDT()

    def incremental_training(self, data, class_label):
        """ Warm Start Implementation by Mario Michael Krell

        The saved status of the algorithm, including the Matrix M, is used
        as a starting point for the iteration.
        Only the problem has to be lifted up one dimension.
        """
        raise NotImplementedError(
            "The node %s does not implement incremental training." \
            % self.__class__.__name__)

    def learn_CDT(self):
        """ Learn features of the training set to detect changes in the
            underlying distribution
        """
        raise NotImplementedError(
            "The node %s does not implement a CDT." % self.__class__.__name__)

    def change_detection_test(self, data, class_label=None):
        """ Detect a change of the distribution

            :param data:  A new sample for the training set.
            :type  data:  list of float
            :param class_label:    The label of the new sample.
            :type  class_label:    str
            :rtype: bool - If change detected return True
        """
        raise NotImplementedError(
            "The node %s does not implement a change detection test." \
            % self.__class__.__name__)

    def get_platt_prob(self, deci, label, prior1, prior0, data):
        """ Get a probability for the decision of the svm

            :param deci: List of decision made for each sample.
            :type  deci: list of float
            :param label: List of labels from the previous samples.
            :type  label: list of bool
            :param prior1: Number of samples of class 1
            :type  prior1: int
            :param prior0: Number of samples of class 0
            :type  prior0: int
            :param data: Sample under investigation
            :type  data: list of float
            :rtype: [float, int] - probability and the corresponding label
        """
        [A, B] = self.approximate_AB_for_plat_prob(deci, label, prior1, prior0)

        predictionVec = self._execute(data)
        f = predictionVec.prediction

        fApB = f * A + B
        if fApB >= 0:
            p = numpy.exp(-fApB) / (1.0 + numpy.exp(-fApB))
        else:
            p = 1.0 / (1.0 + numpy.exp(fApB))

        if self.classes.index(predictionVec.label) == 1:
            return [p, predictionVec.label]
        else:
            return [1-p, predictionVec.label]

    def approximate_AB_for_plat_prob(self, deci, label, prior1, prior0):
        """ Approximate the distribution of both classes

            :param deci: List of decision made for each sample.
            :type  deci: list of float
            :param label: List of labels from the previous samples.
            :type  label: list of bool
            :param prior1: Number of samples of class 1
            :type  prior1: int
            :param prior0: Number of samples of class 0
            :type  prior0: int
            :rtype: [float, float] - ([A, B] - parameters of sigmoid)
        """
        # Parameter setting
        maxiter = 100
        # Maximum number of iterations
        minstep = 1e-10
        # Minimum step taken in line search
        sigma = 1e-12
        # Set to any value > 0
        # Construct initial values:    target support in array t,
        #                              initial function value in fval
        hiTarget = (prior1 + 1.0) / (prior1 + 2.0)
        loTarget = 1 / (prior0 + 2.0)
        length = prior1 + prior0  # Total number of data
        t = numpy.zeros(length)
        for i in numpy.arange(length):
            if label[i] > 0:
                t[i] = hiTarget
            else:
                t[i] = loTarget

        A = 0.0
        B = numpy.log((prior0 + 1.0) / (prior1 + 1.0))
        fval = 0.0
        for i in numpy.arange(length):
            fApB = deci[i] * A + B
            if fApB >= 0:
                fval += t[i] * fApB + numpy.log(1 + numpy.exp(-fApB))
            else:
                fval += (t[i] - 1) * fApB + numpy.log(1 + numpy.exp(fApB))

        for it in numpy.arange(maxiter):
            # Update Gradient and Hessian (use H' = H + sigma 1)
            h11 = h22 = sigma
            h21 = g1 = g2 = 0.0
            for i in numpy.arange(length):
                fApB = deci[i] * A + B
                if fApB >= 0:
                    p = numpy.exp(-fApB) / (1.0 + numpy.exp(-fApB))
                    q = 1.0 / (1.0 + numpy.exp(-fApB))
                else:
                    p = 1.0 / (1.0 + numpy.exp(fApB))
                    q = numpy.exp(fApB) / (1.0 + numpy.exp(fApB))
                d2 = p * q
                h11 += deci[i] * deci[i] * d2
                h22 += d2
                h21 += deci[i] * d2
                d1 = t[i] - p
                g1 += deci[i] * d1
                g2 += d1
            if abs(g1) < 1e-5 and abs(g2) < 1e-5:  # Stopping criteria
                break
            # Compute modified Newton directions
            det = h11 * h22 - h21 * h21
            dA = -(h22 * g1 - h21 * g2) / det
            dB = -(-h21 * g1 + h11 * g2) / det
            gd = g1 * dA + g2 * dB
            stepsize = 1
            while stepsize >= minstep:  # Line search
                newA = A + stepsize * dA
                newB = B + stepsize * dB
                newf = 0.0
                for i in numpy.arange(length):
                    fApB = deci[i] * newA + newB
                    if fApB >= 0:
                        newf += t[i] * fApB + numpy.log(1 + numpy.exp(-fApB))
                    else:
                        newf += (t[i] - 1) * fApB + \
                            numpy.log(1 + numpy.exp(fApB))
                if newf < fval + 0.0001 * stepsize * gd:
                    A = newA
                    B = newB
                    fval = newf
                    break  # Sufficient decrease satisfied
                else:
                    stepsize /= 2.0
            if stepsize < minstep:
                self._log(
                    "Line search fails. A= " + str(A) + " B= " + str(B),
                    level=logging.WARNING)
                break
        if it >= maxiter:
            self._log("Reaching maximal iterations", level=logging.WARNING)

        return [A, B]

# ------------------------------------------------------------------------------
# plot routines
# ------------------------------------------------------------------------------
    def __intersect(self, rect, line):
        """ Calculate the points of a line in a given rectangle

            :param rect: Parameters of a rectangle (min x, min y, max x, max y).
            :type  rect: list of float
            :param line: line given as y=a*x+b or a*x+b*y+c=0
            :type  line: list of float
            :rtype: list of pairs of float
        """
        l = []
        xmin, xmax, ymin, ymax = rect
        a, b, c = line

        assert a != 0 or b != 0

        if a == 0:
            y = -c/b
            if y <= ymax and y >= ymin:
                l.append((xmin, y))
                l.append((xmax, y))
            return l
        if b == 0:
            x = -c/a
            if x <= xmax and x >= xmin:
                l.append((x, ymin))
                l.append((x, ymax))
            return l

        k = -a / b
        m = -c / b
        for x in (xmin, xmax):
            y = k * x + m
            if y <= ymax and y >= ymin:
                l.append((x,y))

        k = -b / a
        m = -c / a
        for y in (ymin, ymax):
            x = k * y + m
            if x < xmax and x > xmin:
                l.append((x, y))
        return l

    def plot_line(self, coef, *args, **kwargs):
        """ Plot a line (y=a*x+b or a*x+b*y+c=0) with the given coefficients

            :param coef: Coefficients determining the line
            :type  coef: list of floats
            :rtype: list of lines
        """
        coef = numpy.float64(coef[:])
        assert len(coef) == 2 or len(coef) == 3
        if len(coef) == 2:
            a, b, c = coef[0], -1., coef[1]
        elif len(coef) == 3:
            a, b, c = coef
        ax = plt.gca()

        limits = ax.axis()
        points = self.__intersect(limits, (a,b,c))
        if len(points) == 2:
            pts = numpy.array(points)
            l = ax.plot(pts[:, 0], pts[:, 1], *args, **kwargs)
            ax.axis(limits)
            return l
        return None

    def circle_out(self, x, y, s=20, *args, **kwargs):
        """ Circle out points with size 's'.

            :param x: x coordinates.
            :type  x: list of float
            :param y: y coordinates.
            :type  y: list of float
            :param s: Size of circle
            :tyep  s: int
        """
        ax = plt.gca()
        x = [item for sublist in x for item in sublist]
        y = [item for sublist in y for item in sublist]
        if 'edgecolors' not in kwargs:
            kwargs['edgecolors'] = 'g'
        self.scat = ax.scatter(x, y, s, facecolors='none', *args, **kwargs)

    def plot_data(self, x, y, target, s=20, *args, **kwargs):
        """ Plot points with size 's'

            :param x: x coordinates.
            :type  x: list of float
            :param y: y coordinates.
            :type  y: list of float
            :param target: Determine class label.
            :type  target: bool
            :param s: Size of point.
            :type  s: int
        """
        ax = plt.gca()
        x = [item for sublist in x for item in sublist]
        y = [item for sublist in y for item in sublist]
        if 'edgecolors' not in kwargs:
            if target == True:
                kwargs['edgecolors'] = 'r'
                self.scatTarget = ax.scatter(x, y, s, marker='x',\
                                             facecolors='none',\
                                             *args, **kwargs)
            else:
                kwargs['edgecolors'] = 'b'
                self.scatStandard = ax.scatter(x, y, s, marker='o',\
                                               facecolors='none',\
                                               *args, **kwargs)

    def plot_hyperplane(self):
        """ Plot the hyperplane (in 2D a line).
        """
        ax = plt.gca()
        ax.set_title("$wx + b = 0$\n$[%.4f; %.4f]x + %.4f = 0$"\
                     % (self.w[0], self.w[1], self.b))

        coef = [self.w[0], self.w[1], self.b]

        coef1 = coef[:]
        coef2 = coef[:]
        coef1[2] += 1
        coef2[2] -= 1

        i = 0
        for _, line in enumerate(ax.lines):
            ax.lines.remove(line)
            i += 1
        if i != 3:
            if self.show_plot:
                from time import sleep
                sleep(0.25)
            for _, line in enumerate(ax.lines):
                ax.lines.remove(line)
                i += 1

        self.plot_line(coef, 'b', lw=2)
        self.plot_line(coef1, 'g', lw=1, ls='dashed')
        self.plot_line(coef2, 'r', lw=1, ls='dashed')

    def plot_samples(self):
        """ Plot all training samples.

            Plot all training samples and mark the class association.
        """
        class_neg = []
        class_pos = []
        for idx in numpy.arange(self.num_samples):
            if self.labels[idx] == 0:
                class_neg.append(self.samples[idx])
            else:
                class_pos.append(self.samples[idx])

        class_neg = numpy.matrix(class_neg)
        class_pos = numpy.matrix(class_pos)

        if self.scatStandard is not None:
            self.scatStandard.remove()
            self.scatStandard = None
        if self.scatTarget is not None:
            self.scatTarget.remove()
            self.scatTarget = None

        # TODO: determine size of plot
        xmin = -2.5 #min(numpy.min(class_neg[:,0]), numpy.min(class_pos[:,0]))
        xmax = 2.5  #max(numpy.max(class_neg[:,0]), numpy.max(class_pos[:,0]))
        ymin = -2.5 #min(numpy.min(class_neg[:,1]), numpy.min(class_pos[:,1]))
        ymax = 2.5  #max(numpy.max(class_neg[:,1]), numpy.max(class_pos[:,1]))
        ax = plt.gca()
        ax.axis([xmin-1.0, xmax+1.0, ymin-1.0, ymax+1.0])

        if numpy.shape(class_neg)[1] > 0:
            self.plot_data(class_neg[:, 0], class_neg[:, 1], False)
        if numpy.shape(class_pos)[1] > 0:
            self.plot_data(class_pos[:, 0], class_pos[:, 1], True)

    def plot_support_vectors(self):
        """ Mark the support vectors by a circle.
        """
        support_vectors = []
        for idx in numpy.arange(self.num_samples):
            if self.dual_solution[idx] != 0:
                support_vectors.append(self.samples[idx])

        support_vectors = numpy.matrix(support_vectors)

        if self.scat is not None:
            self.scat.remove()

        if support_vectors is not None and\
            numpy.shape(support_vectors)[0] > 1 and\
            numpy.shape(support_vectors)[1] > 0:
            self.circle_out(support_vectors[:, 0], support_vectors[:, 1], s=100)
        else:
            self.scat = None

    def plot_class_borders(self, mStandard, mTarget, R,
                           scaleFactorSmall, scaleFactorTall):
        """ Plot the borders of each class.

            :param mStandard: Center of standard class.
            :type  mStandard: [float, float] - (x,y)
            :param mTarget: Center of target class.
            :type  mTarget: [float, float] - (x,y)
            :param R: Distance between both centers.
            :type  R: float
            :param scaleFactorSmall: Determine inner circle of class border.
            :type  scaleFactorSmall: float
            :param scaleFactorTall: Determine outer circle of class border.
            :type  scaleFactorTall: float
        """
        ax = plt.gca()
        if self.circleStandard0 is not None:
            self.circleStandard0.remove()
        if self.circleStandard1 is not None:
            self.circleStandard1.remove()
        if self.circleTarget0 is not None:
            self.circleTarget0.remove()
        if self.circleTarget1 is not None:
            self.circleTarget1.remove()

        self.circleStandard0 = plt.Circle(
            mStandard, radius=scaleFactorSmall * R, color='b', fill=False)
        self.circleStandard1 = plt.Circle(
            mStandard, radius=scaleFactorTall * R, color='b', fill=False)

        self.circleTarget0 = plt.Circle(
            mTarget, radius=scaleFactorSmall * R, color='r', fill=False)
        self.circleTarget1 = plt.Circle(
            mTarget, radius=scaleFactorTall * R, color='r', fill=False)

        ax.add_patch(self.circleStandard0)
        ax.add_patch(self.circleStandard1)
        ax.add_patch(self.circleTarget0)
        ax.add_patch(self.circleTarget1)

    def plot_data_3D(self, x, y, z, target, s=20, *args, **kwargs):
        """ Plot points with size 's'

            :param x: x coordinates.
            :type  x: list of float
            :param y: y coordinates.
            :type  y: list of float
            :param z: z coordinates:
            :type  z: list of float
            :param target: Determine class label.
            :type  target: bool
            :param s: Size of point.
            :type  s: int
        """
        ax = plt.gca(projection='3d')
        x = [item for sublist in x for item in sublist]
        y = [item for sublist in y for item in sublist]
        z = [item for sublist in z for item in sublist]
        if 'edgecolors' not in kwargs:
            if target:
                self.scatTarget = ax.scatter(x, y, z,  c='r', marker='o')
            else:
                self.scatStandard = ax.scatter(x, y, z,  c='g', marker='x')

    def plot_samples_3D(self):
        """ Plot all training samples.

            Plot all training samples and mark the class association.
        """
        ax = plt.gca(projection='3d')#generate 3d plot

        # TODO: determine size of plot
        xmin = -2.5  # min(numpy.min(class_neg[:,0]), numpy.min(class_pos[:,0]))
        xmax = 2.5   # max(numpy.max(class_neg[:,0]), numpy.max(class_pos[:,0]))
        ymin = -2.5  # min(numpy.min(class_neg[:,1]), numpy.min(class_pos[:,1]))
        ymax = 2.5   # max(numpy.max(class_neg[:,1]), numpy.max(class_pos[:,1]))
        zmin = -2.5
        zmax = 2.5

        ax.set_xlim3d(xmin, xmax)
        ax.set_ylim3d(ymin, ymax)
        ax.set_zlim3d(zmin, zmax)

        class_neg = []
        class_pos = []
        for idx in numpy.arange(self.num_samples):
            if self.labels[idx] == 0:
                class_neg.append(self.samples[idx])
            else:
                class_pos.append(self.samples[idx])

        class_neg = numpy.matrix(class_neg)
        class_pos = numpy.matrix(class_pos)

        if self.scatStandard is not None:
            self.scatStandard.remove()
            self.scatStandard = None
        if self.scatTarget is not None:
            self.scatTarget.remove()
            self.scatTarget = None

        if numpy.shape(class_neg)[1] > 0:
            self.plot_data_3D(
                class_neg[:, 0], class_neg[:, 1], class_neg[:, 2], False)
        if numpy.shape(class_pos)[1] > 0:
            self.plot_data_3D(
                class_pos[:, 0], class_pos[:, 1], class_pos[:, 2], True)

    def plot_hyperplane_3D(self):
        """ Plot the hyperplane (in 3D a surface).
        """
        ax = plt.gca(projection='3d')
        ax.set_title("$wx + b = 0$\n$[%.4f; %.4f; %.4f]x + %.4f = 0$"\
                     % (self.w[0], self.w[1], self.w[2], self.b))

        if self.surf is not None:
            self.surf.remove()
            self.surf = None

        # create x,y
        xx, yy = numpy.meshgrid(numpy.arange(-2.0, 2.0, 0.05),\
                                numpy.arange(-2.0, 2.0, 0.05))

        # calculate corresponding z
        z = (-self.w[0] * xx - self.w[1] * yy - self.b) * 1. / self.w[2]
        self.surf = ax.plot_surface(xx, yy, z, alpha=0.2)

    def visualize(self):
        """ Show the training samples, the support vectors if possible and the
            current decision function
        """
        raise NotImplementedError("The node %s does not implement a"+ \
                                  "visualization." % self.__class__.__name__)

# ------------------------------------------------------------------------------

class TimeoutException(Exception):
    """ Break up for to long simplex iterations """ 
    pass
