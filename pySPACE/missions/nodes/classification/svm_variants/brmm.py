""" Relative Margin Machines (original and some variants) """

# the output is a prediction vector
import timeit

from pySPACE.missions.nodes.decorators import NoOptimizationParameter, BooleanParameter, QNormalParameter, \
    ChoiceParameter, QLogUniformParameter, LogUniformParameter
from pySPACE.resources.data_types.prediction_vector import PredictionVector
from pySPACE.missions.nodes.classification.base import RegularizedClassifierBase
# classification vector may be saved as a feature vector
from pySPACE.resources.data_types.feature_vector import FeatureVector
from pySPACE.resources.data_types.time_series import TimeSeries
# timeout package .
import signal

import numpy
from numpy import dot

from pySPACE.missions.nodes.classification.base import TimeoutException

import warnings
import logging

import random
import copy

from pySPACE.resources.dataset_defs.metric import BinaryClassificationDataset

from pySPACE.missions.nodes.classification.svm_variants.external import LibSVMClassifierNode

from pySPACE.missions.nodes.base_node import BaseNode


@NoOptimizationParameter("use_list")
@NoOptimizationParameter("omega")
@QLogUniformParameter("max_iterations", min_value=1, max_value=1000, q=1)
@LogUniformParameter("range_", min_value=1.01, max_value=1000)
@ChoiceParameter("offset_factor", choices=[0, 1, 10, 100])
@BooleanParameter("squared_loss")
@NoOptimizationParameter("linear_weighting")
class RMM2Node(RegularizedClassifierBase):
    """ Classify with 2-norm SVM relaxation (b in target function) for BRMM

    The balanced relative margin machine (BRMM) is a modification of the
    original relative margin machine (RMM).
    The details to this algorithm can be found in the given reference.

    This node extends a successive over relaxation algorithm
    for adaption on new data with some variants.

    For further details, have a look at the given reference,
    the *reduced_descent* method
    which is an elemental processing step and the *_inc_train*  method,
    which uses the status of the algorithm as a warm start.


    **References**

        ========= ==================================================
        main     
        ========= ==================================================
        author    Krell, M. M. and Feess, D. and Straube, S.
        title     `Balanced Relative Margin Machine - The Missing Piece Between FDA and SVM Classification <http://dx.doi.org/10.1016/j.patrec.2013.09.018>`_
        journal   Pattern Recognition Letters
        publisher Elsevier
        doi       10.1016/j.patrec.2013.09.018
        year      2014
        ========= ==================================================

    **Parameters**

    Most parameters are already included into the
    :class:`RegularizedClassifierBase <pySPACE.missions.nodes.classification.base.RegularizedClassifierBase>`.

        :random:
            *Numerical recipes* suggests to randomize the order of alpha.
            *M&M* suggest to sort the alpha by their magnitude.

            (*optional, default: False*)

        :omega:
            Descent factor of optimization algorithm. Should be between 0 and 2!
            *Numerical recipes* uses 1.3 and *M&M* choose 1.0.

            (*optional, default: 1.0*)

        :version:
            Using the *matrix* with the scalar products or using only the
            *samples* and track changes in w and b for fast calculations.
            Both versions give totally the same result but they are available for
            comparison.
            Samples is mostly a bit faster.
            For  kernel usage only *matrix* is possible.

            (*optional, default: "samples"*)

        :reduce_non_zeros:
            In the inner loops, indices are rejected, if they loose there support.

            (*optional, default: True*)

        :linear_weighting:
            Add a linear weighting in the loss term.
            We use the index as weight and normalize it to have a total sum of 1.

            (*optional, default: False*)

        :calc_looCV:
            Calculate the leave-one-out metrics on the training data

            (*optional, default: False*)

        :range\_:
            Upper bound for the prediction value before its 'outer loss' is
            punished with `outer_complexity`.

            Using this parameter (with value >1) activates the RMM.

            (*optional, default: numpy.inf*)

        :outer_complexity:
            Cost factor for to high values in classification. (see `range\_`)

            (*optional, default: `complexity`*)

        :offset_factor:
            Reciprocal weight, for offset treatment in the model

                :0: Use no offset
                :1: Normal affine approach from augmented feature vectors
                :high: Only small punishment of offset, enabling larger offsets
                      (*danger of numerical instability*)

            If 0 is used, the offset b is set to zero, otherwise it is used via
            augmented feature vectors with different augmentation factors.
            The augmentation value corresponds to 1/*offset_factor*,
            where 1/0 corresponds to infinity.

            (*optional, default: 1*)

        :squared_loss:
            Use L2 loss (optional) instead of L1 loss (default).

            (*optional, default: False*)

    In the implementation we do not use alpha but dual_solution for the
    variables of the dual optimization problem,
    which is optimized with this algorithm.

    In the RMM case dual solution weights are concatenated.
    The RMM algorithm was constructed in the same way
    as in the mentioned references.

    As a stopping criterion we use the maximum change to be less than some
    tolerance.

    **Exemplary Call**

    .. code-block:: yaml

        -
            node : 2RMM
            parameters :
                complexity : 1.0
                weight : [1,3]
                debug : True
                store : True
                class_labels : ['Standard', 'Target']

    :input:    FeatureVector
    :output:   PredictionVector
    :Author:   Mario Michael Krell (mario.krell@dfki.de)
    :Created:  2012/06/27
    """
    def __init__(self, random=False, omega=1.0,
                 max_iterations=numpy.inf,
                 version="samples", reduce_non_zeros=True,
                 linear_weighting=False, calc_looCV=False,
                 range_=numpy.inf, outer_complexity=None,
                 offset_factor=1, squared_loss=False,
                 co_adaptive=False, co_adaptive_index=numpy.inf, history_index=1,
                 **kwargs):
        self.old_difference = numpy.inf
        # instead of lists, arrays are concatenated in training
        super(RMM2Node, self).__init__(use_list=False, **kwargs)

        if not(version in ["samples", "matrix"]):
            self._log("Version %s is not available. Default to 'samples'!"%version, level=logging.WARNING)
            version = "samples"
        if not self.kernel_type == 'LINEAR' and not version == "matrix":
            self._log(
                "Version %s is not available for nonlinear kernel. " % version +
                "Default to 'matrix'!", level=logging.WARNING)
            version = "matrix"
        range_ = numpy.float64(range_)
        if outer_complexity is None:
            outer_complexity = self.complexity
        # factor to only use relation between complexities
        # instead of outer_complexity
        complexity_correction = 1.0 * outer_complexity / self.complexity
        if self.tolerance > 0.1 * min(self.complexity, outer_complexity):
            self.set_permanent_attributes(tolerance=0.1*min(self.complexity,
                                                            outer_complexity))
            warnings.warn("Using to high tolerance. Reduced to 0.1 times " +
                          "complexity (tolerance=%f)." % self.tolerance)
        # mapping of the binary variables to {0,1}
        if not squared_loss:
            squ_factor = 0.0
        else:
            squ_factor = 1.0

        if version == "matrix":
            M = None
        else:
            M = []

        self.set_permanent_attributes(random=random,
                                      omega=omega,
                                      max_iterations_factor=max_iterations,
                                      max_sub_iterations=numpy.inf,
                                      iterations=0,
                                      sub_iterations=0,
                                      version=version,
                                      M=M,
                                      reduce_non_zeros=reduce_non_zeros,
                                      calc_looCV=calc_looCV,
                                      offset_factor=offset_factor,
                                      squ_factor=squ_factor,
                                      num_samples=0,
                                      outer_complexity=outer_complexity,
                                      complexity_correction=
                                      complexity_correction,
                                      linear_weighting=linear_weighting,
                                      range=range_,  # RBF for zero training
                                      b=0,
                                      w=None,
                                      bi=[],
                                      ci=[],
                                      dual_solution=None,
                                      one_class=False,
                                      co_adaptive=co_adaptive,
                                      co_adaptive_index=co_adaptive_index,
                                      history_index=history_index
                                      )

    def _execute(self, x):
        """ Executes the classifier on the given data vector in the linear case

        prediction value = <w,data>+b
        """
        if self.kernel_type == 'LINEAR':
            if self.w is None:
                self.w = numpy.zeros(x.shape[1], dtype=numpy.float)
            return super(RMM2Node, self)._execute(x)
        data = x.view(numpy.ndarray)
        data = data[0, :]
        prediction = self.b
        for i in range(self.num_samples):
            dual = self.dual_solution[i][0] - self.dual_solution[i][1]
            if not dual == 0:
                prediction += dual*self.kernel_func(data,
                                                    self.samples[i])*self.bi[i]
        # one-class multinomial handling of REST class
        if "REST" in self.classes and self.multinomial:
            if "REST" == self.classes[0]:
                label = self.classes[1]
            elif "REST" == self.classes[1]:
                label = self.classes[0]
                prediction *= -1
        # Look up class label
        # prediction --> {-1,1} --> {0,1} --> Labels
        elif prediction > 0:
            label = self.classes[1]
        else:
            label = self.classes[0]
        return PredictionVector(label=label, prediction=prediction,
                                predictor=self)

    def _stop_training(self, debug=False):
        """ Train the SVM with the SOR algorithm on the collected training data
        """
        self._log("Preprocessing of SOR SVM")
        self._log("Instances of Class %s: %s, %s: %s"
                  % (self.classes[0],
                     self.labels.count(self.classes.index(self.classes[0])),
                     self.classes[1],
                     self.labels.count(self.classes.index(self.classes[1]))))
        ## initializations of relevant values and objects ##
        self.calculate_weigts_and_class_factors()
        self.num_samples = len(self.samples)
        self.max_iterations = self.max_iterations_factor*self.num_samples
        # RMM variable
        self.dual_solution = numpy.zeros((self.num_samples, 2))

        if self.version == "matrix" and self.kernel_type == "LINEAR":
            self.A = numpy.array(self.samples)
            self.D = numpy.diag(self.bi)
            self.M = dot(self.D,
                         dot(dot(self.A, self.A.T) + self.offset_factor *
                             numpy.ones((self.num_samples, self.num_samples)),
                             self.D))
        elif self.version == "samples" and self.kernel_type == "LINEAR":
            if not self.squ_factor:
                self.M = \
                    [1.0/(numpy.linalg.norm(sample)**2.0 + self.offset_factor)
                     for sample in self.samples]
            else:
                self.M = [(1 / (numpy.linalg.norm(self.samples[i])**2.0
                          + self.offset_factor + 1 / (2 * self.ci[i])),
                          1 / (numpy.linalg.norm(self.samples[i])**2.0
                          + self.offset_factor + 1 /
                          (2 * self.ci[i] * self.complexity_correction)))
                          for i in range(self.num_samples)]
            # changes of w and b are tracked in the samples version
            self.w = numpy.zeros(self.dim, dtype=numpy.float)
        else:
            ## iterative calculation of M
            self.M = numpy.zeros((self.num_samples, self.num_samples))
            for i in range(self.num_samples):
                bi = self.bi[i]
                si = self.samples[i]
                for j in range(self.num_samples):
                    if i > j:
                        self.M[i][j] = self.M[j][i]
                    else:
                        self.M[i][j] = bi * self.bi[j] * (
                            self.kernel_func(si, self.samples[j])
                            + self.offset_factor)

        ## SOR Algorithm ##
        self.iteration_loop(self.M)

        self.classifier_information["~~Solver_Iterations~~"] = self.iterations
        try:
            self.classifier_information["~~offset~~"] = self.b
            self.classifier_information["~~w0~~"] = self.w[0]
            self.classifier_information["~~w1~~"] = self.w[1]
        except:
            pass
        ## calculate leave one out metrics ##
        if self.calc_looCV:
            self.looCV()

    def looCV(self):
        """ Calculate leave one out metrics """
        # remember original solution
        optimal_w = copy.deepcopy(self.w)
        optimal_b = copy.deepcopy(self.b)
        optimal_dual_solution = copy.deepcopy(self.dual_solution)
        # preparation of sorting
        sort_dual = self.dual_solution[:, 0] + self.dual_solution[:, 1]
        # sort indices --> zero weights do not need any adaption and
        # low weights are less relevant for changes
        sorted_indices = map(list,[numpy.argsort(sort_dual)])[0]
        sorted_indices.reverse()

        prediction_vectors = []
        using_initial_solution = True
        for index in sorted_indices:
            d_i = self.dual_solution[index, 0]-self.dual_solution[index, 1]
            # delete each index from the current observation
            if d_i == 0 and using_initial_solution:
                # no change in classifier necessary
                pass
            else:
                # set weight to zero and track the corresponding changes
                self.reduce_dual_weight(index)
                # reiterate till convergence but skip current index
                temp_iter = self.iterations
                self.iteration_loop(self.M, reduced_indices=[index])
                self.iterations += temp_iter
                using_initial_solution = False
            prediction_vectors.append((self._execute(
                numpy.atleast_2d(self.samples[index])),
                self.classes[self.labels[index]]))
        self.loo_metrics = BinaryClassificationDataset.calculate_metrics(
            prediction_vectors,
            ir_class=self.classes[1],
            sec_class=self.classes[0])
        #undo changes
        self.b = optimal_b
        self.w = optimal_w
        self.dual_solution = optimal_dual_solution

    def reduce_dual_weight(self,index):
        """ Change weight at index to zero """
        if self.version == "sample":
            old_weight = self.dual_solution[index]
            old_weight = old_weight[0] - old_weight[1]
            self.update_classification_function(delta=-old_weight, index=index)
        else:
            # the matrix algorithm doesn't care for the old weights
            pass
        self.dual_solution[index] = [0, 0]

    def calculate_weigts_and_class_factors(self):
        """ Calculate weights in the loss term and map label to -1 and 1 """
        # Weights for soft margin (dependent on class or time)
        self.ci = []
        # Mapping from class to value of classifier (-1,1)
        self.bi = []
        self.num_samples = len(self.samples)
        for label in self.labels:
            self.append_weights_and_class_factors(label)
            #care for zero sum
        if self.linear_weighting:
            c = 1/(self.num_samples*(self.num_samples+1)*0.5)
            self.ci = [c*value for value in self.ci]

    def append_weights_and_class_factors(self, label):
        """ Mapping between labels and weights/class factors

        The values are added to the corresponding list.
        This is done in a separate function, since it is also needed for adaption.
        """
        self.bi.append(label*2-1)
        if label == 0:
            if self.linear_weighting:
                self.ci.append(self.complexity*self.weight[0]*self.num_samples)
            else:
                self.ci.append(self.complexity*self.weight[0])
        else:
            if self.linear_weighting:
                self.ci.append(self.complexity*self.weight[1]*self.num_samples)
            else:
                self.ci.append(self.complexity*self.weight[1])

    def iteration_loop(self, M, reduced_indices=[]):
        """ The algorithm is calling the :func:`reduced_descent<pySPACE.missions.nodes.classifiers.ada_SVM.SORSVMNode.reduced_descent>` method in loops over alpha

        In the first step it uses a complete loop over all components of alpha
        and in the second inner loop only the non zero alpha are observed till
        come convergence criterion is reached.

        *reduced_indices* will be skipped in observation.
        """
        ## Definition of tracking variables ##
        self.max_iterations = self.max_iterations_factor*self.num_samples
        self.iterations = 0
        self.difference = numpy.inf

        # recalculate factor in case of changing complexities
        self.complexity_correction = \
            1.0 * self.outer_complexity / self.complexity
        # initial call on all samples
        self.total_descent(self.dual_solution, M, reduced_indices)
        ## outer iteration loop ##
        while self.difference > self.tolerance and \
                self.iterations < self.max_iterations:
            # inner iteration loop only on active vectors/alpha (non zero) ##
            self.sub_iterations = 0
            # sorting or randomizing non zero indices
            # arrays are mapped to lists for later iteration
            sort_dual = self.dual_solution[:, 0] + self.dual_solution[:, 1]

            num_non_zeros = len(map(list, sort_dual.nonzero())[0])
            max_values = len(map(list,
                                 numpy.where(sort_dual == sort_dual.max()))[0])
            # sort the entries of the current dual
            # and get the corresponding indices
            sorted_indices = map(list, [numpy.argsort(sort_dual)])[0]
            if num_non_zeros == 0 or num_non_zeros == max_values:
                # skip sub iteration if everything is zero or maximal
                active_indices = []
            else:
                active_indices = sorted_indices[-num_non_zeros:-max_values]
            for index in reduced_indices:
                try:
                    active_indices.remove(index)
                except ValueError:
                    pass
            if self.random:
                random.shuffle(active_indices)
                #min(self.max_iterations_factor, 200) * \
            self.max_sub_iterations = self.max_iterations_factor * \
                len(active_indices) * 0.5
            while (self.difference > self.tolerance and
                   self.sub_iterations < self.max_sub_iterations
                   and self.iterations < self.max_iterations):
                ## iteration step ##
                self.reduced_descent(self.dual_solution, M,
                                         active_indices)
                ## outer loop ##
            if not (self.iterations < self.max_iterations):
                break
            # For the first run, the previous reduced descent is skipped
            # but for adaptivity and looCV it is important
            # to have first the small loop, since normally, this is sufficient.
            # Furthermore having it at the end simplifies the stop criterion
            self.max_sub_iterations = numpy.inf
            self.total_descent(self.dual_solution, M, reduced_indices)
            ## Final solution ##
        # in the case without kernels, we have to calculate the result
        # by hand new for each incoming sample
        if self.version == "matrix":
            dual_diff = self.dual_solution[:, 0] - self.dual_solution[:, 1]
            if self.offset_factor:  # else: keep b fixed and do NOT change
                self.b = self.offset_factor * dot(dual_diff, self.bi)
            if self.kernel_type == "LINEAR":
                self.w = numpy.array([dot(dot(self.A.T, self.D),
                                          dual_diff)]).T
        elif self.version == "samples" and self.kernel_type == "LINEAR":
            # w and b are pre-computed in the loop
            # transferring of 1-d array to 2d array
            # self.w = numpy.array([self.w]).T
            pass

    def reduced_descent(self,current_dual, M, relevant_indices):
        """ Basic iteration step over a set of indices, possibly subset of all

        The main principle is to make a descent step with just one index,
        while fixing the other dual_solutions.

        The main formula comes from *M&M*:

        .. math::

            d        = \\alpha_i - \\frac{\\omega}{M[i][i]}(M[i]\\alpha-1)

            \\text{with } M[i][j]  = y_i y_j(<x_i,x_j>+1)

            \\text{and final projection: }\\alpha_i = \\max(0,\\min(d,c_i)).

        Here we use c for the weights for each sample in the loss term,
        which is normally complexity times corresponding class weight.
        y is used for the labels, which have to be 1 or -1.

        In the *sample* version only the diagonal of M is used.
        The sum with the alpha is tracked by using the classification vector w
        and the offset b.

        .. math::

            o        = \\alpha_i

            d        = \\alpha_i - \\frac{\\omega}{M[i][i]}(y_i(<w,x_i>+b)-1)

            \\text{with projection: }\\alpha_i = \\max(0,\\min(d,c_i)),

            b=b+(\\alpha_i-o)y_i

            w=w+(\\alpha_i-o)y_i x_i
        """
        self.irrelevant_indices = []
        self.difference = 0
        if self.version == "matrix":
            dual_diff = current_dual[:, 0] - current_dual[:, 1]
        for i in relevant_indices:
            if not (self.sub_iterations < self.max_sub_iterations
                    and self.iterations < self.max_iterations):
                break
            beta = False
            old_dual = current_dual[i]
            ### Main Function ###
            ### elemental update step of SOR algorithm ###
            dual_1 = current_dual[i,0]
            dual_2 = current_dual[i,1]
            if self.version == "samples":
                xi = self.samples[i]
                bi = self.bi[i]
                fi = bi * (dot(xi.T, self.w) + self.b)
            elif self.version == "matrix":
                fi = dot(M[i], dual_diff)
                if self.one_class:
                    # correction if offset should not be changed
                    fi += self.bi[i] * self.b
            s1 = self.squ_factor / (2 * self.ci[i]) * dual_1
            s2 = self.squ_factor / (2 * self.ci[i] *
                                    self.complexity_correction) * dual_2
            if dual_1 > 0:  # alpha update
                old_dual = dual_1
                if self.version == "matrix":
                    x = old_dual - \
                        self.omega / (M[i][i]
                                      + self.squ_factor / (2 * self.ci[i])) \
                        * (fi - 1 + s1)
                elif self.version == "samples" and self.squ_factor:
                    x = old_dual - self.omega * M[i][0] * (fi - 1 + s1)
                elif self.version == "samples" and not self.squ_factor:
                    x = old_dual - self.omega * M[i] * (fi - 1)
            elif dual_2 > 0:  # beta update
                old_dual = dual_2
                if self.version == "matrix":
                    x = old_dual - \
                        self.omega / (M[i][i] + self.squ_factor /
                        (2 * self.ci[i] * self.complexity_correction)) \
                        * (self.range - fi + s2)
                elif self.version == "samples" and self.squ_factor:
                    x = old_dual - self.omega * M[i][1] \
                        * (self.range - fi + s2)
                elif self.version == "samples" and not self.squ_factor:
                    x = old_dual - self.omega * M[i] \
                        * (self.range - fi)
                beta = True
            else:
                # both values are zero and we need to check,
                #if one gets active alpha or beta dual coefficient
                old_dual = 0 # just used to have the same formulas as above
                if self.version == "matrix":
                    x = old_dual - \
                        self.omega/(M[i][i]
                                    + self.squ_factor / (2 * self.ci[i])) \
                        * (fi - 1 + s1)
                elif self.version == "samples" and self.squ_factor:
                    x = old_dual - self.omega * M[i][0] * (fi - 1 + s1)
                elif self.version == "samples" and not self.squ_factor:
                    x = old_dual - self.omega * M[i] * (fi - 1)
                # no update of alpha but update of beta (eventually zero update)
                if not x > 0:
                    beta = True
                    if self.version == "matrix":
                        x = old_dual - \
                            self.omega / (M[i][i] + self.squ_factor /
                            (2 * self.ci[i] * self.complexity_correction)) \
                            * (self.range - fi + s2)
                    elif self.version == "samples" and self.squ_factor:
                        x = old_dual - self.omega * M[i][1] * (
                            self.range - fi + s2)
                    elif self.version == "samples" and not self.squ_factor:
                        x = old_dual - self.omega * M[i] * (
                            self.range - fi + s2)
            # map dual solution to the interval [0,C] in L1 case or
            # just make it positive in the L2 case
            # current_dual[i]=self.project(x,index=i)
            if x <= 0:
                self.irrelevant_indices.append(i)
                current_dual[i] = [0, 0]
            elif not beta and not self.squ_factor:
                current_dual[i, 0] = min(x, self.ci[i])
            elif beta and not self.squ_factor:
                current_dual[i, 1] = min(x, self.ci[i] *
                                         self.complexity_correction)
            elif not beta and self.squ_factor:
                current_dual[i, 0] = x
            elif beta and self.squ_factor:
                current_dual[i, 1] = x
            if self.version == "matrix":
                old_diff = dual_diff[i]
                dual_diff[i] = current_dual[i, 0] - current_dual[i, 1]
                delta = dual_diff[i] - old_diff
                # update w and b in samples case
            if self.version == "samples":
                delta = (current_dual[i, 0] + current_dual[i, 1]
                         - old_dual) * bi
                # for beta:  difference needed
                if beta:
                    delta = -delta
                # update classification function parameter w and b
                # self.update_classification_function(delta=delta, index=i)
                self.b += self.offset_factor * delta
                self.w += delta * xi
            current_difference = numpy.abs(delta)
            # if current_difference > self.difference:
            #     self.difference = current_difference
            self.difference += current_difference
            self.sub_iterations += 1
            self.iterations += 1
        if self.reduce_non_zeros:
            for index in self.irrelevant_indices:
                relevant_indices.remove(index)
        if self.random:
            random.shuffle(relevant_indices)

    def update_classification_function(self, delta, index):
        """ update classification function parameter w and b """
        bi=self.bi[index]
        self.b = self.b + self.offset_factor * delta * bi
        self.w = self.w + delta * bi * self.samples[index]

    def project(self,value,index):
        """ Projection method of *soft_relax* """
        if value<=0:
            self.irrelevant_indices.append(index)
            return 0
        else:
            return min(value, self.ci[index])

    def total_descent(self,current_dual,M,reduced_indices=[]):
        """ Different sorting of indices and iteration over all indices

        .. todo:: check, which parameters are necessary
        """
        if not self.random:
            sorted_indices = range(self.num_samples)
            # if current_dual.ndim == 2:
            #     sort_dual = current_dual[:,0]+current_dual[:,1]
            # else:
            #     sort_dual = current_dual
            # # highest first
            # sort_dual = sort_dual * -1.0
            # # sort the entries of the current dual and get the corresponding indices
            # sorted_indices = map(list,[numpy.argsort(sort_dual)])[0]
            # #sorted_indices.reverse()
        else:
            sorted_indices = range(self.num_samples)
            random.shuffle(sorted_indices)
        for index in reduced_indices:
            sorted_indices.remove(index)
        self.reduced_descent(current_dual, M,sorted_indices)

    def _inc_train(self,data,label):
        """ Warm Start Implementation by Mario Michael Krell

        The saved status of the algorithm, including the Matrix M, is used
        as a starting point for the iteration.
        Only the problem has to be lifted up one dimension.
        """
        #one vs. REST case
        if "REST" in self.classes and not label in self.classes:
            label = "REST"
        # one vs. one case
        if not self.multinomial and len(self.classes) == 2 and \
                not label in self.classes:
            return
        self._train(data, label)
        # here it is important to use the mapped label
        self.append_weights_and_class_factors(self.labels[-1])
        if self.linear_weighting:
            self.ci[-1] = self.ci[-1] * 2 / (
                self.num_samples * (self.num_samples + 1))
            scaling = self.num_samples/(self.num_samples+2)
            self.ci = [value * scaling for value in self.ci]
            self.b *= scaling
            self.w *= scaling

        self.num_samples += 1

        # The new example is at first assumed to be irrelevant (zero weight).
        if self.dual_solution is None:
            self.dual_solution = numpy.zeros((1,2))
        else:
            self.dual_solution = numpy.append(self.dual_solution, [[0.0, 0.0]],
                                              axis=0)
        # update of the relevant matrix
        if self.version == "matrix":
            # very inefficient!!!
            M = self.M
            self.M = numpy.zeros((self.num_samples,self.num_samples))
            if not M is None:
                self.M[:-1, :-1] = M
            del M
            bj = self.bi[-1]
            d = self.samples[-1]
            # calculation of missing entries of matrix M by hand
            for i in range(self.num_samples):
                if self.kernel_type == "LINEAR":
                    # y_i*y_j*(<x_i,x_j>+1)
                    self.M[-1, i] = bj*self.bi[i]*(
                        self.kernel_func(d, self.samples[i])+self.offset_factor)
                    self.M[i, -1] = self.M[-1, i]
                else:
                    raise NotImplementedError
        elif self.version == "samples":
            # very efficient :)
            if not self.squ_factor:
                self.M.append(1.0/(numpy.linalg.norm(self.samples[-1])**2.0
                                   + self.offset_factor))
            else:
                self.M.append(
                    (1 / (numpy.linalg.norm(self.samples[-1])**2.0
                          + self.offset_factor + 1 / (2 * self.ci[-1])),
                     1 / (numpy.linalg.norm(self.samples[-1])**2.0
                          + self.offset_factor + 1 /
                          (2 * self.ci[-1] * self.complexity_correction))
                    ))
        prediction = self._execute(data)
        if (not prediction.label == label or abs(prediction.prediction) < 1 or
                (self.range and abs(prediction.prediction) > self.range)):
            if self.co_adaptive:
                self.update_data()
            if self.version == "matrix":
                # relevant parameters for getting w and b
                # updates should be done using old variables
                self.A = numpy.array(self.samples)
                self.D = numpy.diag(self.bi)
            temp_iter = self.iterations
            self.iteration_loop(self.M)
            self.iterations += temp_iter

    def reiterate(self, parameters):
        """ Change the parameters in the model and reiterate """
        changed_c = False
        for parameter in parameters:
            p = parameter.strip("~")
            if p.startswith("log"):
                value = 10 ** parameters[parameter]
                p = p[4:]
            else:
                value = parameters[parameter]
            if p in ["range", "R", "Range", "range_"]:
                self.range = value
            elif p in ["complexity","C", "Complexity"] \
                    and not value == self.complexity:
                changed_c = True
                difference = abs(self.complexity-value)
                self.complexity = value
                if self.tolerance > 0.1 * difference:
                    self.set_permanent_attributes(tolerance=0.1*difference)
                # # fix ci
                # scaling = 1.0 * self.complexity / old_complexity
                # self.ci = [value * scaling for value in self.ci]
            elif p in ["w1", "W1", "weight1", "Weight1"] \
                    and not value == self.weight[0]:
                changed_c = True
                difference = abs(self.weight[0]-value)
                if self.tolerance > 0.1 * difference:
                    self.set_permanent_attributes(tolerance=0.1*difference)
                self.weight[0] = value
            elif p in ["w2", "W2", "weight2", "Weight2"] \
                    and not value == self.weight[1]:
                changed_c = True
                difference = abs(self.weight[1]-value)
                if self.tolerance > 0.1 * difference:
                    self.set_permanent_attributes(tolerance=0.1*difference)
                self.weight[1] = value
        # recalculate ci
        if changed_c:
            self.calculate_weigts_and_class_factors()
        # fix M
        if self.version == "samples" and self.kernel_type == "LINEAR" \
                and self.squ_factor and changed_c:
            self.M = [(1 / (numpy.linalg.norm(self.samples[i])**2.0
                      + self.offset_factor + 1 / (2 * self.ci[i])),
                      1 / (numpy.linalg.norm(self.samples[i])**2.0
                      + self.offset_factor + 1 /
                      (2 * self.ci[i] * self.complexity_correction)))
                      for i in range(self.num_samples)]
        self.iteration_loop(self.M)

    def append_sample(self, sample):
        super(RMM2Node, self).append_sample(sample)
        if self.co_adaptive:
            try:
                hist_data = copy.deepcopy(sample.history[self.history_index-1])
            except IndexError:
                self._log("No history data available for classifier! " +
                          "Co-adaptivity is turned off.",
                          level=logging.CRITICAL)
                self.co_adaptive = False
            if not "hist_samples" in self.__dict__:
                self.hist_samples = []
            self.hist_samples.append(hist_data)

    def update_data(self):
        """ Recalculate w, samples and M """
        # initialize w
        self.w = numpy.zeros(self.dim, dtype=numpy.float)
        for i in range(self.num_samples):
            # recalculate sample
            new_sample = self.get_previous_execute(
                data=self.hist_samples[i], number=self.co_adaptive_index)
            data_array = new_sample.view(numpy.ndarray)
            self.samples[i] = data_array[0, :]
            # recalculate w
            self.w += data_array[0, :] * self.bi[i] * \
                (self.dual_solution[i][0] - self.dual_solution[i][1])
        if not self.squ_factor:
            self.M = \
                [1.0/(numpy.linalg.norm(sample)**2.0 + self.offset_factor)
                 for sample in self.samples]
        else:
            self.M = [(1 / (numpy.linalg.norm(self.samples[i])**2.0
                      + self.offset_factor + 1 / (2 * self.ci[i])),
                      1 / (numpy.linalg.norm(self.samples[i])**2.0
                      + self.offset_factor + 1 /
                      (2 * self.ci[i] * self.complexity_correction)))
                      for i in range(self.num_samples)]


# ===========================================================================
@BooleanParameter("normalize_C")
@QNormalParameter("range", mu=2, sigma=1, q=1)
class RMM1ClassifierNode(RegularizedClassifierBase):
    """ Classify with 1-Norm SVM and relative margin

    Implementation via Simplex Algorithms for exact solutions.
        
    It is important, that the data is reduced
    and has not more then 2000 features.

    This algorithm is an extension of the original RMM as outlined in the
    reference.

    **References**

        ========= ==================================================
        main     
        ========= ==================================================
        author    Krell, M. M. and Feess, D. and Straube, S.
        title     `Balanced Relative Margin Machine - The Missing Piece Between FDA and SVM Classification <http://dx.doi.org/10.1016/j.patrec.2013.09.018>`_
        journal   Pattern Recognition Letters
        publisher Elsevier
        doi       10.1016/j.patrec.2013.09.018
        year      2013
        ========= ==================================================
    
    **Parameters**
        :complexity:
            Complexity sets the weighting of punishment for misclassification
            in comparison to generalizing classification from the data.
            Value in the range form 0 to infinity.

            (*optional, default: 1*)

        :outer_complexity:
            Outer complexity sets the weighting of punishment being outside
            the *range*
            in comparison to generalizing classification from the data
            and the misclassification above.
            Value in the range form 0 to infinity.
            By default it uses the outer complexity.
            For using infinity, use numpy.inf, a string containing *inf* or
            a negative value.

            (*recommended, default: complexity_value*)

        :weight:
            Defines parameter for class weights.
            I is an array with two entries. 
            Set the parameter C of class i to weight*C.

            (*optional, default: [1,1]*)
            
        :range:
            Defines constraint radius for the outer boarder of the classification.
            Going to infinity, this classifier will be identical to the 1 Norm
            SVM. This parameter should be always greater then one.
            Going to one, the classifier will be a variant of the
            Regularized Linear Discriminant analysis.

            (*optional, default: 2*)

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


        :debug:
            If *debug* is True one gets additional output 
            concerning the classification.

            (*optional, default: False*)

        :store:
            Parameter of super-class. If *store* is True,
            the classification vector is stored as a feature vector.

            (*optional, default: False*)
            
    **Exemplary Call**
    
    .. code-block:: yaml
    
        -
            node : 1RMM
            parameters :
                complexity : 1.0
                weight : [1,2.5]
                debug : False
                store : True
                range : 2

    :input:    FeatureVector
    :output:   PredictionVector
    :Author: Mario Krell (Mario.krell@dfki.de)
    :Revised: 2010/04/09
    
    """
    def __init__(self, normalize_C=False, outer_complexity=None, range=2,
                 **kwargs):
        super(RMM1ClassifierNode, self).__init__(**kwargs)

        # Check if old parameter name 'sensitivity' was used instead of range
        if 'sensitivity' in kwargs:
            range = kwargs['sensitivity']
            self._log("Please use parameter 'range' instead of 'sensitivity'" +
                      " in 1RMM Node!", level=logging.ERROR)
            
        if outer_complexity is None:
            self._log("No outer complexity given, set to complexity parameter!",
                      level=logging.WARNING)
            outer_complexity = self.complexity
        elif outer_complexity < 0 or outer_complexity == numpy.inf or \
                (type(outer_complexity) == str and "inf" in outer_complexity):
            outer_complexity = None
            
        self.set_permanent_attributes(outer_complexity=outer_complexity,
                                      normalize_C=normalize_C,
                                      range=range)

    # Here we define a function that defines the matrix, so that we can
    # insert it as an argument for the solver and optimize it
    def create_problem_matrix(self, n, e):
        # First, the declaration of variables
        
        # We only need to define within the function those variables which we
        # don't need in the _stop_training() function
        ci = []
        co = []
        bi = numpy.array(self.bi)
        for label in self.labels:
            if label == 0:
                if not self.normalize_C:
                    ci.append(1.0/(1.0*self.complexity*self.weight[0]))
                    co.append(1.0/(1.0*self.outer_complexity*self.weight[0]))
                else:
                    ci.append(1.0/(1.0*self.complexity*self.weight[0]/n))
                    co.append(1.0/(1.0*self.outer_complexity*self.weight[0]/n))
            else:
                if not self.normalize_C:
                    ci.append(1.0/(1.0*self.complexity*self.weight[1]))
                    co.append(1.0/(1.0*self.outer_complexity*self.weight[1]))
                else:
                    ci.append(1.0/(1.0*self.complexity*self.weight[1]/n))
                    co.append(1.0/(1.0*self.outer_complexity*self.weight[1]/n))
        
        if not self.outer_soft_margin:
            co = n*[0]

        # Now we start defining the matrix by chunks, as follows
        #
        #
        #
        #         dim        dim     1       n
        #      *********  *********  *  d**-C*...***
        #      *********  *********  *  *i*-I*...***
        #   n  ***(-D)**  ***(D)***  B     .
        #      *********  *********  I       .
        #      *********  *********  *  ****...**a**
        #      *********  *********  *  ****...***g*
        #                               
        #         dim        dim     1       n
        #      *********  *********  *  d**-C*...***
        #      *********  *********  *  *i*-O*...***
        #   n  ***(D)***  ***(-D)**  B     .
        #      *********  *********  I       .
        #      *********  *********  *  ****...**a**
        #      *********  *********  *  ****...***g*
        #
        #      -1**************...******************    
        #      *-1*************...******************
        #      **-1************...******************
        #          .
        #             .
        # 2dim            .
        #  +n                 .
        #      ****************-1***+****...********
        #      *****************-1**+****...********
        #      ******************-1*+****...********
        #      *******************-1+****...********
        #      ********************+-1***...********
        #      ********************+*-1**...********
        #      ********************+**-1*...********
        #                                .
        #                                   .
        #      ********************+****...******-1*
        #      ********************+****...*******-1
        #
        # Note : there is a skipped vertical row, marked by the "+",
        # at 2*dim + 1 in the matrix.

        output_matrix = numpy.zeros((2*self.dim+3*n,2*self.dim+n+1), 'd')

        output_matrix[:n, :self.dim] = \
            - numpy.multiply(numpy.array(self.samples).T, bi).T
        output_matrix[n:2*n, :self.dim] = - output_matrix[:n, :self.dim]
        output_matrix[:n, self.dim:2*self.dim] = output_matrix[n:2*n, :self.dim]
        output_matrix[n:2*n, self.dim:2*self.dim] = output_matrix[:n, :self.dim]
        output_matrix[:n, 2*self.dim] = - numpy.array(self.bi)
        output_matrix[n:2*n, 2*self.dim] = output_matrix[:n, 2*self.dim]
        output_matrix[:n, 2*self.dim+1:2*self.dim+n+1] = - numpy.diag(ci)
        output_matrix[n:2*n, 2*self.dim+1:2*self.dim+n+1] = - numpy.diag(co)
        output_matrix[2*n:2*self.dim+2*n, :2*self.dim] = - numpy.diag(e+e)
        output_matrix[2*self.dim+2*n:, 2*self.dim+1:] = - numpy.diag(n*[1.0])

        return output_matrix


    def _stop_training(self, debug = False):
        """ Finish the training, i.e. train the SVM.

        This makes the same as the 1-Norm RMM,
        except that there are additional restrictions pushing the classification
        into two closed intervals instead of two open.
        At both ends there is the same kind of soft Margin."""
        self._log("Preprocessing of 1-Norm RMM")
        self._log("Instances of Class %s: %s, %s: %s"
                  % (self.classes[0],
                     self.labels.count(self.classes.index(self.classes[0])),
                     self.classes[1],
                     self.labels.count(self.classes.index(self.classes[1]))))
        # Dimension of the data
        self.num_samples = len(self.samples)
        n = self.num_samples
        e = self.dim*[1.0]
        if self.outer_complexity is None:
            self.outer_soft_margin = False
            self.outer_complexity = self.complexity  # arbitrary value
        else:
            self.outer_soft_margin = True
        # Mapping from class to value of classifier
        self.bi = []
        for label in self.labels:
            #if self.classes.index(label) == 0:
            if label == 0:
                self.bi.append(-1)
            else:
                self.bi.append(1)
        
        # Import the necessary optimization module.
        try:
            import cvxopt
            import cvxopt.solvers
        except ImportError:
            raise Exception("Using the 1-Norm-SVM requires the Python CVXOPT module.")
        
        self._log("optimization preparation")
        # Target function (w+ + w- + \sum t_i)
        # Weighting is done in the inequalities
        c = cvxopt.matrix(e+e+[0]+n*[1.0])
        
        # First the n classification restrictions
        # then  the n restrictions for the outer margin
        # and finally the restrictions for positive variables
        h = numpy.hstack((-numpy.ones(n,'d'), self.range*numpy.ones(n,'d'),
                          numpy.zeros(2*self.dim+n)))
        h = cvxopt.matrix(h)
        
        #Suppress printing of GLPK and cvxopt
        if not self.debug:
            cvxopt.solvers.options['LPX_K_MSGLEV'] = 0
            cvxopt.solvers.options['show_progress'] = self.debug
            # try:
            #     import mosek
            #     cvxopt.solvers.options['MOSEK'] = {mosek.iparam.log: 0}
            # except:
            #     warnings.warn('No Mosek import possible!')

        # Do the optimization
        # (-c_i x_i,c_i x_i, -c_i,-e_i) * (w+,w-,b,t) \leq -1
        # (c_i x_i,-c_i x_i, c_i,-e_i) * (w+,w-,b,t) \leq R
        # (-w+,-w-,-t) \leq 0
        self._log("Classifier under construction")
        
        if not self.max_time is None:
            cvxopt.solvers.options['LPX_K_TMLIM'] = int(self.max_time)
        
        self.sol = cvxopt.solvers.lp(
            c,
            cvxopt.matrix(self.create_problem_matrix(n, e)), h, solver='glpk')

        self._log("Construction complete")

        model = []
        self.calculate_classification_vector(model)
        if self.debug:
            self.calculate_slack_variables(model)
        
            inner_dual_sol = self.sol['z'][:n]
            dual_inner_weights = numpy.multiply(numpy.array(inner_dual_sol).T,
                numpy.array(self.bi))[0]
            self.max_inner_weight = []
            self.max_inner_weight.append(-dual_inner_weights.min())
            self.max_inner_weight.append(dual_inner_weights.max())
            if self.debug:
                print "Maximal used inner weights, depending on class:"
                print self.max_inner_weight
            outer_dual_sol = self.sol['z'][n:2*n]
            dual_outer_weights = numpy.multiply(numpy.array(outer_dual_sol).T,
                numpy.array(self.bi))[0]
            self.max_outer_weight = []
            self.max_outer_weight.append(-dual_outer_weights.min())
            self.max_outer_weight.append(dual_outer_weights.max())
            if self.debug:
                print "Maximal used inner weights, depending on class:"
                print self.max_outer_weight
            self.max_weight = []
            self.max_weight.append(max(self.max_outer_weight[0],self.max_inner_weight[0]))
            self.max_weight.append(max(self.max_outer_weight[1],self.max_inner_weight[1]))
            if self.debug:
                print "Maximal used weights, depending on class:"
                print self.max_weight
            print "RMM 1 parameter results:"
            self.print_variables()
            print str(self.outer_margin), " vectors of ", \
                str(self.num_samples), \
                " have been used for the outer margin and "
            numpy.set_printoptions(edgeitems=100, linewidth=75, precision=2,
                                   suppress=True, threshold=1000)
            print self.to, " are the outer Slack variables."
            numpy.set_printoptions(edgeitems=3, infstr='Inf', linewidth=75,
                                   nanstr='NaN', precision=8, suppress=False,
                                   threshold=1000)
            print "The number of support vectors(", self.num_sv, \
                ") can be split into", self.num_osv, " and ", self.num_isv, \
                "for the outer and inner margin support vectors."
        
        self.delete_training_data()

    # algorithms to stop process after a certain time
    # from http://pguides.net/python-tutorial/python-timeout-a-function/
    # Mainly Code copy from 1SVM
    def get_solution_with_timeout(self,c,n,e,h):
        def timeout_handler(signum, frame):
            raise TimeoutException()
        
        # Import the necessary optimization module.
        try:
            import cvxopt
            import cvxopt.solvers
        except ImportError:
            raise Exception("Using the 1-Norm-RMM requires the Python CVXOPT module.")
        
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(self.max_time) # trigger alarm in self.time seconds
     
        try:
            self.sol = cvxopt.solvers.lp(
                c, cvxopt.matrix(self.create_problem_matrix(n,e)),
                                 h, solver='glpk')
        except TimeoutException:
            self._log("Classifier construction interrupted!", level=logging.CRITICAL)
            self.sol = None
            return
        finally:
            signal.signal(signal.SIGALRM, old_handler) 
     
        signal.alarm(0)
        self._log("Finished classifier construction successfully.")
        return

    def calculate_slack_variables(self,model):
        """ Calculate slack variables from the given SVM model """
        self.t=self.sol['x'][2*self.dim+1:]
        # differentiation between inner and outer soft margin vectors
        self.num_sv = 0
        self.num_osv = 0
        self.num_isv = 0
        # count the margin vectors lying not at the border of the margin
        # but inside
        self.inner_margin = 0
        self.outer_margin = 0
        self.ti = []
        self.to = []
        for i in range(self.num_samples):
            p = 2*(self.labels[i]-0.5)*((numpy.dot(self.w.T,
                                                   self.samples[i]))[0]+self.b)
            if p >= self.range:
                self.ti.append(0)
                self.num_sv += 1
                self.num_osv += 1
                if p < 1e-5+self.range:
                    self.to.append(0)
                else:
                    self.to.append(p-self.range)
                    self.outer_margin += 1
            elif p > 1:
                self.ti.append(0)
                self.to.append(0)
            else:
                self.to.append(0)
                self.num_sv += 1
                self.num_isv += 1
                if (1-p) < 1e-9:
                    self.ti.append(0)
                else:
                    self.ti.append(1-p)
                    self.inner_margin += 1

    def calculate_classification_vector(self, model):
        """ Copy from Norm1ClassifierNode due to avoid cross importing """
        w = numpy.zeros((self.dim,1))
        self.num_retained_features = 0
        try:
            self.b = float(self.sol['x'][2*self.dim])
            self.w = numpy.array(self.sol['x'][:self.dim]
                                 - self.sol['x'][self.dim:2*self.dim])
        except TypeError:
            self.b = 0
            self.w = w
            self._log('Classification failed. C=%f'%self.complexity,
                      level=logging.CRITICAL)
        
        self.features = FeatureVector(numpy.atleast_2d(self.w.T).astype(
            numpy.float64), self.feature_names)
        try:
            wf = []
            for i,feature in enumerate(self.feature_names):
                if not float(self.w[i]) == 0:
                    wf.append((float(self.w[i]), feature))
            wf.sort(key=lambda (x, y): abs(x))
            if not len(wf) == 0 :
                w = numpy.array(wf,dtype='|S200')
            self.num_retained_features = len(wf)
        except ValueError :
            print 'w could not be converted.'
        except IndexError :
            print 'There are more feature names than features. Please check your feature generation and input data.'
            self.b = 0
            w = numpy.zeros(self.dim)
            self.w = w
        self.classifier_information["~~Num_Retained_Features~~"] = \
            self.num_retained_features
        self.print_w = w


# ===========================================================================


class SVR2BRMMNode(LibSVMClassifierNode):
    """ Simple wrapper around epsilon-SVR with BRMM parameter mapping """

    def __init__(self, range=3, complexity=1, **kwargs):
        super(SVR2BRMMNode, self).__init__(
            epsilon=(range-1.0)/(range+1.0),
            complexity=2.0*complexity/(range+1.0),
            svm_type ='epsilon-SVR',
            **kwargs)


@QNormalParameter("range_", mu=2, sigma=1, q=1)
class RMMClassifierMatlabNode(RegularizedClassifierBase):
    """ Classify with Relative Margin Machine using original matlab code
    
    This node integrates of the "original" Shivaswamy RMM code. This RMM is
    implemented in Matlab and uses the mosek optimization suite.
    
    For this node to work, make sure that
    
    1.  Matlab is installed.
    2.  The pymatlab Python package is installed properly. pymatlab can be
        downloaded from http://pypi.python.org/pypi/pymatlab/0.1.3
        For a MacOS setup guide please see
        https://svn.hb.dfki.de/IMMI-Trac/wiki/pymatlab
    3.  mosek is installed and matlab can access it. See http://mosek.com/
        People with university affiliation can request free academic licenses
        within seconds from http://license.mosek.com/cgi-bin/student.py

    **References**

        ========= ==================================================
        main     
        ========= ==================================================
        author    Shivaswamy, P. K. and Jebara, T.
        title     `Maximum relative margin and data-dependent regularization <http://portal.acm.org/citation.cfm?id=1756031>`_
        journal   Journal of Machine Learning Research
        pages     747-788
        volume    11
        year      2010
        ========= ==================================================

    **Parameters**
        :complexity:
            Complexity sets the weighting of punishment for misclassification
            in comparison to generalizing classification from the data.
            Value in the range form 0 to infinity.
            
            (*optional, default: 1*)

        :range:
            Defines constraint radius for the outer boarder of the classification.
            Going to infinity, this classifier will be identical to the SVM.
            This parameter should be always greater then one.
            Going to one, the classifier will be a variant of the
            Regularized Linear Discriminant analysis.
            
            (*optional, default: 2*)

        .. note:: This classification node doesn't have nice debugging outputs
            as most errors will occur in mosek, i.e., from within the matlab
            session. In the rmm.m matlab code one might want to save the 'res'
            variable as it contains mosek error codes.
        
        .. note:: This implementation doesn't use class weights, i.e.,
            w=[1,1] is fixed.
            
    **Exemplary Call**
    
    .. code-block:: yaml
    
        -
            node :  RMMmatlab
            parameters :
                complexity : 1.0
                range : 2.0
                class_labels : ['Standard', 'Target']

    :input:    FeatureVector
    :output:   PredictionVector
    :Author: David Feess (David.Feess@dfki.de)
    :Revised: 2011/03/10
    
    """
    def __init__(self,range=2, **kwargs):
                
        super(RMMClassifierMatlabNode, self).__init__(**kwargs)
        
        self.set_permanent_attributes(range=range)

    def _stop_training(self, debug = False):
        """ Finish the training, i.e. train the RMM.
        Essentially, the data is passed to matlab, and the classification
        vector w and the offset b are returned."""
        
        self._log("Preprocessing of Matlab RMM")
        self._log("Instances of Class %s: %s, %s: %s" \
                  % (self.classes[0],
                     self.labels.count(self.classes.index(self.classes[0])),
                     self.classes[1],
                     self.labels.count(self.classes.index(self.classes[1]))))
        # Dimension of the data
        self.num_samples = len(self.samples)
        # Mapping from class to value of classifier
        self.bi = []
        for label in self.labels:
            #if self.classes.index(label) == 0:
            if label == 0:
                self.bi.append(-1)
            else:
                self.bi.append(1)
        self._log("optimization preparation")
        
        # Import the necessary matlab interface
        try:
            from pymatlab.matlab import MatlabSession
        except ImportError:
            raise Exception("Using the RMMClassifierMatlabNode requires the pymatlab module.")
        try:
            matlab = MatlabSession('matlab -nojvm -nodisplay')
        except OSError:
            raise Exception("pymatlab couldn't start matlab session. Is pymatlab configured properly?")
        
        matlab.run('clear')
        matlab.run("addpath('pySPACE/missions/nodes/classification/svm_variants/')")
        
        labels = numpy.array(self.bi)[:, numpy.newaxis]*1.0
        if self.kernel_type == "LINEAR":
            train_data = numpy.array(self.samples).T
            # Kzz = numpy.dot(train_data.T, train_data)
            matlab.putvalue('labels', labels)
            matlab.putvalue('train_data', train_data)
            matlab.run("[w_rmm,b_rmm] = compute_linear_RMM(labels, train_data, %f, %f);" % (self.complexity, self.range))
        
            self.w = matlab.getvalue('w_rmm')
            self.b = matlab.getvalue('b_rmm')
        else:
            raise NotImplementedError(
                "Nonlinear kernels have not been wrapped, yet. " +
                "The required matlab implementation is already provided.")
            # KZZ = numpy.zeros((self.num_samples,self.num_samples))
            # for i in range(self.num_samples):
            #     si=self.samples[i]
            #     for j in range(self.num_samples):
            #         if i>j:
            #             KZZ[i][j] = KZZ[j][i]
            #         else:
            #             KZZ[i][j] = self.kernel_func(si,self.samples[j])
        matlab.close()
        
        self._log("Construction complete")
        self.delete_training_data()


# ===========================================================================
@NoOptimizationParameter("version")
@NoOptimizationParameter("kernel_type")
class RmmPerceptronNode(RMM2Node, BaseNode):
    """ Online Learning variants of the 2-norm RMM

    **Parameters**

        .. seealso:: :class:`RMM2Node`

        :co_adaptive:
            Integrate backtransformation into classifier to catch
            changing preprocessing

            (*optional, default: False*)

        :co_adaptive_index:
            Number of nodes to go back

            (*optional, default: numpy.inf*)

        :history_index:
            Number of history index of original data used for co-adaptation

            (*optional, default: 1*)

    **Exemplary Call**

    .. code-block:: yaml

        -   node : RmmPerceptronNode
            parameters :
                range : 3
                complexity : 1.0
                weight : [1,3]
                class_labels : ['Standard', 'Target']

    :Author:    Mario Michael Krell
    :Created:   2014/01/02
    """
    def __init__(self, version="samples", kernel_type='LINEAR',
                 co_adaptive=False, co_adaptive_index=numpy.inf,
                 history_index=1, **kwargs):
        RMM2Node.__init__(self, version=version, kernel_type=kernel_type,
                          **kwargs)
        self.set_permanent_attributes(co_adaptive=co_adaptive,
                                      co_adaptive_index=co_adaptive_index,
                                      history_index=history_index,
                                      hist_w=None,
                                      hist_b=0,
                                      zero_s=None)

    def _train(self, data, class_label):
        """ Main method for incremental and normal training

        Code is a shortened version from the batch algorithm.
        """
        if self.co_adaptive:
            try:
                hist_data = copy.deepcopy(data.history[self.history_index-1])
            except IndexError:
                self._log("No history data available for classifier! " +
                          "Co-adaptivity is turned off.",
                          level=logging.CRITICAL)
                self.co_adaptive = False
        data_array = data.view(numpy.ndarray)
        # shortened initialization part from RegularizedClassifierBase
        if self.feature_names is None:
            try:
                self.feature_names = data.feature_names
            except AttributeError as e:
                warnings.warn("Use a feature generator node before a " +
                              "classification node.")
                raise e
            if self.dim is None:
                self.dim = data.shape[1]
        if class_label not in self.classes and not "REST" in self.classes:
            warnings.warn("Please give the expected classes to the classifier! "
                          + "%s unknown. "%class_label
                          + "Therefore define the variable 'class_labels' in "
                          + "your spec file, where you use your classifier. "
                          + "For further info look at the node documentation.")
            if not(len(self.classes) == 2):
                self.classes.append(class_label)
                self.set_permanent_attributes(classes=self.classes)
        # individual initialization of classification vector
        if self.w is None:
            self.w = numpy.zeros(self.dim, dtype=numpy.float)
        if self.co_adaptive and (self.hist_w is None or self.zero_s is None):
            if type(hist_data) == FeatureVector:
                self.hist_w = FeatureVector.replace_data(
                    hist_data, numpy.zeros(hist_data.shape, dtype=numpy.float))
                self.zero_s = FeatureVector.replace_data(
                    hist_data, numpy.zeros(hist_data.shape, dtype=numpy.float))
            elif type(hist_data) == TimeSeries:
                self.hist_w = TimeSeries.replace_data(
                    hist_data, numpy.zeros(hist_data.shape, dtype=numpy.float))
                self.zero_s = TimeSeries.replace_data(
                    hist_data, numpy.zeros(hist_data.shape, dtype=numpy.float))
        if self.samples is None:
            self.samples = ["empty"] # to suppress logging warning
        # update part init for compatibility with batch mode formulas
        i = 0
        weight = self.weight[self.classes.index(class_label)]
        self.ci = [float(self.complexity * weight)]
        if not self.squ_factor:
            M = [1.0/(numpy.linalg.norm(data_array)**2.0 + self.offset_factor)]
        else:
            M = [(1 / (numpy.linalg.norm(data_array)**2.0
                  + self.offset_factor + 1 / (2 * self.ci[i])),
                  1 / (numpy.linalg.norm(data_array)**2.0
                  + self.offset_factor + 1 /
                  (2 * self.ci[i] * self.complexity_correction)))]
        bi = float(self.classes.index(class_label) * 2 - 1)
        xi = data_array[0, :]
        fi = bi * (dot(xi.T, self.w) + self.b)
        s1 = 0.0
        s2 = 0.0
        beta = False
        old_dual = 0  # just used to have the same formulas as in batch version
        current_dual = numpy.zeros((1, 2))
        # update part
        if self.squ_factor:
            x = old_dual - self.omega * M[i][0] * (fi - 1 + s1)
        elif not self.squ_factor:
            x = old_dual - self.omega * M[i] * (fi - 1)
        # no update of alpha but update of beta (eventually zero update)
        if not x > 0:
            beta = True
            if self.squ_factor:
                x = old_dual - self.omega * M[i][1] * (
                    self.range - fi + s2)
            elif not self.squ_factor:
                x = old_dual - self.omega * M[i] * (
                    self.range - fi + s2)
        # map dual solution to the interval [0,C] in L1 case or
        # just make it positive in the L2 case
        # current_dual[i]=self.project(x,index=i)
        if x <= 0:
            current_dual[i] = [0, 0]
        elif not beta and not self.squ_factor:
            current_dual[i, 0] = min(x, self.ci[i])
        elif beta and not self.squ_factor:
            current_dual[i, 1] = min(x, self.ci[i] *
                                     self.complexity_correction)
        elif not beta and self.squ_factor:
            current_dual[i, 0] = x
        elif beta and self.squ_factor:
            current_dual[i, 1] = x
        # update w and b
        delta = (current_dual[i, 0] + current_dual[i, 1]
                 - old_dual) * bi
        # for beta:  difference needed
        if beta:
            delta = -delta
        # update classification function parameter w and b
        # self.update_classification_function(delta=delta, index=i)
        self.b += self.offset_factor * delta
        if not self.co_adaptive:
            self.w += delta * xi
        else:
            self.hist_b += delta
            self.hist_w += delta * hist_data
            self.update_from_history()

    def update_from_history(self):
        """ Update w from historic data to catch changing preprocessing """
        temp_w = self.get_previous_execute(
            data=self.hist_w, number=self.co_adaptive_index)
        temp_T = self.get_previous_execute(
            data=self.zero_s, number=self.co_adaptive_index)
        temp_w += (self.hist_b-1.0)*temp_T
        self.w = temp_w.view(numpy.ndarray)[0, :]

    def train(self,data,label):
        """ Prevent RegularizedClassifierBase method from being called """
        #one vs. REST case
        if "REST" in self.classes and not label in self.classes:
            label = "REST"
        # one vs. one case
        if not self.multinomial and len(self.classes) == 2 and \
                not label in self.classes:
            return
        start_time_stamp = timeit.default_timer()
        BaseNode.train(self, data, label)
        stop_time_stamp = timeit.default_timer()
        if not self.classifier_information.has_key("Training_time(classifier)"):
            self.classifier_information["Training_time(classifier)"] = \
                stop_time_stamp - start_time_stamp
        else:
            self.classifier_information["Training_time(classifier)"] += \
                stop_time_stamp - start_time_stamp

    def _inc_train(self, data, label):
        """ Incremental training and normal training are the same """
        #one vs. REST case
        if "REST" in self.classes and not label in self.classes:
            label = "REST"
        # one vs. one case
        if not self.multinomial and len(self.classes) == 2 and \
                not label in self.classes:
            return
        if self.co_adaptive == "double" or not self.co_adaptive:
            self._train(data, label)
        if self.co_adaptive:
            self.update_from_history()

    def _stop_training(self, debug=False):
        """ Do nothing than suppressing mother method """
        self.classifier_information["~~Solver_Iterations~~"] = 0
        try:
            self.classifier_information["~~offset~~"] = self.b
            self.classifier_information["~~w0~~"] = self.w[0]
            self.classifier_information["~~w1~~"] = self.w[1]
        except:
            pass

_NODE_MAPPING = {"1RMM": RMM1ClassifierNode,
                "2RMM": RMM2Node,
                "RMMmatlab": RMMClassifierMatlabNode,
                }
