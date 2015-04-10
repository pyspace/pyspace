""" SVM variants using the SOR or dual gradient descent algorithm

All these variants have their offset in the target function.
SOR is used as abbreviation for Successive Overrelaxation.
"""
import numpy
from numpy import dot

import logging
import warnings

# the output is a prediction vector
from pySPACE.resources.data_types.prediction_vector import PredictionVector
from pySPACE.missions.nodes.classification.base import RegularizedClassifierBase

# needed for speed up
# order of examined samples is shuffled
import random
import copy

# needed for loo-metrics
from pySPACE.resources.dataset_defs.metric import BinaryClassificationDataset

class SorSvmNode(RegularizedClassifierBase):
    """ Classify with 2-norm SVM relaxation (b in target function) using SOR algorithm

    This node extends the algorithm with some variants.

    For further details, have a look at the given sources
    and the *reduced_descent* which is an elemental processing step.

    **References**

        ========= ==========================================================================================
        main      source: M&M (matrix version)
        ========= ==========================================================================================
        author    Mangasarian, O. L.  and Musicant, David R.
        title     Successive Overrelaxation for Support Vector Machines
        journal   IEEE Transactions on Neural Networks
        year      1998
        volume    10
        pages     1032--1037
        ========= ==========================================================================================

        ========= ==========================================================================================
        minor     source: Numerical Recipes (randomization)
        ========= ==========================================================================================
        author    Press, William H. and Teukolsky, Saul A. and Vetterling, William T. and Flannery, Brian P.
        title     Numerical Recipes 3rd Edition: The Art of Scientific Computing
        year      2007
        isbn      0521880688, 9780521880688
        edition   3
        publisher Cambridge University Press
        address   New York, NY, USA
        ========= ==========================================================================================

        ========= ==========================================================================================
        minor     source: sample version
        ========= ==========================================================================================
        author    Hsieh, Cho-Jui and Chang, Kai-Wei and Lin, Chih-Jen and Keerthi, S. Sathiya and Sundararajan, S.
        title     A dual coordinate descent method for large-scale linear SVM
        booktitle Proceedings of the 25th international conference on Machine learning
        series    ICML '08
        year      2008
        isbn      978-1-60558-205-4
        location  Helsinki, Finland
        pages     408--415
        numpages  8
        url       http://doi.acm.org/10.1145/1390156.1390208
        doi       10.1145/1390156.1390208
        acmid     1390208
        publisher ACM
        address   New York, NY, USA
        ========= ==========================================================================================

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

        :calc_looCV:
            Calculate the leave-one-out metrics on the training data

            (*optional, default: False*)

        :use_offset:
            If False, the offset b is set to zero, otherwise it is used as normal
            and as it is done in the literature.

            (*optional, default: True*)

        :squared_loss:
            Use L2 loss (optional) instead of L1 loss (default).

            (*optional, default: False*)

    In the implementation we do not use the name alpha but dual_solution for the
    variables of the dual optimization problem,
    which is optimized with this algorithm.

    As a stopping criterion we use the maximum change to be less than some tolerance.

    **Exemplary Call**

    .. code-block:: yaml

        -
            node : SOR
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
                 calc_looCV=False, use_offset=True, squared_loss=False,
                 **kwargs):
        self.old_difference=numpy.inf
        # instead of lists, arrays are concatenated in training
        super(SorSvmNode, self).__init__(use_list=False, **kwargs)

        if not(version in ["samples", "matrix"]):
            self._log("Version %s is not available. Default to 'samples'!"%version, level=logging.WARNING)
            version = "samples"
        if not self.kernel_type == 'LINEAR' and not version == "matrix":
            self._log("Version %s is not available for nonlinear kernel. Default to 'matrix'!"%version, level=logging.WARNING)
            version = "matrix"
        if self.tolerance > 0.1 * self.complexity:
            self.set_permanent_attributes(tolerance=0.1*self.complexity)
            warnings.warn("Using to high tolerance." +
                          " Reduced to 0.1 times complexity (tolerance=%f)."
                          % self.tolerance)
        # mapping of the binary variable to {0,1}
        if not use_offset:
            offset_factor = 0
        else:
            offset_factor = 1

        if not squared_loss:
            squ_factor = 0.0
        else:
            squ_factor = 1.0

        # Weights for soft margin (dependent on class or time)
        ci = []
        # Mapping from class to value of classifier (-1,1)
        bi = []

        self.set_permanent_attributes(random=random,
                                      omega=omega,
                                      max_iterations_factor=max_iterations,
                                      max_sub_iterations=numpy.inf,
                                      iterations=0,
                                      sub_iterations=0,
                                      version=version,
                                      M=None,
                                      reduce_non_zeros=reduce_non_zeros,
                                      calc_looCV=calc_looCV,
                                      offset_factor=offset_factor,
                                      squ_factor=squ_factor,
                                      ci=ci,
                                      bi=bi,
                                      num_samples=0,
                                      dual_solution=None,
                                      max_iterations=42,
                                      b=0
                                      )

    def _execute(self, x):
        """ Executes the classifier on the given data vector in the linear case

        prediction value = <w,data>+b
        """
        if self.kernel_type == 'LINEAR':
            return super(SorSvmNode, self)._execute(x)
            # else:
        data = x.view(numpy.ndarray)
        data = data[0,:]
        prediction = self.b
        for i in range(self.num_samples):
            dual = self.dual_solution[i]
            if not dual == 0:
                prediction += dual * self.bi[i] * \
                    self.kernel_func(data, self.samples[i])
        # Look up class label
        # prediction --> {-1,1} --> {0,1} --> Labels
        if prediction >0:
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
        self.dual_solution = numpy.zeros(self.num_samples)

        if self.version == "matrix" and self.kernel_type == "LINEAR":
            self.A = numpy.array(self.samples)
            self.D = numpy.diag(self.bi)
            self.M = dot(self.D,
                         dot(dot(self.A, self.A.T) + self.offset_factor *
                             numpy.ones((self.num_samples, self.num_samples)),
                             self.D))
        elif self.version == "samples" and self.kernel_type == "LINEAR":
            self.M = [1 / (numpy.linalg.norm(self.samples[i])**2.0
                      + self.offset_factor
                      + self.squ_factor / (2 * self.ci[i]))
                      for i in range(self.num_samples)]
            # changes of w and b are tracked in the samples version
            self.w = numpy.zeros(self.dim, dtype=numpy.float)
            self.b = 0.0
        else: # kernel case
            # iterative calculation of M
            self.M = numpy.zeros((self.num_samples, self.num_samples))
            for i in range(self.num_samples):
                bi = self.bi[i]
                si = self.samples[i]
                for j in range(self.num_samples):
                    if i>j:
                        self.M[i][j] = self.M[j][i]
                    else:
                        self.M[i][j] = bi * self.bi[j] * (
                            self.kernel_func(si, self.samples[j])
                            + self.offset_factor)

        ## SOR Algorithm ##
        self.iteration_loop(self.M)

        self.classifier_information["~~Solver_Iterations~~"] = self.iterations
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
        sort_dual = self.dual_solution
        # sort indices --> zero weights do not need any changing and
        # low weights are less relevant for changes
        sorted_indices = map(list, [numpy.argsort(sort_dual)])[0]
        sorted_indices.reverse()

        prediction_vectors = []
        using_initial_solution = True
        for index in sorted_indices:
            d_i = self.dual_solution[index]
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
            prediction_vectors.append((
                self._execute(numpy.atleast_2d(self.samples[index])),
                                    self.classes[self.labels[index]]))
        self.loo_metrics = BinaryClassificationDataset.calculate_metrics(
            prediction_vectors,
            ir_class=self.classes[1],
            sec_class=self.classes[0])
        #undo changes
        self.b = optimal_b
        self.w = optimal_w
        self.dual_solution = optimal_dual_solution

    def reduce_dual_weight(self, index):
        """ Change weight at index to zero """
        if self.version == "sample":
            old_weight = self.dual_solution[index]
            self.update_classification_function(delta=-old_weight, index=index)
        else:
            # the matrix algorithm doesn't care for the old weights
            pass
        self.dual_solution[index] = 0

    def calculate_weigts_and_class_factors(self):
        """ Calculate weights in the loss term and map label to -1 and 1 """
        self.num_samples=0
        for label in self.labels:
            self.num_samples += 1
            self.append_weights_and_class_factors(label)
            #care for zero sum

    def append_weights_and_class_factors(self, label):
        """ Mapping between labels and weights/class factors

        The values are added to the corresponding list.
        """
        if label == 0:
            self.bi.append(-1)
            self.ci.append(self.complexity*self.weight[0])
        else:
            self.bi.append(1)
            self.ci.append(self.complexity*self.weight[1])

    def iteration_loop(self, M, reduced_indices=[]):
        """ The algorithm is calling the :func:`reduced_descent<pySPACE.missions.nodes.classifiers.ada_SVM.SORSVMNode.reduced_descent>` method in loops over alpha

        In the first step it uses a complete loop over all components of alpha
        and in the second inner loop only the non zero alpha are observed till
        come convergence criterion is reached.

        *reduced_indices* will be skipped in observation.
        """
        ## Definition of tracking variables ##
        self.iterations = 0
        self.difference = numpy.inf
        ## outer iteration loop ##
        while (self.difference > self.tolerance and
               self.iterations <= self.max_iterations):
            # inner iteration loop only on active vectors/alpha (non zero) ##
            self.sub_iterations = 0
            # sorting or randomizing non zero indices
            # arrays are mapped to lists for later iteration
            sort_dual = self.dual_solution

            num_non_zeros = len(map(list,sort_dual.nonzero())[0])
            max_values = len(map(list,
                                 numpy.where(sort_dual == sort_dual.max()))[0])
            # sort the entries of the current dual
            # and get the corresponding indices
            sorted_indices = map(list,[numpy.argsort(sort_dual)])[0]
            if num_non_zeros == 0 or num_non_zeros==max_values:
                # skip sub iteration if everything is zero or maximal
                non_zero_indices = []
            else:
                non_zero_indices = sorted_indices[-num_non_zeros:-max_values]
            for index in reduced_indices:
                try:
                    non_zero_indices.remove(index)
                except ValueError:
                    pass
            if self.random:
                random.shuffle(non_zero_indices)
            self.max_sub_iterations = self.max_iterations_factor * \
                len(non_zero_indices) * 0.5
            while (self.difference > self.tolerance and
                   self.sub_iterations < self.max_sub_iterations
                   and self.iterations < self.max_iterations):
                ## iteration step ##
                self.reduced_descent(self.dual_solution, M, non_zero_indices)
                ## outer loop ##
            if not (self.iterations < self.max_iterations):
                break
            # For the first run, the previous reduced descent is skipped
            # but for retraining it is important
            # to have first the small loop, since normally, this is sufficient.
            # Furthermore having it at the end simplifies the stop criterion
            self.max_sub_iterations = numpy.inf
            self.total_descent(self.dual_solution, M, reduced_indices)
            ## Final solution ##
        # in the case without kernels, we have to calculate the result
        # by hand new for each incoming sample
        if self.version == "matrix":
            self.b = self.offset_factor * dot(self.dual_solution, self.bi)
            # self.w = self.samples[0]*self.dual_solution[0]*self.bi[0]
            # for i in range(self.num_samples-1):
            #     self.w = self.w + self.bi[i+1] * self.samples[i+1] *
            #         self.dual_solution[i+1]
            if self.kernel_type == "LINEAR":
                self.w = numpy.array([dot(dot(self.A.T, self.D),
                                          self.dual_solution)]).T
        elif self.version == "samples" and self.kernel_type == "LINEAR":
            # w and b are pre-computed in the loop
            # transferring of 1-d array to 2d array
            # self.w = numpy.array([self.w]).T
            pass

    def reduced_descent(self,current_dual,M,relevant_indices):
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
        for i in relevant_indices:
            old_dual = current_dual[i]
            ### Main Function ###
            ### elemental update step of SOR algorithm ###

            if self.version == "matrix":
                # this step is kernel independent
                x = old_dual - self.omega / (
                    M[i][i] + self.squ_factor/(2 * self.ci[i])) * \
                    (dot(M[i], current_dual) - 1)
            elif self.version == "samples":
                xi = self.samples[i]
                bi = self.bi[i]
                x = old_dual - self.omega * (M[i]) * \
                    (bi * (dot(xi.T, self.w) + self.b) - 1 +
                     self.squ_factor * old_dual / (2 * self.ci[i]))
            # map dual solution to the interval [0,C]
            if x <= 0:
                self.irrelevant_indices.append(i)
                current_dual[i] = 0
            elif not self.squ_factor:
                current_dual[i] = min(x, self.ci[i])
            else:
                current_dual[i] = x
            if self.version == "matrix":
                delta = (current_dual[i] - old_dual)
                # update w and b in samples case
            if self.version == "samples":
                delta = (current_dual[i] - old_dual) * bi
                # update classification function parameter w and b
                # self.update_classification_function(delta=delta, index=i)
                self.b = self.b + self.offset_factor * delta
                self.w = self.w + delta * xi
            current_difference = numpy.abs(delta)
            if current_difference > self.difference:
                self.difference = current_difference
            self.sub_iterations += 1
            self.iterations += 1
            if not (self.sub_iterations < self.max_sub_iterations
                    and self.iterations < self.max_iterations):
                break
        if self.reduce_non_zeros:
            for index in self.irrelevant_indices:
                try:
                    relevant_indices.remove(index)
                except:
                    # special mapping for RMM case
                    if index < self.num_samples:
                        relevant_indices.remove(index+self.num_samples)
                    else:
                        relevant_indices.remove(index-self.num_samples)
        if self.random:
            random.shuffle(relevant_indices)

    def update_classification_function(self,delta, index):
        """ update classification function parameter w and b """
        bi = self.bi[index]
        self.b = self.b + self.offset_factor * delta * bi
        self.w = self.w + delta * bi * self.samples[index]

    def project(self,value,index):
        """ Projection method of *soft_relax* """
        if value <= 0:
            self.irrelevant_indices.append(index)
            return 0
        else:
            return min(value, self.ci[index])

    def total_descent(self,current_dual,M,reduced_indices=[]):
        """ Different sorting of indices and iteration over all indices

        .. todo:: check, which parameters are necessary
        """
        if not self.random:
            sort_dual = current_dual
            # sort the entries of the current dual
            # and get the corresponding indices
            sorted_indices = map(list, [numpy.argsort(sort_dual)])[0]
            # highest first
            sorted_indices.reverse()
        else:
            sorted_indices = range(self.num_samples)
            random.shuffle(sorted_indices)
        for index in reduced_indices:
            sorted_indices.remove(index)
        self.reduced_descent(current_dual, M,sorted_indices)

_NODE_MAPPING = {"SOR": SorSvmNode}
