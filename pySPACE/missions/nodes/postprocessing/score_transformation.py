# -*- coding: UTF-8 -*
""" Transform the classification score (especially the one of the SVM) """
import numpy

from pySPACE.missions.nodes.base_node import BaseNode
from pySPACE.resources.data_types.prediction_vector import PredictionVector

class EmptyBinException(Exception): pass

class PlattsSigmoidFitNode(BaseNode):
    """ Map prediction scores to probability estimates with a sigmoid fit
    
    This node uses a sigmoid fit to map a prediction score to a class 
    probability estimate, i.e. a value between 0 and 1, where e.g. 0.5 means
    50% probability of the positive class which must not necessarily correspond
    to a SVM score of 0.
    For more information see 'Probabilistic Outputs for Support Vector
    Machines and Comparisons to Regularized Likelihood Methods' (Platt) 1999.
    The parametric form of the sigmoid function is:
    
    .. math::
        P(c|x) =\\text{appr } P_{A,B}(s(x)) = \\frac{1}{1+e^{As(x)+B}}
    
    where c is the actual class, x the data, s(x) the prediction score and
    A and B are calculated through the training examples.
    
    .. note:: Learning this transformation on the same training data
        than the classifier was trained is not recommended
        for non-linear kernels (due to over-fitting).
    
    The best parameter setting z*=(A*,B*) is determined by solving the 
    following regularized maximum likelihood problem:
    
    .. math::
        \\min F(z)= - \\sum_{i=1}^l{(t_i \\log(p_i) + (1-t_i) \\log(1-p_i))}, 

    for :math:`p_i=P_{A,B}(s(x_i))` and :math:`t_i` are target probabilities defined according
    to *priors* and :math:`c_i`. 
    
    The implementation is improved to ensure convergence and to avoid numerical
    difficulties (see 'A Note on Platt's Probabilistic Outputs for Support
    Vector Machines' (HT Lin, RC Weng) 2007).

    **Parameters**
    
        :priors:
            A tuple that consists the number of examples expected for
            each class (first element negative class, second element
            positive class). If the parameter is not specified, the numbers in
            the training set are used.
            
            (*optional, default: None*)
        
        :class_labels:
            Determines the order of classes, i.e. the mapping of class labels
            onto integers. The first element of the list should be the negative
            class, the second should be the positive class.
            If this parameter is not specified, the order is determined based on
            the order of occurrence in the training data (which is more or less
            arbitrary). 
    
            (*optional, default: []*)
            
        :oversampling:
            If True different class distributions are balanced by oversampling
            and random drawing where appropriate (if the overrepresented class
            is not divisible by the underrepresented class).
            
            (*optional, default: False*)
        
        :store_plots:
            If True 'reliable diagrams' of the training and test data are stored.
            A discretization of the scores is made to calculate empirical 
            probabilities. The number of scores per bin is displayed on every
            data point in the figure and shows how accurate the estimate
            is (the higher the number the better). If the fit is reliable the
            empirical probabilities should scatter around the diagonal in the
            right plots. Although the store variable is set to True if this
            variable is set.
            
        :store_probabilities:        
            If True the calculated probability and the corresponding label for 
            each prediction is pickeled and saved in the results directory. 
            Although the store variable is set to True if this variable is set. 
            
            (*optional, default: False*)
            
        :store:
            If True store_plots and store_probabilities are set to True. 
            This is the "simple" way to store both the plots and the 
            probabilities. 
            
    **Exemplary Call**
    
    .. code-block:: yaml
    
        -
            node : PSF
            parameters :
                class_labels : ['Target','Standard']
                
            
    """
    def __init__(self, priors = None, class_labels = [], oversampling = False,
                 store_plots = False, store_probabilities = False, **kwargs):
        super(PlattsSigmoidFitNode, self).__init__(**kwargs)
        if ( store_plots or store_probabilities ):
            self.store = True
        elif(self.store):
            store_plots = True
            store_probabilities = True

        
        self.set_permanent_attributes(priors = priors,
                                      class_labels = class_labels,
                                      oversampling = oversampling,
                                      scores = [],
                                      labels = [],
                                      probabilities = [],
                                      store_plots = store_plots,
                                      store_probabilities = store_probabilities)
                    
    def is_trainable(self):
        return True
    
    def is_supervised(self):
        return True

    def _train(self, data, class_label):
        """ Collect SVM output and true labels. """
        self._train_phase_started = True
        self.scores.append(data.prediction)
        
        if class_label not in self.class_labels:
            self.class_labels.append(class_label)
        self.labels.append(self.class_labels.index(class_label))
            
    def _stop_training(self):
        """ Compute parameter A and B for sigmoid fit."""
        
        def func_value(s,t,a,b):
            """ Compute the function value avoiding 'catastrophic cancellation'.
                   -(t_i log(p_i) + (1-t_i)log(1- p_i))
                =  t_i log(1+exp(as_i+b)) + (t_i-1)((as_i+b)-log(1+exp(as_i+b)))
                =  t_i log(1+exp(as_i+b)) + (t_i-1)(as_i+b) - 
                                     t_i log(1+exp(as_i+b)) + log(1+exp(as_i+b))
                =  (t_i-1)(as_i+b) + log(1+exp(as_i+b)) *
                =  t_i(as_i+b) + log(exp(-as_i-b)) + log(1+exp(as_i+b))
                =  t_i(as_i+b) + log((1+exp(as_i+b))/exp(as_i+b))
                =  t_i(as_i+b) + log(1+exp(-as_i-b))   **
            """
            fapb = s*a+b
            # * is used if fapb[i]<0, ** otherwise
            f = sum([(t[i]-1)*fapb[i] + numpy.log1p(numpy.exp(fapb[i])) if fapb[i]<0 \
                      else t[i]*fapb[i] + numpy.log1p(numpy.exp(-fapb[i])) \
                      for i in range(len(fapb))])
            return f
            
        self._log("Performing training of sigmoid mapping.")
        if self.oversampling:
            # first assume that the negative class is the overrepresented class
            overrepresented_inst = [score for score,label in \
                                       zip(self.scores,self.labels) if label==0]
            underrepresented_inst = [score for score,label in \
                                       zip(self.scores,self.labels) if label==1]
            if len(overrepresented_inst) != len(underrepresented_inst):
                # check if assumption was correct
                if len(overrepresented_inst) < len(underrepresented_inst):
                    tmp = overrepresented_inst
                    overrepresented_inst = underrepresented_inst
                    underrepresented_inst = tmp
                oversampling_factor = len(overrepresented_inst) / \
                                                      len(underrepresented_inst)
                self.scores.extend((oversampling_factor-1)*underrepresented_inst)
                self.labels.extend((oversampling_factor-1) * \
                                        len(underrepresented_inst) * range(1,2))
                if len(overrepresented_inst) % len(underrepresented_inst) != 0: 
                    num_draw_random = len(overrepresented_inst) - \
                                oversampling_factor * len(underrepresented_inst)
                    # Randomizer has to be fixed for reproducibility
                    import random
                    randomizer = random.Random(self.run_number)
                    for i in range(num_draw_random):
                        selected_score=randomizer.choice(underrepresented_inst)
                        self.scores.append(selected_score)
                        underrepresented_inst.remove(selected_score)
                    self.labels.extend(num_draw_random * range(1,2))
        
        if self.priors == None:
            self.priors = (self.labels.count(0),self.labels.count(1))
        self.scores = numpy.array(self.scores)
        
        # Parameter settings
        maxiter = 100 # Maximum number of iterations
        minstep = 1.0e-10 # Minimum step taken in line search
        sigma = 1.0e-12 # Set to any value > 0
        
        # Construct initial value: target support in array targets, 
        hiTarget = (self.priors[1]+1.0)/(self.priors[1]+2.0)
        loTarget = 1/(self.priors[0]+2.0)
        h = (loTarget,hiTarget)
        targets = numpy.array([h[index] for index in self.labels])
        # initial function value in fval
        self.A = 0
        self.B = numpy.log((self.priors[0]+1.0)/(self.priors[1]+1.0))
        fval = func_value(self.scores,targets,self.A,self.B)
        
        for it in range(maxiter):
            # update gradient and Hessian (use H' = H + sigma I)
            h11 = h22 = sigma
            h21 = g1 = g2 = 0.0
            
            fApB = self.scores*self.A+self.B
            
            pq= numpy.array([[1/(1.0+numpy.exp(fApB[i])),
                numpy.exp(fApB[i])/(1.0+numpy.exp(fApB[i]))] if fApB[i]<0 \
                else [numpy.exp(-fApB[i])/(1.0+numpy.exp(-fApB[i])),
                      1/(1.0+numpy.exp(-fApB[i]))] \
                for i in range(len(fApB))])
            
            d1 = targets - pq[:,0]
            d2 = pq[:,0] * pq[:,1]
            h11 = sum(self.scores**2*d2)
            h21 = sum(self.scores*d2)
            h22 = sum(d2)
            g1 = sum(self.scores*d1)
            g2 = sum(d1)
            
            # stopping criteria: if gradient is really tiny, stop
            if (abs(g1) < 1.0e-5) and (abs(g2) < 1.0e-5):
                break
            
            # finding Newton direction: -inv(H')*g
            det = h11*h22-h21**2
            dA = -(h22*g1-h21*g2)/det
            dB = -(-h21*g1+h11*g2)/det
            gd = g1*dA+g2*dB
            
            # line search
            stepsize = 1
            while stepsize >= minstep:
                newA = self.A+stepsize*dA
                newB = self.B+stepsize*dB
                newf = func_value(self.scores,targets,newA,newB)
                # check sufficient decrease
                if (newf < fval+0.0001*stepsize*gd):
                    self.A = newA
                    self.B = newB
                    fval = newf
                    break
                else:
                    stepsize /= 2.0
            
            if stepsize < minstep:
                import logging
                self._log("Line search fails. A= "+str(self.A)+" B= " \
                             +str(self.B)+" g1= "+str(g1)+" g2= "+str(g2) \
                             +" dA= "+str(dA)+" dB= "+str(dB)+" gd= "+str(gd),
                             level = logging.WARNING)
                break
            
        if it>=maxiter-1:
            import logging
            self._log("Reaching maximal iterations. g1= "+str(g1)+" g2= " \
                      +str(g2), level=logging.WARNING)
        
        self._log("Finished training of sigmoid mapping in %d iterations." % it)
        
        # Clean up of not needed variables
        self.scores = []
        self.labels = []

    def _execute(self, x):
        """ Evaluate each prediction with the sigmoid mapping learned."""
        
        fApB = x.prediction * self.A + self.B
        if fApB<0:
            new_prediction=1/(1.0+numpy.exp(fApB))
        else:
            new_prediction=numpy.exp(-fApB)/(numpy.exp(-fApB)+1.0)
        # enforce mapping to interval [0,1]
        new_prediction = max(0,min(1,new_prediction))
        new_label = self.class_labels[0] if new_prediction <= 0.5 \
                                                       else self.class_labels[1]
        # Safe the new calculated probabilities
        if self.store_probabilities:
            self.probabilities.append( [new_prediction , new_label] )
        return PredictionVector(label=new_label,
                                prediction=new_prediction,
                                predictor=x.predictor)
 
    def _discretize(self, predictions, labels, bins=12):
        """ Discretize predictions into bins. 
        
        Return bin scores and 2d list of discretized labels. """
        while(True):
            try:
                cut = (abs(predictions[0])+ abs(predictions[-1]))/bins
                current_bin=0
                l_discrete={0:[]}
                bin_scores = [predictions[0]+cut/2.0]
                for p,l in zip(predictions,labels):
                    if p > predictions[0]+cut*(current_bin+2):
                        raise EmptyBinException("One bin without any examples!")
                    if p > predictions[0]+cut*(current_bin+1):
                        current_bin += 1
                        bin_scores.append(bin_scores[-1]+cut)
                        l_discrete[current_bin]=[l]
                    else:
                        l_discrete[current_bin].append(l)
                if len(l_discrete)==bins+1:
                    l_discrete[bins-1].extend(l_discrete[bins])
                    del l_discrete[bins]
                    del bin_scores[-1]
                    bin_scores[-1]= bin_scores[-1]-cut/2.0 + \
                                  (predictions[-1]-(bin_scores[-1]-cut/2.0))/2.0
            except EmptyBinException:
                if bins>1:
                    bins-=1
                else:
                    raise Exception("Could not discretize data!")
            else:
                return bin_scores, l_discrete
    
    def _empirical_probability(self, l_discrete):
        """ Return dictionary of empirical class probabilities for discretized label list."""
        plot_emp_prob = {}
        len_list = {}
        for label in range(len(self.class_labels)):
            plot_emp_prob[label]=[]
            len_list[label]=[]
            for score_list in l_discrete.values():
                len_list[label].append(len(score_list))
                plot_emp_prob[label].append(score_list.count(label)/ \
                                                         float(len(score_list)))
        return len_list, plot_emp_prob
    
    def store_state(self, result_dir, index=None):
        """ Stores plots of score distribution and sigmoid fit or/and 
        the calculated probabilities with the corresponding label.

        .. todo:: change plot calculations to upper if else syntax
        .. todo:: add the corresponding data point to the saved probabilities
        """
        if self.store :
            # Create the directory for the stored results
            from pySPACE.tools.filesystem import  create_directory
            import os
            node_dir = os.path.join(result_dir, self.__class__.__name__)
            create_directory(node_dir)
            # Safe the probabilities in a pickle file 
            if( self.store_probabilities ):
                import pickle
                f_name=node_dir + "/probabilities_%d.pickle" % self.current_split
                pickle.dump(self.probabilities, open(f_name,'w'))
            if self.store_plots:
                # reliable plot of training (before sigmoid fit)
                sort_index = numpy.argsort(self.scores)
                labels = numpy.array(self.labels)[sort_index]
                predictions = numpy.array(self.scores)[sort_index]
                
                plot_scores_train,l_discrete_train=self._discretize(predictions, labels)
                len_list_train, plot_emp_prob_train = self._empirical_probability(l_discrete_train)
                
                # training data after sigmoid fit
                fApB = predictions * self.A + self.B
                new_predictions = [(int(fApB[i]<0)+int(fApB[i]>=0)*numpy.exp(-fApB[i]))/ \
                                 (1.0+numpy.exp((-1)**int(fApB[i]>=0)*fApB[i])) \
                                 for i in range(len(fApB))]
                
                plot_scores_train_fit, l_discrete_train_fit = \
                                                self._discretize(new_predictions,labels)
                len_list_train_fit, plot_emp_prob_train_fit = \
                                       self._empirical_probability(l_discrete_train_fit)
                
                # test data before sigmoid fit
                test_scores = []
                test_labels = []
                for data, label in self.input_node.request_data_for_testing():
                    test_scores.append(data.prediction)
                    test_labels.append(self.class_labels.index(label))
                
                sort_index = numpy.argsort(test_scores)
                labels = numpy.array(test_labels)[sort_index]
                predictions = numpy.array(test_scores)[sort_index]
                
                plot_scores_test,l_discrete_test = self._discretize(predictions, labels)
                len_list_test, plot_emp_prob_test = self._empirical_probability(l_discrete_test)
                
                # test data after sigmoid fit
                fApB = predictions * self.A + self.B
                new_predictions = [(int(fApB[i]<0)+int(fApB[i]>=0)*numpy.exp(-fApB[i]))/ \
                                 (1.0+numpy.exp((-1)**int(fApB[i]>=0)*fApB[i])) \
                                 for i in range(len(fApB))]
                
                plot_scores_test_fit, l_discrete_test_fit = \
                                                self._discretize(new_predictions,labels)
                len_list_test_fit, plot_emp_prob_test_fit = \
                                       self._empirical_probability(l_discrete_test_fit)
                
                
                
                import pylab
                from matplotlib.transforms import offset_copy
                pylab.close()
                fig = pylab.figure(figsize=(10,10))
                ax = pylab.subplot(2,2,1)
                transOffset=offset_copy(ax.transData,fig=fig,x=0.05,y=0.1,units='inches')
                for x,y,s in zip(plot_scores_train,plot_emp_prob_train[1],len_list_train[1]):
                    pylab.plot((x,),(y,),'ro')
                    pylab.text(x,y,'%d' % s, transform=transOffset)
                
                pylab.plot((plot_scores_train[0],plot_scores_train[-1]),(0,1),'-')
                x = numpy.arange(plot_scores_train[0],plot_scores_train[-1],.02)
                y = 1/(1+numpy.exp(self.A*x+self.B))
                pylab.plot(x,y,'-')
                pylab.xlim(plot_scores_train[0],plot_scores_train[-1])
                pylab.ylim(0,1)
                pylab.xlabel("SVM prediction Score (training data)")
                pylab.ylabel("Empirical Probability")
                
                ax = pylab.subplot(2,2,2)
                transOffset=offset_copy(ax.transData,fig=fig,x=0.05,y=0.1,units='inches')
                for x, y, s in zip(plot_scores_train_fit, plot_emp_prob_train_fit[1], 
                                                                 len_list_train_fit[1]):
                    pylab.plot((x,),(y,),'ro')
                    pylab.text(x,y,'%d' % s, transform=transOffset)
                
                pylab.plot((plot_scores_train_fit[0],plot_scores_train_fit[-1]),(0,1),'-')
                pylab.xlim(plot_scores_train_fit[0],plot_scores_train_fit[-1])
                pylab.ylim(0,1)
                pylab.xlabel("SVM Probability (training data)")
                pylab.ylabel("Empirical Probability")
                
                ax = pylab.subplot(2,2,3)
                transOffset=offset_copy(ax.transData,fig=fig,x=0.05,y=0.1,units='inches')
                for x,y,s in zip(plot_scores_test,plot_emp_prob_test[1],len_list_test[1]):
                    pylab.plot((x,),(y,),'ro')
                    pylab.text(x,y,'%d' % s, transform=transOffset)
                
                pylab.plot((plot_scores_test[0],plot_scores_test[-1]),(0,1),'-')
                x = numpy.arange(plot_scores_test[0],plot_scores_test[-1],.02)
                y = 1/(1+numpy.exp(self.A*x+self.B))
                pylab.plot(x,y,'-')
                pylab.xlim(plot_scores_test[0],plot_scores_test[-1])
                pylab.ylim(0,1)
                pylab.xlabel("SVM prediction Scores (test data)")
                pylab.ylabel("Empirical Probability")
                
                ax = pylab.subplot(2,2,4)
                transOffset=offset_copy(ax.transData,fig=fig,x=0.05,y=0.1,units='inches')
                for x, y, s in zip(plot_scores_test_fit, plot_emp_prob_test_fit[1], 
                                                                  len_list_test_fit[1]):
                    pylab.plot((x,),(y,),'ro')
                    pylab.text(x,y,'%d' % s, transform=transOffset)
                
                pylab.plot((plot_scores_test_fit[0],plot_scores_test_fit[-1]),(0,1),'-')
                pylab.xlim(plot_scores_test_fit[0],plot_scores_test_fit[-1])
                pylab.ylim(0,1)
                pylab.xlabel("SVM Probability (test data)")
                pylab.ylabel("Empirical Probability")
                
                pylab.savefig(node_dir + "/reliable_diagrams_%d.png" % self.current_split)


class SigmoidTransformationNode(BaseNode):
    """ Transform score to interval [0,1] with a sigmoid function
    
    The new decision border will be at 0.5.
    
    .. warning::
        This is NOT a probability mapping and parameters should be set for
        the function.
    
    This node is intended to be externally optimized, such that it
    generalizes the threshold optimization for soft metrics.
    
    The used sigmoid fit function is :math:`\\frac{1}{1+e^{Ax+B}}`. 
    It is 0.5 at :math:`x = -\\frac{B}{A}`.
    
    **Parameters**
    
        :A:
            Scaling of prediction value. See above.
            
            (*optional, default: -1*)
        
        :B:
            Shifting of scaled prediction. See above.
            
            (*optional, default: 0*)
            
        :offset:
            Has the meaning of :math:`-\\frac{B}{A}` and replaces the parameter B if used.
            
            (*optional, default: None*) 
        
        :class_labels:
            Determines the order of classes, i.e. the mapping of class labels
            onto integers. The first element of the list should be the negative
            class, the second should be the positive class.
            In the context positive should be the class mapped greater than 0.5
            and the other class should be the negative one.
            If the original prediction value had the same orientation,
            *A* should be chosen negative.
        
            (*optional, default: ['Standard','Target']*)
    
    **Exemplary Call**
    
    .. code-block:: yaml
    
        -
            node : SigTrans
            parameters :
                class_labels : ['Standard','Target']
    """
    input_types=["PredictionVector"]
    def __init__(self, class_labels = ['Standard','Target'], 
                A = -1, B = 0, offset = None,
                **kwargs):
        super(SigmoidTransformationNode, self).__init__(**kwargs)
        
        if not(offset is None):
            B = -A *offset
        
        self.set_permanent_attributes(class_labels = class_labels,
                                      A = A,
                                      B = B)
    
    def is_trainable(self):
        return False
    
    def is_supervised(self):
        return False
    
    def _execute(self, data):
        """ Evaluate each prediction with the sigmoid mapping learned. """
        
        # code simply copied from PlattsSigmoidFitNode fur eventual future changes
        fApB = data.prediction * self.A + self.B
        if fApB<0:
            new_prediction=1/(1.0+numpy.exp(fApB))
        else:
            new_prediction=numpy.exp(-fApB)/(numpy.exp(-fApB)+1.0)
        # enforce mapping to interval [0,1]
        new_prediction = max(0,min(1,new_prediction))
        new_label = self.class_labels[0] if new_prediction <= 0.5 \
                                                       else self.class_labels[1]

        return PredictionVector(label=new_label,
                                prediction=new_prediction,
                                predictor=data.predictor)


class LinearTransformationNode(BaseNode):
    """ Scaling and offset shift, and relabeling due to new decision boundary

    Having a prediction value x it is mapped to (x+*offset*)*scaling*.
    If the result is lower than the *decision boundary* it is mapped to the
    first class label for the negative class and otherwise to the second
    positive class.

    **Parameters**

        :class labels:  This mandatory parameter defines the ordering of class
                        labels for the mapping after the transformation.
                        If this parameter is not specified, the label remains
                        unchanged. This is for example feasible for regression
                        mappings.

                        .. note:: This parameter could be also used to change
                                  class label strings, but this would probably
                                  cause problems in the evaluation step.

                        (*recommended, default: None*)

        :offset: Shift of the prediction value.

                 (*optional, default: 0*)

        :scaling: Scaling factor applied after offset shift.

                  (*optional, default: 1*)

        :decision_boundary: Everything lower this value is classified as
            class one and everything else as class two. By default
            no labels are changed.

    **Exemplary Call**

    .. code-block:: yaml

        -   node : LinearTransformation
            parameters :
                class_labels : ['Standard', 'Target']
                offset : 1
                scaling : 42
                decision_boundary : 3
    """
    def __init__(self, class_labels=None, offset=0, scaling=1,
                 decision_boundary=None, **kwargs):
        super(LinearTransformationNode, self).__init__(**kwargs)
        if class_labels is None or decision_boundary is None:
            decision_boundary = None
            class_labels = None

        self.set_permanent_attributes(class_labels=class_labels,
                                      scaling=scaling,
                                      offset=offset,
                                      decision_boundary=decision_boundary,
                                      )

    def _execute(self, x):
        """ (x+o)*s < d """
        p = x.prediction
        prediction = (p+self.offset)*self.scaling
        if self.decision_boundary is None:
            label = x.label
        elif self.decision_boundary < prediction:
            label = self.class_labels[0]
        else:
            label = self.class_labels[1]
        return PredictionVector(prediction=prediction, label=label,
                                predictor=x.predictor)

class LinearFitNode(BaseNode):
    """ Linear mapping between score and [0,1]
    
    This node maps the unbounded SVM score linear to bound it between [0,1].
    If the result can be interpreted as probability can be seen in the
    reliable diagrams.

    **Parameters**
        
        :class_labels:
            Determines the order of classes, i.e. the mapping of class labels
            onto integers. The first element of the list should be the negative
            class, the second should be the positive class.
            If this parameter is not specified, the order is determined based on
            the order of occurrence in the training data (which is more or less
            arbitrary). 
        
            (*optional, default: []*)
        
        :store:
            If True 'reliable diagrams' of the training and test data are stored.
            A discretization of the scores is made to calculate empirical 
            probabilities. The number of scores per bin is displayed on every
            data point in the figure and shows how accurate the estimate
            is (the higher the number the better). If the fit is reliable the
            empirical probabilities should scatter around the diagonal in the
            right plots.
            
    **Exemplary Call**
    
    .. code-block:: yaml
    
        -
            node : LinearFit
            parameters :
                class_labels : ['Standard','Target']
    """
    def __init__(self, class_labels = [], **kwargs):
        super(LinearFitNode, self).__init__(**kwargs)
        
        self.set_permanent_attributes(class_labels = class_labels,
                                      scores = [],
                                      labels = [])
                    
    def is_trainable(self):
        return True
    
    def is_supervised(self):
        return True

    def _train(self, data, class_label):
        """ Collect SVM output and true labels. """
        self._train_phase_started = True
        self.scores.append(data.prediction)
        
        if class_label not in self.class_labels:
            self.class_labels.append(class_label)
        self.labels.append(self.class_labels.index(class_label))       
            
    def _stop_training(self):
        """ Compute max range of the score according to the class."""
        positive_inst = [score for score,label in \
                                       zip(self.scores,self.labels) if label==1]
        negative_inst = [score for score,label in \
                                       zip(self.scores,self.labels) if label==0]
        self.max_range = (abs(min(negative_inst)),max(positive_inst))
        
    def _execute(self, x):
        """ Evaluate each prediction with the linear mapping learned."""
        
        if x.prediction < -1.0*self.max_range[0]:
            new_prediction = 0.0
        elif x.prediction < self.max_range[1]:
            new_prediction = (x.prediction + \
                        self.max_range[self.class_labels.index(x.label)]) / \
                        (2.0 * self.max_range[self.class_labels.index(x.label)])
        else:
            new_prediction = 1.0

        return PredictionVector(label=x.label,
                                prediction=new_prediction,
                                predictor=x.predictor)
 
    def _discretize(self, predictions, labels, bins=12):
        """ Discretize predictions into bins. Return bin scores and 2d list of discretized labels. """
        while(True):
            try:
                cut = (abs(predictions[0])+ abs(predictions[-1]))/bins
                current_bin=0
                l_discrete={0:[]}
                bin_scores = [predictions[0]+cut/2.0]
                for p,l in zip(predictions,labels):
                    if p > predictions[0]+cut*(current_bin+2):
                        raise EmptyBinException("One bin without any examples!")
                    if p > predictions[0]+cut*(current_bin+1):
                        current_bin += 1
                        bin_scores.append(bin_scores[-1]+cut)
                        l_discrete[current_bin]=[l]
                    else:
                        l_discrete[current_bin].append(l)
                if len(l_discrete)==bins+1:
                    l_discrete[bins-1].extend(l_discrete[bins])
                    del l_discrete[bins]
                    del bin_scores[-1]
                    bin_scores[-1]= bin_scores[-1]-cut/2.0 + \
                                  (predictions[-1]-(bin_scores[-1]-cut/2.0))/2.0
            except EmptyBinException:
                if bins>1:
                    bins-=1
                else:
                    raise Exception("Could not discretize data!")
            else:
                return bin_scores, l_discrete
    
    def _empirical_probability(self, l_discrete):
        """ Return dictionary of empirical class probabilities for discretized label list."""
        plot_emp_prob = {}
        len_list = {}
        for label in range(len(self.class_labels)):
            plot_emp_prob[label]=[]
            len_list[label]=[]
            for score_list in l_discrete.values():
                len_list[label].append(len(score_list))
                plot_emp_prob[label].append(score_list.count(label)/ \
                                                         float(len(score_list)))
        return len_list, plot_emp_prob
        
        
    def store_state(self, result_dir, index=None):
      """ Stores plots of score distribution and sigmoid fit. """
      if self.store :
        # reliable plot of training (before linear fit)
        sort_index = numpy.argsort(self.scores)
        labels = numpy.array(self.labels)[sort_index]
        predictions = numpy.array(self.scores)[sort_index]

        plot_scores_train,l_discrete_train=self._discretize(predictions, labels)
        len_list_train, plot_emp_prob_train = self._empirical_probability(l_discrete_train)
        
        # training data after linear fit
        new_predictions = []
        for score in predictions:
            if score < 0.0:
                new_predictions.append((score + self.max_range[0]) / \
                                                      (2.0 * self.max_range[0]))
            else:
                new_predictions.append((score + self.max_range[1]) / \
                                                      (2.0 * self.max_range[1]))
        
        plot_scores_train_fit, l_discrete_train_fit = \
                                        self._discretize(new_predictions,labels)
        len_list_train_fit, plot_emp_prob_train_fit = \
                               self._empirical_probability(l_discrete_train_fit)

        # test data before sigmoid fit
        test_scores = []
        test_labels = []
        for data, label in self.input_node.request_data_for_testing():
            test_scores.append(data.prediction)
            test_labels.append(self.class_labels.index(label))
        
        sort_index = numpy.argsort(test_scores)
        labels = numpy.array(test_labels)[sort_index]
        predictions = numpy.array(test_scores)[sort_index]
        
        plot_scores_test,l_discrete_test = self._discretize(predictions, labels)
        len_list_test, plot_emp_prob_test = self._empirical_probability(l_discrete_test)

        # test data after sigmoid fit
        new_predictions = []
        for score in predictions:
            if score < -1.0*self.max_range[0]:
                new_predictions.append(0.0)
            elif score < 0.0:
                new_predictions.append((score + self.max_range[0]) / \
                                                      (2.0 * self.max_range[0]))
            elif score < self.max_range[1]:
                new_predictions.append((score + self.max_range[1]) / \
                                                      (2.0 * self.max_range[1]))
            else:
                new_predictions.append(1.0)
        
        plot_scores_test_fit, l_discrete_test_fit = \
                                        self._discretize(new_predictions,labels)
        len_list_test_fit, plot_emp_prob_test_fit = \
                               self._empirical_probability(l_discrete_test_fit)

        from pySPACE.tools.filesystem import  create_directory
        import os
        node_dir = os.path.join(result_dir, self.__class__.__name__)
        create_directory(node_dir)
        
        import pylab
        from matplotlib.transforms import offset_copy
        pylab.close()
        fig = pylab.figure(figsize=(10,10))
        ax = pylab.subplot(2,2,1)
        transOffset=offset_copy(ax.transData,fig=fig,x=0.05,y=0.1,units='inches')
        for x,y,s in zip(plot_scores_train,plot_emp_prob_train[1],len_list_train[1]):
            pylab.plot((x,),(y,),'ro')
            pylab.text(x,y,'%d' % s, transform=transOffset)
        
        pylab.plot((plot_scores_train[0],plot_scores_train[-1]),(0,1),'-')
        x1 = numpy.arange(-1.0*self.max_range[0],0.0,.02)
        x2 = numpy.arange(0.0,self.max_range[1],.02)
        y1 = (x1+self.max_range[0])/(2*self.max_range[0])
        y2 = (x2+self.max_range[1])/(2*self.max_range[1])
        pylab.plot(numpy.concatenate((x1,x2)),numpy.concatenate((y1,y2)),'-')
        pylab.xlim(plot_scores_train[0],plot_scores_train[-1])
        pylab.ylim(0,1)
        pylab.xlabel("SVM prediction Score (training data)")
        pylab.ylabel("Empirical Probability")
        
        ax = pylab.subplot(2,2,2)
        transOffset=offset_copy(ax.transData,fig=fig,x=0.05,y=0.1,units='inches')
        for x, y, s in zip(plot_scores_train_fit, plot_emp_prob_train_fit[1], 
                                                         len_list_train_fit[1]):
            pylab.plot((x,),(y,),'ro')
            pylab.text(x,y,'%d' % s, transform=transOffset)
        
        pylab.plot((plot_scores_train_fit[0],plot_scores_train_fit[-1]),(0,1),'-')
        pylab.xlim(plot_scores_train_fit[0],plot_scores_train_fit[-1])
        pylab.ylim(0,1)
        pylab.xlabel("SVM Probability (training data)")
        pylab.ylabel("Empirical Probability")
        
        ax = pylab.subplot(2,2,3)
        transOffset=offset_copy(ax.transData,fig=fig,x=0.05,y=0.1,units='inches')
        for x,y,s in zip(plot_scores_test,plot_emp_prob_test[1],len_list_test[1]):
            pylab.plot((x,),(y,),'ro')
            pylab.text(x,y,'%d' % s, transform=transOffset)
        
        pylab.plot((plot_scores_test[0],plot_scores_test[-1]),(0,1),'-')
        x1 = numpy.arange(-1.0*self.max_range[0],0.0,.02)
        x2 = numpy.arange(0.0,self.max_range[1],.02)
        y1 = (x1+self.max_range[0])/(2*self.max_range[0])
        y2 = (x2+self.max_range[1])/(2*self.max_range[1])
        pylab.plot(numpy.concatenate([[plot_scores_test[0],self.max_range[0]],
                               x1,x2,[self.max_range[1],plot_scores_test[-1]]]),
                   numpy.concatenate([[0.0,0.0],y1,y2,[1.0,1.0]]),'-')
        pylab.xlim(plot_scores_test[0],plot_scores_test[-1])
        pylab.ylim(0,1)
        pylab.xlabel("SVM prediction Score (test data)")
        pylab.ylabel("Empirical Probability")
        
        ax = pylab.subplot(2,2,4)
        transOffset=offset_copy(ax.transData,fig=fig,x=0.05,y=0.1,units='inches')
        for x, y, s in zip(plot_scores_test_fit, plot_emp_prob_test_fit[1], 
                                                          len_list_test_fit[1]):
            pylab.plot((x,),(y,),'ro')
            pylab.text(x,y,'%d' % s, transform=transOffset)
        
        pylab.plot((plot_scores_test_fit[0],plot_scores_test_fit[-1]),(0,1),'-')
        pylab.xlim(plot_scores_test_fit[0],plot_scores_test_fit[-1])
        pylab.ylim(0,1)
        pylab.xlabel("SVM Probability (test data)")
        pylab.ylabel("Empirical Probability")
        
        pylab.savefig(node_dir + "/reliable_diagrams_%d.png" % self.current_split)


_NODE_MAPPING = {"PSF": PlattsSigmoidFitNode,
                "SigTrans": SigmoidTransformationNode,
                "LinearFit":LinearFitNode
                }

