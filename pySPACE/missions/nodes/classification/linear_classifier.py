""" Implement standard linear classifiers """


import numpy
from copy import deepcopy

from pySPACE.missions.nodes.base_node import BaseNode
# the output is a prediction vector
from pySPACE.resources.data_types.prediction_vector import PredictionVector

try:
    import mdp
    from mdp.nodes import FDANode
except:
    pass

class NaiveBayesClassifierNode(BaseNode):
    """ NaiveBayes Classifier Algorithm"""
    def __init__(self, class_labels = [],**kwargs):
        super(NaiveBayesClassifierNode, self).__init__(**kwargs)
       
        self.set_permanent_attributes(class_labels=class_labels, data = [] , mu=[],var=[], ap = [] )

    def is_trainable(self):
        """ Returns whether this node is trainable. """
        return True
    
    def is_supervised(self):
        """ Returns whether this node requires supervised training """
        return True
    
    def _stop_training(self, debug=False):

        for d in self.data:
            self.mu.append( d.mean(axis=0) )
            self.var.append( d.var(axis=0) )
            self.ap.append(len(d) )

        return

    def _execute(self, data):
        """ Executes the classifier on the given data vector x"""
        
        res = numpy.zeros( len(self.ap) );
        for index, item in enumerate(self.ap):
            fac =  1.0/ ( numpy.sqrt(2.0 * numpy.pi * self.var[index] ) )
            term = numpy.exp( - 0.5* ( ( data - self.mu[index] )**2 / (self.var[index] ) ) )
            c = fac*term
            res[index] =  self.ap[index] * c.prod()
        
        classifications = res.argmax();
        return PredictionVector(label = self.class_labels[classifications], prediction =classifications   , predictor = self)
    
    def _train(self, data, class_label):
        """ Collect data and labels """

        # Remember the labels
        if class_label not in self.class_labels: 
            self.class_labels.append(class_label)
            self.data.append( numpy.array(data) ) 
        else:
            index = self.class_labels.index(class_label)
            self.data[index] = numpy.vstack((self.data[index],data))


class FDAClassifierNode(BaseNode):
    """Classify with Fisher's linear discriminant analysis
    
    .. note:: Wrapper around the MDP FDA node
    
    **Parameters**
    
        :class_labels:
            Determines the order of the two classes.
            This is important, when you want that the prediction
            value is negative for the first class and
            positive for the other one.
            Otherwise this variable is set by adding the labels,
            when they first occur.
        
            (*optional, default: []*)
    
    **Exemplary Call**
    
    .. code-block:: yaml
    
        -
            node : FDA_Classifier
            parameters :
                class_labels : ["Target","Standard"]
    
    :Author: Jan Hendrik Metzen (jhm@informatik.uni-bremen.de)
    :Created: 2008/08/26
    :Last change: 2010/08/13 by Mario Krell
    
    """
    def __init__(self, class_labels = [],**kwargs):
        super(FDAClassifierNode, self).__init__(**kwargs)
        
        self.set_permanent_attributes(class_labels=class_labels,
                                      positive_class=None,
                                      negative_class=None,
                                      MDPflow=None,
                                      data=[])
    def is_trainable(self):
        """ Returns whether this node is trainable """
        return True
    
    def is_supervised(self):
        """ Returns whether this node requires supervised training """
        return True
    
    def _execute(self, x, fda_range=None):
        """ Executes the classifier on the given data vector x"""
        if self.positive_class == None:
            # The FDA_Classifier_Node does not provide a mapping
            # from its continuous output to a class label
            # In order to do that, we test the class means and see 
            # whether the yield in positive or negative results
            label_0 = self.class_labels[0]
            classifier_result = \
                 self.MDPflow.execute(self.MDPflow[0].means[label_0])
            if classifier_result[:, 0] > 0.0:
                self.positive_class = self.class_labels[0]
                self.negative_class = self.class_labels[1]
            else:
                self.positive_class = self.class_labels[1]
                self.negative_class = self.class_labels[0]
        data=x.view(numpy.ndarray)
        f_projection = self.MDPflow.execute(data, fda_range)
        
        classifications = numpy.where(f_projection[:, 0] > 0.0,
                                         self.positive_class,
                                         self.negative_class)
        return PredictionVector(label = classifications[0], prediction = float(f_projection[:, 0]), predictor = self)
    
    def _train(self, x, class_label):
        """ Collect data for later training """
        # Remember the labels
        if class_label not in self.class_labels: 
            self.class_labels.append(class_label)
        data=x.view(numpy.ndarray)
        self.data.append((data,class_label))

    def _stop_training(self):
        """ Delegate training to superclass method """
        if self.MDPflow is None:
            self.MDPflow=mdp.Flow([FDANode()])
        self.MDPflow.train([self.data])

class BayesianLinearDiscriminantAnalysisClassifierNode(BaseNode):
    """ Classify with the bayesian linear discriminant analysis
    
    A detailed description can be found in:
    
    Ulrich Hoffmann et al.
    "An efficient P300-based brain-computer interface for disabled subjects", 
    Journal of Neuroscience Methods, Volume 167, Issue 1,
    
    Bishop, 2006 C.M. Bishop, "Pattern recognition and machine learning", 
    Springer (2006)
    
    MacKay, 1992 D.J.C. MacKay, "Bayesian interpolation", 
    Neural Comput 4 (3) (1992) pp. 415-447
    
    **Parameters**
    
        :class_labels:
            Determines the order of the two classes.
            This is important, when you want that the prediction
            value is negative for the first class and
            positive for the other one.
            Otherwise this variable is set by adding the labels,
            when they first occur.
        
            (*optional, default: []*)
    
    **Exemplary Call**
    
    .. code-block:: yaml
    
        -
            node : BLDA_Classifier
            parameters :
                class_labels : ["Target","Standard"]
    
    :Author: Hendrik Woehrle (hendrik.woehrle@dfki.de)
    :Created: 2011/07/25
    """
    def __init__(self, class_labels = [],**kwargs):
        super(BayesianLinearDiscriminantAnalysisClassifierNode, self).__init__\
                                                (**kwargs)
        
        self.set_permanent_attributes(class_labels=class_labels,
                                      x=None,
                                      y=None)
        
    def is_trainable(self):
        """ Returns whether this node is trainable. """
        return True
    
    def is_supervised(self):
        """ Returns whether this node requires supervised training """
        return True
    
    def _train(self, data, label):
        """ Train node on given example *data* for class *label*.
        
            In this method, all data items and labels are buffered 
            for batch training in a matrices. 
        """
            
        # construct list of all labels
        if not label in self.class_labels:
            self.class_labels.append(label)
        
        # map label to [-1,1], assuming that "target label" 
        # is at position 0, and standards at position 1,
        # skip all other labels
        label_index = self.class_labels.index(label)    
        if label_index==0:
            label_index=1
        elif label_index==1:
            label_index=-1
        else:
            return
            
        if self.x is None:
            self.x = deepcopy(data)
            self.y = numpy.array(label_index)
        else:
            self.x = numpy.vstack((self.x, data))
            self.y = numpy.vstack((self.y, label_index))

                
    def _stop_training(self, debug=False):
        """ Perform the actual model building by performing bayesian regression"""
        
        self.x = self.x.transpose()
        self.y = self.y.transpose()
        
        # compute regression targets from class labels (to do lda via regression)
        n_posexamples = numpy.float((self.y == 1).sum());
        n_negexamples = numpy.float((self.y == -1).sum());
        n_examples    = n_posexamples + n_negexamples;
        
        p_ratio = n_examples/n_posexamples;
        n_ratio = -n_examples/n_negexamples;
        
        self.y = numpy.where(self.y==1.,p_ratio,n_ratio)
        
        # dimension of feature vectors
        n_features = self.x.shape[0]
        
        # add feature that is constantly one (bias term)
        self.x = numpy.vstack((self.x, numpy.ones((1,self.x.shape[1]))))
        
        # initialize variables for fast iterative estimation of alpha and beta
        d_beta = numpy.inf;                # (initial) diff. between new and old beta  
        d_alpha = numpy.inf;               # (initial) diff. between new and old alpha 
        alpha    = 25;                     # (initial) inverse variance of prior distribution
        biasalpha = 0.00000001;            # (initial) inverse variance of prior for bias term
        beta     = 1;                      # (initial) inverse variance around targets
        stopeps  = 0.0001;                 # desired precision for alpha and beta
        i        = 1;                      # keeps track of number of iterations
        maxit    = 500;                    # maximal number of iterations 
        
        # needed for fast estimation of alpha and beta
        [d,v]=numpy.linalg.eig(numpy.dot(self.x,self.x.transpose()))
        
        vx = numpy.dot(v.transpose(),self.x)
        vxy = numpy.dot(vx,self.y.transpose());
        e = numpy.ones((n_features,1));
        
        #print vxy.shape
        #print e.shape
        d = numpy.reshape(d,(d.shape[0],1))
        #print d.shape
        
        # estimate alpha and beta iteratively
        while ((d_alpha > stopeps) or (d_beta > stopeps)) and (i < maxit):
            alphaold = alpha
            betaold  = beta

            beta_d_alpha_e_inv = 1./(beta*d+numpy.vstack((alpha*e,biasalpha)))
            
            beta_d_alpha_e_vxy = beta_d_alpha_e_inv * vxy
                       
            m = numpy.dot(beta*v,beta_d_alpha_e_vxy)
                    
            err = numpy.dot(m.transpose(),self.x)
            err = self.y-err
            err = numpy.sum(err**2)

            alpha_e = numpy.vstack((alpha*e,biasalpha))
            #print alpha_e.shape
            
            beta_d = (beta*d)
            #beta_d = beta_d.reshape((n_features+1,1))
            #print beta_d.shape
            
            alpha_e_beta_d = alpha_e + beta_d
            #print alpha_e_beta_d.shape 
                        
            beta_d_alpha_e_inv = 1./(beta_d+numpy.vstack((alpha*e,biasalpha)))

            #print beta_d_alpha_e_inv.shape

            
            gamma = sum((beta*d)*beta_d_alpha_e_inv)
            alpha = gamma/numpy.dot(m.transpose(),m)
            beta  = (n_examples - gamma)/err
            #if b.verbose
            #    fprintf('Iteration %i: alpha = %f, beta = %f\n',i,alpha,beta);
        
            d_alpha = abs(alpha-alphaold);
            d_beta  = abs(beta-betaold);
            i = i + 1;
        
        
        # process results of estimation 
        if i < maxit:
            # compute the log evidence
            # this can be used for simple model selection tasks
            # (see MacKays paper)
            beta_d_alpha_e = beta*d+numpy.vstack((alpha*e,biasalpha))
            
            self.evidence = (n_features/2)*numpy.log(alpha) + (n_examples/2)*numpy.log(beta) - \
                        (beta/2)*err - (alpha/2)*numpy.dot(m.transpose(),m) - \
                        0.5*sum(numpy.log(beta_d_alpha_e)) - (n_examples/2)*numpy.log(2*numpy.pi);
                        
            #print beta_d_alpha_e.shape
            #print v.shape
            
            # store alpha, beta, the posterior mean and the posterior precision-
            # matrix in class attributes
            self.b_alpha = alpha;
            self.b_beta  = beta;
            self.b_w     = m;
            beta_d_alpha_e = numpy.diagflat(1./beta_d_alpha_e)
            
            b_p     = numpy.dot(v,beta_d_alpha_e)
            self.b_p = numpy.dot(b_p,v.transpose())
        
        else:
        
            raise Exception("No convergence of Baye's fitting")

    
    def _execute(self, data):
        """ Executes the classifier on the given data vector x"""

        predicted_class = None
        
        # add feature that is constantly one (bias term)
        data=data.transpose()
        data = numpy.vstack((data,numpy.array(1)))
        
        # compute mean of predictive distributions
        m = float(numpy.dot(self.b_w.transpose(),data));
        
        if m < 0:
            predicted_class = self.class_labels[0]
        else:
            predicted_class = self.class_labels[1]
        
        return PredictionVector(label = predicted_class, 
                                prediction = m, 
                                predictor = self)


_NODE_MAPPING = {"FDA_Classifier": FDAClassifierNode,
                "BLDA_Classifier": BayesianLinearDiscriminantAnalysisClassifierNode,
                }
