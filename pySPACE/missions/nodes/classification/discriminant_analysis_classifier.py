""" Discriminant analysis type classifiers """
import numpy
from copy import deepcopy
import warnings

from pySPACE.missions.nodes.base_node import BaseNode
from pySPACE.resources.data_types.prediction_vector import PredictionVector

class DiscriminantAnalysisClassifierBase(BaseNode):
    """ Template for discriminant analysis type classifier nodes
    
    This class has the is_trainable method and so on. Also a generic training
    method, which simply collects all training data, exists here.
    A classifier that inherits from here should implement stop_training and
    execute.   
    
    **Parameters**
    
        :class_labels:
            Determines the order of the two classes.
            This is important, when you want that the prediction
            value is negative for the first class and
            positive for the other one.
            Otherwise this variable is set by adding the labels,
            when they first occur.
        
            (*optional, default: []*)

        :prior_probability:
            The prior probability for any given sample to belong to either class.
            Pass a list with two entries in the same order as in class_labels. The
            values in prior_probability don't have to be actual probabilities, i.e.,
            they don't have to add up to 1: [1,3] is equivalent to [.25,.75].
            Note that this parameter is in some sense inverse to the SVM weights: 
            The underrepresented class will typically get assigned a higher SVM
            weight but the smaller prior probability.
            
            (*optional, default: [1.,1.]*)
        
    :Author: David Feess (David.Feess@dfki.de)
    :Created: 2012/05/30
    """
    def __init__(self, prior_probability = [1.,1.],
                 class_labels = [], **kwargs):
        super(DiscriminantAnalysisClassifierBase, self).__init__(**kwargs)
        
        self.set_permanent_attributes(classes=class_labels,
                                      prior_probability = prior_probability,
                                      x=None, # training data
                                      y=None) # training labels
        
    def is_trainable(self):
        """ Returns whether this node is trainable. """
        return True
    
    def is_supervised(self):
        """ Returns whether this node requires supervised training """
        return True

    def _train(self, data, label):
        """ Train node on given example *data* for class *label*.
        
            In this method, all data items and labels are buffered 
            in a matrix for batch training. 
        """
        # construct list of all labels
        if label not in self.classes:
            warnings.warn("Please give the expected classes to the classifier!"
                +" %s unknown. "%label +"Therefore define the variable "
                +"'class_labels' in your spec file, where you use your "
                +"classifier. For further info look at the node documentation.")
            self.classes.append(label)
        
        # map label to [-1,1], assuming that "target label"
        # is at position 0 and standards at position 1; skip all other labels
        label_index = self.classes.index(label)
        if label_index == 0:
            label_index = 1
        elif label_index == 1:
            label_index = -1
        else:
            return
        
        if self.x is None: # initialize data variables
            self.x = deepcopy(data)
            self.y = numpy.array(label_index)
        else: # stack data
            self.x = numpy.vstack((self.x, data))
            self.y = numpy.vstack((self.y, label_index))

class LinearDiscriminantAnalysisClassifierNode(DiscriminantAnalysisClassifierBase):
    """ Classify by linear discriminant analysis

    A detailed description can be found in:
    [1] Bishop, 2006 C.M. Bishop, "Pattern recognition and machine learning", 
    Springer (2006), 4.1.3-4.1.5
    
    Implementation strategies originate from
    [2] Schloegl et al., Adaptive Methods in BCI Research - An Introductory
    Tutorial. Brain-Computer Interfaces (2010) pp. 331
    
    
    **Parameters**
    
        See description of :class:`~DiscriminantAnalysisClassifierBase`
    
    **Exemplary Call**
    
    .. code-block:: yaml
    
        -
            node : LDA
            parameters :
                class_labels : ["Target","Standard"]
                prior_probability : [1,6]

    :Author: David Feess (David.Feess@dfki.de)
    :Created: 2012/05/29
    """
    def __init__(self, class_labels = [],**kwargs):
        super(LinearDiscriminantAnalysisClassifierNode, self).__init__\
                                                (class_labels=class_labels,
                                                 **kwargs)

        self.set_permanent_attributes(iECM=None, # inv extended cov matrix
                                      mu_m1=None, # class specific means
                                      mu_p1=None) #,
#                                      bw=None) # classification vector 


    def _stop_training(self, debug=False):
        """ Perform the actual model building """
        # this calculations strongly follow [2]
        self.x = self.x.T # samples x channels
        self.y = self.y.T
        # stack a row of ones to the data; (samples + 1) x channels
        self.x = numpy.vstack((numpy.ones_like(self.x[0,:]),self.x))
        # claculate extended cov matrix and pseudo inverse
        # ECM has entries [a,b;c,D] with a = NrSamples, b.T=c=data mean, D=cov
        ECM = numpy.dot(self.x, self.x.T) # eq. 16 in [2]
        self.iECM = numpy.linalg.inv(ECM/ECM[0,0])
        # calculate class-specific means
        self.mu_m1 = numpy.mean(self.x[1:,self.y[0,:]==-1], axis=1)
        self.mu_p1 = numpy.mean(self.x[1:,self.y[0,:]==1], axis=1)

        ### analytically equivalent but more elegant: use of w and b. ###
        ## calculate w and b (eqs. 40f in [2])
        # w = numpy.dot((mu_p1 - mu_m1), self.iECM[1:,1:]) # delta_mu*inv(cov)
        # b = -numpy.dot(ECM[1:,0],w.T) # -mu_x*w.T
        ## stack b and w to a joint model parameter 
        # self.bw = numpy.hstack((b,w))
        ## and then in execute: prediciton is [b,w]*[1,x].T (eq. 39 in [2])
        # m = float(numpy.dot(self.bw,data))
    
    def _execute(self, data):
        """ Executes the classifier on the given data vector """
        predicted_class = None
        # add feature that is constantly one (bias term)
        data = numpy.vstack((numpy.array([1]),data.T))
        
        # offset due to prior probabilities
        prior_shift =  numpy.log(float(self.prior_probability[0])/ \
                                 float(self.prior_probability[1]))
        
        # prediciton is [0,delta mu]*iECM*[1,x] (eq. 45 in [2])
        # (this is eqivalent to [b,w]*[1,x].T (eq. 39))
        m = numpy.dot(numpy.dot( 
                    numpy.hstack((numpy.array([0]),self.mu_p1 - self.mu_m1)).T,
                    self.iECM),
                    data)[0] + prior_shift
                    
        if m > 0:
            predicted_class = self.classes[0]
        else:
            predicted_class = self.classes[1]
        
        return PredictionVector(label = predicted_class, 
                                prediction = m, 
                                predictor = self)

class QuadraticDiscriminantAnalysisClassifierNode(DiscriminantAnalysisClassifierBase):
    """ Classify by quadratic discriminant analysis
    
    Performs a QDA classification (basically evaluates the log of a
    likelihood ratio test).
    
    Implementation originates from
    [1] Schloegl et al., Adaptive Methods in BCI Research - An Introductory
    Tutorial. Brain-Computer Interfaces (2010) pp. 331
    
    **Parameters**
    
        See description of DiscriminantAnalysisClassifierBase
    
    **Exemplary Call**
    
    .. code-block:: yaml
    
        -
            node : QDA
            parameters :
                class_labels : ["Target","Standard"]
                prior_probability : [1,6]

    :Author: David Feess (David.Feess@dfki.de)
    :Created: 2012/05/29
    """
    def __init__(self, class_labels = [],**kwargs):
        super(QuadraticDiscriminantAnalysisClassifierNode, self).__init__\
                                                (class_labels=class_labels,
                                                 **kwargs)

        self.set_permanent_attributes(ECM_p1=None, # class +1 ext cov matrix
                                      ECM_m1=None, # class -1 ECM
                                      iECM_p1=None, # class +1 inv ext cov matrix
                                      iECM_m1=None, # class -1 iECM
                                      logdetCM_p1=None, # logdet of cov
                                      logdetCM_m1=None)
    
    def _stop_training(self, debug=False):
        """ Perform the actual model building """
        # this calculations strongly follow [1]
        # stack a row of ones to the data; (samples + 1) x channels
        self.x = numpy.vstack((numpy.ones_like(self.x[:,0]),self.x.T))
        self.y = self.y.T
        # data for each class individually:
        x_p1 =  self.x[:,self.y[0,:]==1]
        x_m1 =  self.x[:,self.y[0,:]==-1]
        
        # calculate extended cov matrix and pseudo inverse for each class
        # ECM has entries [a,b;c,D] with a = NrSamples, b.T=c=data mean, D=cov
        # the logdet terms are needed for the classification function
        self.ECM_p1 = numpy.dot(x_p1, x_p1.T) # eq. 16 in [1]
        # the paper does not really use the inverse but the scaled inverse
        self.iECM_p1 = numpy.linalg.inv(self.ECM_p1/self.ECM_p1[0,0])
        self.logdet_p1 = self.logdet_from_ECM(self.ECM_p1)
        
        self.ECM_m1 = numpy.dot(x_m1, x_m1.T) # eq. 16 in [1]
        # the paper does not really use the inverse but the scaled inverse
        self.iECM_m1 = numpy.linalg.inv(self.ECM_m1/self.ECM_m1[0,0])
        self.logdet_m1 = self.logdet_from_ECM(self.ECM_m1)
        
    
    def _execute(self, data):
        """ Executes the classifier on the given data vector """
        predicted_class = None
        # add feature that is constantly one (bias term)
        data = numpy.vstack((numpy.array([1]),data.T)).T
        # The QDA evaluation currently uses the wikipedia formula, because
        # I didn't find a textbook that has it -.-
        # Basically, we perform a likelihood ratio test. the likelihood for 
        # class j is
        # (2*pi*det(Sgima_j))^(-1/2) * exp(-1/2 xF_jx.T) where
        # F_j = (x-mu_j) *        iSigma_j       * (x-mu_j).T
        #     =    [1,x] * {iECM_j - [1,0; 0,0]} * [1,x].T)
        # we use the log of the likelihood ratio, which boils down to:
        # {xFx+log(det(Sigma))}_i - {xFx+log(det(Sigma))}_j
        c=numpy.zeros_like(self.iECM_p1); c[0,0]=1 # c:=[1,0; 0,0]
        # xFx terms:
        xFx_p1 =  float(numpy.dot(data,numpy.dot(self.iECM_p1-c, data.T)))
        xFx_m1 =  float(numpy.dot(data,numpy.dot(self.iECM_m1-c, data.T)))
        
        # offset due to prior probabilities
        prior_shift = 2 * numpy.log(float(self.prior_probability[0])/ \
                                    float(self.prior_probability[1]))
        
        D = (xFx_p1 + self.logdet_p1) - (xFx_m1 + self.logdet_m1) + prior_shift
        if D < 0:
            predicted_class = self.classes[0]
        else:
            predicted_class = self.classes[1]
        
        return PredictionVector(label = predicted_class, 
                                prediction = D, 
                                predictor = self)

    def logdet_from_ECM(self, ECM):
        """ Compute logdet of cov matrix from extended cov matrix (ECM) """
        # This has to be done for both classes in trainng
        # first extract cov matric from extended cov matrix:
        Sigma = ECM[1:,1:]/ECM[0,0] - \
                                numpy.dot(ECM[:,1]/ECM[0,0], ECM[1,:]/ECM[0,0])
        return numpy.linalg.slogdet(Sigma)[1]

_NODE_MAPPING = {"QDA": QuadraticDiscriminantAnalysisClassifierNode,
                "LDA": LinearDiscriminantAnalysisClassifierNode}


