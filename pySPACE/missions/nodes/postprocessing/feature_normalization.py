""" Normalize :class:`~pySPACE.resources.data_types.feature_vector.FeatureVector`

"""

import os
import cPickle
import scipy.stats
import numpy
from collections import defaultdict

from pySPACE.missions.nodes.base_node import BaseNode
from pySPACE.resources.data_types.feature_vector import FeatureVector

from pySPACE.tools.filesystem import  create_directory

class InconsistentFeatureVectorsException(Exception):
    pass

class FeatureNormalizationNode(BaseNode):
    """ General node for Feature Normalization 

    The node should simply shift the data with the *translation* variable and
    afterwards scale it with the *mult* variable.
    
    This transformation can be loaded and stored
    and given to visualization tools.
    
    **Parameter**
        :load_path:
            An absolute path from which the normalization vectors are loaded.
            If not specified, these vectors are learned from the training data.

            (*optional, default: None*)


    **Exemplary Call**
    
    .. code-block:: yaml
    
        -
            node : FeatureNormalizationNode
            parameters :
                load_path: "/Users/mustermann/proj/examples/FN.pickle"

    .. warning:: This base node only works alone, when load_path is specified.

    :input:  FeatureVector
    :output: FeatureVector
    :Author: Mario Krell (mario.krell@dfki.de)
    :Created: 2012/03/28
    """
    def __init__(self, load_path = None, **kwargs):
        self.load_path = load_path
        super(FeatureNormalizationNode, self).__init__(**kwargs)
        self.set_permanent_attributes(samples = [], feature_names = [],
                                      load_path = load_path,
                                      feature_indices = None,
                                      tolerance = 10**-9)

    def is_trainable(self):
        return self.load_path == None

    def store_state(self, result_dir, index=None): 
        """ Stores transformation and feature names in the given directory *result_dir* """
        if self.store:
            node_dir = os.path.join(result_dir, self.__class__.__name__)
            # self.__class__.__name__)
            create_directory(node_dir)
            name = "%s_sp%s.pickle" % ("FN", self.current_split)
            result_file = open(os.path.join(node_dir, name), "wb")
            result_file.write(cPickle.dumps((self.translation, 
                                             self.mult, 
                                             self.feature_names), protocol=2))
            result_file.close()
        super(FeatureNormalizationNode,self).store_state(result_dir)

    def _train(self, data):
        """ Collects the values each feature takes on in the training set. """
        # Check that feature vectors are compatible
        self.extract_feature_names(data)
        data_array = data.view(numpy.ndarray)
        self.collect_data(data_array)
        
    def extract_feature_names(self, data):
        if self.feature_names == []:
            self.feature_names = data.feature_names
            self.dim=len(self.feature_names)
        elif self.feature_names != data.feature_names:
            raise InconsistentFeatureVectorsException("Two feature vectors used during training do not contain the same features!")
        

    def _execute(self, data):
        """ Normalizes the feature vector data.
        
        Normalizes the feature vector data by subtracting
        the *translation* variable and scaling it with *mult*.
        
        .. todo:: check if problems in data transformation still occur
        """
        if not (self.load_path is None or self.load_path=="already_loaded"):
            self.replace_keywords_in_load_path()
            load_file = open(self.load_path, 'r')
            self.translation, self.mult, self.feature_names = cPickle.load(load_file)
            self.load_path = "already_loaded"
        self.extract_feature_names(data)
        # mapping of feature names if current features are a subset
        # of loaded feature normalization in the training
        if self.feature_indices is None:
            try:
                self.feature_indices = [self.feature_names.index(feature_name) 
                                        for feature_name in data.feature_names]
            except ValueError:
                raise InconsistentFeatureVectorsException("Cannot normalize a feature vector "
                                                          "with an unknown feature dimension!")
        # The data reference is not changed or deleted but here it is
        # temporarily replaced. 
        if not self.translation is None:  
            data = (data - self.translation[self.feature_indices]) \
                    * self.mult[self.feature_indices]
        else :
            data = data * 0
        # Handle cases where lower and upper bound are identical
        # This is for example the case, when 
        # one feature generating measurement device is off or out of order
        # TODO check if still needed
        data[numpy.isnan(data)] = 0.0
        data[numpy.isinf(data)] = 0.0
        # for i, v in enumerate(data[0,:]):
        #     if v > 1:
        #         data[0,i] = 1 + self.scaling*(1 - math.exp(1-v))
        #     elif v < 0:
        #         data[0,i] = self.scaling*(math.exp(v)-1)
        return FeatureVector(data,
                             data.feature_names)

    def collect_data(self,data):
        self.samples.append(numpy.array(data[0,:]))


class OutlierFeatureNormalizationNode(FeatureNormalizationNode):
    """ Map the feature vectors of the training set to the range [0,1]^n
    
    A class that normalizes each dimension of the feature 
    vector so that an upper boundary value (learned from in the training set)
    is mapped to 1, and a lower boundary value to 0.
    All other values are linearly interpolated. Optionally, one can specify
    an *outlier_percentage* that determines which ratio of the training data
    is considered to be a potential outlier. *outlier_percentage*/2 samples
    are allowed to be larger than the determined upper boundary, and
    *outlier_percentage*/2 samples are allowed to be smaller than the 
    determined lower boundary. 

    **Parameters**
        :outlier_percentage:
            The percentage of training instances that are potential outliers.

            (*optional, default: 0*)

    **Exemplary Call**
    
    .. code-block:: yaml
    
        -
            node : OutlierFeatureNormalization
            parameters :
                outlier_percentage : 10
                load_path: "/Users/mustermann/proj/examples/FN.pickle"
    
    :Author: Jan Hendrik Metzen (jhm@informatik.uni-bremen.de)
    :Created: ??
    :Revised (1): 2009/07/16
    :Revised (2): 2009/09/03
    
    """
    def __init__(self, outlier_percentage = 0, **kwargs):
        super(OutlierFeatureNormalizationNode, self).__init__(**kwargs)
        self.set_permanent_attributes(outlier_percentage = outlier_percentage,
                                      samples = defaultdict(list))

    def collect_data(self,data):
        for feature_index, feature_value in enumerate(data[0,:]):
            self.samples[feature_index].append(feature_value)

    def _stop_training(self):
        """ Computes the upper and lower boundary for normalization.
        
        For this computation, the largest and smallest *outlier_percentage*/2
        examples for each feature dimension are ignored. 
        The smallest and largest remaining example are used as lower and upper 
        boundary.
        """
        self.lower_bounds = numpy.zeros((1, len(self.samples)))
        self.upper_bounds = numpy.zeros((1, len(self.samples)))
        for feature_index, feature_values in self.samples.iteritems():
            self.lower_bounds[0, feature_index] = \
                scipy.stats.scoreatpercentile(feature_values, 
                                              self.outlier_percentage/2)
            self.upper_bounds[0, feature_index] = \
                scipy.stats.scoreatpercentile(feature_values, 
                                              100 - self.outlier_percentage/2)
        # Cleaning up...
        self.samples = defaultdict(list)
        # name unification
        self.translation = self.lower_bounds[0,:]
        self.mult = 1/(self.upper_bounds[0,:]-self.lower_bounds[0,:])
        self.mult[numpy.isinf(self.mult)] = 0.0
        self.mult[numpy.isnan(self.mult)] = 0.0

class GaussianFeatureNormalizationNode(FeatureNormalizationNode):
    """ Transform the features, such that they have zero mean and variance one
    
    A class that normalizes each dimension of the feature 
    vector so that it has zero mean and variance one.
    The relevant values are learned from the training set.

    **Exemplary Call**
    
    .. code-block:: yaml
    
        -
            node : Gaussian_Feature_Normalization
            parameters :
                load_path: "/Users/mustermann/proj/examples/GFN.pickle"
    
    :Author: Mario Krell (Mario.Krell@dfki.de)
    :Created: 2011/04/15
    
    """
    
    def __init__(self, **kwargs):
        self.n = 0
        self.mean_diff = None
        self.translation = None
        self.mult = None
        super(GaussianFeatureNormalizationNode, self).__init__(**kwargs)
        
    def _stop_training(self):
        """ Computes mean and std deviation of each feature"""
        if not self.is_retrainable():
            self.translation = numpy.mean(numpy.array(self.samples),axis=0)
            self.mult = numpy.std(numpy.array(self.samples),axis=0)
            for i in range(self.dim):
                if not(abs(self.mult[i]) < self.tolerance):
                    self.mult[i] = 1/self.mult[i]
                else:
                    self.mult[i] = 1
            self.n = len(self.samples)
                
    def _train(self, data):
        if not self.is_retrainable():
            super(GaussianFeatureNormalizationNode,self)._train(data)
        else:
            self.extract_feature_names(data)
            data_array = data.view(numpy.ndarray)
            data_array = data_array[0,:]
            if self.translation is None:
                self.translation = numpy.zeros(data_array.shape)
                self.mean_diff = numpy.zeros(data_array.shape)
                self.mult = numpy.zeros(data_array.shape)
            self.n += 1
            delta = data_array - self.translation
            self.translation += delta / self.n;
            self.mean_diff = self.mean_diff + delta * (data_array - self.translation)
            for i in range(self.dim):
                if not(self.mean_diff[i] == 0):
                    self.mult[i] = max((self.n - 1),1)/self.mean_diff[i]
            

    def _inc_train(self, data, class_label=None):
        self._train(data)

class HistogramFeatureNormalizationNode(FeatureNormalizationNode):
    """ Transform the features, such that they have zero mean in the main bit in the histogram and variance one on that bit.

    The relevant values are learned from the training set.

    **Exemplary Call**
    
    .. code-block:: yaml
    
        -
            node : Histogram_Feature_Normalization
            parameters :
                load_path: "/Users/mustermann/proj/examples/FN.pickle"
    
    :Author: Mario Krell (Mario.Krell@dfki.de)
    :Created: 2011/04/15
    
    """
    def _stop_training(self):
        """ Computes mean and std deviation of each feature"""
        mean=[]
        std=[]
        self.feature_values = numpy.array(self.samples).T
        for values in self.feature_values:
            hvalues,bins = numpy.histogram(values, bins = 3)
            maxindex = hvalues.argmax()
            min_bound = bins[maxindex]
            max_bound = bins[maxindex+1]
            i=0
            max_sum=0
            relevant_values=[]
            for value in values:
                    if min_bound <= value <= max_bound:
                        relevant_values.append(value)
#                        max_sum += value
#            mean.append(1.0*max_sum/i)
            mean.append(numpy.mean(relevant_values))
            std.append(numpy.std(relevant_values))
        self.translation = numpy.array(mean)
        self.mult = numpy.array(std)
        #self.mult = numpy.std(numpy.array(self.samples),axis=0)
        for i in range(self.dim):
            if not(abs(self.mult[i]) < self.tolerance):
                self.mult[i] = 1/self.mult[i]
            else:
                self.mult[i] = 1
        # Cleaning up...
        self.samples = []
        self.feature_values = []
        mean = []
        std = []

class EuclideanFeatureNormalizationNode(BaseNode):
    """ Normalize feature vectors to Euclidean norm with respect to dimensions

    **Parameters**
    
        :dimension_scale:
            Scale the output to ||x|| * dim(x)
            (to get bigger values) 

            (*optional, default: False*)

    **Exemplary Call**
    
    .. code-block:: yaml
    
        -
            node : Euclidian_Feature_Normalization
            parameters :
                dimension_scale : True
    
    :Author: Mario Krell (Mario.Krell@dfki.de)
    :Created: 2011/04/15
    
    """
    def __init__(self, dimension_scale = False, **kwargs):
        super(EuclideanFeatureNormalizationNode, self).__init__(**kwargs)
        self.set_permanent_attributes(dim = None, 
                                      dimension_scale=dimension_scale,
                                      feature_names=[])

    def _execute(self, data):
        """ Normalizes the samples vector to norm one """
        if self.feature_names == []:
            self.feature_names = data.feature_names
        elif self.feature_names != data.feature_names:
            raise InconsistentFeatureVectorsException("Two feature vectors used during training do not contain the same features!")
        x = data.view(numpy.ndarray)
        a = x[0,:]
        if self.dim == None:
            self.dim = len(a)
        a = a*numpy.float128(1)/numpy.linalg.norm(a)
        if self.dimension_scale:
            a = FeatureVector([len(a)*a],self.feature_names)
            return a
        else:
            return FeatureVector([a],self.feature_names)
        
    def store_state(self, result_dir, index=None): 
        """ Stores this node in the given directory *result_dir* """
        if self.store:
            pass


_NODE_MAPPING = {"Feature_Normalization": OutlierFeatureNormalizationNode,
                "Outlier_Feature_Normalization": OutlierFeatureNormalizationNode,
                "FN": OutlierFeatureNormalizationNode,
                "O_FN": OutlierFeatureNormalizationNode,
                "Euclidean_Feature_Normalization": EuclideanFeatureNormalizationNode,
                "E_FN": EuclideanFeatureNormalizationNode,
                "Gaussian_Feature_Normalization": GaussianFeatureNormalizationNode,
                "G_FN": GaussianFeatureNormalizationNode,
                "Histogram_Feature_Normalization": HistogramFeatureNormalizationNode,
                "H_FN": HistogramFeatureNormalizationNode}
