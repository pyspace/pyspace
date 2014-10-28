""" Extract statistical properties like moments or correlation coefficients

**Known issues**
    No unit tests!
"""
import numpy
import scipy.stats
import copy
from matplotlib import mlab
import warnings

from pySPACE.missions.nodes.base_node import BaseNode
from pySPACE.resources.data_types.feature_vector import FeatureVector


class StatisticalFeatureNode(BaseNode):
    """ Compute statistical features, e.g. central/raw moments, median, min etc.
        
    This node computes statistical features like raw and central moments
    of k-th order:
        
    1. raw_moment_of_k_th_order

        .. math::

            \\frac{1}{n} \\cdot \\sum_{i=0}^n (x_i^k)

        - k=1: mean;
        - k=2: square root  of value is quadratic mean

    2. central_moment_of_k_th_order

        .. math::

            \\frac{1}{n} \\cdot  sum_{i=0}^n (x_i - mean)^k

        - k=1: 0;
        - k=2: empirical variance;
        - k=3: empirical skewness;
        - k=4: kurtosis

    In addition there could be computed: median, quadratic mean, 
    standard deviation, minimum/maximum amplitude and 
    the number of amplitudes above a given threshold.
    
    **Parameters**
    
        :raw_moment_order:
            The highest order (as an integer number) of raw moments that will
            be computed. For example:

            :0: - no raw moment computation,
            :1: - mean
            :2: - mean *and* raw moment 2nd order
            :3: - etc.

            .. note::
                Values greater then 2 might take long computation time and
                are expected not to have any useful information.
                   
            (*optional, default: 1*)
        
        :central_moment_order: 
            The highest order (as an integer number) of central moments that
            will be computed. For example:

            :1: - no central moment computation
            :2: - empirical variance
            :3: - empirical variance *and* empirical skewness
            :etc.:
            
            (*optional, default: 1*)
             
        :std:  
            True or False. If True the standard deviation of each channel is
            computed and retained as feature.
            
            (*optional, default: False*)
            
        :std_function:
            A string of a Python *eval*able function. Specifies a function,
            e.g. numpy.std, if the algorithm should not use use the one in
            *standard_deviation*. Specification of this parameter makes only
            sense if *std* is "True".
            
            (*optional, default: None*)
            
        :quadratic_mean:
            True or False. If True the square root of the second raw moment
            (quadratic mean) is computed and retained as feature.
            
            (*optional, default: False*)
            
        :median:
            True or False. If True the median of each channel is computed
            and retained as feature.
            
            (*optional, default: False*)
            
        :minimum: 
            True or False. If True the minimum value of each channel is stored
            as a feature.
           
            (*optional, default: False*)
            
        :maximum:
            True or False. If True the maximum value of each channel is stored
            as a feature.
            
            (*optional, default: False*)
            
        :artifact_threshold:
            A float value. The number of amplitudes within a window above the
            given *artifact_threshold* is calculated and used as feature.
            
            .. note::
                    The threshold depends on the actual resolution of the values.
                    Check header file (for EEG data).
            
            .. todo::   Integrate resolution in TimeSeries data type and then scale.
                        (The input here should be in micro-volt.)
            
            (*optional, default: None*)
    
    **Known issues**

    .. todo:: Try to use the methods on the whole data array
    
    **Exemplary Call**
    
    .. code-block:: yaml
    
        -
            node : Statistical_Features
            parameters :
                central_moment_order : 0
                std : True
                median : True
                artifact_threshold : 500.0
    
    :Author: Anett Seeland (anett.seeland@dfki.de)
    :Created: 2009/10/06
    :Refactoring: 2014/09/11 by Mario Michael Krell
    """

    def __init__(self, raw_moment_order=1, central_moment_order=1,
                 std=False, std_function=None, quadratic_mean=False,
                 median=False, minimum=False, maximum=False,
                 artifact_threshold=None, **kwargs):
        super(StatisticalFeatureNode, self).__init__(**kwargs)
        self.set_permanent_attributes(raw_moment_order=raw_moment_order,
                                      central_moment_order=central_moment_order,
                                      std=std, std_function=std_function,
                                      quadratic_mean=quadratic_mean,
                                      median=median, minimum=minimum,
                                      maximum=maximum,
                                      artifact_threshold=artifact_threshold,
                                      feature_names=None)

    def raw_moment(self, one_channel_time_series, k):
        """Compute the k'th-order raw moment of a given channel.
        
        **Arguments**
        
            :one_channel_time_series:
                a one dimensional time series
            
            :k:
                order of raw moment
        """
        # raising int arrays to higher power can cause overflow!!!
        # Hence we do the power-raising manually and NOT by "array ** k" 
        moment_sum = 0.0
        for index,  value in enumerate(one_channel_time_series):
            power = value ** k
            moment_sum += power
        raw_moment = moment_sum / len(one_channel_time_series)
        return raw_moment
    
    def amplitudes_above(self, one_channel_time_series, threshold):
        """ Return the number of values about the given threshold in the given time series.
        
        **Arguments**
        
            :window:
                a tow dimensional time series

            :threshold:
                a number above the values are count.
        """
        count = 0
        for value in one_channel_time_series:
            if value > threshold or value < -threshold:
                count += 1
        return count
    
    def _execute(self, x):
        """ Calculates statistical features """
        # determine what has to be computed and initialize data structures
        data = x.get_data()
        if self.central_moment_order == 0:
            self.central_moment_order = 1  # only for feature_size computation
        feature_size = self.raw_moment_order * len(x.channel_names) + \
            (self.central_moment_order-1) * len(x.channel_names) + \
            self.std * len(x.channel_names) + \
            self.quadratic_mean * len(x.channel_names) + \
            self.median * len(x.channel_names) + \
            self.minimum * len(x.channel_names) + \
            self.maximum * len(x.channel_names)
        if not self.artifact_threshold is None:
            feature_size += 1
        statistical_features = numpy.zeros((feature_size, ), numpy.float64)
        # initialize the actual feature_index
        feature_index = 0

        # for every channel...
        # TODO: Try to use the methods on the whole data array
        for index in range(len(x.channel_names)):  
            current_channel = data[:, index]
            # in these cases it is auspicious to compute and store variables
            # cause we will need them again
            if self.raw_moment_order > 0 or self.central_moment_order > 1 or \
                    self.std or self.quadratic_mean:
                average = numpy.mean(current_channel)
            if self.raw_moment_order > 1 or self.quadratic_mean:
                # self.raw_moment(current_channel, 2) 
                second_raw_moment = scipy.stats.ss(current_channel) 
                # print second_raw_moment
            if self.raw_moment_order > 0:  # raw_moment_of_1_th_order needed?
                # it's the mean, so don't compute it again
                statistical_features[feature_index] = average
                feature_index += 1  # update feature_index
                if self.raw_moment_order > 1:  # raw_moment_2nd_order needed?
                    # we have already computed it
                    statistical_features[feature_index] = second_raw_moment
                    feature_index += 1
                    # for the other orders of raw_moments
                    for order in range(3, self.raw_moment_order+1): 
                        statistical_features[feature_index] = \
                            self.raw_moment(current_channel, order)
                        feature_index += 1
            # central_moment
            for order in range(2, self.central_moment_order+1): 
                statistical_features[feature_index] = \
                    scipy.stats.moment(current_channel, order)
                feature_index += 1
            # standard_deviation
            if self.std:
                statistical_features[feature_index] = \
                    numpy.std(current_channel)
                feature_index += 1
            # quadratic_mean    
            if self.quadratic_mean: 
                # we stored relevant results before
                statistical_features[feature_index] = second_raw_moment ** 0.5
                feature_index += 1
            # median
            if self.median:
                statistical_features[feature_index] = \
                    numpy.median(current_channel)
                feature_index += 1
            # minimum
            if self.minimum:
                statistical_features[feature_index] = \
                    numpy.amin(current_channel)
                feature_index += 1
            # maximum
            if self.maximum:
                statistical_features[feature_index] = \
                    numpy.amax(current_channel)
                feature_index += 1
                
            if self.artifact_threshold is None:
                statistical_features[-1] += self.amplitudes_above(
                    current_channel, self.artifact_threshold)
        # if feature_names have to be determined
        if self.feature_names is None:
            # initialize data structure
            self.feature_names = []
            for name in x.channel_names:
                # raw_moment
                for order in range(1,  self.raw_moment_order+1):
                    self.feature_names.append(
                        "RAW_MOMENT_%d_%s" % (order, name))
                # central_moment
                for order in range(2, self.central_moment_order+1):
                    self.feature_names.append(
                        "CENTRAL_MOMENT_%d_%s" % (order, name))
                # standard_deviation
                if self.std:
                    self.feature_names.append("STD_%s" % (name))
                # quadratic_mean
                if self.quadratic_mean: 
                    self.feature_names.append("QUAD_MEAN_%s" %(name))
                # median
                if self.median: 
                    self.feature_names.append("MEDIAN_%s" % (name))
                # minimum
                if self.minimum: 
                    self.feature_names.append("MIN_%s" % (name))
                # maximum
                if self.maximum:
                    self.feature_names.append("MAX_%s" % (name))
            if not self.artifact_threshold is None:
                self.feature_names.append(
                    "AMP_ABOVE_%.2f" % (self.artifact_threshold))
                    
        feature_vector = FeatureVector(numpy.atleast_2d(statistical_features),
                                       self.feature_names)
        return feature_vector


class PearsonCorrelationFeatureNode(BaseNode):
    """ Compute pearson correlation of all pairs of channels
    
    This node computes for all pairs of channels the Pearson product-moment 
    correlation coefficient of certain time segments and 
    returns each of correlation coefficient as feature.

    .. todo: Calculate maximum number of segments and catch wrong usage.

    **Parameters**
    
        :segments:
            The number of segments the time series window is split.
            
            (*optional, default: 1*)
            
        :max_segment_shift:
            If 0, only the same segments of the two channels are compared.
            For n, each segment is also compared with the n,n-1,...,0 previous
            and later segments of the other channel.
                           
            (*optional, default: 0*)
     
    **Exemplary Call**
     
    .. code-block:: yaml
    
        -
            node : Pearson_Correlation_Features
            parameters :
                segments : 3
                max_segment_shift : 2
            
    :Author: Jan Hendrik Metzen (jhm@informatik.uni-bremen.de)
    :Created: 2009/03/18
    """
    
    def __init__(self, segments=1, max_segment_shift=0, *args, **kwargs):
        super(PearsonCorrelationFeatureNode, self).__init__(*args, **kwargs)
       
        assert(max_segment_shift < segments)
     
        self.set_permanent_attributes(segments=segments,
                                      segment_border_indices=None,
                                      max_segment_shift=max_segment_shift)

    def _execute(self, x):
        
        # Compute the indices of the segment borders lazily when the data is
        # known
        if self.segment_border_indices == None:
            datapoints = x.shape[0]
            borders = [k * datapoints / (self.segments + 1) 
                            for k in range(0, self.segments + 2)]
            self.segment_border_indices = [(borders[i], borders[i + 2])
                                                for i in range(self.segments)]
        data = x.view(numpy.ndarray)
        features = []
        feature_names = []
        # Iterate over all segment combinations:
        for segment_index_channel1 in range(self.segments):
            segment_borders1 = \
                           self.segment_border_indices[segment_index_channel1]
            for segment_index_channel2 in range(0, min(self.segments, 
                        segment_index_channel1 + self.max_segment_shift + 1)):
                segment_borders2 = \
                           self.segment_border_indices[segment_index_channel2]
                # Iterate over all channel pairs
                for i, channel1_name in enumerate(x.channel_names):
                    channel1_index = x.channel_names.index(channel1_name)
                    for channel2_name in x.channel_names[i+1:]:
                        channel2_index = x.channel_names.index(channel2_name)
                        
                        # Get segments whose correlation should be computed
                        segment1 = data[segment_borders1[0]:segment_borders1[1], 
                                     channel1_index]
                        segment2 = data[segment_borders2[0]:segment_borders2[1], 
                                     channel2_index]
                        
                        # Bring segments to the same shape
                        if segment1.shape[0] != segment2.shape[0]:
                            min_shape = min(segment1.shape[0],
                                                            segment2.shape[0])
                            segment1 = segment1[0:min_shape]
                            segment2 = segment2[0:min_shape]
                        
                        # Compute the pearson correlation of the two segments
                        correlation = scipy.corrcoef(segment1, segment2)[0,1]

                        features.append(correlation)
                        feature_names.append("Correlation_%s_%s_%ssec_%ssec_%s"
                                % (channel1_name, channel2_name,
                                   segment_borders1[0] / x.sampling_frequency,
                                   segment_borders1[1] / x.sampling_frequency,
                                   segment_index_channel2 ))
     
        feature_vector = \
         FeatureVector(numpy.atleast_2d(features).astype(numpy.float64),
         feature_names)
        
        return feature_vector


class ClassAverageCorrelationFeatureNode(BaseNode):
    """ Compute pearson correlation between channel segments and class averages
    
    This node computes for all channels the Pearson product-moment 
    correlation coefficient between certain time segments and the class
    averages. Then each of the correlation coefficients is returned as
    feature.
    
    **Parameters**
    
        :segments:
            The number of segments the time series window is split.
            
            (*optional, default: 1*)
     
    **Exemplary Call**
    
    .. code-block:: yaml
    
        -
            node : Class_Average_Correlation_Features
            parameters :
                segments : 3
    
    :Author: Jan Hendrik Metzen (jhm@informatik.uni-bremen.de)
    :Created: 2009/03/18
    """
    
    def __init__(self, segments=1, *args, **kwargs):
        super(ClassAverageCorrelationFeatureNode, self).__init__(*args,
                                                                     **kwargs)
        
        self.set_permanent_attributes(segments=segments,
                                      segment_border_indices=None,
                                      class_averages=dict(),
                                      class_examples=dict())
    
    def is_trainable(self):
        """ Returns whether this node is trainable """
        return True
    
    def is_supervised(self):
        """ Returns whether this node is trainable """
        return True

    def _execute(self, x):
        
        # Compute the indices of the segment borders lazily when the data is
        # known
        if self.segment_border_indices == None:
            datapoints = x.shape[0]
            borders = [k * datapoints / (self.segments + 1) 
                            for k in range(0, self.segments + 2)]
            self.segment_border_indices = [(borders[i], borders[i + 2])
                                                for i in range(self.segments)]
        
        features = []
        feature_names = []
        # Iterate over all segments:
        for segment_borders in self.segment_border_indices:
            # Iterate over all channels
            for channel_name in x.channel_names:
                channel_index = x.channel_names.index(channel_name)
                
                # Correlation of the channel to the class average
                for label in self.class_averages.keys():
                    channel_seg_avg = self.class_averages[label] \
                        [segment_borders[0]:segment_borders[1], channel_index]
                    sample_seq = \
                       x[segment_borders[0]:segment_borders[1], channel_index]
                    correlation = scipy.corrcoef(channel_seg_avg,
                                 sample_seq) # 0,1 or 1.0 doesn't matter
                    features.append(correlation)
                    feature_names.append("Pearson_%s_Class%s_%ssec_%ssec"
                          % (channel_name,
                             label,
                             segment_borders[0] / x.sampling_frequency,
                             segment_borders[1] / x.sampling_frequency))

#                    if segment_borders[0] == 14 and row == 0:
#                        print correlation
#                        import pylab
#                        pylab.plot(avg, label = ("Avg %s" % label))
#                        pylab.plot(x[:,row], label = "Sample")
#                        pylab.legend()
#                        pylab.show()
#                        raw_input()
#                        pylab.gca().clear()
            
        feature_vector = \
         FeatureVector(numpy.atleast_2d(features).astype(numpy.float64),
         feature_names)
        
        return feature_vector
    
    def _train(self, x, label):
        # Accumulate the examples for each class
        if label not in self.class_averages.keys():
            self.class_averages[label] = copy.deepcopy(x)
            self.class_examples[label] = 1
        else:
            self.class_averages[label] += x
            self.class_examples[label] += 1
            
    def _stop_training(self):
        for label in self.class_averages.keys():
#            print "Examples for class %s: %s" 
#                            % (label, self.class_examples[label])
            self.class_averages[label] /= self.class_examples[label]
    
class CoherenceFeatureNode(BaseNode):
    """ Compute pairwise coherence of two channels with *matplotlib.mlab.cohere*
    
    **Parameters**
    
        :frequency_band:
            The frequency band which is used to extract features from the
            spectrogram. If None, the whole frequency band is used.
            
            (*Optional, default: None*)
    
    **Exemplary Call**
    
    .. code-block:: yaml
    
        -
            node : Coherence_Features
            parameters :
                frequency_band : [0.4,3.5]
                
    :Author: Jan Hendrik Metzen (jhm@informatik.uni-bremen.de)
    :Created: 2009/03/18
    """
    
    def __init__(self, frequency_band = None, frequency_resolution = None,
                  *args, **kwargs):
        super(CoherenceFeatureNode, self).__init__(*args, **kwargs)
        
        if frequency_band != None:
            min_frequency, max_frequency = frequency_band
        else:
            min_frequency, max_frequency = (-numpy.inf, numpy.inf) 

        if frequency_resolution is None:
            frequency_resolution = 1
            warnings.warn("No initial frequency resolution set. Default"
                          "value of 1 used!")


        self.set_permanent_attributes(min_frequency = min_frequency,
                                      max_frequency = max_frequency,
                               frequency_resolution = frequency_resolution)

    def _execute(self, x):
        # Lazy computation of NFFT and noverlap
        if not hasattr(self, "NFFT"):
            # Compute NFFT to obtain the desired frequency resolution 
            # (if possible)
            # self.NFFT has to be even
            self.NFFT = int(round(0.5 * x.sampling_frequency / \
                                               self.frequency_resolution) * 2)
            self.noverlap = 0
        
        # For each pair of channels, we compute the STFT
        features = []
        feature_names = []
        for i, channel_name1 in enumerate(x.channel_names):
            for j, channel_name2 in enumerate(x.channel_names[i + 1:]):
                (Cxy, freqs) = mlab.cohere(x[:, i], x[:, i + 1 + j],
                                           Fs = x.sampling_frequency,
                                           NFFT = self.NFFT,
                                           noverlap = self.noverlap)
                
                # TODO: This would be more efficient without the explicit loop
                for index1, freq in enumerate(freqs):
                    if not (self.min_frequency <= freq <= self.max_frequency):
                        continue
                    # Append as feature
                    features.append(Cxy[index1])
                    feature_names.append("Coherence_%s_%s_%.2fHz" % 
                                                (channel_name1, channel_name2,
                                                 freq))
        
        feature_vector = \
            FeatureVector(numpy.atleast_2d(features).astype(numpy.float64),
                          feature_names)
        
        return feature_vector
    
#    #UNCOMMENT FOR GRAPHICAL ANALYZATION
#    def _train(self, x, label):
#        #if label != "Target": return
#        print label
#        
#        channel_name1 = 'CP2'
#        channel_index1 = x.channel_names.index(channel_name1)
#        channel_name2 = 'Pz'
#        channel_index2 = x.channel_names.index(channel_name2)
#
#        import pylab        
#        Cxy, Phase, freqs =  pylab.mlab.cohere_pairs(x, [(channel_index1, channel_index2)],
#                                                     Fs = x.sampling_frequency,
#                                                     NFFT = self.NFFT,
#                                                     noverlap = self.noverlap)
#
#        pylab.subplot(411)
#        pylab.plot(freqs, Cxy[(channel_index1, channel_index2)])
#        pylab.subplot(412)
#        pylab.plot(freqs, Phase[(channel_index1, channel_index2)])
#
#
#        pylab.subplot(421)
#        pylab.plot(x[:, channel_index1])
#        
#        pylab.subplot(422)
#        pylab.plot(x[:, channel_index2])
#        
#        pylab.show()
#        raw_input()
#        pylab.gca().clear()

_NODE_MAPPING = {"Pearson_Correlation_Features": PearsonCorrelationFeatureNode,
                "Class_Average_Correlation_Features": ClassAverageCorrelationFeatureNode,
                "Coherence_Features": CoherenceFeatureNode, 
                "Statistical_Features" : StatisticalFeatureNode}

