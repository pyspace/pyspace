""" Setting adaptive threshold"""

import numpy
import math
import warnings

from pySPACE.missions.nodes.base_node import BaseNode

from pySPACE.resources.data_types.time_series import TimeSeries

from pySPACE.resources.data_types.prediction_vector import PredictionVector



class AdaptiveThresholdPreprocessingNode(BaseNode):
    """ Setting adaptive threshold as described by Semmaoui, H., etal. (2012)
    
    This node can be used to threshold a continuous signal with a adaptive threshold.
    The advantage over a simple fixed threshold method is the adaption to the signal.
    For example if a sensor value drifts over time either in positive or negative 
    direction, a fixed threshold method can have big problems with this one. For 
    a negative drift the "zero" value may get so low that the fixed threshold is 
    never reached again, the other way round a positive drift can lead to a continuous 
    overcoming of the fixed threshold. The adaptive threshold is based on the following
    publication:
    
    Semmaoui, H., etal. (2012).
    Setting adaptive spike detection threshold for smoothed TEO based on robust statistics theory.
    IEEE Transactions on Biomedical Engineering, 59(2):474 - 482.
    (http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=06070974)

    The formula is given as:
    
    .. math:: T(t) = mean(t)_N + p*std(t)_N, 
    
    where T(t) is the threshold at a given timepoint t, mean(t)_N is the mean at 
    timepoint t calculated over the last N samples p is the sensitivity factor and 
    std(t)_N is the standard deviation at timepoint t calculated over the last N 
    samples.
    
    
    The processing is split into two parts, this node implements the first part 
    which does the actual thresholding and safes a timeseries containing zeros, 
    accept at those timepoints where the signal exceeded the threshold. NOTICE 
    only the very first timepoint where the signal overcame the threshold is 
    marked with the value 1 all other values remain a zero. In a second step 
    see "AdaptiveThresholdClassifierNode" below the results are transfered into 
    prediction vectors. This is done since the threshold methods needs the whole 
    data in order to continuously calculate the mean and std. dev., otherwise the 
    first N samples of each window could not be used for analysis. IMPORTANT the 
    preprocessing has to be done without any windowing accept the NULL marker, 
    with a fixed nullmarkerstride.
    
    **Parameters**
    
        :width_adaptiveThreshold:
            Specifies the width of the window used for calculating the mean and 
            the standard deviation for the threshold in ms
            
            (*optional, default:2000*)
        
        :p_adaptiveThreshold:
            Specifies the p for the adaptive threshold
                
            (*optional, default:8*)
                
        :time_below_threshold:
            Specifies how long the signal has to be below the signal before a new 
            thresholding is allowed in ms. This is helpful if only the beginning 
            of some event should be detected in the signal.
            
            (*optional, default:1000*) 
    
    **Exemplary Call**
    
    .. code-block:: yaml
    
        -
            node : AdaptiveThreshold_Preprocessing
            parameters :
                width_adaptive_threshold : 2000
                p_adaptive_threshold : 8
                time_below_threshold : 1000
    
    :Author: Marc Tabie (mtabie@informatik.uni-bremen.de)
    :Created: 2013/01/17
    :Last change: 2013/01/23 by Marc Tabie
    
    """
    input_types=["TimeSeries"]
    def __init__(self, width_adaptive_threshold = 2000, p_adaptive_threshold = 8, time_below_threshold = 1000,  **kwargs):
        super(AdaptiveThresholdPreprocessingNode, self).__init__(**kwargs)
        self.set_permanent_attributes(width_AT = width_adaptive_threshold,                  #Width of the adaptive threshold
                                      ringbuffer_AT = None,                                 #Ringbuffer for storing old data for the adaptive Threshold
                                      p_AT = p_adaptive_threshold,                          #p of the adaptive threshold
                                      variables_AT = [0,0,0,0],                             #Values for calculating the adaptive threshold see function adaptive_threshold()
                                      below_threshold = None,                               #Array which indicates how long each signal was below the threshold
                                      time_below_threshold = time_below_threshold)          #Time in ms where the signal has to below the threshold in order to make a new detection
        
    def is_trainable(self):
        """ Returns whether this node is trainable. """
        return False
    
    def is_supervised(self):
        """ Returns whether this node requires supervised training """
        return False

    def _execute(self, x):
        """ Executes the preprocessing on the given data vector x"""
        #Number of retained channels
        num_channels = numpy.size(x,1)
        if(self.below_threshold == None):
            # When the node is called for the first time initialize all parameters/variables
            self.width_AT = int((self.width_AT*x.sampling_frequency)/1000.)
            
            #Convert the time from ms to samples
            self.time_below_threshold = int((self.time_below_threshold*x.sampling_frequency)/1000.)
            
            #Create and prefill the array which indicates how long a signal was below the threshold
            self.below_threshold = numpy.zeros(num_channels)
            self.below_threshold.fill(self.time_below_threshold+1)
            
            #Create the ringbuffer and the variables list for the adaptive threshold 
            self.ringbuffer_AT=numpy.zeros((self.width_AT,num_channels))
            self.variables_AT=numpy.zeros((4,num_channels))
        data=x.view(numpy.ndarray)
        #Create the array for the thresholded data
        threshold_data = numpy.zeros(data.shape)
        #For each sample of each retained channel
        for i in range(num_channels):
            data_index = 0
            for sample in data[:,i]:
                #calculate the adaptive threshold
                value = self.adaptive_threshold(sample, i)
                #if the actual sample exceeds the threshold...
                if(sample >= value):
                    #and the resting time was observed
                    if(self.below_threshold[i] > self.time_below_threshold):
                        #store a 1 indicating a onset
                        threshold_data[data_index][i] = 1
                    #reset the resting time counter
                    self.below_threshold[i] = 0
                #increase the time the signal was below the signal
                else:
                    self.below_threshold[i] += 1
                data_index += 1

        #return the thresholded data
        result_time_series = TimeSeries.replace_data(x, threshold_data)
        return result_time_series

    def get_output_type(self, input_type, as_string=True):
        return self.string_to_class("TimeSeries")

    def adaptive_threshold(self, data_point, channel_counter):
        """Adaptive threshold for single values
    
        data_point = new datapoint
        channel_counter = index for the retained channels in the ringbuffer
        """
        i=int(self.variables_AT[1][channel_counter])
        n = self.width_AT

        S1 = float(self.variables_AT[2][channel_counter] + (data_point - self.ringbuffer_AT[i][channel_counter])\
         * ((n-1.0) * data_point + (n+1.0) * self.ringbuffer_AT[i][channel_counter] - (2.0 * self.variables_AT[3][channel_counter])))
        self.variables_AT[2][channel_counter] = S1
        self.variables_AT[3][channel_counter] = self.variables_AT[3][channel_counter]+(data_point-self.ringbuffer_AT[i][channel_counter])
        self.variables_AT[0][channel_counter] = self.p_AT*math.sqrt(S1/(n*n)) + (self.variables_AT[3][channel_counter]/n)

        self.ringbuffer_AT[i][channel_counter] = data_point
        i = i+1.0
        if(i>=n):
            i = 0.0;
        self.variables_AT[1][channel_counter] = i

        return self.variables_AT[0][channel_counter]

class AdaptiveThresholdClassifierNode(BaseNode):
    """ Adaptive threshold onset detection classifier
    
    This node parses timeseries generated by the "AdaptiveThresholdPreprocessingNode"
    Basically each data channel of the windows passed to this node are scanned for 
    values equal to 1. If in enough channels specified by num_channels_above_threshold
    the value 1 is found this window is labeled with the positive class otherwise it 
    belongs to the negative class

    **Parameters**
    
        :labels:
            Specifies the names corresponding to the two classes separated
            by the threshold method. NOTICE first give the negative class 
            followed by the positive one
            
            (*optional, default:['noMovement','Movement']*)
            
        :num_channels_above_threshold:
            Specifies how many channels inside a window have to exceed the 
            threshold in order to detect an onset
            
            (*optional, default:1*)


    **Exemplary Call**

    .. code-block:: yaml

        -
            node : AdaptiveThreshold_Classifier

    :Author: Marc Tabie (mtabie@informatik.uni-bremen.de)
    :Created: 2013/01/17
    :Last change: 2013/01/23 by Marc Tabie

    """

    def __init__(self, labels = ['no_movement','movement'], num_channels_above_threshold=1, **kwargs):
        super(AdaptiveThresholdClassifierNode, self).__init__(**kwargs)
        self.set_permanent_attributes(labels = labels,               #Labels for the different classes
                                      num_channels_above=num_channels_above_threshold,test=0)

    def is_trainable(self):
        """ Returns whether this node is trainable. """
        return False

    def is_supervised(self):
        """ Returns whether this node requires supervised training """
        return False

    def _execute(self, x):
        """ Executes the classifier on the given data vector x"""
        num_channels = numpy.size(x,1)
        
        data=x.view(numpy.ndarray)
        if(self.num_channels_above <= 0):
            warnings.warn("num_channels_above_threshold was set to %d. The value has to be greater then zero, now its set to 1" %(self.num_channels_above))
            self.num_channels_above = 1
        elif(self.num_channels_above > num_channels):
            warnings.warn("num_channels_above_threshold was set to %d. But only %d channels are retained, now its set to %d" %(self.num_channels_above,num_channels,num_channels))
            self.num_channels_above = num_channels
        movements_found = numpy.zeros(num_channels)

        #For each sample of each retained channel
        for i in range(num_channels):
            if(numpy.any(data[:,i])):
                movements_found[i] = 1
        # If onsets in enough channels were found label with positive vale else with negative
        label = self.labels[1] if numpy.sum(movements_found) >= self.num_channels_above else self.labels[0]
        return PredictionVector(label=label,
                                prediction=self.labels.index(label),
                                predictor=self)


_NODE_MAPPING = {"AdaptiveThreshold_Preprocessing": AdaptiveThresholdPreprocessingNode,
                 "AdaptiveThreshold_Classifier": AdaptiveThresholdClassifierNode}

