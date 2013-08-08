""" Extract features in time domain like simple amplitudes, signal differentiation or polynomial fit """

import numpy
import warnings

from pySPACE.missions.nodes.base_node import BaseNode
from pySPACE.resources.data_types.feature_vector import FeatureVector

class TimeDomainFeaturesNode(BaseNode):
    """ Use the samples of the time series as features
    
    This node uses the values of the channels at certain points of time
    within the window directly as features.
    
    **Parameters**
    
        :datapoints:
            The indices of the data points that are used as features. If None,
            all data points are used.
            
            (*optional, default: None*)

        :absolute:
            If True, the absolute value of each amplitude value is used.
            Recommended for classification using only the EMG signal.
            
            (*optional, default: False*)

    **Exemplary Call**
    
    .. code-block:: yaml
    
        -
            node : Time_Domain_Features
            parameters :
                datapoints : [-4,-3,-2,-1] # means last 4 samples
    
    :Author: Jan Hendrik Metzen (jhm@informatik.uni-bremen.de)
    :Created: 2008/08/26
    :Revised: 2009/07/16
    """
    def __init__(self, 
                 datapoints = None,
                 absolute = False,
                 **kwargs):
        super(TimeDomainFeaturesNode, self).__init__(**kwargs)
        
        self.set_permanent_attributes(datapoints=datapoints,
                                      absolute=absolute,
                                      feature_names=[])

    def _execute(self, x):
        """ Extract the TD features from the given data x """
        y = x.view(numpy.ndarray)
        if self.datapoints is None or self.datapoints == 0:
            self.datapoints = range(y.shape[0])
        # Mapping from data point index to relative time to onset
        def indexToTime(index):
            if index >= 0:
                return float(index) / x.sampling_frequency
            else:
                return (x.end_time - x.start_time)/ 1000.0 \
                            + float(index) / x.sampling_frequency 
        # We project onto the data points that should be used as features
        y = y[self.datapoints,:]
        if self.absolute:
            y = numpy.fabs(y)
        y = y.T
        # Use all remaining values as features
        features = y.reshape((1, y.shape[0] * y.shape[1]))
        # If not already done, we determine the name of the features
        if self.feature_names == []:
            for channel_name in x.channel_names:
                for index in self.datapoints:
                    self.feature_names.append("TD_%s_%.3fsec" % 
                                                    (channel_name,
                                                     indexToTime(index)))
        # Create and return the feature vector
        feature_vector = \
            FeatureVector(numpy.atleast_2d(features).astype(numpy.float64),
                          self.feature_names)
        return feature_vector

class CustomChannelWiseFeatureNode(TimeDomainFeaturesNode):
    """ Use the result of a transformation of the time series as features.
    
    This node applies a given transformation to the data of each individual
    data channel. The result is consequently used as features.
    
    
    **Parameters**
    
        :feature_function:
            A string that defines the transformation of the univariate time
            series. This string will be evaluated in a
            ``eval('lambda x:' + feature_function)``
            statement. Therefore, ``'x'`` has to be used as placeholder for the
            input data. The output has to be array-like, dimensions do not
            matter. Note that numpy can directly be used. To use other external
            libraries, use, e.g., the following syntax:
            ``"__import__('statsmodels.tsa.api').tsa.api.AR(x).fit(maxlag=3).params[1:]"``
    
    
    **Note**
    
        The *datapoints* parameter provided by the TimeDomainFeaturesNode can
        also be used here. The *absolute* parameter, however, is not supported.
        If the absolute value shall be computed, this can be done in the
        *feature_function*.


    **Exemplary Call**
    
    .. code-block:: yaml
    
        -
            node : Custom_Features
            parameters :
                feature_function : "numpy.dot(x,x)"
    
    :Author: David Feess (david.feess@dfki.de)
    :Created: 2012/07/29
    """
    def __init__(self, feature_function, **kwargs):
        super(CustomChannelWiseFeatureNode, self).__init__(**kwargs)
        
        if 'absolute' in kwargs.keys():
            warnings.warn("Custom_Features does not support 'absolute'!")
        
        ### Place to define abbreviations. The feature_function strings are
        ### matched with some patterns (using regexp) and then replaced by the
        ### actual string that describes the functions. Add your stuff here!
        import re
        # check if an abbreviation is used:
        # AR[p]:
        if re.match("AR\[([1-9])+\]",feature_function)!=None:
            p = feature_function.strip('AR[]')
            feature_function = "__import__('statsmodels.tsa.api').tsa.api.AR(x).fit(maxlag="+p+").params[1:]"        
        # ARMA[p,q]: (note that ARMA is extremely slow)
        if re.match("ARMA\[([1-9])+,([1-9])+\]",feature_function)!=None:
            (p,q) = feature_function.strip('ARMA[]').split(',')
            feature_function = "__import__('statsmodels.tsa.api').tsa.api.ARMA(x).fit(order=("+p+","+q+")).params[1:]"
        
        self.set_permanent_attributes(feature_function = feature_function, #str
                                      feat_func = None) # the actual function
        
    def _execute(self, x):
        """ Extract the TD features from the given data x """
        y=x.view(numpy.ndarray)
        if self.datapoints == None or self.datapoints == 0:
            self.datapoints = range(y.shape[0])
        # We project onto the data points that should be used as features
        y = y[self.datapoints,:]
        
        # generate feat_func from string representation if not done yet
        if self.feat_func == None:
            self.feat_func = eval("lambda x: numpy.atleast_1d(" + 
                                                self.feature_function + 
                                                                ").flatten()")
        
        # initialize 2D array for transformation results
        nr_feats_per_channel = len(self.feat_func(y[:,0]))
        res = numpy.zeros((nr_feats_per_channel,y.shape[1]))
        
        # eval fet_func for each channel
        for curr_chan in range(y.shape[1]):
            try:
                res[:,curr_chan] = self.feat_func(y[:,curr_chan])
            except: # pass zeros as features
                res[:,curr_chan] = numpy.zeros_like(res[:,curr_chan])
                warnings.warn("Feature Function failed or delivered wrong " +
                              "dimensions for channel %s in window: %s. " 
                               % (x.channel_names[curr_chan],x.tag) + 
                               "Wrote zeros in the feature vector instead.")
                
        
        # flatten, such that feats from one channel stay grouped together
        features = res.flatten('F') 
        
        # Feature names
        if self.feature_names == []:
            for channel_name in x.channel_names:
                for i in range(nr_feats_per_channel):
                    self.feature_names.append("CustomFeature1_%s_%d" % 
                                                            (channel_name, i))
        
        # Create and return the feature vector
        feature_vector = \
            FeatureVector(numpy.atleast_2d(features).astype(numpy.float64),
                          self.feature_names)
        return feature_vector
        
    
# TODO: These two nodes need memory optimization ...
class TimeDomainDifferenceFeatureNode(BaseNode):
    """ Use differences between channels and/or times as features.
    
    This node uses differences between channels (Inter_Channel) at the same
    time and between different times (Intra_Channel) on the same channel as
    features.
    
    **Parameters**
    
        :datapoints:
            The indices of the data points that are used. If None, all 
            data points are used.
            
            (*Optional, default: None*)
        
        :moving_window_length:
            If this parameter is greater than one, then not the data point x[i]
            is used but the average of the k=*moving_window_length* elements
            around x[i], i.e. avg([x[i-k/2],...,x[i+k/2]).
            
            (*Optional, default: 1*)
            
    **Known issues** 
        In the current version this produces to much data, even just for one
        choice.
    
    **Exemplary Call**
    
    .. code-block:: yaml
    
        -
            node : Time_Domain_Difference_Features
            parameters :
                datapoints : None
                moving_window_length : 1
                
    :Author: Jan Hendrik Metzen (jhm@informatik.uni-bremen.de)
    :Created: 2008/08/26
    :Revised: 2009/07/16
    """
    def __init__(self, 
                datapoints = None,
                moving_window_length = 1, 
                **kwargs):
        super(TimeDomainDifferenceFeatureNode, self).__init__(**kwargs)
       
        self.set_permanent_attributes(datapoints = datapoints,
                            moving_window_length = moving_window_length,
                                   feature_names = []) #bug: should be feature_names

    def _execute(self, x):
        """
        Extract the TD features from the given data x
        """
        #TODO: Shorten maybe this code
        if self.datapoints == None:
            self.datapoints = range(x.shape[0]) 
        y=x.view(numpy.ndarray)
        #From each selected channel we extract the specified datapoints
        indices = []
        for datapoint in self.datapoints:
            indices.append(range(max(0, datapoint - \
               self.moving_window_length / 2), min(x.shape[0], datapoint + \
               (self.moving_window_length + 1) / 2)))
       
        channel_features = dict()
        for channel_name in x.channel_names:
            channel_index = x.channel_names.index(channel_name)
            for number, index_range in enumerate(indices):
                channel_features[(channel_name, number)] = \
                                  numpy.mean(y[index_range, channel_index])
       
        # Mapping from datapoint index to relative time to onset
        def indexToTime(index):
            if index >= 0:
                return float(index) / x.sampling_frequency
            else:
                return (x.end_time - x.start_time)/ 1000.0 \
                           + float(index) / x.sampling_frequency

        features = []
        for channel1, number1 in  channel_features.iterkeys():
            for channel2, number2 in  channel_features.iterkeys():
                if channel1 == channel2 and number1 > number2:
                    features.append(channel_features[(channel1, number1)] - \
                                        channel_features[(channel2, number2)])
                elif number1 == number2 and channel1 != channel2:
                    features.append(channel_features[(channel1, number1)] - \
                                        channel_features[(channel2, number2)])
                   
        if self.feature_names == []:
            for channel1, number1 in  channel_features.iterkeys():
                for channel2, number2 in  channel_features.iterkeys():
                    if channel1 == channel2 and number1 > number2:
                        self.feature_names.append( \
                            "TDIntraChannel_%s_%.3fsec_%.3fsec" % (channel1,
                                  indexToTime(number1), indexToTime(number2)))
                    elif number1 == number2 and channel1 != channel2:
                        self.feature_names.append( \
                            "TDInterChannel_%s-%s_%.3fsec" % (channel1,
                                              channel2, indexToTime(number1)))
    
        feature_vector = \
         FeatureVector(numpy.atleast_2d(features).astype(numpy.float64),
         self.feature_names)
       
        return feature_vector

class SimpleDifferentiationFeatureNode(BaseNode):
    """ Use differences between successive times on the same channel.
    
    This node uses differences between successive times on the same channel
    of the time series as features to simulate differentiation.
    
    **Parameters**
    
        :datapoints:
            The indices of the data points that are used. If None, all 
            data points are used.
            
            (*Optional, default: None*)
        
        :moving_window_length:
            If this parameter is greater than one, then not the data point x[i]
            is used but the average of the k=*moving_window_length* elements
            around x[i], i.e. avg([x[i-k/2],...,x[i+k/2]).
            
            (*Optional, default: 1*)
            
    **Known Issues**
        .. todo:: new node with same result as a new time series
    
    **Exemplary Call**
    
    .. code-block:: yaml
    
        -
            node : Derivative_Features
            parameters :
                datapoints : None
                moving_window_length = 1
    
    :Author: Mario Krell (Mario.Krell@dfki.de)
    """
    # Simple copy of the TimeDifferenceFeatureNode with slight change at the
    # end. More complex would be a short time regression between more than two
    # points. Yet the possiblity to make it a "real" differentiation is in 
    # comment.
    def __init__(self, 
                datapoints = None,
                moving_window_length = 1, 
                **kwargs):
        super(SimpleDifferentiationFeatureNode, self).__init__(**kwargs)
        self.set_permanent_attributes(datapoints = datapoints,
                            moving_window_length = moving_window_length) 

    def _execute(self, x):
        """
        Extract the TD features from the given data x
        """
        if self.datapoints == None:
            self.datapoints = range(x.shape[0])
        y=x.view(numpy.ndarray)
        # From each selected channel we extract the specified data points
        indices = []
        for datapoint in self.datapoints:
            indices.append(range(max(0, datapoint - \
                self.moving_window_length / 2), min(x.shape[0], datapoint + \
                (self.moving_window_length + 1) / 2)))
       
        channel_features = dict()
        for channel_name in x.channel_names:
            channel_index = x.channel_names.index(channel_name)
            for number, index_range in enumerate(indices):
                channel_features[(channel_name, number)] = \
                                  numpy.mean(y[index_range, channel_index])
       
        # Mapping from datapoint index to relative time to onset
        def indexToTime(index):
            if index >= 0:
                return float(index) / x.sampling_frequency
            else:
                return (x.end_time - x.start_time)/ 1000.0 \
                                         + float(index) / x.sampling_frequency

        features = []
        feature_names = []
        for channel1, number1 in  channel_features.iterkeys():
            # intuitive derivative quotient
            number2 = number1 + 1
            if (channel1, number2) in  channel_features.iterkeys():
                features.append(channel_features[(channel1, number2)] - \
                    channel_features[(channel1, number1)])#*sampling_frequency
                feature_names.append("Df2_%s_%.3fsec" % 
                                             (channel1, indexToTime(number1)))
            # Method taken frome http://www.holoborodko.com/pavel/?page_id=245
            # f'(x)=\\frac{2(f(x+h)-f(x-h))-(f(x+2h)-f(x-2h))}{8h}
            # Further smoothing functions are available, but seemingly not 
            # necessary, because we have already a smoothing of the signal
            # when doing the subsampling.
            number3 = number1 + 4
            number = number1 + 2
            if (channel1, number3) in  channel_features.iterkeys():
                features.append(2.0 * (channel_features[(channel1, number+1)]\
                 - channel_features[(channel1, number-1)]) - \
                 (channel_features[(channel1, number+2)] - channel_features[(\
                 channel1, number-2)]))#*8*sampling_frequency
                feature_names.append("Df5_%s_%.3fsec" % 
                                              (channel1, indexToTime(number)))
        feature_vector = \
         FeatureVector(numpy.atleast_2d(features).astype(numpy.float64),
         feature_names)
        features = []
        feature_names = []
        channel_features = dict()
        return feature_vector

class LocalStraightLineFeatureNode(BaseNode):
    """ Fit straight lines to channel segments and uses coefficients as features.
    
    Fit first order polynomials (straight lines) to subsegments of the 
    channels and use the learned coefficients as features.
    
    **Parameters**
    
        :segment_width:
            The width of the segments (in milliseconds) to which straight lines 
            are fitted. 
            
            .. note:: segment_width is rounded such that it becomes a multiple
                      of the sampling interval.

        :stepsize:
            The time (in milliseconds) the segments are shifted. Extracted 
            segments are [0, segment_width], [stepsize, segment_width+stepsize],
            [2*stepsize, segment_width+2*stepsize], ...
             
            .. note:: stepsize is rounded such that it becomes a multiple
                      of the sampling interval.
           
        :coefficients_used:
            List of the coefficients of the straight line that are actually used 
            as features. The offset of the straight line is the coefficient 0 and
            the slope is coefficient 1. Per default, both are used as features. 

            (*optional, default: [0, 1]*)
           
         
    **Exemplary Call**
    
    .. code-block:: yaml
    
        -
            node : Local_Straightline_Features
            parameters :
                  segment_width : 400
                  stepsize : 200
    
    :Author: Jan Hendrik Metzen (jhm@informatik.uni-bremen.de)
    :Created: 2011/01/04
    :Refactored: 2012/01/18
    """
    
    def __init__(self, segment_width, stepsize, coefficients_used=[0, 1],
                 *args, **kwargs):
        super(LocalStraightLineFeatureNode, self).__init__(*args, **kwargs)
        
        assert len(coefficients_used) <= 2 and \
                    len(set(coefficients_used).difference([0,1])) == 0, \
                        "Only the coefficients 0 (offset) and 1 (slope) are " \
                        "supported!"
        
        # Externally, coefficient 0 denotes the offset and coefficient 1
        # denotes the slope. polyfit handles this the other way around.
        coefficients_used = 1 - numpy.array(coefficients_used)
        
        self.set_permanent_attributes(segment_width = segment_width,
                                      stepsize = stepsize,
                                      coefficients_used = coefficients_used)

    def _execute(self, data):
        # Convert window_width and step size from milliseconds to data points
        segment_width = self.segment_width / 1000.0 * data.sampling_frequency
        segment_width = int(round(segment_width))
        
        stepsize = self.stepsize / 1000.0 * data.sampling_frequency
        stepsize = int(round(stepsize))
        
        sample_width = int(1000 / data.sampling_frequency)
        
        # The subwindows of the time series to which a straight line is fitted
        num_windows = \
            data.shape[1] * ((data.shape[0] - segment_width) / stepsize + 1)
        windows = numpy.zeros((segment_width, num_windows))
        
        feature_names = []
        counter = 0
        data_array = data.view(numpy.ndarray)
        for channel_index, channel_name in enumerate(data.channel_names):
            start = 0 # Start of segment (index)
            while start + segment_width <= data.shape[0]:
                # Compute and round start and end of segment
                end = start + segment_width
                
                # calculate sub-windows
                windows[:, counter] = \
                                    data_array[start:end, channel_index]
                
                #coefficients_used is inverted (see __init__)
                #feature name consists of start and end time
                if 0 in self.coefficients_used:
                    feature_names.append("LSFSlope_%s_%.3fsec_%.3fsec" \
                                            % (channel_name, 
                                               float(start * sample_width)/1000.0,
                                               float(end * sample_width)/1000.0))
                if 1 in self.coefficients_used:
                    feature_names.append("LSFOffset_%s_%.3fsec_%.3fsec" \
                                            % (channel_name, 
                                               float(start * sample_width)/1000.0,
                                               float(end * sample_width)/1000.0))

                # Move to next segment
                start = start + stepsize
                counter += 1
        assert counter == windows.shape[1]
        
        # Compute the local straight line features
        coeffs = numpy.polyfit(range(windows.shape[0]), windows, 1)
        coeffs =  coeffs[self.coefficients_used].flatten('F') 

        feature_vector = \
            FeatureVector(numpy.atleast_2d(coeffs).astype(numpy.float64),
                          feature_names)
        
        return feature_vector

_NODE_MAPPING = {"Time_Domain_Features": TimeDomainFeaturesNode,
                "TDF": TimeDomainFeaturesNode,
                "Time_Domain_Difference_Features": TimeDomainDifferenceFeatureNode,
                "Derivative_Features": SimpleDifferentiationFeatureNode,
                "Local_Straightline_Features" : LocalStraightLineFeatureNode,
                "Custom_Features": CustomChannelWiseFeatureNode}
