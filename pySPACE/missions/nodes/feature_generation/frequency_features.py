""" Extract frequency properties like band powers

:Known issues:
    No unit tests!
"""

import numpy
from matplotlib import mlab

from pySPACE.missions.nodes.base_node import BaseNode
from pySPACE.resources.data_types.feature_vector import FeatureVector

class STFTFeaturesNode(BaseNode):
    """ Extract features based on the Short-term Fourier Transform
    
    This node converts the signal into the frequency domain using the
    Short-term Fourier transform. The power of certain cells of the STFT
    (i.e. the power in a frequency band in a certain time interval) are
    returned as features.
    
    **Parameters**
    
        :frequency_band:
            The frequency band which is used to extract features from the
            spectrogram. If this parameter is not specified (None), the
            complete frequency range (-inf,+inf) is used.
            
            (*optional, default: None*)
            
        :frequency_resolution:
            The desired frequency resolution of the spectrogram, i.e. the
            smallest distance between two frequencies being distinguishable.
            Increasing the frequency resolution decreases the time resolution.
            .. note:: *frequency_resolution* << sampling_frequency
    
    **Exemplary Call**
    
    .. code-block:: yaml
    
        -
            node : STFT_Features
            parameters :
                frequency_band : [0.4,3.5]
                frequency_resolution : 1.0
    
    :Author: Jan Hendrik Metzen (jhm@informatik.uni-bremen.de)
    :Created: 2009/03/09
    """
    
    def __init__(self, frequency_band = None, frequency_resolution = None,
                 *args, **kwargs):
        super(STFTFeaturesNode, self).__init__(*args, **kwargs)
        
        if frequency_band != None:
            min_frequency, max_frequency = frequency_band
        else:
            min_frequency, max_frequency = (-numpy.inf, numpy.inf) 
        
        self.set_permanent_attributes(min_frequency = min_frequency,
                                      max_frequency = max_frequency,
                               frequency_resolution = frequency_resolution) 
                               
        assert (frequency_resolution != None), \
                "*frequency_resolution* is a required parameter! It can't " \
                "be None."

    def _execute(self, x):
        """ Extract the Fourier features from the given data x """
        # Lazy computation of NFFT (mlab documentation: The number of data
        # points used in each block for the FFT. Must be even; a power 2 is
        # most efficient. The default value is 256.) and noverlap
        if not hasattr(self, "NFFT"):
            # Compute NFFT to obtain the desired frequency resolution 
            # (if possible)
            # self.NFFT has to be even
            self.NFFT = int(round(0.5 * x.sampling_frequency / \
                                               self.frequency_resolution) * 2)
            self.noverlap = 0 # Number of points of overlap between blocks.
               
        # For each channel, we compute the STFT
        features = []
        feature_names = []
        for channel_name in x.channel_names:
            channel_index = x.channel_names.index(channel_name)
           
            (Pxx, freqs, bins) = mlab.specgram(x[:, channel_index],
                                               Fs = x.sampling_frequency,
                                               NFFT = self.NFFT,
                                               noverlap = self.noverlap)

            # TODO: This would be more efficient without the explicit loop
            for index1, freq in enumerate(freqs):
                if not (self.min_frequency <= freq <=  self.max_frequency):
                    continue
                for index2, bin in enumerate(bins):
                    # Convert to decibels and append as feature
                    features.append(10 * numpy.log10(Pxx[index1, index2]))
                    feature_names.append("STFT_%s_%.2fHz_%.3fsec" % 
                                                (channel_name, freq, bin))
        
        feature_vector = \
         FeatureVector(numpy.atleast_2d(features).astype(numpy.float64),
         feature_names)

        return feature_vector
        
class FrequencyBandFeatureNode(BaseNode):
    """ Extract features based on the Frequency-band power
    
    This node computes the power contained in certain frequency bands. For
    this, the STFT is used to compute a spectrogram and the power of the cells
    within the specified frequency band are summed to obtain the power of the
    frequency band in the respective time bin. Different criteria can be given
    to specify how features are derived from the change of band power over
    time.
    
    **Parameters**
    
        :frequency_bands:
            A list of tuple. The frequency bands whose band power is used as
            feature.
            
        :criterion:
            An criterion that specifies how the change of band power over time
            is used to create a feature: One of the following Strings is
            valid:
            
            * "max" - returns the maximum power in the band over time
            * "min" - returns the minimum power in the band over time
            * "median" - returns the median power in the band over time
            * "max_change" - returns the maximum minus the minimum power in
                             the band over time
            * "all" - returns all powers in the band over time as single
                      features
                      
            (*Optional, default: "max"*)
    
    **Exemplary Call**
    
    .. code-block:: yaml
    
        -
            node : Frequency_Band_Features
            parameters :
                frequency_bands: [(7.0, 10.0), (13.0,26.0)]
                criterion : "all"
                
    :Author: Jan Hendrik Metzen (jhm@informatik.uni-bremen.de)
    :Created: 2009/03/09
    """
    
    def __init__(self, frequency_bands, criterion = "max", *args, **kwargs):
        super(FrequencyBandFeatureNode, self).__init__(*args, **kwargs)
        
        allowed_criteria = ["max", "min", "median", "max_change", "all"]
        if not criterion in allowed_criteria:
            raise Exception("Invalid criterion %s. Please use one of %s" \
                            % (criterion, allowed_criteria))

        self.set_permanent_attributes(frequency_bands = eval(frequency_bands),
                                      criterion = criterion) 

    def _execute(self, x):
        """ Extract the Fourier features from the given data x """
        # Lazily set NFFT and noverlap
        if not hasattr(self, "NFFT"):
            self.NFFT = (x.shape[0] // 4) * 2
            self.noverlap = self.NFFT * 9 // 10
            
        # For each channel, we compute the band power in the specified
        # frequency bands
        features = []
        feature_names = []
        for channel_name in x.channel_names:
            channel_index = x.channel_names.index(channel_name)
            # Compute spectrogram
            (Pxx, freqs, bins) = mlab.specgram(x[:, channel_index],
                                               Fs = x.sampling_frequency,
                                               NFFT = self.NFFT,
                                               noverlap = self.noverlap)
            
            for min_frequency, max_frequency in self.frequency_bands:
                # Just to make sure....
                min_frequency = float(min_frequency)
                max_frequency = float(max_frequency)
    
                # Compute band powers in the specified frequency range for all
                # bins
                selected_freqs = Pxx[numpy.where((freqs >= min_frequency)
                                                & (freqs <= max_frequency))]
                band_powers = numpy.sum(selected_freqs, 0)
                                                          
                # Compute the corresponding feature based on the criterion
                # that was specified
                if self.criterion == "max":
                    channel_features = [10 * numpy.log10(max(band_powers))]
                elif self.criterion == "min":
                    channel_features = [10 * numpy.log10(min(band_powers))]
                elif self.criterion == "median":
                    channel_features = \
                           [10 * numpy.log10(numpy.median(band_powers))]
                elif self.criterion == "max_change":
                    channel_features = [10*numpy.log10((max(band_powers)- \
                                                           min(band_powers)))]
                elif self.criterion == "all":
                    channel_features = 10 * numpy.log10(band_powers)
    
                # Append the feature
                features.extend(channel_features)
                # Determine feature names
                if self.criterion != "all":
                    feature_names.append("BP_%s_%.2fHz_%.2fHz_%s" % 
                                         (channel_name, min_frequency,
                                          max_frequency, self.criterion))
                else:
                    feature_names.extend("BP_%s_%.2fHz_%.2fHz_%s_%.2fsec" % 
                                         (channel_name, min_frequency,
                                          max_frequency, self.criterion,
                                          bin) for bin in bins)
        
        feature_vector = \
         FeatureVector(numpy.atleast_2d(features).astype(numpy.float64),
         feature_names)
        return feature_vector


_NODE_MAPPING = {"STFT_Features": STFTFeaturesNode,
                "Frequency_Band_Features":FrequencyBandFeatureNode}
