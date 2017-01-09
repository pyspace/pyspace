# This Python file uses the following encoding: utf-8
# The upper line is needed for the copyright symbol in this module.
""" Reduce sampling rate of the :class:`~pySPACE.resources.data_types.time_series.TimeSeries` by a specified fraction

Here different combinations with filters are possible.

.. todo:: Move gcd function to better place.
"""

from scipy.interpolate import interp1d
from pySPACE.missions.nodes.base_node import BaseNode
from pySPACE.missions.nodes.decorators import ChoiceParameter, BooleanParameter
from pySPACE.missions.nodes.preprocessing import filtering
from pySPACE.resources.data_types.time_series import TimeSeries
from pySPACE.tools import prime_factors

from numpy import linspace

import warnings

import logging
import numpy
import scipy
import scipy.signal


def gcd(a, b):
    """ Return the greatest common divisor of a and b.
    
    :Input:
    
        * a -- an integer
        * b -- an integer
        
    :Output:
    
        * an integer -- the gcd of a and b
        
    Examples:
    
    >>> gcd(97,100)
    1
    >>> gcd(97 * 10**15, 19**20 * 97**2)
    97L
    
    © William Stein, 2004
    
    http://modular.math.washington.edu/ent/ent_py
    """
    if a < 0:  a = -a
    if b < 0:  b = -b
    if a == 0: return b
    if b == 0: return a
    while b != 0: 
        (a, b) = (b, a % b)
    return a


class SubsamplingNode(BaseNode):
    """ Downsampling with a simple low pass filter
    
    This is done by upsampling with a linear interpolation and downsampling 
    with a corresponding low pass filter.

    **Exemplary Call**

    .. code-block:: yaml

        -
            node : Subsampling
            parameters :
                target_frequency : 1.0

    :Author: Jan Hendrik Metzen (jhm@informatik.uni-bremen.de)
    :Created: 2008/08/25
    :Revised: 2010/02/25 by Mario Krell
    """
    def __init__(self,
                 target_frequency,
                 **kwargs):
        super(SubsamplingNode, self).__init__(**kwargs)
        
        self.set_permanent_attributes(target_frequency = target_frequency)
                        
        warnings.warn("Use Decimation to reduce the sampling rate",DeprecationWarning)

    def _execute(self, x):
        """ Subsample the given data x and return a new time series """
        # Compute the upsampling and downsampling factor
        source_frequency = int(round(x.sampling_frequency * 100))
        target_frequency = int(round(self.target_frequency * 100))
        reduce_fraction_factor = gcd(source_frequency, target_frequency)
        upsampling_factor = target_frequency / reduce_fraction_factor
        downsampling_factor = source_frequency /reduce_fraction_factor
        # Upsampling
        if upsampling_factor == 1:
            upsampled_time_series = x
        else:
            # We make a linear interpolation for upsampling as an easy version.
            # This should be enough because of the following Lowpassfilter.
            # It is maybe a better version to construction a C^1 spline, 
            # because it prevents artifacts in frequency, but this would  
            # decrease processing speed significantly.
            
            # The new Data Array for the upsampled data
            # The last timepoint does not bring additional points. 
            # This results in the -1+1 calculation.
            self._log("Using upsampling.", level=logging.WARNING)
            
            interpolated_time_series = numpy.zeros((upsampling_factor*(len(x)-1)+1,
                                                       len(x.channel_names)))

            # Array that corresponds to the x value of the fictive function 
            # (original data)
            time = linspace(0, len(x)-1, len(x))
            # Array that corresponds to the x value of the fictive function 
            # (upsampled data)
            newTime = linspace(0, len(x)-1, len(interpolated_time_series))
            # Linear interpolation of each channel
            for channel_name in x.channel_names:
                channel_index = x.channel_names.index(channel_name)
                f = interp1d(time, x[:,channel_index])
                interpolated_time_series[:,channel_index] = f(newTime)
            upsampled_time_series = TimeSeries.replace_data(x,
                                                            interpolated_time_series)
                        
        # Low Pass filtering: According to Shannon-Nyquist's sampling theorem, 
        # we should only retain frequency components below 1/2*target_frequency
        # We multiply with 0.45 instead of 0.5 because of the finite window 
        # length
        lpf_node = filtering.SimpleLowPassFilterNode(cutoff_frequency =  self.target_frequency * 0.45)
        filtered_time_series = lpf_node.execute(upsampled_time_series)
        
        # Downsampling (can be achieved by a simple re-striding)
        # Note: We have to create a new array since otherwise the off-strided 
        #       data remains in memory
        downsampled_data = numpy.array(filtered_time_series[::downsampling_factor, :])
        downsampled_time_series = TimeSeries.replace_data(x, downsampled_data) 
        downsampled_time_series.sampling_frequency = self.target_frequency
        
#        # Uncomment for graphical analyzation
#        import pylab
#        pylab.figure(1)
#        pylab.plot(pylab.linspace(0.0, 1.0, x.shape[0]),
#                   x[:,x.channel_names.index('Pz')],
#                   label = 'Pz' + "_subsampled")
#        pylab.figure(2)
#        pylab.subplot(312)
#        pylab.specgram(x[:,x.channel_names.index('Pz')], Fs = x.sampling_frequency,
#                       NFFT = 64, noverlap = 63)
#        pylab.colorbar()
#        pylab.show()

        return downsampled_time_series


class FFTResamplingNode(BaseNode):
    """ Downsampling with a FFT filter
    
    
    **Exemplary Call**
    
    .. code-block:: yaml
    
        - 
            node : Resampling
            parameters :
                target_frequency : 25.0
                window : "nut" # optional, window to apply before downsampling
                mirror : True #  optional, to make the data periodic before applying the FFT
    
    
    possible windows are:
    'flattop' - 'flat', 'flt' 'boxcar' - 'ones', 'box'
    'triang' - 'traing', 'tri' 'parzen' - 'parz',
    'par' 'bohman' - 'bman', 'bmn' 'blackmanharris' - 'blackharr',
    'bkh', 'nuttall' - 'nutl', 'nut' 'barthann' - 'brthan',
    'bth' 'blackman' - 'black', 'blk' 'hamming' - 'hamm',
    'ham' 'bartlett' - 'bart', 'brt' 'hanning' - 'hann',
    'han' ('kaiser', beta) - 'ksr' ('gaussian', std) - 'gauss',
    'gss' ('general gauss', power, width) - 'general',
    'ggs' ('slepian', width) - 'slep', 'optimal', 'dss'

    :Author: Mario Krell (Mario.Krell@dfki.de)
    :Created: 2010/02/25
    """
    def __init__(self,
                 target_frequency,
                 window=None,
                 mirror =False,
                 **kwargs):
        super(FFTResamplingNode, self).__init__(**kwargs)
        
        self.set_permanent_attributes(target_frequency = target_frequency, 
                                      window = window, new_len = 0, mirror=mirror)

    def _execute(self, data):
        """ Subsample the given data and return a new time series """
        if self.new_len == 0 :
            self.new_len = int(round(self.target_frequency*len(data)/(1.0*data.sampling_frequency)))
        if not self.mirror:
            downsampled_time_series = \
                TimeSeries.replace_data(data, 
                                        scipy.signal.resample(data, self.new_len,
                                                            t=None, axis=0,
                                                            window=self.window))
        else:
            downsampled_time_series = \
                TimeSeries.replace_data(data, 
                                        scipy.signal.resample(numpy.vstack((data,numpy.flipud(data))), self.new_len*2,
                                                            t=None, axis=0,
                                                            window=self.window)[:self.new_len])
        downsampled_time_series.sampling_frequency = self.target_frequency
        return downsampled_time_series


class DownsamplingNode(BaseNode):
    """ Pure downsampling without filter
    
    Reduce sampling rate by picking the values according to the downsampling factor.
    No low pass filter is used as needed for a proper decimation.

    **Exemplary Call**
    
    .. code-block:: yaml
    
        - 
            node : Downsampling
            parameters :
                target_frequency : 2.5
                phase_shift : 1

    .. todo:: Perhaps it is better to not rely on frequencies, but on factors?
    
    :Author: Hendrik Woehrle (Hendrik.Woehrle@dfki.de)
    """
    def __init__(self,
                 target_frequency,
                 **kwargs):
        super(DownsamplingNode, self).__init__(**kwargs)
        
        self.set_permanent_attributes(target_frequency = target_frequency,
                                      downsampling_factor = None)

    def _execute(self, data):
        """ Subsample the given data data and return a new time series """
        # Compute the downsampling factor
        source_frequency = int(round(data.sampling_frequency * 100))
        target_frequency = int(round(self.target_frequency * 100))
        reduce_fraction_factor = gcd(source_frequency, target_frequency)
        new_downsampling_factor = source_frequency /reduce_fraction_factor
        
        upsampling_factor = target_frequency / reduce_fraction_factor
        if not(upsampling_factor == 1):
            self._log("Upsampling used but not implemented!"
                      +" Check target frequency and real frequency!",
                      level=logging.WARNING)
        
        # reset filter if downsampling factor has changed
        if new_downsampling_factor != self.downsampling_factor:
            self.downsampling_factor = new_downsampling_factor

            if self.downsampling_factor <= 1:
                self._log("Inapplicable downsampling factor: "
                          + str(self.downsampling_factor)
                          +" Downsampling will be skipped",
                          level=logging.WARNING)

        # omit downsampling, if inapplicable
        if self.downsampling_factor <= 1:
            return data
                        
        # Downsampling (can be achieved by a simple re-striding)
        # Note: We have to create a new array since otherwise the off-strided 
        #       data remains in memory
        downsampled_data = numpy.array(data[::self.downsampling_factor, :])
        downsampled_time_series = TimeSeries.replace_data(data, downsampled_data) 
        downsampled_time_series.sampling_frequency = self.target_frequency
                
        return downsampled_time_series

@BooleanParameter("Multi_step")
@ChoiceParameter("comp_type", ["normal", "parallel", "gpu"])
class DecimationBase(BaseNode):
    """ Decimate the signal to a given sampling frequency
    
    This class is the base class for the other decimation nodes
    and can not be used directly.

    The decimation is performed by doing a downsampling 
    with a preceding filtering by a corresponding low pass filter beforehand. 
    According to Shannon-Nyquist's sampling theorem, 
    only frequencies below 1/2*target_frequency should be retained.
    
    For the decimation, one should use the FIR decimation (preferred)
    or IIR decimation, see below.
    
    **Parameters**
        :target_frequency:
            The frequency after the decimation.

            (*required*)
            
        :multi_step:
            The decimation is done in multiple sets,
            if the downsampling factor is high.
            The steps are chosen by the node.
            
            (*optional, default: True*)
            
        :comp_type:
            Type of computation for the filtering.
            One of the following Strings: 'normal', 'parallel', 'gpu'....
            
            (*optional, default: 'normal'*)
            
        :norm_cutoff_freq:
            Cutoff frequency for the low pass filter.
            Normalized to the target frequency.
            
             (*optional, default: 'norm_cutoff_freq'*)
            
        :norm_trans_region_width:
            Width of transition region for the low pass filter.
            Normalized to the target frequency.
            
            (*optional, default: 'norm_trans_region_width'*)
            
        :filter_frequency:
            Optional frequency of the filter, if the value
            should not be chosen automatically.

            (*optional, default: None*)
   
    :Author: Hendrik Woehrle (Hendrik.Woehrle@dfki.de)
    """
    def __init__(self,
                 target_frequency,
                 multi_step = True,
                 comp_type = 'normal',
                 norm_cutoff_freq = 1./3.,
                 norm_trans_region_width = 1./6.,
                 filter_frequency = None, 
                 **kwargs):
        super(DecimationBase, self).__init__(**kwargs)
        
        self.set_permanent_attributes(target_frequency = target_frequency,
                                        filter_frequency = filter_frequency,
                                        low_pass_nodes = None,
                                        downsampling_nodes = None,
                                        downsampling_factor = 0,
                                        comp_type = comp_type,
                                        multi_step = multi_step,
                                        multi_step_factors = None,
                                        trans_regions = None,
                                        norm_trans_region_width = norm_trans_region_width,
                                        norm_cutoff_freq = norm_cutoff_freq)
                                        
    def compute_filter_factors(self):
        # compute the optimal downsamplinf factor according to Crochiere and Rabiner, 
        # IEEE Trans. on Acoust. Speech and Signal proc, Vol. 23, N0.5, 1975
        # "Optimum FIR digital filter implementations for decimation,
        # interpolation, and narrow-band filtering"
        # it is asummed that two steps are used
        
        # get filter transition region width 
        f = self.norm_cutoff_freq - self.norm_trans_region_width
        
        # compute theoretically optimal filter
        d = self.downsampling_factor
        d_opt_num = 1. - numpy.sqrt(d*f/(2.-f))
        
        d_opt_denom = 2. - f * (d + 1.)
        
        d_opt = 2 * self.downsampling_factor * d_opt_num / d_opt_denom
        
        # compute final (rounded) downsampling factors
        d_small = prime_factors.next_least_nice_integer_divisor(d, d_opt)
        
        # store multi step factors
        self.multi_step_factors.append(d_small)
        self.multi_step_factors.append(d/d_small)
        
        # use biggest factor first to save computation time
        self.multi_step_factors.sort()
        self.multi_step_factors.reverse()
        
        #compute the individual transition regions
        norm_pass_band = 1 - self.norm_cutoff_freq - self.norm_trans_region_width
        # first (wide) transition region
        # full width between pass band in final filter response to
        # stopband of intermediate filter  
        self.trans_regions.append(self.norm_cutoff_freq - norm_pass_band / self.multi_step_factors[0])
        # second, more steep transition region
        self.trans_regions.append(self.norm_trans_region_width)
        
    def initialize_filters(self, data):
        # if the decimation factor is to big, the decimation is performed in several steps
        self.low_pass_nodes = []
        self.downsampling_nodes = []
        self.trans_regions = []
        self.multi_step_factors = []
        
        if self.downsampling_factor > 10:
            self.compute_filter_factors()
        else:
            self.multi_step_factors.append(self.downsampling_factor)
            self.trans_regions.append(self.norm_trans_region_width)
           
        target_frequency = data.sampling_frequency
                  
        # create all filters
        i = 0
        for dec_factor, trans_reg_width in zip(self.multi_step_factors,self.trans_regions):

            # compute general target frequencies
            target_frequency = target_frequency / dec_factor 
            filter_frequency = target_frequency * self.norm_cutoff_freq
            
            # in the last step, use (eventually) the manually specified frequency
            if (i+1) == len(self.multi_step_factors) and self.filter_frequency is not None:
                filter_frequency = self.filter_frequency
            i = i + 1
            
            self.low_pass_nodes.append(self.create_filter(
                                        target_frequency=filter_frequency,
                                        downsampling_factor=dec_factor,
                                        transition_region_width = trans_reg_width))
        
            self.downsampling_nodes.append(DownsamplingNode(target_frequency = target_frequency))
            
    def create_filter(self, target_frequency, downsampling_factor, transition_region_width):
        return None

    def _execute(self, data):
        """ Subsample the given data data and return a new time series """
        # compute the downsampling factor, if needed (first one or changed)
        
        if data.sampling_frequency != self.downsampling_factor * self.target_frequency:
            self.downsampling_factor = data.sampling_frequency / self.target_frequency
            
            if int(self.downsampling_factor) != self.downsampling_factor:
                self.downsampling_factor = int(self.downsampling_factor)
                self._log("Upsampling not implemented."+
                          " Downsampling factor needs to be an integer.",
                          level=logging.WARNING)
            if self.downsampling_factor <= 1:
                self._log("Inapplicable downsampling factor: "
                          + str(self.downsampling_factor)
                          +" Downsampling will be skipped",
                          level=logging.WARNING)
            else:
                self.initialize_filters(data)
        
        # omit downsampling, if inapplicable
        if self.downsampling_factor <= 1:
            return data
        
        for filter, downsampler in zip(self.low_pass_nodes,self.downsampling_nodes):
            data = filter.execute(data)
        
            data = downsampler.execute(data)
                
        return data
    
    def store_state(self, result_dir, index=None):
        """ Stores this node in the given directory *result_dir* """
        if self.store:
            for index, node in enumerate(self.low_pass_nodes):
                node.store_state(result_dir, index)
            super(DecimationBase, self).store_state(result_dir, index)


class DecimationIIRNode(DecimationBase):
    """ Downsampling with a preceding IIR filter
    
    The decimation is performed by doing a downsampling 
    with a preceding filtering by a corresponding low pass filter beforehand. 
    According to Shannon-Nyquist's sampling theorem, 
    only frequencies below 1/2*target_frequency should be retained.
    
    The decimation is applied in multiple steps, if the sampling factor 
    is too big.
        
    The Filering is done using a IIR filter.

    **Exemplary Call**
    
    .. code-block:: yaml

        -
            node : DecimationIIR
            parameters :
                target_frequency : 25 

    :Author: Hendrik Woehrle (Hendrik.Woehrle@dfki.de)
    """
    
    def __init__(self,
                 **kwargs):
        super(DecimationIIRNode, self).__init__(**kwargs)
    
    
    def create_filter(self,target_frequency,downsampling_factor,transition_region_width):
                
        return filtering.IIRFilterNode([target_frequency], 
                                         comp_type = self.comp_type)


@ChoiceParameter("comp_type", ["normal", "parallel"])
@ChoiceParameter("time_shift", ["normal", "middle", "end", "stream"])
@BooleanParameter("skipping")
class DecimationFIRNode(DecimationBase):
    """ Downsampling with a preceding FIR filter
    
    The decimation is performed by doing a downsampling 
    with a preceding filtering by a corresponding low pass filter beforehand. 
    According to Shannon-Nyquist's sampling theorem, 
    only frequencies below 1/2*target_frequency should be retained.
        
    The decimation is applied in multiple steps, if the sampling factor 
    is too big.
    
    The filtering procedure is applied by an FIR filter, and only for values that are 
    significant due to the downsampling procedure.
    
    **Parameters**
        :comp_type:
            Computation type of the filter, see FIRFilterNode for further information.

            (*optional, default: 'normal'*)

        :skipping:
            If output samples should be skipped in the filtering process,
            because they are discarded in the downsampling process.
            
            (*optional, default: True*)
            
        :time_shift:
            Parameter time_shift of the filter, see FIRFilterNode for further information.

            (*optional, default: 'middle'*)
    
    **Exemplary Call**
    
    .. code-block:: yaml

        -
            node : DecimationFIR
            parameters :
                target_frequency : 25 
       
    :Author: Hendrik Woehrle (Hendrik.Woehrle@dfki.de)
    """
    input_types=["TimeSeries"]
    def __init__(self,
                 comp_type = 'normal',
                 skipping = False,
                 time_shift = "middle",
                 **kwargs):
        super(DecimationFIRNode, self).__init__(comp_type = comp_type,
                                                **kwargs)

        self.set_permanent_attributes(skipping=skipping,
                                      time_shift = time_shift)

    def create_filter(self,target_frequency,downsampling_factor,transition_region_width):
        
        skip = 0
        if self.skipping:
            skip = int(round(downsampling_factor-1))
                
        return filtering.FIRFilterNode([target_frequency], 
                                         comp_type=self.comp_type,
                                         width=transition_region_width,
                                         skip=skip,
                                         time_shift = self.time_shift,
                                         store = self.store)


_NODE_MAPPING = {"Resampling": FFTResamplingNode,
                "Decimation": DecimationFIRNode,
                "DecimationIIR": DecimationIIRNode,
                "Subsampling":SubsamplingNode,
                "Downsampling": DownsamplingNode}

