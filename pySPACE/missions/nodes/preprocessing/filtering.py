""" Digital filtering of :class:`~pySPACE.resources.data_types.time_series.TimeSeries`"""

import numpy
import math
import scipy.signal
import scipy.fftpack

import warnings

from pySPACE.missions.nodes.base_node import BaseNode
from pySPACE.resources.data_types.time_series import TimeSeries

try:
    import pyublas
except:
    pass

try:
    from pySPACE.missions.support.CPP.variance_tools import variance_tools as vt
except:
    pass


class SimpleLowPassFilterNode(BaseNode):
    """ Low-pass filtering with the given cutoff frequency using SciPy

    This node performs low pass filtering with the given *cutoff_frequency*. It
    uses a FIR filter whose *taps*, *width*, and *window* can be specified.

    .. note:: Deprecated, because functionality is contained in the other nodes,
              with much more important features.
              Use the FIRFilterNode or IIRFilterNode.

    :Author: Jan Hendrik Metzen (jhm@informatik.uni-bremen.de)
    :Revisited: Hendrik Woehrle (hendrik.woehrle@dfki.de)
    :Created: 2008/08/18
    """
    def __init__(self,
                 cutoff_frequency,
                 taps = None,
                 width = None,
                 window = 'hamming',
                 selected_channels= None,
                 **kwargs):
        assert(cutoff_frequency > 0.0)

        super(SimpleLowPassFilterNode, self).__init__(**kwargs)

        self.set_permanent_attributes(cutoff_frequency = cutoff_frequency,
                                      taps = taps,
                                      width = width,
                                      window = window,
                                      selected_channels = selected_channels,
                                                      b = None)

        warnings.warn("Use LowPassFilterNode",DeprecationWarning)

    def _execute(self, x):
        """ Apply low pass filter to data x and return the result """
        #Determine the indices of the channels which will be filtered
        selected_channel_names =  self.selected_channels \
                           if self.selected_channels != None else x.channel_names
        selected_channel_indices = [x.channel_names.index(channel_name) \
                                      for channel_name in selected_channel_names]

        # Compute the FIR window which is required for the low pass filter
        # This is quite slow!
        # filter_order = 2 * x.sampling_frequency / self.cutoff_frequency
        filter_order = 31
        if self.b is None:
            try:
                b = \
                    scipy.signal.firwin(numtaps = filter_order,
                                    cutoff = self.cutoff_frequency * 2.0 / x.sampling_frequency,
                                    width = self.width,
                                    window = self.window)
            except TypeError:
                b = \
                    scipy.signal.firwin(N = filter_order-1,
                                    cutoff = self.cutoff_frequency * 2.0 / x.sampling_frequency,
                                    width = self.width,
                                    window = self.window)
            self.set_permanent_attributes(b=b)
        #Do the actual filtering
        filtered_data = numpy.zeros(x.shape)
        y=x.view(type=numpy.ndarray)
        for channel_index in selected_channel_indices:
            filtered_data[:,channel_index] = scipy.signal.convolve(self.b, \
                              y[:,channel_index])[len(self.b)/2:-len(self.b)/2+1]
        result_time_series = TimeSeries.replace_data(x, filtered_data)

        return result_time_series

class HighPassFilterNode(BaseNode):
   """ High-pass filtering with a FIR filter

   .. todo:: This nodes needs revision concerning computation time and
             correctness.

   **Parameters**

    :cutoff_frequency:
        A frequency in Hz. Frequencies above the cutoff frequency can pass, but
        below are reduced (attenuated).
        Recommended cutoff_frequency for EMG preprocessing: 40 Hz

    :taps:
        Number of taps of the filter kernel. Also called filter order.
        For EMG preprocessing the recommended filter order is 150.

    :width:
        Approximate width of transition region (normalized so that 1
        corresponds to pi) for use in kaiser FIR filter design.

        (*optional, default: None*)

    :window:
        Window function to use. See Scipy documentation http://docs.scipy.org/
        doc/scipy/reference/generated/scipy.signal.firwin.html#scipy.signal.firwin
        for possible windows.

        (*optional, default: ('kaiser', 0.5)*)

    :selected_channels:
        A list of channel names for which the filter should be applied.
        E.g. the names of the EMG channels.

        (*optional, default: None*)

    **Exemplary Call**

    .. code-block:: yaml

        -
            node : High_Pass_Filter
            parameters :
                cutoff_frequency : 40.0
                taps : 150
                selected_channels : ['EMG1',EMG2']


   :Author: Judith Suttrup
   :Created: 2010/02/10
   """
   def __init__(self,
                cutoff_frequency,
                taps,
                width = None,
                window = ("kaiser", 0.5),
                selected_channels= None,
                **kwargs):
       assert(cutoff_frequency > 0.0)

       super(HighPassFilterNode, self).__init__(**kwargs)

       self.set_permanent_attributes(cutoff_frequency = cutoff_frequency,
                                     taps = taps,
                                     width = width,
                                     window = window,
                                     selected_channels = selected_channels,
                                     b=None)

   def _execute(self, x):
       """ Apply high pass filter to data x and return the result """

       #Determine the indices of the channels which will be filtered
       selected_channel_names =  self.selected_channels \
                           if self.selected_channels != None else x.channel_names
       selected_channel_indices = [x.channel_names.index(channel_name) \
                                      for channel_name in selected_channel_names]
       if self.b is None:
       #Compute the FIR window which is required for the high pass filter
            try:
                b = scipy.signal.firwin(numtaps = self.taps,
                                   cutoff = self.cutoff_frequency * 2.0 / x.sampling_frequency,
                                   width = self.width,
                                   window = self.window)
            except TypeError:
                b = scipy.signal.firwin(N = self.taps-1,
                                   cutoff = self.cutoff_frequency * 2.0 / x.sampling_frequency,
                                   width = self.width,
                                   window = self.window)
            b = -b
            b[self.taps/2] = b[self.taps/2]+1
            self.set_permanent_attributes(b=b)

       #Do the actual filtering
       y=x.view(numpy.ndarray)
       filtered_data = numpy.zeros(x.shape)
       for channel_index in selected_channel_indices:
           filtered_data[:,channel_index] = \
                                 scipy.signal.lfilter(self.b, [1], y[:,channel_index])

       result_time_series = TimeSeries.replace_data(x, filtered_data)

       return result_time_series

class FFTBandPassFilterNode(BaseNode):
    """ Band-pass filtering using a Fourier transform

    This node  performs a band-pass filtering for a given *pass_band* by
    converting the signal  into the frequency domain using an FFT, setting
    all bands outside the pass band to zero, and going back to the time domain
    using an IFFT.

    .. note:: Deprecated. Use the FIRFilterNode or IIRFilterNode.

    :Author: Jan Hendrik Metzen (jhm@informatik.uni-bremen.de)
    :Revisited: Hendrik Woehrle (hendrik.woehrle@dfki.de)
    :Created: 2008/08/18
    """
    def __init__(self,
                 pass_band,
                 selected_channels= None,
                 **kwargs):
        super(FFTBandPassFilterNode, self).__init__(**kwargs)

        self.set_permanent_attributes(pass_band = pass_band,
                                      selected_channels = selected_channels)

        warnings.warn("Use BandPassFilterNode",DeprecationWarning)

    def _execute(self, x):
        """ Apply band pass filter to data x and return the result """
        #Determine the indices of the channels which will be filtered
        selected_channel_names = self.selected_channels \
                           if self.selected_channels != None else x.channel_names
        selected_channel_indices = [x.channel_names.index(channel_name) \
                                      for channel_name in selected_channel_names]


        #Do the actual filtering
        x_data=x.view(numpy.ndarray) # More efficient slicing, without memory copy...
        filtered_data = numpy.zeros(x.shape)
        for channel_index in selected_channel_indices:
            #Fourier transform
            fourier_transformed = scipy.fftpack.fft(x_data[:, channel_index])

            #Compute the pass band indices
            lower_bound = int(round(float(self.pass_band[0]) / \
                                x.sampling_frequency * len(fourier_transformed)))
            upper_bound = int(round(float(self.pass_band[1]) / \
                                x.sampling_frequency * len(fourier_transformed)))

            #Setting frequencies outside the pass band to 0
            for i in range(0, lower_bound):
                fourier_transformed[i] = 0
                fourier_transformed[-i-1] = 0
            for i in range(upper_bound,len(fourier_transformed)/2):
                fourier_transformed[i] = 0
                fourier_transformed[-i-1] = 0

            #Inverse Fourier transform and project to real component
            filtered_data[:, channel_index] = \
                                      scipy.fftpack.ifft(fourier_transformed).real

        result_time_series = TimeSeries.replace_data(x, filtered_data)

        return result_time_series


class FIRFilterNode(BaseNode):
    """ Band-pass or low-pass filtering with a time domain convolution based on a FIR filter kernel

    This node performs a finite impulse response filtering for a given
    *pass_band* by applying a time domain convolution with a FIR filter kernel.

    **Parameters**

        :pass_band:
            The pass band. Tuple for band pass, single value for low pass filtering.

        :taps:
            Number of taps of the filter kernel (i.e. order-1)

            (*optional, default: 33*)

        :width:
            Approximate width of transition region (normalized so that 1 corresponds
            to pi) for use in kaiser FIR filter design.

            (*optional, default: None*)

        :window:
            Window function to use. See scipy doc for possible windows.

            (*optional, default: 'hamming'*)

        :time_shift:
            Normally, the convolution is performed as follows:
            If a signal of length N is convolved
            with a signal of length M,
            the result has the length N+M-1.
            Scipy picks the first N values to assure, that the
            resulting signal is of valid length.
            If time_shift is set to 'normal,
            the filter behaves as stated above.
            If time_shift is set to 'middle',
            the values of the interval [M/2,N+M/2-1]
            are picked.
            If time_shift is set to 'end',
            the values of the interval [(N+M-1)/2,N+M-1]
            are picked.
            If time_shift is set to 'stream',
            a block-wise computation is performed, i.e. all incoming time
            series objects are assumed to be adjacent sub-blocks of a larger
            data stream. Therefore, the internal filter state is preserved between
            different executions of the filter.
            (*optional, default: False*)


    **Exemplary Call**

    .. code-block:: yaml

        -
            node : FIRBandPassFilter
            parameters :
                pass_band : [0,4]
                comp_type : "normal"
                window : "hamming"
                taps : 33
                skip : 0


    :Author: Hendrik Woehrle (hendrik.woehrle@dfki.de)
    """
    def __init__(self,
                  pass_band,
                  taps = 33,
                  width = None,
                  window = 'hamming',
                  skip = 0,
                  comp_type = 'normal',
                  time_shift = "middle",
                  selected_channels= None,
                  **kwargs):

        super(FIRFilterNode, self).__init__(**kwargs)

        self.set_permanent_attributes(pass_band = pass_band,
                                      taps = taps,
                                      width = width,
                                      window = window,
                                      selected_channels = selected_channels,
                                      comp_type = comp_type,
                                      skip = skip,
                                      time_shift = time_shift)

        self.comp_hw = kwargs.pop('comp_hw', None)

        self.filter_kernel = None
        self.filtered_data = None
        self.shrinked_data = None
        self.data_buffer = None

        # the internal filter state
        self.internal_state = None

    def initialize_data_dependencies(self, data):
        """ Initialize several data dependent buffer variables
        and data items
        """

        # compute the filter kernel
        self.calc_filter_kernel(data)

        #Determine the indices of the channels which will be filtered
        self.selected_channel_names =  self.selected_channels \
                        if self.selected_channels != None else data.channel_names
        self.selected_channel_indices = [data.channel_names.index(channel_name) \
                                 for channel_name in self.selected_channel_names]

        # buffer for all filter results
        if self.time_shift == "middle":
            self.filtered_data = numpy.zeros((data.shape[0]+ \
                 (len(self.filter_kernel)-1)/2,data.shape[1]), dtype=numpy.float)
            self.shrinked_data = numpy.zeros(data.shape, dtype=numpy.float)
            self.data_buffer = numpy.zeros((data.shape[0]+ \
                 (len(self.filter_kernel)-1)/2,data.shape[1]), dtype=numpy.float)
        elif self.time_shift == "end":
            self.filtered_data = numpy.zeros((data.shape[0]+ \
                 (len(self.filter_kernel)-1),data.shape[1]), dtype=numpy.float)
            self.shrinked_data = numpy.zeros(data.shape, dtype=numpy.float)
            self.data_buffer = numpy.zeros((data.shape[0]+ \
                 (len(self.filter_kernel)-1),data.shape[1]), dtype=numpy.float)
        else:
            self.filtered_data = numpy.zeros(data.shape, dtype=numpy.float)
            self.data_buffer = numpy.zeros(data.shape, dtype=numpy.float)


    def calc_filter_kernel(self, data):
        """ Calculate filter kernel """
        self.sampling_frequency = data.sampling_frequency

        if len(self.pass_band) == 2:
            # highpass with spectral inversion
            try:
                highpass_kernel = - scipy.signal.firwin(numtaps = self.taps,
                         cutoff = float(self.pass_band[0])*2/self.sampling_frequency,
                                                    width = self.width,
                                                   window = self.window)

                # lowpass
                lowpass_kernel = scipy.signal.firwin(numtaps = self.taps,
                         cutoff = float(self.pass_band[1])*2/self.sampling_frequency,
                                                 width = self.width,
                                                window = self.window)
            except TypeError:
                highpass_kernel = - scipy.signal.firwin(N = self.taps-1,
                         cutoff = float(self.pass_band[0])*2/self.sampling_frequency,
                                                    width = self.width,
                                                   window = self.window)

                # lowpass
                lowpass_kernel = scipy.signal.firwin(N = self.taps-1,
                         cutoff = float(self.pass_band[1])*2/self.sampling_frequency,
                                                 width = self.width,
                                                window = self.window)
            lowpass_kernel[self.taps/2] = lowpass_kernel[self.taps/2] + 1

            bandpass_kernel = - (highpass_kernel+lowpass_kernel)
            bandpass_kernel[self.taps/2] = bandpass_kernel[self.taps/2] + 1

            self.filter_kernel = bandpass_kernel

        elif len(self.pass_band) == 1:
            try:
                lowpass_kernel = \
                    scipy.signal.firwin(numtaps = self.taps,
                                        cutoff = float(self.pass_band[0])*2/self.sampling_frequency,
                                        width = self.width,
                                        window = self.window)
            except TypeError:
                lowpass_kernel = \
                    scipy.signal.firwin(N = self.taps-1,
                                        cutoff = float(self.pass_band[0])*2/self.sampling_frequency,
                                        width = self.width,
                                        window = self.window)

            self.filter_kernel = lowpass_kernel

        else:
            raise ValueError("No valid number of pass band arguments: " + \
              "pass_band must be a tuple (band pass) or single value (low pass)")


        # the optional blockwise filtering
        if self.time_shift == "stream":
            self.internal_state = dict()
            for channel_index in xrange(data.shape[1]):
                    self.internal_state[channel_index] = numpy.zeros(len(self.filter_kernel)-1)



    def _execute(self, data):
        """ Apply filter to data and return the result """
        # compute the FIR window which is required for the low pass filter, if
        # it not exists
        if self.filter_kernel is None:
            self.initialize_data_dependencies(data)
            self.time_offset = 0

        assert(len(self.filter_kernel)>0), "Filter construction failed."

        if numpy.dtype('float64') != data.dtype:
            data = data.astype(numpy.float)
        data_array = data.view(numpy.ndarray)
        if self.time_shift == "middle":
            # append zeros to the selected channels and copy them to the data
            # buffer (e.g. just copy the relevant data to data_buffer that
            # contains zeros)
            self.time_offset = (len(self.filter_kernel)-1)/2
            for channel_index in self.selected_channel_indices:
                self.data_buffer[0:data.shape[0],channel_index] = \
                                                    data_array[:,channel_index]
        elif self.time_shift == "end":
            # append zeros to the selected channels and copy them to the data
            # buffer (e.g. just copy the relevant data to data_buffer that
            # contains zeros)
            self.time_offset = (len(self.filter_kernel)-1)
            for channel_index in self.selected_channel_indices:
                self.data_buffer[0:data.shape[0],channel_index] = \
                                                    data_array[:,channel_index]

        else:
            self.data_buffer = data_array

        #Do the actual filtering
        if self.comp_type == 'normal':
            # do sequential filtering
            if self.time_shift == "stream":
                for channel_index in self.selected_channel_indices:
                    (self.filtered_data[:,channel_index],
                     self.internal_state[channel_index]) = \
                        scipy.signal.lfilter(
                            self.filter_kernel, 1,
                            x = self.data_buffer[:,channel_index],
                            zi = self.internal_state[channel_index])
            else:
                for channel_index in self.selected_channel_indices:
                    self.filtered_data[:,channel_index] = \
                        scipy.signal.lfilter(
                             self.filter_kernel, 1,
                             x = self.data_buffer[:,channel_index])

        else:
            raise ValueError("Computation type %s unknown" % self.comp_type)

        if self.time_shift == "middle" \
            or self.time_shift=="end":
            # cut away the irrelevant data
            # (e.g. just copy relevant data back)
            for channel_index in self.selected_channel_indices:
                self.shrinked_data[0:data.shape[0],channel_index] = \
                 self.filtered_data[self.time_offset:
                    self.filtered_data.shape[0]+self.time_offset,
                    channel_index]
            result_time_series = TimeSeries.replace_data(data,self.shrinked_data)
        else:
            result_time_series = TimeSeries.replace_data(data,self.filtered_data)

        return result_time_series

    def __setstate__(self, sdict):
        """ Restore object from its pickled state"""
        super(FIRFilterNode, self).__setstate__(sdict)
        self.filter_kernel = None
        self.filtered_data = None
        self.shrinked_data = None
        self.data_buffer = None
        self.internal_state = None

class IIRFilterNode(BaseNode):
    """ Band-pass or low-pass filtering with a direct form IIR filter

    **Parameters**

        :pass_band:
            The pass band. Tuple for band pass, single value for low pass filtering.

        :pass_band_loss:
            Allowed pass band loss in dB.

            (*optional, default: 0.5*)

        :stop_band_rifle:
            Allowed remaining stop band rifle in dB.

            (*optional, default: 60*)

        :ftype:
            Type of used filter, e.g. elliptic or butterworth.
            See scipy.signal.filter_design.iirdesign for further information.

        :selected_channels:
            selected_channels

        :comp_type:
            Type of computation, e.g. 'normal'

            (*optional, default: 'normal'*)

    **Exemplary Call**

    .. code-block:: yaml

        -
            node : IIRBandPassFilter
            parameters :
                pass_band : [0,4]
                comp_type : "normal"
                window : "hamming"
                taps : 33
                skip : 0

    :Author: Hendrik Woehrle (hendrik.woehrle@dfki.de)
    """
    def __init__(self,
                 pass_band,
                 pass_band_loss = 0.5,
                 stop_band_rifle = 60,
                 ftype = 'ellip',
                 selected_channels = None,
                 comp_type = 'normal',
                 **kwargs):

        super(IIRFilterNode, self).__init__(**kwargs)

        self.set_permanent_attributes(pass_band = pass_band,
                                      pass_band_loss = pass_band_loss,
                                      stop_band_rifle = stop_band_rifle,
                                      ftype = ftype,
                                      selected_channels = selected_channels,
                                      comp_type = comp_type,
                                      filter_kernel = None)


        self.comp_hw = kwargs.pop('comp_hw', None)


    def calc_filter_kernel(self, data):
        """ Calculate filter kernel """
        self.sampling_frequency = data.sampling_frequency
        if len(self.pass_band) == 1:
            wp = self.pass_band[0] * 2. * 0.9 / self.sampling_frequency

            ws = self.pass_band[0] * 2. * 1.0 / self.sampling_frequency

        elif len(self.pass_band) == 2:

            wp = [self.pass_band[0] * 2. * 1.0 / self.sampling_frequency,
                  self.pass_band[1] * 2. * 0.9 / self.sampling_frequency]

            ws = [self.pass_band[0] * 2. * 0.9 / self.sampling_frequency,
                  self.pass_band[1] * 2. * 1.0 / self.sampling_frequency]
        else:
            raise ValueError("No valid number of pass band arguments: pass_band"\
                     + " must be a tuple (band pass) or single value (low pass)")

        b,a = scipy.signal.iirdesign(wp, ws, self.pass_band_loss,
                                    self.stop_band_rifle, ftype=self.ftype)

        self.filter_kernel=[b,a]

    def _execute(self, data):
        """ Apply filter to data and return the result
        .. todo:: check if other view is needed here please
        """
        #Compute the FIR window which is required for the low pass filter, if it
        #not exists
        if self.filter_kernel is None:
            self.calc_filter_kernel(data)
            #Determine the indices of the channels which will be filtered
            self.selected_channel_names =  self.selected_channels \
                        if self.selected_channels != None else data.channel_names
            self.selected_channel_indices = \
                    [data.channel_names.index(channel_name) for channel_name in \
                                                     self.selected_channel_names]

        if self.comp_type == 'normal': #normal filtering with scipy

            filtered_data = numpy.zeros(data.shape)

            for channel_index in self.selected_channel_indices:
                filtered_data[:,channel_index] = \
                                      scipy.signal.lfilter(self.filter_kernel[0],
                                    self.filter_kernel[1], data[:,channel_index])

            result_time_series = TimeSeries.replace_data(data, filtered_data)

        elif self.comp_type == 'mirror':
            #filtering with scipy, mirror the data beforehand on the right border
            data_mirrored = numpy.vstack((data,numpy.flipud(data)))
            pre_filtered_data = numpy.zeros(data_mirrored.shape)
            for channel_index in self.selected_channel_indices:
                pre_filtered_data[:,channel_index] = \
                                      scipy.signal.lfilter(self.filter_kernel[0],
                           self.filter_kernel[1], data_mirrored[:,channel_index])

                pre_filtered_data[:,channel_index] = \
                                      scipy.signal.lfilter(self.filter_kernel[0],
                                                           self.filter_kernel[1],
                                numpy.flipud(pre_filtered_data[:,channel_index]))

            result_time_series = \
                     TimeSeries.replace_data(data, pre_filtered_data[:len(data)])

        else:
            raise ValueError("Computation type unknown")

        return result_time_series

class VarianceFilterNode(BaseNode):
    """ Take the variance as filtered data or standardize with moving variance and mean

    This node can perform a low-pass filtering using the variance,
    for example used to enhance the SNR of raw EMG Signals,
    or calculates a standardization with the variance.

    Filter:
        The variance is calculated for each sample at time point t using the following formula:

        .. math:: n*var(t) = n*var(t-1) + (x(t)-x(t-n)) * ((n-1)*x(t) + (n+1)*x(t-n) - 2*n*m(t-1)),

        where:
            - n is the width of the "filter", number of samples used for calculating the variance
            - Var(t) is the variance as time point t
            - x(t) is the sample at time point t
            - m(t) is the mean at time point t

    Standardization:
        The standardization is calculated for each sample at time point t using the following formula:

        .. math:: S(t) = \\frac{x(t)-m(t)}{Std(t)},

        where:
            - S(t) = is the standardization for the sample at time point t
            - x(t) is the sample at time point t
            - m(t) is the mean at time point t
            - Std(t) is the standard deviation at time point t

        The standard deviation is calculated using the formula for the variance explained above,
        followed by a applying the normal square root

    **Parameters**

        :width:
            Size of the window used to calculate the variance,
            the higher the value the smoother is the resulting signal.
            The value is given in ms.

            (*optional, default: 50*)

        :standardization:
            Flag which indicates if the filter should simply calculate the
            variance for a given size (False), or if a standardization
            should be calculated, meaning the mean for the given size is
            subtracted from the sample followed by a division by the variance.

            (*optional, default: False*)

    **Exemplary Call**

       .. code-block:: yaml

           -
               node : VarianceFilter
               parameters :
                   width : 250
                   standardization : False

    :Author: Marc Tabie (mtabie@informatik.uni-bremen.de)
    :Created: 2012/05/02
    """
    def __init__(self,
                 width = 50,
                 standardization = False,
                 **kwargs):

        super(VarianceFilterNode, self).__init__(ringbuffer = None,variables = None,**kwargs)
        if(width < 1):
            print "Width have to be greater or equal to 1!\nWidth is now set to 1, therefore the data won't be changed!!!"
            width = 1

        #Try to import the c-implementation of the variance and the standardization
        try:
            import pyublas
            var_tools = True
        except:
            warnings.warn("Pyublas is not installed\nIf you are using the varianceFilterNode\nit is going to be very slow...")
            var_tools = False

        if var_tools:
            try:
                from pySPACE.missions.support.CPP.variance_tools import variance_tools as vt
            except:
                warnings.warn("The variance_tools module is not compiled\nIt is located in missions/support/CPP/variance_tools\nPlease compile it using qmake and make")
                var_tools = False

        self.set_permanent_attributes(ringbuffer = None,    # List with ringbuffers for the last n samples used for calculating the variance
                                      variables = None,     # List with the variables needed to calculate
                                      index = None,         # List with the current indices for the ringbuffers
                                      width = width,        # Window size of the variance
                                      standardization = standardization, # Use standardization?
                                      nChannels = None,     # Number of channels
                                      var_tools = var_tools)# C-implementation of var/std

    def _execute(self, data):
        # Initialize the ringbuffers and variables one for each channel
        if(self.ringbuffer == None):
            self.width /= 1000.0
            self.width = int(self.width * data.sampling_frequency)
            self.nChannels = len(data.channel_names)
            self.ringbuffer = numpy.zeros((self.width,self.nChannels),dtype=numpy.double)
            self.variables = numpy.zeros((2,self.nChannels),dtype=numpy.double)
            self.index = numpy.zeros(self.nChannels,'i')

        # Convert the input data to double
        x = data.view(numpy.ndarray).astype(numpy.double)
        # Initialize the result data array
        filtered_data = numpy.zeros(x.shape)
        # Lists which are passed to the standardization
        processing_filtered_data = None
        processing_ringbuffer = None
        processing_variables = None
        processing_index = None
        if(self.standardization):
            for channel_index in range(self.nChannels):
                # Copy the different data to the processing listst
                processing_filtered_data = numpy.array(filtered_data[:,channel_index],'d')
                processing_ringbuffer = numpy.array(self.ringbuffer[:,channel_index],'d')
                processing_variables = numpy.array(self.variables[:,channel_index],'d')
                processing_index = int(self.index[channel_index])
                if self.var_tools:
                    # Perform the standardization
                    # The module vt (variance_tools) is implemented in c using boost to wrap the code in python
                    # The module is located in trunk/library/variance_tools and have to be compiled
                    self.index[channel_index] = vt.standardization(processing_filtered_data, numpy.array(x[:,channel_index],'d'), processing_ringbuffer, processing_variables, self.width, processing_index)
                else:
                    self.index[channel_index] = self.standardisation(processing_filtered_data, numpy.array(x[:,channel_index],'d'), processing_ringbuffer, processing_variables, self.width, processing_index)
                # Copy the processing lists back to the local variables
                filtered_data[:,channel_index] = processing_filtered_data
                self.ringbuffer[:,channel_index] = processing_ringbuffer
        else:
            for channel_index in range(self.nChannels):
                # Copy the different data to the processing listst
                processing_filtered_data = numpy.array(filtered_data[:,channel_index],'d')
                processing_ringbuffer = numpy.array(self.ringbuffer[:,channel_index],'d')
                processing_variables = numpy.array(self.variables[:,channel_index],'d')
                processing_index = int(self.index[channel_index])
                if self.var_tools:
                    # Perform the filtering with the variance
                    # The module vt (variance_tools) is implemented in c using boost to wrap the code in python
                    # The module is located in trunk/library/variance_tools and have to be compiled
                    self.index[channel_index] = vt.filter(processing_filtered_data, numpy.array(x[:,channel_index],'d'), processing_ringbuffer, processing_variables, self.width, processing_index)
                else:
                    self.index[channel_index] = self.variance(processing_filtered_data, numpy.array(x[:,channel_index],'d'), processing_ringbuffer, processing_variables, self.width, processing_index)
                # Copy the processing lists back to the local variables
                filtered_data[:,channel_index] = processing_filtered_data
                self.ringbuffer[:,channel_index] = processing_ringbuffer
                self.variables[:,channel_index] = processing_variables
        # Return the result
        result_time_series = TimeSeries.replace_data(data, filtered_data)
        return result_time_series

    #Fallback functions if the c implementation of the variance filter and the standardisation could not be loaded
    def variance(self, outData, inData, ringbuffer, variables, width, index):
        #Some local variables for speed up
        ww = width*width
        wm1 = width-1.0
        wp1 = width+1.0

        ringbufferValue=0.0
        variable1 = 0.0

        for i in range(len(inData)):
            #Speedup for array entries which are needed several times
            ringbufferValue = ringbuffer[index];
            inDataValue = inData[i]
            variable1 = variables[1];

            #Calculating the new variance
            variables[0] = variables[0] + (inDataValue - ringbufferValue) * ( ((wm1) * inDataValue) + ((wp1) * ringbufferValue) - (2.0*variable1));

            #Calculating the new mean value
            variables[1] = variable1 + (inDataValue-ringbufferValue);

            #Store the actual sample in the ringbuffer
            ringbuffer[index] = inDataValue;

            #Increment the ringbuffer index
            index = index + 1 if (index < wm1) else 0

            #Calculate the standardization
            outData[i] = variables[0]/(ww)

        return index

    def standardisation(self, outData, inData, ringbuffer, variables, width, index):
        #Some local variables for speed up
        ww = width*width
        wm1 = width-1.0
        wp1 = width+1.0

        ringbufferValue=0.0
        variable1 = 0.0

        for i in range(len(inData)):
            #Speedup for array entries which are needed several times
            ringbufferValue = ringbuffer[index];
            inDataValue = inData[i]
            variable1 = variables[1]

            #Calculating the new variance
            variables[0] = variables[0] + (inDataValue - ringbufferValue) * ( ((wm1) * inDataValue) + ((wp1) * ringbufferValue) - (2.0*variable1));

            #Calculating the new mean value
            variables[1] = variable1 + (inDataValue-ringbufferValue);

            #Store the actual sample in the ringbuffer
            ringbuffer[index] = inDataValue;

            #Increment the ringbuffer index
            index = index + 1 if (index < wm1) else 0
            print index
            #Calculate the standardization

            outData[i] = (inDataValue-(variables[1]/(width))) / math.sqrt(variables[0]/(ww)) if (math.sqrt(variables[0]/(ww)) != 0.0) else 0.0

        return index


class TkeoNode(BaseNode):
    """ Calculate the energy of a signal with the Teager Kaiser Energy Operator (TKEO) as new signal

    This is a quadratic filter with the formula:

    .. math::

        x_{i-1}^2 - x_{i-2} \\cdot x_i

    The formula is taken from the following publication::

        Kaiser J. F. (1990)
        On a simple algorithm to calculate 'energy' of a signal.
        In Proceedings:
        International Conference on Acoustics, Speech, and Signal Processing (ICASSP-90)
        Pages 381-384
        (http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=115702)

    For processing EMG-Data a high pass filter at 20 Hz before the node
    and a low pass filter at 50 Hz after the node is recommended.

    **Parameters**

        :selected_channels:
            A list of channel names the algorithm should work with. I.e. the
            EMG channel names. If this parameter is not specified, all
            channels are used.

            (*optional, default: None*)

    **Exemplary Call**

    .. code-block:: yaml

        -
            node : TKEO
            parameters :
                selected_channels : ["EMG1","EMG2"]


    :Author: Marc Tabie (mtabie@informatik.uni-bremen.de)
    :Created: 2012/05/02
    """


    def __init__(self, selected_channels= None, **kwargs):
        super(TkeoNode, self).__init__(**kwargs)

        self.set_permanent_attributes(selected_channels = selected_channels,
                                     old_data = None,
                                     selected_channel_indices = None)

    def _execute(self, x):
        """ Compute the energy of the given signal x using the TKEO """
        #Determine the indices of the channels which will be filtered
        #Done only once...
        if(self.selected_channel_indices == None):
            self.selected_channels = self.selected_channels \
            if self.selected_channels != None else x.channel_names
            self.selected_channel_indices = [x.channel_names.index(channel_name) \
                                            for channel_name in self.selected_channels]
            self.old_data = numpy.zeros((2,len(self.selected_channel_indices)))

        filtered_data = numpy.zeros(x.shape)
        channel_counter = -1
        for channel_index in self.selected_channel_indices:
            channel_counter += 1
            for i in range(len(x)):
                if i==0:
                    filtered_data[i][channel_index] = math.pow(self.old_data[1][channel_counter],2) - (self.old_data[0][channel_counter] * x[0][channel_index])
                elif i==1:
                    filtered_data[i][channel_index] = math.pow(x[0][channel_index],2) - (self.old_data[1][channel_counter] * x[1][channel_index])
                else:
                    filtered_data[i][channel_index] = math.pow(x[i-1][channel_index],2) - (x[i-2][channel_index] * x[i][channel_index])
            self.old_data[0][channel_counter] = x[-2][channel_index]
            self.old_data[1][channel_counter] = x[-1][channel_index]
        result_time_series = TimeSeries.replace_data(x, filtered_data)

        return result_time_series

_NODE_MAPPING = {"Simple_Low_Pass_Filter": SimpleLowPassFilterNode,
                "High_Pass_Filter": HighPassFilterNode,
                "FFT_Band_Pass_Filter": FFTBandPassFilterNode,
                "BandPassFilter": FIRFilterNode,
                "LowPassFilter": FIRFilterNode,
                "FIRBandPassFilter": FIRFilterNode,
                "FIRLowPassFilter": FIRFilterNode,
                "IIRBandPassFilter": IIRFilterNode,
                "IIRLowPassFilter": IIRFilterNode,
                "TKEO": TkeoNode,
                }