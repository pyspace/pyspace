""" Data generation facilities to test algorithms or node chains e.g. in unittests """

import numpy
import pylab
import scipy
import abc


import warnings

from pySPACE.resources.data_types.time_series import TimeSeries


###################################################################

class DataGenerator(object):
    """ Abstract base class for data generation for different test data patterns
    
        To implement an arbitrary data generation class, 
        subclass from this class
        and override the method generate() .
        This can be sine waves, different types of noise, etc.
    """
    def __init__(self,sampling_frequency=1.,
                 *args,**kwargs):
        self.__sampling_frequency = sampling_frequency
        
    def set_sampling_frequency(self,sampling_frequency):
        self.__sampling_frequency = sampling_frequency
        
    def get_sampling_frequency(self):
        return self.__sampling_frequency
    
    sampling_frequency = property(get_sampling_frequency, set_sampling_frequency)

    __metaclass__ = abc.ABCMeta
    
    def __call__(self):
        """ Helper function that returns, how often it was called"""
        try:
            self.__getattribute__("index")
        except AttributeError:
            self.index = 0
        
        temp = self.generate()
        self.index += 1
        return temp
    
    def next_channel(self):
        """ Goes to the next channel"""
        pass
    
    @abc.abstractmethod
    def generate(self):
        pass


# several different test data generation functions
class Zero(DataGenerator):
    """ Helper function for data generation that simply returns zero"""
    
    def __init__(self,*args,**kwargs):
        super(Zero,self).__init__(*args,**kwargs)
    
    def generate(self):
        return 0.0


class One(DataGenerator):
    """ Helper function for data generation that simply returns one"""
    
    def __init__(self,*args,**kwargs):
        super(One,self).__init__(*args,**kwargs)
        
    def generate(self):
        return 1.0


class Constant(DataGenerator):
    """ Helper function for data generation that simply returns one"""
    
    def __init__(self,value,*args,**kwargs):
        self.value = value
        super(Constant,self).__init__(*args,**kwargs)
        
    def generate(self):
        return self.value


class Counter(DataGenerator):
    """ Counts the number of calls and returns the value"""
    
    def __init__(self,start=0,*args,**kwargs):
        self.index = start
        super(Counter,self).__init__(*args,**kwargs)
    
    def generate(self):
        return self.index


class Channel(DataGenerator):
    """ Generated the number of the actual channel"""
    def __init__(self,
                 num_channels,
                 num_time_pts,
                 *args,
                 **kwargs):
        self.num_channels = num_channels
        self.num_time_pts = num_time_pts
        
        super(Channel,self).__init__(*args,**kwargs)
    
    def generate(self):
        return self.index / self.num_time_pts
    
    
class TimePoint(DataGenerator):
    """ Generated the index of the actual time point"""
    
    def __init__(self,
                 num_channels,
                 num_time_pts,
                 *args,
                 **kwargs):
        self.num_channels = num_channels
        self.num_time_pts = num_time_pts
        
        super(TimePoint,self).__init__(*args,**kwargs)
    
    def generate(self):
        return self.index % self.num_channels


class Triangle(DataGenerator):
    """ Generates a triangle with a given width and height
    """
        
    def __init__(self,width,height,*args,**kwargs):
        
        self.width = numpy.double(width)
        self.height = numpy.double(height)
        
        super(Triangle,self).__init__(*args,**kwargs)
        
    def generate(self):
        buffer = numpy.mod(self.index,self.width)
        if buffer <= self.width/2.:
            buffer /= self.width / 2
        else:
            buffer = (self.width - buffer)/(self.width/2)
        return self.height * buffer

    
class GaussianNoise(DataGenerator):
    """ Generates normal distributed noise"""
    
    def __init__(self, mean=0., std=1.,
                 seed = None, *args, **kwargs):
        
        self.mean = numpy.double(mean)
        self.std = numpy.double(std)
        
        if seed != None:
            numpy.random.seed(seed)
            
        super(GaussianNoise,self).__init__(*args,**kwargs)
        
    def generate(self):
        return scipy.randn() * self.std + self.mean
    
    
class Sine(DataGenerator):
    """ Generates a sine wave """
    
    def __init__(self,phase=0.0,frequency=1.,amplitude=1.,sampling_frequency=1.,*args,**kwargs):
        self.phase = phase
        self.frequency = frequency
        self.amplitude = amplitude
        super(Sine,self).__init__(sampling_frequency=sampling_frequency,
                                           *args,**kwargs)
        
    def generate(self):
        t = 2.0 * numpy.pi * self.index * self.frequency / self.sampling_frequency + self.phase
        return self.amplitude * numpy.sin(t)


class ChannelDependentSine(Sine):
    """ Generates a sine wave with channel scaled frequency"""
    
    def __init__(self,*args,**kwargs):
        self.channel_index = 1
        super(ChannelDependentSine, self).__init__(*args,**kwargs)
        
    def next_channel(self):
        """ Goes to the next channel"""
        self.channel_index += 1
        self.frequency = self.channel_index


class Cosine(DataGenerator):
    """ Generates a cosine wave """
    
    def __init__(self,phase=0.0,frequency=1.,amplitude=1.,sampling_frequency=1.,*args,**kwargs):
        self.phase = phase
        self.frequency = frequency
        self.amplitude = amplitude
        super(Cosine).__init__(sampling_frequency=sampling_frequency,
                                      *args,**kwargs)
        
    def generate(self):
        t = 2.0 * numpy.pi * self.index * self.frequency / self.__sampling_frequency + self.phase
        return self.amplitude * numpy.cos(t)


class ChannelDependentCosine(Sine):
    """ Generates a cosine wave with channel scaled frequency"""
    
    def __init__(self,*args,**kwargs):
        self.channel_index = 1
        super(ChannelDependentCosine, self).__init__(*args,**kwargs)
        
    def next_channel(self):
        """ Goes to the next channel"""
        self.channel_index += 1
        self.frequency = self.channel_index


class Delta(Sine):
    """ Generates a delta impulse, i.e. 1 if t==-k, 0 else """
    
    def __init__(self, k=0, *args, **kwargs):
        self.k = k
        super(Delta, self).__init__(*args,**kwargs)
        
    def generate(self):
        if self.index == -self.k:
            return 1
        else:
            return 0


class ChannelDependentDelta(Delta):
    """ Generates a sine wave with channel scaled frequency"""

    def __init__(self,*args,**kwargs):
        self.channel_index = 1
        super(ChannelDependentDelta, self).__init__(k=0, *args, **kwargs)

    def next_channel(self):
        """ Goes to the next channel"""
        self.k -= 2 # to have difference between channels and
        self.index = 0
        self.channel_index += 1

    def generate(self):
        if self.index == -self.k:
            return self.channel_index
        else:
            return 0


class Combiner(DataGenerator):
    """ Combines several generators"""
    def __init__(self,generator_list=[],*args,**kwargs):
        self.generator_list = generator_list
        super(Combiner, self).__init__(*args,**kwargs)
        
    def add_generator(self,generator):
        self.generator_list.append(generator)
        
    def set_sampling_frequency(self,sampling_frequency):
        self.__sampling_frequency = sampling_frequency
        for gen in self.generator_list:
            gen.sampling_frequency = sampling_frequency
        
    def get_sampling_frequency(self):
        return self.__sampling_frequency
    
    sampling_frequency = property(get_sampling_frequency, set_sampling_frequency)


class Adder(Combiner):
    """ Combines several signal by adding them together"""
    
    def __init__(self,generator_list=[],*args,**kwargs):
        super(Adder, self).__init__(generator_list, *args,**kwargs)
                
    def generate(self):
        datum = 0
        for generator in self.generator_list:
            datum += generator()
            
        return datum


class Multiplier(Combiner):
    """ Combines several signal by adding them together"""
    
    def __init__(self,generator_list=[],*args,**kwargs):
        super(Multiplier, self).__init__(generator_list, *args,**kwargs)
                
    def generate(self):
        datum = 1
        for generator in self.generator_list:
            datum *= generator()
            
        return datum


class TestTimeSeriesGenerator(object):
    """ Helper class to generate time series objects e.g. by DataGenerator classes
    
    .. todo:: Documentation is wrong.
    .. todo:: Fix dependencies and function names.
              Why no inheritance from DataGenerator?
              Why use of generate_test_data instead of generate?
    """  
    
    def init(self,**kwargs):
        pass

    def generate_test_data(self,
                           channels=1, 
                           time_points=100, 
                           function=Sine(phase=0.0, frequency=2., amplitude=1.),
                           sampling_frequency=1000,
                           channel_order=True,
                           channel_names=None,
                           dtype=numpy.float):
        """
        A method which generates a signal for testing, with
        the specified number of "channels" which are all generated using
        the given function.
        
        **Keyword arguments**

            :channels: number of channels
            :time_points: number of time points
            :function: the function used for sample generation
            :sampling_frequency: the frequency which is used for sampling,
                                 e.g. the signal corresponds to a time frame of
                                 time_points/sampling frequency
            :channel_names: the names of the channels (alternative to the
                            channels parameter, if not None, it also specifies
                            the number of channels)
            :channel_order: the channel values are computed first, use False
                            for first computation of the row values
            :dtype: data type of the array
        """
       
        if channel_names:
            if len(channel_names) != channels:
                channels = len(channel_names)
                warnings.warn("Ambiguous number of channels in TestTimeSeriesGenerator")
        else:
            channel_names = [("test_channel_%s" % i) for i in range(channels)]
       
        #Generate an empty ndarray
        data = numpy.zeros((time_points, channels),dtype=dtype)

        if channel_order:
            #Compute the values for all channels
            for channel_index in xrange(channels):
                for time_index in xrange(time_points):
                    data[time_index, channel_index] = function()
                function.next_channel()
        else:
            for time_index in xrange(time_points):
                for channel_index in xrange(channels):
                    data[time_index, channel_index] = function()
                
        #Generate a time series build out of the data
        test_data = TimeSeries(input_array = data, 
                               channel_names = channel_names,
                               sampling_frequency = sampling_frequency,
                               start_time = 0,
                               end_time = float(time_points) / sampling_frequency )
        
        return test_data
    
    
    def generate_test_data_simple(self,
                                  channels, 
                                  time_points, 
                                  function, 
                                  sampling_frequency, 
                                  initial_phase = 0.0):
        """
        A method which generates a signal by using function for testing, with
        the specified number of "channels" which are all generated using
        the given function.
        
        **Keyword arguments**

            :channels:  number of channels
            :time_points: number of time points
            :function: the function used for sample generation
            :sampling_frequency: the frequency which is used for sampling,
                                 e.g. the signal corresponds to a time frame
                                 of time_points/sampling frequency
          
        """
       
        #Generate an empty ndarray
        data = numpy.zeros((time_points, channels))
        
        #Compute the values for all channels
        for channel_index in range(channels):
            for time_index in range(time_points):
                data[time_index, channel_index] = function(time_index / sampling_frequency + initial_phase)
                
        #Generate a time series build out of the data
        test_data = TimeSeries(
            input_array=data,
            channel_names=[("test_channel_%s" % i) for i in range(channels)],
            sampling_frequency=sampling_frequency,
            start_time=initial_phase,
            end_time=float(time_points) / sampling_frequency + initial_phase)
        return test_data
    
    def add_to_test_data_single_channel(self, time_series, channel_index, function):
        
        (num_time_points,num_channels) = time_series.shape
        
        sampling_frequency = time_series.sampling_frequency
        
        for time_index in range(num_time_points):
            time_series[time_index, channel_index] = function( time_index/sampling_frequency )

    def add_to_test_data(self, time_series, function):
        """
        Function to add an additional signal generated by function to an existing time series
        
        **Keyword arguments**

            :timeSeries: the time series object
            :function: function to generate signal
        
        """
        (num_time_points,num_channels) = time_series.shape
            
        for channel_index in range(num_channels):
            self.add_to_test_data_single_channel(time_series, channel_index, function)
    
    def generate_normalized_test_data(self,
                                      channels, 
                                      time_points, 
                                      function, 
                                      sampling_frequency, 
                                      initial_phase=0.0):
        """
        A method which generates a normalized (mu = 0, sigma =1) signal for testing, with
        the specified number of "channels" which are all generated using the given function
        """
       
        #Generate an empty ndarray
        data = numpy.zeros((time_points, channels))
        
        #Compute the values for all channels
        for channel_index in range(channels):
            for time_index in range(time_points):
                data[time_index, channel_index] = function(2.0 * numpy.pi * (channel_index + 1) * (time_index / sampling_frequency + initial_phase))
            current_channel = data[:, channel_index]
            current_channel = (current_channel - pylab.mean(current_channel))/pylab.std(current_channel)
            data[:, channel_index] = current_channel
            
        #Generate a time series build out of the data
        test_data = TimeSeries(input_array = data, 
                               channel_names = [("test_channel_%s" % i) for i in range(channels)],
                               sampling_frequency = sampling_frequency,
                               start_time = initial_phase,
                               end_time = float(time_points) / sampling_frequency + initial_phase)
        
        return test_data
