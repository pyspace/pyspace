""" 2d array of channels x time series for windowed time series """

import numpy
import warnings
from pySPACE.resources.data_types import base


class TimeSeries(base.BaseData):
    """ Time Series object
    
    Represents a finite length time series consisting (potentially) of
    several channels. Objects of this type are called "Windows" in other
    contexts. 

    The time series object is a
    2d array of channels times time series amplitudes
    with some additional properties.
    The additional properties are:

    * channel names
    * sampling frequency
    * start time
    * end time
    * marker name (the name of the marker used to create this object,
                   dictionary of included marker names and time stamps)
    * name & tag (text format of object meta info)

    Channels can also be pseudo channels after spatial filtering.

    :Author: Jan Hendrik Metzen  (jhm@informatik.uni-bremen.de)
    :Created: 2008/03/05
    :Completely Refactored: 2008/08/18
    :BaseData compatibility: David Feess, 2010/09/27
    """
           
    def __new__(subtype, input_array, channel_names, sampling_frequency, 
                start_time=None, end_time=None, name=None,
                marker_name=None, tag=None):
        if type(input_array) == dict:
            data = []
            for channel in channel_names:
                data.append(input_array[channel])
            input_array = numpy.array(data)
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = base.BaseData.__new__(subtype, numpy.atleast_2d(input_array))
        if obj.ndim > 2:
            input_array = obj[0]
            obj = base.BaseData.__new__(subtype, input_array)
            warnings.warn("To many dimensions for Time Series Object!")
        # add subclasses attributes to the created instance
        obj.channel_names = channel_names
        try:
            assert(len(channel_names) == obj.shape[1]),\
                "Channel names (%s) do not match array dimensions (%s)! Fix this!" \
                % (str(channel_names), str(obj.shape))
        except:
            warnings.warn(
                "Array dimensions (%s) do not match channel names (len: %i, names: %s)! Fix this!"
                % (str(obj.shape), len(channel_names), str(channel_names)))
        obj.sampling_frequency = float(sampling_frequency)
        obj.start_time = start_time
        obj.end_time = end_time
        obj.name = name
        obj.marker_name = marker_name
        if not tag is None:
            obj.tag = tag
        
        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        super(TimeSeries, self).__array_finalize__(obj)
        if not obj is None and not type(obj)==numpy.ndarray:
            self.channel_names_hash = getattr(obj, 'channel_names_hash', None)
            self.sampling_frequency = getattr(obj, 'sampling_frequency', None)
            self.start_time = getattr(obj, 'start_time', None)
            self.end_time = getattr(obj, 'end_time', None)
            self.name = getattr(obj, 'name', None)
            self.marker_name = getattr(obj, 'marker_name', None)
        else:
            # TODO: do we need this part or the other one?
            self.channel_names_hash = None
            self.sampling_frequency =  None
            self.start_time = None
            self.end_time = None
            self.name = None
            self.marker_name = None

    def __reduce__(self):
        # Refer to 
        # http://www.mail-archive.com/numpy-discussion@scipy.org/msg02446.html
        # for infos about pickling ndarray subclasses
        object_state = list(super(TimeSeries, self).__reduce__())
        subclass_state = (self.channel_names, self.sampling_frequency,
                          self.start_time, self.end_time, self.name,
                          self.marker_name)
        object_state[2].append(subclass_state)
        object_state[2] = tuple(object_state[2])
        return tuple(object_state)
    
    def __setstate__(self, state):
        if len(state) == 2: # For compatibility with old TS implementation
            nd_state, own_state = state
            numpy.ndarray.__setstate__(self, nd_state)
        else: # len == 3: new BaseData timeseries.

            nd_state, base_state, own_state = state
            super(TimeSeries, self).__setstate__((nd_state, base_state))
        
        (self.channel_names, self.sampling_frequency, self.start_time, 
         self.end_time, self.name, self.marker_name) = own_state

    @staticmethod
    def _generate_tag(obj):
        """generate new tag based on time series attributes start_time,
        end_time and name. The name is usually a sentence, with the last word
        indicating the class. """
        # if no information present: return None
        if getattr(obj, 'name', None) == None and \
           getattr(obj, 'start_time', None) == None and \
           getattr(obj, 'end_time', None) == None:
            return None
        else:
            # If attribute name is provided, the last word should represent class:
            if getattr(obj, 'name', None) == None:
                class_name = 'na'
            else:
                class_name = obj.name.split(' ')[-1]
            
            if getattr(obj, 'start_time', None) == None:
                start = 'na'
            else:
                start = str(int(obj.start_time))

            if getattr(obj, 'end_time', None) == None:
                end = 'na'
            else:
                end = str(int(obj.end_time))
            
            return 'Epoch Start: %sms; End: %sms; Class: %s' % \
                        (start, end, class_name)
        
    # In order to reduce the memory footprint, we do not store the channel
    # names once per instance but only once per occurence. Instead we store a
    # unique hash once per instance  that allows to retrieve the channel names
    channel_names_dict = {}

    def get_channel_names(self):
        return TimeSeries.channel_names_dict[self.channel_names_hash]

    def set_channel_names(self, channel_names):
        self.channel_names_hash = hash(str(channel_names))
        if not TimeSeries.channel_names_dict.has_key(self.channel_names_hash):
            TimeSeries.channel_names_dict[self.channel_names_hash] = channel_names

    def del_channel_names(self):
        pass

    channel_names = property(get_channel_names, set_channel_names,
                             del_channel_names, 
                             "The channel_names property.")

    @staticmethod
    def replace_data(old, data, **kwargs):
        """ Create a new time series with the given data but the old metadata.
        
        A factory method which creates a time series object with the given
        data and the metadata from the old time_series
        """
        data = TimeSeries(data,
                   channel_names=kwargs.get('channel_names',
                                            old.channel_names),
                   sampling_frequency=kwargs.get('sampling_frequency',
                                                 old.sampling_frequency),
                   start_time=kwargs.get('start_time', old.start_time),
                   end_time=kwargs.get('end_time', old.end_time),
                   name=kwargs.get('name', old.name),
                   marker_name=kwargs.get('marker_name', old.marker_name))
        data.inherit_meta_from(old)
        if "tag" in kwargs.keys():
            data.tag=kwargs["tag"]
        return data

    def get_channel(self, channel_name):
        """ Return the values of the channel with name *channel_name* """
        channel_index = self.channel_names.index(channel_name)
        data=self.view(numpy.ndarray)
        return data[:, channel_index]
    
    def reorder(self, ordered_channel_list):
        """ Reorder TimeSeries according to ordered_channel_list
        
        This function takes the list given as argument as list of channel names, 
        orders the given TimeSeries object according to this list and returns a 
        reordered TimeSeries object. 
        """
        
        for elem in ordered_channel_list:
            assert elem in self.channel_names, \
                "TimeSeries:: Reordering impossible. %s is not present in original data!" % elem
            
            current_pos=ordered_channel_list.index(elem)
            
            #if True, the lines are swapped
            if current_pos is not self.channel_names.index(elem):
                old=self[:, current_pos]
                self[:, current_pos]=self[:, self.channel_names.index(elem)]
                self[:, self.channel_names.index(elem)]=old
            
        self.channel_names=ordered_channel_list
    
    def _ms_to_samples(self, ms):
        return ms/1000.0*self.sampling_frequency
        
    def _samples_to_ms(self, samples):
        return samples/float(self.sampling_frequency)*1000

    def __str__(self):
        str_repr =  "TimeSeriesObject \nChannel_names: "
        str_repr+= str(self.channel_names)
        str_repr+= "\n"
#        str_repr+=str(self.view(numpy.ndarray))
        va = self.view(numpy.ndarray)
        for index, channel_name in enumerate(self.channel_names):
            str_repr += "%s : %s \n" % (channel_name, va[:,index])
        str_repr+="\n"
        return str_repr

    def __eq__(self,other):
        """ Same channels (names) and values """
        if not type(self)==type(other):
            return False
        if not set(self.channel_names)==set(other.channel_names):
            return False
        if not self.shape==other.shape:
            return False
        if self.channel_names==other.channel_names:
            if (self.view(numpy.ndarray)-other.view(numpy.ndarray)).any():
                return False
            else:
                return True
        else:
            # Comparison by hand
            for channel in self.channel_names:
                if ((self[self.channel_names.index(channel)]-self[other.channel_names.index(channel)]).any()):
                    return False
            return True
