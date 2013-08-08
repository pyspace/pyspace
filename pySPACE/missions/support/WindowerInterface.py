
""" This file defines the template AbstractStreamReader
which is needed for all stream readers (e.g. eegreader)
to work together with the Windower or any classes
derived from the Windower Base-class.
"""

import abc

class AbstractStreamReader():
    """ Property and method definitions of any reader class to be able to interact with the windower. """
    __metaclass__ = abc.ABCMeta
    
    @abc.abstractproperty
    def dSamplingInterval(self):
        """ actually the sampling frequency """
        return 
        
    @abc.abstractproperty
    def stdblocksize(self):
        """ standard block size (int) """
        return
        
    @abc.abstractproperty
    def markerids(self):
        """ mapping of markers/events in stream and unique integer (dict) 
        
        The dict has to contain the mapping 'null' -> 0 to use the
        nullmarkerstride option in the windower. 
        """
        return
        
    @abc.abstractproperty
    def channelNames(self):
        """ list of channel/sensor names """
        return
        
    @abc.abstractproperty
    def markerNames(self):
        """ inverse mapping of markerids (dict) """
        return
        
    @abc.abstractmethod
    def regcallback(self, func):
        """ register a function as consumer of the stream """
        return
        
    @abc.abstractmethod
    def read(self, nblocks):
        """ Read *nblocks* of the stream and pass it to registers functions 
        
        The callback function that is registered by the windower has the
        signature 'func_name(self, ndsamples, ndmarkers)' where ndsamples
        is a numpy 2d-array with shape (number_of_sensors x stdblocksize) and
        ndmarkerks is a numpy ndarray of length stdblocksize filled with the
        unique marker ids (ints) where the events occurred and -1 otherwise.
        The read function has to provide this two arrays and then pass it
        to the callback functions. It should in addition return the number of
        read blocks.
        """
        return
