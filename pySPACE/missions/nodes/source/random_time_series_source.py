""" Generate random data for TimeSeries """

from pySPACE.missions.nodes.base_node import BaseNode
from pySPACE.missions.nodes.source.time_series_source import TimeSeriesSourceNode
from pySPACE.tools.memoize_generator import MemoizeGenerator
from pySPACE.resources.data_types.time_series import TimeSeries

import random
import numpy


class RandomTimeSeriesSourceNode(TimeSeriesSourceNode):
    """ Generate random data and act as a source for windowed TimeSeries
    
    This node acts as a source for windowed TimeSeries. The TimeSeries
    are generated randomly according to the given parameters and
    forwarded.
    
    The time series are generated according to the given generating function,
    and the class label by a uniform distribution according with a given threshold
    Only two classes are supported by now.
    
    **Parameters**

        :num_instances:
            The number of instances to be generated.

            (*optional, default: 20*)

        :generating_function_class_0:
            A function to generate data for class 0.
            Receives an index, which states the 
            number of already generated samples. 
            
            (*optional, default: lambda i: numpy.ones((2,2))*i*)
            
        :generating_function_class_1:
            A function to generate data for class 1.
            Receives an index, which states the 
            number of already generated samples. 
            
            (*optional, default: lambda i: numpy.ones((2,2))*i*)
            
        :channel_names: Channel names of the time series objects.
            
        :class_labels: The class labels of the generated time series.
        
        :choice_threshold:
            The threshold class assignment. The classes are
            generated randomly by generating a random number r
            between 0 and 1. If r < threshold, the class label is
            class_labels[0], and class_labels[1] otherwise.
            
        :sampling_frequency:
            Sampling frequency of the generated time series.
            
        :random:
            If true, the order of the data is randomly shuffled. 

            (*optional, default: True*)
    
    **Exemplary Call**
    
    .. code-block:: yaml
    
        - 
            node : RandomTimeSeriesSource
    
    :Author: Hendrik Woehrle (hendrik.woehrle@dfki.de)
    :Created: 2010/09/22
    """
    
    def __init__(self, num_instances = 20, 
                        generating_function_class_0 = lambda i: numpy.ones((2,2))*i,
                        generating_function_class_1 = lambda i: numpy.ones((2,2))*i,
                        channel_names = ["X", "Y"],
                        class_labels = ['A','B'],
                        class_choice_function = random.random,
                        choice_threshold = 0.33,
                        sampling_frequency = 2,
                        **kwargs):
        super(RandomTimeSeriesSourceNode, self).__init__(**kwargs)
        # We have to create a dummy collection
        class DummyObject(object): pass
        collection = DummyObject()
        collection.meta_data = {'runs' : 1}
        collection.data = {}
        
        # only binary classification supported by now
        assert( len(class_labels) == 2)
        
        self.set_permanent_attributes(collection = collection, 
                                        num_instances = num_instances,
                                        generating_function_class_0 = generating_function_class_0,
                                        generating_function_class_1 = generating_function_class_1,
                                        channel_names = channel_names,
                                        class_labels = class_labels,
                                        class_choice_function = class_choice_function,
                                        choice_threshold = choice_threshold,
                                        sampling_frequency = sampling_frequency)
        
    def generate_random_data(self):
        """ Method that is invoked by train and test data generation functions"""
        # invokes the given generating functions
        generated_data = []
        
        for i in range(self.num_instances):
            choice = self.class_choice_function()
            label = None
            
            if choice < self.choice_threshold:
                input_array = self.generating_function_class_0(i)
                label = self.class_labels[0]
            else:
                input_array = self.generating_function_class_1(i)
                label = self.class_labels[1]
            
            generated_data.append( (TimeSeries(input_array = input_array,
                                        channel_names = self.channel_names, 
                                        sampling_frequency = self.sampling_frequency ),
                                        label))
        return generated_data
    
    def request_data_for_testing(self):
        """
        Returns the data that can be used for testing of subsequent nodes

        .. todo:: to document
        """
        # If we haven't read the data for testing yet
        if self.data_for_testing == None:
            
            generated_data = self.generate_random_data()
                                    
            # Create a generator that emits the windows
            test_data_generator = ((sample, label) \
                                     for (sample, label) in generated_data)
            
            self.data_for_testing = MemoizeGenerator(test_data_generator,
                                                     caching = True)
            
        
        # Return a fresh copy of the generator
        return self.data_for_testing.fresh()   
        
        
    def request_data_for_training(self, use_test_data):
        """
        Returns the data that can be used for testing of subsequent nodes

        .. todo:: to document
        """
        if use_test_data:
            return self.request_data_for_testing()
            
        # If we haven't read the data for testing yet
        if self.data_for_training == None:
            
            generated_data = self.generate_random_data()
                                    
            # Create a generator that emits the windows
            train_data_generator = ((sample, label) \
                                     for (sample, label) in generated_data)
            
            self.data_for_training = MemoizeGenerator(train_data_generator,
                                                     caching = True)
            
        
        # Return a fresh copy of the generator
        return self.data_for_training.fresh()       
        
        
    def get_metadata(self, key):
        """ This source node does not contain collection meta data. """
        return None
