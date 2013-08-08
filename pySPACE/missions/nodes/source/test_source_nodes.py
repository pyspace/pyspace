""" Source nodes to generate test data with specific properties

Using these nodes, the data with defined properties can be used
to have a 'ground truth'.
This can be used to test the properties and functionality of 
entire node chains.

"""

import numpy
import random

from pySPACE.missions.nodes.source.time_series_source import TimeSeriesSourceNode
from pySPACE.resources.data_types.time_series import TimeSeries

from pySPACE.tests.utils.data.test_data_generation import *

from pySPACE.tools.memoize_generator import MemoizeGenerator

from pySPACE.resources.dataset_defs.time_series import TimeSeriesDataset


class SimpleTimeSeriesSourceNode(TimeSeriesSourceNode):
    """ A simple test class for unit tests 
    
    Generates the same data for test and training.
    """
    
    def __init__(self, *args, **kwargs):
        super(SimpleTimeSeriesSourceNode, self).__init__(*args, **kwargs)
        
        run_number = 0
        
        # We have to create a dummy dataset
        class DummyObject(object): pass
        dataset = DummyObject()
        dataset.meta_data = {'runs' : 1}
        dataset.data = {}
        
        self.set_permanent_attributes(dataset = dataset,
                                      run_number=run_number)
    
    def request_data_for_testing(self):
        """
        Returns the data that can be used for testing of subsequent nodes

        .. todo:: to document
        """
        
        # If we haven't read the data for testing yet
        if self.data_for_testing is None:
            self.time_series = [(TimeSeries(input_array = numpy.ones((2,2))*i,
                                            channel_names = ["X", "Y"], 
                                            sampling_frequency = 2),
                                            random.choice(["A", "B"]))
                                            for i  in range(23)]
            # Create a generator that emits the windows
            test_data_generator = ((sample, label) \
                                     for (sample, label) in self.time_series)
            

            self.data_for_testing = MemoizeGenerator(test_data_generator,
                                                     caching = True)
        # Return a fresh copy of the generator
        return self.data_for_testing.fresh()


class DataGenerationTimeSeriesSourceNode(TimeSeriesSourceNode):
    """ Generate data of two classes for testing
    
    This node can generate data according to the specifications
    of two different DataGenerators. 
    
    It generates objects of the type TimeSeries
    
    **Parameters**
        :ir_generator:
            A generator of type DataGenerator for data items of the 
            information relevant class.
            If it is specified in a node chain, it should be given as a
            string.
            
            (*optional, default: 100*)
            
        :nir_generator:
            A generator of type DataGenerator for data items of the 
            not information relevant class.
            If it is specified in a node chain, it should be given as a
            string.
            
            (*optional, default: 100*)
            
        :ir_items:
            Number of items that should be generated for the ir class.
            
            (*optional, default: 100*)
            
        :nir_items:
            Number of items that should be generated for the non ir class.
            
            (*optional, default: 100*)
            
        :channel_names:
            List of strings for the channel names. 
            Determines also the number of generated
            channels.
            
            (*optional*)

        :num_channels:
            Number of channels. Unused, if channel_names is set.
            
            (*optional, default: 16*)
            
        :ir_label:
            The label for the ir_class.
            
            (*optional, default: 'Target'*)
            
        :nir_label:
            The label for the ir_class.
            
            (*optional, default: 'Standard'*)
            
        :shuffle:
            If the data items for the two classes are shuffled.
            
            (*optional, default: True*)
            
        :time_points:
            Number of points per channel in a generated TimeSeries
            object. 
            
            (*optional, default: 100*)
            
         :sampling_frequency:
             Sampling rate of the generated data.
             Important for sines etc.
             
             A generated time series object has a
             temporal length of time_points/sampling_frequency
         
             (*optional, default: 1000*)
             
          :ir_drift_vector:
              Drift of the ir class data.
              Specify a vector (numpy array) of shape (time_points,num_channels)
              and the a linear drift in this direction will be added to the
              generated data:
              [0 * ir_drift_vector]                       added to first sample,
              [1/(ir_items+nir_items) * ir_drift_vector]  to the second sample
              [...]                                       and so on, until
              [1 * ir_drift_vector]                       added to last sample.
              
              The specification of the drift vector in the specification can, e.g.,
              be done like this:
              ir_drift_vector : "eval(__import__('numpy').asarray([[1,1],[2,2]]))"
              
              (*optional, default: None*)
            
          :nir_drift_vector:
              Drift of the ir class data. See ir_drift_vector. 
              
              (*optional, default: None*)              
              
    **Exemplary Call**
    
    .. code-block:: yaml
    
        -
            node : Data_Generation_Source
            parameters :
                ir_generator : "Adder([SineGenerator(),GaussianNoiseGenerator()])"
                nir_generator : "GaussianNoiseGenerator()"
                
    
    :Author: Hendrik Woehrle
    :Created: 201/07/27
    """
    
    def __init__(self, 
                 ir_generator="Adder([Sine(),GaussianNoise()])",
                 nir_generator="GaussianNoise()",
                 ir_items=100,
                 nir_items=100,
                 ir_drift_vector=None,
                 nir_drift_vector=None,
                 channel_names=None,
                 num_channels=16,
                 ir_label='Target',
                 nir_label='Standard',
                 time_points=100,
                 sampling_frequency=1000,
                 shuffle=True,
                 **kwargs):
        super(DataGenerationTimeSeriesSourceNode, self).__init__(**kwargs)
        
        if type(ir_generator) == str:
            ir_generator = eval(ir_generator)
        
        if type(nir_generator) == str:
            nir_generator = eval(nir_generator)

        ir_generator.sampling_frequency = sampling_frequency
        nir_generator.sampling_frequency = sampling_frequency
        
        run_number = 0
        
        dataset = None

        if not channel_names is None:
            num_channels = len(channel_names)
        else:
            channel_names = []
            for i in xrange(num_channels):
                channel_names.append(str(i))
        
        # Translate drift "None" to zero-vector 
        if ir_drift_vector is None:
            ir_drift_vector = numpy.zeros((time_points,num_channels))
        if nir_drift_vector is None:
            nir_drift_vector = numpy.zeros((time_points,num_channels))
        
        self.set_permanent_attributes(dataset=dataset,
                                      ir_generator=ir_generator,
                                      nir_generator=nir_generator,
                                      ir_items=ir_items,
                                      nir_items=nir_items,
                                      channel_names=channel_names,
                                      num_channels=num_channels,
                                      ir_label=ir_label,
                                      nir_label=nir_label,
                                      time_points=time_points,
                                      sampling_frequency=sampling_frequency,
                                      shuffle=shuffle,
                                      run_number=run_number,
                                      data_for_testing=None,
                                      data_for_training=None,
                                      ir_drift_vector=ir_drift_vector,
                                      nir_drift_vector=nir_drift_vector)
        
        self.generate_data_set()
        
    def set_input_dataset(self, dataset):
        """ Instead of using a given dataset, a new one is generated """
        self.generate_data_set()
        
        
    def generate_data_set(self):
        """ Generate a dataset using the given generators """
        
        self.dataset = TimeSeriesDataset()
        
        # generate a set of dummy labels to know which class is used later
        label_sequence = numpy.hstack((numpy.ones(self.ir_items),numpy.zeros(self.nir_items)))
        
        if self.shuffle:
            random.shuffle(label_sequence)
            
        ts_generator = TestTimeSeriesGenerator()
        
        current_item = 0 # count produced data objects for drift
        for label in label_sequence:
            if label == 1:
                #generate a data item using the ir_generator
                data_item = \
                    ts_generator.generate_test_data(
                        channels=len(self.channel_names), 
                        time_points=self.time_points, 
                        function=self.ir_generator,
                        sampling_frequency=self.sampling_frequency,
                        channel_order=True,
                        channel_names=self.channel_names,
                        dtype=numpy.float)
                # Drift:
                data_item = data_item + current_item*self.ir_drift_vector
                self.dataset.add_sample(data_item,self.ir_label,False)
                
            else:
                #generate a data item using the nir_generator
                data_item = \
                    ts_generator.generate_test_data(
                        channels=len(self.channel_names), 
                        time_points=self.time_points, 
                        function=self.nir_generator,
                        sampling_frequency=self.sampling_frequency,
                        channel_order=True,
                        channel_names=self.channel_names,
                        dtype=numpy.float)
                # Drift:
                data_item = data_item + current_item*self.nir_drift_vector
                self.dataset.add_sample(data_item,self.nir_label,False)

            current_item += 1./(self.ir_items+self.nir_items)


_NODE_MAPPING = {"Data_Generation_Source": DataGenerationTimeSeriesSourceNode,
                "Simple_Test_Source": SimpleTimeSeriesSourceNode}
