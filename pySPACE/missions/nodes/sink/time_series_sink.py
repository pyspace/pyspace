""" Gather all time series objects that are passed through

:Author: Jan Hendrik Metzen (jhm@informatik.uni-bremen.de)
:Created: 2008/11/28
"""

import itertools
import copy
import numpy

from pySPACE.missions.nodes.base_node import BaseNode
from pySPACE.resources.dataset_defs.time_series import TimeSeriesDataset

from pySPACE.resources.data_types.time_series import TimeSeries


class TimeSeriesSinkNode(BaseNode):
    """ Collect all :mod:`time series objects <pySPACE.resources.data_types.time_series>` in a :mod:`collection <pySPACE.resources.dataset_defs.time_series>`
    
    **Parameters**
    
      :sort_string: 
          A lambda function string that is passed to the TimeSeriesDataset and
          evaluated before the data is stored.
                    
          (*optional, default: None*)
          
      :max_num_stored_objects:
          Number of maximal stored time series objects. Can be used if only a part
          of a dataset should be exported, e.g. for size purposes in debugging.
          Applies to train and test set separately.
          
          (*optional, default: numpy.inf*)
    
      :merge:
         Can be set to true if the use wants to get one timeseries containing the
         entier input data
         
         (*optional, default: False*)

    **Exemplary Call**

    .. code-block:: yaml

        - 
            node: Time_Series_Sink

    :Author: Jan Hendrik Metzen (jhm@informatik.uni-bremen.de)
    :Created: 2008/11/28    
    :LastChange: 2011/04/13 Anett Seeland (anett.seeland@dfki.de)        
    """
    input_types = ["TimeSeries"]

    def __init__(self, sort_string=None, merge = False, **kwargs):
        super(TimeSeriesSinkNode, self).__init__(**kwargs)
        
        self.set_permanent_attributes(sort_string=sort_string,
                                      merge = merge,
                                      # This will be created lazily
                                      time_series_collection = None,
                                      max_num_stored_objects = numpy.inf) 
    
    def reset(self):
        """
        Reset the state of the object to the clean state it had after its
        initialization
        """
        # We have to create a temporary reference since we remove 
        # the self.permanent_state reference in the next step by overwriting
        # self.__dict__
        tmp = self.permanent_state
        # TODO: just a hack to get it working quickly...
        tmp["time_series_collection"] = self.time_series_collection 
        self.__dict__ = copy.copy(tmp)
        self.permanent_state = tmp
    
    def is_trainable(self):
        """ Returns whether this node is trainable. """
        # Though this node is not really trainable, it returns true in order
        # to get trained. The reason is that during this training phase, 
        # it stores all time windows along with their class label
        return True
    
    def _get_train_set(self, use_test_data):
        """ Returns the data that can be used for training """
        # We take data that is provided by the input node for training
        # NOTE: This might involve training of the preceding nodes
        train_set = self.input_node.request_data_for_training(use_test_data)
        
        # Add the data provided by the input node for testing to the
        # training set
        # NOTE: This node is not really learning but creating a labeled set
        #       of time windows. Because of that it must take all
        #       data for training (even when use_test_data is False) 
        train_set = itertools.chain(train_set,
                                    self.input_node.request_data_for_testing())
        return train_set
    
    def is_supervised(self):
        """ Returns whether this node requires supervised training """
        return True
    
    def _train(self, data, label):
        # We do nothing
        pass
        
    def process_current_split(self):
        """ 
        Compute the results of this sink node for the current split of the data
        into train and test data
        """
        index = 0
        # Compute the time series for the data used for training
        for time_series, label in self.input_node.request_data_for_training(False):
            # Do lazy initialization of the class 
            if self.time_series_collection == None:
                self.time_series_collection = \
                            TimeSeriesDataset(sort_string=self.sort_string)
            
            if index < self.max_num_stored_objects:
                # Add sample
                self.time_series_collection.add_sample(time_series,
                                                       label = label,
                                                       train = True,
                                                       split = self.current_split,
                                                       run = self.run_number)
            index += 1
            
        # Compute the time series for the data used for testing
        index = 0
        for time_series, label in self.input_node.request_data_for_testing():
            # Do lazy initialization of the class 
            # (maybe there were no training examples)
            if self.time_series_collection == None:
                self.time_series_collection = \
                            TimeSeriesDataset(sort_string=self.sort_string)
            
            if index < self.max_num_stored_objects:
                # Add sample
                self.time_series_collection.add_sample(time_series,
                                                   label = label,
                                                   train = False,
                                                   split = self.current_split,
                                                   run = self.run_number)
            index += 1

    
    def merge_time_series(self, input_collection):
        """ Merges all timeseries of the input_collection to one big timeseries """
        # Retriev the time series from the input_collection
        input_timeseries = input_collection.get_data(0,0,'test')
        # Get the data from the first timeseries
        output_data = input_timeseries[0][0]
        skiped_range = output_data.start_time

        # Change the endtime of the first timeseries to the one of the last
        # timeseries inside the input_collection
        input_timeseries[0][0].end_time = input_timeseries[-1][0].end_time
        # For all the remaining timeseries

        for ts in input_timeseries[1:]:
            # Concatenate the data...
            output_data = numpy.vstack((output_data,ts[0]))
            # ... and add the marker to the first timeseries
            if(len(ts[0].marker_name) > 0):
                for k in ts[0].marker_name:
                    if(not input_timeseries[0][0].marker_name.has_key(k)):
                        input_timeseries[0][0].marker_name[k] = []
                    for time in ts[0].marker_name[k]:
                        input_timeseries[0][0].marker_name[k].append(time+ts[0].start_time - skiped_range)
        # Use the meta information from the first timeseries e.g. marker start/end_time
        # and create a new timeseries with the concatenated data
        merged_time_series = TimeSeries.replace_data(input_timeseries[0][0],output_data)
        # Change the name of the merged_time_series
        merged_time_series.name = "%s, length %d ms, %s" % (merged_time_series.name.split(',')[0], \
                                                            (len(merged_time_series)*1000.0)/merged_time_series.sampling_frequency,\
                                                            merged_time_series.name.split(',')[-1])
        
        return merged_time_series

        
    def get_result_dataset(self):
        """ Return the result """
        # Merges all timeseries inside the collection if merge flag is set to true
        if self.merge:
            merged_time_series = self.merge_time_series(self.time_series_collection)
            self.time_series_collection = None
            self.time_series_collection = \
                      TimeSeriesDataset(sort_string=self.sort_string)
            self.time_series_collection.add_sample(merged_time_series,
                                                 label = 'Window',
                                                 train = False)
        return self.time_series_collection


_NODE_MAPPING = {"Time_Series_Sink": TimeSeriesSinkNode}
