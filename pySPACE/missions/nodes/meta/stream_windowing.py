""" Perform windowing on stream of windows

.. todo:: documentation: Delete upper summaries!
          Check if *meta* is the correct module.
          If yes the documentation should make clear, why.
          The documentation should show in detail the use case of this node.

.. todo:: this module should work without EEG acquisition
"""

import itertools

from pySPACE.missions.nodes.base_node import BaseNode
from pySPACE.resources.dataset_defs.time_series import TimeSeriesClient
from pySPACE.tools.memoize_generator import MemoizeGenerator
from pySPACE.missions.support.windower import Windower, MarkerWindower


class StreamWindowingNode(BaseNode):
    """Get a stream of time series objects and window them inside a flow. 

    Node that interprets a stream of incoming time series objects as
    a raw data stream.
    The markers stored in marker_name attribute are used as the markers
    for a :class:`~pySPACE.missions.support.windower.MarkerWindower`.
    This should done *before* any splitter, since all incoming windows
    are regarded as parts of a consecutive data stream.

    **Parameters**

     :windower_spec_file:
         The window specification file for the
         :class:`~pySPACE.missions.support.windower.MarkerWindower`.
         Used for testing and training, if windower_spec_file_train
         is not specified. 

     :windower_spec_file_train:
         A separate window file for training only.
         If not specified, windower_spec_file is used for training
         and testing.

    
    **Parameters**


    **Exemplary Call**

    .. code-block:: yaml

        -
            node : Stream_Windowing
            parameters :
                windower_spec_file : "example_lrp_window_spec.yaml"

    :Authors: Hendrik Woehrle (hendrik.woehrle@dfki.de)
    :Created: 2012/07/09
    """
    def __init__(self,
                 windower_spec_file,
                 windower_spec_file_train = None,
                 local_window_conf=False,
                 nullmarker_stride_ms=None,
                 *args,
                 **kwargs):
        super(StreamWindowingNode, self).__init__(*args, **kwargs)
        
        if windower_spec_file_train is None:
            windower_spec_file_train = windower_spec_file

        self.set_permanent_attributes(client = None,
                                      marker_windower = None,
                                      window_definition = None,
                                      local_window_conf = local_window_conf,
                                      windower_spec_file = windower_spec_file,
                                      windower_spec_file_train = windower_spec_file_train,
                                      nullmarker_stride_ms=nullmarker_stride_ms)

    def request_data_for_training(self, use_test_data):
        """ Returns the data that can be used for training of subsequent nodes

        .. todo:: to document
        """
        
        # set window definition for train phase windower file
        self.window_definition = \
            Windower._load_window_spec(self.windower_spec_file_train,
                                       self.local_window_conf)

        self._log("Requesting train data...")
        if self.data_for_training is None:
            if not use_test_data:
                # Get training and test data (with labels)
                train_data = \
                    list(self.input_node.request_data_for_training(use_test_data=use_test_data))
                # If training or test data is an empty list
                if train_data == []:
                    self.data_for_training=MemoizeGenerator(
                        (x for x in [].__iter__()), caching=True)
                    return self.data_for_training.fresh()
                # create stream of 
                self.window_stream(train_data)

                # Create a generator that emits the windows
                train_data_generator = ((sample, label) for (sample, label)
                                        in self.marker_windower)
                self.data_for_training = MemoizeGenerator(train_data_generator, 
                                                          caching=True)
                return self.data_for_training.fresh()
        
            else:
                # Return the test data as there is no additional data that
                # was dedicated for training
                self.data_for_training = self.request_data_for_testing()
                return self.data_for_training.fresh()
        else: 
            return self.data_for_training.fresh()

    def request_data_for_testing(self):
        """ Returns the data for testing of subsequent nodes

        .. todo:: to document
        """

        if self.data_for_testing is None:
            # set window definition for test phase windower file
            self.window_definition = \
                Windower._load_window_spec(self.windower_spec_file,
                                           self.local_window_conf)
            test_data = list(self.input_node.request_data_for_testing())

            # create stream of windows
            self.window_stream(test_data)
    
            # Create a generator that emits the windows
            test_data_generator = ((sample, label) \
                                   for (sample, label) in self.marker_windower)
    
            self.data_for_testing = MemoizeGenerator(test_data_generator)
    
            # Return a fresh copy of the generator
            return self.data_for_testing.fresh()
        else: 
            return  self.data_for_testing.fresh()
    
    
    def process(self):
        """ Processes all data that is provided by the input node

        Returns a generator that yields the data after being processed by this
        node.
        """
        assert(self.input_node != None), "No input node specified!"
        # Assert  that this node has already been trained
        assert(not self.is_trainable() or
               self.get_remaining_train_phase() == 0), "Node not trained!"
               
        data_generator = \
                itertools.imap(lambda (data, label):
                               (self.execute(data), label),
                               self.input_node.process())
                
        self.client = TimeSeriesClient(ts_stream = data_generator)
        
        self.client.connect()
        self.marker_windower = MarkerWindower(data_client=self.client,
                                              windowdefs=self.window_definition,
                                              stridems=self.nullmarker_stride_ms)
        
        if self.marker_windower == None:
            self.window_stream()

        # Create a generator that emits the windows
        test_data_generator = ((sample, label) \
                               for (sample, label) in self.marker_windower)

        self.data_for_testing = MemoizeGenerator(test_data_generator)

        # Return a fresh copy of the generator
        return self.data_for_testing.fresh()
        
    def window_stream(self, data):
        # Creates a windower that splits the given data data into windows
        # based in the window definitions provided
        # and assigns correct labels to these windows
        self.client = TimeSeriesClient(ts_stream = iter(data))
        
        self.client.connect()
        self.marker_windower = MarkerWindower(data_client=self.client,
                                              windowdefs=self.window_definition,
                                              stridems=self.nullmarker_stride_ms)
        


_NODE_MAPPING = {"Stream_Windowing": StreamWindowingNode}
