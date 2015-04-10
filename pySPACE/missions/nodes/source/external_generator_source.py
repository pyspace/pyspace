""" Use external data as input """

from pySPACE.missions.nodes.base_node import BaseNode

class ExternalGeneratorSourceNode(BaseNode):
    """ Yield the data provided by an external generator
    
    This node enables to easy input data from an external
    data generator within an internal data node chain.

    The easiest usage is together with meta nodes, where
    the data is given explicitly to the node with the `set_generator` function.
    
    This node only uses one data generator which is taken as testing data.
    For training usage an additional splitter is needed as usual.

    .. note:: Below you can see a more complex example on how to make use of this node.
    
        .. code-block:: python
          
          # Training is done in separate threads, we send the time series
          # windows to these threads via two queues
          self.queueS2 = Queue.Queue()
          self.queueS3 = Queue.Queue()

          # The two classification threads access the two queues via two 
          # generators
          def s2_generator():
              # Yield all windows until a None item is found in the queue
              while True:
                  window = self.queueS2.get(block = True, timeout = None)
                  if window == None: break
                  yield window

          def s3_generator():
              # Yield all windows until a None item is found in the queue
              while True:
                  window = self.queueS3.get(block = True, timeout = None)
                  if window == None: break
                  yield window

          # Create the actual data chains for S1 vs S2 discrimination
          # and S1 vs S3 discrimination
          self.S1S2 = NodeChainFactory.flow_from_yaml(Flow_Class = NodeChain,
                                            flow_spec_file = dataflow_spec_S2)
          self.S1S2[0].set_generator(s2_generator())

          self.S1S3 = NodeChainFactory.flow_from_yaml(Flow_Class = NodeChain,
                                            flow_spec_file = dataflow_spec_S3)
          self.S1S3[0].set_generator(s3_generator())

    **Parameters**

    **Exemplary Call**

    .. code-block:: yaml

        - 
            node: External_Generator_Source_Node

    :Author: Jan Hendrik Metzen (jhm@informatik.uni-bremen.de)
    :Created: 2009/09/17

    """
    input_types = ["TimeSeries", "FeatureVector", "PredictionVector"]

    def __init__(self, **kwargs):
        super(ExternalGeneratorSourceNode, self).__init__(**kwargs)
        
        self.set_permanent_attributes(
            current_split=0)

    def set_generator(self, generator):
        """ Sets the generator from which this node reads the data """
        self.set_permanent_attributes(generator=generator)

    def register_input_node(self, node):
        """ Register the given node as input """
        raise Exception("No nodes can be registered as inputs for source nodes")

    def use_next_split(self):
        """ Use the next split (if any) """
        assert(False)
    
    def train_sweep(self, use_test_data):
        """ Performs the actual training of the node.
        
        .. note:: Source nodes cannot be trained
        """
        raise Exception("Source nodes cannot be trained")
    
    def request_data_for_training(self, use_test_data):
        """ Returns time windows usable for training of subsequent nodes

        .. todo:: to document
        """
        if not use_test_data:
            # Returns an iterator that iterates over an empty sequence
            # (i.e. an iterator that is immediately exhausted), since
            # this node does not provide any data that is explicitly
            # dedicated for training 
            return (x for x in [].__iter__())
        else:
            # Return the test data as there is no additional data that
            # was dedicated for training
            return self.request_data_for_testing()
    
    def request_data_for_testing(self):
        """ Returns time windows usable for testing of subsequent nodes
        
        .. todo:: to document
        """
        def logging_generator():
            for data, label in self.generator:
                yield (data, label)
        
        return logging_generator()
    
    def process(self):
        """ Returns a generator that yields all data contained in the generator
        """
        def logging_generator():
            for data, label in self.generator:
                yield (data, label)
                
        return logging_generator()
    
    def __getstate__(self):
        """ Return a pickable state for this object """
        odict = super(ExternalGeneratorSourceNode, self).__getstate__()
        odict['generator'] = None
        
        return odict
    
    
    def get_metadata(self, key):
        """ This source node does not contain collection meta data. """
        return None


_NODE_MAPPING = {"External_Generator_Source_Node": ExternalGeneratorSourceNode}
