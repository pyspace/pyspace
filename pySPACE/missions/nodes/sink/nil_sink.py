""" Store only meta data of dataset

This is useful to save 
disk space in scenarios, where only the output of any preceding node
is desired, and where there is no need to save the whole data set again.

"""

from pySPACE.missions.nodes.sink.time_series_sink import TimeSeriesSinkNode
from pySPACE.resources.dataset_defs.dummy import DummyDataset


class NilSinkNode(TimeSeriesSinkNode):
    """ Store only meta information and perform training and testing
    
    The node inherits from TimeSeriesSinkNode, but instead of
    collecting the data, this node passes a DummyDataset.
    
    
    **Parameters**
    
    **Exemplary Call**

    .. code-block:: yaml

        - 
            node: Nil_Sink

    :Input: any
    :Output: DummyDataset
    :Author: David Feess (david.feess@dfki.de)
    :Created: 2010/03/30
    """
    
    def __init__(self,**kwargs):
        self.dummy_collection=DummyDataset()
        super(NilSinkNode, self).__init__(**kwargs)

    def process_current_split(self):
        """Request the data from the input node and count splits. """
        for _,_ in self.input_node.request_data_for_training(False):
            pass
        for _,_ in self.input_node.request_data_for_testing():
            pass
        
        # Count Splits for meta data. Usually this is done by
        # BaseDataset.add_sample. But here, obviously, no samples are added.
        if self.current_split + 1 > self.dummy_collection.meta_data["splits"]: 
            self.dummy_collection.meta_data["splits"] = self.current_split + 1
            
            
    def get_result_dataset(self):
        """ Return the empty dummy collection """
        return self.dummy_collection


class OnlyTrainSinkNode(NilSinkNode):
    """ Store only meta information and perform training but not testing
    
    The node performs only training on the node chain,
    so that the test procedure can be performed manually,
    e.g. for debug and testing reasons.
    
    The node is very similar to the NilSinkNode.
    
    .. todo:: Merge the nil-nodes

    .. todo:: Change name to more meaningful.
    
    **Parameters**
    
    **Exemplary Call**

    .. code-block:: yaml

        - 
            node: Only_Train_Sink

    :Author: Hendrik Woehrle (hendrik.woehrle@dfki.de)
    :Created: 2011/07/14
    """
    
    def __init__(self, **kwargs):
            super(OnlyTrainSinkNode, self).__init__(**kwargs)
            
            self.set_permanent_attributes(dummy_collection = \
                                                     DummyDataset())
       
        
    def process_current_split(self):
        """ Request the data from the input node and count splits """
        for _,_ in self.input_node.request_data_for_training(False): #feature_vector, label
            pass
        # Count Splits for meta data. Usually this is done by
        # BaseDataset.add_sample. But here, obviously, no samples are added.
        if self.current_split + 1 > self.dummy_collection.meta_data["splits"]: 
            self.dummy_collection.meta_data["splits"] = self.current_split + 1
            
            
    def request_data_for_testing(self):
        """ Request data for testing, just call the predecessors method 
        
        This is possible, since this node does not process any data and slightly
        shortens processing time.
        """
        return self.input_node.request_data_for_testing()

_NODE_MAPPING = {"Nil_Sink": NilSinkNode,
                "Only_Train_Sink": OnlyTrainSinkNode}