""" Use all available data for training """

import itertools
import logging

from pySPACE.missions.nodes.base_node import BaseNode

class AllTrainSplitterNode(BaseNode):
    """ Use all available data for training
    
    This node allows subsequent nodes to use all available labeled
    data for training. Accordingly, no data for testing is provided.
    
    **Parameters**

    **Exemplary Call**
    
    .. code-block:: yaml
    
        -
            node : All_Train_Splitter
    
    :Author: Jan Hendrik Metzen (jhm@informatik.uni-bremen.de)
    :Created: 2009/01/07
    """
    
    def __init__(self, non_persistent = False, use_test_data = True,
                 *args, **kwargs):
        super(AllTrainSplitterNode, self).__init__(*args, **kwargs)

        self.set_permanent_attributes(non_persistent=non_persistent)
        self.set_permanent_attributes(use_test_data=use_test_data)

    def is_split_node(self):
        """ Returns whether this is a split node. """
        return True
    
    def use_next_split(self):
        """ Use the next split of the data into training and test data.
        
        Returns True if more splits are available, otherwise False.
        
        This method is useful for benchmarking
        """
        # This source node provides only one single split of the data,
        # namely using all data as training data.
        return False
    
    def train_sweep(self, use_test_data):
        """ Performs the actual training of the node.
        
        .. note:: Split nodes cannot be trained
        """
        raise Exception("Split nodes cannot be trained")
    
    def request_data_for_training(self, use_test_data):
        """ Returns the data for training of subsequent nodes

        .. todo:: to document
        """
        self._log("Data for training is requested.", level = logging.DEBUG)
        # This splitter uses all data points for training
        # self.use_test_data is True
        train_data = \
            itertools.imap(lambda (data, label) : (data, label),
                self.input_node.request_data_for_training(use_test_data = self.use_test_data))
        self._log("Data for training finished", level = logging.DEBUG)
        
        return train_data
    
    def request_data_for_testing(self):
        """ Returns the data for testing of subsequent nodes

         .. todo:: to document
         """
        self._log("Data for testing is requested.", level = logging.DEBUG)
        self._log("Returning iterator over empty sequence.", level = logging.DEBUG)
        return (x for x in [].__iter__())

#    def __getstate__(self):
#        """ Return a pickable state for this object """
#        odict = super(AllTrainSplitterNode, self).__getstate__()
#        if self.non_persistent == True:
#            odict['data_for_training'] = None
#            odict['data_for_testing'] = None
#        return odict


_NODE_MAPPING = {"All_Train_Splitter": AllTrainSplitterNode}
