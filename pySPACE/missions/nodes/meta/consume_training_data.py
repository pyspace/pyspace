""" Splits training data for internal usage and usage of successor nodes 

.. todo:: Documentation: node-name should not be module name: Generalization?
"""

import random
import logging
import itertools
from collections import defaultdict
from itertools import repeat

from pySPACE.missions.nodes.base_node import BaseNode
from pySPACE.tools.memoize_generator import MemoizeGenerator

class ConsumeTrainingDataNode(BaseNode):
    """ Split training data for internal usage and usage of successor nodes
    
    This node allows to handle situations where some model needs to be trained 
    and later on evaluated on the given training data (using test data may not 
    be allowed for certain reasons). Simply training and evaluating the model
    on the same data is not an option, since the evaluation would have a strong
    optimistic bias (model is well adapted to the data it was trained on).
    
    One example of such a situation is when a node chain is trained on the data that
    should be combined later on with an ensemble of node chains trained on historic
    data. The ensemble training should not happen on the same data as
    training.  
    
    This node therefore splits the training data into two parts: one for internal
    use (training the model) and one for usage of successor nodes
    (model evaluation). The ratio of training data that should be used 
    internally can be controlled with the argument *consumption_rate* (a value
    between 0.0 and 1.0).
    
    .. note:: 
            When defining  this node in the pySPACE YAML syntax, "wrapped_node"
            can be the definition of a node in YAML syntax (see below).
            The node object is then created automatically based on this definition.
    
    **Parameters**
         
     :wrapped_node: 
         The node that is trained with the internally used training data.
        
     :consumption_rate:
        The rate of training data that is used internally for training 
        *wrapped_node*. The remaining data is supplied for the successor nodes.
    
     :random_seed:
        The seed of the random generator. Defaults to 0.
       

    **Exemplary Call**
    
    
    .. code-block:: yaml
    
        -
            node: ConsumeTrainingData
            parameters : 
                 consumption_rate : 0.8
                 wrapped_node : 
                      node : Flow_Node
                      parameters :
                           input_dim : 64
                           output_dim : 1
                           nodes : 
                              ......
                              
    :Author: Jan Hendrik Metzen (jhm@informatik.uni-bremen.de)
    :Created: 2010/08/06
    """
    
    def __init__(self, wrapped_node, consumption_rate, random_seed=0,
                 *args, **kwargs):
        self.wrapped_node = wrapped_node # Necessary to determine whether trainable.
        super(ConsumeTrainingDataNode, self).__init__(*args, **kwargs)
        
        #############################################
        self.set_permanent_attributes(wrapped_node = wrapped_node,
                                      consumption_rate = consumption_rate,
                                      internal_training_set = [],
                                      external_training_set = [],
                                      r = random.Random(random_seed))
        
    @staticmethod
    def node_from_yaml(node_spec):
        """ Creates a node based on the node_spec to overwrite default """
        # This node requires one parameters, namely a list of nodes
        assert("parameters" in node_spec 
                and "wrapped_node" in node_spec["parameters"]),\
                   "ConsumeTrainingDataNode requires specification of a wrapped node!"
        # Create all nodes that are packed together in this layer
        wrapped_node = BaseNode.node_from_yaml(node_spec["parameters"]["wrapped_node"])
        node_spec["parameters"].pop("wrapped_node")
        # Create the node object
        node_obj = ConsumeTrainingDataNode(wrapped_node = wrapped_node,
                                           **node_spec["parameters"])
            
        return node_obj
         
    def is_trainable(self):
        """ Returns whether this node is trainable. """
        return self.wrapped_node.is_trainable()
    
    def is_supervised(self):
        """ Returns whether this node requires supervised training """
        return self.wrapped_node.is_supervised()

    def _get_train_set(self, use_test_data = False):
        """ Returns the data that can be used for training """
        # We take data that is provided by the input node for training
        # NOTE: This might involve training of the preceding nodes
        train_set = list(self.input_node.request_data_for_training(use_test_data))
        
        # Divide available instances according to label
        all_instances = defaultdict(list)
        for instance, label in train_set:
            all_instances[label].append(instance)
        
        # Split into training data used internally and training data that is 
        # available for successor nodes
        self.internal_training_set = []
        self.external_training_set = []
        for label, instances in all_instances.iteritems():
            self.r.shuffle(instances)
            split_index = int(round(len(instances) * self.consumption_rate))
            self.internal_training_set.extend(zip(instances[:split_index],
                                                  repeat(label)))
            self.external_training_set.extend(zip(instances[split_index:],
                                                  repeat(label)))   
        
        return self.internal_training_set
    
    def request_data_for_training(self, use_test_data):
        """ Returns data for training of subsequent nodes
        
        .. todo:: to document
        """
        assert(self.input_node != None)
        
        self._log("Data for training is requested.", level = logging.DEBUG)
        
        # If we haven't computed the data for training yet
        if self.data_for_training == None:
            self._log("Producing data for training.", level = logging.DEBUG)
            # Train this node
            self.train_sweep(use_test_data)
            
            # Compute a generator the yields the train data and
            # encapsulate it in an object that memoizes its outputs and
            # provides a "fresh" method that returns a new generator that'll
            # yield the same sequence
            train_data_generator = \
                     itertools.imap(lambda (data, label) : (self.execute(data), label),
                                    self.external_training_set) 
                     
            self.data_for_training = MemoizeGenerator(train_data_generator,
                                                      caching=self.caching) 
        
        self._log("Data for training finished", level = logging.DEBUG)
        # Return a fresh copy of the generator  
        return self.data_for_training.fresh()
    
    def _train(self, data, label):
        """ Trains the wrapped nodes on the given data vector *data* """
        self.wrapped_node.train(data, label)
        
    def _stop_training(self):
        """ Finish the training of the node."""
        self.wrapped_node.stop_training()
    
    def _execute(self, data):
        """ Executes the node on the given data vector *data* """
        return self.wrapped_node.execute(data)
    
    def store_state(self, result_dir, index=None):
        """ Stores this node in the given directory *result_dir* """
        self.wrapped_node.store_state(result_dir, index=None)

    def get_output_type(self, input_type, as_string=True):
        """ Return the output type

        The method calls the corresponding method in the wrapped node
        """
        return self.wrapped_node.get_output_type(input_type, as_string)
