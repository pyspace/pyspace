""" Encapsulate complete :mod:`~pySPACE.environments.chains.node_chain` into a single node """

import operator
import cPickle
import copy
import logging
import warnings
import itertools

import pySPACE.missions.nodes
from pySPACE.missions.nodes.base_node import BaseNode
from pySPACE.missions.nodes.source.external_generator_source\
import ExternalGeneratorSourceNode
from pySPACE.missions.nodes.splitter.all_train_splitter import AllTrainSplitterNode
from pySPACE.environments.chains.node_chain import NodeChain
from pySPACE.tools.memoize_generator import MemoizeGenerator


class FlowNode(BaseNode):
    """ Encapsulate a whole node chain from YAML specification or path into a single node
    
    The FlowNode encapsulates a whole node chain so that it can be used like a node.
    The encapsulated chain can either be passed directly via the *nodes*
    parameter. Alternatively, the path to a pickled node chain can be passed
    via *load_path*. In the second case, the object is loaded lazily
    (i.e. only when required). This is important in situations where the
    FlowNode is pickled again (for instance when using
    :class:`~pySPACE.environments.backends.multicore.MulticoreBackend`).
    
    .. note:: 
          When defining  this node in YAML syntax, one can pass
          a "nodes" parameter instead of the "subflow" parameter (see 
          exemplary call below). The value of this parameter must be a 
          NodeChain definition in YAML syntax (properly indented). This
          NodeChain definition is converted into the actual "subflow" parameter
          passed to the constructor in the class' static method 
          "node_from_yaml" (overwriting the default implementation of
          BaseNode). Furthermore, it is determined whether trainable and
          supervised must be True. Thus, these parameters need not be
          specified explicitly.    
    
    **Parameters**

        :subflow:
            The NodeChain object that is encapsulated in this node. Must
            be provided when no load_path is given.   

            (*semi-optional, default: None*)

        :load_path:
            The path to the pickled NodeChain object that is loaded and
            encapsulated in this flow node. Must be given when no subflow is 
            provided. The path string can contain phrases like __SPLIT__ - they
            are replaced in the super node. 

            (*semi-optional, default: None*)

        :trainable: 
            If True, the nodes of the NodeChain require training,
            thus this node itself must be trainable.

            When reading the specification, it is tested, if the subnodes need
            training.

            (*optional, default: False*)

        :supervised:
            If True, the nodes require supervised training, thus this
            node itself must be supervised.

            (*optional, default: False*)

        :input_dim:
            This node may require in contrast to the other nodes that the
            dimensionality of the input data is explicitly set. This is the case
            when the input dimensionality cannot be inferred from the passed
            subflow parameter.

            (*optional, default: None*)

        :output_dim:
            This node may require in contrast to the other  nodes that the
            dimensionality of the output data is explicitly set. This is the case
            when the output dimensionality cannot be inferred from the passed
            subflow parameter.

            (*optional, default: None*)

        :change_parameters:
            List of tuple specifying, which parameters to change in the internal nodes

            Each tuple is a dictionary with the keys:

                :node: Name of the node,
                :parameters: dictionary of new parameters,
                :number: optional number of occurrence in the node (default: 1).

            By default we assume, that node parameters and program variables are
            identical. This default is implemented in the BaseNOde and
            can be overwritten by the relevant node with the function
            *_change_parameters*.

            (*optional, default: []*)

    **Exemplary Call**

    .. code-block:: yaml

         -
             node : FlowNode
             parameters :
                  input_dim : 64
                  output_dim : 1612
                  nodes : 
                            -
                                node : ChannelNameSelector
                                parameters : 
                                    inverse : True
                                    selected_channels: ["EMG1","EMG2","TP7","TP8"]
                            - 
                                node : Decimation
                                parameters : 
                                    target_frequency : 25.0
                            -
                                node : FFT_Band_Pass_Filter
                                parameters : 
                                    pass_band : [0.0, 4.0]
                            -
                                node : Time_Domain_Features
                                parameters :
                                      moving_window_length : 1
                  change_parameters :
                        -
                            node : ChannelNameSelector
                            parameters : 
                                inverse : False
                                selected_channels: ["EMG1","EMG2"]
              
    :Author: Jan Hendrik Metzen (jhm@informatik.uni-bremen.de)
    """
    
    def __init__(self, nodes=None, load_path=None, trainable=False,
                 supervised=False, input_dim=None, output_dim=None, dtype=None,
                 change_parameters=[], **kwargs):
        # We need either a nodes or the path to one
        assert (nodes or load_path)
        
        # set trainable permanently
        self.trainable = trainable
        
        if load_path:
            # assume that all splits have same relevant parameters
            check_nodes_path = load_path.replace("__SPLIT__","0")
            # We load the flow only temporarily, since we might need to
            # pickle this object (depending on the backend), and storing the flow
            # object too early causes overhead
            flow = cPickle.load(open(check_nodes_path, 'r'))
        # Determine some properties of the flow (input_dim, output_dim)
        if input_dim==None and load_path:
            if not input_dim:
                if flow[0].is_source_node(): 
                    input_dim = flow[1].input_dim
                else:
                    input_dim = flow[0].input_dim
            if not output_dim:
                if flow[-1].is_sink_node():
                    output_dim = flow[-2].output_dim
                else:
                    output_dim = flow[-1].output_dim
        elif input_dim==None:
            input_dim = nodes[0].input_dim
            
#            assert input_dim is not None, "You must specify the input dim of " \
#                                          "node %s explicitly!" % self.__class__.__name__
            output_dim = nodes[-1].output_dim
        
        if load_path:
            trainable = reduce(operator.or_, 
                                    [node.is_retrainable() for node in flow])
            supervised = trainable
        # flow is not made permanent but later on loaded
        try:
            del(flow)
        except:
            pass
        # Now we can call the superclass constructor
        super(FlowNode, self).__init__(input_dim=input_dim, 
                                        output_dim=output_dim,
                                       dtype=dtype, **kwargs)
        
        self.set_permanent_attributes(trainable = trainable,
                                      supervised = supervised,
                                      train_instances = [],
                                      change_parameters = change_parameters,
                                      changed=False)
        
        if nodes: # if we got a flow object
            # Remove dtype of the  nodes
            for node in nodes:
                node._dtype = None
            self.set_permanent_attributes(flow = nodes)
        else:
            # Do not load now, but only store path to pickled object
            self.set_permanent_attributes(load_path = load_path,
                                          flow = None) # We will load the nodes lazily

    @staticmethod
    def node_from_yaml(nodes_spec):
        """ Creates the FlowNode node and the contained chain based on the node_spec """
        node_obj = FlowNode(**FlowNode._prepare_node_chain(nodes_spec))
        
        return node_obj
            
    @staticmethod
    def _prepare_node_chain(nodes_spec):
        """ Creates the FlowNode node and the contained chain based on the node_spec """
        assert "parameters" in nodes_spec
        if "load_path" in nodes_spec["parameters"]:
            # Let node load pickled object
            return nodes_spec["parameters"]  
        else:
            # The node chain has to be specified in YAML syntax
            assert "nodes" in nodes_spec["parameters"], \
                       "FlowNode requires specification of a list of nodes " \
                       "or of a load_path to a pickled node chain."
    
            node_sequence = [ExternalGeneratorSourceNode(),
                             AllTrainSplitterNode()]
            # For all nodes in the specs
            for node_spec in nodes_spec["parameters"]["nodes"]:
                # Use factory method to create node
                node_obj = BaseNode.node_from_yaml(node_spec)
                    
                # Append this node to the sequence of node
                node_sequence.append(node_obj) 
                
            # Check if the nodes have to cache their outputs
            for index, node in enumerate(node_sequence):
                # If a node is trainable, it uses the outputs of its input node
                # at least twice, so we have to cache.
                if node.is_trainable():
                    node_sequence[index - 1].set_permanent_attributes(caching=True)
                # Split node might also request the data from their input nodes
                # (once for each split), depending on their implementation. We 
                # assume the worst case and activate caching
                if node.is_split_node():
                    node_sequence[index - 1].set_permanent_attributes(caching=True)
            
            # Determine if any of the nodes is trainable
            trainable = reduce(operator.or_, 
                               [node.is_trainable() for node in node_sequence])
            
            # Determine if any of the nodes requires supervised training
            supervised = reduce(operator.or_, 
                                [node.is_trainable() for node in node_sequence])
            
            # Create the nodes
            flow = NodeChain(node_sequence)
            nodes_spec["parameters"].pop("nodes")
            
            # Evaluate all remaining parameters if they are eval statements
            for key, value in nodes_spec["parameters"].iteritems():
                if isinstance(value, basestring) and value.startswith("eval("):
                    nodes_spec["parameters"][key] = eval(value[5:-1])
            
            # Create the node object
            member_dict = copy.deepcopy(nodes_spec["parameters"])
            member_dict["nodes"] = flow
            member_dict["trainable"] = trainable
            member_dict["supervised"] = supervised
            
            
            return member_dict
        

    def is_trainable(self):
        """ Returns whether this node is trainable. """
        return self.trainable
    
    def is_supervised(self):
        """ Returns whether this node requires supervised training """
        return self.supervised

    def _get_flow(self):
        """ Return flow (load flow lazily if not yet loaded).
    
        .. todo:: Check if first node is  source node and if yes remove
        .. todo:: Add ExternalGeneratorSourceNode if self.trainable
        .. todo:: Check if last node is sink node and remove 
        """
        if not self.flow: # Load nodes lazily
            nodes = cPickle.load(open(self.load_path, 'r'))
            for node in nodes:
                node._dtype = None
            self.flow = nodes
        if not self.changed:
            self.change_flow()
            self.changed=True
        return self.flow

    def change_flow(self):
        for changeset in self.change_parameters:
            number=changeset.get("number",1)
            if not changeset.has_key("node") or not changeset.has_key("parameters"):
                import warnings
                warnings.warn("Could not change change set: "+str(changeset)+"!")
                continue
            i=1
            for node in self.flow:
                if pySPACE.missions.nodes.NODE_MAPPING[changeset["node"]]==type(node):
                    if i==number:
                        node._change_parameters(changeset["parameters"])
                        break
                    else:
                        i+=1

    def _execute(self, data):
        """ Executes the flow on the given data vector *data* """
        # Delegate to internal flow object
        return self._get_flow().execute(data)

    def _train(self, data, label):
        """ Trains the flow on the given data vector *data* """
        self.train_instances.append((data, label))
        
    def _stop_training(self):
        self._get_flow()[0].set_generator(self.train_instances)
        self._get_flow().train()
        self.train_instances = [] # We do no longer need the training data

    def _inc_train(self, data, class_label=None):
        """ Iterate through the nodes to train them """
        for node in self._get_flow():
            if node.is_retrainable() and not node.buffering and hasattr(node, "_inc_train"):
                if not node.retraining_phase:
                    node.retraining_phase=True
                    node.start_retraining()
                node._inc_train(data,class_label)
            data = node._execute(data)
            
    def _batch_retrain(self,data_list, label_list):
        """ Batch retraining for node chains
        
        The input data is taken, to change the first retrainable node.
        After the change, the data is processed and given to the next node,
        which is trained with the data coming from the retrained algorithm.
        """
        for node in self._get_flow():
            for i in range(len(label_list)):
                if node.is_retrainable() and not node.buffering and hasattr(node, "_inc_train"):
                    if not node.retraining_phase:
                        node.retraining_phase=True
                        node.start_retraining()
                    node._inc_train(data_list[i],label_list[i])
            data_list = [node._execute(data) for data in data_list]
        data_list  = None
        label_list = None

    def is_retrainable(self):
        """ Retraining needed if one node is retrainable """
        if self.is_retrainable:
            return True
        else:
            for node in self._get_flow():
                if node.is_retrainable():
                    return True
        return False

    def present_label(self, label):
        """ Forward the label to the nodes
        
        *buffering* must be set to *True* only for the main node for
        using incremental learning in the application (live environment).
        The inner nodes must not have set this parameter.
        
        .. todo::
            Implement check on, if the inner nodes do not buffer.
        """
        super(FlowNode, self).present_label(label)


    def reset(self):
        """ Reset the state to the clean state it had after its initialization """
        # Reset not only the node but also all nodes of the encapsulated node chain.
        # Irrelevant, since the node chain is made permanent or later on loaded.
#        if self._get_flow():
 #           for node in self._get_flow():
  #              node.reset()

        super(FlowNode, self).reset()

    def store_state(self, result_dir, index=None):
        """ Stores this node in the given directory *result_dir* """
        if self._get_flow():
            for node in self._get_flow():
                node.store_state(result_dir, index)


class BatchAdaptSubflowNode(FlowNode):
    """ Load and retrain a pre-trained NodeChain for recalibration
    
    This node encapsulates a whole NodeChain so that it can be
    used like a node. The path to a pickled NodeChain object has to be passed
    via *load_path*. The NodeChain object is loaded lazily (i.e. only when
    required). This is important in situations where this
    node is pickled as part of a NodeChain again (for instance
    when using :class:`~pySPACE.environments.backends.multicore`).
    
    In contrast to the FlowNode, this node allows also to retrain the loaded
    NodeChain to novel training data. All nodes of the loaded NodeChain for which
    *retrain* is set ``True`` are provided with the training data.
    Before this, the method "start_retraining" is called on this node. The
    training data is then provided to the "_inc_train" method.
    
    **Parameters**
         
     :load_path:
         The path to the pickled NodeChain object that is loaded and
         encapsulated in this node. This parameter is not optional!
        
    **Exemplary Call**
    
    .. code-block:: yaml
    
        -
            node : BatchAdaptSubflow
            parameters :
                load_path : "some_path"


    :Author: Mario Krell (mario.krell@dfki.de)
    :Created: 2012/06/20
    """
    
    def __init__(self, load_path,**kwargs):
        super(BatchAdaptSubflowNode, self).__init__(load_path=load_path,**kwargs)
        self.set_permanent_attributes(batch_labels=None)
    
    @staticmethod
    def node_from_yaml(nodes_spec):
        """ Creates the FlowNode node and the contained chain based on the node_spec """
        node_obj = BatchAdaptSubflowNode(**FlowNode._prepare_node_chain(nodes_spec))
        
        return node_obj
    
    def _train(self, data, label):
        """ Expects the nodes to buffer the training samples,
        when they are executed on the data. """
        if self.batch_labels is None:
            self.batch_labels=[]
        # save labels
        self.batch_labels.append(label)
        # save examples via the buffering parameter in the node
        # Only the relevant nodes will save there data
        self._get_flow().execute(data)
        
    def _stop_training(self):
        # Present_label is a BaseNode method.
        # It recursively goes through all nodes.
        # There _batch_retrain is called with our label list and
        # the nodes are retrained with this labels on the previously buffered samples.
        self._get_flow()[-1].present_label(self.batch_labels)
        self.batch_labels = None

_NODE_MAPPING = {"Flow_Node": FlowNode,
                "Batch_Adapt_Subflow" : BatchAdaptSubflowNode}
