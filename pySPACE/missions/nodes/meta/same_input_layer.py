""" Combine several other nodes together in parallel

This is useful to be combined with the
:class:`~pySPACE.missions.nodes.meta.flow_node.FlowNode`.
"""

import numpy
from pySPACE.environments.chains.node_chain import NodeChainFactory

from pySPACE.missions.nodes.base_node import BaseNode
from pySPACE.resources.data_types.feature_vector import FeatureVector
from pySPACE.resources.data_types.time_series import TimeSeries
from pySPACE.resources.data_types.prediction_vector import PredictionVector

# ensemble imports
import os
try:
    import portalocker
except ImportError, e:
    pass
# import fcntl
import fnmatch
import cPickle
import logging
import warnings
from collections import defaultdict

from pySPACE.missions.nodes.meta.flow_node import FlowNode
from pySPACE.tools.filesystem import locate


class SameInputLayerNode(BaseNode):
    """ Encapsulates a set of other nodes that are executed in parallel in the flow. 
    
    This node was a thin wrapper around MDP's SameInputLayer node
    but is now an own implementation.

    **Parameters**

     :enforce_unique_names: 
         When combining time series channels or feature vectors,
         the node adds the index of the current node to the channel names or
         feature names as a prefix to enforce unique names.

        (*optional, default: True*)

    **Exemplary Call**
    
    
    .. code-block:: yaml
    
        - 
            node : Same_Input_Layer
            parameters : 
                 enforce_unique_names  : True
                 nodes : 
                        -
                            node : Time_Domain_Features
                            parameters :
                                  moving_window_length : 1
                        -
                            node : STFT_Features
                            parameters :
                                  frequency_band : [2.0, 8.0]
                                  frequency_resolution : 1.0
    """
    input_types = ["TimeSeries"]
    def __init__(self, nodes,enforce_unique_names=True,
                 store=False, **kwargs):
        self.nodes = nodes  # needed to find out dimensions and trainability,...
        super(SameInputLayerNode, self).__init__(**kwargs)
        self.permanent_state.pop("nodes")
        self.set_permanent_attributes(output_type=None,
                                      names=None,
                                      unique=enforce_unique_names)

    @staticmethod
    def node_from_yaml(layer_spec):
        """ Load the specs and initialize the layer nodes """
        # This node requires one parameters, namely a list of nodes
        assert("parameters" in layer_spec 
               and "nodes" in layer_spec["parameters"]),\
            "SameInputLayerNode requires specification of a list of nodes!"
        # Create all nodes that are packed together in this layer
        layer_nodes = []
        for node_spec in layer_spec["parameters"]["nodes"]:
            node_obj = BaseNode.node_from_yaml(node_spec)
            layer_nodes.append(node_obj)
        layer_spec["parameters"].pop("nodes")
        # Create the node object
        node_obj = SameInputLayerNode(
            nodes=layer_nodes, **layer_spec["parameters"])

        return node_obj

    def reset(self):
        """ Also reset internal nodes """
        nodes = self.nodes
        for node in nodes:
            node.reset()
        super(SameInputLayerNode, self).reset()
        self.nodes = nodes

    def register_input_node(self, input_node):
        """ All sub-nodes have the same input node """
        super(SameInputLayerNode, self).register_input_node(input_node)
        # Register the node as the input for all internal nodes
        for node in self.nodes:
            node.register_input_node(input_node) 
    
    def _execute(self, data):
        """ Process the data through the internal nodes """
        names = []
        result_array  = None
        result_label  = []
        result_predictor  = []
        result_prediction = []
        # For all node-layers
        for node_index, node in enumerate(self.nodes):
            # Compute node's result
            node_result = node.execute(data)
            # Determine the output type of the node
            if self.output_type is None:
                self.output_type = type(node_result)
            else:
                assert (self.output_type == type(node_result)), \
                       "SameInputLayerNode requires that all of its layers return "\
                       "the same type. Types found: %s %s" \
                                % (self.output_type, type(node_result))
            
            # Merge the nodes' outputs depending on the type 
            if self.output_type == FeatureVector:
                result_array = \
                        self.add_feature_vector(node_result, node_index,
                                                result_array, names)
            elif self.output_type == PredictionVector:
                if type(node_result.label) == list:
                    result_label.extend(node_result.label)
                else:
                    # a single classification is expected here
                    result_label.append(node_result.label)
                if type(node_result.prediction) == list:
                    result_prediction.extend(node_result.prediction)
                else:
                    result_prediction.append(node_result.prediction)
                if type(node_result.predictor) == list:
                    result_predictor.extend(node_result.predictor)
                else:
                    result_predictor.append(node_result.predictor)
            else: 
                assert (self.output_type == TimeSeries), \
                        "SameInputLayerNode can not merge data of type %s." \
                                % self.output_type
                if self.names is None and not self.unique:
                    names.extend(node_result.channel_names)
                elif self.names is None and self.unique:
                    for name in node_result.channel_names:
                        names.append("%i_%s" % (node_index, name))
                        
                if result_array == None:
                    result_array = node_result
                    if self.dtype == None:
                        self.dtype = node_result.dtype
                else :
                    result_array = numpy.concatenate((result_array,
                                                         node_result), axis=1)
        # Construct output with correct type and names
        if self.names is None:
            self.names = names
            
        if self.output_type == FeatureVector:
            return FeatureVector(result_array, self.names)
        elif self.output_type == PredictionVector:
            return PredictionVector(label=result_label, 
                                    prediction=result_prediction,
                                    predictor=result_predictor)
        else:
            return TimeSeries(result_array, self.names,
                              node_result.sampling_frequency, 
                              node_result.start_time, node_result.end_time, 
                              node_result.name, node_result.marker_name)
    
    def add_feature_vector(self, data, index, result_array, names):
        """ Concatenate feature vectors, ensuring unique names """
        if self.names is None and self.unique:
            for name in data.feature_names:
                names.append("%i_%s" % (index,name))
        elif self.names is None and not self.unique:
            names.extend(data.feature_names)
            
        if result_array == None:
            result_array = data
        else:
            result_array = numpy.concatenate((result_array,data), axis=1)
        return result_array
    
    
    def is_trainable(self):
        """ Trainable if one subnode is trainable """
        for node in self.nodes:
            if node.is_trainable():
                return True
        return False
    
    def is_supervised(self):
        """ Supervised if one subnode requires supervised training """
        for node in self.nodes:
            if node.is_supervised():
                return True
        return False
#
#    def train_sweep(self, use_test_data):
#        """ Train all internal nodes """
#        for node in self.nodes:
#            node.train_sweep(use_test_data)

    def _train(self, x, *args, **kwargs):
        """ Perform single training step by training the internal nodes """
        for node in self.nodes:
            if node.is_training():
                node.train(x, *args, **kwargs)

    def _stop_training(self):
        """ Perform single training step by training the internal nodes """
        for node in self.nodes:
            if node.is_training():
                node.stop_training()

    def store_state(self, result_dir, index=None):
        """ Stores all nodes in subdirectories of *result_dir* """
        for i, node in enumerate(self.nodes):
            node_dir = os.path.join(result_dir, (self.__class__.__name__+str(index).split("None")[0]+str(i)))
            node.store_state(node_dir, index=i)

    def _inc_train(self, data, label):
        """ Forward data to retrainable nodes
        
        So the single nodes do not need to buffer or *present_labels* does not
        have to be reimplemented.
        """
        for node in self.nodes:
            if node.is_retrainable():
                node._inc_train(data, label)

    def set_run_number(self, run_number):
        """ Informs all subnodes about the number of the current run """
        for node in self.nodes:
            node.set_run_number(run_number)
        super(SameInputLayerNode, self).set_run_number(run_number)

    def get_output_type(self, input_type, as_string=True):
        """ Returns expected output from first node

        Additionally the type is compared with the expected output of
        the other nodes to ensure consistency.
        """
        output = None
        for node in self.nodes:
            if output is None:
                output = node.get_output_type(input_type, as_string)
            elif output != node.get_output_type(input_type, as_string):
                warnings.warn("Node %s yields has a different output from"
                              "the rest of the nodes.", str(node))
            else:
                continue
        return output


class EnsembleNotFoundException(Exception):
    """Error when loading of ensembles is not possible"""
    pass


class ClassificationFlowsLoaderNode(BaseNode):
    """ Combine an ensemble of pretrained node chains

    This node loads all "pickled" flows whose file names match 
    *ensemble_pattern* and are contained in the directory tree rooted at
    *ensemble_base_dir*.  If the *flow_select_list* is not empty, only the 
    flows with indices contained in flow_select_list are used. The index "-1"
    corresponds to "all flows".
    
    **Parameters**
    
     :ensemble_base_dir:
         The root directory under which the stored flow objects which constitute 
         the ensemble are stored.
        
     :ensemble_pattern:
         Pickled flows must match the given pattern to be included into the
         ensemble.
    
     :flow_select_list:
         This optional parameter allows to select only a subset of the flows 
         that are found in ensemble_base_dir. It must be a list of indices.
         Only the flows with the given index are included into the ensemble.
         If -1 is contained in the list, all flows are automatically added to
         the ensemble.
        
         .. note::
               The order of the flows in the ensemble is potentially random or at
               least hard to predict. Thus, this parameter should not be used 
               to select a specific flow. In contrast, this parameter can be used
               to select a certain number of flows from the available flows 
               (where it doesn't matter which ones). This can be useful for instance
               in benchmarking experiments when one is interested in
               the average performance of an ensemble of a certain size. 
              
         (*optional, default: [-1]*)
    
     :cache_dir:
         If this argument is given, all results of all ensembles are remembered
         and stored in a persistent cache file in the given cache_dir. These
         cached results can be later reused without actually loading and 
         executing the ensemble.

         (*optional, default: None*)
    
    **Exemplary Call**
    
    .. code-block:: yaml
    
        -
            node : Ensemble_Node
            parameters :
                  ensemble_base_dir : "/tmp/" # <- insert suitable directory here
                  ensemble_pattern : "flow*.pickle"
                  flow_select_list : "eval(range(10))"
              
    :Author: Jan Hendrik Metzen (jhm@informatik.uni-bremen.de)
    :Created: 2010/05/20
    """
    
    def __init__(self, ensemble_base_dir, ensemble_pattern,
                 flow_select_list=[-1], cache_dir=None, **kwargs):
        super(ClassificationFlowsLoaderNode, self).__init__(**kwargs)
        try:
            import portalocker
        except ImportError, e:
            warnings.warn("Before running " + self.__class__.__name__ +
                          ", please install the portalocker module, e.g.," +
                          " via pip install")
            raise(e)
        # Load all flow-pickle files that match the given ensemble_pattern
        # in the directory tree rooted in ensemble_base_dir
        flow_pathes = tuple(locate(ensemble_pattern, ensemble_base_dir))
        if -1 not in flow_select_list:
            # Select only flows for ensemble whose index is contained in 
            # flow_select_list
            flow_pathes = tuple(flow_pathes[index] for index in flow_select_list)

        if len(flow_pathes) == 0: 
            raise EnsembleNotFoundException(
                "No ensemble found in %s for pattern %s" %
                (ensemble_base_dir, ensemble_pattern))
        
        self.feature_names = \
            map(lambda s: "_".join(s.split(os.sep)[-1].split('_')[0:2]),
            flow_pathes)

        self.set_permanent_attributes(ensemble=None,
                                      flow_pathes=flow_pathes,
                                      cache_dir=cache_dir,
                                      cache=None,
                                      cache_updated=False,
                                      store=True) # always store cache
    
    def _load_cache(self):
        self.cache = defaultdict(dict)
        # Check if there are cached results for this ensemble
        for flow_path in self.flow_pathes: 
            file_path = self.cache_dir + os.sep + "ensemble_cache" + os.sep \
                            + "cache_%s" % hash(flow_path)
            if os.path.exists(file_path):
                # Load ensemble cache
                self._log("Loading flow cache from %s" % file_path)
                lock_file = open(file_path + ".lock", 'w')
                portalocker.lock(lock_file, portalocker.LOCK_EX)
                # fcntl.flock(lock_file, fcntl.LOCK_EX)
                self._log("Got exclusive lock on %s" % (file_path + ".lock"),
                          logging.INFO)
                cache_file = open(file_path, 'r')
                self.cache[flow_path] = cPickle.load(cache_file)
                cache_file.close()
                self._log("Release exclusive lock on %s" % (file_path + ".lock"),
                          logging.INFO)
                # fcntl.flock(lock_file, fcntl.LOCK_UN)
                portalocker.unlock(lock_file)

    def _load_ensemble(self):
        self._log("Loading ensemble")
        # Create a flow node for each  flow pickle
        flow_nodes = [FlowNode(load_path=flow_path)
                      for flow_path in self.flow_pathes]

        # Create an SameInputLayer node that executes all flows independently
        # with the same input
        ensemble = SameInputLayerNode(flow_nodes, enforce_unique_names=True)
        
        # We can now set the input dim and output dim
        self.input_dim = ensemble.input_dim
        self.output_dim = ensemble.output_dim
            
        self.set_permanent_attributes(ensemble = ensemble)
        
    def _train(self, data, label):
        """ Trains the ensemble on the given data vector *data* """
        if self.ensemble == None:
            # Load ensemble since data is not cached
            self._load_ensemble()            
        return self.ensemble.train(data, label)
        
    def _execute(self, data):
        # Compute data's hash
        data_hash = hash(tuple(data.flatten()))

        # Load ensemble's cache
        if self.cache == None:
            if self.cache_dir:
                self._load_cache()
            else: # Caching disabled
                self.cache = defaultdict(dict)
        # Try to lookup the result of this ensemble for the given data in the cache
        labels = []
        predictions = []
        for i, flow_path in enumerate(self.flow_pathes):
            if data_hash in self.cache[flow_path]:
                label, prediction = self.cache[flow_path][data_hash]
            else:
                self.cache_updated = True
                
                if self.ensemble == None:
                    # Load ensemble since data is not cached
                    self._load_ensemble()
                
                node_result = self.ensemble.nodes[i].execute(data)
                label = node_result.label
                prediction = node_result.prediction
                
                self.cache[flow_path][data_hash] = (label, prediction)
                
            labels.append(label)
            predictions.append(prediction)

        result = PredictionVector(label=labels, 
                                  prediction=predictions,
                                  predictor=self)
        result.dim_names = self.feature_names
        
        return result
    
    def store_state(self, result_dir, index=None):
        """ Stores this node in the given directory *result_dir* """
        # Store cache if caching is enabled and cache has changed
        if self.cache_dir and self.cache_updated:
            if not os.path.exists(self.cache_dir + os.sep + "ensemble_cache"):
                os.makedirs(self.cache_dir + os.sep + "ensemble_cache")
            
            for flow_path in self.flow_pathes:
                file_path = self.cache_dir + os.sep + "ensemble_cache" + os.sep \
                                  + "cache_%s" % hash(flow_path)
                if os.path.exists(file_path):
                    self._log("Updating flow cache %s" % file_path)
                    # Update existing cache persistency file
                    lock_file = open(file_path + ".lock", 'w')
                    portalocker.lock(lock_file, portalocker.LOCK_EX)
                    # fcntl.flock(lock_file, fcntl.LOCK_EX)
                    self._log("Got exclusive lock on %s" % (file_path + ".lock"),
                              logging.INFO)
                    cache_file = open(file_path, 'r')
                    self.cache[flow_path].update(cPickle.load(cache_file))
                    cache_file.close()
                    cache_file = open(file_path, 'w')
                    cPickle.dump(self.cache[flow_path], cache_file)
                    cache_file.close()
                    self._log("Release exclusive lock on %s" % (file_path + ".lock"),
                              logging.INFO)
                    portalocker.unlock(lock_file)
                    # fcntl.flock(lock_file, fcntl.LOCK_UN)
                else:
                    self._log("Writing flow cache %s" % file_path)
                    # Create new cache persistency file
                    lock_file = open(file_path + ".lock", 'w')
                    portalocker.lock(lock_file, portalocker.LOCK_EX)
                    # fcntl.flock(lock_file, fcntl.LOCK_EX)
                    self._log("Got exclusive lock on %s" % (file_path + ".lock"),
                              logging.INFO)
                    cache_file = open(file_path, 'w')
                    cPickle.dump(self.cache[flow_path], cache_file)
                    cache_file.close()
                    self._log("Release exclusive lock on %s" % (file_path + ".lock"),
                              logging.INFO)
                    portalocker.unlock(lock_file)
                    # fcntl.flock(lock_file, fcntl.LOCK_UN)

    def get_output_type(self, input_type, as_string=True):
        if as_string:
            return "PredictionVector"
        else:
            return PredictionVector

class MultiClassLayerNode(SameInputLayerNode):
    """ Wrap the one vs. rest or one vs. one scheme around the given node
    
    The given class labels are forwarded to the internal nodes.
    During training, data is relabeled. 
    Everything else is the same as in the base node.
    
    Though this scheme is most important for classification it permits
    other trainable algorithms to use this scheme.
    
    **Parameters**
        :class_labels:
            This is the complete list of expected class labels.
            It is needed to construct the necessary flows in the
            initialization stage.

        :node:
            Specification of the wrapped node for the used scheme
            
            As class labels , for the *1vsR* scheme,
            this node has to use *REST* and *LABEL*.
            *LABEL* is replaced with the different `class_labels`.
            The other label should be *REST*.

            For the *1vs1* scheme *LABEL1* and *LABEL2* have to be used.

        :scheme:
            One of *1v1* (One vs. One) or *1vR* (One vs. Rest)

            .. note:: The one class approach is included by simply not giving
                      'REST' label to the classifier, but filtering it out.

            (*optional, default:'1v1'*)

    **Exemplary Call**

    .. code-block:: yaml

        - 
            node : MultiClassLayer
            parameters :
                class_labels : ["Target", "Standard","Artifact"]
                scheme : "1vR"
                node : 
                    -
                        node : 1SVM
                        parameters :
                            class_labels : ["LABEL","REST"]
                            complexity : 1
    """
    input_types=["FeatureVector"]
    @staticmethod
    def node_from_yaml(layer_spec):
        """ Load the specs and initialize the layer nodes """
        assert("parameters" in layer_spec
               and "class_labels" in layer_spec["parameters"]
               and "node" in layer_spec["parameters"]),\
                   "Node requires specification of a node and classification labels!"
        scheme = layer_spec["parameters"].pop("scheme","1vs1")
        # Create all nodes that are packed together in this layer
        layer_nodes = []
        node_spec = layer_spec["parameters"]["node"][0]
        classes = layer_spec["parameters"]["class_labels"]
        if scheme=='1vR':
            for label in layer_spec["parameters"]["class_labels"]:
                node_obj = BaseNode.node_from_yaml(NodeChainFactory.instantiate(node_spec,{"LABEL":label}))
                layer_nodes.append(node_obj)
        else:
            n=len(classes)
            for i in range(n-1):
                for j in range(i+1,n):
                    replace_dict = {"LABEL1":classes[i],"LABEL2":classes[j]}
                    node_obj = BaseNode.node_from_yaml(NodeChainFactory.instantiate(node_spec,replace_dict))
                    layer_nodes.append(node_obj)
        layer_spec["parameters"].pop("node")
        layer_spec["parameters"].pop("class_labels")
        # Create the node object
        node_obj = MultiClassLayerNode(nodes = layer_nodes,**layer_spec["parameters"])

        return node_obj

_NODE_MAPPING = {"Ensemble_Node": ClassificationFlowsLoaderNode,
                 "Same_Input_Layer": SameInputLayerNode,
                 }
