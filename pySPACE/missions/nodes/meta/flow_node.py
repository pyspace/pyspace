""" Encapsulate complete :mod:`~pySPACE.environments.chains.node_chain` into a single node """

import operator
import cPickle
import copy
import logging
import warnings
import itertools
import numpy
import os

import pySPACE.missions.nodes
from pySPACE.missions.nodes.base_node import BaseNode
from pySPACE.missions.nodes.source.external_generator_source\
import ExternalGeneratorSourceNode
from pySPACE.missions.nodes.splitter.all_train_splitter import AllTrainSplitterNode
from pySPACE.environments.chains.node_chain import NodeChain
from pySPACE.tools.memoize_generator import MemoizeGenerator

# BacktransformationNode imports
from pySPACE.resources.data_types.feature_vector import FeatureVector
from pySPACE.resources.data_types.time_series import TimeSeries
from pySPACE.missions.nodes.feature_generation.time_domain_features import TimeDomainFeaturesNode


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
            identical. This default is implemented in the BaseNode and
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
    :Created: 2010/07/28
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

    def set_run_number(self, run_number):
        """ Forward run number to flow """
        if self.load_path is None:
            self._get_flow()[-1].set_run_number(run_number)
        super(FlowNode, self).set_run_number(run_number)

    def set_temp_dir(self, temp_dir):
        """ Forward temp_dir to flow """
        self._get_flow()[-1].set_temp_dir(temp_dir)
        super(FlowNode, self).set_semp_dir(temp_dir)

    def _get_flow(self):
        """ Return flow (load flow lazily if not yet loaded).
    
        .. todo:: Check if first node is  source node and if yes remove
        .. todo:: Add ExternalGeneratorSourceNode if self.trainable
        .. todo:: Check if last node is sink node and remove 
        """
        if not self.flow: # Load nodes lazily
            self.replace_keywords_in_load_path()
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
            i = 1
            for node in self.flow:
                if pySPACE.missions.nodes.NODE_MAPPING[changeset["node"]]==type(node):
                    if i == number:
                        node._change_parameters(changeset["parameters"])
                        break
                    else:
                        i += 1

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
        self._get_flow()._inc_train(data, class_label)

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
        if self.retrainable:
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

    def get_output_type(self, input_type, as_string=True):
        """ Get the output type of the flow

        The method calls the method with the same name from the
        NodeChain module where the output of an entire flow is
        determined
        """
        flow = self._get_flow()
        return flow.get_output_type(input_type, as_string)


class UnsupervisedRetrainingFlowNode(FlowNode):
    """ Use classified label for retraining

    All the other functionality is as described in :class:`FlowNode`.

    **Parameters**

      :confidence_boundary:
        Minimum distance to decision boundary which is required for retraining.
        By default every result is used.
        For regression algorithms, this option cannot be used.

        (*optional, default: 0*)

      :decision_boundary:
        Threshold for decision used for calculating classifier confidence.

        (*optional, default: 0*)

      .. seealso:: :class:`FlowNode`

    **Exemplary Call**

    .. code-block:: yaml

        - node : UnsupervisedRetrainingFlow
          parameters :
            retrain : True
            nodes :
              - node : 2SVM
                parameters :
                  retrain : True

    :Author: Mario Michael Krell (mario.krell@dfki.de)
    :Created: 2015/02/07
    """
    def __init__(self, decision_boundary=0, confidence_boundary=0, **kwargs):
        super(UnsupervisedRetrainingFlowNode, self).__init__(**kwargs)
        self.set_permanent_attributes(decision_boundary=decision_boundary,
                                      confidence_boundary=confidence_boundary)

    @staticmethod
    def node_from_yaml(nodes_spec):
        """ Create the FlowNode node and the contained chain """
        node_obj = UnsupervisedRetrainingFlowNode(
            **FlowNode._prepare_node_chain(nodes_spec))
        return node_obj

    def _inc_train(self, data, class_label=None):
        """ Execute for label guess and retrain if appropriate """
        result = self._execute(data)
        if not type(result.prediction) == list and\
                (abs(result.prediction - self.decision_boundary) >=
                    self.confidence_boundary):
            super(UnsupervisedRetrainingFlowNode, self)._inc_train(
                data, result.label)
        else:  # no adaptation because prediction is not confident
            pass


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
    
    def __init__(self, load_path, **kwargs):
        super(BatchAdaptSubflowNode, self).__init__(
            load_path=load_path, **kwargs)
        self.set_permanent_attributes(batch_labels=None)
    
    @staticmethod
    def node_from_yaml(nodes_spec):
        """ Create the FlowNode node and the contained chain """
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


class BacktransformationNode(FlowNode):
    """ Determine underlying linear transformation of classifier or regression algorithm

    The resulting linear transformation can be accessed with the method:
    *get_previous_transformations* of the following node, e.g., for
    visualization and sensor ranking.
    It is stored in the same format as the input data.

    .. warning:: This node makes sense if and only if the underlying
        transformations are linear. For nonlinear transformations a
        more generic approach needs to be implemented.
        This implementation is not using direct access to the internal
        algorithms but determining the transformation by testing a large
        number of samples, which is not efficient but most generic.

    .. warning:: Currently this node requires stationary processing and does
        not catch the changing transformation from incremental learning.

    **References**

        ========= ==========================================================================================
        main      source: Backtransformation
        ========= ==========================================================================================
        author    Krell, M. M. and Straube, S.
        journal   Advances in Data Analysis and Classification
        title     `Backtransformation: a new representation of data processing chains with a scalar decision function <http://dx.doi.org/10.1007/s11634-015-0229-3>`_
        year      2015
        doi       10.1007/s11634-015-0229-3
        pages     1-25
        ========= ==========================================================================================

    **Parameters**

        .. seealso:: :class:`FlowNode`

        :eps:
            the step of the difference method :math:`\\varepsilon`. Should be set manually for each
            differentiation

            (*default: 2.2e-16*)

        :method:
            the method that should be used for the derivation. The available
            methods are encoded as strings

            * Forward difference method -> ``method="forward_difference"``
            * Central difference method -> ``method="central_difference"``
            * Central difference method using a half step -> ``method="central_difference_with_halfstep"``

        :mode:
            the method used to obtain the backtransformation. The choice of
            method depends to the current dataset and hence

            * Linear, affine datasets -> ``mode="linear"``
            * Non-linear datasets -> ``mode="nonlinear"``

        :store_format:
            specify the format in which the data is to be stored. The options
            here are:

                * `txt` file - this file is generated automatically
                    using `numpy.savetxt`
                * `pickle` file - this file is generated using `pickle.dump`
                * `mat` file - saved using the scipy matlab interface

            If no format is specified, no file will be stored.

            (*optional, default: None*)

    **Exemplary Call**

    .. code-block:: yaml

        -   
            node : Backtransformation
            parameters :
                nodes :
                    -   
                        node : FFTBandPassFilter
                        parameters :
                            pass_band : [0.0, 4.0]
                    -   
                        node : TimeDomainFeatures
                    -
                        node : LinearDiscriminantAnalysisClassifier

    :Author: Mario Michael Krell
    :Created: 2013/12/24
    """
    input_types=["TimeSeries", "FeatureVector"]
    def __init__(self, mode="linear", method="central_difference",
                 eps=2.2*1e-16, store_format=None, **kwargs):
        super(BacktransformationNode, self).__init__(**kwargs)
        self.set_permanent_attributes(trafo=None, offset=0.0, example=None,
                                      mode=mode, method=method, eps=eps,
                                      num_samples=0, covariance=None,
                                      store_format=store_format)

    def _execute(self, data):
        """ Determine example at first call, forward normal processing """
        # generate example at first call
        if self.example is None:
            self.example = copy.deepcopy(data)
            result = super(BacktransformationNode, self)._execute(data)
            return result
        return super(BacktransformationNode, self)._execute(data)

    def _train(self, data, label):
        """ Update covariance matrix and forward training """
        super(BacktransformationNode, self)._train(data, label)
        flattened_data = numpy.atleast_2d(data.get_data().flatten())
        if self.covariance is None:
            self.covariance = flattened_data * flattened_data.T
        else:
            self.covariance += flattened_data * flattened_data.T
        self.num_samples += 1

    def _stop_training(self):
        """ Update covariance matrix and forward training """
        super(BacktransformationNode, self)._stop_training()
        self.covariance = 1.0 * self.covariance / self.num_samples

    def get_own_transformation(self, sample=None):
        """ Return the transformation parameters """
        if sample is None:
            sample = self.example
        if self.example is None:
            self._log("No transformation generated!", level=logging.ERROR)
            return None
        elif self.trafo is None and self.mode == "linear":
            self.generate_affine_backtransformation()
        elif self.mode == "nonlinear":
            self.get_derivative(sample=sample)
        if type(self.example) == TimeSeries:
            return (self.trafo, (self.offset, self.covariance),
                    self.example.channel_names, "generic_backtransformation")
        elif type(self.example) == FeatureVector:
            return (self.trafo, (self.offset, self.covariance),
                    self.example.feature_names, "generic_backtransformation")

    def generate_affine_backtransformation(self):
        """ Generate synthetic examples and test them to determine transformation

        This is the key method!
        """
        if type(self.example) == FeatureVector:
            testsample = FeatureVector.replace_data(
                self.example, numpy.zeros(self.example.shape))
            self.offset = numpy.longdouble(self._execute(testsample))
            self.trafo = FeatureVector.replace_data(
                self.example, numpy.zeros(self.example.shape))
            for j in range(len(self.example.feature_names)):
                testsample = FeatureVector.replace_data(
                    self.example,
                    numpy.zeros(self.example.shape))
                testsample[0][j] = 1.0
                self.trafo[0][j] = \
                    numpy.longdouble(self._execute(testsample) - self.offset)
        elif type(self.example) == TimeSeries:
            testsample = TimeSeries.replace_data(
                self.example, numpy.zeros(self.example.shape))
            self.offset = numpy.longdouble(numpy.squeeze(
                self._execute(testsample)))
            self.trafo = TimeSeries.replace_data(
                self.example, numpy.zeros(self.example.shape))
            for i in range(self.example.shape[0]):
                for j in range(self.example.shape[1]):
                    testsample = TimeSeries.replace_data(
                        self.example, numpy.zeros_like(self.example))
                    testsample[i][j] = 1.0
                    self.trafo[i][j] = \
                        numpy.longdouble(numpy.squeeze(self._execute(testsample))
                                       - self.offset)

    def normalization(self, sample):
        """ normalizes the results of the transformation to the same norm as the input

        **Principle**

        The function first computes the norm of the input and then applies the same norm to
        the self.trafo variable such that the results will be on the same scale

        .. note::

            If either the input or the derivative have not been computed already
            the node will will raise an IOError.
        """
        if self.trafo is None:
            raise IOError("The derivative has not be computed. Cannot perform normalization.")
        if sample is None:
            raise IOError("The initial sample has not been given. Cannot perform normalization.")
        initial = sample.view(numpy.ndarray)
        a = initial[0,:]
        norm_a = numpy.linalg.norm(a)
        if norm_a == 0:
            norm_a = 1

        initial = self.trafo.view(numpy.ndarray)
        b = initial[0,:]
        norm_b = numpy.linalg.norm(b)
        if norm_b == 0:
            norm_b = 1

        self.trafo = FeatureVector.replace_data(self.trafo, b*norm_a/norm_b)

    def get_derivative(self, sample=None):
        """ obtain the derivative of the entire transformation

        The method is just a wrapper for different methods of derivation
        that are called by the method. The first order derivative is saved
        to a variable called ``self.trafo`` and can be visualised using
        specific methods

        The methods used in the following pieces of code are described in
        `Numerical Methods in Engineering with Python <http://books.google.de/books?id=WiDie-hev1kC>`_
        by Jaan Kiusalaas. Namely, the three methods implemented here are:

        * Forward difference method
        * Central difference method
        * Central difference method using a half step

        More details about the implementations can be found in the descriptions
        of the functions

        **Parameters**

            :sample:
                the initial values on which the derivative is to be computed. If no
                sample is provided, the default ``self.example`` variable is used.

            (*default: None*)
        """
        if sample is None:
            warnings.warn("No new sample was given. Using the default example.")
            sample = self.example
        if self.method == "forward_difference":
            self.forward_difference_method(sample=sample)
        elif self.method == "central_difference":
            self.central_difference_method(sample=sample)
        elif self.method == "central_difference_with_halfstep":
            self.central_difference_with_halfstep_method(sample=sample)
        else:
            warnings.warn("Method " + self.method + " is not know. "
                                               "Using the forward difference approach")
            self.forward_difference_method(sample=sample)

        #self.normalization(sample)

    def forward_difference_method(self, sample):
        """ implementation of the forward difference method

        **Principle**

        The principle applied by this method of numerical differentiation is

        .. math::

            f'(x)=\\frac{f(x+h)-f(x)}{h}

        where :math:`h` is the step of the differentiation that is computed
        as :math:`h(x)=\sqrt{\\varepsilon} \\cdot x` for :math:`x \\neq 0` and
        :math:`h(0)=\\sqrt{\\varepsilon}` for :math:`x=0`.

        The differentiation method distinguishes between ``FeatureVector`` and
        ``TimeSeries`` inputs and applies the derivative according to the
        input type.

        **Parameters**

            :sample:
                the initial value used for the derivation

        .. note::

            Out of the three numerical differentiation methods, this one has the
            least overhead. Nonetheless, this method is less accurate than the
            half step method.
        """
        initial_value = self._execute(sample)
        if type(sample) == FeatureVector:
            self.trafo = FeatureVector.replace_data(
                    self.example, numpy.zeros(self.example.shape))
            for j in range(len(sample.feature_names)):
                data_with_offset = copy.deepcopy(sample)
                if data_with_offset[0][j] == 0.:
                    diff = numpy.sqrt(self.eps)
                else:
                    diff = numpy.sqrt(self.eps)*data_with_offset[0][j]
                orig = data_with_offset[0][j]
                data_with_offset[0][j] += diff
                diff = data_with_offset[0][j] - orig
                new_feature_vector = FeatureVector.replace_data(
                    sample,
                    data_with_offset
                )
                self.trafo[0][j] = \
                    numpy.longdouble((self._execute(new_feature_vector) -
                                   initial_value)/diff)
        elif type(sample) == TimeSeries:
            self.trafo = TimeSeries.replace_data(
                self.example, numpy.zeros(self.example.shape))
            for i in range(sample.shape[0]):
                for j in range(sample.shape[1]):
                    data_with_offset = copy.deepcopy(sample)
                    if data_with_offset[i][j] == 0.:
                        diff = numpy.sqrt(self.eps)
                    else:
                        diff = numpy.sqrt(self.eps)*data_with_offset[0][j]
                    data_with_offset[i][j] += diff
                    new_time_series = TimeSeries.replace_data(
                        sample,
                        data_with_offset)
                    self.trafo[i][j] = \
                        numpy.longdouble((numpy.squeeze(self._execute(new_time_series))
                                       - numpy.squeeze(initial_value))/diff)

    def central_difference_method(self, sample):
        """ implementation of the central difference method

        **Principle**

        The principle applied by the central difference method is

        .. math::

            f'(x)=\\frac{f(x+h)-f(x-h)}{2h}

        where :math:`h` is the step of the differentiation that is computed
        as :math:`h(x)=\sqrt{\\varepsilon} \\cdot x` for :math:`x \\neq 0` and
        :math:`h(0)=\\sqrt{\\varepsilon}` for :math:`x=0`.

        **Parameters**

            :sample:
                the initial value used for the derivation
        """
        if type(sample) == FeatureVector:
            self.trafo = FeatureVector.replace_data(
                    sample, numpy.zeros(sample.shape))
            for j in range(len(sample.feature_names)):
                positive_offset = copy.deepcopy(sample)
                negative_offset = copy.deepcopy(sample)
                if positive_offset[0][j] == 0.:
                    diff = numpy.sqrt(self.eps)
                else:
                    diff = numpy.sqrt(self.eps)*positive_offset[0][j]
                positive_offset[0][j] += diff
                negative_offset[0][j] -= diff
                diff = (positive_offset[0][j]-negative_offset[0][j])/2.

                positive_vector = FeatureVector.replace_data(
                    sample,
                    positive_offset
                )
                negative_vector = FeatureVector.replace_data(
                    sample,
                    negative_offset
                )
                self.trafo[0][j] = \
                    numpy.longdouble((self._execute(positive_vector) -
                                   self._execute(negative_vector))/(2.*diff))
        elif type(sample) == TimeSeries:
            self.trafo = TimeSeries.replace_data(
                self.example, numpy.zeros(self.example.shape))
            for i in range(sample.shape[0]):
                for j in range(sample.shape[1]):
                    positive_offset = copy.deepcopy(sample)
                    negative_offset = copy.deepcopy(sample)
                    if positive_offset[i][j] == 0.:
                        diff = numpy.sqrt(self.eps)
                    else:
                        diff = numpy.sqrt(self.eps)*positive_offset[i][j]

                    positive_offset[i][j] += diff
                    negative_offset[i][j] -= diff
                    diff = (positive_offset[i][j]-negative_offset[i][j])/2.

                    positive_series = TimeSeries.replace_data(
                        sample,
                        positive_offset
                    )
                    negative_series = TimeSeries.replace_data(
                        sample,
                        negative_offset
                    )
                    self.trafo[i][j] = \
                        numpy.longdouble((self._execute(positive_series) -
                                       self._execute(negative_series))/(2.*diff))

    def central_difference_with_halfstep_method(self, sample):
        """ implementation of the central difference method with a half step

        **Principle**

        The principle applied by the central difference method with a half step is

        .. math::

            f'(x)=\\frac{f(x-h)-8f(x-\\frac{h}{2})+8f(x+\\frac{h}{2})-f(x-h)}{6h}

        where :math:`h` is the step of the differentiation that is computed
        as :math:`h(x)=\sqrt{\\varepsilon} \\cdot x` for :math:`x \\neq 0` and
        :math:`h(0)=\\sqrt{\\varepsilon}` for :math:`x=0`.

        **Parameters**

            :sample:
                the initial value used for the derivation

        .. note::

            This method is the most accurate differentiation method but also
            has the greatest overhead.
        """
        if type(sample) == FeatureVector:
            self.trafo = FeatureVector.replace_data(
                    self.example, numpy.zeros(self.example.shape))
            for j in range(len(sample.feature_names)):
                positive_offset = copy.deepcopy(sample)
                negative_offset = copy.deepcopy(sample)
                half_positive_offset = copy.deepcopy(sample)
                half_negative_offset = copy.deepcopy(sample)

                if positive_offset[0][j] == 0.:
                    diff = numpy.sqrt(self.eps)
                else:
                    diff = numpy.sqrt(self.eps)*positive_offset[0][j]
                positive_offset[0][j] += diff
                negative_offset[0][j] -= diff
                half_positive_offset[0][j] += diff/2.
                half_negative_offset[0][j] -= diff/2.
                
                diff = (positive_offset[0][j]-negative_offset[0][j])/2.

                positive_vector = FeatureVector.replace_data(
                    sample,
                    positive_offset
                )
                negative_vector = FeatureVector.replace_data(
                    sample,
                    negative_offset
                )
                half_positive_vector = FeatureVector.replace_data(
                    sample,
                    half_positive_offset
                )
                half_negative_vector = FeatureVector.replace_data(
                    sample,
                    half_negative_offset
                )
                self.trafo[0][j] = \
                    numpy.longdouble((self._execute(negative_vector) -
                                    8*self._execute(half_negative_vector) +
                                    8*self._execute(half_positive_vector) -
                                    self._execute(positive_vector))/(6.*diff))
        elif type(sample) == TimeSeries:
            self.trafo = TimeSeries.replace_data(
                self.example, numpy.zeros(self.example.shape))
            for i in range(sample.shape[0]):
                for j in range(sample.shape[1]):
                    positive_offset = copy.deepcopy(sample)
                    negative_offset = copy.deepcopy(sample)
                    half_positive_offset = copy.deepcopy(sample)
                    half_negative_offset = copy.deepcopy(sample)

                    if positive_offset[i][j] == 0.:
                        diff = numpy.sqrt(self.eps)
                    else:
                        diff = numpy.sqrt(self.eps)*positive_offset[i][j]
                    positive_offset[i][j] += diff
                    negative_offset[i][j] -= diff
                    half_positive_offset[i][j] += diff/2.
                    half_negative_offset[i][j] -= diff/2.
                    
                    diff = (positive_offset[i][j]-negative_offset[i][j])/2.

                    positive_series = TimeSeries.replace_data(
                        sample,
                        positive_offset
                    )
                    negative_series = TimeSeries.replace_data(
                        sample,
                        negative_offset
                    )
                    half_positive_series = TimeSeries.replace_data(
                        sample,
                        half_positive_offset
                    )
                    half_negative_series = TimeSeries.replace_data(
                        sample,
                        half_negative_offset
                    )
                    self.trafo[i][j] = \
                        numpy.longdouble((self._execute(negative_series) -
                                        8*self._execute(half_negative_series) +
                                        8*self._execute(half_positive_series) -
                                        self._execute(positive_series))/(6.*diff))

    def get_sensor_ranking(self):
        """ Transform the transformation to a sensor ranking by adding the respective absolute values

        This method is following the principles as implemented in
        :class:`~pySPACE.missions.nodes.classification.base.RegularizedClassifierBase`.
        There might be some similarities in the code.
        """
        self.generate_backtransformation()
        ## interfacing to code from RegularizedClassifierBase
        if type(self.trafo) == FeatureVector:
            trafo = self.trafo
        elif type(self.trafo) == TimeSeries:
            # canonic mapping of time series to feature vector for simplicity
            node = TimeDomainFeaturesNode()
            trafo = node._execute(self.trafo)
        ## code from RegularizedClassifierBase with ``trafo`` instead of
        ## ``self.features``
        # channel name is what comes after the first underscore
        feat_channel_names = [chnames.split('_')[1]
                              for chnames in trafo.feature_names]
        from collections import defaultdict
        ranking_dict = defaultdict(float)
        for i in range(len(trafo[0])):
            ranking_dict[feat_channel_names[i]] += abs(trafo[0][i])
        ranking = sorted(ranking_dict.items(),key=lambda t: t[1])
        return ranking

    def _inc_train(self, data, class_label=None):
        """ This method is not yet implemented """
        self._log("Incremental backtransformation is not yet available!",
                  level=logging.ERROR)
        super(BacktransformationNode,self)._inc_train(data, class_label)

    @staticmethod
    def node_from_yaml(nodes_spec):
        """ Creates the FlowNode node and the contained chain based on the node_spec """
        node_obj = BacktransformationNode(**FlowNode._prepare_node_chain(nodes_spec))
        return node_obj

    def store_state(self, result_dir, index=None):
        """ Store the results

        This method stores the transformation matrix, the offset, the
        covariance matrix and the channel names. The `store_format` variable
        must be set to either of the 3 corresponding formats: `txt`, `pickle`
        or `mat`. If the `store_format` variable is `None`, the output will
        not be stored.
        """
        super(BacktransformationNode, self).store_state(result_dir,index)
        if self.store_format is None:
            return
        stored = False
        if "txt" in self.store_format:
            numpy.set_printoptions(threshold='nan')
            name = "%s_sp%s.txt" % ("backtransformation", self.current_split)
            file_name = os.path.join(result_dir, name)
            numpy.savetxt(file_name, self.get_own_transformation(), delimiter=" ", fmt="%s")
            stored = True
        if "pickle" in self.store_format:
            import pickle
            name = "%s_sp%s.pickle" % ("backtransformation", self.current_split)
            file_name = os.path.join(result_dir, name)
            pickle.dump(self.get_own_transformation(), open(file_name, "w"))
            stored = True
        if "mat" in self.store_format:
            import scipy.io
            name = "%s_sp%s.mat" % ("backtransformation", self.current_split)
            file_name = os.path.join(result_dir, name)
            result = self.get_own_transformation()
            result_dict = {
                "Transformation matrix":result[0],
                "Offset":result[1][0],
                "Covariance matrix":result[1][1],
                "Feature/Channel names": result[2],
                "Transformation name": result[3]
            }
            scipy.io.savemat(open(file_name, "w"), result_dict)
            stored = True
        if self.store_format is not None and not stored:
            message = ("Storage format \"%s\" unrecognized. " +
                       "Please choose between \"mat\",\"txt\""+
                       " and \"pickle\"") % self.store_format
            warnings.warn(message)


_NODE_MAPPING = {"Flow_Node": FlowNode,
                "Batch_Adapt_Subflow" : BatchAdaptSubflowNode}
