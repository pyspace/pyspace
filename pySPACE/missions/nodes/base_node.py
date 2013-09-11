""" Skeleton for an elemental transformation of the signal

This includes some exception and metaclass handling, but the most important part
is the :class:`~pySPACE.missions.nodes.base_node.BaseNode`.

.. note::
    This module includes a reimplementation of the MDP node class that
    is better suited for the purposes of pySPACE. For instance
    it provides methods to allow the benchmarking of supervised training,
    storing, loading, cross validation, logging ...
    Furthermore, it takes care for the totally different data types,
    because in our case, the input data is 2-dimensional.
    These differences in concept are quite essential and resulted in
    creating an 'own' implementation, comprising the code into one module,
    instead of keeping the inheritance of the MDP node class.
    Nevertheless a lot of code was copied from this great library.

.. image:: ../../graphics/node.png
   :width: 500

:Author: Jan Hendrik Metzen (jhm@informatik.uni-bremen.de)
:Created: 2008/11/25

MDP (version 3.3) is distributed under the following BSD license::

    This file is part of Modular toolkit for Data Processing (MDP).
    All the code in this package is distributed under the following conditions:

    Copyright (c) 2003-2012, MDP Developers <mdp-toolkit-devel@lists.sourceforge.net>

    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

        * Redistributions of source code must retain the above copyright
          notice, this list of conditions and the following disclaimer.
        * Redistributions in binary form must reproduce the above copyright
          notice, this list of conditions and the following disclaimer in the
          documentation and/or other materials provided with the distribution.
        * Neither the name of the Modular toolkit for Data Processing (MDP)
          nor the names of its contributors may be used to endorse or promote
          products derived from this software without specific prior written
          permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
    ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
    FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
    DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
    CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
    OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import itertools
import copy

# logging imports
import logging
import logging.handlers
import warnings

import socket
import os
import time
import cPickle
import numpy

import pySPACE
from pySPACE.tools.memoize_generator import MemoizeGenerator


# Exceptions from MDP
class NodeException(Exception):
    """Base class for exceptions in `Node` subclasses."""
    pass


class InconsistentDimException(NodeException):
    """Raised when there is a conflict setting the dimensions

    Note that incoming data with conflicting dimensionality raises a normal
    `NodeException`.
    """
    pass


class TrainingException(NodeException):
    """Base class for exceptions in the training phase."""
    pass


class TrainingFinishedException(TrainingException):
    """Raised when the `Node.train` method is called although the
    training phase is closed.
    """
    pass


class IsNotTrainableException(TrainingException):
    """Raised when the `Node.train` method is called although the
    node is not trainable.
    """
    pass


class NodeMetaclass(type):
    """ General meta class for future features """
    def __new__(cls, classname, bases, members):
        """ Forward to standard method from type """
        return super(NodeMetaclass, cls).__new__(cls, classname, bases, members)


class BaseNode(object):
    """ Main base class for nodes which forwards data without processing

    It provides methods to allow the benchmarking of supervised training,
    storing, loading, cross validation, logging, ...
    Furthermore, it takes care for different data types.
    The input data is currently two-dimensional.
    It can be:

        * :class:`~pySPACE.resources.data_types.time_series.TimeSeries` or
        * :class:`~pySPACE.resources.data_types.feature_vector.FeatureVector` or
        * :class:`~pySPACE.resources.data_types.prediction_vector.PredictionVector`
        * which all inherit from a common :class:`~pySPACE.resources.data_types.base.BaseData`.

    In the following parameters are introduced which do not give any
    functionality but which could generally be used by inheriting nodes.

    **Parameters**

        :input_dim:
            Dimension(s) of the input data.
            By default determined automatically.

            (*optional, default: None*)

        :output_dim:
            Dimension(s) of the output data.
            By default determined automatically.

            (*optional, default: None*)

        :dtype:
            Data type of the data array.
            By default determined automatically.

            (*optional, default: None*)

        :keep_in_history:
            This parameter is a specialty, which comes with the
            :class:`~pySPACE.resources.data_types.base.BaseData`.
            The execution result of the node
            is copied into the *history* parameter of the object.
            Additionally, the *specs* of the object receive an entry labeled '
            node_specs' containing a dictionary of additional information
            from the saving node.

            Especially :mod:`~pySPACE.missions.nodes.visualization` nodes
            may use this functionality to visualize the change ifn the
            processing of the data.

            (*optional, default: False*)

        :load_path:
            This is the standard variable to load processing information for
            the node especially from previous seen data.

            Examples for the usage, are the loading of spatial filters,
            classifiers or feature normalizations.
            If a parameter load_path is provided for any node, the
            node is able to replace some keywords.

            So far implemented replacements:

                :__RUN__: current run number
                :__SPLIT__: current split number

            Be aware that corresponding split and run numbers don't
            necessarily mean that you're operating on the same data.

            Especially if cross validations generated the splits, there
            is no reason to believe that the current splitting has
            anything to do with a previous one!

            .. note::
                The keywords **__INPUT_DATASET__** and **__RESULT_DIRECTORY__** can
                also be used. The replacement of these keyword is done
                in the :class:`~pySPACE.missions.operations.node_chain.NodeChainOperation`.

            (*optional, default: None*)

        :store:
            If the node parameter *store* is set to 'True', before each reset
            the internal state of the node is stored (pickled) with the
            store_state method.

            (*optional, default: False*)

        :retrain:
            If your node has the method *_inc_train* and you want to use
            *incremental* training during testing or application phase,
            this parameter has to be set to True.
            After processing the data, the node will immediately get the label
            to learn changes in the data.

            For more subtle retraining in the online application,
            you will additionally have to use the
            parameter *buffering* ('True') to save all occurring samples
            in the testing phase. The retraining is then activated by
            calling the method *present_label(label)*:

                If the the label is *None*, only the first buffered element
                is deleted. This is used, if we don't get a label,
                if we are insecure of the true label or
                if we simply do not want to retrain on this sample.
                In the other case, the presented label belongs to the first
                buffered element, which is then given to the *_inc_train* method
                together with its label.
                Afterwards the buffered element is deleted.

                The method could be called in different ways in a sink node,
                to simulate different ways of getting labels and different ways
                of incremental learning.

                Furthermore, it could used by node_chain_scripts
                as they can be found in the
                :mod:`~pySPACE.environments.live` environment,
                where we have the real
                situation, that we have to check after the classification,
                what was the right label of the data.

            .. note:: Before using this parameter you should always check, if
                      the node is able for incremental learning!

            (*optional, default: False*)

        :buffering:
            This switch is responsible for real time *incremental*
            learning of the node in applications (live environment),
            by mainly buffering all samples in the execute method in the testing
            phase.

            If *buffering* is set to 'True', the *retrain* parameter should also be
            and the node must have an *_inc_train* method.
            Furthermore the *present_label* method must be called externally.
            Otherwise you will run into memory issues.

            For more details see the documentation of the *retrain* parameter.

            (*optional, default: False*)

        :zero_training:
            This enforces the node to be not trained, though it is trainable.

            .. warning:: For usage in nodes, the algorithms need to define
                proper defaults in the initialization, e.g. by using the
                *load_path* parameter.

            (*optional, default: True*)

    **Implementing your own Node**

    For finding out, how to implement your own node, have a look at the
    :mod:`~pySPACE.missions.nodes.templates`.

    **Exemplary Call**

    .. code-block:: yaml

        -
            node : Noop
            parameters:
                input_dim : 42
                output_dim : 42


    :input:    Any (e.g. FeatureVector)
    :output:   Any1 (e.g. FeatureVector)
    :Author: Mario Michael Krell and many more (krell@uni-bremen.de)
    :Created: before 2008/09/28
    """
    # setting the meta class
    __metaclass__ = NodeMetaclass

    def __init__(self, store=False, retrain=False, input_dim=None, output_dim=None, dtype=None, **kwargs):
        """ This initialization is necessary for every node

        So make sure, that you use it via the *super* method in each new node.
        The method cares for the setting of the basic parameters, including
        parameters for storing,
        and handling of training and test data.
        """
        # Sanity checks
        assert store in [True, False], \
            "Passing inappropriate value %s for parameter 'store'." % store

        assert retrain in [True, False], \
            "Passing inappropriate value %s for parameter 'retrain'." \
                                                                 % retrain
        self.store = store
        self.retrainable = retrain

        #: parameter for retraining in application
        #: see *present_label*
        self.buffering = False
        if kwargs.has_key("buffering"):
            self.buffering = kwargs["buffering"]
        
        self.zero_training = False
        if kwargs.has_key("zero_training"):
            self.zero_training = kwargs["zero_training"]
            
        if self.buffering and not self.retrainable:
            warnings.warn("Buffering nodes should retrains!")
        self.retraining_phase=False
        
        # whether to save data or not
        self.save_intermediate_results = False
        if kwargs.has_key("save_intermediate_results"):
            self.save_intermediate_results = kwargs["save_intermediate_results"]

        # initialize basic attributes
        self._input_dim = None
        self._output_dim = None
        self._dtype = None
        # call set functions for properties
        self.set_input_dim(input_dim)
        self.set_output_dim(output_dim)
        self.set_dtype(dtype)

        # skip the training phase if the node is not trainable
        if not self.is_trainable() or self.zero_training:
            self._training = False
            self._train_phase = -1
            self._train_phase_started = False
        else:
            # this var stores at which point in the training sequence we are
            self._train_phase = 0
            # this var is False if the training of the current phase hasn't
            #  started yet, True otherwise
            self._train_phase_started = False
            # this var is False if the complete training is finished
            self._training = True

        self.input_node = None

        self.data_for_training = None
        self.data_for_testing = None

        self.root_logger = None

        # distinguish execution on training and test data
        # since some nodes only want to handle test data and ignore
        # training data
        self._training_execution_phase = False

        self.current_split = 0

        self.trace = False

        #: Do we have to remember the outputs of this node for later reuse?
        self.caching = False

        self.load_path=kwargs.get('load_path', None)
        self.keep_in_history=kwargs.get('keep_in_history', False)

        self.node_specs = {}
        self.node_name = str(type(self)).split(".")[-1].split("'")[0]

        self.retrain_data = None
        self.retrain_label = None

        # Every Parameter is stored since we reset them with every new spit.
        self.permanent_state = copy.deepcopy(self.__dict__)

    ###### Methods, which can be overwritten by inheriting nodes ######
    def _train(self, x):
        """ Give the training data to the node

        If a node is trainable, this method is called and *has to be* implemented.
        Optionally the :func:`_stop_training` method can be additionally implemented.
        """
        if self.is_trainable():
            raise NotImplementedError("The node %s is not trainable"%self.__class__.__name__)

    def _stop_training(self, *args, **kwargs):
        """ Called method after the training data went through the node

        It can be overwritten by the inheriting node.
        Normally, the :func:`_train` method only collects the data
        and this method does the real (batch) training.

        By default this method does nothing.
        """
        pass

    def _execute(self, x):
        """ Elemental processing step (**key component**)

        This method should be overwritten by the inheriting node.
        It implements the final processing of the data of the node.

        By default the data is just forwarded.

        Some nodes only visualize or analyze training data
        or only handle the data sets without changing the data
        and so they do not need this method.
        """
        return x

    def _check_train_args(self, x, *args, **kwargs):
        """ Checks if the arguments are correct for training

        Implemented by subclasses if needed.

        .. todo:: Check if this method copy is needed and
                  if there is a good use-case.
        """
        pass

    def _inc_train(self,data, class_label=None):
        """ Method to be overwritten by subclass for incremental training after initial training """
        raise NotImplementedError("The node %s does not implement incremental training."%self.__class__.__name__)

    #@staticmethod
    def is_trainable(self):
        """ Return True if the node can be trained, False otherwise

        *Default*: False
        """
        return False

    #@staticmethod
    def is_supervised(self):
        """ Returns whether this node requires supervised training

        *Default*: False
        """
        return False

    ###### Reimplementation of some MDP methods that have some flaws, ######
    ###### when used with the different concepts, used here.          ######
    ### check functions
    def _check_input(self, x):
        """ Check the input_dim and array consistency

        Here input_dim are the dimensions of the input array
        """
        data_array=x.view(numpy.ndarray)

        # check input rank
        if not x.ndim == 2:
            error_str = "Class %s: x has rank %d, should be 2" \
                            % (self.__class__.__name__, x.ndim)
            raise NodeException(error_str)
        # check for NaN
        if not numpy.isfinite(data_array).all():
            error_str = "Class %s: Not finite number in data: %s!" \
                            % (self.__class__.__name__, str(data_array))
            raise NodeException(error_str)
        # set the dtype if necessary
        if self.dtype is None:
            self.dtype = x.dtype
        # set the input dimension if necessary
        if self.input_dim is None:
            shape = x.shape
            if len(shape)==1:
                self.input_dim=shape[0]
            else:
                self.input_dim=shape
        # check the input dimension
        if not x.shape == self.input_dim and not x.shape[1]==self.input_dim:
            error_str = "Class %s: x has dimension %s, should be %s" \
                % (self.__class__.__name__, str(x.shape), str(self.input_dim))
            raise NodeException(error_str)

    ### Handle the data types of the data
    def _get_supported_dtypes(self):
        """ Return the list of dtypes supported by this node.

        The types can be specified in any format allowed by *numpy* `dtype`.

        .. todo:: In future we should use as default float and double
                    and specify explicitly for each node
                    if it can use other input formats!
        """
        def get_dtypes(typecodes_key):
            """Return the list of dtypes corresponding to the set of
            typecodes defined in numpy.typecodes[typecodes_key].
            E.g., get_dtypes('Float') = [dtype('f'), dtype('d'), dtype('g')].
            """
            types = []
            for c in numpy.typecodes[typecodes_key]:
                try:
                    dtype = numpy.dtype(c)
                    types.append(dtype)
                except TypeError:
                    pass
            return types

        return get_dtypes('All')

    def get_dtype(self):
        """ Return dtype."""
        return self._dtype

    def set_dtype(self, t):
        """Set internal structures' dtype.

        Perform sanity checks and then calls `self._set_dtype(n)`, which
        is responsible for setting the internal attribute `self._dtype`.

        .. note:: Subclasses should overwrite self._set_dtype when needed.
        """
        # Difference to MDP's standard set_dtype: Setting of dtypes allows
        # now also dtypes that are inherited from supported dtypes
        # (in order to allow for var-length strings)
        if t is None:
            return
        t = numpy.dtype(t)
        if (self._dtype is not None) and (self._dtype != t):
            errstr = ("Class %s: dtype is already set to '%s' "
                      "('%s' given)!" % (self.__class__.__name__, t,
                                         self.dtype.name))
            raise Exception(errstr)
        else:
            for dt in self.get_supported_dtypes():
                if numpy.issubdtype(t, dt):
                    self._set_dtype(t)
                    return
            errstr = ("\ndtype '%s' is not supported.\n"
                      "Supported dtypes: %s" %
                            (t.name, [numpy.dtype(t).name for t in
                                                 self.get_supported_dtypes()]))
            raise Exception(errstr)

    def _set_dtype(self, t):
        t = numpy.dtype(t)
        if t not in self.get_supported_dtypes():
            raise NodeException('dtype %s not among supported dtypes (%s) in node %s'
                                % (str(t), self.get_supported_dtypes(),self.__class__.__name__))
        self._dtype = t

    def get_supported_dtypes(self):
        """Return dtypes supported by the node as a list of numpy `dtype` objects.

        Note that subclasses should overwrite `self._get_supported_dtypes`
        when needed."""
        return [numpy.dtype(t) for t in self._get_supported_dtypes()]

    supported_dtypes = property(get_supported_dtypes,
                                doc="Supported dtypes")
    dtype = property(get_dtype,
                     set_dtype,
                     doc = "dtype")
    ################ Reimplementation END ################


    @staticmethod
    def node_from_yaml(node_spec):
        """ Creates a node based on the dictionary *node_spec* """
        # The node_spec from the calling method should not be changed,
        # hence there are maybe several recalls with the same node_spec
        node_spec = copy.deepcopy(node_spec)
        if node_spec is None:
            warnings.warn("Maybe you have a wrong minus with no following "
                          "entry in your spec file? Please correct and "
                          "restart!")
            return
        # evaluation of components of the form "eval(command)"
        if isinstance(node_spec["node"], basestring) \
                            and node_spec["node"].startswith("eval("):
            node_name = eval(node_spec["node"][5:-1])
        else:
            node_name = node_spec["node"]
        try:
            node_class = pySPACE.missions.nodes.NODE_MAPPING[node_name]
        except KeyError:
            raise UserWarning("No node with name %s exists" % node_name)

        # If the node overwrites this method we delegate node creation
        if node_name not in ['Noop', "Base", "BaseNode"] and \
                'node_from_yaml' in node_class.__dict__:
            return node_class.node_from_yaml(node_spec)
        elif node_class.__module__ == "pySPACE.missions.nodes.external":
            # do not interface the wrapper but the real class
            if 'node_from_yaml' in node_class.__bases__[0].__dict__:
                return node_class.node_from_yaml(node_spec)

        # If parameters need to be passed to the class
        if "parameters" in node_spec:
            # All parameters which are eval() statements
            # are considered to be python expressions and are evaluated
            BaseNode.eval_dict(node_spec["parameters"])
            # Create the node object
        #try:
            node_obj = node_class(**node_spec["parameters"])
        #except TypeError, e:
            #raise TypeError("%s: %s" % (node_class.__name__, e))
        else:
            node_obj = node_class()
        return node_obj

    @staticmethod
    def eval_dict(dictionary):
        """ Check dictionary entries starts and evaluate if needed

        Evaluation is switched on, by using ``eval(statement)`` to
        evaluate the *statement*.
        Dictionary entries are replaced with evaluation result.

        .. note:: No additional string mark up needed, contrary to normal
                python evaluate syntax
        """
        for key, value in dictionary.iteritems():
            if isinstance(value, basestring) and value.startswith("eval("):
                try:
                    dictionary[key] = eval(value[5:-1])
                except:
                    warnings.warn("Could not evaluate:" + value)

    def set_permanent_attributes(self, **kwargs):
        """ Add all the items of the given kwargs dictionary as permanent attributes of this object

        Permanent attribute are reset, when using the reset method.
        The other attributes are deleted.

        .. note:: Parameters of the basic init function are always set permanent.
        .. note:: The memory of permanent attributes is doubled.
                  When having large objects, like the data in source nodes,
                  you should handle this by overwriting the reset method.

        The main reason for this method is the reset of nodes during cross
        validation. Here the parameters of the algorithms have to be reset,
        to have independent evaluations.

        """
        self.__dict__.update(kwargs)
        # Deepcopy the permanent state (except the input node and generator)
        for key, value in kwargs.iteritems():
            if key == "input_node" or (key == "generator"):
                self.permanent_state[key] = value
            else:
                self.permanent_state[key] = copy.deepcopy(value)
        # Track changes in node_specs. This is a dictionary with all local
        # variables of the __init__ function.
        # The variable names can be found with the co_varnames attribute
        # of the function object code object.
        self.node_specs = dict([ (key, val)
                for key, val in self.__dict__.items()
                if key in self.__init__.im_func.func_code.co_varnames])
        self.node_specs.update({'node_name': self.node_name})

    def reset(self):
        """ Reset the state of the object to the clean state it had after its initialization

        .. note:: Attributes in the permanent state are not overwritten/reset.
                  Parameters were set into permanent state with the method:
                  *set_permanent_attributes*.
        """
        # We have to create a temporary reference since we remove
        # the self.permanent_state reference in the next step by overwriting
        # self.__dict__
        tmp = self.permanent_state
        # The input node should not be deepcopied since otherwise the input
        # node and the node in the node list that precedes this node are
        # different objects
        input_node = self.permanent_state.pop("input_node")
        self.__dict__ = copy.deepcopy(tmp)
        self.input_node = input_node
        self.permanent_state = tmp
        self.permanent_state["input_node"] = input_node

    def reset_attribute(self, attribute_string):
        """ Reset a single attribute with its previously saved permanent state """
        if not isinstance(attribute_string, basestring):
            warnings.warn("You did not use a string for reset."+
                          "Instead you used:%s."%str(attribute_string))
        else:
            try:
                self.__dict__[attribute_string]=copy.deepcopy(self.permanent_state[attribute_string])
            except KeyError:
                warnings.warn("You did not use a valid attribute for reset."+
                          "Instead you used:%s."%str(attribute_string))

    def is_retrainable(self):
        """ Returns if node supports retraining """
        return self.retrainable

    def is_source_node(self):
        """ Returns whether this node is a source node that can yield data """
        # A source node is identified by its name ending
        return self.__class__.__name__.endswith("SourceNode")

    def is_sink_node(self):
        """ Returns if this node is a sink node that gathers results"""
        # A sink node is identified by its property of having a method
        # with the name "store_results"
        return hasattr(self, "get_result_dataset")

    def is_split_node(self):
        """ Returns whether this is a split node. """
        return False

    def register_input_node(self, node):
        """ Register the given node as input """
        self.set_permanent_attributes(input_node=node)

    def set_run_number(self, run_number):
        """ Informs the node about the number of the current run

        Per default, a node is not interested in the run number and simply
        hands the information back to its input node.
        For nodes like splitter that are interested in the run_number, this method
        can be overwritten.
        """
        self.set_permanent_attributes(run_number=run_number)
        if not self.is_source_node():
            self.input_node.set_run_number(run_number)

    def set_temp_dir(self, temp_dir):
        """ Give directory name for temporary data saves """
        self.set_permanent_attributes(temp_dir = temp_dir)
        try:
            self.input_node.set_temp_dir(temp_dir)
        except:
            pass

    def get_source_file_name(self):
        """ Returns the name of the source file.

        This works for the Stream2TimeSeriesSourceNode and the
        Stream2TimeSeriesSourceNode, for other nodes None
        is returned.
        """
        try:
            return self.input_node.get_source_file_name()
        except:
            pass

    def perform_final_split_action(self):
        """ Perform automatic action when the processing of the current split is finished.

        This method does nothing in the default case, but can be overwritten by child nodes if desired.
        """
        pass

    def use_next_split(self):
        """ Use the next split of the data into training and test data.

        Returns True if more splits are available, otherwise False.

        This method is useful for benchmarking
        """
        assert(self.input_node != None)

        has_more_splits = self.input_node.use_next_split()

        self.perform_final_split_action()

        if has_more_splits:
            # Counting the number of the current split
            self.increase_split_number()
            # Resetting the node for the next run
            self.reset()

        return has_more_splits

    def increase_split_number(self):
        """ Method for increasing split number (needed for access by meta nodes)

        .. todo:: Better exception handling. Move code to meta/Layer nodes?
        """
        self.set_permanent_attributes(current_split=self.current_split + 1)
        try:
            for node in self.nodes:
                node.increase_split_number()
        except:
            pass
        try:
            self.node.increase_split_number()
        except:
            pass

    def _get_train_set(self, use_test_data = False):
        """ Returns the data that can be used for training """
        # We take data that is provided by the input node for training
        # NOTE: This might involve training of the preceding nodes
        train_set = self.input_node.request_data_for_training(use_test_data)
        # If we should also use the test data for training (i.e. we are not
        # doing benchmarking...)
        if use_test_data:
            # Add the data provided by the input node for testing to the
            # training set
            train_set = \
                itertools.chain(train_set,
                                self.input_node.request_data_for_testing())
        return train_set

    def train_sweep(self, use_test_data):
        """ Performs the actual training of the node.

        If use_test_data is True, we use all available data for training,
        otherwise only the data that is explicitly marked as data for training.
        This is a requirement e.g. for benchmarking.
        """
        # If this node does not require training
        if not self.is_trainable() or self.zero_training:
            self._log("Does not require training.")
            # Get train data since that causes the predecessor nodes
            # to be trained
            train_set = self._get_train_set(use_test_data)
        # Check whether the node requires supervised training
        elif self.is_supervised(): # Supervised learning
            self._log("Supervised training started.")
            # For all train phases
            while self.get_remaining_train_phase() > 0:
                self._log("Supervised train stage %s started."
                                            %  self._train_phase)
                train_set = self._get_train_set(use_test_data)
                # Present all available data (along with the corresponding
                # label) to this node
                for data, label in train_set:
                    self.train(data, label)
                # Stop this train phase
                self.stop_training()

                self._log("Supervised train stage %s finished."
                                                         % self._train_phase)
            self._log("Supervised training finished.")
        elif self.is_trainable(): # Unsupervised learning
            self._log("Unsupervised training started.")
            train_set = self._get_train_set(use_test_data)
            # For all train phases
            while self.get_remaining_train_phase() > 0:
                self._log("Unsupervised train stage %s started."
                                        %  self._train_phase)
                # Present all available data to this node, but
                # skip the label (since we are doing unsupervised training)
                for data, label in train_set:
                    self.train(data)

                self._log("Unsupervised train stage %s finished."
                                         %  self._train_phase)
                # Stop this train phase
                self.stop_training()
            self._log("Unsupervised training finished.")

    def process(self):
        """ Processes all data that is provided by the input node

        Returns a generator that yields the data after being processed by this
        node.
        """
        assert(self.input_node != None), "No input node specified!"
        # Assert  that this node has already been trained
        assert(not self.is_trainable() or
               self.get_remaining_train_phase() == 0), "Node not trained!"
        self._log("Processing data.", level = logging.DEBUG)
        data_generator = \
                itertools.imap(lambda (data, label):
                      (self._trace(self.execute(self._trace(data, "entry")),
                                   "exit"), label),
                        self.input_node.process())
        return data_generator

    def request_data_for_training(self, use_test_data):
        """ Returns data for training of subsequent nodes of the node chain

        A call to this method might involve training of the node chain up this
        node. If use_test_data is true, all available data is used for
        training, otherwise only the data that is explicitly for training.
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
            # This line crashes without the NodeMetaclass bug fix
            train_data_generator = \
                 itertools.imap(lambda (data, label) :
                                            (self.execute(data,in_training=True), label),
                                self.input_node.request_data_for_training(
                                                                use_test_data))
            self.data_for_training = MemoizeGenerator(train_data_generator,
                                                      caching=self.caching)

        self._log("Data for training finished", level = logging.DEBUG)
        # Return a fresh copy of the generator
        return self.data_for_training.fresh()

    def request_data_for_testing(self):
        """ Returns data for testing of subsequent nodes of the node chain

        A call to this node might involve evaluating the whole node chain
        up to this node.
        """
        assert(self.input_node != None)

        self._log("Data for testing is requested.", level = logging.DEBUG)

        # If we haven't computed the data for testing yet
        if self.data_for_testing == None:
            # Assert  that this node has already been trained
            assert(not self.is_trainable() or
                   self.get_remaining_train_phase() == 0)
            # Compute a generator the yields the test data and
            # encapsulate it in an object that memoizes its outputs and
            # provides a "fresh" method that returns a new generator that'll
            # yield the same sequence
            self._log("Producing data for testing.", level = logging.DEBUG)
            test_data_generator = \
                itertools.imap(lambda (data, label):
                                                self.test_retrain(data, label),
                               self.input_node.request_data_for_testing())
            self.data_for_testing = MemoizeGenerator(test_data_generator,
                                                     caching=self.caching)
        self._log("Data for testing finished", level = logging.DEBUG)
        # Return a fresh copy of the generator
        return self.data_for_testing.fresh()


    def test_retrain(self,data,label):
        """ Wrapper method for offline incremental retraining

        The parameter *retrain* has to be set to True to activate offline retraining.
        The parameter *buffering* should be False, which is the default.

        .. note:: The execute method of the node is called implicitly
                  in this node instead of being called in the
                  request_data_for_testing-method.
                  For the incremental retraining itself
                  the method _inc_train (to be implemented)
                  is called.

        For programming, we first train on the old data and then execute
        on the new one. This is necessary, since the following nodes
        may need the status of the transformation.
        So we must not change it after calling execute.

        .. note :: Currently there is no retraining to the last sample.
                      This could be done by modifying the :func:`present_label`
                      method and calling  it in the last node after the
                      last sample was processed.
        """
        if self.is_retrainable() and not self.buffering and hasattr(self, "_inc_train"):
            if not self.retraining_phase:
                self.retraining_phase=True
                self.start_retraining()
            else:
                self._inc_train(self.retrain_data,self.retrain_label)
            new_data = self.execute(data)
            self.retrain_data = data
            self.retrain_label = label
            return (new_data, label)
        else:
            return (self.execute(data), label)

    def start_retraining(self):
        """ Method called for initialization of retraining """
        pass

    def present_label(self, label):
        """ Wrapper method for incremental training in application case (live)

        The parameters *retrain* and *buffering* have to be
        set to True to activate this functionality.

        For skipping examples, you can use None, "null" or an empty string as label.

        .. note:: For the incremental training itself
                  the method _inc_train (to be implemented)
                  is called.
        """
        if not self.input_node is None:
            self.input_node.present_label(label)
        if self.buffering and self.is_retrainable():
            if not self.retraining_phase:
                self.retraining_phase=True
                self.start_retraining()
            if not label in [None, "null", ""]:
                if type(label) is str:
                    self._inc_train(self.data_buffer[0],label)
                elif type(label) is list:
                    self._batch_retrain(self.data_buffer[0:len(label)],label)
                    for i in range(len(label-1)):
                        self.data_buffer.pop(0)
            self.data_buffer.pop(0)

    def _batch_retrain(self,data_list, label_list):
        """ Interface for retraining with a set of data

        A possible application is a calibration phase, where we may want to
        improve non-incremental algorithms.

        If this method is not overwritten, it uses the incremental training
        as a default.
        """
        for i in range(label_list):
            self._inc_train(data_list[i],label_list[i])
        data_list  = None
        label_list = None

    def _change_parameters(self,parameters):
        """ Overwrite parameters of a node e.g. when it is loaded and
        parameters like *retrain* or *recalibrate* have to be set to True.

        The node only provides the simple straight forward way,
        of permanently replacing the parameters.
        For more sophisticated parameter handling, nodes have to replace this
        method by their own.
        """
        self.set_permanent_attributes(**parameters)

    def store_state(self, result_dir, index=None):
        """ Stores this node in the given directory *result_dir*

        This method is automatically called during benchmarking
        for every node.
        The standard convention is, that nodes only store their state,
        if the parameter *store* in the specification is set *True*.
        """
        def format_dict(item):
            if issubclass(type(item), BaseNode):
                    # the item is a nested pyspace node -> get dictionary and clean it
                    attr_dict = item.__getstate__()
                    attr_dict = format_dict(attr_dict)
                    return attr_dict
            elif isinstance(item, dict):
                # the item is a attribute dict -> remove inappropriate value
                new_dict = {}
                if item.has_key('input_node'):
                    del item['input_node'] 
                for key, value in item.iteritems():
                    if isinstance(value, dict):
                        value = format_dict(value)    
                    elif isinstance(value, list):
                        temp = []
                        for subitem in value:
                            if isinstance(subitem, dict) or issubclass(type(subitem), BaseNode):
                                temp.append(format_dict(subitem))
                            else:
                                temp.append(subitem)
                        new_dict[key] = temp
                    elif value is None:
                        new_dict[key] = -1
                    else:
                        new_dict[key] = value
                return new_dict
            else:
                # the item is a primitive type
                return item

        if self.store:
            import scipy.io
            node_index = 0
            result_file = None
            # export to text file 
            while not result_file:
                filename = os.path.join(result_dir, self.node_name + "_"
                                        + str(node_index))
                if os.path.isfile(filename+".mat"):
                    node_index += 1
                    continue
                result_file = open(filename+".mat","w")
            attr_dict = self.__getstate__()
            # matlab doesn't like Nones.. replace Nones with 0's
            attr_dict = format_dict(attr_dict)
            scipy.io.savemat(result_file,mdict=attr_dict)
            

    def _log(self, message, level = logging.INFO):
        """ Log the given message into the logger of this class """
        if pySPACE.configuration.min_log_level>level:
            return

        if not hasattr(self,"root_logger") or self.root_logger == None:
            self.root_logger = logging.getLogger("%s-%s.%s" % (socket.gethostname(),
                                                               os.getpid(),
                                                               self))
        if len(self.root_logger.handlers)==0:
            self.root_logger.addHandler(logging.handlers.SocketHandler('localhost',
                    logging.handlers.DEFAULT_TCP_LOGGING_PORT))
        self.root_logger.log(level, message)

    def _trace(self, x, key_str):
        """ Every call of this function creates a time-stamped log entry """
        if self.trace:
            self._log("%s time: %f" % (key_str , time.time()))
        return x

    def __getstate__(self):
        """ Return a pickable state for this object """
        self._log("Pickling instance of class %s." % self.__class__.__name__,
                  level = logging.DEBUG)
        odict = self.__dict__.copy() # copy the dict since we change it
        odict['data_for_training'] = None
        odict['data_for_testing'] = None
        odict['root_logger'] = None
        del odict['permanent_state']
        # Remove other non-pickable stuff
        remove_keys=[]
        for key, value in odict.iteritems():
            if key == "input_node":
                continue
            try:
                cPickle.dumps(value)
            except (TypeError, cPickle.PicklingError):
                remove_keys.append(key)

        for key in remove_keys:
            self._log("Removing attribute %s of class %s (type %s) because of "
                      "it can not be pickled."
                            % (key, self.__class__.__name__, type(odict[key])),
                       level = logging.INFO)
            odict.pop(key)
        return odict

    def __setstate__(self, sdict):
        """ Restore object from its pickled state"""
        self._log("Restoring instance of class %s." % self.__class__.__name__,
                  level = logging.DEBUG)
        self.__dict__.update(sdict)   # update attributes
        # Reconstruct the permanent state of the object
        # This should be a deepcopy except for the input node...
        if "input_node" in self.__dict__:
            input_node = self.__dict__.pop("input_node")
            self.permanent_state = copy.deepcopy(self.__dict__)
            self.__dict__["input_node"] = input_node
            self.permanent_state['input_node'] = self.input_node
        else:
            self.permanent_state = copy.deepcopy(self.__dict__)

    def replace_keywords_in_load_path(self):
        """ Replace keywords in the load_path parameter

        .. note::
            The keywords **__INPUT_DATASET__** and **__RESULT_DIRECTORY__** can
            also be used. The replacement of these keyword is done by
            the :class:`~pySPACE.missions.operations.node_chain.NodeChainOperation`.
        """
        self.load_path = self.load_path.replace('__SPLIT__',
                                                '%i' % self.current_split)
        self.load_path = self.load_path.replace('__RUN__',
                                                '%i' % self.run_number)

    ### properties, copied from MDP without change

    def get_input_dim(self):
        """Return input dimensions."""
        return self._input_dim

    def set_input_dim(self, n):
        """Set input dimensions.

        Perform sanity checks and then calls ``self._set_input_dim(n)``, which
        is responsible for setting the internal attribute ``self._input_dim``.
        Note that subclasses should overwrite `self._set_input_dim`
        when needed.
        """
        if n is None:
            pass
        elif (self._input_dim is not None) and (self._input_dim != n):
            msg = ("Input dim are set already (%d) "
                   "(%d given) in node %s!" % (self.input_dim, n, self.__class__.__name__))
            raise InconsistentDimException(msg)
        else:
            self._set_input_dim(n)

    def _set_input_dim(self, n):
        self._input_dim = n

    input_dim = property(get_input_dim,
                         set_input_dim,
                         doc="Input dimensions")

    def get_output_dim(self):
        """Return output dimensions."""
        return self._output_dim

    def set_output_dim(self, n):
        """Set output dimensions.

        Perform sanity checks and then calls ``self._set_output_dim(n)``, which
        is responsible for setting the internal attribute ``self._output_dim``.
        Note that subclasses should overwrite `self._set_output_dim`
        when needed.
        """
        if n is None:
            pass
        elif (self._output_dim is not None) and (self._output_dim != n):
            msg = ("Output dim are set already (%d) "
                   "(%d given) in node %s!" % (self.output_dim, n, self.__class__.__name__))
            raise InconsistentDimException(msg)
        else:
            self._set_output_dim(n)

    def _set_output_dim(self, n):
        self._output_dim = n

    output_dim = property(get_output_dim,
                          set_output_dim,
                          doc="Output dimensions")

    ### Definition of training sequence from MDP
    _train_seq = property(lambda self: self._get_train_seq(),
                          doc="""\
        List of tuples::

          [(training-phase1, stop-training-phase1),
           (training-phase2, stop_training-phase2),
           ...]

        By default::

          _train_seq = [(self._train, self._stop_training)]
        """)

    def _get_train_seq(self):
        return [(self._train, self._stop_training)]

    def has_multiple_training_phases(self):
        """Return True if the node has multiple training phases."""
        return len(self._train_seq) > 1

    ### Node states from MDP
    def is_training(self):
        """Return True if the node is in the training phase,
        False otherwise."""
        return self._training

    def get_current_train_phase(self):
        """Return the index of the current training phase.

        The training phases are defined in the list `self._train_seq`."""
        return self._train_phase

    def get_remaining_train_phase(self):
        """Return the number of training phases still to accomplish.

        If the node is not trainable then return 0.
        """
        if self.is_trainable() and not self._train_phase<0:
            return len(self._train_seq) - self._train_phase
        else:
            return 0

    ### check functions from mdp
    def _check_output(self, y):
        # check output rank
        if not y.ndim == 2:
            error_str = "y has rank %d, should be 2 in node %s" % (y.ndim, self.__class__.__name__)
            raise NodeException(error_str)

        # check the output dimension
        if not y.shape[1] == self.output_dim:
            error_str = "y has dimension %d, should be %d in node %s" % (y.shape[1],
                                                              self.output_dim,
                                                              self.__class__.__name__)
            raise NodeException(error_str)

    def _if_training_stop_training(self):
        if self.is_training():
            self.stop_training()
            # if there is some training phases left we shouldn't be here!
            if self.get_remaining_train_phase() > 0:
                error_str = "The training phases of node %s are not completed yet."%self.__class__.__name__
                raise TrainingException(error_str)

    def _pre_execution_checks(self, x):
        """This method contains all pre-execution checks.

        It can be used when a subclass defines multiple execution methods.
        """
        # if training has not started yet, assume we want to train the node
        if (self.get_current_train_phase() == 0 and
            not self._train_phase_started):
            while True:
                self.train(x)
                if self.get_remaining_train_phase() > 1:
                    self.stop_training()
                else:
                    break

        self._if_training_stop_training()

        # control the dimension x
        self._check_input(x)

        # set the output dimension if necessary
        if self.output_dim is None:
            self.output_dim = self.input_dim

    ### casting helper functions from MDP

    def _refcast(self, x):
        """Helper function to cast arrays to the internal dtype."""
        def refcast(array, dtype):
            """
            Cast the array to dtype only if necessary, otherwise return a reference.

            .. todo:: move to tools?
            """
            dtype = numpy.dtype(dtype)
            if array.dtype == dtype:
                return array
            return array.astype(dtype)
        return refcast(x, self.dtype)

    ### User interface to the overwritten methods
    def execute(self, x, in_training=False, *args, **kwargs):
        """ Process the data contained in 'x'

        If the object is still in the training phase, the function
        'stop_training' will be called.
        'x' is NOT a matrix having different variables on different columns
        and observations on the rows as in MDP.
        'x' is a data type object, which can be a TimeSeries,
        a FeatureVector or a PredictionVector.

        .. note:: This method changes the original MDP implementation.
                  The main difference to the MDP's standard execute method is that here
                  the output_dim of the node is set per default to the size of the
                  node's first result (and not to the size of the input data).
                  Furthermore we have a possible buffering mode for retraining
                  and suppress the setting of the dtype.
        """
        # data buffering for training in live usage
        # needed for the delayed training, which is called
        # by the present_label method
        if hasattr(self, "buffering") and self.buffering and not in_training:
            if not hasattr(self, "data_buffer"):
                self.data_buffer = []
            # no buffering in request_data_for_training
            self.data_buffer.append(x)
        self._training_execution_phase = in_training

        # Additional feature, that standard keywords are replaced in the
        # loading path of the node.
        if self.load_path is not None:
            self.replace_keywords_in_load_path()

#        # if training has not started yet, assume we want to train the node
#        # MDP-SPECIFIC CODE, WHICH SHOULD NOT BE USED ANYMORE
#        if (self.get_current_train_phase() == 0 and
#                                    not self._train_phase_started):
#            while True:
#                self.train(x)
#                if self.get_remaining_train_phase() > 1:
#                    self.stop_training()
#                else:
#                    break
#
#        self._if_training_stop_training()

        # control the dimension x
        self._check_input(x)

        # Do the actual computation
        result = self._execute(self._refcast(x), *args, **kwargs)

        # Make sure key, tag, specs and history are passed
        if x.has_meta():
            result.inherit_meta_from(x)
        else:
            result.generate_meta()

        if self.keep_in_history:
            result.add_to_history(result, self.node_specs) # Append current data to history
        if self.save_intermediate_results:
            self.export_intermediate_results(result)

        # # set the dtype if necessary
        # if self.dtype is None:
        #     self.dtype = result.dtype

        # set the output dimension if necessary
        if self.output_dim is None:
            shape = result.shape
            if len(shape)==1:
                self.output_dim = shape[0]
            else:
                self.output_dim = shape
        elif not (self.output_dim in [result.shape,result.shape[1]]):
            error_str = "y has dimension %d, should be %d in node %s" % (result.shape[1],
                                                              self.output_dim[1],
                                                              self.__class__.__name__)
            raise Exception(error_str)

        return result

    def train(self, x, *args, **kwargs):
        """Update the internal structures according to the input data `x`.

        `x` is a matrix having different variables on different columns
        and observations on the rows.

        By default, subclasses should overwrite `_train` to implement their
        training phase. The docstring of the `_train` method overwrites this
        docstring.

        .. note::
            A subclass supporting multiple training phases should implement
            the *same* signature for all the training phases and document the
            meaning of the arguments in the `_train` method doc-string. Having
            consistent signatures is a requirement to use the node in a node chain.
        """

        if not self.is_trainable():
            raise IsNotTrainableException("The node %s is not trainable."%self.__class__.__name__)

        if not self.is_training():
            err_str = "The training phase of node %s has already finished."%self.__class__.__name__
            raise TrainingFinishedException(err_str)

        self._check_input(x)
        self._check_train_args(x, *args, **kwargs)

        self._train_phase_started = True
        self._train_seq[self._train_phase][0](self._refcast(x), *args, **kwargs)

    def stop_training(self, *args, **kwargs):
        """Stop the training phase.

        By default, subclasses should overwrite `_stop_training` to implement
        this functionality. The docstring of the `_stop_training` method
        overwrites this docstring.
        """
        if self.is_training() and self._train_phase_started == False:
            raise TrainingException("The node %s has not been trained. "%self.__class__.__name__ +
            "Check if you specified training data or a validation scheme (splitter). "+
            "Furthermore you should check the node parameters. "+
            "Did you specify relevant labels correct?")

        if not self.is_training():
            err_str = "The training phase of node %s has already finished."%self.__class__.__name__
            raise TrainingFinishedException(err_str)

        # close the current phase.
        self._train_seq[self._train_phase][1](*args, **kwargs)
        self._train_phase += 1
        self._train_phase_started = False
        # check if we have some training phase left
        if self.get_remaining_train_phase() == 0:
            self._training = False

    def __call__(self, x, *args, **kwargs):
        """Calling an instance of `Node` is equivalent to calling
        its `execute` method."""
        return self.execute(x, *args, **kwargs)

    ###### string representation

    def __str__(self):
        return str(type(self).__name__)

    def __repr__(self):
        # print input_dim, output_dim, dtype
        name = type(self).__name__
        inp = "input_dim=%s" % str(self.input_dim)
        out = "output_dim=%s" % str(self.output_dim)
        if self.dtype is None:
            typ = 'dtype=None'
        else:
            typ = "dtype='%s'" % self.dtype.name
        args = ', '.join((inp, out, typ))
        return name + '(' + args + ')'

    def copy(self, protocol=None):
        """Return a deep copy of the node.

        :param protocol: the pickle protocol (deprecated).

        .. todo:: check if needed
        """
        if protocol is not None:
            warnings.warn("protocol parameter to copy() is ignored")
        return copy.deepcopy(self)

    def save(self, filename, protocol=-1):
        """Save a pickled serialization of the node to `filename`.
        If `filename` is None, return a string.

        Note: the pickled `Node` is not guaranteed to be forwards or
        backwards compatible.

        .. todo:: check if needed
        """
        if filename is None:
            return cPickle.dumps(self, protocol)
        else:
            # if protocol != 0 open the file in binary mode
            mode = 'wb' if protocol != 0 else 'w'
            with open(filename, mode) as flh:
                cPickle.dump(self, flh, protocol)
                
    def getMetadata(self, key):
        return self.input_node.getMetadata(key)


# Specify special node names, different to standard names
_NODE_MAPPING = {"Noop": BaseNode}

