# coding=utf-8
""" NodeChains are sequential orders of :mod:`~pySPACE.missions.nodes`

.. image:: ../../graphics/node_chain.png
   :width: 500

There are two main use cases:

    * the application for :mod:`~pySPACE.run.launch_live` and the
        :mod:`~pySPACE.environments.live` using the default
        :class:`NodeChain` and
    * the benchmarking with :mod:`~pySPACE.run.launch` using
        the :class:`BenchmarkNodeChain` with the
        :mod:`~pySPACE.missions.operations.node_chain` operation.

.. seealso::

    - :mod:`~pySPACE.missions.nodes`
    - :ref:`node_list`
    - :mod:`~pySPACE.missions.operations.node_chain` operation

.. image:: ../../graphics/launch_live.png
   :width: 500

.. todo:: Documentation

This module extends/reimplements the original MDP flow class and
has some additional methods like reset(), save() etc.

Furthermore it supports the construction of NodeChains and
also running them inside nodes in parallel.

MDP is distributed under the following BSD license::

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
import sys
import os

if __name__ == '__main__':
    # add root of the code to system path
    file_path = os.path.dirname(os.path.abspath(__file__))
    pyspace_path = file_path[:file_path.rfind('pySPACE')-1]
    if not pyspace_path in sys.path:
        sys.path.append(pyspace_path)

import cPickle
import gc
import logging
import multiprocessing
import shutil
import socket
import time
import uuid
import yaml
import pySPACE
from pySPACE.tools.filesystem import create_directory
from pySPACE.tools.socket_utils import talk, inform
from pySPACE.tools.conversion import python2yaml, replace_parameters_and_convert, replace_parameters
import copy

import warnings
import traceback
import numpy

class CrashRecoveryException(Exception):
    """Class to handle crash recovery """
    def __init__(self, *args):
        """Allow crash recovery.
        Arguments: (error_string, crashing_obj, parent_exception)
        The crashing object is kept in self.crashing_obj
        The triggering parent exception is kept in ``self.parent_exception``.
        """
        errstr = args[0]
        self.crashing_obj = args[1]
        self.parent_exception = args[2]
        # ?? python 2.5: super(CrashRecoveryException, self).__init__(errstr)
        super(CrashRecoveryException,self).__init__(self, errstr)

    def dump(self, filename = None):
        """
        Save a pickle dump of the crashing object on filename.
        If filename is None, the crash dump is saved on a file created by
        the tempfile module.
        Return the filename.
        """
        import cPickle
        import tempfile
        if filename is None:
            (fd, filename)=tempfile.mkstemp(suffix=".pic", prefix="NodeChainCrash_")
            fl = os.fdopen(fd, 'w+b', -1)
        else:
            fl = open(filename, 'w+b', -1)
        cPickle.dump(self.crashing_obj, fl)
        fl.close()
        return filename

class NodeChainException(Exception):
    """Base class for exceptions in node chains."""
    pass

class NodeChainExceptionCR(CrashRecoveryException, NodeChainException):
    """Class to handle crash recovery """

    def __init__(self, *args):
        """Allow crash recovery.

        Arguments: (error_string, flow_instance, parent_exception)

        The triggering parent exception is kept in self.parent_exception.
        If ``flow_instance._crash_recovery`` is set, save a crash dump of
        flow_instance on the file self.filename
        """
        CrashRecoveryException.__init__(self, *args)
        rec = self.crashing_obj._crash_recovery
        errstr = args[0]
        if rec:
            if isinstance(rec, str):
                name = rec
            else:
                name = None
            name = CrashRecoveryException.dump(self, name)
            dumpinfo = '\nA crash dump is available on: "%s"' % name
            self.filename = name
            errstr = errstr+dumpinfo

        Exception.__init__(self, errstr)


class NodeChain(object):
    """ Reimplement/overwrite mdp.Flow methods e.g., for supervised learning """

    def __init__(self, node_sequence, crash_recovery=False, verbose=False):
        """ Creates the NodeChain based on the node_sequence

        .. note:: The NodeChain cannot be executed before not all trainable
                  nodes have been trained, i.e. self.trained() == True.
        """
        self._check_nodes_consistency(node_sequence)
        self.flow = node_sequence
        self.verbose = verbose
        self.set_crash_recovery(crash_recovery)
        # Register the direct predecessor of a node as its input
        # (i.e. we assume linear flows)
        for i in range(len(node_sequence) - 1):
            node_sequence[i+1].register_input_node(node_sequence[i])

        self.use_test_data = False

        # set a default run number
        self[-1].set_run_number(0)
        # give this flow a unique identifier
        self.id = str(uuid.uuid4())
        self.handler = None
        self.store_intermediate_results = True

    def train(self, data_iterators=None):
        """  Train NodeChain with data from iterator or source node

        The method can proceed in two different ways:

        *   If no data is provided, it is checked that the first node of
            the flow is a source node. If that is the case, the data provided
            by this node is passed forward through the flow. During this
            forward propagation, the flow is trained.
            The request of the data is done in the last node.

        *   If a list of data iterators is provided,
            then it is checked that no source
            and split nodes are contained in the NodeChain.
            these nodes only include already a data handling
            and should not be used, when training is done in different way.
            Furthermore, split nodes are relevant for benchmarking.

            One iterator for each node has to be given.
            If only one is given, or no list, it is mapped to a list
            with the same iterator for each node.

            .. note:: The iterator approach is normally not used in pySPACE,
                      because pySPACE supplies the data with special
                      source nodes and is doing the training automatically
                      without explicit calls on data samples.
                      The approach came with MDP.

            .. todo:: The iterator approach needs some use cases and testings,
                      especially, because it is not used in the normal setting.
        """
        if data_iterators is not None:
            # Check if no source and split nodes are contained in the node chain
            assert(not self[0].is_source_node()), \
                 "Node chains with source nodes cannot be trained "\
                 "with external data_iterators!"
            for node in self:
                assert(not node.is_split_node()), \
                    "Node chains with split nodes cannot be trained "\
                    "with external data_iterators!"
            # prepare iterables
            if not type(data_iterators) == list:
                data_iterators =  [data_iterators] * len(self.flow)
            elif not len(data_iterators)==len(self.flow):
                data_iterators =  [data_iterators] * len(self.flow)
            # Delegate to iterative training
            self.iter_train(data_iterators)
        else: # Use the pySPACE train semantic and not MDP type
            # Check if the first node of the node chain is a source node
            assert(self[0].is_source_node()), \
                 "Training of a node chain without source node requires a "\
                 "data_iterator argument!"
            # Training is accomplished by requesting the iterator
            # of the last node of the chain. This node will recursively call
            # the train method of all its predecessor nodes.
            # As soon as the first element is yielded the node has been trained.
            for _ in self[-1].request_data_for_training(
                    use_test_data=self.use_test_data):
                return

    def iter_train(self, data_iterables):
        """ Train all trainable nodes in the NodeChain with data from iterator

        *data_iterables* is a list of iterables, one for each node in the chain.
        The iterators returned by the iterables must return data arrays that
        are then used for the node training (so the data arrays are the data for
        the nodes).

        Note that the data arrays are processed by the nodes
        which are in front of the node that gets trained, so the data dimension
        must match the input dimension of the first node.

        If a node has only a single training phase then instead of an iterable
        you can alternatively provide an iterator (including generator-type
        iterators). For nodes with multiple training phases this is not
        possible, since the iterator cannot be restarted after the first
        iteration. For more information on iterators and iterables see
        http://docs.python.org/library/stdtypes.html#iterator-types .

        In the special case that *data_iterables* is one single array,
        it is used as the data array *x* for all nodes and training phases.

        Instead of a data array *x* the iterators can also return a list or
        tuple, where the first entry is *x* and the following are args for the
        training of the node (e.g., for supervised training).
        """

        data_iterables = self._train_check_iterables(data_iterables)

        # train each Node successively
        for i in range(len(self.flow)):
            if self.verbose:
                print "Training node #%d (%s)" % (i, str(self.flow[i]))
            self._train_node(data_iterables[i], i)
            if self.verbose:
                print "Training finished"

        self._close_last_node()

    def trained(self):
        """
        Returns whether the complete training is finished, i.e. if all nodes have been trained.
        """
        return self[-1].get_remaining_train_phase() == 0

    def execute(self, data_iterators=None):
        """ Process the data through all nodes """
        if data_iterators is not None:
            # Delegate to super class
            return self.iter_execute(data_iterators)
        else: # Use the evaluate semantic
            # Check if the first node of the flow is a source node
            assert (self[0].is_source_node()), \
                 "Evaluation of a node chain without source node requires a " \
                 "data_iterator argument!"
            # This is accomplished by calling the request_data_for_testing
            # method of the last node of the chain. This node will recursively
            # call the request_data_for_testing method of all its predecessor
            # nodes
            return self[-1].process()

    def iter_execute(self, iterable, nodenr = None):
        """ Process the data through all nodes in the chain till *nodenr*

        'iterable' is an iterable or iterator (note that a list is also an
        iterable), which returns data arrays that are used as input.
        Alternatively, one can specify one data array as input.

        If 'nodenr' is specified, the flow is executed only up to
        node nr. 'nodenr'. This is equivalent to 'flow[:nodenr+1](iterable)'.

        .. note:: In contrary to MDP, results are not concatenated
                  to one big object. Each data object remains separate.
        """
        if isinstance(iterable, numpy.ndarray):
            return self._execute_seq(iterable, nodenr)
        res = []
        empty_iterator = True
        for x in iterable:
            empty_iterator = False
            res.append(self._execute_seq(x, nodenr))
        if empty_iterator:
            errstr = ("The execute data iterator is empty.")
            raise NodeChainException(errstr)
        return res

    def _inc_train(self, data, class_label=None):
        """ Iterate through the nodes to train them """
        for node in self:
            if node.is_retrainable() and not node.buffering and hasattr(node, "_inc_train"):
                if not node.retraining_phase:
                    node.retraining_phase=True
                    node.start_retraining()
                node._inc_train(data,class_label)
            if not (hasattr(self, "buffering") and self.buffering):
                data = node.execute(data)
            else: # workaround to inherit meta data
                self.buffering = False
                data = node.execute(data)
                self.buffering = True

    def save(self, filename, protocol = -1):
        """ Save a pickled representation to *filename*

        If *filename* is None, return a string.

        .. note:: the pickled NodeChain is not guaranteed to be upward or
                    backward compatible.
        .. note:: Having C-Code in the node might cause problems with saving.
                  Therefore, the code has special handling for the
                  LibSVMClassifierNode.
        .. todo:: Intrinsic node methods for storing should be used.
                  .. seealso:: :func:`store_node_chain`
        """
        if self[-1].__class__.__name__ in ["LibSVMClassifierNode"] \
            and self[-1].multinomial:
            indx = filename.find(".pickle")
            if indx != -1:
                self[-1].save_model(filename[0:indx]+'.model')
            else:
                self[-1].save_model(filename+'.model')

        import cPickle

        odict = self.__dict__.copy() # copy the dict since we change it
        # Remove other non-pickable stuff
        remove_keys=[]
        k = 0
        for key, value in odict.iteritems():
            if key == "input_node" or key == "flow":
                continue
            try:
                cPickle.dumps(value)
            except (ValueError, TypeError, cPickle.PicklingError):
                remove_keys.append(key)


        for key in remove_keys:
            odict.pop(key)

        self.__dict__ = odict
        if filename is None:
            return cPickle.dumps(self, protocol)
        else:
            # if protocol != 0 open the file in binary mode
            if protocol != 0:
                mode = 'wb'
            else:
                mode = 'w'

            flh = open(filename , mode)
            cPickle.dump(self, flh, protocol)
            flh.close()

    def get_output_type(self, input_type, as_string=True):
        """
        Returns the output type of the entire node chain

        Recursively iterate over nodes in flow
        """
        output = input_type
        for i in range(len(self.flow)):
            if i == 0:
                output = self.flow[i].get_output_type(
                    input_type, as_string=True)
            else:
                output = self.flow[i].get_output_type(output, as_string=True)

        if as_string:
            return output
        else:
            return self.string_to_class(output)

    @staticmethod
    def string_to_class(string_encoding):
        """ given a string variable, outputs a class instance

        e.g. obtaining a TimeSeries
        """
        from pySPACE.resources.data_types.time_series import TimeSeries
        from pySPACE.resources.data_types.feature_vector import FeatureVector
        from pySPACE.resources.data_types.prediction_vector import PredictionVector
        if "TimeSeries" in string_encoding:
            return TimeSeries
        elif "PredictionVector" in string_encoding:
            return PredictionVector
        elif "FeatureVector" in string_encoding:
            return FeatureVector
        else:
            raise NotImplementedError

#################
# MDP Code copy #


    def _propagate_exception(self, exception, nodenr):
        # capture exception. the traceback of the error is printed and a
        # new exception, containing the identity of the node in the NodeChain
        # is raised. Allow crash recovery.
        (etype, val, tb) = sys.exc_info()
        prev = ''.join(traceback.format_exception(exception.__class__,
                                                   exception,tb))
        act = "\n! Exception in node #%d (%s):\n" % (nodenr,
                                                     str(self.flow[nodenr]))
        errstr = ''.join(('\n', 40*'-', act, 'Node Traceback:\n', prev, 40*'-'))
        raise NodeChainExceptionCR(errstr, self, exception)

    def _train_node(self, data_iterable, nodenr):
        """ Train a single node in the flow.

        nodenr -- index of the node in the flow
        """
        node = self.flow[nodenr]
        if (data_iterable is not None) and (not node.is_trainable()):
            # attempted to train a node although it is not trainable.
            # raise a warning and continue with the next node.
            # wrnstr = "\n! Node %d is not trainable" % nodenr + \
            #        "\nYou probably need a 'None' iterable for"+\
            #         " this node. Continuing anyway."
            #warnings.warn(wrnstr, UserWarning)
            return
        elif (data_iterable is None) and node.is_training():
            # None instead of iterable is passed to a training node
            err_str = ("\n! Node %d is training"
                       " but instead of iterable received 'None'." % nodenr)
            raise NodeChainException(err_str)
        elif (data_iterable is None) and (not node.is_trainable()):
            # skip training if node is not trainable
            return

        try:
            train_arg_keys = self._get_required_train_args(node)
            train_args_needed = bool(len(train_arg_keys))
            ## We leave the last training phase open for the
            ## CheckpointFlow class.
            ## Checkpoint functions must close it explicitly if needed!
            ## Note that the last training_phase is closed
            ## automatically when the node is executed.
            while True:
                empty_iterator = True
                for x in data_iterable:
                    empty_iterator = False
                    # the arguments following the first are passed only to the
                    # currently trained node, allowing the implementation of
                    # supervised nodes
                    if (type(x) is tuple) or (type(x) is list):
                        arg = x[1:]
                        x = x[0]
                    else:
                        arg = ()
                    # check if the required number of arguments was given
                    if train_args_needed:
                        if len(train_arg_keys) != len(arg):
                            err = ("Wrong number of arguments provided by " +
                                   "the iterable for node #%d " % nodenr +
                                   "(%d needed, %d given).\n" %
                                   (len(train_arg_keys), len(arg)) +
                                   "List of required argument keys: " +
                                   str(train_arg_keys))
                            raise NodeChainException(err)
                    # filter x through the previous nodes
                    if nodenr > 0:
                        x = self._execute_seq(x, nodenr-1)
                    # train current node
                    node.train(x, *arg)
                if empty_iterator:
                    if node.get_current_train_phase() == 1:
                        err_str = ("The training data iteration for node "
                                   "no. %d could not be repeated for the "
                                   "second training phase, you probably "
                                   "provided an iterator instead of an "
                                   "iterable." % (nodenr+1))
                        raise NodeChainException(err_str)
                    else:
                        err_str = ("The training data iterator for node "
                                   "no. %d is empty." % (nodenr+1))
                        raise NodeChainException(err_str)
                self._stop_training_hook()
                # close the previous training phase
                node.stop_training()
                if node.get_remaining_train_phase() > 0:
                    continue
                else:
                    break
        except self.flow[-1].TrainingFinishedException, e:
            # attempted to train a node although its training phase is already
            # finished. raise a warning and continue with the next node.
            wrnstr = ("\n! Node %d training phase already finished"
                      " Continuing anyway." % nodenr)
            warnings.warn(wrnstr, UserWarning)
        except NodeChainExceptionCR, e:
            # this exception was already propagated,
            # probably during the execution  of a node upstream in the flow
            (exc_type, val) = sys.exc_info()[:2]
            prev = ''.join(traceback.format_exception_only(e.__class__, e))
            prev = prev[prev.find('\n')+1:]
            act = "\nWhile training node #%d (%s):\n" % (nodenr,
                                                         str(self.flow[nodenr]))
            err_str = ''.join(('\n', 40*'=', act, prev, 40*'='))
            raise NodeChainException(err_str)
        except Exception, e:
            # capture any other exception occurred during training.
            self._propagate_exception(e, nodenr)

    def _stop_training_hook(self):
        """Hook method that is called before stop_training is called."""
        pass

    @staticmethod
    def _get_required_train_args(node):
        """Return arguments in addition to self and x for node.train.

        Arguments that have a default value are ignored.
        """
        import inspect
        train_arg_spec = inspect.getargspec(node.train)
        train_arg_keys = train_arg_spec[0][2:]  # ignore self, x
        if train_arg_spec[3]:
            # subtract arguments with a default value
            train_arg_keys = train_arg_keys[:-len(train_arg_spec[3])]
        return train_arg_keys

    def _train_check_iterables(self, data_iterables):
        """Return the data iterables after some checks and sanitizing.

        Note that this method does not distinguish between iterables and
        iterators, so this must be taken care of later.
        """
        # verifies that the number of iterables matches that of
        # the signal nodes and multiplies them if needed.
        flow = self.flow

        # # if a single array is given wrap it in a list of lists,
        # # note that a list of 2d arrays is not valid
        # if isinstance(data_iterables, numpy.ndarray):
        #     data_iterables = [[data_iterables]] * len(flow)

        if not isinstance(data_iterables, list):
            err_str = ("'data_iterables' must be either a list of "
                       "iterables or an array, but got %s" %
                       str(type(data_iterables)))
            raise NodeChainException(err_str)

        # check that all elements are iterable
        for i, iterable in enumerate(data_iterables):
            if (iterable is not None) and (not hasattr(iterable, '__iter__')):
                err = ("Element number %d in the data_iterables"
                       " list is not an iterable." % i)
                raise NodeChainException(err)

        # check that the number of data_iterables is correct
        if len(data_iterables) != len(flow):
            err_str = ("%d data iterables specified,"
                       " %d needed" % (len(data_iterables), len(flow)))
            raise NodeChainException(err_str)

        return data_iterables

    def _close_last_node(self):
        if self.verbose:
            print "Close the training phase of the last node"
        try:
            self.flow[-1].stop_training()
        except self.flow[-1].TrainingFinishedException:
            pass
        except Exception, e:
            self._propagate_exception(e, len(self.flow)-1)

    def set_crash_recovery(self, state = True):
        """Set crash recovery capabilities.

        When a node raises an Exception during training, execution, or
        inverse execution that the flow is unable to handle, a NodeChainExceptionCR
        is raised. If crash recovery is set, a crash dump of the flow
        instance is saved for later inspection. The original exception
        can be found as the 'parent_exception' attribute of the
        NodeChainExceptionCR instance.

        - If 'state' = False, disable crash recovery.
        - If 'state' is a string, the crash dump is saved on a file
          with that name.
        - If 'state' = True, the crash dump is saved on a file created by
          the tempfile module.
        """
        self._crash_recovery = state

    def _execute_seq(self, x, nodenr = None):
        """ Executes input data 'x' through the nodes 0..'node_nr' included

        If no *nodenr* is specified, the complete node chain is used for
        processing.
        """
        flow = self.flow
        if nodenr is None:
            nodenr = len(flow)-1
        for node_index in range(nodenr+1):
            try:
                x = flow[node_index].execute(x)
            except Exception, e:
                self._propagate_exception(e, node_index)
        return x

    def copy(self, protocol=None):
        """Return a deep copy of the flow.

        The protocol parameter should not be used.
        """
        import copy
        if protocol is not None:
            warnings.warn("protocol parameter to copy() is ignored",
                           DeprecationWarning, stacklevel=2)
        return copy.deepcopy(self)

    def __call__(self, iterable, nodenr = None):
        """Calling an instance is equivalent to call its 'execute' method."""
        return self.iter_execute(iterable, nodenr=nodenr)

    ###### string representation

    def __str__(self):
        nodes = ', '.join([str(x) for x in self.flow])
        return '['+nodes+']'

    def __repr__(self):
        # this should look like a valid Python expression that
        # could be used to recreate an object with the same value
        # eval(repr(object)) == object
        name = type(self).__name__
        pad = len(name)+2
        sep = ',\n'+' '*pad
        nodes = sep.join([repr(x) for x in self.flow])
        return '%s([%s])' % (name, nodes)

    ###### private container methods

    def __len__(self):
        return len(self.flow)

    def _check_dimension_consistency(self, out, inp):
        """Raise ValueError when both dimensions are set and different."""
        if ((out and inp) is not None) and out != inp:
            errstr = "dimensions mismatch: %s != %s" % (str(out), str(inp))
            raise ValueError(errstr)

    def _check_nodes_consistency(self, flow = None):
        """Check the dimension consistency of a list of nodes."""
        if flow is None:
            flow = self.flow
        len_flow = len(flow)
        for i in range(1, len_flow):
            out = flow[i-1].output_dim
            inp = flow[i].input_dim
            self._check_dimension_consistency(out, inp)

    def _check_value_type_isnode(self, value):
        if not isinstance(value, pySPACE.missions.nodes.base.BaseNode):
            raise TypeError("flow item must be Node instance")

    def __getitem__(self, key):
        if isinstance(key, slice):
            flow_slice = self.flow[key]
            self._check_nodes_consistency(flow_slice)
            return self.__class__(flow_slice)
        else:
            return self.flow[key]

    def __setitem__(self, key, value):
        if isinstance(key, slice):
            [self._check_value_type_isnode(item) for item in value]
        else:
            self._check_value_type_isnode(value)

        # make a copy of list
        flow_copy = list(self.flow)
        flow_copy[key] = value
        # check dimension consistency
        self._check_nodes_consistency(flow_copy)
        # if no exception was raised, accept the new sequence
        self.flow = flow_copy

    def __delitem__(self, key):
        # make a copy of list
        flow_copy = list(self.flow)
        del flow_copy[key]
        # check dimension consistency
        self._check_nodes_consistency(flow_copy)
        # if no exception was raised, accept the new sequence
        self.flow = flow_copy

    def __contains__(self, item):
        return self.flow.__contains__(item)

    def __iter__(self):
        return self.flow.__iter__()

    def __add__(self, other):
        # append other to self
        if isinstance(other, NodeChain):
            flow_copy = list(self.flow).__add__(other.flow)
            # check dimension consistency
            self._check_nodes_consistency(flow_copy)
            # if no exception was raised, accept the new sequence
            return self.__class__(flow_copy)
        elif isinstance(other, pySPACE.missions.nodes.base.BaseNode):
            flow_copy = list(self.flow)
            flow_copy.append(other)
            # check dimension consistency
            self._check_nodes_consistency(flow_copy)
            # if no exception was raised, accept the new sequence
            return self.__class__(flow_copy)
        else:
            err_str = ('can only concatenate flow or node'
                       ' (not \'%s\') to flow' % (type(other).__name__))
            raise TypeError(err_str)

    def __iadd__(self, other):
        # append other to self
        if isinstance(other, NodeChain):
            self.flow += other.flow
        elif isinstance(other, pySPACE.missions.nodes.base.BaseNode):
            self.flow.append(other)
        else:
            err_str = ('can only concatenate flow or node'
                       ' (not \'%s\') to flow' % (type(other).__name__))
            raise TypeError(err_str)
        self._check_nodes_consistency(self.flow)
        return self

    ###### public container methods

    def append(self, x):
        """flow.append(node) -- append node to flow end"""
        self[len(self):len(self)] = [x]

    def extend(self, x):
        """flow.extend(iterable) -- extend flow by appending
        elements from the iterable"""
        if not isinstance(x, NodeChain):
            err_str = ('can only concatenate flow'
                       ' (not \'%s\') to flow' % (type(x).__name__))
            raise TypeError(err_str)
        self[len(self):len(self)] = x

    def insert(self, i, x):
        """flow.insert(index, node) -- insert node before index"""
        self[i:i] = [x]

    def pop(self, i = -1):
        """flow.pop([index]) -> node -- remove and return node at index
        (default last)"""
        x = self[i]
        del self[i]
        return x

    def reset(self):
        """ Reset the flow and obey permanent_attributes where available

        Method was moved to the end of class code, due to program environment
        problems which needed the __getitem__ method beforehand.
        """
        for i in range(len(self)):
            self[i].reset()

class BenchmarkNodeChain(NodeChain):
    """ This subclass  overwrites the train method in order
    to provide a more convenient way of doing supervised learning.
    Furthermore, it contains a benchmark method that can be used for
    benchmarking.

    This includes logging, setting of run numbers,
    delivering the result collection, handling of source and sink nodes, ...

    :Author: Jan Hendrik Metzen (jhm@informatik.uni-bremen.de)
    :Created: 2008/08/18
    """

    def __init__(self, node_sequence):
        """ Creates the BenchmarkNodeChain based on the node_sequence """
        super(BenchmarkNodeChain, self).__init__(node_sequence)
        # Each BenchmarkNodeChain must start with an source node
        # and end with a sink node
        assert(self[0].is_source_node()), \
                "A benchmark flow must start with a source node"
        assert(self[-1].is_sink_node()), \
                "A benchmark flow must end with a sink node"

    def use_next_split(self):
        """
        Use the next split of the data into training and test data

        This method is useful for pySPACE-benchmarking
        """
        # This is handled by calling use_next_split() of the last node of
        # the flow which will recursively call predecessor nodes in the flow
        # until a node is found that handles the splitting
        return self[-1].use_next_split()

    def benchmark(self, input_collection, run=0,
                  persistency_directory=None, store_node_chain=False):
        """ Perform the benchmarking of this data flow with the given collection

        Benchmarking is accomplished by iterating through all splits of the
        data into training and test data.

        **Parameters**:

            :input_collection:
                A sequence of data/label-tuples that serves as a generator or a
                BaseDataset which contains the data to be processed.

            :run:
                The current run which defines all random seeds within the flow.

            :persistency_directory:
                Optional information of the nodes as well as the trained node chain
                (if *store_node_chain* is not False) are stored to the given
                *persistency_directory*.

            :store_node_chain:
                If True the trained flow is stored to *persistency_directory*.
                If *store_node_chain* is a tuple of length 2---lets say (i1,i2)--
                only the subflow starting at the i1-th node and ending at the
                (i2-1)-th node is stored. This may be useful when the stored
                flow should be used in an ensemble.
        """
        # Inform the first node of this flow about the input collection
        if hasattr(input_collection,'__iter__'):
            # assume a generator is given
            self[0].set_generator(input_collection)
        else: # assume BaseDataset
            self[0].set_input_dataset(input_collection)

        # Inform all nodes recursively about the number of the current run
        self[-1].set_run_number(int(run))
        # set temp file folder
        if persistency_directory != None:
            self[-1].set_temp_dir(persistency_directory+os.sep+"temp_dir")

        split_counter = 0

        # For every split of the dataset
        while True: # As long as more splits are available
            # Compute the results for the current split
            # by calling the method on its last node
            self[-1].process_current_split()

            if persistency_directory != None:
                if store_node_chain:
                    self.store_node_chain(persistency_directory + os.sep + \
                                "node_chain_sp%s.pickle" % split_counter, store_node_chain)

                # Store nodes that should be persistent
                self.store_persistent_nodes(persistency_directory)

            # If no more splits are available
            if not self.use_next_split():
                break

            split_counter += 1

        # print "Input benchmark"
        # print gc.get_referrers(self[0].collection)

        # During the flow numerous pointers are put to the flow but they are
        # not deleted. So memory is not given free, which can be seen by the
        # upper comment. Therefore we now free the input collection and only
        # then the gc collector can free the memory. Otherwise under not yet
        # found reasons, the pointers to the input collection will remain even
        # between processes.
        if hasattr(input_collection,'__iter__'):
            self[0].set_generator(None)
        else:
            self[0].set_input_dataset(None)
        gc.collect()
        # Return the result collection of this flow
        return self[-1].get_result_dataset()

    def __call__(self, iterable=None, train_instances=None, runs=[]):
        """ Call *execute* or *benchmark* and return (id, PerformanceResultSummary)

        If *iterable* is given, calling an instance is equivalent to call its
        'execute' method.
        If *train_instances* and *runs* are given, 'benchmark' is called for
        every run number specified and results are merged. This is useful for
        e.g. parallel execution of subflows with the multiprocessing module,
        since instance methods can not be serialized in Python but whole objects.
        """
        if iterable != None:
            return self.execute(iterable)
        elif train_instances != None and runs != []: # parallelization case
            # we have to reinitialize logging cause otherwise deadlocks occur
            # when parallelization is done via multiprocessing.Pool
            self.prepare_logging()
            for ind, run in enumerate(runs):
                result = self.benchmark(train_instances, run=run)
                if ind == 0:
                    result_collection = result
                else:
                    result_collection.data.update(result.data)
                # reset node chain for new training if another call of
                # :func:`benchmark` is expected.
                if not ind == len(runs) - 1:
                    self.reset()
            self.clean_logging()
            return (self.id, result_collection)
        else:
            import warnings
            warnings.warn("__call__ methods needs at least one parameter (data)")
            return None

    def store_node_chain(self, result_dir, store_node_chain):
        """ Pickle this flow into *result_dir* for later usage"""
        if isinstance(store_node_chain,basestring):
            store_node_chain = eval(store_node_chain)
        if isinstance(store_node_chain,tuple):
            assert(len(store_node_chain) == 2)
            # Keep only subflow starting at the i1-th node and ending at the
            # (i2-1) node.
            flow = NodeChain(self.flow[store_node_chain[0]:store_node_chain[1]])
        elif isinstance(store_node_chain,list):
            # Keep only nodes with indices contained in the list
            # nodes have to be copied, otherwise input_node-refs of current flow
            # are overwritten
            from copy import copy
            store_node_list = [copy(node) for ind, node in enumerate(self.flow) \
                                                           if ind in store_node_chain]
            flow = NodeChain(store_node_list)
        else:
            # Per default, get rid of source and sink nodes
            flow = NodeChain(self.flow[1:-1])
        input_node = flow[0].input_node
        flow[0].input_node = None
        flow.save(result_dir)

    def prepare_logging(self):
        """ Set up logging

        This method is only needed if one forks subflows, i.e. to execute them
        via multiprocessing.Pool
        """
        # Prepare remote logging
        root_logger = logging.getLogger("%s-%s" % (socket.gethostname(),
                                                   os.getpid()))
        root_logger.setLevel(logging.DEBUG)
        root_logger.propagate = False

        if len(root_logger.handlers)==0:
            self.handler = logging.handlers.SocketHandler(socket.gethostname(),
                                      logging.handlers.DEFAULT_TCP_LOGGING_PORT)
            root_logger.addHandler(self.handler)

    def clean_logging(self):
        """ Remove logging handlers if existing

        Call this method only if you have called *prepare_logging* before.
        """
        # Remove potential logging handlers
        if self.handler is not None:
            self.handler.close()
            root_logger = logging.getLogger("%s-%s" % (socket.gethostname(),
                                            os.getpid()))
            root_logger.removeHandler(self.handler)

    def store_persistent_nodes(self, result_dir):
        """ Store all nodes that should be persistent """
        # For all node
        for index, node in enumerate(self):
            # Store them in the result dir if they enabled storing
            node.store_state(result_dir, index)


class NodeChainFactory(object):
    """ Provide static methods to create and instantiate data flows

    :Author: Jan Hendrik Metzen (jhm@informatik.uni-bremen.de)
    :Created: 2009/01/26
    """

    @staticmethod
    def flow_from_yaml(Flow_Class, flow_spec):
        """ Creates a Flow object

        Reads from the given *flow_spec*, which should be a valid YAML
        specification of a NodeChain object, and returns this dataflow
        object.

        **Parameters**

            :Flow_Class:
                The class name of node chain to create. Valid are 'NodeChain' and
                'BenchmarkNodeChain'.

            :flow_spec:
                A valid YAML specification stream; this could be a file object,
                a string representation of the YAML file or the Python
                representation of the YAML file (list of dicts)
        """
        from pySPACE.missions.nodes.base_node import BaseNode
        # Reads and parses the YAML file if necessary
        if type(flow_spec) != list:
            dataflow_spec = yaml.load(flow_spec)
        else:
            dataflow_spec = flow_spec
        node_sequence = []
        # For all nodes of the flow
        for node_spec in dataflow_spec:
            # Use factory method to create node
            node_obj = BaseNode.node_from_yaml(node_spec)

            # Append this node to the sequence of node
            node_sequence.append(node_obj)

        # Check if the nodes have to cache their outputs
        for index, node in enumerate(node_sequence):
            # If a node is trainable, it uses the outputs of its input node
            # at least twice, so we have to cache.
            if node.is_trainable():
                node_sequence[index - 1].set_permanent_attributes(caching = True)
            # Split node might also request the data from their input nodes
            # (once for each split), depending on their implementation. We
            # assume the worst case and activate caching
            if node.is_split_node():
                node_sequence[index - 1].set_permanent_attributes(caching = True)

        # Create the flow based on the node sequence and the given flow class
        # and return it
        return Flow_Class(node_sequence)

    @staticmethod
    def instantiate(template, parametrization):
        """ Instantiate a template recursively for the given parameterization

        Instantiate means to replace the parameter in the template by the
        chosen value.

        **Parameters**

        :template:
            A dictionary with key-value pairs, where values might contain
            parameter keys which have to be replaced. A typical example of a
            template would be a Python representation of a node read from YAML.

        :parametrization:
            A dictionary with parameter names as keys and exact one value for
            this parameter as value.

        """
        instance = {}
        for key, value in template.iteritems():
            if value in parametrization.keys():  # Replacement
                instance[key] = parametrization[value]
            elif isinstance(value, dict):  # Recursive call
                instance[key] = NodeChainFactory.instantiate(value, parametrization)
            elif isinstance(value, basestring):  # String replacement
                for param_key, param_value in parametrization.iteritems():
                    try:
                        value = value.replace(param_key, repr(param_value))
                    except:
                        value = value.replace(param_key, python2yaml(param_value))
                instance[key] = value
            elif hasattr(value, "__iter__"):
                # Iterate over all items in sequence
                instance[key] = []
                for iter_item in value:
                    if iter_item in parametrization.keys():  # Replacement
                        instance[key].append(parametrization[iter_item])
                    elif isinstance(iter_item, dict):
                        instance[key].append(NodeChainFactory.instantiate(
                            iter_item, parametrization))
                    elif isinstance(value, basestring): # String replacement
                        for param_key, param_value in parametrization.iteritems():
                            try:
                                iter_item = iter_item.replace(param_key,
                                                              repr(param_value))
                            except:
                                iter_item = iter_item.replace(
                                    param_key, python2yaml(param_value))
                        instance[key] = value
                    else:
                        instance[key].append(iter_item)
            else: # Not parameterized
                instance[key] = value
        return instance

    @staticmethod
    def replace_parameters_in_node_chain(node_chain_template, parametrization):
        node_chain_template = copy.copy(node_chain_template)
        if parametrization == {}:
            return node_chain_template
        elif type(node_chain_template) == list:
            return [NodeChainFactory.instantiate(
                template=node,parametrization=parametrization)
                for node in node_chain_template]
        elif isinstance(node_chain_template, basestring):
            node_chain_template = \
                replace_parameters(node_chain_template, parametrization)
        return node_chain_template

class SubflowHandler(object):
    """ Interface for nodes to generate and execute subflows (subnode-chains)

    A subflow means a node chain used inside a node for processing data.

    This class provides functions that can be used by nodes to generate and
    execute subflows. It serves thereby as a communication daemon to the
    backend (if it is used).

    Most important when inheriting from this class is that the subclass MUST be
    a node. The reason is that this class uses node functionality, e.g. logging,
    the *temp_dir*-variable and so on.

    **Parameters**

        :processing_modality:
            One of the valid strings: 'backend', 'serial', 'local'.

                :backend:
                    The current backends modality is used. This is implemented
                    at the moment only for 'LoadlevelerBackend' and 'LocalBackend'.

                :serial:
                    All subflows are executed sequentially, i.e. one after the
                    other.

                :local:
                    Subflows are executed in a Pool using *pool_size* cpus. This
                    may be also needed when no backend is used.

            (*optional, default: 'serial'*)

        :pool_size:
            If a parallelization is based on using several processes on a local
            system in parallel, e.g. option 'backend' and
            :class:`pySPACEMulticoreBackend`
            or option
            'local', the number of worker processes for subflow evaluation has
            to be specified.

            .. note:: When using the LocalBackend, there is also the possibility
                      to specify the pool size of parallel executed
                      processes, e.g. data sets. Your total number of cpu's
                      should be pool size (pySPACE) + pool size (subflows).

            (*optional, default: 2*)

        :batch_size:
            If parallelization of subflow execution is done together with the
            :class:`~pySPACE.environments.backends.ll_backend.LoadLevelerBackend`,
            *batch_size* determines how many subflows are executed in one
            serial LoadLeveler job. This option is useful if execution of a
            single subflow is really short (range of seconds) since there is
            significant overhead in creating new jobs.

            (*optional, default: 1*)

    :Author: Anett Seeland (anett.seeland@dfki.de)
    :Created: 2012/09/04
    :LastChange: 2012/11/06 batch_size option added
    """
    def __init__(self, processing_modality='serial', pool_size=2, batch_size=1,
                 **kwargs):
        self.modality = processing_modality
        self.pool_size = int(pool_size)
        self.batch_size = int(batch_size)
        # a flag to send pool_size / batch_size only once to the backend
        self.already_send = False
        self.backend_com = None
        self.backend_name = None
        # to indicate the end of a message received over a socket
        self.end_token = '!END!'

        if processing_modality not in ["serial", "local", "backend"]:
            import warnings
            warnings.warn("Processing modality not found! Serial mode is used!")
            self.modality = 'serial'

    @staticmethod
    def generate_subflow(flow_template, parametrization=None, flow_class=None):
        """ Return a *flow_class* object of the given *flow_template*

        This methods wraps two function calls (NodeChainFactory.instantiate and
        NodeChainFactory.flow_from_yaml.

        **Parameters**

            :flow_template:
                List of dicts - a valid representation of a node chain.
                Alternatively, a YAML-String representation could be used,
                which simplifies parameter replacement.

            :parametrization:
                A dictionary with parameter names as keys and exact one value for
                this parameter as value. Passed to NodeChainFactory.instantiate

                (*optional, default: None*)

            :flow_class:
                The flow class name of which an object should be returned

                (*optional, default: BenchmarkNodeChain*)

        """
        if flow_class is None:
            flow_class = BenchmarkNodeChain
        flow_spec = NodeChainFactory.replace_parameters_in_node_chain(
            flow_template,parametrization)
        # create a new Benchmark flow
        flow = NodeChainFactory.flow_from_yaml(flow_class, flow_spec)

        return flow

    def execute_subflows(self, train_instances, subflows, run_numbers=None):
        """ Execute subflows and return result collection.

        **Parameters**
            :training_instances:
                List of training instances which should be used to execute
                *subflows*.

            :subflows:
                List of BenchmarkNodeChain objects.

                ..note:: Note that every subflow object is stored in memory!

            :run_numbers:
                All subflows will be executed with every run_number specified in
                this list. If None, the current self.run_number (from the node
                class) is used.

                (*optional, default: None*)
        """
        if run_numbers == None:
            run_numbers = [self.run_number]
        # in case of serial backend, modality is mapped to serial
        # in the other case communication must be set up and
        # jobs need to be submitted to backend
        if self.modality == 'backend':
            self.backend_com = pySPACE.configuration.backend_com
            if not self.backend_com is None:
                # ask for backend_name
                # create a socket and keep it alive as long as possible since
                # handshaking costs really time
                client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                client_socket.connect(self.backend_com)
                client_socket, self.backend_name = talk('name' + self.end_token,
                client_socket, self.backend_com)
            else:
                import warnings #necessary for serial backend!
                warnings.warn("Seems that no backend is used! Modality of subflow execution "\
                "has to be specified! Assuming serial backend.")
                self.backend_name = 'serial'
            self._log("Preparing subflows for backend execution.")
            if self.backend_name in ['loadl','mcore'] :
                # we have to pickle training instances and store it on disk
                store_path = os.path.join(self.temp_dir,
                                                    "sp%d" % self.current_split)
                create_directory(store_path)
                filename = os.path.join(store_path, "subflow_data.pickle")
                if not os.path.isfile(filename):
                    cPickle.dump(train_instances, open(filename,'wb'),
                                              protocol=cPickle.HIGHEST_PROTOCOL)
                subflows_to_compute = [subflows[ind].id for ind in \
                                                           range(len(subflows))]
                if self.backend_name == 'loadl':
                    # send batch_size to backend if not already done
                    if not self.already_send:
                        client_socket = inform("subflow_batchsize;%d%s" % \
                                            (self.batch_size, self.end_token),
                                            client_socket, self.backend_com)
                        self.already_send = True
                    for subflow in subflows:
                        cPickle.dump(subflow, open(os.path.join(store_path,
                                                         subflow.id+".pickle"),"wb"),
                                     protocol=cPickle.HIGHEST_PROTOCOL)
                    send_flows = subflows_to_compute
                else: # backend_name == mcore
                    # send pool_size to backend if not already done
                    if not self.already_send:
                        client_socket = inform("subflow_poolsize;%d%s" % \
                                            (self.pool_size, self.end_token),
                                            client_socket, self.backend_com)
                        self.already_send = True
                    # send flow objects via socket
                    send_flows = [cPickle.dumps(subflow, cPickle.HIGHEST_PROTOCOL) \
                                  for subflow in subflows]
                # inform backend
                client_socket,msg  = talk('execute_subflows;%s;%d;%s;%s%s' % \
                                   (store_path, len(subflows), str(send_flows),
                                    str(run_numbers), self.end_token),
                                               client_socket, self.backend_com)
                time.sleep(10)

                not_finished_subflows = set(subflows_to_compute)
                while len(not_finished_subflows) != 0:
                    # ask backend for finished jobs
                    client_socket, msg = talk('is_ready;%d;%s%s' % \
                            (len(not_finished_subflows), str(not_finished_subflows),
                             self.end_token), client_socket, self.backend_com)
                    # parse message
                    finished_subflows = eval(msg) #should be a set
                    # set difference
                    not_finished_subflows -= finished_subflows
                    time.sleep(10)

                if self.backend_name == 'loadl':
                    # read results and delete store_dir
                    result_pattern = os.path.join(store_path, '%s_result.pickle')
                    result_collections = [cPickle.load(open(result_pattern % \
                        subflows[ind].id,'rb')) for ind in range(len(subflows))]
                    # ..todo:: check if errors have occurred and if so do not delete!
                    shutil.rmtree(store_path)
                else: # backend_name == mcore
                    # ask backend to send results
                    client_socket, msg = talk("send_results;%s!END!" % \
                            subflows_to_compute, client_socket, self.backend_com)
                    # should be a list of collections
                    results = eval(msg)
                    result_collections = [cPickle.loads(result) for result in results]
                self._log("Finished subflow execution.")
                client_socket.shutdown(socket.SHUT_RDWR)
                client_socket.close()
                return result_collections
            elif self.backend_name == 'serial':
                # do the same as modality=='serial'
                self.modality = 'serial'
            else: # e.g. mpi backend    :
                import warnings
                warnings.warn("Subflow Handling with %s backend not supported,"\
                              " serial-modality is used!" % self.backend_name)
                self.modality = 'serial'
        if self.modality == 'serial':
            # serial execution
            # .. note:: the here executed flows can not store anything.
            #           meta data of result collection is NOT updated!
            results = [subflow(train_instances=train_instances,
                               runs=run_numbers) for subflow in subflows]
            result_collections = [result[1] for result in results]
            return result_collections
        else: # modality local, e.g. usage without backend in application case
            self._log("Subflow Handler starts processes in pool.")
            pool = multiprocessing.Pool(processes=self.pool_size)
            results = [pool.apply_async(func=subflow,
                                        kwds={"train_instances": train_instances,
                                              "runs": run_numbers}) \
                       for subflow in subflows]
            pool.close()
            self._log("Waiting for parallel processes to finish.")
            pool.join()
            result_collections = [result.get()[1] for result in results]
            del pool
            return result_collections
