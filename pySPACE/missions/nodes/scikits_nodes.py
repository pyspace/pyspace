# -*- coding:utf-8; -*-
""" Wrap the algorithms defined in `scikits.learn <http://scikit-learn.org/>`_ in pySPACE nodes

For details on parameter usage look at the
`scikits documentation <http://scikit-learn.org/>`_ or
the wrapped documentation of pySPACE: :ref:`scikit_nodes`.
The parameters given in the node specification are filtered, to check if they
are available, and then directly forwarded to the scikit algorithm.

This module is based heavily on the scikits.learn wrapper for the "Modular
toolkit for Data Processing"
(MDP, version 3.3, http://mdp-toolkit.sourceforge.net/).
All credit goes to the MDP authors.

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
__docformat__ = "restructuredtext en"

try:
    import sklearn
    _sklearn_prefix = 'sklearn'
except ImportError:
    try:
        import scikits.learn as sklearn
        _sklearn_prefix = 'scikits.learn'
    except ImportError:
        _sklearn_prefix = False

import inspect
import re

import numpy
import logging
import warnings
import sys

from pySPACE.missions.nodes.base_node import BaseNode
from pySPACE.missions.nodes import NODE_MAPPING, DEFAULT_NODE_MAPPING
from pySPACE.resources.data_types.prediction_vector import PredictionVector
from pySPACE.resources.data_types.feature_vector import FeatureVector


class ScikitsException(Exception):
    """Base class for exceptions in nodes wrapping scikits algorithms."""
    pass


# import all submodules of sklearn (to work around lazy import)
def _version_too_old(version, known_good):
    """ version comparison """
    for part,expected in zip(version.split('.'), known_good):
        try:
            p = int(part)
        except ValueError:
            return None
        if p < expected:
            return True
        if p > expected:
            break
    return False

if not _sklearn_prefix:
    scikits_modules = []
elif _version_too_old(sklearn.__version__, (0, 8)):
    scikits_modules = ['ann', 'cluster', 'covariance', 'feature_extraction',
                       'feature_selection', 'features', 'gaussian_process', 'glm',
                       'linear_model', 'preprocessing', 'svm',
                       'pca', 'lda', 'hmm', 'fastica', 'grid_search', 'mixture',
                       'naive_bayes', 'neighbors', 'qda']
elif _version_too_old(sklearn.__version__, (0, 9)):
    # package structure has been changed in 0.8
    scikits_modules = ['svm', 'linear_model', 'naive_bayes', 'neighbors',
                       'mixture', 'hmm', 'cluster', 'decomposition', 'lda',
                       'covariance', 'cross_val', 'grid_search',
                       'feature_selection.rfe', 'feature_extraction.image',
                       'feature_extraction.text', 'pipelines', 'pls',
                       'gaussian_process', 'qda']
elif _version_too_old(sklearn.__version__, (0, 11)):
    # from release 0.9 cross_val becomes cross_validation and hmm is deprecated
    scikits_modules = ['svm', 'linear_model', 'naive_bayes', 'neighbors',
                       'mixture', 'cluster', 'decomposition', 'lda',
                       'covariance', 'cross_validation', 'grid_search',
                       'feature_selection.rfe', 'feature_extraction.image',
                       'feature_extraction.text', 'pipelines', 'pls',
                       'gaussian_process', 'qda', 'ensemble', 'manifold',
                       'metrics', 'preprocessing', 'tree']
else:
    scikits_modules = ['svm', 'linear_model', 'naive_bayes', 'neighbors',
                       'mixture', 'cluster', 'decomposition', 'lda',
                       'covariance', 'cross_validation', 'grid_search',
                       'feature_selection', 'feature_extraction',
                       'pipeline', 'pls', 'gaussian_process', 'qda',
                       'ensemble', 'manifold', 'metrics', 'preprocessing',
                       'semi_supervised', 'tree', 'hmm']


for name in scikits_modules:
    # not all modules may be available due to missing dependencies
    # on the user system.
    # we just ignore failing imports
    try:
        __import__(_sklearn_prefix + '.' + name)
    except ImportError:
        pass


_WS_LINE_RE = re.compile(r'^\s*$')
_WS_PREFIX_RE = re.compile(r'^(\s*)')
_HEADINGS_RE = re.compile(r'''^(Parameters|Attributes|Methods|Examples|Notes)\n
                           (----+|====+)''', re.M + re.X)
_UNDERLINE_RE = re.compile(r'----+|====+')
_VARWITHUNDER_RE = re.compile(r'(\s|^)([a-zA-Z_][a-zA-Z0-9_]*_)(\s|$|[,.])')

_HEADINGS = set(['Parameters', 'Attributes', 'Methods', 'Examples',
                 'Notes', 'References'])

_DOC_TEMPLATE = """
%s

This node has been automatically generated by wrapping the ``%s.%s`` class
from the ``sklearn`` library.  The wrapped instance can be accessed
through the ``scikits_alg`` attribute.

%s
"""


def _gen_docstring(object, docsource=None):
    """ Generate and modify the docstring for each wrapped node """
    module = object.__module__
    name = object.__name__
    if docsource is None:
        docsource = object
    docstring = docsource.__doc__
    if docstring is None:
        return None

    lines = docstring.strip().split('\n')
    for i, line in enumerate(lines):
        if _WS_LINE_RE.match(line):
            break
    header = [line.strip() for line in lines[:i]]

    therest = [line.rstrip() for line in lines[i + 1:]]
    body = []

    if therest:
        prefix = min(len(_WS_PREFIX_RE.match(line).group(1))
                     for line in therest if line)
        quoteind = None
        for i, line in enumerate(therest):
            line = line[prefix:]
            if line in _HEADINGS:
                body.append('**%s**' % line)
            elif _UNDERLINE_RE.match(line):
                body.append('')
            else:
                line = _VARWITHUNDER_RE.sub(r'\1``\2``\3', line)
                if quoteind:
                    if len(_WS_PREFIX_RE.match(line).group(1)) >= quoteind:
                        line = quoteind * ' ' + '- ' + line[quoteind:]
                    else:
                        quoteind = None
                        body.append('')
                body.append(line)

            if line.endswith(':'):
                body.append('')
                if i + 1 < len(therest):
                    next = therest[i + 1][prefix:]
                    quoteind = len(_WS_PREFIX_RE.match(next).group(1))

    return _DOC_TEMPLATE % ('\n'.join(header), module, name, '\n'.join(body))

# TODO: generalize dtype support
# TODO: have a look at predict_proba for Classifier.prob
# TODO: inverse <-> generate/rvs
# TODO: deal with input_dim/output_dim
# TODO: change signature of overwritten functions
# TODO: wrap_scikits_instance
# TODO: add sklearn availability to test info strings
# TODO: which tests ? (test that particular algorithm are / are not trainable)
# XXX: if class defines n_components, allow output_dim, otherwise throw exception
#      also for classifiers (overwrite _set_output_dim)
#      Problem: sometimes they call it 'k' (e.g., algorithms in sklearn.cluster)


def apply_to_scikits_algorithms(current_module, action,
                                processed_modules=None,
                                processed_classes=None):
    """ Function that traverses a module to find scikits algorithms.

    'sklearn' algorithms are identified by the 'fit' 'predict',
    or 'transform' methods. The 'action' function is applied to each found
    algorithm.

    action -- a function that is called with as ``action(class_)``, where
              ``class_`` is a class that defines the 'fit' or 'predict' method
    """

    # only consider modules and classes once
    if processed_modules is None:
        processed_modules = []
    if processed_classes is None:
        processed_classes = []

    if current_module in processed_modules:
        return
    processed_modules.append(current_module)

    for member_name, member in current_module.__dict__.items():
        if not member_name.startswith('_'):
            # classes
            if inspect.isclass(member) and member not in processed_classes:
                if ((hasattr(member, 'fit')
                        or hasattr(member, 'predict')
                        or hasattr(member, 'transform'))
                        and not member.__module__.endswith('_')):
                    processed_classes.append(member)
                    action(member)

            # other modules
            elif (inspect.ismodule(member) and
                  member.__name__.startswith(_sklearn_prefix)):
                apply_to_scikits_algorithms(member, action, processed_modules,
                                            processed_classes)
    return processed_classes


_OUTPUTDIM_ERROR = """'output_dim' keyword not supported.

Please set the output dimensionality using sklearn keyword
arguments (e.g., 'n_components', or 'k'). See the docstring of
this class for details."""


def wrap_scikits_classifier(scikits_class):
    """Wrap a sklearn classifier as a BaseNode subclass.

    The wrapper maps these node methods to their sklearn equivalents:

    - _stop_training -> fit
    - _execute -> predict
    """

    newaxis = numpy.newaxis

    # create a wrapper class for a sklearn classifier
    class ScikitsClassifier(BaseNode):

        def __init__(self, input_dim=None, output_dim=None, dtype=None,
                     class_labels=None, **kwargs):

            if output_dim is not None:
                # output_dim and n_components cannot be defined at the same time
                if 'n_components' in kwargs:
                    msg = ("Dimensionality set both by "
                           "output_dim=%d and n_components=%d""")
                    raise ScikitsException(msg % (output_dim,
                                                  kwargs['n_components']))

            super(ScikitsClassifier, self).__init__(input_dim=input_dim,
                                                    output_dim=output_dim,
                                                    dtype=dtype)
            try:
                accepted_args = inspect.getargspec(scikits_class.__init__)[0]
                for key in kwargs.keys():
                    if key not in accepted_args:
                        kwargs.pop(key)
            except TypeError:  # happens for GaussianNBSklearnNode
                kwargs = {}
            self.kwargs = kwargs

            self.set_permanent_attributes(kwargs=kwargs,
                                          scikits_alg=scikits_class(**self.kwargs),
                                          data=[],
                                          labels=[],
                                          class_labels=class_labels)

        # ---- re-direct training and execution to the wrapped algorithm

        def _train(self, data, y):
            x = data.view(numpy.ndarray)
            self.data.append(x[0])
            self.labels.append(y)

        def _stop_training(self, **kwargs):
            super(ScikitsClassifier, self)._stop_training(self)
            if self.class_labels is None:
                self.class_labels = sorted(list(set(self.labels)))

            data = numpy.array(self.data)
            label_values = \
                numpy.array(map(lambda s: self.class_labels.index(s),
                                self.labels))
            try:
                return self.scikits_alg.fit(data, label_values, **kwargs)
            except Exception as e:
                raise type(e), \
                type(e)("in node %s:\n\t"%self.__class__.__name__+e.args[0]),\
                sys.exc_info()[2]

        def _execute(self, data):
            x = data.view(numpy.ndarray)
            try:
                prediction = self.scikits_alg.predict(x)[0]
            except Exception as e:
                raise type(e), \
                type(e)("in node %s:\n\t"%self.__class__.__name__+e.args[0]), \
                sys.exc_info()[2]
            if hasattr(self.scikits_alg, "predict_proba"):
                try:
                    score = self.scikits_alg.predict_proba(x)[0, 1]
                except Exception as e:
                    warnings.warn("%s in node %s:\n\t"\
                            %(type(e).__name__,self.__class__.__name__)+e.args[0])
                    try:
                        score = self.scikits_alg.decision_function(x)[0]
                    except:
                        score = prediction
            elif hasattr(self.scikits_alg, "decision_function"):
                score = self.scikits_alg.decision_function(x)[0]
            else:
                score = prediction
            label = self.class_labels[prediction]
            return PredictionVector(label=label, prediction=score,
                                    predictor=self)

        # ---- administrative details

        @staticmethod
        def is_trainable():
            """Return True if the node can be trained, False otherwise."""
            return hasattr(scikits_class, 'fit')

        @staticmethod
        def is_supervised():
            """Return True if the node requires labels for training, False otherwise."""
            return True

        # NOTE: at this point scikits nodes can only support up to
        # 64-bits floats because some call numpy.linalg.svd, which for
        # some reason does not support higher precisions
        def _get_supported_dtypes(self):
            """Return the list of dtypes supported by this node.
            The types can be specified in any format allowed by numpy.dtype."""
            return ['float32', 'float64']

    # modify class name and docstring
    ScikitsClassifier.__name__ = scikits_class.__name__ + 'SklearnNode'
    ScikitsClassifier.__doc__ = _gen_docstring(scikits_class)

    # Class must be permanently accessible from module level
    globals()[ScikitsClassifier.__name__] = ScikitsClassifier

    # change the docstring of the methods to match the ones in sklearn

    # methods_dict maps ScikitsNode method names to sklearn method names
    methods_dict = {'__init__': '__init__',
                    'stop_training': 'fit',
                    'execute': 'predict'}
    #if hasattr(scikits_class, 'predict_proba'):
    #    methods_dict['prob'] = 'predict_proba'

    for pyspace_name, scikits_name in methods_dict.items():
        pyspace_method = getattr(ScikitsClassifier, pyspace_name)
        scikits_method = getattr(scikits_class, scikits_name)
        if hasattr(scikits_method, 'im_func'):
            # some scikits algorithms do not define an __init__ method
            # the one inherited from 'object' is a
            # "<slot wrapper '__init__' of 'object' objects>"
            # which does not have a 'im_func' attribute
            pyspace_method.im_func.__doc__ = _gen_docstring(scikits_class,
                                                           scikits_method.im_func)

    if scikits_class.__init__.__doc__ is None:
        ScikitsClassifier.__init__.im_func.__doc__ = _gen_docstring(scikits_class)

    return ScikitsClassifier


def wrap_scikits_transformer(scikits_class):
    """ Wrap a sklearn transformer as a pySPACE BaseNode subclass

    The wrapper maps these pySPACE methods to their sklearn equivalents:

    - _stop_training -> fit
    - _execute -> transform
    """

    # create a wrapper class for a sklearn transformer
    class ScikitsTransformer(BaseNode):

        def __init__(self, input_dim=None, output_dim=None, dtype=None, **kwargs):
            if output_dim is not None:
                raise ScikitsException(_OUTPUTDIM_ERROR)
            super(ScikitsTransformer, self).__init__(input_dim=input_dim,
                                                     output_dim=output_dim,
                                                     dtype=dtype)
            accepted_args = inspect.getargspec(scikits_class.__init__)[0]
            for key in kwargs.keys():
                if key not in accepted_args:
                    kwargs.pop(key)
            self.kwargs = kwargs

            self.set_permanent_attributes(kwargs=kwargs,
                                          scikits_alg=scikits_class(**self.kwargs),
                                          data=[],
                                          feature_names=None)

        # ---- re-direct training and execution to the wrapped algorithm

        def _train(self, data):
            assert type(data) == FeatureVector, \
                "Scikits-Learn Transformer nodes only support FeatureVector inputs."
            x = data.view(numpy.ndarray)
            self.data.append(x[0])

        def _stop_training(self, **kwargs):
            super(ScikitsTransformer, self)._stop_training(self)
            data = numpy.array(self.data)
            return self.scikits_alg.fit(data, **kwargs)

        def _execute(self, data):
            x = data.view(numpy.ndarray)
            out = self.scikits_alg.transform(x[0])
            if self.feature_names is None:
                self.feature_names = \
                    ["%s_%s" % (self.__class__.__name__, i)
                        for i in range(out.shape[1])]
            return FeatureVector(out, self.feature_names)

        # ---- administrative details

        @staticmethod
        def is_trainable():
            """Return True if the node can be trained, False otherwise."""
            return hasattr(scikits_class, 'fit')

        @staticmethod
        def is_supervised():
            """Return True if the node requires labels for training, False otherwise."""
            return False

        # NOTE: at this point scikits nodes can only support up to
        # 64-bits floats because some call numpy.linalg.svd, which for
        # some reason does not support higher precisions
        def _get_supported_dtypes(self):
            """Return the list of dtypes supported by this node.
            The types can be specified in any format allowed by numpy.dtype."""
            return ['float32', 'float64']

    # modify class name and docstring
    ScikitsTransformer.__name__ = scikits_class.__name__ + 'SklearnNode'
    ScikitsTransformer.__doc__ = _gen_docstring(scikits_class)

    # Class must be permanently accessible from module level
    globals()[ScikitsTransformer.__name__] = ScikitsTransformer


    # change the docstring of the methods to match the ones in sklearn

    # methods_dict maps ScikitsNode method names to sklearn method names
    methods_dict = {'__init__': '__init__',
                    'stop_training': 'fit',
                    'execute': 'transform'}

    for pyspace_name, scikits_name in methods_dict.items():
        pyspace_method = getattr(ScikitsTransformer, pyspace_name)
        scikits_method = getattr(scikits_class, scikits_name, None)
        if hasattr(scikits_method, 'im_func'):
            # some scikits algorithms do not define an __init__ method
            # the one inherited from 'object' is a
            # "<slot wrapper '__init__' of 'object' objects>"
            # which does not have a 'im_func' attribute
            pyspace_method.im_func.__doc__ = _gen_docstring(scikits_class,
                                                            scikits_method.im_func)

    if scikits_class.__init__.__doc__ is None:
        ScikitsTransformer.__init__.im_func.__doc__ = _gen_docstring(scikits_class)
    return ScikitsTransformer


def wrap_scikits_predictor(scikits_class):
    """ Wrap a sklearn predictor as an pySPACE BaseNode subclass

    The wrapper maps these pySPACE methods to their sklearn equivalents:

    * _stop_training -> fit
    * _execute -> predict
    """

    # create a wrapper class for a sklearn predictor
    class ScikitsPredictor(BaseNode):

        def __init__(self, input_dim=None, output_dim=None, dtype=None, **kwargs):
            if output_dim is not None:
                raise ScikitsException(_OUTPUTDIM_ERROR)
            super(ScikitsPredictor, self).__init__(input_dim=input_dim,
                                                   output_dim=output_dim,
                                                   dtype=dtype)
            accepted_args = inspect.getargspec(scikits_class.__init__)[0]
            for key in kwargs.keys():
                if key not in accepted_args:
                    kwargs.pop(key)
            self.kwargs = kwargs

            self.set_permanent_attributes(kwargs=kwargs,
                                          scikits_alg=scikits_class(**self.kwargs))

        # ---- re-direct training and execution to the wrapped algorithm

        def _train(self, data, y):
            x = data.view(numpy.ndarray)
            self.data.append(x[0])
            self.labels.append(y)

        def _stop_training(self, **kwargs):
            super(ScikitsPredictor, self)._stop_training(self)
            data = numpy.array(self.data)
            label_values = numpy.array(self.labels)
            try:
                return self.scikits_alg.fit(data, label_values, **kwargs)
            except Exception as e:
                raise type(e), \
                    type(e)("in node %s:\n\t"%self.__class__.__name__+e.args[0]), \
                    sys.exc_info()[2]

        def _execute(self, data):
            x = data.view(numpy.ndarray)
            try:
                prediction = self.scikits_alg.predict(x)[0]
            except Exception as e:
                raise type(e), \
                    type(e)("in node %s:\n\t"%self.__class__.__name__+e.args[0]), \
                    sys.exc_info()[2]
            if hasattr(self.scikits_alg, "predict_proba"):
                try:
                    score = self.scikits_alg.predict_proba(x)[0, 1]
                except Exception as e:
                    warnings.warn("%s in node %s:\n\t" \
                                  %(type(e).__name__,self.__class__.__name__)+e.args[0])
                    try:
                        score = self.scikits_alg.decision_function(x)[0]
                    except:
                        score = prediction
            elif hasattr(self.scikits_alg, "decision_function"):
                score = self.scikits_alg.decision_function(x)[0]
            else:
                score = prediction
            label = self.class_labels[prediction]
            return PredictionVector(label=label, prediction=score,
                                    predictor=self)
        # ---- administrative details

        @staticmethod
        def is_trainable():
            """Return True if the node can be trained, False otherwise."""
            return hasattr(scikits_class, 'fit')

        # NOTE: at this point scikits nodes can only support up to 64-bits floats
        # because some call numpy.linalg.svd, which for some reason does not
        # support higher precisions
        def _get_supported_dtypes(self):
            """Return the list of dtypes supported by this node.
            The types can be specified in any format allowed by numpy.dtype."""
            return ['float32', 'float64']

    # modify class name and docstring
    ScikitsPredictor.__name__ = scikits_class.__name__ + 'SklearnNode'
    ScikitsPredictor.__doc__ = _gen_docstring(scikits_class)

    # Class must be permanently accessible from module level
    globals()[ScikitsPredictor.__name__] = ScikitsPredictor

    # change the docstring of the methods to match the ones in sklearn

    # methods_dict maps ScikitsPredictor method names to sklearn method names
    methods_dict = {'__init__': '__init__',
                    'stop_training': 'fit',
                    'execute': 'predict'}

    for pyspace_name, scikits_name in methods_dict.items():
        pyspace_method = getattr(ScikitsPredictor, pyspace_name)
        scikits_method = getattr(scikits_class, scikits_name)
        if hasattr(scikits_method, 'im_func'):
            # some scikits algorithms do not define an __init__ method
            # the one inherited from 'object' is a
            # "<slot wrapper '__init__' of 'object' objects>"
            # which does not have a 'im_func' attribute
            pyspace_method.im_func.__doc__ = _gen_docstring(scikits_class,
                                                        scikits_method.im_func)

    if scikits_class.__init__.__doc__ is None:
        ScikitsPredictor.__init__.im_func.__doc__ = _gen_docstring(scikits_class)
    return ScikitsPredictor


#list candidate nodes
def print_public_members(class_):
    """ Print methods of sklearn algorithm """
    print '\n', '-' * 15
    print '%s (%s)' % (class_.__name__, class_.__module__)
    for attr_name in dir(class_):
        attr = getattr(class_, attr_name)
        #print attr_name, type(attr)
        if not attr_name.startswith('_') and inspect.ismethod(attr):
            print ' -', attr_name

#apply_to_scikits_algorithms(sklearn, print_public_members)


def wrap_scikits_algorithms(scikits_class, nodes_list):
    """ Check *scikits_class* and append new wrapped class to *nodes_list*

    Currently only classifiers subclassing ``sklearn.base.ClassifierMixin``
    and having a *fit* method were integrated and tested.
    Algorithms with the *transform* function are also available.
    *predict* nodes will be available soon but require more testing especially
    of regression in pySPACE.
    """

    name = scikits_class.__name__
    if (name[:4] == 'Base' or name == 'LinearModel'
            or name.startswith('EllipticEnvelop')
            or name.startswith('ForestClassifier')):
        return

    if issubclass(scikits_class, sklearn.base.ClassifierMixin) and \
            hasattr(scikits_class, 'fit'):
        nodes_list.append(wrap_scikits_classifier(scikits_class))
    # Some (abstract) transformers do not implement fit.
    elif hasattr(scikits_class, 'transform') and hasattr(scikits_class, 'fit'):
        nodes_list.append(wrap_scikits_transformer(scikits_class))
    elif hasattr(scikits_class, 'predict') and hasattr(scikits_class, 'fit'):
        pass  # for the moment, we don't support predictors (regression in pySPACE)
        # nodes_list.append(wrap_scikits_predictor(scikits_class))

if _sklearn_prefix:
    scikits_nodes = []
    apply_to_scikits_algorithms(sklearn,
                                lambda c: wrap_scikits_algorithms(
                                    c, scikits_nodes))
    # add scikits nodes to dictionary
    #scikits_module = new.module('scikits')
    for wrapped_c in scikits_nodes:
        DEFAULT_NODE_MAPPING[wrapped_c.__name__] = wrapped_c
        NODE_MAPPING[wrapped_c.__name__] = wrapped_c
        NODE_MAPPING[wrapped_c.__name__[:-4]] = wrapped_c
    del(wrapped_c)