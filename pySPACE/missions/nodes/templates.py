""" Tell the developer about general coding and documentation approaches for nodes

The first line in each module/class/function docstring should be short,
in imperative, without stop and explain, what it does.
You should **not** copy it from elsewhere, because if you overwrite
existing routines, they should be different and such differ in documentation.
If you really can not help, copying code, you should at least document,
where you took it from.

Detailed information on documentation guidelines can be found
:ref:`here <doc>`.
Before writing your own node, you have to have a look at it, because every node
must be documented!
You should also have a look at the documentation of the
:mod:`~pySPACE.missions.nodes` package

.. note:: Some parts of the guidelines, will be copied to this place,
          because of their importance.

Coding Guidelines
-----------------

Before programming a node or overwriting an method, have a look at this
documentation and the documentation in the
:class:`~pySPACE.missions.nodes.base_node.BaseNode`, because otherwise
you might get unwanted side effects.
Furthermore, you should have a look at existing code and mainly follow
the `PEP 8 coding style <http://www.python.org/dev/peps/pep-0008/>`_

How is the node accessed? - Some naming conventions
----------------------------------------------------

The :mod:`~pySPACE.missions.nodes` package supports automatic loading of nodes.
The easiest access then is via the
:mod:`node chains <pySPACE.environments.chains.node_chain>` in the
:mod:`node chain operation <pySPACE.missions.operations.node_chain>`.
Their, your new implemented node needs a name.
To be detected as node, your node has to end with **`Node`**.
Node names are written in CamelCase.
Automatic node names are the class name and the class name without the
ending **Node**.
If you want to have extra names or backward compatibility after changing names,
you can define or extend a dictionary with the name
**_NODE_MAPPING** at the end of each module implementation.
As keys you use the new names and the value is the corresponding class name.
Be careful to use meaningful names or abbreviations, which are not already
:ref:`in use <node_list>`.

When the documentation is generated, the available names are added
automatically to the documentation of each node and additionally
:ref:`complete lists <node_list>` are generated to give information
on available nodes and their functionality.

Base nodes should always include **"Base"** in their class name
and if they are not usable they should end with it instead of **"Node"**.

The First Step: Where to put the node?
--------------------------------------

Finding the Category
+++++++++++++++++++++

Before implementing your own node, you should find out, where to put it.
The :mod:`~pySPACE.missions.nodes` has several categories of algorithms
and you should check their documentation to find out, where your node belongs to.

Finding the Module
+++++++++++++++++++

As the next step, you should find out, if there is already a fitting module in
there, which fits to your algorithm or needs only small change in
documentation, to be fitting.

If there is no module, you have to open up your own new one.

.. warning:: Be careful, when creating new modules. The module name should be
             meaningful. The module should include some basic documentation and
             most importantly describe a general concept of a group
             of algorithms and not repeat the documentation of your new class.

.. note:: When the documentation is generated, at the end of each module
          documentation, a summary of its functions and classes is generated.
          So you should not do this manually.

Finding the Class Name
+++++++++++++++++++++++

The class name is written with the above mentioned conventions.
But before starting with it, you should first check,
if there is already a class, which only has to be modified for your new part.
Second you should check if your algorithms is developed for a very special
purpose but could be generalized.
Finally you decide for a short meaningful algorithm name, which
contrasts your algorithm from the exiting ones.

Base Nodes
++++++++++

For some highly sophisticated types of nodes you can find a corresponding
base node in the package.
The :mod:`~pySPACE.missions.nodes.visualization` package is the best example
in this case.
Theses nodes define a special interface for your algorithm and you will only
have to implement some special functions for these nodes,
which can be found in their documentation.

Currently the default is, that your node is not inheriting from any special
generalizing base node but only from the basic
:class:`~pySPACE.missions.nodes.base_node.BaseNode`.
This node is by default just forwarding the data, but implements all
functionality a node needs.

The Main Principle
------------------

Every node, which is no base node should inherit from the base node.

.. code-block:: python

    from pySPACE.missions.nodes.base_node import BaseNode

Implementing a node now is nothing more than carefully overwriting the
default methods of this node.
Depending on the complexity of your algorithm, this might be very easy or a bit
more complicated. In the following we will give advice and examples therefore.

The Processed Data Types - Input and Output
--------------------------------------------

For processing only special input and output types are required,
which are subclasses of numpy arrays.
All currently available types can be found in
:mod:`pySPACE.resources.data_types`.


Further Minor Information
-------------------------

* Randomization should be done be setting a fixed seed with the help of
  *self.run_number* to enable reproducibility.
* The here presented nodes are only examples
  for better documentation of the node creation process
  and should not be taken to serious
  or used as base nodes for real algorithms.
* A special end of a node chain can be a *sink node*.
  It is defined by implementing the method *get_result_dataset*.
  Its function is to gather all data.
* The variable *self.temp_dir* an be used to store some data temporarily,
  e.g. for security reasons.

General Concept of a Node
-------------------------

.. image:: ../../graphics/node.png
   :width: 500

Integration of Nodes in a :mod:`~pySPACE.environments.chains.node_chain`
-------------------------------------------------------------------------

.. image:: ../../graphics/node_chain.png
   :width: 500

Usage of :mod:`node chains <pySPACE.environments.chains.node_chain>`
--------------------------------------------------------------------

.. image:: ../../graphics/launch_live.png
   :width: 500

Visualization of :mod:`~pySPACE.missions.nodes.splitter` Nodes
--------------------------------------------------------------

.. image:: ../../graphics/splitter.png
    :width: 500
"""
from pySPACE.missions.nodes.base_node import BaseNode
import warnings
import logging
import numpy
from pySPACE.resources.data_types.feature_vector import FeatureVector

class SimpleDataTransformationTemplateNode(BaseNode):
    """ Parametrized algorithm, transforming the data without training

    Describe your algorithm in detail.

    In the simplest case, an algorithm only implements its initialization
    and execution function like this node.

    The list of parameters should always be complete and correct to avoid
    hidden functionality.

    **References**

        If this node is using code from other implementations or
        is described in detail in a publication,
        mention the reference here.

    **Parameters**
        :Parameter1: Describe effect and specialties

            (*recommended, default: 42*)

        :Parameter2: Describe the effect, and if something special happens by
            default. It is also important to mention, which entries are possible
            (e.g. only True and False are accepted values).

            (*optional, default: False*)

    **Exemplary Call**

    .. code-block:: yaml

        -
            node : SimpleDataTransformationTemplate
            parameters:
                Parameter1 : 77
                Parameter2 : False


    :input:    Type1 (e.g. FeatureVector)
    :output:   Type2 (e.g. FeatureVector)
    :Author: Mario Muster (muster@informatik.exelent-university.de)
    :Created: 2013/02/25
    """
    def __init__(self, Parameter1=42, Parameter2=False, **kwargs):
        """ Set the basic parameters special for this algorithm

        If your init is not doing anything special, it does not need any
        documentation. The relevant class documentation
        is expected to be in the class docstring.

        .. note::
            The mapping from the call of the function with a YAML file
            and this init is totally straightforward.
            Every parameter in the dictionary description in the
            YAML file is directly used at the init call.
            The value of the parameter is transformed with the help
            of the YAML syntax (see: :ref:`yaml`).

        It is important to also use `**kwargs`, because they have to be
        forwarded to the base class, using:

        .. code-block:: python

            super(SimpleDataTransformationTemplateNode, self).__init__(**kwargs)

        .. warning::
            With the call of `super` comes some hidden functionality.
            Every self parameter in the init is made permanent via the
            function:
            :func:`~pySPACE.missions.nodes.base_node.BaseNode.set_permanent_attributes`
            from the base node.
            Normally all self parameters are instantiated after this call
            and have to be made permanent on their own.
            Permanent means, that these parameters are reset to the
            defined value, when the
            :func:`~pySPACE.missions.nodes.base_node.BaseNode.reset`
            method is called.
            This is for example done during k-fold cross validation, when
            the training fold is changed.
            For special variable types you may run into trouble,
            because set_permanent_attributes needs to copy them.

        .. warning::
            The init function is called before the distribution of node_chains
            in the parallel execution. So the node parameters need to be
            able to be stored into the pickle format.
            If you need parameters, which have not this functionality,
            just initialize them with the first call of the training or execute
            method.

        .. code-block:: python

            self.set_permanent_attributes(  P1 : Parameter1,
                                            P2 : Parameter2,
                                            P3 : "Hello"
                                         )

        Here `self.P3` will be an internal parameter.
        """
        super(SimpleDataTransformationTemplateNode, self).__init__(**kwargs)
        if not type(Parameter1) == int:
            warnings.warn("Parameter 1 is having wrong type %s." %
                str(type(Parameter1)))
            Parameter1 = 42
        self.set_permanent_attributes(P1=Parameter1,
                                      P2=Parameter2,
                                      P3="Hello")

    def _execute(self, x):
        """ General description of algorithm maybe followed by further details

        E.g. log "Hello" during first call and if P2 is set to True,
        always multiply data with P1 and in the other case forward the data.

        Logging is done using
        :func:`~pySPACE.missions.nodes.base_node.BaseNode._log`:

        .. code-block:: python

            self._log(self.P3, level=logging.DEBUG)

        To access only the data array and not the attached meta data, use
        `data = x.view(numpy.ndarray)` for preparation.
        """
        if self.P3:
            self._log(self.P3, level=logging.DEBUG)
            self.P3 = False
        data = x.view(numpy.ndarray)
        if self.P2:
            data = self.P1 * data
            x = FeatureVector.replace_data(x, data)
        return x


class TrainableAlgorithmTemplateNode(SimpleDataTransformationTemplateNode):
    """ Template for trainable algorithms

    :class:`SimpleDataTransformationTemplateNode` is the base node for this node
    and so, this node does not have to implement an _execute or __init__
    function. Often these methods have to be implemented nevertheless,
    but not here, to keep the example short.

    For trainable methods, a minimum of two functions has to be implemented:
    :func:`is_trainable` and :func:`_train`.
    Optionally four other functions can be overwritten:
    :func:`is_supervised`, :func:`_stop_training`, :func:`_inc_train`
    and :func:`start_retraining.`
    The first returns by default `False` and the other methods do nothing.

    .. note:: The execute function is applied on all data,
              even the training data, but the true label remains unknown.

    **Parameters**
        Please refer to :class:`SimpleDataTransformationTemplateNode`

        .. note:: Parameter1 is determined, by counting the training examples.

    **Exemplary Call**

    .. code-block:: yaml

        -
            node : TrainableAlgorithmTemplateNode
            parameters:
                Parameter1 : 77
                Parameter2 : False

    :input:    Type1 (e.g. FeatureVector)
    :output:   Type2 (e.g. FeatureVector)
    :Author: Mario Muster (muster@informatik.exelent-university.de)
    :Created: 2013/02/25
    """
    def is_trainable(self):
        """ Define trainable node, by returning True in this function """
        return True

    def is_supervised(self):
        """ Return True to get access to labels in training functions """
        return True

    def _train(self,data,class_label):
        """ Called for each element in training data to be processed

        Incremental algorithms, simply use the example to change their
        parameters and batch algorithms preprocess data and only store it.

        If :func:`is_supervised` were not overwritten or set `False`,
        this function is defined without the parameter *class_label*
        """
        if self.P3 == "Hello":
            self.P3 = ""
            self.P1 = 0
        self.P1 += 1
        self.P3 += class_label

    def _stop_training(self):
        """ Called after processing of all training examples

        For simplicity, we just reimplement the default.
        """
        pass

    def _inc_train(self,data, class_label):
        """ Train on new examples in testing phase

        During testing phase in the application phase,
        new labeled examples may occur  and this function is
        used to improve the already trained algorithm on these examples.

        .. note:: This method should always be as fast as possible.

        For simplicity, we only forward everything to :func:`_train`.

        For more details on retraining (how to turn it on, and how it works),
        have a look at the documentation of the *retrain* parameter
        in the :class:`~pySPACE.missions.nodes.base_node.BaseNode`.
        """
        self._train(data, class_label)

    def start_retraining(self):
        """ Prepare retraining

        Normally this method is not needed and does nothing,
        but maybe some parameters
        have to be changed, before the first retraining with
        the _inc_train method should be done.
        This method is here, to give this possibility.

        In our case, we simply reset the starting parameter *self.P3*.
        """
        self.P3 = "Hello"


class SpecialPurposeFunctionsTemplate(BaseNode):
    """ Introduce additional available functions

    Additional to the aforementioned methods,
    some algorithms have to overwrite the default behavior of nodes,
    directly change the normal data flow,
    manipulate data or labels,
    or communicate information to other nodes.

    Some of these methods will be introduced in the following and some use cases
    will be given.

    .. warning::
        Every method in the :class:`~pySPACE.missions.nodes.base_node.BaseNode`
        could be overwritten but this should be done very carefully to avoid
        bad side effects.
    """
    def store_state(self, result_dir, index=None):
        """ Store some additional results or information of this node

        Here the parameter *self.store* should be used to switch on the saving,
        since this method is called in every case,
        but should only store information, if this parameter is set true

        This method is automatically called during benchmarking
        for every node.
        It is for example used
        to store visualization of algorithms or data.

        Additionally to the result_dir, the node name should be used.
        If you expect this node to occur multiple times in a node chain,
        also use the index. This can be done for example like:

        .. code-block:: python

            import os
            from pySPACE.tools.filesystem import create_directory
            if self.store:
                #set the specific directory for this particular node
                node_dir = os.path.join(result_dir, self.__class__.__name__)
                #do we have an index-number?
                if index is None:
                    #add the index-number...
                    node_dir += "_%d" % int(index)
                create_directory(node_dir)

        Furthermore, it is very important to integrate the split number into
        the file name when storing, because otherwise your results will be
        overwritten. The convention in pySPACE is, to have a meaningful name
        of the part of the node you store followed by an underscore and 'sp'
        and the split number as done in

        .. code-block:: python

            file_name = "%s_sp%s.pickle" % ("patterns", self.current_split)
        """
        pass

    def reset(self):
        """ Resets the node to a clean state

        Every parameter set with
        :func:`~pySPACE.missions.nodes.base_node.BaseNode.set_permanent_attributes`
        is by default reset here to its specified value or deleted,
        if no value is specified.

        Since this method copies every parameter or some variables escape
        from the normal class variables scope, some methods need to overwrite
        this method.

        When you really need to overwrite this method some points have to be
        considered. For the normal functionality of the node, the super method
        needs to be called. To avoid deleting of the special variables,
        they have to be made local variables beforehand and afterwards again
        cast to class variables. This is depicted in the following example code,
        taken from the
        :class:`~pySPACE.missions.nodes.meta.same_input_layer.SameInputLayerNode`.

        .. code-block:: python

            def reset(self):
                ''' Also reset internal nodes '''
                nodes = self.nodes
                for node in nodes:
                    node.reset()
                super(SameInputLayerNode, self).reset()
                self.nodes = nodes
        """
        pass

    def get_result_dataset(self):
        """ Implementing this function, makes a node a :mod:`~pySPACE.missions.nodes.sink` """
        pass

    def request_data_for_training(self, use_test_data):
        """ Returns generator for training data for subsequent nodes of the node chain

        If *use_test_data* is true, all available data is used for
        training, otherwise only the data that is explicitly for training.

        These methods normally use the
        :class:`~pySPACE.tools.memoize_generator.MemoizeGenerator`
        to define their generator.
        When implementing such a method, one should always try not to double
        data but only redirect it, without extra storing it.

        The definition or redefinition of training data is done
        by :mod:`~pySPACE.missions.nodes.source` and
        :mod:`~pySPACE.missions.nodes.splitter` nodes.
        """
        pass

    def request_data_for_testing(self):
        """ Returns data for testing of subsequent nodes of the node chain

        When defining :func:`request_data_for_training` this method
        normally has to be
        implemented/overwritten, too and vice versa.
        """
        pass

    def process_current_split(self):
        """ Main processing part on test and training data of current split

        This method is called in the usage with benchmark node chains
        and defines the gathering of the result data of the node chain
        for a :mod:`~pySPACE.missions.nodes.sink` node.

        Hereby it gets the data by calling
        :func:`request_data_for_training` and :func:`request_data_for_testing`.

        In the case of using the
        :class:`~pySPACE.missions.nodes.cv_splitter.CrossValidationSplitterNode`,
        this method is called multiple times
        for each split and stores every time the result in the result dataset
        separately.

        Though this approach seems on first sight very complicated on first
        sight, it gives three very strong advantages.

        * The cross validation can be done exactly before the first trainable
          node in the node chain and circumvents unnecessary double processing.

        * By handling indices instead of real data,
          the data for training and testing is not copied and memory is saved.

        * The cross validation is very easy to use.
          Moving this functionality to the
          :mod:`~pySPACE.resources.dataset_types`
          would make the usage muh mor complicated and inefficient.
          Especially for nodes, which internally use node chains,
          like the :mod:`~pySPACE.missions.nodes.meta.parameter_optimization`
          nodes, this easy access pays off.
        """
        pass

    def get_sensor_ranking(self):
        """ Return sensor ranking fitting to the algorithm

        For usage with the ranking variant in the
        :class:`~pySPACE.missions.nodes.spatial_filtering.sensor_selection.SensorSelectionRankingNode`
        this method of the node is called to get the ranking to reduce sensors.

        The ranking is a sorted list of tuple (sensor name, weight).
        The first element has to correspond to the
        sensor with the lowest weight, meaning it is the most unimportant.

        .. note:: The code here is a copy from
            :class:`~pySPACE.missions.nodes.classification.base`
            which takes the classification vector `self.features`
            and sums up the absolute values fitting to one channel.
            It is only used as an example.
        """
        # channel name is what comes after the first underscore
        feat_channel_names = [chnames.split('_')[1]
                              for chnames in self.features.feature_names]
        from collections import defaultdict
        ranking_dict = defaultdict(float)
        for i in range(len(self.features[0])):
            ranking_dict[feat_channel_names[i]] += abs(self.features[0][i])
        ranking = sorted(ranking_dict.items(),key=lambda t: t[1])
        return ranking
