""" Tell the developer about general coding and documentation approaches for nodes

A very useful tutorial can be found under :ref:`t_new_node`.
"""
from pySPACE.missions.nodes.base_node import BaseNode
import warnings
import logging
import numpy
from pySPACE.resources.data_types.feature_vector import FeatureVector
from pySPACE.tools.memoize_generator import MemoizeGenerator

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


class SimpleSourceTemplateNode(BaseNode):
    """ A simple template that illustrates the basic principles of a source node

    In `pySPACE`, source nodes are used at the beginning of the node chain.
    The source nodes are responsible for the input of data, be it from a
    static source or from a live stream.

    It is very important to note that these nodes just serve the purpose of
    providing the node chain with an input dataset and do not perform any
    changes on the data itself. That being said, these nodes are **do not**
    have an **input node** and are **not trainable**!

    In the following we will discuss the general strategy for building a new
    source node for a static input data set which has been saved to disk.
    In the case of more complicated inputs, please consult the documentation of
    :mod:`~pySPACE.missions.nodes.source.external_generator_source.ExternalGeneratorSourceNode`
    and :mod:`~pySPACE.missions.nodes.source.time_series_source.Stream2TimeSeriesSourceNode`
    """
    def __init__(self, **kwargs):
        """ Initialize some values to 0 or `None`

        The initialization routine of the source node is basically completely
        empty. Should you feel the need to do something in this part of the
        code, you can initialize the ``input_dataset`` to ``None``. This
        attribute will then later be changed when the ``set_input_dataset``
        method is called.

        If the user wants to generate the dataset inside the SourceNode,
        this should be done in the ``__init__`` method though. A good example
        of this practice can be found in the
        :mod:`~pySPACE.missions.nodes.source.random_time_series_source.RandomTimeSeriesSourceNode`
        """
        super(SimpleSourceTemplateNode, self).__init__(**kwargs)

        self.set_permanent_attributes(dataset=None)

    def set_input_dataset(self, dataset):
        """ Sets the dataset from which this node reads the data

        This method is the beginning of the node. Put simply, this method
        starts the feeding process of your node chain by telling the node chain
        where to get the data from.
        """
        self.set_permanent_attributes(dataset=dataset)

    def request_data_for_training(self, use_test_data):
        """ Returns the data that can be used for training of subsequent nodes

        This method streams training data and sends it to the subsequent nodes.
        If one looks at the tutorial related to building new nodes (available in
        the tutorial section), one can see exactly where the ``request_data``
        methods are put to use.

        The following example is one that was extracted from the
        :mod:`~pySPACE.missions.nodes.source.feature_vector_source.FeatureVectorSourceNode`

        which should(in theory at least) be implementable for all types of data.
        """
        if not use_test_data:
            # If the input dataset consists only of one single run,
            # we use this as input for all runs to be conducted (i.e. we
            # rely on later randomization of the order). Otherwise
            # we use the data for this run number
            if self.dataset.meta_data["runs"] > 1:
                key = (self.run_number, self.current_split, "train")
            else:
                key = (0, self.current_split, "train")
            # Check if there is training data for the current split and run
            if key in self.dataset.data.keys():
                self._log("Accessing input dataset's training feature vector windows.")
                self.data_for_training = MemoizeGenerator(self.dataset.get_data(*key).__iter__(),
                                                          caching=self.caching)
            else:
                # Returns an iterator that iterates over an empty sequence
                # (i.e. an iterator that is immediately exhausted), since
                # this node does not provide any data that is explicitly
                # dedicated for training
                self._log("No training data available.")
                self.data_for_training = MemoizeGenerator((x for x in [].__iter__()),
                                                          caching=self.caching)
        else:
            # Return the test data as there is no additional data that
            # was dedicated for training
            return self.request_data_for_testing()

        # Return a fresh copy of the generator
        return self.data_for_training.fresh()


    def request_data_for_testing(self):
        """ Returns the data that can be used for testing of subsequent nodes

        The principle of obtaining the testing data are the same as the principles
        used in obtaining the training data set. The only difference here is that,
        in the case in which there is no testing data available, we allow for the
        training data to be used as testing data.
        """
        # If we haven't read the data for testing yet
        if self.data_for_testing == None:
            self._log("Accessing input dataset's test feature vector windows.")
            # If the input dataset consists only of one single run,
            # we use this as input for all runs to be conducted (i.e. we
            # rely on later randomization of the order). Otherwise
            # we use the data for this run number
            if self.dataset.meta_data["runs"] > 1:
                key = (self.run_number, self.current_split, "test")
            else:
                key = (0, self.current_split, "test")

            test_data_generator = self.dataset.get_data(*key).__iter__()

            self.data_for_testing = MemoizeGenerator(test_data_generator,
                                                     caching=self.caching)

        # Return a fresh copy of the generator
        return self.data_for_testing.fresh()


    def getMetadata(self, key):
        """ Return the value corresponding to the given key from the dataset meta data of this source node

        At some point in time, you might need to know the metadata of some
        specific input in your input and this is when you would use this method.
        """
        return self.dataset.meta_data.get(key)

    def use_next_split(self):
        """ Return False

        The method will always return `False` since the SourceNode
        should(in the case of more than 1 split) execute the splits in
        parallel and not in series.
        """
        return False


class SimpleSinkTemplateNode(BaseNode):
    """ A simple template that illustrates the basic principles of a sink node

    The sink node is always placed at the end of the node chain. You can think
    of a sink node as a place in which you can throw all your data and it will
    do something with this data e.g. saving it to disk.

    Of course, this is not the only possibility for a Sink node but it is the
    most basic one. One example of a more complex process happening inside the
    Sink node is that of the
    :mod:`~pySPACE.missions.nodes.sink.classification_performance_sink.PerformanceSinkNode`
    whereby the classification results are collected into a complex structure
    that reflects the performance of the entire node chain.

    That being said, this template addresses the very simple case of just
    collecting the results of the node chain and doing something with them.

    For a complete list of the available nodes, please consult
    :mod:`~pySPACE.missions.nodes.sink`
    """

    def __init__(self, selection_criterion=None, data=None, **kwargs):
        """ Initialize some criterion of selection for the data

        In the initialization stage, the node is expected to just save some
        permanent attributes that it might use at a later point in time.
        In the case of :class:`~pySPACE.resources.data_types.feature_vector.FeatureVector`
        data, this criterion might represent selected channel names(as
        implemented in
        :mod:`~pySPACE.missions.nodes.sink.feature_vector_sink.FeatureVectorSinkNode`
        while for :mod:`~pySPACE.resources.data_types.time_series.TimeSeries`
        it might represent a sorting criterion, as implemented in
        :mod:`~pySPACE.missions.nodes.sink.time_series_sink.TimeSeriesSinkNode`
        Since this is only a mere template, we will call our selection criterion
        `selection_criterion` and leave it up to the user to implement specific
        selection criteria.
        """
        super(SimpleSinkTemplateNode, self).__init__(**kwargs)

        self.set_permanent_attributes(selection_crit=selection_criterion,
                                      data=data)

    def is_trainable(self):
        """ Return True if the node is trainable

        While the sink nodes do not need to be trained, they do need access to
        the training data that is sent through the node chain. In order to
        achieve this, the :func:`~pySPACE.missions.nodes.base_node.BaseNode.is_trainable`
        function from the `BaseNode` is overwritten such that it
        always returns `True` when access to the training data is required.
        """
        return True

    def is_supervised(self):
        """ Returns True if the node requires supervised training

        The function will almost always return True. If the node requires access
        to the training data i.e. if the node `is_trainable` it will almost
        surely also be supervised.
        """
        return True

    def _train(self, data, label):
        """ Tell the node what to do with specific data inputs

        In the case of Sink nodes, the `_train` function is usually overwritten
        with a dummy function that either returns the input data e.g.
        :mod:`~pySPACE.missions.nodes.sink.analyzer_sink.AnalyzerSinkNode`
        or just does not(as we will implement it here)
        """
        pass

    def reset(self):
        """ Reset the permanent parameters of the node chain

        When used inside a node chain, the Sink node should also be
        responsible for saving the permanent state parameters. These
        parameters get reinitialized whenever the node chain reaches its
        end. Nevertheless, the parameters should be saved such that they
        can be inspected after the entire procedure has finished.


        The following piece of code was adapted from
        :mod:`~pySPACE.missions.nodes.sink.feature_vector_sink.FeatureVectorSinkNode`
        with the `FeatureVector` specific parameters changed to dummy
        variables.
        """
        import copy
        tmp = self.permanent_state
        tmp["dataset"] = self.data
        self.__dict__ = copy.copy(tmp)
        self.permanent_state = tmp

    def process_current_split(self):
        """  The final processing step for the current split

        This function should contain the last activities that need to be run in
        the current split. You should include any method that combines, selects
        or transforms the result data set in any way into this function.
        """
        pass

    def get_result_dataset(self):
        """ Return the result dataset

        This function should be built such that it returns the result dataset.
        """
        return self.data
