""" Sources for windowed times series, e.g from streaming data

.. seealso::

    - :class:`~pySPACE.resources.data_types.time_series.TimeSeries`
    - :class:`~pySPACE.resources.dataset_defs.time_series.TimeSeriesDataset`
    - :class:`~pySPACE.resources.dataset_defs.stream.StreamDataset`

"""
import logging
import os

from pySPACE.missions.nodes.base_node import BaseNode
from pySPACE.missions.support.windower import Windower
from pySPACE.tools.memoize_generator import MemoizeGenerator


class TimeSeriesSourceNode(BaseNode):
    """ Source for windowed :class:`~pySPACE.resources.data_types.time_series.TimeSeries` saved in pickle format via :class:`~pySPACE.missions.nodes.sink.time_series_sink.TimeSeriesSinkNode`
    
    **Parameters**
    
    **Exemplary Call**
    
    .. code-block:: yaml
    
        - 
            node : TimeSeriesSource
    
    :Author: Jan Hendrik Metzen (jhm@informatik.uni-bremen.de)
    :Created: 2008/11/25
    """
    
    def __init__(self, **kwargs):
        super(TimeSeriesSourceNode, self).__init__(**kwargs)
        
        self.set_permanent_attributes(input_types=["time_series"],
                                      dataset=None)

    def set_input_dataset(self, dataset):
        """ Sets the dataset from which this node reads the data """
        self.set_permanent_attributes(dataset=dataset)

    def register_input_node(self, node):
        """ Register the given node as input """
        raise Exception("No nodes can be registered as inputs for source nodes")

    def use_next_split(self):
        """
        Use the next split of the data into training and test data.
        Returns True if more splits are available, otherwise False.
        
        This method is useful for benchmarking
        """
        # if the input dataset has more than one split/run we will compute
        # the splits in parallel, i.e. we don't return any further splits
        return False
    
    def train_sweep(self, use_test_data):
        """
        Performs the actual training of the node.
        .. note:: Source nodes cannot be trained
        """
        raise Exception("Source nodes cannot be trained")
    
    def request_data_for_training(self, use_test_data):
        """
        Returns the time windows that can be used for training of subsequent nodes

        .. todo:: to document
        """
        if not use_test_data:
            # If the input dataset consists only of one single run,
            # we use this as input for all runs to be conducted (i.e. we
            # rely on later randomization of the order). Otherwise
            # we use the data for this run number
            if self.dataset.meta_data["runs"] > 1:
                key = (self.run_number, self.current_split, "train")
                self._log("Run %s." % self.run_number)
            else: 
                key = (0, self.current_split, "train")
                self._log("Run %s. Using input data of run 0." % self.run_number)
                
            # Check if there is training data for the current split and run
            if key in self.dataset.data.keys():
                self._log("Accessing input dataset's training time series windows.")
                self.data_for_training = \
                    MemoizeGenerator(self.dataset.get_data(*key).__iter__(),
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
        """
        Returns the data that can be used for testing of subsequent nodes

        .. todo:: to document
        """
        # If we haven't read the data for testing yet
        if self.data_for_testing is None:
            self._log("Accessing input dataset's test time series windows.")
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
        """ Return the value corresponding to the given key from the dataset meta data of this source node. """
        return self.dataset.meta_data.get(key)

    def __del__(self):
        del self.dataset
        self.dataset = None


class BCICompetitionSourceNode(TimeSeriesSourceNode):
    """ The source node for a BCI Competition data set 

    The data of the .mat file is transferred to :class:`~pySPACE.resources.data_types.time_series.TimeSeries` objects.
    
    **Parameters**
    
    **Exemplary Call**
    
    .. code-block:: yaml
    
        - 
            node : BCICompetitionSource
    
    :Author: Hendrik Woehrle (hendrik.woehrle@dfki.de)
    """
    pass


class Stream2TimeSeriesSourceNode(TimeSeriesSourceNode):
    """ Transformation of streaming data to windowed time series

    This node contains an interfaces the streaming dataset to provide
    it with a windowing specification and then get a data generator.
    This is a main difference, since other source nodes, get access to
    the real data and generate a generator object.

    For the segmentation of the data, the
    :class:`~pySPACE.resources.dataset_defs.stream.StreamDataset`
    uses the :class:`~pySPACE.missions.support.windower.MarkerWindower`.

    **Parameters**

        :windower_spec_file:
            The specification file for the Windower containing information which
            data should be windowed and which data can be discarded.

            For a detailed description look at the module description.

            (*recommended, default: windower.WindowFactory.default_windower_spec*)

        :local_window_conf:
            Can be set to True if the user wants to specify the location of the
            windower spec file manually. In the default situation, the spec file
            is looked up according to the location of the spec files. When
            set to True, the windower spec file can be specified with path (e.g.
            '/home/myuser/myspecs/mywindow.yaml') or without path, which indicates
            that the window specs file is located in current local folder
            or the specification file folder of the node chain.
            For the parameterization of the windower configuration file,
            you should have a look at the documentation
            of the :class:`~pySPACE.missions.support.windower.MarkerWindower`

            (*optional, default: False*)

        :nullmarker_stride_ms:
            An integer to specify the interval of the null marker. The null marker
            is than inserted into the data stream every *null_marker_stride_ms* ms.
            This marker can be used to cut out sliding windows at a constant rate.

            Either *nullmarker_stride_ms* or *windower_spec_file*
            should be specified!

            (*recommended, default: 1000*)

        :no_overlap:
            When having streamed windows, the last data point of the previous
            window might be the same as the one of the current window,
            since when using a fixed window size, first and last point of the
            window are normally used. This effect can be turned off with this
            parameter.

            When a window spec file is given, the default is *False*.
            If not, the default is *True*.

            (*recommended, default: None*)
            
        :data_consistency_check:
            When True it will be checked if cut windows contain channels with
            zero standard deviation and the user will be informed.
            
            (*optional, default: False*)

    **Exemplary Call**

    .. code-block:: yaml

        -
            node: Stream2TimeSeriesSourceNode
            parameters :
                windower_spec_file : "example_lrp_window_spec.yaml"

    :Author: Johannes Teiwes (johannes.teiwes@dfki.de)
    :Created: 2010/10/12
    :LastChanges: Mario Michael Krell

    """

    def __init__(self, windower_spec_file=None, local_window_conf=False,
                 nullmarker_stride_ms=None, no_overlap=False,
                 continuous=False,
                 data_consistency_check=False, **kwargs):

        super(Stream2TimeSeriesSourceNode, self).__init__(**kwargs)

        assert not(nullmarker_stride_ms is None and windower_spec_file is None),\
            "No segmentation parameters specified!"
        if windower_spec_file is None:
            windower_spec_file = ""
            no_overlap = True
            continuous = True
        elif nullmarker_stride_ms is None:
            nullmarker_stride_ms = 1000

        self.set_permanent_attributes(
            input_types=["stream"],
            window_definition=Windower._load_window_spec(
                windower_spec_file, local_window_conf),
            nullmarker_stride_ms=nullmarker_stride_ms,
            no_overlap=no_overlap,
            data_consistency_check=data_consistency_check,
            dataset=None,
            continuous=continuous)

    def get_source_file_name(self):
        """ Returns the file name of the source file"""
        return self.dataset.data_file.split('/')[-1]

    def process(self):
        """ Returns a generator that yields all data received by the client

        This is helpful, when using this source node in online application,
        since for most other source nodes, :func:`request_data_for_testing`
        is used instead.

        ..todo:: check code
        """

        # self._log("Processing data.", level = logging.DEBUG)
        #
        # # Create a generator that emits the windows
        # data_generator = ((sample, label) for (sample, label) in \
        #                                                    self.marker_windower)
        # return data_generator
        return self.request_data_for_testing()

    def request_data_for_training(self, use_test_data):
        """
        Returns the data that can be used for training of subsequent nodes

        .. todo:: to document
        """
        self._log("Requesting train data...")
        if not use_test_data:
            # If we haven't read the data for training yet
            if self.data_for_training is None:

                self._log("Start streaming.")

                self.dataset.set_window_defs(
                    window_definition=self.window_definition,
                    nullmarker_stride_ms=self.nullmarker_stride_ms,
                    no_overlap=self.no_overlap,
                    data_consistency_check=self.data_consistency_check)

                if self.dataset.meta_data["runs"] > 1:
                    key = (self.run_number, self.current_split, "train")
                else:
                    key = (0, self.current_split, "train")

                # Create a generator that emits the windows
                train_data_generator = (
                    (sample, label)
                    for (sample, label) in self.dataset.get_data(*key))

                self.data_for_training = \
                    MemoizeGenerator(train_data_generator,
                                     caching=self.caching)

            # Return a fresh copy of the generator
            return self.data_for_training.fresh()
        else:
            # Return the test data as there is no additional data that
            # was dedicated for training
            return self.request_data_for_testing()

    def request_data_for_testing(self):
        """
        Returns the data that can be used for testing of subsequent nodes

        .. todo:: to document
        """
        self._log("Requesting test data...")
        # If we haven't read the data for testing yet
        if self.data_for_testing is None:

            self._log("Start streaming.")

            self.dataset.set_window_defs(
                window_definition=self.window_definition,
                nullmarker_stride_ms=self.nullmarker_stride_ms,
                no_overlap=self.no_overlap,
                data_consistency_check=self.data_consistency_check)

            if self.dataset.meta_data["runs"] > 1:
                key = (self.run_number, self.current_split, "test")
            else:
                key = (0, self.current_split, "test")

            # Create a generator that emits the windows
            test_data_generator = (
                (sample, label)
                for (sample, label) in self.dataset.get_data(*key))

            self.data_for_testing = \
                MemoizeGenerator(test_data_generator,
                                 caching=self.caching)

        # Return a fresh copy of the generator
        return self.data_for_testing.fresh()

    def store_state(self, result_dir, index=None):
        """ Stores this node in the given directory *result_dir* """
        from pySPACE.tools.filesystem import  create_directory
        node_dir = os.path.join(result_dir, self.__class__.__name__)
        create_directory(node_dir)

        result_file = open(os.path.join(node_dir, "window_definitions.txt"), "w")
        for window_def in self.window_definition:
            result_file.write(str(window_def))
        result_file.close()


class TimeSeries2TimeSeriesSourceNode(Stream2TimeSeriesSourceNode):
    """ Source for streamed time series data for later windowing

    Source node that interprets a stream of time series windows as
    raw data stream.
    The markers stored in marker_name attribute are used as the markers
    for a :class:`~pySPACE.missions.support.windower.MarkerWindower`.

    This node pretends to be a Stream2TimeSeriesSourceNode
    but takes real time series data and interprets it as a stream.

    Main functionality is implemented in the :class:`TimeSeriesDataset`
    inspired by the :class:`StreamDataset`


    **Parameters**

     :windower_spec_file:
         The specification file for the windower containing information which
         data should be windowed and which data can be discarded.

         For a detailed description look at the module description.

    :...: For further parameters check the :class:`Stream2TimeSeriesSourceNode`

    **Exemplary Call**

    .. code-block:: yaml

        -
            node: Time_Series_Stream_Source
            parameters :
                windower_spec_file : "example_lrp_window_spec.yaml"

    :Author: Hendrik Woehrle (hendrik.woehrle@dfki.de)
    :Created: 2011/08/12
    """
    def __init__(self, **kwargs):

        super(TimeSeries2TimeSeriesSourceNode, self).__init__(**kwargs)
        self.set_permanent_attributes(input_types=["time_series"])

    def get_source_file_name(self):
        """ Source file name is unknown or preprocessing specific

        .. todo:: check possibility for access source file name if possible,
                  e.g. by using metadata.yaml
        """
        pass

_NODE_MAPPING = {"Time_Series_Source": TimeSeriesSourceNode,
                "BCI_Competition_Source": BCICompetitionSourceNode,
                "EEG_Source": Stream2TimeSeriesSourceNode,
                "Offline_EEG_Source": Stream2TimeSeriesSourceNode,
                "Time_Series_Stream_Source": TimeSeries2TimeSeriesSourceNode,
                "TimeSeriesStreamSource": TimeSeries2TimeSeriesSourceNode}