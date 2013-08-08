""" Script to run the actual online classification of data
"""
import os
import logging.handlers
import multiprocessing
import time
import numpy
import warnings
import cPickle
import xmlrpclib
import sys

online_logger = logging.getLogger("OnlineLogger")

from pySPACE.environments.live import eeg_stream_manager
import pySPACE.environments.live.communication.socket_messenger
from pySPACE.tools.logging_stream_colorer import ColorFormatter, COLORS


class SimpleResultCollection(object):

    """ Base Class for Result collection.

    Default behaviour is counting occurring events in
    a dictionary.

    **Parameters**

        :name: potential-name (e.g. LRP)
        :params: parameter/meta-data of the potential
        :store: write result to a pickle file

    """

    def __init__(self, name, params, store=True):
        self.result = None
        self.name = name
        self.params = params
        self.store = store

        online_logger.info(str(self))

    def __repr__(self):
        return str("simple result collection for %s (%s) " % (self.name, self.params))

    def event_notification(self, event_str):
        """ simple event counting """
        if self.result is None:
            self.result = dict()
        if event_str not in self.result.keys():
            self.result[event_str] = 0
        self.result[event_str] += 1

    def dump(self):
        """ simple print function """
        online_logger.info(str("result for %s: %s" % (self.name, str(self.result))))

        # write the result to the file system
        if self.store:
            f = open(str("%s.result" % self.name), 'w')
            cPickle.dump(self.result, f)
            f.close()

class ConfusionMatrix(SimpleResultCollection):
    """ A confusion matrix.

    Stores and handles a confusion matrix.
    The confusion matrix is assumed to have the following form:

        +--------+---+-----+------+
        |            | prediction |
        +            +-----+------+
        |            | P   | N    |
        +========+===+=====+======+
        | actual | P | TP  | FN   |
        +        +---+-----+------+
        |        | N | FP  | TN   |
        +--------+---+-----+------+

     **Parameters**

         :name: potential-name (e.g. LRP)
         :params: parameter/meta-data of the potential

     """

    def __init__(self, name, params):
        super(ConfusionMatrix, self).__init__(name, params)
        self.last_result = None

    def __repr__(self):
        return str("confusing matrix for %s (%s) " % (self.name, self.params))

    def event_notification(self, event_str):
        """ Update the confusion matrix

        It is assumed that we receive a trigger event
        *after* the classification result (either pos. or neg.-event).
        If the trigger_event occurrs it is a validation
        for a positive prediction.
        If no trigger_event but instead another
        classification result comes in it is assumed
        that no reaction on the previously presented
        target appeared.

        """

        # online_logger.info("Got event with event string:" + str(event_str))


        if self.result is None:
            self.result = numpy.zeros((2,2))

        if self.params.has_key("trigger_event"):

            # if current event is a trigger event we use
            # it as a validation for the last prediction
            if event_str == self.params["trigger_event"]:
                if self.last_result == self.params["negative_event"]:
                    self.result[1,0] += 1
                elif self.last_result == self.params["positive_event"]:
                    self.result[0,0] += 1
                elif self.last_result is None:
                    online_logger.warn("Received trigger event without positive event!")
                    online_logger.warn("Possible reasons:")
                    online_logger.warn("- Subject reacted although no target was presented")
                    online_logger.warn("- The streaming is much faster than realtime and the prediction appeared later than the respone.")
                    online_logger.warn("- Marker for trigger event appeared twice.")

                self.last_result = None

            # current event is a classification result:
            # - either infer, that no reaction occurred
            # - or just store it
            if event_str == self.params["positive_event"] or \
                    event_str == self.params["negative_event"]:

                if self.last_result == self.params["positive_event"]:
                    self.result[0,1] += 1
                elif self.last_result == self.params["negative_event"]:
                    self.result[1,1] += 1
                elif self.last_result is not None:
                    online_logger.info(str("unknwon event type: event_str=%s last_result=%s" % (event_str, self.last_result[key])))

                # store it
                self.last_result = event_str

    def dump(self):
        """
        prints the collected result of the confusion matrix

        """
        super(ConfusionMatrix, self).dump()

        online_logger.info(str("Confusion matrix for %s:" % self.name))
        online_logger.info(str("+--------+---+------+------+"))
        online_logger.info(str("|            |  prediction |"))
        online_logger.info(str("+            +------+------+"))
        online_logger.info(str("|            | P    | N    |"))
        online_logger.info(str("+========+===+======+======+"))
        online_logger.info(str("| actual | P | %4d | %4d |" % (self.result[0,0], self.result[1,0])))
        online_logger.info(str("+        +---+------+------+"))
        online_logger.info(str("|        | N | %4d | %4d |" % (self.result[0,1], self.result[1,1])))
        online_logger.info(str("+--------+---+------+------+"))


class Predictor(object):
    """ Class that is responsible to perform the actual predictions.
    """

    def __init__(self, live_processing = None, configuration = None):

        self.configuration = configuration

        self.predicting_active_potential = {}
        self.abri_flow = {}
        self.prewindowing_flow = {}
        self.postprocessing_flow = {}

        # init the message handling
        self.event_queue = dict() # multiprocessing.Queue()
        self.command_queue = multiprocessing.Queue()

        # initialize the live_processing
        if live_processing == None:
            self.messenger = pySPACE.environments.live.communication.socket_messenger.SocketMessenger()
        else:
            self.messenger = live_processing

        self.create_processor_logger()

        self.predict_process = {}
        self.predicting_fct_stream_data_process = {}

        self.event_notification_process = None

        self.controller_host = None
        self.controller_port = None

        self.mars_host = None
        self.mars_port = None

        self.queue = {}

        self.stream_manager = None

        self.window_stream = {}

    def __del__(self):
        self.messenger.end_transmission()

    def initialize_xmlrpc(self,
                          controller_host,
                          controller_port,
                          mars_host = '127.0.0.1',
                          mars_port = 8080):
        """ Setup communication to remote listeners

        This method tells ABRIProcessing which remote processes are interested
        in being informed about its classification results.
        """

        # Create Server Proxy for control center
        self.controller_host = controller_host
        self.controller_port = controller_port
        self.controller = xmlrpclib.ServerProxy('http://%s:%s' % (controller_host,
                                                                  controller_port))

        # Create Server Proxy for MARS simulation
        self.mars_host = mars_host
        self.mars_port = mars_port
        self.marsXmlRpcServer = \
            xmlrpclib.ServerProxy('http://%s:%s' % (mars_host,
                                                    mars_port))

    def set_eeg_stream_manager(self, stream_manager):
        """ Set manager class that provides the actual data for the prediction """
        self.stream_manager = stream_manager

    def load_model(self, directory, datasets):
        """ Store the learned models """
        online_logger.info( "Reloading learned models ...")
        self.datasets = datasets
        for key in self.datasets.keys():
            adapted_flow_path = "%s/%s.pickle" % (directory , "abri_flow_adapted_"+ key)
            trained_flow_path = "%s/%s.pickle" % (directory , "train_flow_"+ key)
            trained_flow_path_svm_model = "%s/%s.model" % (directory , "train_flow_"+ key)
            prewindowing_flow_path = "%s/%s.pickle" % (directory , "prewindowing_flow_"+ key)
            prewindowing_offline_flow_path = "%s/%s.pickle" % (directory , "prewindowing_offline_flow_"+ key)
            # check if adapted flow exists, if yes, use it for prediction
            if os.path.exists(adapted_flow_path):
                flh = {}
                flh[key] = open(adapted_flow_path, 'r')
                self.abri_flow[key] = cPickle.load(flh[key])
                flh[key].close()
                online_logger.info( "Predicting using adapted " + key +" flow...")
            # check if trained flow exists, if yes, use it for prediction
            elif os.path.exists(trained_flow_path):
                flh = {}
                flh[key] = open(trained_flow_path, 'r')
                self.abri_flow[key] = cPickle.load(flh[key])
                flh[key].close()
                online_logger.info( "Predicting using trained " + key +" flow...")
            # try to get the flow from prewindowing flow and postprocessing flow
                if os.path.exists(trained_flow_path_svm_model):
                    self.abri_flow[key][-1].load_model(trained_flow_path_svm_model)
                    online_logger.info( "Predicting using trained " + key +" flow...")
            else:
                flh_1 = {}
                flh_2 = {}
                if os.path.exists(prewindowing_flow_path):
                    flh_1[key] = open(prewindowing_flow_path, 'r')
                elif os.path.exists(prewindowing_offline_flow_path):
                    flh_1[key] = open(prewindowing_offline_flow_path, 'r')
                flh_2[key] = open("%s/%s.pickle" % (directory , "prewindowed_train_flow_"+ key), 'r')
                self.prewindowing_flow[key] = cPickle.load(flh_1[key])
                self.prewindowing_flow[key].pop(-1)
                self.prewindowing_flow[key].pop(-1)
                self.postprocessing_flow[key] = cPickle.load(flh_2[key])
                self.postprocessing_flow[key].pop(0)
                self.postprocessing_flow[key].pop(0)
                self.abri_flow[key] = self.prewindowing_flow[key] + self.postprocessing_flow[key]
                flh_1[key].close()
                flh_2[key].close()
                online_logger.info( "Predicting using prewindowed trained " + key +" flow...")

        time.sleep(5)
        online_logger.info( "Reloading learned models ... Done!")
        return 0

    def prepare_predicting(self, datasets, testing_data=None):
        """Prepares the trained aBRI-DP flows to classify new instances.
        """

        self.messenger.register()

        if testing_data is not None:
            if self.stream_manager is not None:
                online_logger.warn("deleting stream manager %s - this should not happen" % self.stream_manager)
                self.stream_manager = None

            self.stream_manager = eeg_stream_manager.LiveEegStreamManager(online_logger)
            self.stream_manager.stream_local_file(testing_data)


        # create window streams for all potentials
        spec_base = self.configuration.spec_dir
        for key in self.datasets.keys():
            online_logger.info( "Creating " + key + " windower stream")
            window_spec = os.path.join(spec_base,"node_chains","windower", self.datasets[key]["windower_spec_path_prediction"])
            if self.datasets[key].has_key("stream") and self.datasets[key]["stream"] == True:
                self.window_stream[key] = \
                    self.stream_manager.request_window_stream(window_spec, nullmarker_stride_ms=50, no_overlap = True)
            else:
                self.window_stream[key] = \
                    self.stream_manager.request_window_stream(window_spec, nullmarker_stride_ms=50)
        # Classification is done in separate threads, we send the time series
        # windows to these threads via two queues
        for key in self.datasets.keys():
            self.queue[key] = multiprocessing.Queue()
            self.predicting_active_potential[key] = multiprocessing.Value("b",False)
        self.predicting_paused_potential = multiprocessing.Value('b',False)

        # The two classification threads access the two queues via two
        # generators
        def flow_generator(key):
            """create a generator to yield all the abri flow windows"""
            # Yield all windows until a None item is found in the queue
            while True:
                window = self.queue[key].get(block = True, timeout = None)
                if window == None: break
                yield window

        for key in self.datasets.keys():
            self.abri_flow[key][0].set_generator(flow_generator(key))

        return 0

    def start_predicting(self, trace = False):
        """ Classify new instances based on the learned aBRI-DP flows. """

        if trace:
            for key in self.datasets.keys():
                for node in self.abri_flow[key]:
                    node.trace = True

        def handle_event_notification(key):
            online_logger.info(str("handling event notification for %s" % key))
            if self.datasets[key].has_key("trigger_event"):
                result_collector = ConfusionMatrix(name=key, params=self.datasets[key])
            else:
                result_collector = SimpleResultCollection(name=key, params=self.datasets[key])

            event = self.event_queue[key].get(block = True, timeout = None)
            while event != None:
                result_collector.event_notification(event)
                event = self.event_queue[key].get(block = True, timeout = None)

            result_collector.dump()

        def predicting_fct(key):
            """ A function that is executed in a separate thread, in which pyspace detects whether a
                target is perceived or not and put them in the event queue
            """

            self.predicting_active_potential[key].value = True
            online_logger.debug(key +" detection process started")
            for result in self.abri_flow[key].execute():
                if self.predicting_paused_potential.value:
                    continue
                if not self.datasets[key].get("messenger",True):
                    continue
                if self.datasets[key].has_key("trigger_event"):
                    self.messenger.send_message((key, result[0].label in self.datasets[key]["positive_event"]))
                    if str(result[0].label) in self.datasets[key]["positive_event"]:
                        self.event_queue[key].put(self.datasets[key]["positive_event"])
                    else:
                        self.event_queue[key].put(self.datasets[key]["negative_event"])

                    online_logger.info("Classified target as " + str(result[0].label) + " with score " + str(result[0].prediction))

                else:
                    self.messenger.send_message((key,result[0].prediction))

                    if str(result[0].label) == self.datasets[key]["positive_event"]:
                        self.lrp_logger.info("Classified movement window as "
                                              + str(result[0].label) + " with score " + str(result[0].prediction))
                        self.event_queue[key].put(self.datasets[key]["positive_prediction"])
                    else:
                        self.no_lrp_logger.info("Classified movement window as "
                                               + str(result[0].label) + " with score " + str(result[0].prediction))
                        self.event_queue[key].put(self.datasets[key]["negative_prediction"])


            # when finished put a none in the event queue
            self.event_queue[key].put(None)
            self.predicting_active_potential[key].value = False
            online_logger.info(str("predicition of %s finished!" % key))


        def predicting_fct_stream_data(key):
            """ A function that decides whether the window stream in p3 is a response or a
            NoResponse or a Standard and put them in an event queue
            """

            active = True
            visualize = False

            # distribute all windows to the responsible flows
            for data, label in self.window_stream[key]:
                if self.predicting_paused_potential.value:
                    continue

                # distribution is performed according to different preconditions

                # detection is performed if there is a preceding trigger event
                if self.datasets[key].has_key("trigger_event"):
                    if label in self.datasets[key]["trigger_event"]:
                        self.event_queue[key].put(self.datasets[key]["trigger_event"])
                    else:
                        self.queue[key].put((data, label))

                # switch detection on or off depending on activation label
                elif self.datasets[key].has_key("activation_label"):
                    if label in self.datasets[key]["activation_label"]:
                        online_logger.info("Detection of " + key + "started")
                        active = True
                    elif label in self.datasets[key]["deactivation_label"]:
                        online_logger.info("Detection of " + key + "stopped")
                        active = False

                    if label in self.datasets[key]["positive_event"] and active:
                        self.event_queue[key].put(self.datasets[key]["positive_event"])
                        self.queue[key].put((data, label))
                    time.sleep(0.1)

                # just put data into the flow
                else:
                    self.queue[key].put((data, label))

            # Put a None into the data-queue to stop classification threads
            self.queue[key].put(None)

            # self.predicting_active_potential[key].value = False

            online_logger.info("Finished stream data " + key)



        online_logger.info( "Starting Evaluation")

        # Start two threads for predicting
        for key in self.datasets.keys():
            if not key in self.event_queue.keys():
                self.event_queue[key] = multiprocessing.Queue()

            if not key in self.predict_process.keys():
                self.predict_process[key] = \
                    multiprocessing.Process(target = predicting_fct, args = (key,))
                self.predict_process[key].start()

            if not key in self.predicting_fct_stream_data_process.keys():
                self.predicting_fct_stream_data_process[key] = \
                    multiprocessing.Process(target = predicting_fct_stream_data, args = (key,))
                self.predicting_fct_stream_data_process[key].start()

        self.predicting_paused_potential.value = False

        if not self.event_notification_process:
            self.event_notification_process = dict()
            for key in self.datasets.keys():
                self.event_notification_process[key] = \
                    multiprocessing.Process(target = handle_event_notification, args=(key,))
                self.event_notification_process[key].start()

        return 0
        # Put all windows into the queues so that they can be processed by
        # the two classification threads

    def is_predicting_active(self):
        """ Returns whether prediction phase is finished or still running """
        for key in self.datasets.keys():
            return self.predicting_active_potential[key].value == True #or self.predicting_active_LRP.value == True

    def process_external_command(self, command):
        if command == "STOP":
            self.pause_prediction()

    def pause_prediction(self):
        self.predicting_paused_potential.value = True

    def stop_predicting(self):
        """ Force the end of the predicting """
        # We stop the aBRI-DP training by disconnecting the EEG stream from it
        def read(**kwargs):
            online_logger.info( "Canceling EEG transfer")
            return 0

        online_logger.info( "Stopping predicting ...")

        # Wait until aBRI-DP has finished predicting
        for key in self.datasets.keys():
            self.predict_process[key].join()
            self.predicting_fct_stream_data_process[key].join()

        self.event_notification_process.join()
        online_logger.info("Prediction finished")

        return 0

    def set_controller(self,controller):
        """ Set reference to the controller """
        self.controller = controller

    def create_processor_logger(self):
        """ Create specific logger for the prediction """
        # Setting up log level
        # create a logger
        # create logger for test output
        self.lrp_logger = logging.getLogger('abriOnlineProcessorLoggerForLrps')
        self.lrp_logger.setLevel(logging.DEBUG)
        self.no_lrp_logger = logging.getLogger('abriOnlineProcessorLoggerForNoLrps')
        self.no_lrp_logger.setLevel(logging.DEBUG)

        formatterResultsStreamNoLrp = ColorFormatter("%(asctime)s - %(name)s: %(message)s",
                                                color = COLORS.RED)
        formatterResultsStreamLrp = ColorFormatter("%(asctime)s - %(name)s: %(message)s",
                                                color = COLORS.GREEN)

        formatterResultsFile = logging.Formatter("%(asctime)s - %(name)s: %(message)s")
        loggingFileHandlerResults = logging.handlers.TimedRotatingFileHandler("log"+os.path.sep+ \
            "prediction_lrp.log",backupCount=5)

        loggingStreamHandlerResultsNoLrp = logging.StreamHandler()
        loggingStreamHandlerResultsNoLrp.setFormatter(formatterResultsStreamNoLrp)

        loggingStreamHandlerResultsLrp = logging.StreamHandler()
        loggingStreamHandlerResultsLrp.setFormatter(formatterResultsStreamLrp)

        loggingFileHandlerResults.setFormatter(formatterResultsFile)
        loggingStreamHandlerResultsLrp.setLevel(logging.DEBUG)
        loggingStreamHandlerResultsNoLrp.setLevel(logging.DEBUG)
        loggingFileHandlerResults.setLevel(logging.DEBUG)

        self.lrp_logger.addHandler(loggingStreamHandlerResultsLrp)
        self.no_lrp_logger.addHandler(loggingStreamHandlerResultsNoLrp)
        self.lrp_logger.addHandler(loggingFileHandlerResults)
        self.no_lrp_logger.addHandler(loggingFileHandlerResults)

