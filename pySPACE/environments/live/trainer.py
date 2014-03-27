""" The module that trains pyspace flows.
"""
import os
import glob
import multiprocessing
import time
import shutil
import logging
import yaml
import datetime
import re

from pySPACE.environments.chains.node_chain import NodeChain, NodeChainFactory
from pySPACE.resources.dataset_defs.base import BaseDataset
from pySPACE.resources.dataset_defs.time_series import TimeSeriesDataset
from pySPACE.environments.live import eeg_stream_manager

online_logger = logging.getLogger("OnlineLogger")


class LiveTrainer(object):
    """ The class is responsible to perform all tasks in pyspace live that
        are related to the training process of pyspace The trained
        flows are stored in the flow_storage directory.
    """

    def __init__(self, flow_storage = "flow_storage",
                 prewindowed_data_directory = "prewindowed_data_storage"):

        self.training_active_potential = {}
        # : path to storage location for node_chain defs and pickles
        self.flow_storage = flow_storage
        # : path to storage location for prewindowed data
        self.prewindowed_data_directory = prewindowed_data_directory
        # : stores node_chain definition as dictionary
        self.node_chain_definitions = {}
        # : stores executable node_chains
        self.node_chains = {}

        self.train_process = {}
        self.prewindowed_data = {}
        self.queue = {}
        self.data_stream_process = {}
        self.window_stream = {}
        self.target_shown = {}
        self.last_target_data = {}
        self.marker_windower = {}
        self.training_paused_potential = multiprocessing.Value('b',False)
        self.nullmarker_stride_ms = None

    def set_controller(self,controller):
        """ Set reference to the calling controller """
        self.controller = controller

    def set_eeg_stream_manager(self, stream_manager):
        """ Set the stream manager that provides the training data """
        self.stream_manager = stream_manager

    def prepare_training(self, training_files, potentials, operation, nullmarker_stride_ms = None):
        """ Prepares pyspace live for training.

        Prepares everything for training of pyspace live,
        i.e. creates flows based on the dataflow specs
        and configures them.
        """
        online_logger.info( "Preparing Training")
        self.potentials = potentials
        self.operation = operation
        self.nullmarker_stride_ms = nullmarker_stride_ms
        if self.nullmarker_stride_ms == None:
            online_logger.warn( 'Nullmarker stride interval is %s. You can specify it in your parameter file.' % self.nullmarker_stride_ms)
        else:
            online_logger.info( 'Nullmarker stride interval is set to %s ms ' % self.nullmarker_stride_ms)

        online_logger.info( "Creating flows..")
        for key in self.potentials.keys():
            spec_base = self.potentials[key]["configuration"].spec_dir
            if self.operation == "train":
                self.potentials[key]["node_chain"] = os.path.join(spec_base, self.potentials[key]["node_chain"])
                online_logger.info( "node_chain_spec:" + self.potentials[key]["node_chain"])

            elif self.operation in ("prewindowing", "prewindowing_offline"):
                self.potentials[key]["prewindowing_flow"] = os.path.join(spec_base, self.potentials[key]["prewindowing_flow"])
                online_logger.info( "prewindowing_dataflow_spec: " + self.potentials[key]["prewindowing_flow"])

            elif self.operation == "prewindowed_train":
                self.potentials[key]["postprocess_flow"] = os.path.join(spec_base, self.potentials[key]["postprocess_flow"])
                online_logger.info( "postprocessing_dataflow_spec: " + self.potentials[key]["postprocess_flow"])

            self.training_active_potential[key] = multiprocessing.Value("b",False)

        online_logger.info("Path variables set for NodeChains")

        # check if multiple potentials are given for training
        if isinstance(training_files, list):
            self.training_data = training_files
        else:
            self.training_data = [training_files]

        # Training is done in separate processes, we send the time series
        # windows to these threads via two queues
        online_logger.info( "Initializing Queues")
        for key in self.potentials.keys():
            self.queue[key] = multiprocessing.Queue()


        def flow_generator(key):
            """create a generator to yield all the abri flow windows"""
            # Yield all windows until a None item is found in the queue
            while True:
                window = self.queue[key].get(block = True, timeout = None)
                if window == None: break
                yield window

        # Create the actual data flows
        for key in self.potentials.keys():

            if self.operation == "train":
                self.node_chains[key] = NodeChainFactory.flow_from_yaml(Flow_Class = NodeChain,
                                                         flow_spec = file(self.potentials[key]["node_chain"]))
                self.node_chains[key][0].set_generator(flow_generator(key))
                flow = open(self.potentials[key]["node_chain"])
            elif self.operation in ("prewindowing", "prewindowing_offline"):
                online_logger.info("loading prewindowing flow..")
                online_logger.info("file: " + str(self.potentials[key]["prewindowing_flow"]))

                self.node_chains[key] = NodeChainFactory.flow_from_yaml(Flow_Class = NodeChain,
                                                             flow_spec = file(self.potentials[key]["prewindowing_flow"]))
                self.node_chains[key][0].set_generator(flow_generator(key))
                flow = open(self.potentials[key]["prewindowing_flow"])
            elif self.operation == "prewindowed_train":
                self.node_chains[key] = NodeChainFactory.flow_from_yaml(Flow_Class = NodeChain, flow_spec = file(self.potentials[key]["postprocess_flow"]))
                replace_start_and_end_markers = False

                final_collection = TimeSeriesDataset()
                final_collection_path = os.path.join(self.prewindowed_data_directory, key, "all_train_data")
                # delete previous training collection
                if os.path.exists(final_collection_path):
                    online_logger.info("deleting old training data collection for " + key)
                    shutil.rmtree(final_collection_path)

                # load all prewindowed collections and
                # append data to the final collection
                prewindowed_sets = \
                    glob.glob(os.path.join(self.prewindowed_data_directory, key, "*"))
                if len(prewindowed_sets) == 0:
                    online_logger.error("Couldn't find data, please do prewindowing first!")
                    raise Exception
                online_logger.info("concatenating prewindowed data from " + str(prewindowed_sets))

                for s,d in enumerate(prewindowed_sets):
                    collection = BaseDataset.load(d)
                    data = collection.get_data(0, 0, "train")
                    for d,(sample,label) in enumerate(data):
                        if replace_start_and_end_markers:
                            # in case we concatenate multiple 'Window' labeled
                            # sets we have to remove every start- and endmarker
                            for k in sample.marker_name.keys():
                                # find '{S,s}  8' or '{S,s}  9'
                                m = re.match("^s\s{0,2}[8,9]{1}$", k, re.IGNORECASE)
                                if m is not None:
                                    online_logger.info(str("remove %s from %d %d" % (m.group(), s, d)))
                                    del(sample.marker_name[m.group()])

                            if s == len(prewindowed_sets)-1 and \
                                d == len(data)-1:
                                # insert endmarker
                                sample.marker_name["S  9"] = [0.0]
                                online_logger.info("added endmarker" + str(s) + " " + str(d))

                            if s == 0 and d == 0:
                                # insert startmarker
                                sample.marker_name["S  8"] = [0.0]
                                online_logger.info("added startmarker" + str(s) + " " + str(d))

                        final_collection.add_sample(sample, label, True)

                # save final collection (just for debugging)
                os.mkdir(final_collection_path)
                final_collection.store(final_collection_path)

                online_logger.info("stored final collection at " + final_collection_path)

                # load final collection again for training
                online_logger.info("loading data from " + final_collection_path)
                self.prewindowed_data[key] =  BaseDataset.load(final_collection_path)
                self.node_chains[key][0].set_input_dataset(self.prewindowed_data[key])

                flow = open(self.potentials[key]["postprocess_flow"])

            # create window_stream for every potential

            if self.operation in ("prewindowing"):
                window_spec_file = os.path.join(spec_base,"node_chains","windower",
                             self.potentials[key]["windower_spec_path_train"])

                self.window_stream[key] = \
                        self.stream_manager.request_window_stream(window_spec_file,
                                                              nullmarker_stride_ms = self.nullmarker_stride_ms)
            elif self.operation in ("prewindowing_offline"):
                pass
            elif self.operation in ("train"):
                pass

            self.node_chain_definitions[key] = yaml.load(flow)
            flow.close()

        # TODO: check if the prewindowing flow is still needed when using the stream mode!
        if self.operation in ("train"):
            online_logger.info( "Removing old flows...")
            try:
                shutil.rmtree(self.flow_storage)
            except:
                online_logger.info("Could not delete flow storage directory")
            os.mkdir(self.flow_storage)
        elif self.operation in ("prewindowing", "prewindowing_offline"):
            # follow this policy:
            # - delete prewindowed data older than 12 hours
            # - always delete trained/stored flows
            now = datetime.datetime.now()
            then = now - datetime.timedelta(hours=12)

            if not os.path.exists(self.prewindowed_data_directory):
                os.mkdir(self.prewindowed_data_directory)
            if not os.path.exists(self.flow_storage):
                os.mkdir(self.flow_storage)

            for key in self.potentials.keys():
                found = self.find_files_older_than(then, \
                        os.path.join(self.prewindowed_data_directory, key))
                if found is not None:
                    for f in found:
                        online_logger.info(str("recursively deleting files in \'%s\'" % f))
                        try:
                            shutil.rmtree(os.path.abspath(f))
                        except Exception as e:
                            # TODO: find a smart solution for this!
                            pass # dir was probably already deleted..

                if os.path.exists(os.path.join(self.prewindowed_data_directory, key, "all_train_data")):
                    shutil.rmtree(os.path.join(self.prewindowed_data_directory, key, "all_train_data"))
                    online_logger.info("deleted concatenated training data for " + key)


        online_logger.info( "Training preparations finished")
        return 0

    def find_files_older_than(self, then, dir):
        # recursively find files in 'dir' which are older
        # then 'date' and add their basepath to 'found'

        found = None
        for r,d,f in os.walk(dir):
            for file in f:
                if file.startswith("."):
                    continue
                abs_file = os.path.abspath(os.path.join(r, file))
                if os.path.getmtime(abs_file) < time.mktime(then.timetuple()):
                    if found is None:
                        found = list()
                    print f, " -> adding -> ", r
                    found.append(r)

        online_logger.info(str("pathes to delete: %s" % found))
        return found

    def training_fct(self, key):
        """ Function that performs the real training """
        self.training_active_potential[key].value= True
        online_logger.info( key + " " + self.operation + " started")

        if self.operation in ("train", "prewindowed_train"):
            self.node_chains[key].train()

        elif self.operation in ("prewindowing", "prewindowing_offline"):
            result_collection = {}
            self.node_chains[key][-1].process_current_split()

            result_collection[key] = self.node_chains[key][-1].get_result_dataset()
            save_dir = os.path.abspath(self.prewindowed_data_directory + os.path.sep + key)
            if not os.path.exists(save_dir):
                os.mkdir(os.path.abspath(self.prewindowed_data_directory + os.path.sep + key))

            if result_collection[key] != None:
                online_logger.info("storing result collection for " + key)

                now = datetime.datetime.now()
                now_folder = str("%04d%02d%02d-%02d%02d%02d" % \
                    (now.year, now.month, now.day, now.hour, now.minute, now.second))

                p = os.path.abspath(os.path.join(self.prewindowed_data_directory, \
                                        key, now_folder))
                if not os.path.exists(p):
                    os.mkdir(p)

                result_collection[key].store(p)
                online_logger.info( key + " Prewindowed data stored!")

            else:
                online_logger.warn(str("result-collection for %s was None - nothing stored.." % key))

        online_logger.info( key + " " + self.operation + " finished")
        online_logger.info( "Storing " + key +" flow model...")
        self.node_chains[key].save("%s/%s.pickle" % (self.flow_storage, self.operation + "_flow_"+ key))
        f = open('%s/%s.yaml' % (self.flow_storage, self.operation +"_flow_"+ key),"w")
        yaml.dump(self.node_chain_definitions[key], f, default_flow_style=False)
        f.close()
        online_logger.info( key + " Flow Model stored!")
        self.training_active_potential[key].value = False


    def triggered_queue_filler_training(self,data,label, key):

        if label in self.potentials[key]['positive_event']:
            self.target_shown[key] = True
            self.last_target_data[key] = data
        elif label in self.potentials[key]['trigger_event']:
            if self.target_shown[key] == True:
                self.queue[key].put((self.last_target_data[key], self.potentials[key]['positive_event']))
                self.target_shown[key] = False
        elif label in self.potentials[key]['negative_event']:
            self.queue[key].put((data, label))


    def classification_thread(self, key):
        """ Thread that processes external training commands """
        window_counter = 0
        active = False

        for data, label in self.window_stream[key]:
            if self.training_paused_potential.value == True:
                break
            online_logger.info("Got instance number "+ str(window_counter) + " with class %s" % label)
            window_counter += 1
            # Skip the first few training examples since there might be no
            # clear distinction between standards and targets
            if "ignore_num_first_examples" in self.potentials[key]:
                if window_counter < int(self.potentials[key]["ignore_num_first_examples"]):
                    online_logger.info("Ignoring first " + str(window_counter) + " " + key + " training samples")
                    continue

            if self.potentials[key].has_key("trigger_event"):
                self.triggered_queue_filler_training(data, label, key)

            # distribution is performed only if it is activated beforehand
            elif self.potentials[key].has_key("activation_label"):
                if label in self.potentials[key]["activation_label"]:
                    online_logger.warn("Detection of " + key + "started")
                    active = True

                if label in self.potentials[key]["positive_event"] and active:
                    self.event_queue.put(self.potentials[key]["positive_event"])
                    self.queue[key].put((data, label))

                if label in self.potentials[key]["deactivation_label"]:
                    online_logger.warn("Detection of " + key + "stopped")
                    active = False


            else:
                if label in self.potentials[key]["positive_event"]:
                    self.queue[key].put((data, label))
                elif label in self.potentials[key]["negative_event"]:
                    self.queue[key].put((data, label))

        online_logger.info( "Streaming data finished")
        online_logger.debug("Submit stream end data item...")
        self.queue.put(None)
        online_logger.debug("Stream end data item submitted")

    def stream_data(self, key):
        """ A function that forwards the data to the worker threads """
        spec_base = self.potentials[key]["configuration"].spec_dir
        window_spec_file = {}
 
        if self.operation in ("prewindowing"):
            online_logger.info(str("streaming data for %s started" % key))
            
            self.classification_thread(key)
 
            # all done!
            self.queue[key].put(None)
            online_logger.info(str("%s for %s finished" % (self.operation, key)))
 
        elif self.operation in ("prewindowing_offline"):
            data_set_count = 0
 
            # create local stream manager
            local_streaming = eeg_stream_manager.LiveEegStreamManager(online_logger)
 
            for train_dataset in self.training_data:
                # continue if we are not supposed to train any further
                if self.training_paused_potential.value == True:
                    continue
 
                # stream local file
                local_streaming.stream_local_file(train_dataset)
 
                # create window stream
                window_spec_file[key] = os.path.join(spec_base, 
                                                    "node_chains", 
                                                    "windower", 
                                                    self.potentials[key]["windower_spec_path_train"])
                     
                self.window_stream[key] = local_streaming.request_window_stream(window_spec_file[key], \
                                         nullmarker_stride_ms=self.nullmarker_stride_ms)
                # process the data
                online_logger.info(str("streaming data for %s started" % key))
                self.classification_thread(key)
                
                data_set_count += 1
                online_logger.info(str("dataset %d completely streamed for %s" % (data_set_count, key)))
                local_streaming.stop()
 
            # all done!
            self.queue[key].put(None)
            online_logger.info(str("training for %s finished" % key))
 
        elif self.operation in ("train"):
            data_set_count = 0
            local_streaming = eeg_stream_manager.LiveEegStreamManager(online_logger)
            for train_dataset in self.training_data:
                if self.training_paused_potential.value == True:
                    continue
                online_logger.info("Start streaming training dataset " + train_dataset)
                # Start EEG client
                local_streaming.stream_local_file(train_dataset)
                # create windower
                
                
                print 'window specs: ' , self.potentials[key]["windower_spec_path_train"]
                 
                window_spec_file[key] = \
                    os.path.join(spec_base, 
                                "node_chains", 
                                "windower", 
                                self.potentials[key]["windower_spec_path_train"])
                     
                self.window_stream[key] = \
                    local_streaming.request_window_stream(window_spec_file[key], \
                    nullmarker_stride_ms=self.nullmarker_stride_ms)
                         
                online_logger.info(key + " windower: " + str(self.window_stream[key]))
                self.classification_thread(key)
                data_set_count += 1
                online_logger.info(str("dataset %d completely streamed for %s" % (data_set_count, key)))
                    #local_streaming.stop()
 
            self.queue[key].put(None)

    def start_training(self, operation, profiling=False):
        """ Trains flows on the streamed data """

        for key in self.potentials.keys():
            assert(not self.training_active_potential[key].value == True)

            # Stream the data
            if self.operation in ("train", "prewindowing", "prewindowing_offline"):
                self.data_stream_process[key] = multiprocessing.Process(target = self.stream_data, args = (key,))
                self.data_stream_process[key].start()
                time.sleep(0.1)
            else:
                pass

            if not key in self.train_process.keys():
                # Start multiple threads for training
                self.train_process[key] = multiprocessing.Process(target = self.training_fct, args = (key,))
                #start all processes
                self.train_process[key].start()


        self.training_paused_potential.value = False
        # wait until training processes are set up and running
        # they should run after 30s
        setup_timer = 0
        while not self.is_training_active():
            if setup_timer > 30:
                online_logger.error("Training processes not started")
                raise RuntimeError("Training processes not started")
            else:
                time.sleep(1)
                setup_timer+=1
        if self.operation in ("train", "prewindowing", "prewindowing_offline", "prewindowed_train"):
            for key in self.train_process.iterkeys():
                while self.train_process[key].is_alive():
                    time.sleep(1)

    def is_training_active(self):
        """ Returns whether training is finished or still running """
        active = False
        alive = False

        for key in self.potentials.keys():
            active |= self.training_active_potential[key].value

        for key in self.train_process.iterkeys():
            alive |=  self.train_process[key].is_alive()

        if not alive:
            active = False

        return active

    def process_external_command(self, command):
        """ Process external stop command """
        if command == "STOP":
            for key in self.data_stream_process.keys():
                self.data_stream_process[key].terminate()
            self.pause_training()

    def pause_training(self):
        """ Pause the training phase """
        self.training_paused_potential.value = True

    def stop_training(self):
        """ Force the end of the training """
        # We stop the training by disconnecting the data stream from it
        def read(**kwargs):
            online_logger.info( "Canceling EEG transfer")
            return 0

        online_logger.info( "Stopping training ...")

        # Wait until training has finished
        for key in self.potentials.keys():
            online_logger.info("Check if training is still active ...")
            while self.is_training_active():
                time.sleep(1)
                online_logger.info("Training is still active ...")
            self.train_process[key].join()

        online_logger.info("Training finished")

        return 0
