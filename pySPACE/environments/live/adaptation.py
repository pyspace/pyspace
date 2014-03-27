""" Script for running threshold adaptation
"""

import os
import multiprocessing
import time
import shutil
import logging
import cPickle

online_logger = logging.getLogger("pySPACELiveLogger")

from pySPACE.environments.chains.node_chain import NodeChain, NodeChainFactory


class LiveAdaptor(object):
    """ The class that performs the threshold to a given
        cost function in order to scale the relation of
        false positives to false negatives.
    """

    def __init__(self):
        self.adaptation_active_potential = {}
        self.queue = {}
        self.pyspace_flow = {}
        self.data_stream_process = {}
        self.train_process = {}
        self.target_shown = {}
        self.last_target_data = {}
        self.window_stream = {}
        self.nullmarker_stride_ms = None

    def set_eeg_stream_manager(self, stream_manager):
        self.stream_manager = stream_manager

    def load_model(self, directory, datasets):
        """ Load only the model """
        self.directory = directory
        self.datasets = datasets
        online_logger.info( "Look for original flow...")
        # test, if a copy of the original (e.g. without threshold optimization) flow exists
        for key in self.datasets.keys():
            if "threshold_adaptation_flow" in self.datasets[key]:
                try:
                    flh = open("%s/%s.pickle" % (self.directory, "abri_flow_" + key + "_unadapted.pickle"), 'r')
                    flh.close()
                except IOError:
                    # there exists no copy of the orignal flow
                    online_logger.info("Create backup copy of original flow")
                    self.copy_flow(key)

        time.sleep(2)
        online_logger.info( "Reloading " + key + " models ... Done!")
        return 0

    def copy_flow(self, key):
        trained_flow_path = "%s/%s.pickle" % (self.directory , "train_flow_"+ key)
        prewindowing_flow_path = "%s/%s.pickle" % (self.directory , "prewindowing_flow_"+ key)
        prewindowing_offline_flow_path = "%s/%s.pickle" % (self.directory , "prewindowing_offline_flow_"+ key)
        prewindowed_train_flow_path = "%s/%s.pickle" % (self.directory , "prewindowed_train_flow_"+ key)
        # using the trained flow for adaptation
        if os.path.exists(trained_flow_path):
            shutil.copyfile(trained_flow_path, "%s/%s.pickle" % (self.directory, "abri_flow_" + key + "_unadapted"))
        # using the prewindowing flow and prewindowed-trained flow for adaptation
        else:
            flh_1 = {}
            flh_2 = {}
            prewindowing_flow = {}
            postprocessing_flow = {}
            unadapted_flow = {}
            if os.path.exists(prewindowing_flow_path):
                flh_1[key] = open(prewindowing_flow_path, 'r')
            elif os.path.exists(prewindowing_offline_flow_path):
                flh_1[key] = open(prewindowing_offline_flow_path, 'r')
            flh_2[key] = open("%s/%s.pickle" % (self.directory , "prewindowed_train_flow_"+ key), 'r')
            prewindowing_flow[key] = cPickle.load(flh_1[key])
            prewindowing_flow[key].pop(-1)
            prewindowing_flow[key].pop(-1)
            postprocessing_flow[key] = cPickle.load(flh_2[key])
            postprocessing_flow[key].pop(0)
            postprocessing_flow[key].pop(0)
            unadapted_flow[key] = prewindowing_flow[key] + postprocessing_flow[key]
            flh_1[key].close()
            flh_2[key].close()
            unadapted_file = open("%s/%s.pickle" % (self.directory, "abri_flow_" + key + "_unadapted"), 'w+')
            cPickle.dump(unadapted_flow[key], unadapted_file)

    def prepare_adaptation(self, adaptation_files, datasets, nullmarker_stride_ms = None):
        """ Prepares the threshold adaptation.
        """

        online_logger.info( "Preparing Adaptation")
        online_logger.info( "adaptation files:" + str(adaptation_files))
        
        self.nullmarker_stride_ms = nullmarker_stride_ms
        if self.nullmarker_stride_ms == None:
            online_logger.warn( 'Nullmarker stride interval is %s. You can specify it in your parameter file.' % self.nullmarker_stride_ms)
        else:
            online_logger.info( 'Nullmarker stride interval is set to %s ms' % self.nullmarker_stride_ms)
        
        for key in self.datasets.keys():
            if "threshold_adaptation_flow" in self.datasets[key]:
                spec_base = self.datasets[key]["configuration"].spec_dir
                self.datasets[key]["threshold_adaptation_flow"] = os.path.join(spec_base, self.datasets[key]["threshold_adaptation_flow"])
                online_logger.info( "windower_spec_path:" + self.datasets[key]["windower_spec_threshold_adaptation"])
                online_logger.info( "dataflow_spec_" + key + ":" + self.datasets[key]["threshold_adaptation_flow"])
                self.adaptation_active_potential[key] = multiprocessing.Value('b',False)

        # start the eeg server
        # check if multiple datasets are given for adaptation
        if hasattr(adaptation_files,'__iter__'):
            self.adaptation_data = adaptation_files
            online_logger.debug("Using multiple data sets:" + str(self.adaptation_data))
        else:
            self.adaptation_data = [adaptation_files]


        # Adaptation is done in separate threads, we send the time series
        # windows to these threads via two queues
        online_logger.info( "Initializing Queues")
        for key in self.datasets.keys():
            self.queue[key] = multiprocessing.Queue()
        online_logger.info( "Creating flows")

        def flow_generator(key):
            """create a generator to yield all the windows"""
            # Yield all windows until a None item is found in the queue
            while True:
                window = self.queue[key].get(block = True, timeout = None)
                if window == None: break
                yield window

        # Create the actual data flows for S1 vs P3 discrimination
        # and S1 vs LRP discrimination
        for key in self.datasets.keys():
            if "threshold_adaptation_flow" in self.datasets[key]:
                self.aBRI_flow[key] = NodeChainFactory.flow_from_yaml(Flow_Class = NodeChain,
                                                         flow_spec = file(self.datasets[key]["threshold_adaptation_flow"]))
                self.aBRI_flow[key][0].set_generator(flow_generator(key))

        online_logger.info( "threshold adaptation preparations finished")
        return 0

    def queue_filler(self,data,label,key):

        if label in self.datasets[key]['positive_event']:
            # if the second traget is shown without any resonse,
            # the previous target was a "Missed"
            if self.target_shown[key] == True:
                self.queue[key].put((self.last_target_data[key], self.datasets[key]['negative_event']))
                self.last_target_data[key] = data
                return
            else:
                self.last_target_data[key] = data
                self.target_shown[key] = True
        elif label in self.datasets[key]['trigger_event']:
            if self.target_shown[key] == True:
                if not self.last_target_data[key] == None:
                    self.queue[key].put((self.last_target_data[key], self.datasets[key]['positive_event']))
                self.target_shown[key] = False
        elif label in self.datasets[key]['negative_event']:
            self.queue[key].put((data, label))

    def adaptation_fct(self, key):
        """ A function that is executed in a separate thread """
        self.adaptation_active_potential[key].value = True
        self.copy_flow(key)
        online_logger.info( "Adaptation of " + key + " started")
        self.aBRI_flow[key].train()
        online_logger.info( "Adaptation of " + key + " finished")
        online_logger.info( "Storing " + key + " model...")
        self.aBRI_flow[key].save("%s/%s.pickle" % (self.directory, "abri_flow_adapted_"+ key))
        online_logger.info( key + " Model stored!")
        self.adaptation_active_potential[key].value = False

    def stream_data(self, key):
        """ A function that forwards the data to the worker threads """
        adaptation_data_set_counter = 0

        for dataset in self.adaptation_data:
            online_logger.info("Start streaming adaptation dataset " + dataset)
            # Start EEG client
            self.stream_manager.start_eeg_stream(dataset)

            # create windower
            online_logger.info( "Creating Windower")
            spec_base = self.datasets[key]["configuration"].spec_dir
            self.datasets[key]["windower_spec_threshold_adaptation"] = os.path.join(spec_base, self.datasets[key]["windower_spec_threshold_adaptation"])
            self.window_stream[key] = self.stream_manager.create_windower(self.datasets[key]["windower_spec_threshold_adaptation"], nullmarker_stride_ms = 1000)
            online_logger.info(key + " windower: " + str(self.window_stream[key]))

            window_counter  = 0

            self.target_shown[key] = False
            self.last_target_data[key] = None

            # Put all windows into the queues so that they can be processed by
            # the two adaptation threads
            online_logger.info( "Streaming data started")
            for data, label in self.window_stream[key]:
                online_logger.info( "Got instance number "+ str(window_counter) + " with class %s" % label)
                window_counter += 1
                # Skip the first few adaptation examples since there might be no
                # clear distinction between standards and targets
                if "ignore_num_first_examples" in self.datasets[key]:
                    if window_counter < self.datasets[key]["ignore_num_first_examples"]:
                        online_logger.info("Ignoring first " + str(window_counter) + " " + key + " training samples")
                        continue
                self.queue_filler(data,label,key)

            if adaptation_data_set_counter > 0:
                online_logger.info( "Dataset number" + str(adaptation_data_set_counter) + "streamed")

            adaptation_data_set_counter += 1

        online_logger.info( "Streaming data finished")
        online_logger.debug( "Submit stream end data item...")

        # Put a None into the queues to stop classification threads
        self.queue[key].put(None)

        online_logger.debug( "Stream end data item submitted")

    def start_adaptation(self):
        """ Adapts the threshold to a specified error function"""
        for key in self.datasets.keys():
            if "threshold_adaptation_flow" in self.datasets[key]:
                if not key in self.data_stream_process.keys():
                    # Stream the data
                    self.data_stream_process[key] = multiprocessing.Process(target = self.stream_data, args = (key,))
                    self.data_stream_process[key].start()
                if not key in self.train_process.keys():
                    # Start two threads for adaptation
                    self.train_process[key] = multiprocessing.Process(target = self.adaptation_fct, args = (key,))
                    self.train_process[key].start()

        for key in self.train_process.iterkeys():
            while self.train_process[key].is_alive():
                time.sleep(1)
        self.stream_manager.stop_server()
        online_logger.info("EEG manager stopped!")


    def is_adaptation_active(self):
        """ Returns whether adaptation is finished or still running """
        for key in self.datasets.keys():
            return self.adaptation_active_potential[key].value == True

    def stop_adaptation(self):
        """ Force the end of the adaptation """
        # We stop the pyspace adaptation by disconnecting the EEG stream from it
        def read(**kwargs):
            online_logger.info( "Cancelling data transfer")
            return 0

        online_logger.info( "Stopping adaptation ...")

        # Wait until pysapce has finished adaptation
        online_logger.debug( "Check if adaptation is still active ...")
        while self.is_adaptation_active():
            time.sleep(1)
            online_logger.debug( "Adaptation is still active ...")

        online_logger.info( "Adaptation finished")

        # Close the EEG client's socket
        return 0


