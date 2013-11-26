""" Script for running pyspace live controlling

.. image:: ../../graphics/launch_live.png
   :width: 500


A script for running pyspace live. The script contains
a class to control the other related classes needed in the online mode,
and several methods that are used for the general startup of the suite.
"""

import sys
import os
import time
import traceback
import logging
import yaml
import datetime
import optparse
import multiprocessing

file_path = os.path.dirname(os.path.realpath(__file__))
pyspace_path = file_path[:file_path.rfind('pySPACE')-1]
if not pyspace_path in sys.path:
    sys.path.append(pyspace_path)

import pySPACE

# create logger with handlers
from pySPACE.environments.live import online_utilities

online_logger = logging.getLogger("OnlineLogger")

class LiveController(object):
    """ Controlling suite.

    This class provides a clean interface to the live environment.
    It provides contains objects of the classes that are used
    for the online mode and configures them as needed.

    The controller uses the config-files for user related configuration,
    and additional parameter files for scenario/task specific parameterization.

    """
    def __init__(self,
                    parameters,
                    configuration,
                    live_processing = None):

        # fetch mandatory parameters
        datafile_info = parameters["data_files"]
        eeg_server_info = parameters["eeg_server"]
        live_server_info = parameters["live_server"]
        potentials = parameters["potentials"]
        flow_persistency_directory = parameters["flow_persistency_directory"]

        # try to fetch optional parameters, set them to None if not present
        try:
            prewindowed_data_directory = parameters["prewindowed_data_directory"]
        except:
            prewindowed_data_directory = None

        self.datafile_train = \
            datafile_info["eeg_data_file_train"]
        self.datafile_test = \
            datafile_info["eeg_data_file_test"]

        self.eeg_server_train_ip = \
            eeg_server_info["eeg_server_train_ip"]
        self.eeg_server_predict_ip = \
            eeg_server_info["eeg_server_predict_ip"]
        self.eeg_server_offline_predict_ip = \
            eeg_server_info["eeg_server_offline_predict_ip"]
        self.eeg_server_eeg_port = \
            eeg_server_info["eeg_server_eeg_port"]

        # check if recording data is set
        try:
            self.subject = parameters["record"]["subject"]
            self.experiment = parameters["record"]["experiment"]
        except:
            self.subject = None
            self.experiment = None


        # try to fetch optional parameters, set them to None if not present
        try:
            self.eeg_server_prewindow_ip = \
                eeg_server_info["eeg_server_prewindow_ip"]
        except:
            self.eeg_server_prewindow_ip = None

        self.live_server_ip = \
            live_server_info["live_server_ip"]
        self.live_xmlrpc_port = \
            live_server_info["live_xmlrpc_port"]

        self.flow_persistency_directory = \
            flow_persistency_directory
        self.prewindowed_data_directory = \
            prewindowed_data_directory

        self.configuration = configuration

        self.label = None

        self.prediction_process = None
        self.live_processing = None
        self.live_prewindower = None

        if live_processing == None:
            self.messenger = pySPACE.environments.live.communication.log_messenger.LogMessenger()
        else:
            self.messenger = live_processing

        # figure out all potentials and
        # store all relevant information
        self.erps = dict()
        if not isinstance(potentials, list):
            for (_, potential) in potentials.iteritems():
                potential["configuration"] = self.configuration
                self.erps[potential["flow_id"]] = potential
        else:
            for potential in potentials:
                potential["configuration"] = self.configuration
                online_logger.info(potential)
                self.erps[potential["flow_id"]] = potential


    def prewindowing(self, online = True):
        """ Prewindows the pyspace flows on the data streamed from
            an external EEG-Server
        """

        online_logger.info("starting prewindowing")
        # Create prewindower
        self.live_prewindower = trainer.LiveTrainer()
        # online_logger.info(self.erps)
        prewindowing_files = []

        if online:
            # create the stream manager
            stream_manager = eeg_stream_manager.LiveEegStreamManager(online_logger, self.configuration)
            stream_manager.initialize_eeg_server(self.eeg_server_prewindow_ip,
                                                 self.eeg_server_eeg_port)
            # setup recording if option is set
            if self.subject is not None and \
                    self.experiment is not None:
                stream_manager.record_with_options(self.subject, self.experiment)
            else:
                online_logger.error("RAW DATA IS NOT RECORDED!")

            # set the stream manager into the trainer object
            self.live_prewindower.set_eeg_stream_manager(stream_manager)

            # in online case just connect to the streaming server
            self.live_prewindower.prepare_training(prewindowing_files,
                                                   self.erps,
                                                   "prewindowing")



        else:
            # when running offline prepare local streaming
            if isinstance(self.datafile_train, str):
                prewindowing_files = \
                    os.path.join(self.configuration.storage, self.datafile_train)
            else:
                for datafile in self.datafile_train:
                    if os.path.isabs(datafile):
                        prewindowing_files = prewindowing_files + [datafile]
                    else:
                        prewindowing_files = prewindowing_files + \
                            [os.path.join(self.configuration.storage, datafile)]

            online_logger.info("prewindowing files:")
            online_logger.info(prewindowing_files)

            self.live_prewindower.prepare_training(prewindowing_files,
                                          self.erps,
                                          "prewindowing_offline")
        self.start_prewindowing(online)

    def start_prewindowing(self, online = True):
        """ Start the prewindowing process """
        online_logger.info("Start prewindowing")
        self.live_prewindower.start_training("prewindowing") # pass an additional True for profiling

    def stop_prewindowing(self):
        """ Create pyspace live processing server """
        self.live_prewindower.process_external_command("STOP")


    def prewindowed_train(self):
        """ Trains the pyspace flows which have been prewindowed using the prewindower"""
        # Create trainer and initialize the eeg data stream
        pw_trainer = trainer.LiveTrainer()

        postprocessing_files = []
        pw_trainer.prepare_training(postprocessing_files,
                                  self.erps,
                                  "prewindowed_train")

        # Let pyspace live train on this data
        online_logger.info("Start pyspace live training")
        pw_trainer.start_training("prewindowed_train") # pass an additional True for profiling


    def train(self):
        """ Trains the pyspace flows on the data streamed from
            an external EEG-Server
        """

        # Create trainer and initialize the eeg data stream
        online_trainer = trainer.LiveTrainer()
        stream_manager = \
            eeg_stream_manager.LiveEegStreamManager(online_logger)

        stream_manager.initialize_eeg_server(self.eeg_server_train_ip,
                                             self.eeg_server_eeg_port)

        # Prepare trainer for training
        online_trainer.set_eeg_stream_manager(stream_manager)


        training_files = []

        if isinstance(self.datafile_train, str):
            training_files = self.datafile_train
        else:
            for datafile in self.datafile_train:
                if os.path.isabs(datafile):
                    datafile_train = [datafile]
                else:
                    datafile_train = \
                        [os.path.join(self.configuration.storage, datafile)]

                training_files = training_files + datafile_train


        online_logger.info(training_files)
        online_logger.info("#"*30)
        online_trainer.prepare_training(training_files,
                                      self.erps,
                                      "train")

        # Let pyspace live train on this data
        online_logger.info("Start pyspace live training")
        online_trainer.start_training("train") # pass an additional True for profiling


    def adapt_classification_threshold(self, load_model=True):
        """ Adapts classification threshold on a special function """

        # Create pyspace live processing server
        live_adaptor = adaptation.LiveAdaptor()

        # Reloading stored models
        if load_model:
            online_logger.info("Reloading Models")
            live_adaptor.load_model(self.flow_persistency_directory, self.erps)

        online_logger.info("Creating eeg stream")
        # Start EEG server that streams data for testing
        stream_manager = \
            eeg_stream_manager.LiveEegStreamManager(online_logger)

        stream_manager.initialize_eeg_server(self.eeg_server_train_ip,
                                             self.eeg_server_eeg_port)

        # Prepare live_adaptor for adaptation
        live_adaptor.set_eeg_stream_manager(stream_manager)
        adaptation_files = []

        if isinstance(self.datafile_train, str):
            adaptation_files = self.datafile_train
        else:
            for datafile in self.datafile_train:
                if os.path.isabs(datafile):
                    datafile_train = [datafile]
                else:
                    datafile_train = \
                        [os.path.join(self.configuration.storage, datafile)]

                adaptation_files = adaptation_files + datafile_train


        online_logger.info(adaptation_files)
        online_logger.info("#"*30)

        live_adaptor.prepare_adaptation(adaptation_files,
                                        self.erps)
        # Prepare for adaptation
        # register the pyspace live module with the ControlManager

        # Let pyspace live train on this data
        online_logger.info("Start pyspace live adaptation")
        live_adaptor.start_adaptation()

        # We block and wait until either adaptation is finished or
        # the user enters a 'X'
        time.sleep(5)
        try:
            while live_adaptor.is_adaptation_active():
                time.sleep(1)

        except Exception as exc:
            online_logger.log(logging.ERROR,"Training interrupted")
            exc_type, exc_value, exc_traceback = sys.exc_info()
            online_logger.log(logging.ERROR, repr(
                                traceback.format_exception(
                                    exc_type, exc_value, exc_traceback)))
            online_logger.log(logging.ERROR, str(exc))

        live_adaptor.stop_adaptation()

        online_logger.info("Adaptation Finished")



    def predict(self, load_model = True, online = True, remote = False):
        """ Classifies new instances based on the trained pyspace flows"""

        # do all preparations only if there is no prepared prediction process
        if self.live_processing == None:

            # create pyspace live processing server
            self.live_processing = prediction.Predictor(self.messenger, self.configuration)
            self.live_processing.set_controller(self)
            self.prediction_process = self.live_processing

            # reloading stored models
            if load_model:
                online_logger.info("Reloading Models")
                self.live_processing.load_model(self.flow_persistency_directory, self.erps)

            # connect to the server
            if online:

                # init eeg streaming and recording
                stream_manager = eeg_stream_manager.LiveEegStreamManager(online_logger, self.configuration)
                stream_manager.initialize_eeg_server(self.eeg_server_predict_ip,
                                                     self.eeg_server_eeg_port)

                # setup recording if option is set
                if self.subject is not None and \
                        self.experiment is not None:
                    stream_manager.record_with_options(self.subject, self.experiment, online=True)
                else:
                    online_logger.warn("RAW DATA IS NOT RECORDED!")

                # set teht stream manager into the trainger object
                self.live_processing.set_eeg_stream_manager(stream_manager)

                # prepare the prediction
                self.live_processing.prepare_predicting(self.erps)

            else:
                # when running offline prepare local streaming
                if isinstance(self.datafile_test, str):
                    testing_file = \
                        os.path.join(self.configuration.storage, self.datafile_test)
                elif isinstance(self.data_test, list):
                    testing_file = \
                        os.path.join(self.configuration.storage, self.datafile_test[0])
                else:
                    raise Exception, "could not determine testing data!"

                online_logger.info(str("testing file: %s " % testing_file))
                self.live_processing.prepare_predicting(self.erps, testing_file)

            online_logger.info("Finished")

            if not remote:
                raw_input("\nPress Enter to start predicting ")

        # Let pyspace live classify the test data
        online_logger.info("Start pyspace live classification")
        # pass an additional True for profiling

        self.live_processing.start_predicting()


    def stop_prediction(self):
        # Create pyspace live processing server
        self.live_processing.process_external_command("STOP")


    def get_live_flow (self):

        f = open("%s/abri_flow_P3.yaml" % self.flow_persistency_directory, 'r')
        abri_flow_p3 = f.read()
        f.close()
        f = open("%s/abri_flow_LRP.yaml" % self.flow_persistency_directory, 'r')
        abri_flow_lrp = f.read()
        f.close()

        return abri_flow_p3, abri_flow_lrp

def parse_arguments():
    """ Parses the command line arguments to create options object"""
    usage = "Usage: %prog [--config <configuration.yaml>] "\
            "[--params <params.yaml>] "
    parser = optparse.OptionParser(usage=usage)
    parser.add_option("-c", "--configuration",
                      help="Choose the configuration file",
                      action="store")
    parser.add_option("-p", "--params",
                      help="Specify parameter file that contains information about the data and environment",
                      action="store")
    parser.add_option("-t","--train",
                      help="Train a flow according to parameters in parameter file",
                      action="store_true",
                      dest="train",
                      default=False)
    parser.add_option("--prewindowing",
                      help="Prewindow a flow according to parameters in parameter file",
                      action="store_true",
                      dest="prewindowing",
                      default=False)
    parser.add_option("--prewindowing_offline",
                      help="Prewindow an offline flow for test purpose",
                      action="store_true",
                      dest="prewindowing_offline",
                      default=False)
    parser.add_option("--prewindowed_train",
                      help="Train a prewindowed flow according to parameters in parameter file",
                      action="store_true",
                      dest="prewindowed_train",
                      default=False)
    parser.add_option("-a","--adapt",
                      help="Adapt the threshold of the flow according to parameters in parameter file",
                      action="store_true",
                      dest="adapt",
                      default=False)
    parser.add_option("--predict",
                      help="Predict with trained flow",
                      action="store_true",
                      dest="predict",
                      default=False)
    parser.add_option("--predict_offline",
                      help="Prediction using an offline flow for testing purposes",
                      action="store_true",
                      dest="predict_offline",
                      default=False)
    parser.add_option("--all",
                      help="First train a flow according to parameters in parameter file and then do prediction using the trained flow",
                      action="store_true",
                      dest="all",
                      default=False)
    parser.add_option("--remote",
                      help="Start remote control",
                      action="store_true",
                      dest="remote",
                      default=False)



    (parse_options, parse_args) = parser.parse_args()


    return (parse_options, parse_args)

def read_parameter_file(parameter_file_name):
    """ Reads and interprets the given parameter file """
    # interpret parameter file
    online_logger.info(parameter_file_name)

    param_path = os.path.join(pySPACE.configuration.spec_dir, "live_settings", parameter_file_name)

    stream = file(param_path, 'r')

    online_logger.info( "Loading parameter file..")
    parameters = yaml.load(stream)
    online_logger.info( "Done.")
    online_logger.debug(yaml.dump(parameters))

    return parameters

def create_and_start_rpc_server(controller_instance, rpc_port=16254):
    """  Creates and starts the server for the remote procedure calls """
        # starting rpc server
    rpc_server_ip = "localhost"
    rpc_server_port = rpc_port
    online_logger.info(str("Starting RPC server on port %d .." % rpc_server_port))
    from SimpleXMLRPCServer import SimpleXMLRPCServer
    server = \
        SimpleXMLRPCServer((rpc_server_ip, rpc_server_port), logRequests=False)
    online_logger.info( "RPCServer listens on "+str(rpc_server_ip)+":"+str(rpc_server_port))

    # register and start
    server.register_instance(controller_instance)
    server_process = multiprocessing.Process(target = server.serve_forever)
    server_process.start()

    return server_process

def create_backup(liveControl, options):
    """Create backup files"""
    online_logger.info( "Creating backup...")

    #path to be created
    path = os.path.realpath(__file__)
    dir_path = os.path.dirname(path)

    newdir = dir_path + os.path.sep + "backup"
    if not os.path.exists(newdir):
        os.makedirs (newdir)

    date_time = datetime.datetime.now()

    path_datetime = newdir + os.path.sep + date_time.strftime("%Y%m%d_%H%M%S")
    os.mkdir (path_datetime)
    path_flow = path_datetime + os.path.sep + "flow_storage"
    path_node_chain = path_datetime + os.path.sep + "node_chains"
    path_windower = path_datetime + os.path.sep + "windower"
    path_param = path_datetime + os.path.sep + "live_settings"
    os.mkdir (path_flow)
    os.mkdir (path_node_chain)
    os.mkdir (path_windower)
    os.mkdir (path_param)

    import distutils.dir_util
    distutils.dir_util.copy_tree(
        liveControl.flow_persistency_directory, path_flow)
    if os.path.isdir (path_flow):
        online_logger.info( "flow storage backup successful!")


    param_path = os.path.join(pySPACE.configuration.spec_dir, "live_settings", options.params)
    if param_path == None:
        return
    distutils.file_util.copy_file(param_path, path_param)
    if os.path.isdir (path_param):
        online_logger.info( "parameters file backup successful!")


    online_logger.info("Creating backup finished!")

if __name__ == "__main__":

    (options,args) = parse_arguments()

    server_process = None

    if options.remote:
        online_logger.info("Starting remote modus")
        conf_file_name = options.configuration
        conf = pySPACE.load_configuration(conf_file_name)
        adrf = pySPACE.environments.live.communication.adrf_messenger.AdrfMessenger()
        adrf.register()
        # register the interface with ADRF
        online_logger.info("Starting event loop")
        while True:
            online_logger.info("Check register status")
            time.sleep(0.5)
            while adrf.is_registered():
                #online_logger.info("Get command")
                command = adrf.adrf_receive_command()

                if command[0] == 3: # 3 = C_CONFIGURE
                    online_logger.info( "received command: C_CONFIGURE")
                    online_logger.info( "Loading parameter file..")

                    online_logger.info( "Done")
                    adrf.set_state(5) # 5 = S_CONFIGURED

                    # starting controller
                    cfg = adrf.get_config()
                    online_logger.info( "Constructing Controller...")
                    liveControl = LiveController(cfg,
                                                       conf,
                                                       adrf)

                    online_logger.info( "Constructing Controller finished")
                    if server_process == None:
                        online_logger.info("Starting XMLRPCServer.. ")
                        server_process = create_and_start_rpc_server(liveControl)
                    else :
                        online_logger.info(str("XMLRPCServer already running (%s)" % server_process))

                elif command[0] == 4: # 4 = C_STARTAPP
                    online_logger.info( "received command: C_STARTAPP")
                    adrf.set_state(6) # 6 = S_RUNNING
                    cfg = adrf.get_config()
                    # mode can be defined in the configuration file, predict_offline as an example
                    if cfg["mode"] == 'prewindowing_offline':
                        liveControl.prewindowing(online=False)
                        create_backup(liveControl, options)
                    elif cfg["mode"] == 'prewindowing':
                        # first start eegclient
                        liveControl.prewindowing(online=True)
                        create_backup(liveControl, options)
                    elif cfg["mode"] == 'prewindowed_train':
                        liveControl.prewindowed_train()
                        create_backup(liveControl, options)
                    elif cfg["mode"] == 'train':
                        liveControl.train()
                        create_backup(liveControl, options)
                    elif cfg["mode"] == 'adapt':
                        liveControl.adapt_classification_threshold()
                    elif cfg["mode"] == 'predict':
                        liveControl.predict(online=True, remote=True)
                    elif cfg["mode"] == 'predict_offline':
                        liveControl.predict(online=False, remote=True)

                    elif cfg["mode"] == 'all':
                        liveControl.train()
                        create_backup(liveControl, options)
                        liveControl.predict(online=False, remote=True)
                    else :
                        online_logger.warn(str("mode \'%s\' was not recognized!" % cfg["mode"]))

                elif command[0] == 5: # 5 = C_STOPAPP
                    online_logger.info( "received command: C_STOPAPP")
                    adrf.set_state(8) # 8 = S_STOPPED
                    if cfg["mode"] in ('prewindowing', 'prewindowing_offline'):
                        liveControl.stop_prewindowing()
                    elif cfg["mode"] in ('predict', 'predict_offline'):
                        liveControl.stop_prediction()
                    else:
                        pass

        adrf.undo_registration()


    elif options.all:
        online_logger.info("Starting training and then predicting...")

        param_file_name = options.params
        parameters = read_parameter_file(param_file_name)

        conf_file_name = options.configuration
        if conf_file_name is not None:
            conf = pySPACE.load_configuration(conf_file_name)
        else:
            conf = None
        # starting controller
        online_logger.info( "Constructing Controller...")


        liveControl = LiveController(parameters,
                                           conf)

        online_logger.info( "Constructing Controller finished")


        server_process = create_and_start_rpc_server(liveControl)

        liveControl.prewindowing()
        liveControl.prewindowed_train()
        create_backup(liveControl, options)
        server_process.terminate()
        server_process.join()
        liveControl.predict(online=False)
        server_process.terminate()
        server_process.join()



    else:
        pySPACE.load_configuration(options.configuration)
        conf = pySPACE.configuration

        param_file_name = options.params
        parameters = read_parameter_file(param_file_name)



        from pySPACE.environments import big_bang
        from pySPACE.environments.live import eeg_stream_manager, prediction, adaptation, communication, trainer
        import pySPACE.environments.live.communication.log_messenger

        # starting controller
        online_logger.info( "Constructing Controller...")


        liveControl = LiveController(parameters, conf)

        online_logger.info( "Constructing Controller finished")


        server_process = create_and_start_rpc_server(liveControl)


        # start main work....
        if options.prewindowing:
            # first start eegclient
            liveControl.prewindowing(online=True)
            create_backup(liveControl, options)
            server_process.terminate()
            server_process.join()
        elif options.prewindowing_offline:
            liveControl.prewindowing(online=False)
            create_backup(liveControl, options)
            server_process.terminate()
            server_process.join()
        elif options.prewindowed_train:
            liveControl.prewindowed_train()
            create_backup(liveControl, options)
            server_process.terminate()
            server_process.join()
        elif options.train:
            liveControl.train()
            create_backup(liveControl, options)
            server_process.terminate()
            server_process.join()
        elif options.adapt:
            liveControl.adapt_classification_threshold()
            create_backup(liveControl, options)
            server_process.terminate()
            server_process.join()
        elif options.predict:
            liveControl.predict(online=True)
            server_process.terminate()
            server_process.join()
        elif options.predict_offline:
            liveControl.predict(online=False)
            server_process.terminate()
            server_process.join()
