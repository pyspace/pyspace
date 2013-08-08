import os
import sys
import subprocess
import random
import time
import xmlrpclib
import yaml
import logging
import getopt
import signal
import cPickle

file_path = os.path.dirname(os.path.abspath(__file__))
pyspace_path = file_path[:file_path.rfind('pySPACE')-1]
if not pyspace_path in sys.path:
    sys.path.append(pyspace_path)

import pySPACE

from pySPACE.environments import big_bang
from pySPACE.environments.big_bang import Configuration
# from pySPACE.missions.nodes.source.eeg_source import *
# from eeg_stream_manager import AbriOnlineEegStreamManager

class VIBotTestSuite(object):

    # receive configuration and
    # parameters for running abri-online
    def __init__(self, logger=None, conf_file="", param_file=""):

        """

        :param logger:
            the logger object which is used for logging

        :param conf_file:
            the pySPACE conf file

        :param param_file:
            the live-parameter file

        """
        self.logger = logger

        # save raw file names for later
        self.conf_file = conf_file
        self.param_file = param_file

        # load configuration
        pySPACE.load_configuration(conf_file)
        self.conf = pySPACE.configuration


        self.executable_path = os.path.join(pyspace_path, "pySPACE", "tools", "live", "eegmanager", "eegmanager", "eegmanager")
        if not os.path.isfile(self.executable_path):
            self.logger.error("cannot find eegmanager executable!")
            self.logger.error("it should be in %s", self.executable_path)
            exit(0)

        # reference to abri-online subprocess
        self.abri_online = None

        # create eeg-streaming subprocess
        self.eegmanager = None
        self.eegmanager_remote = None
        self.create_eegmanager()

        # load parameter file
        self.params = yaml.load(open(os.path.join(self.conf.spec_dir, "live_settings", param_file)))


    # before we leave stop the subprocess!
    def cleanup(self):
        self.eegmanager_remote.shut_down()
        self.logger.debug("eegmanager process was shut down")

    # creates an eegmanager subprocess for data streaming
    def create_eegmanager(self):


        self.logger.info(str("starting process with executable %s" % self.executable_path))

        xml_range = range(16253, 17253)
        random.shuffle(xml_range)
        for xmlport in xml_range:
            try:
                self.logger.info("Creating eegserver process with server \'eegmanager %d\'" % xmlport)
                self.eegmanager = subprocess.Popen([self.executable_path, str(xmlport)])
            except OSError as err:
                self.logger.error(str("launching subprocess with excutable at %s failed!" % self.executable_path))
                raise err

            time.sleep(.3)
            self.eegmanager.poll()
            if self.eegmanager.returncode == None:
                # exit the loop of the process is still running..
                break

        self.eegmanager_remote = xmlrpclib.ServerProxy("http://%s:%s" % ("127.0.0.1", xmlport))

        self.logger.debug("created eegmanager process")

    # start abri online subprocess in prewindowing mode
    def do_prewindowing(self, training_data):

        work_dir = os.path.join(pySPACE.configuration.root, "run")

        # create param file for testing
        # set prewindow ip to localhost
        self.params["eeg_server"]["eeg_server_prewindow_ip"] = "127.0.0.1"
        yaml.dump(self.params, open(os.path.join(self.conf.spec_dir, "live_settings", str("%s_for_testing.yaml" % (self.param_file.split(".")[0]))), "wt"))
        self.params = yaml.load(open(os.path.join(self.conf.spec_dir, "live_settings", self.param_file)))

        for filename in training_data:

            self.logger.info(str("running prewindowing with %s" % filename))

            full_file_name = os.path.join(pySPACE.configuration.storage, filename)
            # add signal source
            self.eegmanager_remote.add_module("FILEAcquisition", str("--filename %s --blocksize 100" % full_file_name))
            # add network interface to abri
            self.eegmanager_remote.add_module("NETOutput", "--port 51244 --blocking")
            self.eegmanager_remote.start()

            p = ["python", "launch_live.py", "--conf", self.conf_file, \
                    "--param", str("%s_for_testing.yaml" % (self.param_file.split(".")[0])), "--prewindowing"]
            self.abri_online = subprocess.Popen(p, cwd=work_dir)

            time.sleep(1)
            self.abri_online.poll()
            if self.abri_online.returncode is not None:
                self.logger.error("abri-online subprocess exited!")
                exit(0)

            while(self.abri_online.returncode is None):
                self.abri_online.poll()
                self.logger.debug(".. test running ..")
                time.sleep(1)

            self.logger.info("abri-streaming finished!")
            self.eegmanager_remote.stop()
            self.eegmanager_remote.stop()

        # remove the altered param file
        os.remove(os.path.join(self.conf.spec_dir, "live_settings", str("%s_for_testing.yaml" % (self.param_file.split(".")[0]))))

        # self.logger.info("            ######################")
        # self.logger.info("            prewindowing finished!")
        # self.logger.info("            ######################")

    # start abri online for training
    def do_prewindowed_train(self):

        work_dir = os.path.join(pySPACE.configuration.root, "run")

        p = ["python", "launch_live.py", "--conf", self.conf_file, \
                "--param", self.param_file, "--prewindowed_train"]
        self.abri_online = subprocess.Popen(p, cwd=work_dir)

        time.sleep(1)
        self.abri_online.poll()
        if self.abri_online.returncode is not None:
            self.logger.error("abri-online subprocess exited!")
            exit(0)

        while(self.abri_online.returncode is None):
            self.abri_online.poll()
            self.logger.debug(".. test running ..")
            time.sleep(1)


        # self.logger.info("            ##############################")
        # self.logger.info("            prewindowed training finished!")
        # self.logger.info("            ##############################")



    # start abri online for prediction
    def do_prediction(self, test_data):

        work_dir = os.path.join(pySPACE.configuration.root, "run")

        # create param file for testing
        # set prediction ip to localhost
        self.params["eeg_server"]["eeg_server_predict_ip"] = "127.0.0.1"
        yaml.dump(self.params, open(os.path.join(self.conf.spec_dir, "live_settings", str("%s_for_testing.yaml" % (self.param_file.split(".")[0]))), "wt"))
        self.params = yaml.load(open(os.path.join(self.conf.spec_dir, "live_settings", self.param_file)))

        for filename in test_data:

            self.logger.info(str("running prediction with %s" % filename))

            full_file_name = os.path.join(pySPACE.configuration.storage, filename)
            # add signal source
            self.eegmanager_remote.add_module("FILEAcquisition", str("--filename %s --blocksize 100" % full_file_name))
            # add network interface to abri
            self.eegmanager_remote.add_module("NETOutput", "--port 51244 --blocking")
            self.eegmanager_remote.start()

            p = ["python", "launch_live.py", "--conf", self.conf_file, \
                    "--param", str("%s_for_testing.yaml" % (self.param_file.split(".")[0])), "--predict"]
            self.abri_online = subprocess.Popen(p, stdin=subprocess.PIPE, cwd=work_dir)

            time.sleep(1)
            self.abri_online.poll()
            if self.abri_online.returncode is not None:
                self.logger.error("abri-online subprocess exited!")
                exit(0)

            # let it run for seven seconds (startup etc.)
            time.sleep(7)

            self.abri_online.poll()
            if self.abri_online.returncode is not None:
                break

            # send the enter-keystroke for the actual prediction to start
            self.logger.info("pressing enter..")
            self.abri_online.stdin.write("\n")

            while(self.abri_online.returncode is None):
                self.abri_online.poll()
                self.logger.debug(".. test running ..")
                time.sleep(1)

            self.logger.info("abri-prediction finished!")
            self.eegmanager_remote.stop()
            self.eegmanager_remote.stop()

            # check on the results
            self.check_results(filename)

        # remove the altered param file
        os.remove(os.path.join(self.conf.spec_dir, "live_settings", str("%s_for_testing.yaml" % (self.param_file.split(".")[0]))))

        # self.logger.info("            #########################")
        # self.logger.info("            prediction test finished!")
        # self.logger.info("            #########################")

    # compare reference results to actual results
    # or store current results as reference
    def check_results(self, filename):

        work_dir = os.path.join(pySPACE.configuration.root, "run")

        for key in self.params["potentials"]:
            name = key["flow_id"]
            result = cPickle.load(open(os.path.join(work_dir,str("%s.result" % name)), "r"))
            reference_file = os.path.join(pySPACE.configuration.storage, str("%s.%s.result"%(filename, name)))
            if not os.path.isfile(reference_file):
                self.logger.info(str("no reference results found for %s" % filename))
                self.logger.info(str("storing current result as reference in file %s" % reference_file))
                cPickle.dump(result, open(reference_file, "w+"))
                continue

            reference = cPickle.load(open(reference_file, "r"))
            self.logger.info("-------")
            self.logger.info(str("%s-%s-Result:    %s" % (name, filename, result)))
            self.logger.info(str("%s-%s-Reference: %s" % (name, filename, reference)))
            self.logger.info("-------")

    def stop_streaming(self):
        self.eegmanager_remote.stop()
        self.logger.info("streaming was stopped")
        self.logger.info(self.eegmanager_remote.get_state())

    # process for creating instances from wdefs
    def stimulate(self):
        pass

    # process for receiving abri onlines classification results
    def validate(self):
        pass



#### MAIN ####
if __name__ == "__main__":

    # create logger
    logger = logging.getLogger("test-vi-bot")
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(pathname)s:%(lineno)d - %(message)s")
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.propagate = False

    # write results also to a file
    logfile = logging.handlers.TimedRotatingFileHandler("test.log",backupCount=5)
    logfile.setFormatter(formatter)
    logfile.setLevel(logging.INFO)
    logger.addHandler(logfile)

    conf_file = None
    param_file = None

    try:
        opts, args = getopt.getopt(sys.argv[1:], "p:c:", ["param=", "conf="])
    except getopt.GetoptError as err:
        # print help information and exit:
        print str(err) # will print something like "option -a not recognized"
        sys.exit(2)

    for o, a in opts:
        if o in ("-p", "--param"):
            param_file = a
        elif o in ("-c", "--conf"):
            conf_file = a
        else:
            assert False, "unhandled option"

    if conf_file is None or param_file is None:
        logger.error("please provide parameter and configuration file!")
        exit(0)

    pySPACE.load_configuration(conf_file)

    # create test-suite object
    suite = VIBotTestSuite(logger=logger, conf_file=conf_file, param_file=param_file)

    training_data = ["eeg_examples/test_data_live/Set1/example1.eeg"]

    test_data = ["eeg_examples/test_data_live/Set2/example2.eeg"]


    logger.info("PREWINDOWING -->")
    # start abri-online in prewindowing mode
    suite.do_prewindowing(training_data)
    logger.info("PREWINDOWING        --- DONE")

    logger.info("PREWINDOWED-TRAIN -->")
    # train classifier on prewindowed data
    suite.do_prewindowed_train()
    logger.info("PREWINDOWED-TRAIN   --- DONE")

    logger.info("PREDICTION -->")
    # start abri-online in predict mode
    suite.do_prediction(test_data)
    logger.info("PREDICTION          --- DONE")

    suite.cleanup()