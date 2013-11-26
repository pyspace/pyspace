""" Script for managing of eeg data streams

Here the :class:`~pySPACE.missions.support.windower.MarkerWindower` is used.
"""

import os
import sys
import time
import xmlrpclib
import subprocess
import random

file_path = os.path.dirname(os.path.abspath(__file__))
pyspace_path = file_path[:file_path.rfind('pySPACE')-1]
if not pyspace_path in sys.path:
    sys.path.append(pyspace_path)

import pySPACE

from pySPACE.missions.support.windower import MarkerWindower, WindowFactory
from pySPACE.tools.live import eeg_stream

class LiveEegStreamManager(object):
    """ This module controls the eegmanager related configuration
    and provides a meaningful interface of the eegmanager for pyspace live.
    """
    def __init__(self, logger, configuration=dict()):
        self.logger = logger
        self.configuration = configuration
        self.server_process = None
        self.eeg_client = list()
        self.remote = None

        self.ip = None
        self.port = None

        # create a new eeg server process
        # it is needed either for streaming of local data
        # or recording of incoming data
        self.executable_path = os.path.join(pyspace_path, "pySPACE", "tools", "live", "eegmanager", "eegmanager", "eegmanager")
        if not os.path.isfile(self.executable_path):
            self.logger.error("cannot find eegmanager executable!")
            self.logger.error("it should be in %s", self.executable_path)
            exit(0)

        self.logger.info(str("starting process with executable %s" % self.executable_path))

        xml_range = range(16253, 17253)
        random.shuffle(xml_range)
        for xmlport in xml_range:
            try:
                self.logger.info("Creating eegserver process with server \'eegmanager %d\'" % xmlport)
                self.server_process = subprocess.Popen([self.executable_path, str(xmlport)])
            except OSError as err:
                self.logger.error(str("launching subprocess with excutable at %s failed!" % self.executable_path))
                raise err

            time.sleep(.3)
            self.server_process.poll()
            if self.server_process.returncode == None:
                # exit the loop of the process is still running..
                break

        self.remote = xmlrpclib.ServerProxy("http://%s:%s" % ("127.0.0.1", xmlport))


    def __del__(self):
        # self.remote.shut_down()
        pass

    def stream_local_file(self, filename):
        # start streaming of file filename using
        # the local eegmanager process

        if self.remote.get_state()[0] is not "IDLE":
            self.remote.stop()

        tcp_range = range(41244, 61244)
        random.shuffle(tcp_range)
        for port in tcp_range:

            self.remote.stop() # clear possible dangling setup

            # file acquisition
            ret = self.remote.add_module("FILEAcquisition", str("--blocksize 100 --filename %s" % filename))
            if ret < 0:
                self.remote.stop()
                self.logger.error(str("failed to adding fileacquisition for file %s" % filename))
                self.logger.error(str(self.remote.stdout()))
                self.remote.shut_down()
                raise Exception, "Check your paths!"

            # to the network!
            ret = self.remote.add_module("NETOutput", str("--port %d --blocking" % port))
            if ret < 0:
                self.logger.warn("failed to add netoutput with port %d" % port)
                self.logger.error(str(self.remote.stdout()))
                continue

            self.port = port
            break

        self.remote.start()
        self.logger.debug(str("started local streaming of file %s" % filename))

    def receive_remote_stream(self, ip, port):
        # set the ip/port which is used during the
        # request window stream function
        self.port = port
        self.ip = ip


    def record_with_options(self, subject, experiment, online=False):
        # use local eegmanager process for
        # raw-data-recording of incoming data

        if self.port is None or self.ip is None:
            raise Exception, "Dont know how to record without remote info!"

        if self.remote.get_state()[0] is not "IDLE":
            self.remote.stop()

        self.remote.stop() # clear possible dangling setup

        directory = self.configuration.storage

        ret = self.remote.add_module("NETAcquisition", str("--host %s --port %d" % (self.ip, self.port)))
        if ret < 0:
            self.remote.stop()
            self.logger.error(str("failed to add netacquisition with host(%s) and ip(%d)" % \
                        (self.ip, self.port)))
            self.logger.error(str(self.remote.stdout()))
            self.remote.shut_down()
            raise Exception, "Connection failed!"

        if online:
            ret = self.remote.add_module("FILEOutput", str("--subject %s --trial %s --dir %s --online" % \
                    (subject, experiment, directory)))
        else:
            ret = self.remote.add_module("FILEOutput", str("--subject %s --trial %s --dir %s" % \
                    (subject, experiment, directory)))
        if ret < 0:
            self.remote.stop()
            self.logger.error(str("failed to add fileoutput with directory %s" % directory))
            self.logger.error(str(self.remote.stdout()))
            raise Exception, "Check if this directory exists!"

        self.remote.start()
        self.logger.info("started raw-data-recording")

    def request_window_stream(self, window_spec, nullmarker_stride_ms = 1000, no_overlap = False):
        # function to connect a client to a running
        # remote streaming server or local process
        if self.ip is None:
            self.ip = "127.0.0.1" # for the local mode

        if self.port is None:
            # without a port we cannot do anything
            raise Exception, "Port for stream reception is not set!"

        # connect and start client
        eeg_client = eeg_stream.EEGClient(host=self.ip,
                                               port=self.port)
        eeg_client.connect()
        self.logger.info( "Started EEG-Client")
        self.eeg_client.append(eeg_client)

        # load windower spec file
        windower_spec_file = open(window_spec, 'r')
        window_definitions = \
            WindowFactory.window_definitions_from_yaml(windower_spec_file)
        windower_spec_file.close()
        self.logger.info( "Finished loading windower spec file")
        self.logger.info(window_definitions)

        # create windower
        self.marker_windower = MarkerWindower(eeg_client,
                                              window_definitions,
                                              nullmarker_stride_ms=nullmarker_stride_ms,
                                              no_overlap = no_overlap)
        self.logger.info( "Created windower instance")

        # return an iterator over the yielded windows
        window_stream = ((sample, label) for (sample, label) in self.marker_windower)
        self.logger.info( "Created window-stream")

        return window_stream

    def initialize_eeg_server(self, eeg_server_ip,
                              eeg_server_port):
        self.ip = eeg_server_ip
        self.port = eeg_server_port
        self.eeg_client = list()



    def stop(self):
        self.remote.stop()

