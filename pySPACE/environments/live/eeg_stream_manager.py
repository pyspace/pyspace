""" Script for managing of eeg data streams

Here the :class:`~pySPACE.missions.support.windower.MarkerWindower` is used.
"""

import os
import sys
import time
import xmlrpclib
import subprocess
import random
import warnings

file_path = os.path.dirname(os.path.abspath(__file__))
pyspace_path = file_path[:file_path.rfind('pySPACE')-1]
if not pyspace_path in sys.path:
    sys.path.append(pyspace_path)

import pySPACE

from pySPACE.missions.support.windower import MarkerWindower, WindowFactory
from pySPACE.tools.live import eeg_stream
from pySPACE.environments.live.recorder import Recorder

class LiveEegStreamManager(object):
    """ This module controls the eegmanager related configuration
    and provides a meaningful interface of the eegmanager for pyspace live.
    """
    def __init__(self, logger):
        self.logger = logger
        self.configuration = pySPACE.configuration
        self.eeg_client = None

        # tools for eegmanager subprocess
        self.server_process = None
        self.remote = None
        self.ip = None
        self.port = None

        # tools for python based recorder
        self.recorder = None

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
        if self.remote is not None:
            self.remote.shut_down()

    def stream_local_file(self, filename):
        # start streaming of file filename using
        # the local eegmanager process

        self.ip = ip = "127.0.0.1"

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
                self.logger.error(str("failed to add fileacquisition for file %s" % filename))
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

    def initialize_eeg_server(self, ip=None, port=None, usb=None):
        self.ip = ip
        self.port = port
        self.usb = usb

    def record_with_options(self, subject, experiment, online=False):
        # create new Recorder instance to handle raw data storage

        self.recorder = Recorder(client=self.eeg_client, folder=None,
                                 subject=subject, task=experiment, online=online)
        self.logger.info("started raw-data-recording")

    def request_window_stream(self, window_spec=None, nullmarker_stride_ms=1000, no_overlap=True):

        # load windower spec file
        if window_spec is None:
            window_definitions = WindowFactory.default_windower_spec()
            no_overlap = True
            self.logger.info("Using default windower spec %s" % window_definitions)
        else:
            windower_spec_file = open(window_spec, 'r')
            window_definitions = \
                WindowFactory.window_definitions_from_yaml(windower_spec_file)
            windower_spec_file.close()
            self.logger.info(str("Finished loading windower spec file from %s" % window_spec))

        if nullmarker_stride_ms != window_definitions[0].endoffsetms:
            warnings.warn("defined nullmarker stride (%d) is different from "
                          "endoffset (%d) in window-definitions[0]!" %
                          (nullmarker_stride_ms, window_definitions[0].endoffsetms))

        eeg_client = self.setup_client()

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


    def setup_client(self):

        # connect and start client
        if self.ip is not None and self.port is not None:
            eeg_client = eeg_stream.EEGClient(host=self.ip, port=self.port)
        elif self.usb is not None:
            self.logger.info("Using USB Client")
            eeg_client = eeg_stream.EEGClientUsb()
        eeg_client.connect()

        if self.eeg_client is None:
            self.eeg_client = eeg_client

        if self.recorder is not None:
            if not self.recorder.has_client():
                self.recorder.set_eeg_client(eeg_client)

        self.logger.info("Started EEG-Client")

        return eeg_client

    def stop(self):
        self.eeg_client.disconnect()
        del self.eeg_client

        if self.remote is not None:
            self.remote.stop()
            self.remote.shut_down()
