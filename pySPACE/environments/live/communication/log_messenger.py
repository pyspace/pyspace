import logging
import multiprocessing
import Queue
import socket
import select
import struct

from pySPACE.environments.live.communication import messenger
from pySPACE.tools.logging_stream_colorer import ColoredLevelFormatter

class LogMessenger(messenger.Messenger):
    
    def __init__(self):
        self.logger = logging.getLogger('LogMessengerLogger')
        self.logger.setLevel(logging.DEBUG)

        self.loggingStreamHandler = logging.StreamHandler()
        self.loggingStreamHandler.setLevel(logging.INFO)

        self.formatter = ColoredLevelFormatter("%(asctime)s - %(name)s - %(levelname)s - %(pathname)s:%(lineno)d - %(message)s")
        self.loggingStreamHandler.setFormatter(self.formatter)

        self.logger.addHandler(self.loggingStreamHandler)
        
    def register(self):
        pass

    def end_transmission(self):
        pass

    def send_message(self,message):
        pass
        #print "++++++++++++++++++++++"
        self.logger.log(logging.DEBUG, str(message))
        #print "_______________________"
               