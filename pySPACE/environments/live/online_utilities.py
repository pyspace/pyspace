''' Utility functions for pyspace live, like special logging functionality
'''
import logging.handlers
import os
import errno
import sys
import socket

file_path = os.path.dirname(os.path.abspath(__file__))
pyspace_path = file_path[:file_path.rfind('pySPACE')-1]
if not pyspace_path in sys.path:
    sys.path.append(pyspace_path)

# create directory for logging
try:
    os.mkdir("./log")
except OSError as e:
    if e.errno == errno.EEXIST:
        pass
    else: raise

# import other modules for online prediction
from pySPACE.tools.logging_stream_colorer import ColoredLevelFormatter


online_logger = logging.getLogger("OnlineLogger")
online_logger.setLevel(logging.INFO)

loggingFileHandler = \
    logging.handlers.TimedRotatingFileHandler("log" + \
    os.path.sep + \
    "controller.log",backupCount=5)

loggingStreamHandler = logging.StreamHandler()
loggingStreamHandler.setLevel(logging.INFO)

format_string = "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
stream_formatter = ColoredLevelFormatter(format_string)
file_formatter = logging.Formatter(format_string)
loggingFileHandler.setFormatter(file_formatter)
loggingStreamHandler.setFormatter(stream_formatter)

online_logger.addHandler(loggingFileHandler)
online_logger.addHandler(loggingStreamHandler)
online_logger.propagate = False


# redirect the standard abri logger to the online logger
from pySPACE.tools.logging_stream_redirection import RedirectionHandler
redirection_handler = RedirectionHandler(online_logger)

# Prepare remote logging
root_logger = logging.getLogger("%s-%s" % (socket.gethostname(),
                                           os.getpid()))

root_logger.addHandler(redirection_handler)

root_logger.propagate = False

# TODO: hide tcpserver somewhere
tcpserver = None

def start_logging_server():
    """ Start the socket server """
    from pySPACE.tools.socket_logger import LogRecordSocketReceiver
    # Determining host ip
    host, aliaslist, lan_ip = socket.gethostbyname_ex(socket.gethostname())
    host = lan_ip[0]
    # Search for an available port
    port = logging.handlers.DEFAULT_TCP_LOGGING_PORT
    while True:
        try:
            tcpserver = LogRecordSocketReceiver(host=host,
                                                port=port)
            break
        except socket.error:
            port += 1

    tcpserver.start()
    online_logger.log(logging.info,"Started the TCP logging server on port %s!" % port)

def stop_logging_server(self):
    """ Stop the logging of this operation """
    online_logger.log(logging.info, "Stopping the TCP logging server...")
    tcpserver.abort = True
    tcpserver.join()
    tcpserver.shutdown()
    online_logger.log(logging.info, "Stopping the TCP logging server... Done!")

    online_logger.log(logging.info, "Logging stopped")

