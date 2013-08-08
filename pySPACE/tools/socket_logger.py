""" Code based on `network-logging <http://www.python.org/doc/2.5.2/lib/network-logging.html>`_

This code is mainly a code copy.
For license issues of python code, we refer to:
http://docs.python.org/3/license.html
"""

import cPickle
import logging
import logging.handlers
import SocketServer
import socket
import struct
import select
import threading
import warnings
    
class LogRecordStreamHandler(SocketServer.StreamRequestHandler):
    """Handler for a streaming logging request.

    This basically logs the record using whatever logging policy is
    configured locally.
    """

    def handle(self):
        """
        Handle multiple requests - each expected to be a 4-byte length,
        followed by the LogRecord in pickle format. Logs the record
        according to whatever policy is configured locally.
        """
        while True:
            chunk = self.connection.recv(4)
            if len(chunk) < 4:
                return
            slen = struct.unpack(">L", chunk)[0]
            chunk = self.connection.recv(slen)
            while len(chunk) < slen:
                chunk = chunk + self.connection.recv(slen - len(chunk))
            obj = self.unPickle(chunk)
            record = logging.makeLogRecord(obj)
            self.handleLogRecord(record)

    def unPickle(self, data):
        return cPickle.loads(data)

    def handleLogRecord(self, record):
        # if a name is specified, we use the named logger rather than the one
        # implied by the record.
        if self.server.logname is not None:
            name = self.server.logname
        else:
            name = record.name
        logger = logging.getLogger('')
        logger.handle(record)

class LogRecordSocketReceiver(threading.Thread, SocketServer.ThreadingTCPServer):
    """ Simple TCP socket-based logging receiver suitable for testing

    ..note:: If this class crashes your software, there are problems with your
            network configuration. Change your network connection or restart
            computer.
    """

    allow_reuse_address = True
    try:
        host, aliaslist, lan_ip = socket.gethostbyname_ex(socket.gethostname())
    except socket.gaierror,e:
        warnings.warn(str(e)+" Your network configurations seems to be broken."+
            " You should restart the connection or your PC."+
            " The connection is used for logging communication.")

    def __init__(self, host=lan_ip[0],
                 port=logging.handlers.DEFAULT_TCP_LOGGING_PORT,
                 handler=LogRecordStreamHandler):
        threading.Thread.__init__(self)
        SocketServer.ThreadingTCPServer.__init__(self, (host, port), handler)
        self.abort = False
        self.timeout = True
        self.logname = None

    def run(self):
        abort = 0
        while not abort:
            rd, wr, ex = select.select([self.socket.fileno()],
                                       [], [],
                                       self.timeout)
            if rd:
                self.handle_request()
            abort = self.abort
            
    def shutdown(self):
        SocketServer.ThreadingTCPServer.server_close(self)
        
