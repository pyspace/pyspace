""" Backend Base Class and Methods """

import socket
import logging
import logging.handlers
import os
import sys

file_path = os.path.dirname(os.path.abspath(__file__))
pyspace_path = file_path[:file_path.rfind('pySPACE')-1]
if not pyspace_path in sys.path:
    sys.path.append(pyspace_path)

import pySPACE

class Backend(object):
    """ Interface for backends
    
    All other backends must implement several methods of 
    this interface in order to execute an operation
    on a specific modality. 
    """
    STATES = set(["idling", "staged", "executing", "retrieved", "consolidated"])
    
    def __init__(self):
        # Start logging
        self._start_logging()
        
        # Install signal handlers
        #signal.signal(signal.SIGINT, self._grace)
        #signal.signal(signal.SIGTERM, self._grace)
        #signal.signal(signal.SIGQUIT, self._grace)
        #signal.signal(signal.SIGPIPE, self._grace)
        
        self.file_handler = None
        self.current_operation = None 
        
        self.SERVER_IP = socket.gethostbyname(socket.gethostname())
        self.SERVER_PORT = None
        self.listener = None
        
    def __del__(self):
        # Stop logging
        self._stop_logging()
                  
    def __str__(self):
        return str(type(self).__name__)

    def stage_in(self, operation):
        """
        Stage the current operation
        """
        self.current_operation = operation
        
        # Add a handler that logs the operations output to a file
        log_path = self.current_operation.result_directory
        self.file_handler = logging.FileHandler(os.path.join(log_path,
                                                               "operation.log"))
        # Determining level of the logger based on pySPACE configuration file
        try:
            log_level = eval(pySPACE.configuration.file_log_level) \
                            if hasattr(pySPACE.configuration, "file_log_level") \
                            else logging.INFO
            if not isinstance(log_level, int):
                raise NameError()                
        except (AttributeError, NameError):
            import warnings
            warnings.warn(
                    "%s is not a valid log level! Falling back to logging.INFO."
                    % pySPACE.configuration.log_level)
            log_level = logging.INFO
            
        self.file_handler.setLevel(log_level)
        # set a format which is simpler for console use
        formatter = logging.Formatter(
                          '%(asctime)s %(name)-40s %(levelname)-8s %(message)s')
        # tell the handler to use the formatter
        self.file_handler.setFormatter(formatter)
        # add the handler to the root logger
        logging.getLogger('').addHandler(self.file_handler)
        
        # prepare socket connection and listener thread    
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM) # TCP
        self.sock.bind((self.SERVER_IP,0))
        self.sock.setblocking(0)
        self.SERVER_PORT = self.sock.getsockname()[1]
    
    def execute(self):
        """
        Executes all processes specified in the currently staged
        operation.
        """
        raise NotImplementedError(
                        "Method execute has not been implemented in subclass %s" 
                         % self.__class__.__name__)
    
    def check_status(self):
        """
        Returns a description of the current state of the operations
        execution.
        """
        raise NotImplementedError(
                   "Method check_status has not been implemented in subclass %s" 
                    % self.__class__.__name__)
    
    def retrieve(self):
        """
        Fetches the results of the operation's processes.
        
        ... note:: This call might block until all processes are finished
        """
        raise NotImplementedError(
                       "Method retrieve has not been implemented in subclass %s" 
                       % self.__class__.__name__)
    
    def consolidate(self):
        """
        Consolidates the results of the single processes into a consistent result of the whole
        operation
        """
        raise NotImplementedError(
                    "Method consolidate has not been implemented in subclass %s"
                    % self.__class__.__name__)
    
    def cleanup(self):
        """
        Remove the current operation and all potential results that
        have been stored in this object
        """
        raise NotImplementedError(
                        "Method cleanup has not been implemented in subclass %s" 
                        % self.__class__.__name__)
        
    def get_result_directory(self):
        """ Return the result directory of the current operation (if any) """
        if self.current_operation == None:
            raise Exception("No operation staged!")
        return self.current_operation.result_directory
        
    def _start_logging(self):
        """ Configures and starts the logging of this operation """
        # Remove the default handler
        if len(logging.getLogger('').handlers) > 0:
            logging.getLogger('').removeHandler(logging.getLogger('').handlers[0])
        # define a handler which writes WARNING messages 
        # or higher to the sys.stderr
        console = logging.StreamHandler()
        
        # Determining level of the logger based on pySPACE configuration file
        try:
            log_level = eval(pySPACE.configuration.console_log_level) \
                            if hasattr(pySPACE.configuration, "console_log_level") \
                            else logging.WARNING
            if not isinstance(log_level, int):
                raise NameError()
        except (AttributeError, NameError):
            import warnings
            warnings.warn("%s is not a valid log level! Falling back to " \
                          "logging.WARNING." % pySPACE.configuration.log_level)
            log_level = logging.WARNING
            
        console.setLevel(log_level)
        # set a format which is simpler for console use
        formatter = logging.Formatter(
                          '%(asctime)s %(name)-40s %(levelname)-8s %(message)s')
        # tell the handler to use this format
        console.setFormatter(formatter)
        # add the handler to the root logger
        logging.getLogger('').addHandler(console)
        logging.getLogger('').setLevel(logging.DEBUG)
        
        self._log("Logging started")
        
        # Starting the socket server
        from pySPACE.tools.socket_logger import LogRecordSocketReceiver
        self._log("Starting the TCP logging server...")
        # Determining host ip
        host, aliaslist, lan_ip = socket.gethostbyname_ex(socket.gethostname())
        self.host = lan_ip[0]
        # Search for an available port
        self.port = logging.handlers.DEFAULT_TCP_LOGGING_PORT
        while True:
            try:
                self.tcpserver = LogRecordSocketReceiver(host=self.host,
                                                         port=self.port)
                break
            except socket.error:
                self.port += 1
                
        self.tcpserver.start()
        self._log("Started the TCP logging server on port %s!" % self.port)
        
    def _stop_logging(self):
        """ Stops the logging of this operation """
        self._log("Stopping the TCP logging server...")
        self.tcpserver.abort = True
        self.tcpserver.join()
        self.tcpserver.shutdown()
        self._log("Stopping the TCP logging server... Done!")
        
        self._log("Logging stopped")

    def _log(self, message, level = logging.INFO):
        """ Logs  the given message  with the given logging level """
        root_logger = logging.getLogger("%s-%s.%s" % (socket.gethostname(),
                                        os.getpid(),
                                        self))
        if len(root_logger.handlers)==0:
            root_logger.addHandler(logging.handlers.SocketHandler('localhost',
                    logging.handlers.DEFAULT_TCP_LOGGING_PORT))

        root_logger.log(level, message)

    #def _grace(self, signal_number, stack_frame):
        #self._log("Signal %s thrown! Gracing backend..." % signal_number, logging.WARNING)
        #self.cleanup()
        #self._stop_logging()
        #import sys
        #sys.exit(1)


def create_backend(backend_type = "serial"):
    """ Creates the :mod:`backend object<pySPACE.environments.backends>` based on the given options

    The following backends are available:

        :``serial``: :class:`~pySPACE.environments.backends.serial.SerialBackend`
        :``mcore``:  :class:`~pySPACE.environments.backends.multicore.MulticoreBackend`
        :``mpi``:    :class:`~pySPACE.environments.backends.mpi_backend.MpiBackend`
        :``loadl``:  :class:`~pySPACE.environments.backends.ll_backend.LoadLevelerBackend`
    """
    if backend_type == "serial":
        from pySPACE.environments.backends.serial import SerialBackend
        backend = SerialBackend()
    elif backend_type == "mcore":
        from pySPACE.environments.backends.multicore import MulticoreBackend
        if hasattr(pySPACE.configuration, "pool_size"):
            backend = MulticoreBackend(pool_size=pySPACE.configuration.pool_size)
        else:
            backend = MulticoreBackend()
    elif backend_type == "mpi":
        from pySPACE.environments.backends.mpi_backend import MpiBackend
        backend = MpiBackend()
    elif backend_type == "loadl":
        from pySPACE.environments.backends.ll_backend import LoadLevelerBackend
        backend = LoadLevelerBackend()
    else:
        raise Exception("Invalid backend (must be either serial, mcore, or mpi). Is %s." % backend_type)

    return backend