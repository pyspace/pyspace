""" Serial process execution on the local machine for easy debugging

All processes are executed in sequence in the main
process thread. 
This does not exploit multiple cores or grids but 
simplifies debugging and gives a simple implementation.
"""

import logging
import logging.handlers
import traceback

import pySPACE
from pySPACE.environments.backends.base import Backend
from pySPACE.tools.progressbar import ProgressBar, Percentage, ETA, Bar

class SerialBackend(Backend):
    """ A backend that allows for easy debugging since the program flow
    is not threaded or distributed over several OS processes.
    """
    
    def __init__(self):
        super(SerialBackend, self).__init__()
              
        self.state = "idling"  
        self.current_process = 0
        
        
    def stage_in(self, operation):
        """
        Stage the current operation
        """
        super(SerialBackend, self).stage_in(operation)
        
        # Set up progress bar
        widgets = ['Operation progress: ', Percentage(), ' ', Bar(), ' ', ETA()]
        self.progress_bar = ProgressBar(widgets = widgets, 
                                        maxval = self.current_operation.number_processes)
        self.progress_bar.start()
        
        self._log("Operation - staged")
        self.state = "staged"
        
    def execute(self):
        """
        Executes all processes specified in the currently staged
        operation.
        """
        assert(self.state == "staged")
        
        self.state = "executing" 
        self._log("Operation - executing")
        
        # The handler that is used remotely for logging
        handler_class = logging.handlers.SocketHandler
        handler_args = {"host" : self.host, "port" : self.port}
        
        try:
            process = self.current_operation.processes.get()
        except KeyboardInterrupt:
            self._log(traceback.format_exc(), level = logging.ERROR)
            process = False
        # while there are Processes in the queue ...
        while process != False:   
            process.prepare(pySPACE.configuration, handler_class, handler_args)
            # Execute process, update progress bar and get next queue-element
            try:
                process()
            # if an exception is raised somewhere in the code we maybe want to
            # further try other processes
            except Exception: 
                self._log(traceback.format_exc(), level = logging.ERROR)
                process.post_benchmarking()
            # if ctrl+c is pressed we want to immediately stop everything
            except KeyboardInterrupt:
                self._log(traceback.format_exc(), level = logging.ERROR)
                process.post_benchmarking()
                process = False
            else:    
                self.current_process += 1
                self.progress_bar.update(self.current_process)
                process = self.current_operation.processes.get()
            
    def check_status(self):
        """
        Returns a description of the current state of the operations
        execution.
        
        .. todo:: do we really need this method???
        """
        # Returns which percentage of processes of the current operation
        # is already finished
        return float(self.current_process)/self.current_operation.number_processes
    
    def retrieve(self):
        """
        Returns the result of the operation.
        
        This is trivial in the Debug-Backend since execute blocks.
        """
        assert(self.state == "executing")
        
        self._log("Operation - retrieved")
        
        self.current_operation.processes.close()
        # if process creation has another thread
        if hasattr(self.current_operation, "create_process") \
                        and self.current_operation.create_process != None:
            self.current_operation.create_process.join()
            
        # Change the state to retrieved
        self.state = "retrieved"
    
    
    def consolidate(self):
        """
        Consolidates the results of the single processes into a consistent result of the whole
        operation
        """
        assert(self.state == "retrieved")
        
        try:
            self.current_operation.consolidate()
        except Exception:
            self._log(traceback.format_exc(), level = logging.ERROR)
        
        self._log("Operation - consolidated")
        self.state = "consolidated"
    
    def cleanup(self):
        """
        Remove the current operation and all potential results that
        have been stored in this object
        """
        self.state = "idling"
        
        self._log("Operation - cleaned up")
        self._log("Idling...")

        # Remove the file logger for this operation
        logging.getLogger('').removeHandler(self.file_handler)
        # close listener socket
        self.sock.close()
        
        self.current_operation = None
        self.current_process = 0      
    
