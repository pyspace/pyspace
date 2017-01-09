""" Execute as many processes in parallel as there are (logical) CPUs on the local machine

This backend is based on the multiprocessing package and should work on every
multicore system without additional settings even on virtual machines.
"""

import os
import time
import multiprocessing
import logging
import logging.handlers
import threading
import socket
import select
import cPickle
import warnings
from functools import partial

import pySPACE
from pySPACE.environments.backends.base import Backend
from pySPACE.tools.progressbar import ProgressBar, Percentage, ETA, Bar


class MulticoreBackend(Backend):
    """ Execute as many processes in parallel as there are (logical) CPUs on the local machine
    
    This backend is based on the multiprocessing package and should work on every
    multicore system without additional settings even on virtual machines.
    Each process corresponds to one combination of input data set and
    parameter choice.
    
    :Author: Anett Seeland (anett.seeland@dfki.de)
    :LastChange: 2012/09/24
    
    """
    def __init__(self, pool_size=None):
        super(MulticoreBackend, self).__init__()
        
        # Set the number of processes in the pool
        # per default to the number of CPUs
        if pool_size is None:
            pool_size = MulticoreBackend.detect_CPUs()
        self.pool_size = pool_size
        self.state = "idling"

        # queue for execution
        self.result_handlers = []
        self.pool = None
        self.current_process = 0
        self._log("Created MulticoreBackend with pool size %s" % pool_size)
    
    def reset_queue(self):
        """ Resets the execution queue"""
        self.result_handlers = []
        
    def stage_in(self, operation):
        """ Stage the current operation """
        super(MulticoreBackend, self).stage_in(operation)
        self.pool = multiprocessing.Pool(processes = self.pool_size)
        
        # Set up progress bar
        widgets = ['Operation progress: ', Percentage(), ' ', Bar(), ' ', ETA()]
        self.progress_bar = ProgressBar(widgets = widgets, 
                               maxval = self.current_operation.number_processes)
        self.progress_bar.start()
        
        self._log("Operation - staged")
        self.state = "staged"
        
    def execute(self, timeout=1e6):
        """ Execute all processes specified in the currently staged operation """
        assert(self.state == "staged")

        self._log("Operation - executing")
        self.state = "executing"

        # The handler that is used remotely for logging
        handler_class = logging.handlers.SocketHandler
        handler_args = {"host" : self.host, "port" : self.port}
        backend_com = (self.SERVER_IP, self.SERVER_PORT)

        # A socket communication thread to handle e.g. subflows
        self.listener = LocalComHandler(self.sock)
        self.listener.start()

        # Until not all Processes have been created prepare all processes
        # from the queue for remote execution and execute them
        get_process = partial(self.current_operation.processes.get, timeout=timeout)
        for process in iter(get_process, False):
            process.prepare(pySPACE.configuration, handler_class, handler_args,
                            backend_com)
            # Execute all functions in the process pool but return immediately
            self.result_handlers.append(
                self.pool.apply_async(process, callback=self.dequeue_process))

    def dequeue_process(self, result):
        """ Callback function for finished processes """
        self.current_process += 1
        self.progress_bar.update(self.current_process)
    
    def check_status(self):
        """ Return a description of the current state of the operations execution
        
        .. todo:: do we really need this method???
        """
        # Returns which percentage of processes of the current operation
        # is already finished
        return float(self.current_process) / self.current_operation.number_processes
    
    def retrieve(self, timeout=1e10):
        """ Wait for all results of the operation
        
        This call blocks until all processes are finished.
        """
        assert(self.state == "executing")
        
        # Prevent any other processes from being submitted to the pool
        # (necessary for join)
        self.pool.close() 
        self._log("Closing pool", level=logging.DEBUG)
            
        self._log("Operation - retrieved")
        self.current_operation.processes.close()
        # if process creation has another thread
        if hasattr(self.current_operation, "create_process") \
            and self.current_operation.create_process != None:
            self.current_operation.create_process.join(timeout=timeout)
        # Close the result handler and wait for every process
        # to terminate
        try:
            for result in self.result_handlers:
                result.wait(timeout=timeout)
        except multiprocessing.TimeoutError:
            # A timeout occurred, terminate the pool
            self._log("Timeout occurred, terminating worker processes")
            self.pool.terminate()
            return False
        finally:
            self.pool.join() # Wait for worker processes to exit
            # inform listener that its time to die
            self.listener.operation_finished = True
            time.sleep(1)
            self.listener.join(timeout=timeout)
            # Change the state to finished
            self.state = "retrieved"
        self._log("Worker processes have exited gracefully")
        return True
    
    def consolidate(self):
        """ Consolidate the single processes' results into a consistent result of the whole operation """
        assert(self.state == "retrieved")
        try:
            self.current_operation.consolidate()
        except Exception:
            import traceback
            self._log(traceback.format_exc(), level = logging.ERROR)
        self._log("Operation - consolidated")
        self.state = "consolidated"
    
    def cleanup(self):
        """ Remove the current operation and all potential results that have been stored in this object """        
        self.state = "idling"
        self._log("Operation - cleaned up")
        self._log("Idling...")
        # Remove the file logger for this operation
        logging.getLogger('').removeHandler(self.file_handler)
        # close listener socket
        self.sock.close()
        self.current_operation = None
        self.current_process = 0

    @classmethod
    def detect_CPUs(cls):
        """ Detects the number of CPUs on a system. Cribbed from pp.
        
        :from: http://codeliberates.blogspot.com/2008/05/detecting-cpuscores-in-python.html
        """
        ncpus = None
        # Linux, Unix and MacOS:
        if hasattr(os, "sysconf"):
            if os.sysconf_names.has_key("SC_NPROCESSORS_ONLN"):
                # Linux & Unix:
                 ncpus = os.sysconf("SC_NPROCESSORS_ONLN")
            if isinstance(ncpus, int) and ncpus > 0:
                return ncpus
            else: # OSX:
                return int(os.popen2("sysctl -n hw.ncpu")[1].read())
        # Windows:
        if os.environ.has_key("NUMBER_OF_PROCESSORS"):
            ncpus = int(os.environ["NUMBER_OF_PROCESSORS"]);
            if ncpus > 0:
                return ncpus
        return 1 # Default        

class LocalComHandler(threading.Thread):
    """ Server socket thread for accepting connections and reacting on it
    
    A helper class for :class:`~pySPACE.environments.backends.multicore.MulticoreBackend`, 
    which handles incoming connections (e.g. from nodes that want to
    compute subflows).
    
    **Parameters**
    
        :sock:
            The socket object to which messages are send.
    """
    def __init__(self, sock):
        threading.Thread.__init__(self)
        self.sock = sock
        self.subflow_pool = None
        self.results = {}
        # variables for monitoring
        self.subflow_ids_running = set()
        self.subflow_ids_finished = set()
        # flag from backend to stop run-method
        self.operation_finished = False
        # initialize select concept (multiplexing of socket connections)
        self.sock.listen(socket.SOMAXCONN)
        # define potentially readers and writers 
        self.readers = [self.sock]
        self.writers = []
        # data structure to store all established connections and messages for
        # reading and writing: data[connection] = [message_read, message_to_write]
        self.data = {}
        # end flag of messages
        self.end_token = "!END!"
        
    def run(self):
        """ Accept, read and write on connections until the operation is finished """ 
        while not (self.operation_finished):
            # multiplexing on potentially requests (in self.readers/writers)
            readable, writable, others = select.select(self.readers, 
                                                          self.writers, [], 1.0)
            if self.sock in readable:
                conn, _ = self.sock.accept()
                self.readers.append(conn)
                self.data[conn] = ["",""]
                readable.remove(self.sock)
            for reader in readable:
                try:
                    tmp = reader.recv(4096)
                except socket.error,e:
                    warnings.warn('recv '+ str(e))
                    self.close_sock(reader)
                else:                
                    if tmp:
                        self.data[reader][0] += tmp
                        # Complete messages are processed
                        if self.end_token in self.data[reader][0]:
                            self.parse_message(reader)
                            # New data to send.  Make sure client is in the
                            # server's writer queue.
                            if self.data[reader][1] != "" and reader not in self.writers:
                                self.writers.append(reader)
                    else:
                        self.close_sock(reader)
            for writer in writable:
                try:
                    # send data; tmp is #chars sent (may not be all in writbuf).
                    tmp = writer.send(self.data[writer][1])
                except socket.error,e:
                    warnings.warn('send: '+ str(e))
                    self.close_sock(writer)
                else:
                    # Removed sent characters from writbuf.
                    self.data[writer][1] = self.data[writer][1][tmp:]
                    # If writbuf is empty, remove socket from potentially writers
                    if not self.data[writer][1]:
                        self.writers.remove(writer)
        if not self.subflow_pool is None:
            self.subflow_pool.close()
            self.subflow_pool.join(timeout=1e10)
    
    def close_sock(self, conn):
        """ Close connection and remove it from lists of potentially readers/writers """
        conn.close()
        if conn in self.readers:
            self.readers.remove(conn)
        if conn in self.writers:
            self.writers.remove(conn)
        del self.data[conn]
    
    def parse_message(self, conn):
        """ Parse incoming message and react 
        
        The following string messages can be send:
        
            :name:
                Sends back the name of the backend, i.e. 'mcore'.
            
            :subflow_poolsize;*poolsize*:
                Create a multiprocessing Pool object with *poolsize* worker
                threads for executing subflows.
            
            :is_ready;*nr_subflows*;*subflow_ids*:
                Asks the listener which of the *nr_subflows* subflows 
                (identified by their subflow_id) have already finished executing. 
                *subflow_ids* must be a string representation of a set. The
                listener sends the set of finished ids back. 
            
            :execute_subflows;*path*;*nr_subflows*;*subflow_obj*;*runs*:
                Asks the listener to execute *nr_subflows* subflows via a
                multiprocessing Pool. *path* is the absolute path where the 
                training data is stored, e.g. the *temp_dir* of a node. 
                *subflow_obj* are pickled strings of the subflows. *runs* is a 
                list containing the run numbers the flow should be executed 
                with: the *run_number* determines the random seed, e.g., for a 
                splitter node.
            
            :send_results;*subflow_ids*:
                Sends back a list of results (PerformanceResultSummary) of *subflow_ids*.
            
        """
        end_ind = self.data[conn][0].find(self.end_token)
        message = self.data[conn][0][:end_ind]
        if message == 'name':
            self.data[conn][1] = 'mcore' + self.end_token
        elif message.startswith('subflow_poolsize'):
            if self.subflow_pool == None:
                text = message.split(';')
                self.subflow_pool = multiprocessing.Pool(processes=int(text[1]))  
        elif message.startswith('execute_subflows'):
            text = message.split(';')
            if len(text) > 5: # splitted within pickled object :-(
                subflow_str = eval(";".join(text[3:-1]))
            else:
                subflow_str = eval(text[3])
            path, runs, nr_subflows = text[1], eval(text[-1]), eval(text[2])
            subflows = [cPickle.loads(s) for s in subflow_str]
            subflow_ids = [s.id for s in subflows]
            assert(nr_subflows == len(subflows)), "incorrect number of subflows"
            # load training data and submit calculation to the pool
            training_data_path = os.path.join(path,'subflow_data.pickle')
            train_instances = cPickle.load(open(training_data_path, 'rb'))            
            for subflow in subflows:
                self.subflow_pool.apply_async(func=subflow, 
                                        kwds={"train_instances":train_instances,
                                              "runs": runs}, 
                                        callback=self.subflow_finished)
            # minitor running jobs
            self.subflow_ids_running.update(subflow_ids)
        elif message.startswith('is_ready'):
            text = message.split(';')
            nr_requested, requested_subflows = [eval(s) for s in text[1:]]
            assert(nr_requested == len(requested_subflows)), "incorrect number"\
                                                                  " of subflows"
            # check which subflows have already finished and tell to client
            finished = requested_subflows & self.subflow_ids_finished
            # .. todo: maybe reduced self.subflow_ids_finished since they are
            # unique and will never be requested again
            self.data[conn][1] = str(finished) + self.end_token
        elif message.startswith('send_results'):
            text = message.split(';')
            subflow_ids = eval(text[1])
            requested_results = [cPickle.dumps(self.results[i], 
                                               cPickle.HIGHEST_PROTOCOL) \
                                     for i in subflow_ids]
            # delete requested results to free memory
            for key in subflow_ids:
                del self.results[key]
            self.data[conn][1] = str(requested_results) + self.end_token
        else:
            warnings.warn("Got unknown message: %s" % message)
        self.data[conn][0] = self.data[conn][0][end_ind+len(self.end_token):]
        
    def subflow_finished(self, result):
        """ Callback method for pool execution of subflows """
        # result is a tuple of flow_id and PerformanceResultSummary
        flow_id, result_collection = result
        self.results[flow_id]= result_collection
        self.subflow_ids_running.remove(flow_id)
        self.subflow_ids_finished.add(flow_id)
