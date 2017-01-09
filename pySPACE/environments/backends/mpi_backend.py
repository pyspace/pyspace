""" Distribute the processes on a High Performance Cluster

Very simple Implementation.

.. todo:: Documentation

:Author: Yohannes Kassahun (kassahun@informatik.uni-bremen.de)
:Created: 2011/01/11
"""

import os
import sys
import logging
import logging.handlers
import cPickle
import shutil
import time
import subprocess


import pySPACE
from pySPACE.environments.backends.base import Backend
from pySPACE.tools.progressbar import ProgressBar, Percentage, ETA, Bar


class MpiBackend(Backend):
    """ 
    A message passing interface (mpi) backend to pySPACE
    
    In order to use this backend, you need a working MPI distribution and mpi4py. 
    You can download mpi4py from http://code.google.com/p/mpi4py/. mpi4py is 
    compatible with a python 2.3  to 2.7 or 3.0 to 3.1 distribution. 

    This backend assumes a global file system that is seen by all nodes running 
    the processes. 
    
    **Parameters**
        :pool_size: Define how many MPI processes should be started in parallel.
                    This should not exceed the amount of available processors.
                    (or the number of mpi slots defined in the hostsfile)
        
            (*recommended, default: 156*)
 
    """
    
    def __init__(self, pool_size = 156):
        super(MpiBackend, self).__init__()
        #self.COMMAND_MPI = '/usr/lib64/openmpi/bin/mpirun'
        self.COMMAND_MPI = 'mpirun'
        self.COMMAND_PYTHON = sys.executable
        self.runner_script = os.sep.join([pySPACE.configuration.root_dir,
                             "environments",
                             "backends",
                             "mpi_runner.py"])
        # start as many processes as the total number of processors
        # available
        self.NumberOfProcessesToRunAtBeginning = pool_size
        self.NumberOfProcessesToRunLater = pool_size #39
    def __del__(self):
        pass
        

    def stage_in(self, operation):
        """
        Stage the current operation
        """
        super(MpiBackend, self).stage_in(operation)
        # init of process lists, because backend is only initialized once        
        self.process_args_list = []
        self.IndexCopyStart = 0
        self.ProcessingSuccessful = True
        self.TotalProcessesFinished = 0
        self.CrashedProcesses = []
        # Set up progress bar
        widgets = ['Operation progress: ', Percentage(), ' ', Bar(), ' ', ETA()]
        self.progress_bar = ProgressBar(widgets = widgets, 
                                       maxval = self.current_operation.number_processes)
        self.progress_bar.start()

        # The handler that is used remotely for logging
        handler_class = logging.handlers.SocketHandler
        handler_args = {"host" : self.host, "port" : self.port}
        
        # Set up stage in directory
        stagein_dir = os.sep.join([self.current_operation.result_directory,
                                   ".stagein"])
        # Check if hosts file is created in the right directoy
        HostfileCreated = pySPACE.configuration.root_dir+ "/" +'hostsfile'
        if (not os.path.isfile(HostfileCreated)):
            print "***************************************************************************************************"
            print "hostsfile not created !"
            print "Please create the hosts file with a filename 'hostsfile' under ", pySPACE.configuration.root_dir
            print "***************************************************************************************************"
            raise UserWarning('Missing hostsfile.')
        if not os.path.exists(stagein_dir):
            os.mkdir(stagein_dir)   

        process = self.current_operation.processes.get()
        print "Preparing processes. This might take a few minutes...."
        # Until not all Processes have been created prepare all processes
        # from the queue for remote execution and execute them
        i = 0
        while process != False:
            process.prepare(pySPACE.configuration, handler_class, handler_args)
            # since preparing the process might be quite faster than executing
            # it we need another queue where processes get out when they have
            # finished execution
            #self.result_handlers.put(1)
            # Execute all functions in the process pool but return immediately
            #self.pool.apply_async(process, callback=self.dequeue_process)
            proc_file_name = os.sep.join([stagein_dir,
                                          "process_%d.pickle" % i])
            proc_file = open(proc_file_name, "w")
            cPickle.dump(process, proc_file)
            proc_file.close()
            # Add task to job specification
            self.process_args_list.append(proc_file_name)
            # Get the next process
            process = self.current_operation.processes.get()
            i+=1

        self._log("Operation - staged")
        self.state = "staged"        
        
    def execute(self, timeout=1e6):
        """
        Executes all processes specified in the currently staged
        operation.
        """
        assert(self.state == "staged")
        
        
    def check_status(self):
        """
        Returns a description of the current state of the operations
        execution. 
        """
        #self.progress_bar.update(float(self.current_job.info()["percentDone"]))
        #return float(self.current_job.info()["percentDone"]) / 100.0
        #return float(self.current_process) / self.current_operation.number_processes
        return 1.0

    def not_xor(self, a, b):
        return not((a or b) and not (a and b))
    
    def retrieve(self, timeout=1e10):
        """
        Returns the result of the operation.
        """
        
        self.state = "executing" 
        self._log("Operation - executing") 
        if (self.NumberOfProcessesToRunAtBeginning > len(self.process_args_list)):
            args = ([self.COMMAND_MPI] +
                ['--loadbalance']+
                ['--nolocal']+
                ['--hostfile'] +
                [pySPACE.configuration.root_dir+ "/" +'hostsfile'] +
                ['-n', str(len(self.process_args_list))] +
                [self.COMMAND_PYTHON] +  
                [self.runner_script] + 
                self.process_args_list)
            # Start the processes. 
            self._log("mpi-parameters: %s" % args, level=logging.DEBUG);
            self._log("mpi-parameters-joined: %s" % os.path.join(args), level=logging.DEBUG);
            p =subprocess.Popen(args)
            #self.pids.append(p)
            self.IndexCopyStart += self.NumberOfProcessesToRunAtBeginning
            #print args
        else:
            #copy the arguments of the processes to run
            sub_process_args_list = (self.process_args_list[self.IndexCopyStart: 
                                     self.NumberOfProcessesToRunAtBeginning])
            args = ([self.COMMAND_MPI] +
                ['--loadbalance']+
                ['--nolocal']+
                ['--hostfile'] +
                [pySPACE.configuration.root_dir+ "/" +'hostsfile'] +
                ['-n', str(len(sub_process_args_list))] +
                [self.COMMAND_PYTHON] +  
                [self.runner_script] + 
                sub_process_args_list)
            # Start the processes. 
            p = subprocess.Popen(args)
            #self.pids.append(p) # TODO: call p.poll() for p in self.pids after all processes have exited
            self.IndexCopyStart += self.NumberOfProcessesToRunAtBeginning
            #print args

        # Create a list of boolean for processes which are finished.
        # First we assume that all processes are not started, so we set
        # every element of the list to false
        FinishedProcesses=[False for i in range(len(self.process_args_list))] 
        
        # Wait until all processes finish and start new processes
        # when old ones finish

        print "Waiting for the processes to finish...."

        # Counter for the processes which are finished. It will be reset
        # after 'NumberOfProcessesToRunLater' processes are finished
        CounterProcessesFinished = 0
        processes_Finished = False

        while not processes_Finished:
          try:
             processes_Finished = True
             for LoopCounter, process_args in enumerate(self.process_args_list):
                 if (self.not_xor (os.path.isfile(process_args+"_Finished"), 
                               os.path.isfile(process_args+"_Crashed"))):
                    processes_Finished = False
                 else:
                    if (FinishedProcesses[LoopCounter] == False):
                       # Record that the process is finished                       
                       FinishedProcesses[LoopCounter] = True
                       # If the process is crashed take note of that
                       if (os.path.isfile(process_args+"_Crashed")):
                           self.CrashedProcesses.append(process_args)
                       # Increment the counter for the number of processes finished
                       # by one
                       CounterProcessesFinished += 1
                       self.TotalProcessesFinished += 1 
                       # update the progress bar
                       self.progress_bar.update(self.TotalProcessesFinished)
                       if (CounterProcessesFinished == self.NumberOfProcessesToRunLater):
                          # Define a variable for a subset of processes to run
                          sub_process_args_list = []
                          if (self.IndexCopyStart==len(self.process_args_list)):
                              break
                          elif ((self.IndexCopyStart+self.NumberOfProcessesToRunLater)< len(self.process_args_list)):
                              sub_process_args_list = (self.process_args_list[self.IndexCopyStart:
                                                       self.IndexCopyStart +self.NumberOfProcessesToRunLater])
                          else:
                              sub_process_args_list = self.process_args_list[self.IndexCopyStart:len(self.process_args_list)]
                          args = ([self.COMMAND_MPI] +
                                 ['--loadbalance']+
                                 ['--nolocal']+
                                 ['--hostfile'] +
                                 [pySPACE.configuration.root_dir+ "/" +'hostsfile'] +
                                 ['-n', str(len(sub_process_args_list))] +
                                 [self.COMMAND_PYTHON] +  
                                 [self.runner_script] + 
                                 sub_process_args_list)
                          # Start the processes
                          if (len(sub_process_args_list) > 0):
                             p = subprocess.Popen(args)
                          #print args                          
                          # Adjust the start index
                          self.IndexCopyStart += self.NumberOfProcessesToRunLater
                          # Reset the counter for processes finished
                          CounterProcessesFinished = 0
             # sleep for one second                
             time.sleep(1)
          except (KeyboardInterrupt, SystemExit): # if processes hang forever
            self.ProcessingSuccessful = False
            print "*********************************************************************************************************"
            print "pySPACE forced to stop ..."
            print "Please wait until mpi_backend is finished with consolidating the results generated and with clean up ..."
            print "**********************************************************************************************************"
            import pySPACE.resources.dataset_defs.performance_result.PerformanceResultSummary as PerformanceResultSummary
            # merge the remaining files
            print "***************************************************************************************************"
            print "Starting merging . . ."
            PerformanceResultSummary.merge_performance_results(self.current_operation.result_directory)
            print "Merging complete . . ."
            print "***************************************************************************************************"
            break #The while loop will break

        self._log("Operation - processing finished")
        
        # Change the state to retrieved
        self.state = "retrieved"
        
        return None


    def consolidate(self):
        """
        Consolidates the results of the single processes into a consistent result of the whole
        operation
        """
        assert(self.state == "retrieved")
        
        if ((self.ProcessingSuccessful ==True) and (len(self.CrashedProcesses) == 0)):
            self.current_operation.consolidate()
                 
        if ((self.ProcessingSuccessful ==True) and (len(self.CrashedProcesses) != 0)):
            import pySPACE.resources.dataset_defs.performance_result.PerformanceResultSummary as PerformanceResultSummary
            # merge the remaining files
            print "***************************************************************************************************"
            print "Starting merging . . ."
            PerformanceResultSummary.merge_performance_results(self.current_operation.result_directory)
            print "Merging complete . . ."
            print "***************************************************************************************************"

        self._log("Operation - consolidated")
        
        self.state = "consolidated"
        
        
    def cleanup(self):
        """
        Remove the current operation and all potential results that
        have been stored in this object
        """
        self.state = "idling"

        # Cleaning up...
        stagein_dir = os.sep.join([self.current_operation.result_directory,
                                   ".stagein"])
        if ((self.ProcessingSuccessful == True) and (len(self.CrashedProcesses) == 0)):
           deleted = False

           while not deleted:
               try:
                  os.chdir("..")
                  shutil.rmtree(stagein_dir)
                  deleted = True
               except OSError, e:
                  if e.errno == 66:
                     self._log("Could not remove .stagein dir " 
                             ", waiting for NFS lock",
                              level=logging.WARNING)
                  time.sleep(5)
               
        self._log("Operation - cleaned up")
        self._log("Idling...")
        
        # Remove the file logger for this operation
        logging.getLogger('').removeHandler(self.file_handler)       
        # close listener socket
        self.sock.close()
        self.current_operation = None 
        
    
