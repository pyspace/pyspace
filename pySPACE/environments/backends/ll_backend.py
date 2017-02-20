""" Execute operations on a cluster with the LoadLeveler scheduler """

import glob
import logging
import logging.handlers
import multiprocessing
import os
import select
import socket
import subprocess as sub
import sys
import threading
import time
from collections import defaultdict
from functools import partial

from enum import Enum

import pySPACE
from pySPACE.environments.backends.base import Backend
from pySPACE.missions.operations.base import Operation
from pySPACE.tools.progressbar import ProgressBar, Percentage, ETA, Bar
from pySPACE.tools.socket_utils import inform

try:
    # noinspection PyPep8Naming
    import cPickle as pickle
except ImportError:
    import pickle


class LoadLevelerBackend(Backend):
    """ Commits every process to LoadLeveler cluster, which resumes parallel execution

    Each process corresponds to one combination of input data set and
    parameter choice. The process objects are first pickled.
    The path to the pickled object together with a helper script is then
    submitted to LoadLeveler. There the object is unpickled, called and
    the backend is informed when the results are stored.

    Communication between the independent processes and the backend is
    done via TCP socket connection (see
    :class:`~pySPACE.environments.backends.ll_backend.LoadLevelerComHandler` for detailed
    information).

    :Author: Anett Seeland (anett.seeland@dfki.de)
    :Created: 2011/06/08
    :LastChange: 2012/09/06 Add communication to SubflowHandler
    """
    LL_COMMAND_FILE_TEMPLATE = """
# @ job_type = serial
# @ notification = never
# @ class = {job_class}
# @ resources = ConsumableMemory({memory}) ConsumableCPUs({CPUs})
# @ requirements = {requirements}
# @ executable = {executable}
# @ arguments = {arguments}
# @ output = %(op_result_dir)s/log/pySPACE_$(jobid).out
# @ error = %(op_result_dir)s/log/pySPACE_$(jobid).err
# @ queue"""

    def __init__(self):
        super(LoadLevelerBackend, self).__init__()

        self.state = "idling"
        # create command file template for Loadleveler
        if "job_class" not in pySPACE.configuration or not pySPACE.configuration["job_class"]:
            pySPACE.configuration["job_class"] = "general"
        if "consumable_memory" not in pySPACE.configuration or not pySPACE.configuration["consumable_memory"]:
            pySPACE.configuration["consumable_memory"] = "3250mb"
        if "consumable_cpus" not in pySPACE.configuration or not pySPACE.configuration["consumable_cpus"]:
            pySPACE.configuration["consumable_cpus"] = 1
        if "anode" not in pySPACE.configuration:
            pySPACE.configuration["anode"] = ""

        assert (pySPACE.configuration["job_class"] in ['critical', 'critical_forking',
                                                       'general', 'general_forking',
                                                       'longterm', 'longterm_forking',
                                                       'test']),\
            "LL_Backend:: Job class not existing! Check your pySPACE config file!"

        self.template_file = LoadLevelerBackend.LL_COMMAND_FILE_TEMPLATE.format(
            executable=sys.executable,
            arguments=" ".join([os.path.join(pySPACE.configuration.root_dir,
                                             "environments", "backends", "ll_runner.py"),
                                "%(process_file_path)s", self.SERVER_IP, "%(server_port)d"]),
            job_class=pySPACE.configuration["job_class"],
            memory=pySPACE.configuration["consumable_memory"],
            CPUs=pySPACE.configuration["consumable_cpus"],
            requirements=pySPACE.configuration["anode"])

        self._log("Using '%s' as template", logging.DEBUG)

        # queue for execution
        self.result_handlers = None
        self.progress_bar = None
        self.process_dir = ""
        self._log("Created LoadLeveler Backend.")

    def stage_in(self, operation):
        """
        Stage the given operation.

        :param operation: The operation to stage.
        :type operation: Operation
        """
        super(LoadLevelerBackend, self).stage_in(operation)
        # set up queue
        self.result_handlers = multiprocessing.Queue(200)
        # Set up progress bar
        widgets = ['Operation progress: ', Percentage(), ' ', Bar(), ' ', ETA()]
        self.progress_bar = ProgressBar(widgets=widgets,
                                        maxval=self.current_operation.number_processes)
        self.progress_bar.start()

        self._log("Operation - staged")
        self.state = "staged"

    def execute(self, timeout=1e6):
        """ Execute all processes specified in the currently staged operation """
        assert (self.state == "staged")

        self._log("Operation - executing")
        self.state = "executing"

        # The handler that is used remotely for logging
        handler_class = logging.handlers.SocketHandler
        handler_args = {"host": self.host, "port": self.port}
        # the communication properties to talk to LoadLevelerComHandler
        backend_com = (self.SERVER_IP, self.SERVER_PORT)
        self._log('--> Loadleveler Communication : \n\t\t host:%s, port:%s' % (self.SERVER_IP, self.SERVER_PORT))
        # Prepare the directory where processes are stored before submitted
        # to LoadLeveler
        self.process_dir = os.sep.join([self.current_operation.result_directory, ".processes"])
        if not os.path.exists(self.process_dir):
            os.mkdir(self.process_dir)

        process_counter = 0

        # create and start server socket thread
        self.listener = LoadLevelerComHandler(self.sock, self.result_handlers,
                                              self.progress_bar,
                                              self.template_file,
                                              log_func=self._log,
                                              operation_dir=self.current_operation.result_directory)
        self.listener.start()
        # create a client socket to talk to server socket thread
        send_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        send_socket.connect((self.SERVER_IP, self.SERVER_PORT))
        try:
            # Until not all Processes have been created, prepare all processes
            # from the queue for remote execution and execute them
            get_process = partial(self.current_operation.processes.get, timeout=timeout)
            for process in iter(get_process, False):
                process.prepare(pySPACE.configuration, handler_class, handler_args,
                                backend_com)
                # since preparing the process might be quite faster than executing
                # it we need another queue where processes get out when they have
                # finished execution
                self.result_handlers.put(process)
                # pickle the process object
                proc_file_name = os.sep.join([self.process_dir,
                                              "process_%d.pickle" % process_counter])
                with open(proc_file_name, "wb") as proc_file:
                    pickle.dump(process, proc_file, pickle.HIGHEST_PROTOCOL)

                # fill out LoadLeveler template
                llfile = self.template_file % {
                    "process_file_path": proc_file_name,
                    "server_port": self.SERVER_PORT,
                    "op_result_dir": self.current_operation.result_directory}

                llfilepath = os.path.join(self.current_operation.result_directory, "ll_call.cmd")
                with open(llfilepath, 'w') as f:
                    f.write(llfile)

                # submit to LoadLeveler
                error_counter = 0
                while True:
                    outlog, errlog = sub.Popen(["llsubmit", llfilepath],
                                               stdout=sub.PIPE, stderr=sub.PIPE).communicate()
                    if errlog == "":
                        break
                    elif error_counter < 100:
                        self._log("Warning: Job submission to LoadLeveler failed"
                                  " with %s. Job will be resubmitted." % errlog,
                                  logging.WARNING)
                        time.sleep(1)
                        error_counter += 1
                    else:
                        self._log("Warning: Job submission to LoadLeveler failed %d times"
                                  " with %s. skipping job" % (error_counter, errlog),
                                  logging.WARNING)
                        break

                # parse job_id for monitoring
                loadl_id = outlog.split("\"")[1].split(".")[-1]
                # inform listener that we successfully submitted the job
                # noinspection PyTypeChecker
                send_socket = LoadLevelerComHandler.send_message(send_socket, self.SERVER_IP, self.SERVER_PORT,
                                                                 LoadLevelerComHandler.MESSAGES.SUBMITTED,
                                                                 process_counter, loadl_id)
                # update process_counter
                process_counter += 1

            # send message 'creation finished' to listener
            # noinspection PyTypeChecker
            send_socket = LoadLevelerComHandler.send_message(send_socket, self.SERVER_IP, self.SERVER_PORT,
                                                             LoadLevelerComHandler.MESSAGES.CREATION_FINISHED)
        finally:
            self.listener.creation_finished = True
            send_socket.shutdown(socket.SHUT_RDWR)
            send_socket.close()

    def check_status(self):
        """ Return a description of the current state of the operations execution

        .. todo:: do we really need this method???
        """
        # Returns the current state of the operation
        return self.state

    def retrieve(self, timeout=1e6):
        """
        Wait for all results of the operation

        This call blocks until all processes are finished
        or the given timeout is reached. If the timeout is zero,
        the timeout is disabled.

        :param timeout: The time to wait until a job is considered as "finished"
                        and will be stopped.
        :type timeout: int
        """
        assert (self.state == "executing")
        self._log("All processes submitted. Waiting for finishing.")
        # since self.current_operation.number_processes is not reliable (maybe
        # to high) we wait until the listener thread is terminated
        self.listener.finished.wait(timeout=timeout)
        self._log("Worker processes have exited gracefully")

        self.current_operation.processes.close()

        # if process creation has another thread
        if self.current_operation.create_process is not None:
            self.current_operation.create_process.join(timeout=timeout)
        self.result_handlers.close()
        # join also listener thread
        self.listener.join(timeout=timeout)
        # Change the state to finished
        self._log("Operation - retrieved")
        self.state = "retrieved"
        return True

    def consolidate(self):
        """ Consolidate the single processes' results into a consistent result of the whole operation """
        assert (self.state == "retrieved")

        self.current_operation.consolidate()

        self._log("Operation - consolidated")

        # collect all log file
        def _merge_files(file_list, delete=True):
            result_str = ""
            for filename in file_list:
                tmp_str = ""
                try:
                    if os.path.getsize(filename) != 0:
                        tmp_str += filename.split(os.sep)[-1] + "\n" + len(filename.split(os.sep)[-1]) * "-" + "\n"
                        with open(filename, 'r') as f:
                            tmp_str += f.read()
                        tmp_str += 80 * "-" + "\n"
                    if delete:
                        os.remove(filename)
                except (IOError, OSError), e:
                    self._log("Problems with file %s: %s." % (filename, e), logging.WARNING)
                result_str += tmp_str
            return result_str

        outlist = glob.glob(self.current_operation.result_directory + "/log/pySPACE*.out")
        out = _merge_files(outlist)
        errlist = glob.glob(self.current_operation.result_directory + "/log/pySPACE*.err")
        err = _merge_files(errlist)

        with open(self.current_operation.result_directory + "/pySPACE.out", 'w') as merged_out:
            merged_out.write(out)

        with open(self.current_operation.result_directory + "/pySPACE.err", 'w') as merged_err:
            merged_err.write(err)

        try:
            outlist = glob.glob(self.current_operation.result_directory + "/sub_log/pySPACE*.out")
            out = _merge_files(outlist)
            errlist = glob.glob(self.current_operation.result_directory + "/sub_log/pySPACE*.err")
            err = _merge_files(errlist)

            with open(self.current_operation.result_directory + "/pySPACE_sub.out", 'w') as merged_out:
                merged_out.write(out)

            with open(self.current_operation.result_directory + "/pySPACE_sub.err", 'w') as merged_err:
                merged_err.write(err)
        except IOError:
            pass

        self._log("Process Logging - consolidated")
        self.state = "consolidated"

    def cleanup(self):
        """ Remove the current operation and all potential results that have been stored in this object """
        self.state = "idling"
        try:
            # Remove .process dir
            try:
                if os.path.isdir(self.process_dir):
                    os.rmdir(self.process_dir)
            except OSError, e:
                self._log("Deleting process folder failed with error: %s" % e, level=logging.CRITICAL)
            try:
                os.rmdir(self.current_operation.result_directory + "/log")
                self._log("Operation - logging folder cleaned up")
            except OSError, e:
                self._log("Deleting log folder failed with error %s. "
                          "Maybe no processes started. "
                          "Please check your operation file constraints." % e, level=logging.CRITICAL)
            try:
                os.rmdir(self.current_operation.result_directory + "/sub_log")
                self._log("Operation - subflow log folder cleaned up")
            except OSError:
                pass
        finally:
            # Remove the file logger for this operation
            logging.getLogger('').removeHandler(self.file_handler)
            # close listener socket
            self.sock.shutdown(socket.SHUT_RDWR)
            self.sock.close()
            self._log("Operation - logging closed")
            self.current_operation = None
            self._log("Operation - cleaned up")

class LoadLevelerComHandler(threading.Thread):
    """ Server socket thread for accepting connections and reacting on it

    A helper class for
    :class:`LoadLevelerBackend<pySPACE.environments.backends.ll_backend.LoadLevelerBackend>`,
    which releases finished processes from the execution queue making new
    submits to LoadLeveler possible. In addition the Backends progress bar is
    updated.

    It is also possible to communicate with this thread via the
    :class:`~pySPACE.environments.chains.node_chain.SubflowHandler`
    or via terminal (port and ip can be checked in 'll_call.cmd').
    The benefit of communication is to find out, which processes are running
    and to find out the details on the current state of the processing.

    **Parameters**

        :sock:
            The server socket which listens to messages

        :executing_queue:
            A queue where one element has to be removed if 'finished' is sent.

        :progress_bar:
            A progress bar object with is updated, if 'finished' is sent.

        :loadl_temp:
            A template for a cmd-file that is used to submit subflow-jobs.

    :Author: Anett Seeland (anett.seeland@dfki.de)
    :Created: 2011/06/08
    :LastChange: 2012/09/06 added communication with SubflowHandler
     """

    # to label message end when communicating via socket connection
    JOB_END_TOKEN = "!END!"

    class MESSAGES(Enum):
        NAME = 0
        SUBFLOW_BATCHSIZE = 1
        SUBMITTED = 2
        FINISHED = 3
        CREATION_FINISHED = 4
        EXECUTE_SUBFLOWS = 5
        IS_READY = 6
        GET = 7
        SET = 8

    @classmethod
    def send_message(cls, connection, host, port, message_type, *args):
        """
        Sends a message with the given type and optional args via the given connection.
        If the connection is established, it is used. Otherwise a new
        connection is created and will be returned.

        :param connection: The connection to use for sending.
        :type connection: socket.socket
        :param host: The host to send the message to if the connection is not established
        :type host: basestring
        :param port: The port to use for establishing a new connection
        :type port: int
        :param message_type: The type of the message
        :param args: Optional arguments
        :return: The connection that has been used to send the message
        :rtype: socket.socket
        """
        message = str(message_type.value)
        for arg in args:
            message += ";%s" % arg
        message += cls.JOB_END_TOKEN
        return inform(message, conn=connection, ip_port=(host, port))

    def __init__(self, sock, executing_queue, progress_bar, loadl_temp, log_func, operation_dir=None):
        threading.Thread.__init__(self)
        self.sock = sock
        self.executing_queue = executing_queue
        self.progress_bar = progress_bar
        self.loadl_temp = loadl_temp
        # variables for monitoring
        self.process_loadl_mapping = defaultdict(list)
        self.num_running_processes = 0
        self.subflow_ids_running = set()
        self.subflows_waiting = []
        self.MAX_RUNNING = 400
        self.num_finished_processes = 0
        self.subflow_ids_finished = set()
        self.creation_finished = False
        self.finished = threading.Event()
        # initialize select concept (multiplexing of socket connections)
        self.sock.listen(socket.SOMAXCONN)
        # define potentially readers and writers
        self.readers = [self.sock]
        self.writers = []
        # data structure to store all established connections and messages for
        # reading and writing: data[connection] = [message_read, message_to_write]
        self.data = {}
        # a task queue for subflows to be started; starting is swapped to
        # another thread to be faster in handling all incoming requests
        self.subflow_msg = multiprocessing.Queue()
        self.batch_size = 1
        self.operation_dir = operation_dir
        self._log = log_func

    def run(self):
        """ Accept, read and write on connections until all processes are finished """
        # start thread for handling long messages, i.e. executing subflows
        subflow_starter = SubflowStarter(self.subflow_msg, self.sock,
                                         self.loadl_temp,
                                         log_func=self._log,
                                         operation_dir=self.operation_dir)
        subflow_starter.start()
        while not (self.creation_finished and self.num_running_processes == 0):
            # multiplexing on potentially requests (in self.readers/writers)
            readable, writable, others = select.select(self.readers,
                                                       self.writers, [], 1.0)
            if self.sock in readable:
                conn, _ = self.sock.accept()
                self.readers.append(conn)
                self.data[conn] = ["", ""]
                readable.remove(self.sock)
            for reader in readable:
                try:
                    tmp = reader.recv(4096)
                except socket.error, e:
                    self._log('recv %s' % e, logging.WARNING)
                    self.close_sock(reader)
                else:
                    if tmp:
                        self.data[reader][0] += tmp
                        # Complete messages are processed
                        if self.JOB_END_TOKEN in self.data[reader][0]:
                            self.parse_message(reader, subflow_starter)
                            # New data to send.  Make sure client is in the
                            # server's writer queue.
                            if self.data[reader][1] != "" and reader not in self.writers:
                                self.writers.append(reader)
                    else:
                        self.close_sock(reader)
            for writer in writable:
                try:
                    # send data; tmp is #chars sent (may not be all in write buffer)
                    tmp = writer.send(self.data[writer][1])
                except socket.error, e:
                    self._log('send: %s' % e, logging.WARNING)
                    self.close_sock(writer)
                else:
                    # Removed sent characters from write buffer
                    self.data[writer][1] = self.data[writer][1][tmp:]
                    # If write buffer is empty, remove socket from potentially writers
                    if not self.data[writer][1]:
                        self.writers.remove(writer)
        # send all_tasks_finished signal to thread
        self.subflow_msg.put((False, False, False))
        # give thread some time to realize end of tasks
        time.sleep(1)
        self.subflow_msg.close()
        subflow_starter.join(timeout=1e6)
        # raise event so that backend knows we have finished
        self.finished.set()

    def close_sock(self, conn):
        """
        Close the given connection and remove it from lists of potentially readers/writers

        :param conn: The connection to close
        :type conn: socket.socket
        """
        try:
            conn.shutdown(socket.SHUT_RDWR)
        except socket.error:
            pass
        conn.close()
        if conn in self.readers:
            self.readers.remove(conn)
        if conn in self.writers:
            self.writers.remove(conn)
        del self.data[conn]

    def parse_message(self, conn, subflow_starter):
        """
        Parse incoming message and react

        :param conn: The socket to parse the message from
        :type conn: socket.socket
        :param subflow_starter: A callable to start possible subflows with
        :type subflow_starter: SubflowStarter

        The following string messages can be send:

            :name:
                Send name of the Backend back, e.g. 'loadl' or 'local'.

            :subflow_batchsize;*batchsize*:
                *batch_size* determines how many subflows are executed in one
                serial LoadLeveler job.

            :submitted;*process_nr*;*loadl_id*:
                Informs the listener that the pickled process
                `process_*process_nr*.pickle` has been submitted to LoadLeveler.
                In addition the LoadLeveler job id (without step id) is stored
                to be available for debugging purposes.

            :creation_finished:
                Informs the listener that no further process will be created
                anymore. Based on that the listener has only to process further
                messages until the number of running processes equals zero.

            :finished:
                Informs the listener that a process has been successfully
                executed. Hence, the number of running processes is
                decremented (the Backend is able to submit a new job) and the
                progress bar is updated.

            :finished;*flow_id*:
                Informs the listener that a subflow with unique identifier
                *flow_id* has stored its results.

            :is_ready;*nr_subflows*;*subflow_ids*:
                Asks the listener which of the *nr_subflows* subflows
                (identified by their subflow_id) have already finished executing.
                *subflow_ids* must be a string representation of a set. The
                listener sends the set of finished ids back.

            :execute_subflows;*path*;*nr_subflows*;*subflow_ids*;*runs*:
                Asks the listener to execute *nr_subflows* subflows.
                *path* is the absolute path where the subflows (named by their
                unique subflow id) are stored on disk, e.g. the *temp_dir* of
                a node. Hence, *subflow_ids* is a list of the filenames in
                *path* (without file extension) that have been stored before in
                pickle-format. *runs* is a list containing the run numbers the
                flow should be executed with: the *run_number* determines the
                random seed, e.g., for a splitter node. In addition it is
                assumed, that the data the subflows need for execution is also
                present in *path* (as 'subflow_data.pickle').
                Since the reaction of this request may take quite some time, it
                is swapped to another
                :class:`Thread<pySPACE.environments.backends.ll_backend.SubflowStarter>`.

            :get;*attribute_name*:
                Getter methods for debugging purposes. Sends the value of
                *attribute_name* back.

            :set;*attribute_name*;*value*:
                Setter methods for debugging purposes. Equals to
                'self.attribute_name = value'. Note that *value* must be
                evaluable.

        .. note:: It is important that you don't forget the end_flog at each
                  message!
        """
        end_ind = self.data[conn][0].find(self.JOB_END_TOKEN)
        message = self.data[conn][0][:end_ind]
        message_type = message.split(";")[0]
        message_args = message.split(";")[1:]

        # noinspection PyBroadException
        try:
            message_type = self.MESSAGES(int(message_type))
            if message_type == self.MESSAGES.NAME:
                self.data[conn][1] = 'loadl%s' % self.JOB_END_TOKEN
            elif message_type == self.MESSAGES.SUBFLOW_BATCHSIZE:
                batch_size = eval(message_args[-1])
                subflow_starter.batch_size = batch_size
                self.batch_size = batch_size
            elif message_type == self.MESSAGES.SUBMITTED:
                key = eval(message_args[0])
                self.process_loadl_mapping[key].append(message_args[1])
                self.num_running_processes += 1
            elif message_type == self.MESSAGES.FINISHED:
                # it is a Process that has finished
                if not message_args:
                    self.num_finished_processes += 1
                    self.num_running_processes -= 1
                    self.executing_queue.get(timeout=1e100)
                    self.progress_bar.update(self.num_finished_processes)
                # it is a subflow that has finished
                else:
                    self.subflow_ids_running.remove(message_args[0])
                    self.subflow_ids_finished.add(message_args[0])
                    self.data[conn][1] = "OK%s" % self.JOB_END_TOKEN
            elif message_type == self.MESSAGES.CREATION_FINISHED:
                self.creation_finished = True
            elif message_type == self.MESSAGES.EXECUTE_SUBFLOWS:
                path, runs = message_args[0], message_args[-1]
                nr_subflows, subflow_ids = [eval(s) for s in message_args[1:-1]]
                assert (nr_subflows == len(subflow_ids)), "incorrect number of subflows"
                # check if we can submit new jobs
                self.control_subflow_submission(path, runs, subflow_ids)
                self.data[conn][1] = "OK%s" % self.JOB_END_TOKEN
            elif message_type == self.MESSAGES.IS_READY:
                nr_requested, requested_subflows = [eval(s) for s in message_args]
                assert (nr_requested == len(requested_subflows)), "incorrect number" \
                                                                  " of subflows"
                # check which subflows have already finished and tell to client
                finished = requested_subflows & self.subflow_ids_finished
                # .. todo: maybe reduced self.subflow_ids_finished since they are
                # unique and will never be requested again
                self.data[conn][1] = str(finished) + self.JOB_END_TOKEN
            # to be able to communicate with the backend in case something went
            # wrong give the user the possibility to get and set attributes
            elif message_type == self.MESSAGES.GET:
                self._log("variable wanted", logging.INFO)
                name = message_args[0]
                self.data[conn][1] = str(getattr(self, name, None))
            elif message_type == self.MESSAGES.SET:
                name, value = message_args[0], eval(message_args[1])
                setattr(self, name, value)
            else:
                self._log("Got unknown message: %s" % message, logging.WARNING)

            self.data[conn][0] = self.data[conn][0][end_ind + len(self.JOB_END_TOKEN):]

            # try to submit waiting jobs
            if self.subflows_waiting != [] and (
                    self.MAX_RUNNING - (len(self.subflow_ids_running) / self.batch_size) > 0):
                path, runs, subflow_ids = self.subflows_waiting.pop()
                self.control_subflow_submission(path, runs, subflow_ids)
        except Exception, e:
            self._log("Exception while parsing message '%s' %s:" % (message, e), logging.ERROR)

    def control_subflow_submission(self, path, runs, subflow_ids):
        """ Give SubflowStarter only a limited amount of tasks

        The MAX_RUNNING variable specifies how many loadleveler subflow jobs
        should maximal run in parallel. With respect to this number the
        SubflowStarter gets only tasks if there are available slots.
        """
        free_slots = self.MAX_RUNNING - (len(self.subflow_ids_running) / \
                                                              self.batch_size)
        if free_slots < (len(subflow_ids) / self.batch_size):
            # cannot submit all jobs at once
            end_ind = free_slots * self.batch_size
            self.subflow_msg.put((path, runs, subflow_ids[:end_ind]))
            self.subflows_waiting.append((path, runs, subflow_ids[end_ind:]))
            self.subflow_ids_running.update(subflow_ids[:end_ind])
        else:
            self.subflow_msg.put((path, runs, subflow_ids))
            self.subflow_ids_running.update(subflow_ids)


class SubflowStarter(threading.Thread):
    """ Submit subflows to LoadLeveler

    A helper class for
    :class:`~pySPACE.environments.backends.ll_backend.LoadLevelerComHandler`,
    which submits subflows as independent jobs to LoadLeveler. The reason behind
    it is a better (more fair) scheduling of jobs when several users run the software on
    a cluster due to the fact that each single job has a short computation time.

    Since LoadLeveler does not know that Processes have nothing to do than
    wait when their subflows have to be executed, another job class is used for
    the subflow that preempt the Processes. The name of this class ends with
    '_child'.

    **Parameters**

        :task_queue:
            This queue is filled with jobs from incoming requests by the
            :class:`LoadLevelerComHandler<pySPACE.environments.backends.ll_backend.LoadLevelerComHandler>`.
            Each job is specified by a tuple: (*path*,*runs*,*subflow_ids*).

        :sock:
            The listener socket to be able to get information about it. Needed
            to fill out the LoadLeveler command file.

        :loadl_temp:
            LoadLeveler command file template that has been partly filled out
            by :class:`LoadLevelerBackend<pySPACE.environments.backends.ll_backend.LoadLevelerBackend>`
            but needs to be further adjusted.

    .. todo:: create parallel Loadleveler jobs to reduce request from processes;
              its better to execute all subflows of one process first than
              one subflow of every process

    .. todo:: actually the server socket is not needed here, only the server port
    """

    def __init__(self, task_queue, sock, loadl_temp, log_func, operation_dir=None):
        threading.Thread.__init__(self)
        self.task_queue = task_queue
        self.sock = sock
        self.loadl_temp = loadl_temp
        self.first_subflow = True
        self.batch_size = 1
        self.operation_dir = operation_dir
        self._log = log_func

    def submit(self, path, runs, subflow_ids):
        """ Submit one subflow job to LoadLeveler """
        # TODO: Add path, runs and subflow_ids to docstring
        if self.operation_dir is None:
            operation_dir = path
        else:
            operation_dir = self.operation_dir
        llfile = self.loadl_temp % {
            "process_file_path": path,
            "server_port": self.sock.getsockname()[1],
            "op_result_dir": operation_dir,
            "runs": '"' + str(runs) + '"',
            "subflow_ids": str(subflow_ids)}

        llfilepath = os.path.join(path, "ll_call.cmd")
        with open(llfilepath, 'w') as f:
            f.write(llfile)

        # submit to LoadLeveler
        error_counter = 0
        while True:
            outlog, errlog = sub.Popen(["llsubmit", llfilepath],
                                       stdout=sub.PIPE, stderr=sub.PIPE).communicate()
            if errlog == "":
                break
            elif error_counter < 100:
                self._log("Warning: Job submission to LoadLeveler failed"
                          " with %s. Job will be resubmitted." % errlog,
                          logging.WARNING)
                time.sleep(1)
                error_counter += 1
            else:
                self._log("Warning: Job submission to LoadLeveler failed %d times"
                          " with %s. skipping job" % (error_counter, errlog),
                          logging.WARNING)
                break

    def run(self):
        """ Portion and execute subflow tasks until *subflow_ids* equals 'False' """
        rest_ids = []
        path, runs, subflow_ids = self.task_queue.get(timeout=1e10)
        while subflow_ids:
            if self.first_subflow:  # need this modification only once
                # modify template for subflows: class, ll_runner, params
                template = self.loadl_temp.split("\n")
                self.loadl_temp = ""
                for line in template:
                    if line.startswith('# @ class'):
                        assert ("forking" in line), "When using subflow " \
                                                    "paralellization a job class with forking " \
                                                    "functionality has to be used!"
                        line = line.replace("forking", "child")
                    elif line.startswith('# @ arguments'):
                        line = line.replace('ll_runner.py',
                                            'll_subflow_runner.py ')
                        line += ' %(runs)s "%(subflow_ids)s"'
                    elif line.startswith('# @ output') or line.startswith('# @ error'):
                        line = line.replace("/log/pySPACE_", "/sub_log/pySPACE_")
                    self.loadl_temp += line + " \n"
                self.first_subflow = False

            # check whether nr of subflows is dividable by batch_size
            if len(subflow_ids) % self.batch_size:
                rest = len(subflow_ids) % self.batch_size
                rest_ids = subflow_ids[-1 * rest:]
                subflow_ids = subflow_ids[:len(subflow_ids) - rest]

            # iterate over loadlevler jobs
            for ind in range(0, len(subflow_ids), self.batch_size):
                self.submit(path, runs, subflow_ids[ind:ind + self.batch_size])
            if rest_ids:  # need to submit one more job
                self.submit(path, runs, rest_ids)
            path, runs, subflow_ids = self.task_queue.get(timeout=1e10)
