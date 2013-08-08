""" Simple helper script that unpickles and executes a processes using mpi

If a process finishes successfully an empty file with filename 
"processname_Finished" will be created, otherwise a file with filename 
"processname_Crashed" will be created. The file "processname_Crashed"
will contain the reason why the process crashed.

.. todo:: check import statement
 
:Author: Yohannes Kassahun (kassahun@informatik.uni-bremen.de)
:Created: 2011/01/11
"""

import sys
import os
try:
    import cPickle as pickle
except:
    import pickle
try:
    # This import does not work for mac because there is no MPI in mpi4py
    from mpi4py import MPI
except:
    pass

def main():
    size = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    name = MPI.Get_processor_name()
    #sys.stdout.write(
    #  "Starting process %d of %d on %s.\n"
    #   % (rank, size, name))
    # Get the file name of the process to run
    proc_file_name = sys.argv[rank+1]
    # # Extract the pySPACE directory and append root directory to PYTHONPATH
    file_path = os.path.dirname(os.path.abspath(__file__))
    pyspace_path = file_path[:file_path.rfind('pySPACE')-1]
    if not pyspace_path in sys.path:
        sys.path.append(pyspace_path)
    # We need to import pySPACE since the processes are independent of
    # the main program
    # TODO: Check comment!
    try: # process runs normally
        #logging.info("Unpickling process %s" % proc_file_name)
        proc_file = open(proc_file_name, 'r')
        # Openmpi does not like methods which use fork() or system(), so we should
        # change the implementation of the mpi_runner.py to avoid the warning
        # message. This happens in the method pickle.load().
        proc = pickle.load(proc_file)
        proc_file.close()
        #logging.info("Starting process on node %s" % socket.gethostname())
        proc()
        proc_file = open(proc_file_name+"_Finished", "w")
        proc_file.close()
        #sys.stdout.write(
        #  "Finishing process %d of %d on %s.\n"
        #  % (rank, size, name))
    except IOError as (errno, strerror): #process crashes
        proc_file = open(proc_file_name+"_Crashed", "w")
        proc_file.write("Process was running on : %s \n" % name)
        proc_file.write("I/O error({0}): {1}".format(errno, strerror))
        proc_file.close()
    except: #process crashes
        proc_file = open(proc_file_name+"_Crashed", "w")
        e = sys.exc_info()[1]
        proc_file.write("Process was running on : %s \n" % name)
        proc_file.write("Reason for crash: %s " % e)
        proc_file.close()
    
if __name__ == '__main__':
    main()
