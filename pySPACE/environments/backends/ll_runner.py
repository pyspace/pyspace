""" Executable script for Loadleveler to start processes

A simple helper script that unpickles, executes and deletes a given process.
Afterwards the 
:class:`LoadLevelerComHandler<pySPACE.environments.backends.ll_backend.LoadLevelerComHandler>`
of :mod:`LoadLeveler Backend<pySPACE.environments.backends.ll_backend>` is informed that
execution is finished.

:Author: Anett Seeland (anett.seeland@dfki.de)
:Created: 2011/04/06
:Last Change: 2012/09/04
"""
import socket
import os
import sys
import cPickle

def main():
        # Extract the trunk directory and append to PYTHONPATH cause we need
        # the specific operation process class
        # backends, environments, pySPACE are the parts to go back
        file_path = os.path.dirname(os.path.abspath(__file__))
        pyspace_path = file_path[:file_path.rfind('pySPACE')-1]
        if not pyspace_path in sys.path:
            sys.path.append(pyspace_path)
        # Get the file name of the process to unpickle and call
        proc_file_name = sys.argv[1]
        # Unpickle the process
        proc_file = open(proc_file_name, 'rb')
        proc = cPickle.load(proc_file)
        proc_file.close()
        # Do the actual call
        proc()
        # Deleted the proc_file since we don't need it any more
        os.system("rm "+proc_file_name)
        # Inform the listener socket in the Backend that this job has finished 
        send_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        send_sock.connect((sys.argv[2],int(sys.argv[3])))
        send_sock.sendall("finished!END!")
        send_sock.shutdown(socket.SHUT_RDWR)
        send_sock.close()

if __name__ == '__main__':
    main()
