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
import logging
import socket
import os
import sys
from pySPACE.environments.backends.ll_backend import LoadLevelerComHandler
try:
    # noinspection PyPep8Naming
    import cPickle as pickle
except ImportError:
    # C-Extension is not available.. import the python pickler
    import pickle


def main():
    # Extract the trunk directory and append to PYTHONPATH cause we need
    # the specific operation process class
    # backends, environments, pySPACE are the parts to go back
    file_path = os.path.dirname(os.path.abspath(__file__))
    pyspace_path = file_path[:file_path.rfind('pySPACE')-1]
    if pyspace_path not in sys.path:
        sys.path.append(pyspace_path)

    # Get the file name of the process to unpickle and call
    proc_file_name = sys.argv[1]

    # Unpickle the process
    with open(proc_file_name, 'rb') as proc_file:
            proc = pickle.load(proc_file)

    # noinspection PyBroadException
    try:
        # Do the actual call
        proc()
    except Exception:
        logging.exception("Error while executing the process:")
    finally:
        # Deleted the proc_file since we don't need it any more
        try:
            os.unlink(proc_file_name)
        except IOError:
            logging.exception("Error while unlinking the process file:")

        # Inform the listener socket in the Backend that this job has finished
        send_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        host = sys.argv[2]
        port = int(sys.argv[3])
        send_sock.connect((host, port))
        try:
            LoadLevelerComHandler.send_message(send_sock, host, port, LoadLevelerComHandler.MESSAGES.FINISHED)
        finally:
            send_sock.shutdown(socket.SHUT_RDWR)
            send_sock.close()


if __name__ == '__main__':
    main()
