""" Executable script for Loadleveler to start subflows

A simple helper script that unpickles, executes and deletes given
:class:`flows<pySPACE.environments.chains.node_chain.BenchmarkNodeChain>`. After
each execution the
:class:`~pySPACE.environments.backends.ll_backend.LoadLevelerComHandler` is informed.

Call should be 'python ll_subflow_runner.py dir/to/pickled/flows
Com_Handler_IP Com_Handler_Port list_of_run_numbers list_of_flow_ids'.

It is further assumed that training data used by the flows is stored as
'subflow_data.pickle' in the same directory as the flows.

:Author: Anett Seeland (anett.seeland@dfki.de)
:Created: 2012/07/29
:LastChange: 2012/11/05 Added possibility to execute several flows
"""
import socket
import os
import sys
import time
try:
    # noinspection PyPep8Naming
    import cPickle as pickle
except ImportError:
    import pickle


if __name__ == '__main__':
    # add root of the code to system path
    file_path = os.path.dirname(os.path.abspath(__file__))
    pyspace_path = file_path[:file_path.rfind('pySPACE') - 1]
    if pyspace_path not in sys.path:
        sys.path.append(pyspace_path)


from pySPACE.environments.backends.ll_backend import LoadLevelerComHandler


def main():
    # Parse input
    dir_path = sys.argv[1]
    train_instances_file_name = os.path.join(dir_path,
                                             'subflow_data.pickle')
    runs = eval(sys.argv[4])
    # construct socket to communicate with backend
    send_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    send_sock.connect((sys.argv[2], int(sys.argv[3])))
    try:
        flow_ids = eval(sys.argv[5])
        for flow_id in flow_ids:
            flow_file_name = os.path.join(dir_path, flow_id + '.pickle')
            # Unpickle the flow and train_instances
            with open(flow_file_name, 'rb') as flow_file:
                flow = pickle.load(flow_file)
            with open(train_instances_file_name, 'rb') as train_instances_file:
                train_instances = pickle.load(train_instances_file)
            try:
                # Execute the flow
                # .. note:: the here executed flows can not store anything
                #           meta data of result collection is NOT updated!
                _, result_collection = flow(train_instances=train_instances, runs=runs)
                # Store results
                result_file_name = flow_file_name.replace('.pickle', '_result.pickle')
                with open(result_file_name, 'wb') as result_file:
                    pickle.dump(result_collection, result_file, protocol=pickle.HIGHEST_PROTOCOL)
            finally:
                # Delete the flow_file since we don't need it any more
                # training_file is may be used by other flows so don't delete it
                os.remove(flow_file_name)
                # Inform Backend that this flow has finished
                send_sock = LoadLevelerComHandler(send_sock, sys.argv[2], int(sys.argv[3]),
                                                  LoadLevelerComHandler.MESSAGES.FINISHED, flow_id)
        # give backend some time to get information
        time.sleep(1)
    finally:
        send_sock.shutdown(socket.SHUT_RDWR)
        send_sock.close()

if __name__ == "__main__":
    main()
