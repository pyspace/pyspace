""" Concatenation of operations

The basic principle is, that a new main folder is created
for the operation chain
and every operation gets a subfolder for saving results.

When running the operation chain, the operations were told where to get and
where to put their results.

The user only has to specify the list of operations as described
in the tutorials and example files and run the software with the
chain option ``-c``.
"""
import os
import time
import sys
import yaml

file_path = os.path.dirname(os.path.abspath(__file__))
pyspace_path = file_path[:file_path.rfind('pySPACE')-1]
if not pyspace_path in sys.path:
    sys.path.append(pyspace_path)


import pySPACE

class OperationChain(dict):
    """ Create the operation chain object

    .. todo:: This class should not subclass dict.
    """
    def __init__(self, input_dict, base_result_dir):
        self.update(input_dict)
        self.base_result_dir = base_result_dir

    def get_output_directory(self):
        return self.base_result_dir

def create_operation_chain(operation_chain_name):
    """ Creates the operation chain for the name *operation_chain_name*

    Creates the operation for a given operation name. The
    name is used to look up a YAML specification file for
    the operation in the spec dir.
    """
    spec_file_name = os.path.join(pySPACE.configuration.spec_dir,
                                  "operation_chains", operation_chain_name)
    operation_chain_spec = yaml.load(open(spec_file_name, "r"))

    timeString = time.strftime("%Y%m%d_%H_%M_%S")
    base_result_dir =  os.path.join(pySPACE.configuration.storage,
                                    "operation_chain_results", timeString)
    os.makedirs(base_result_dir)

    return OperationChain(operation_chain_spec, base_result_dir)