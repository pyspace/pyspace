""" Interface for the numerous operations and processes

An **operation** takes one summary of data as input and produces a
second one as output with a set of *processes*.

Each **process** can be considered as
an atomic task that can be executed independently from all other processes
of the operation.

Standard Specification File Parameters
++++++++++++++++++++++++++++++++++++++

type
----

Specifies which operation is used.
Type and corresponding module name are the same in lower case (with underscore).
The respective operation name is written in camel case with the ending
``Operation``.

(*obligatory*)

input_path
------------
Path name relative to the *storage* in your configuration file.

The input has to fulfill certain rules,
specified in the fitting: :mod:`pySPACE.resources.dataset_defs` subpackage.

Depending on the operation, the input datasets are combined, or each parameter
combination is applied separately to the data.

(*mostly obligatory*)

storage_format
--------------

Some datasets give the opportunity to choose between different
formats for storing the result. The choice can be specified here.
For details look at the :mod:`~pySPACE.resources.dataset_defs` documentation.
If the format is not specified, the default of the
dataset is used.

(*optional, default: default of dataset*)

hide_parameters
---------------

Normally each parameter is added in the format
*{__PARAMETER__: value}* added to the *__RESULT_DIRECTORY__*.
Every parameter specified in the list *hide_parameters* is not
added to the folder name. Be careful not to have
the same name for different results.

(*optional, default: []*)

runs
----

Number of repetitions for each running process.
The run number is given to the process, to use it for *choosing*
random components, especially when using cross-validation.

(*optional, default: 1*)

backend
-------

Overwrite the backend defined in the command line
or, if not specified, the default (serial) backend

(*optional, default: command line or default backend*)

others
------

For the operation specific parameters have a look at operation
documentation.
"""
import glob

import os
import time
import logging
import socket
import sys
import warnings
import yaml
import copy

file_path = os.path.dirname(os.path.abspath(__file__))
pyspace_path = file_path[:file_path.rfind('pySPACE')-1]
if not pyspace_path in sys.path:
    sys.path.append(pyspace_path)

import pySPACE
from pySPACE.tools.filesystem import create_directory


class Operation(object):
    """ Take one summary of data as input and produces a second one as output with a set of *processes*.,
    that can be executed on an arbitrary backend modality
    """
    def __init__(self, processes, operation_spec, result_directory):
        self.processes = processes
        self.operation_spec = operation_spec
        self.result_directory = result_directory
        
        # Check if the required directories exist 
        # and create them if necessary
        create_directory(self.result_directory)
        
        # Store the specification of this operation in the directory
        # without the base_file entry
        base_file = self.operation_spec.pop("base_file", None)
        source_operation_file = open(os.sep.join([self.result_directory,
                                                  "source_operation.yaml"]), 'w')
        yaml.dump(self.operation_spec, source_operation_file)
        source_operation_file.close()
        if not base_file is None:
            self.operation_spec["base_file"] = base_file
    
    def consolidate(self, results):
        """
        Consolidates the results obtained by the single processes of this 
        operation into a consistent structure of collections.
        """
        raise NotImplementedError(
            "Method consolidate has not been implemented in subclass %s"
            % self.__class__.__name__)
    
    def _log(self, message, level=logging.INFO):
        """ Logs  the given message  with the given logging level """
        logging.getLogger("%s" % self).log(level, message)

    
    @classmethod
    def create(cls, operation_spec, base_result_dir=None):
        """
        A factory method that calls the responsible method
        for creating an operation of the type specified in
        the operation specification dictionary (*operation_spec*).
        """
        # Determine result directory
        result_directory = cls.get_unique_result_dir(base_result_dir)
        print("--> Results will be stored at: \n\t\t %s"%str(result_directory))
        # Check if the required directories exist 
        # and create them if necessary
        create_directory(result_directory)

        # Determine all input datasets (note: they can be specified by
        # extended syntax for the glob package)
        storage = pySPACE.configuration.storage
        if not operation_spec.has_key("input_path"):
            warnings.warn("No input path found in operation specification.")
        input_path_pattern = os.sep.join([storage,
                                          operation_spec.get("input_path", ""),
                                          "*", ""])
        input_paths = glob.glob(input_path_pattern)
        obsolete_paths=[]
        for path in input_paths:
            file_path = os.sep.join([path,"metadata.yaml"])
            if os.path.isfile(os.sep.join([path,"metadata.yaml"])):
                continue
            elif os.path.isfile(os.sep.join([path,"collection.yaml"])):
                continue # warning comes, when data is loaded
            else:
                obsolete_paths.append(path)
                warnings.warn('Folder' + str(path) + ' seems not to be a pySPACE'+
                              ' dataset (no "metadata.yaml" found)! '+
                              'Skipping this folder in operation...')
        for path in obsolete_paths:
            input_paths.remove(path)

        op_type = operation_spec["type"]
        if op_type.endswith("_operation"):
            l=len("_operation")*-1
            op_type=op_type[:l]
            operation_spec["type"] = op_type
            warnings.warn("'%s_operation' has the wrong ending. Using '%s' instead."%(op_type,op_type),DeprecationWarning)
        op_class_name = ''.join([x.title() for x in op_type.split('_')])
        op_class_name += "Operation"
        # dynamic class import: from data_mod_name import col_class_name
        try:
            op_module = __import__('pySPACE.missions.operations.%s' % op_type,
                                        fromlist=[op_class_name])
        except:
            msg = "Operation module %s is unknown. Trying to use node_chain." % (op_type)
            from pySPACE.missions.operations.node_chain import NodeChainOperation
            op_class = NodeChainOperation
        else:
            op_class = getattr(op_module,op_class_name)
        return op_class.create(operation_spec, result_directory,
                               input_paths=input_paths)

    @classmethod
    def get_unique_result_dir(cls, base_result_dir):
        """ Creates a new result directory with current time stamp """
        # Determine the directory in which the operation's result
        # are stored
        time_stamp = time.strftime("%Y%m%d_%H_%M_%S")
        result_directory = os.path.join(base_result_dir,
                                        time_stamp)
    
        return result_directory
    
    @classmethod
    def _get_parameter_space(cls, operation_spec):
        """ Determines all parameter combinations that should be tested """
        # Crossproduct
        crossproduct = lambda ss,row=[],level=0: len(ss)>1 \
            and reduce(lambda x,y:x+y,[crossproduct(ss[1:],row+[i],level+1)
                                         for i in ss[0]]) \
            or [row+[i] for i in ss[0]]
       
        # Compute all possible parameter combinations 
        # for the node chain template instantiation
        if "parameter_ranges" in operation_spec:
            parametrization_def = operation_spec["parameter_ranges"]
            parameter_ranges = [eval(range_expression[5:-1])
                if isinstance(range_expression, basestring)
                    and range_expression.startswith("eval(")
                        else range_expression for range_expression in parametrization_def.values()]
            parameter_settings = map(lambda x: dict(zip(parametrization_def.keys(), x)),
                                    crossproduct(parameter_ranges))
        elif "parameter_settings" in operation_spec: # Just use specified parameters
            parameter_settings = operation_spec["parameter_settings"]
        else:
            parameter_settings = [dict()]

        # Filter the parameter combinations if constraints have been set
        def check_constraint(constraint, parameters):
            for key, value in parameters.iteritems():
                # TODO Fix string mapping for floats
                constraint = constraint.replace(key, str(value))
            return eval(constraint)
        
        if "constraints" in operation_spec:
            for constraint_def in operation_spec["constraints"]:
                # for every elem (*x*) in list *parameter_settings* call 
                # lambda-function, if this returns true, elem will stay in the 
                # list, otherwise it is filtered out
                parameter_settings = filter(lambda x: check_constraint(constraint_def, x),
                                            parameter_settings)
        return parameter_settings            

    def get_output_directory(self):
        """ Returns the directory where the output is stored """
        return self.result_directory
      
    def __repr__(self):
        """ Return a representation of this class"""
        return self.__class__.__name__        
    
class Process(object):
    """
    A process is an atomic task that can be executed on an arbitrary
    backend modality by executing the __call__ method.
    """
    def __init__(self):
        pass

    def prepare(self, configuration, handler_class=None, handler_args=None,
                backend_com=None):
        """
        This method is called by the respective backend in order to prepare
        the process for remote execution 
        """
        self.configuration = copy.deepcopy(configuration)
        self.handler_class = handler_class
        self.handler_args = handler_args
        self.handler = None
        self.backend_com = backend_com

    def __repr__(self):
        """ Return a representation of this class"""
        return self.__class__.__name__

    def pre_benchmarking(self):
        """
        Execute some code which is not specific for the respective operation
        but needs to be executed before benchmarking remotely (e.g. set up logging,
        setting environment variables etc.)
        """
        # Prepare remote logging
        root_logger = logging.getLogger("%s-%s" % (socket.gethostname(),
                                                   os.getpid()))
        root_logger.setLevel(logging.DEBUG)
        root_logger.propagate = False

        if self.handler_class != None:
            self.handler = self.handler_class(**self.handler_args)
            root_logger.addHandler(self.handler)

        new_python_path = self.configuration.python_path
        python_path_set = set(new_python_path)
        import sys
        unique_sys_path = [i for i in sys.path if i not in python_path_set and not python_path_set.add(i)]
        new_python_path.extend(unique_sys_path)
        sys.path = new_python_path

    def post_benchmarking(self):
        """
        Execute some code which is not specific for the respective operation
        but needs to be executed after benchmarking remotely to clean up
        """
        # Remove potential logging handlers
        if self.handler is not None:
            root_logger = logging.getLogger("%s-%s" % (socket.gethostname(),
                                            os.getpid()))
            self.handler.close()
            root_logger.removeHandler(self.handler)

    def _log(self, message, level = logging.INFO):
        """ Log the given message into the logger of this class """
        logging.getLogger("%s-%s.%s" % (socket.gethostname(),
                                        os.getpid(),
                                        self)).log(level, message)


def create_operation_from_file(operation_filename, base_result_dir = None):
    """ Creates the operation for the file *operation_filename*

    Creates the operation for a given operation filename.  If *operation_filename*
    is an absolute path, it is expected to be the path  to the
    YAML specification file of an operation. Alternatively, if *operation_filename*
    is not an absolute path, the name is used to look up a YAML specification
    file for the operation in the spec dir.

    If *base_result_dir* is not None, the results
    of the operation are written into the specified directory.
    """
    # Load operation from specs directory when not an absolute path
    if not os.path.isabs(operation_filename):
        spec_file_name = os.path.join(pySPACE.configuration.spec_dir,
                                      "operations", operation_filename)
    else:
        spec_file_name = operation_filename

    print("--> Loading operation: \n\t\t %s."%spec_file_name)
    try:
        with open(spec_file_name, "r") as op_file:
            operation_spec = yaml.load(op_file)
        with open(spec_file_name, "r") as op_file:
            operation_spec["base_file"] = op_file.read()
    except IOError,e:
        if not spec_file_name.endswith(".yaml"):
            warnings.warn(
                "Operation specification not found. Trying with yaml ending.")
            spec_file_name += ".yaml"
            with open(spec_file_name, "r") as op_file:
                operation_spec = yaml.load(op_file)
            with open(spec_file_name, "r") as op_file:
                operation_spec["base_file"] = op_file.read()
        else:
            raise e
    storage = pySPACE.configuration.storage
    if operation_spec.has_key("input_path"):
        input_path = os.sep.join([storage, operation_spec["input_path"], ""])
        print("--> Input data is taken from: \n\t\t %s"%input_path)
    return create_operation(operation_spec, base_result_dir)


def create_operation(operation_spec, base_result_dir = None):
    """ Creates the operation for the given operation spec

    Simple wrapper to the *create* function of the operation, getting
    the operation result dir from the configuration file.
    """
    if base_result_dir is None:
        base_result_dir =  os.path.join(pySPACE.configuration.storage,
                                        "operation_results")
    # use the operation factory method to create operation
    operation = Operation.create(operation_spec,
                                 base_result_dir = base_result_dir)

    return operation