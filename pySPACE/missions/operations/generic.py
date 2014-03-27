# 2012-11-30, Jan Hendrik Metzen
""" Use a generic external python script in a process

Specification file Parameters
+++++++++++++++++++++++++++++

type
----

Specifies which operation is used. For this operation
you have to use *generic*.

(*obligatory, generic*)

code
----
The code to be executed by this operation. The code MUST contain
the definition of the functions "process" and "consolidate". A
typical pattern is to have the code in a separate file and use
"exec open("file_containing_code.py").read()" for execution.

(*obligatory*)

configuration_template
----------------------

Template which determines the parameters that are passed to the
"process" function via the *config* parameter. The template is
instantiated based on the given parameter_ranges (see below).
For each instantiation, one process is created.

(*mandatory if parameter_setting is not used*)

parameter_ranges
----------------

Parameter range which determines the specific instantiations of the
configuration template. The template is instantiated once for each
element of the cartesian product of the given ranges (if no
constraints are defined)

(*mandatory if parameter_setting is not used*)

parameter_setting
-----------------

If you do not use the *parameter_ranges*, this is a list of
dictionaries giving the parameter combinations, you want to test
the code on.

(*mandatory if parameter_ranges is not used*)

constraints
-----------

List of strings, where the parameters values are replaced and
which is afterwards evaluated to check if is *True*.
If it is not, the parameter is rejected.

(*optional, default: []*)

Exemplary Call
++++++++++++++

.. code-block:: yaml

    type: generic

    # The code to be executed by this operation
    # The code MUST contain the definition of the functions "process" and "consolidate"
    # NOTE: We could just directly put the code into this configuration file (instead of loading it from
    #       an other file). However, if the functions are not trivial, it's typically more convenient
    #       to have the code in a separate file
    # NOTE: The | is actually required. Don't remove it!
    code: |
        exec open('~/pyspace/pySPACE/run/scripts/mandelbrot_set.py').read()

    # The template which determines the parameters that are passed to the "process" function
    # via the *config* parameter. The template is instantiated based on the given
    # parameter_ranges (see below). For each instantiation, one process is created.
    # NOTE: The | is actually required. Don't remove it!
    configuration_template: |
        xind : __XIND__
        yind : __YIND__
        xstep : 0.4
        ystep : 0.4
        resolution: 1000

    # Parameter range which determines the specific instantiations of the configuration template.
    # The template is instantiated once for each element of the cartesian product of the given
    # ranges (if no constraints are defined)
    parameter_ranges:
        __XIND__ : [0, 1, 2, 3, 4, 5, 6]
        __YIND__ : [0, 1, 2, 3, 4, 5, 6]

"""

import os
import sys
import imp
import yaml
import logging
import traceback
# processing was renamed in Python 2.6 to multiprocessing
if sys.version_info[0] == 2 and sys.version_info[1] < 6:
    import processing
else:
    import multiprocessing as processing

from pySPACE.missions.operations.base import Operation, Process


class GenericOperation(Operation):
    """ Generic operation performing computation based on external module

    A GenericOperation consists of a set of GenericProcess instances. Each of
    these processes consists of calling a function "process" defined in an
    external module for a specific parametrization.

    The results of this operation are collected using the
    consolidate method that produces a consistent representation
    of the result. Especially it collects not only the data,
    but also saves information of the in and out coming files and the used
    specification files.

    Objects of this class are typically create using the *create* factory
    method based on a operation_spec dictionary. This dictionary in turn is
    typically create based on a configuration file in YAML syntax. Which
    parameters need to be specified in this configuration file are listed below.
    See example_generic_operation.yaml for an example.
    """
    def __init__(self, processes, operation_spec, result_directory,
                 code_string, number_processes):
        super(GenericOperation, self).__init__(processes, operation_spec,
                                               result_directory)

        self.code_string = code_string
        self.number_processes = number_processes

        # we add this since some backends assume that this attribute exists
        self.create_process = None

    @classmethod
    def create(cls, operation_spec, result_directory, debug=False, input_paths=[]):
        """ Factory method for creating a :class:`GenericOperation`.

        Factory method for creating a GenericOperation based on the
        information given in the operation specification *operation_spec*.
        """
        assert(operation_spec["type"] == "generic")

        configuration_template = operation_spec["configuration_template"]

        # Compute all possible parameter combinations
        parameter_settings = cls._get_parameter_space(operation_spec)
        processes = processing.Queue()
        for process_id, parameter_setting in enumerate(parameter_settings):
            process = GenericProcess(process_id=process_id,
                                     configuration_template=configuration_template,
                                     parameter_setting=parameter_setting,
                                     result_directory=result_directory,
                                     code_string=operation_spec["code"])

            processes.put(process)

        # Indicate that we are at then end of the process queue and no more jobs
        # will be added
        processes.put(False)

        # create and return the operation object
        return cls(processes, operation_spec, result_directory,
                    operation_spec["code"], len(parameter_settings))

    def consolidate(self):
        """Perform cleanup / consolidation at the end of operation. """
        # Write the specification of this operation to the result directory in
        # order to make later analysis of results more easy
        self._log("Storing YAML configuration file ...")
        source_operation_file = open(os.path.join(self.result_directory,
                                                  "source_operation.yaml"), 'w')
        yaml.dump(self.operation_spec, source_operation_file)
        source_operation_file.close()

        # First execute module to get function definitions
        # Create pseudo module
        tempmodule = imp.new_module('tempmodule')
        exec self.code_string in tempmodule.__dict__

        # Call consolidate function
        tempmodule.consolidate(self.result_directory,
                      yaml.load(self.operation_spec["configuration_template"]))


class GenericProcess(Process):
    """ Generic process performing computation specified in external module.

    This process calls the function "process" defined

    **Parameters**
        :process_id:
            Globally unique id of this process. Might be used e.g. for
            creating a file into which the results of this process are stored.

            (*obligatory*)

        :configuration_template:
            Template (string) which determines the parameters that are passed
            to the "process" function contained in the code_string via the
            *config*  parameter. The template is instantiated based on the given
            parameter_setting (see below). It must adhere the YAML standard and
            correspond to a dictionary mapping parameter name to value.

            (*obligatory*)

        :parameter_setting:
            Dictionary containing the mapping from parameter name (must be
            contained in configuration_template string) to parameter value.
            The specific parameter values define this particular computation.

            (*obligatory*)

        :result_directory:
            Directory into which the results of the computation of this
            function are stored.

            (*obligatory*)

        :code_string:
            A string containing a definition of a python function with the name
            "process". Note: We pass the string-definition of the function
            instead of the function itself since GenericProcess objects are
            often serializes using pickle. This wouldn't work when the object
            would contain a reference to a function object.

            (*obligatory*)

    """

    def __init__(self, process_id, configuration_template, parameter_setting,
                 result_directory, code_string):

        super(GenericProcess, self).__init__()

        self.process_id = process_id
        self.configuration_template = configuration_template
        self.parameter_setting = parameter_setting
        self.result_directory = result_directory
        self.code_string = code_string

        # Instantiate the configuration_template for the specific
        # parametrization defined in parameter_setting.
        for key, value in parameter_setting.iteritems():
            # Parameters framed by "#" are considered to be escaped
            if "#"+key+"#" in configuration_template:
                configuration_template = \
                    configuration_template.replace("#"+key+"#", "##")

            # Apply replacement of key-value pairs
            configuration_template = \
                configuration_template.replace(str(key), '%s' % str(value))

            # Reinsert escaped parameters
            if "##" in configuration_template:
                configuration_template = \
                    configuration_template.replace("##", key)

        # Evaluate instantiated configuration template string
        self.config = yaml.load(configuration_template)

    def __call__(self):
        """ Executes this process on the respective modality. """
        ############## Prepare benchmarking ##############
        self.pre_benchmarking()

        ##### Execute computation specified in external python module #########

        tempmodule = imp.new_module('tempmodule')
        try:
            # Create pseudo module
            exec self.code_string in tempmodule.__dict__

            # Perform actual computation
            tempmodule.process(self.process_id, self.result_directory,
                               self.config, self.parameter_setting)
        except:
            # Send exception to Logger
            self._log(traceback.format_exc(), level=logging.ERROR)
            raise

        ############## Clean up after benchmarking ##############
        self.post_benchmarking()
