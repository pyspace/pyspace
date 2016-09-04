""" Interface to :mod:`~pySPACE.environments.chains.node_chain` using the :class:`~pySPACE.environments.chains.node_chain.BenchmarkNodeChain`

.. image:: ../../graphics/node_chain.png
   :width: 500

A process consists of applying
the given `node_chain` on an input for a certain parameters setting.

Factory methods are provided that create this operation based
on  a specification in one or more YAML files.

.. note:: Complete lists of available nodes and namings can be found
          at: :ref:`node_list`.

.. seealso::

    - :mod:`~pySPACE.missions.nodes`
    - :ref:`node_list`
    - :mod:`~pySPACE.environments.chains.node_chain`

Specification file Parameters
+++++++++++++++++++++++++++++

type
----

Specifies which operation is used. For this operation
you have to use *node_chain*.

(*obligatory, node_chain*)

input_path
----------

Path name relative to the *storage* in your configuration file.

The input has to fulfill certain rules,
specified in the fitting: :mod:`~pySPACE.resources.dataset_defs` subpackage.
For each dataset in the input and each parameter combination
you get an own result.

(*obligatory*)

templates
---------

List of node chain file templates which shall be evaluated.
The final node chain is constructed, by substituting specified parameters
of the operation spec file. Therefore you should use the format
*__PARAMETER__*. Already fixed and usable keywords are
*__INPUT_DATASET__* and *__RESULT_DIRECTORY__*.
The spec file name may include a path and it is searched relative
the folder *node_chains* in the specs folder,
given in your configuration file you used, when starting :ref:`pySPACE`
(:mod:`pySPACE.run.launch`).

If no template is given, the software tries to use the
parameters *node_chain*, which simply includes the *node_chain* description.

.. todo:: Check that parameter fulfills format rule!

(*recommended, alternative: `node_chain`*)

node_chain
----------

Instead of using a template you can use your single node chain directly
in the operation spec.

(*optional, default: `templates`*)

runs
----

Number of repetitions for each running process.
The run number is given to the process, to use it for *choosing*
random components, especially when using cross-validation.

(*optional, default: 1*)

parameter_ranges
----------------

Dictionary with parameter names as keys and a list of values it
should be replaced with.
Finally a grid of all possible combinations is build and afterwards
only those remain fulfilling the constraints.

(*mandatory if parameter_setting is not used*)

parameter_setting
-----------------

If you do not use the *parameter_ranges*, this is a list of
dictionaries giving the parameter combinations, you want to test
the operation on.

(*mandatory if parameter_ranges is not used*)

constraints
-----------

List of strings, where the place holders for the parameter values are replaced
as in the node chain template.
After creating all possible combinations of parameters, they are inserted
into this string and the string is evaluated like a Python expression.
If it is not evaluated to *True*, the parameter is rejected.
The test is performed for each constraint (string). For being used later on,
a parameter setting has to pass all tests. Example:

.. code-block:: yaml

    __alg__ in ['A','B','C'] and __par__ == 0.1 or __alg__ == 'D'

(*optional, default: []*)

old_parameter_constraints
-------------------------
Same as *constraints*, but here parameters of earlier
operation calls, e.g. windowing, can be used in the constraint def.

hide_parameters
---------------

Normally each parameter is added in the format
*{__PARAMETER__: value}* added to the *__RESULT_DIRECTORY__*.
Every parameter specified in the list *hide_parameters* is not
added to the folder name. Be careful not to have
the same name for different results.

(*optional, default: []*)

storage_format
--------------

Some datasets give the opportunity to choose between different
formats for storing the result. The choice can be specified here.
For details look at the :mod:`~pySPACE.resources.dataset_defs` documentation.
If the format is not specified, the default of the
dataset is used.

(*optional, default: default of dataset*)

store_node_chain
----------------

If True, the total state of the node chain after the processing is stored to disk.
This is done separately for each split and run.
If *store_node_chain* is a tuple of length 2---lets say (i1,i2)---only the 
subflow starting at the i1-th node and ending at the (i2-1)-th node is stored. 
This may be useful when the stored flow should be used in an ensemble.
If *store_node_chain* is a list with indices, all nodes in the chain with the
corresponding indices are stored. This might be useful to exclude specific nodes
that do not change the data dimension / type, like e.g. visualization nodes.

.. note:: Since yaml cannot recognize tuples, string parameters are evaluated
          before type testing (if boolean, tuple, list) is done.

(*optional, default: False*)

compression
-----------

If your result is a classification summary,
all the created sub-folders are zipped with the zipfile module,
since they are normally not needed anymore,
but you may have numerous folders, making coping difficult.
To switch this of, use the value *False* and if you want no
compression, use *1*. If all the sub-folders (except a single one)
should be deleted, set compression to ``delete``.

(*optional, default: 8*)


Exemplary Call
++++++++++++++

.. code-block:: yaml

    type: node_chain
    input_path: "my_data"
    runs : 2
    templates : ["example_node_chain1.yaml","example_node_chain2.yaml","example_node_chain3.yaml"]

.. code-block:: yaml

    type: node_chain
    input_path: "my_data"
    runs : 2
    parameter_ranges :
        __prob__ : eval(range(1,4))
    node_chain :
        -
            node: FeatureVectorSource
        -
            node: CrossValidationSplitter
            parameters:
                splits : 5
        -
            node: GaussianFeatureNormalization
        -
            node: LinearDiscriminantAnalysisClassifier
            parameters:
                class_labels: ["Standard","Target"]
                prior_probability : [1,__prob__]
        -
            node: ThresholdOptimization
            parameters:
                metric : Balanced_accuracy

        -
            node: Classification_Performance_Sink
            parameters:
                ir_class: "Target"

.. code-block:: yaml

    type: node_chain

    input_path: "example_data"
    templates : ["example_flow.yaml"]
    backend: "local"
    parameter_ranges :
        __LOWER_CUTOFF__ : [0.1, 1.0, 2.0]
        __UPPER_CUTOFF__ : [2.0, 4.0]
    constraints:
        - "__LOWER_CUTOFF__ < __UPPER_CUTOFF__"


    runs : 10

.. code-block:: yaml

    type: node_chain
    input_path: "my_data"
    runs : 2
    templates : ["example_node_chain.yaml"]
    parameter_setting :
        -
            __prob__ : 1
        -
            __prob__ : 2
        -
            __prob__ : 3

Here `example_node_chain.yaml` is defined in the node chain specification folder as

.. literalinclude:: ../../examples/specs/node_chains/example_node_chain.yaml
    :language: yaml

"""

import os
import sys
import time
import yaml
import shutil
import glob
import logging
import warnings
import copy
import random
import gc
import hashlib

# processing was renamed in Python 2.6 to multiprocessing
if sys.version_info[0] == 2 and sys.version_info[1] < 6:
    import processing
else:
    import multiprocessing as processing

# import of all necessary pySPACE packages
import pySPACE
from pySPACE.missions.operations.base import Operation, Process
from pySPACE.resources.dataset_defs.base import BaseDataset
from pySPACE.tools.filesystem import create_directory
from pySPACE.tools.conversion import extract_key_str, replace_parameters_and_convert
from pySPACE.environments.chains.node_chain \
    import BenchmarkNodeChain, NodeChainFactory

from pySPACE.resources.dataset_defs.performance_result \
    import PerformanceResultSummary
from pySPACE.tools.filesystem import get_author


class NodeChainOperation(Operation):
    """ Load configuration file, create processes and consolidate results

    The operation consists of a set of processes. Each of
    these processes consists of applying a given
    :class:`~pySPACE.environments.chains.node_chain.BenchmarkNodeChain` on
    an input  for a certain configuration of parameters.

    The results of this operation are collected using the
    consolidate method that produces a consistent representation
    of the result. Especially it collects not only the data,
    but also saves information of the in and out coming files and the used
    specification files.
    """
    def __init__(self, processes, operation_spec, result_directory,
                 number_processes, create_process=None):
        # temporary replacement of template keyword for better layout when
        # source_operation.yaml file is saved to result folder
        templates = operation_spec.pop("templates")
        if type(templates[0]) is basestring:
            operation_spec["templates"] = [yaml.load(x) for x in templates]
        else:
            operation_spec["templates"] = templates
        super(NodeChainOperation, self).__init__(processes, operation_spec,
                                                 result_directory)
        self.operation_spec = templates
        self.create_process = create_process
        self.number_processes = number_processes

        if operation_spec.has_key("compression"):
            self.compression = operation_spec["compression"]
        else:
            self.compression = 8

    @classmethod
    def create(cls, operation_spec, result_directory, debug=False, input_paths=[]):
        """ A factory method that creates the processes which form an operation
        based on the  information given in the operation specification, *operation_spec*.

        In debug mode this is done in serial. In the other default mode,
        at the moment 4 processes are created in parallel and can be immediately
        executed. So generation of processes and execution are made in parallel.
        This kind of process creation is done independently from the backend.

        For huge parameter spaces this is necessary!
        Otherwise numerous processes are created and corresponding data is loaded
        but the concept of spreading the computation to different processors
        can not really be used, because process creation is blocking only
        one processor and memory space, but nothing more is done,
        till the processes are all created.

        .. todo:: Use :class:`~pySPACE.resources.dataset_defs.dummy.DummyDataset`
                  for empty data, when no input_path is given.
        """
        assert(operation_spec["type"] == "node_chain")

        # Determine all parameter combinations that should be tested
        parameter_settings = cls._get_parameter_space(operation_spec)

        ## Use node_chain parameter if no templates are given ##
        if not operation_spec.has_key("templates"):
            if operation_spec.has_key("node_chain"):
                operation_spec["templates"] = [
                    extract_key_str(operation_spec["base_file"],
                                    keyword="node_chain")]
                operation_spec.pop("node_chain")
            else:
                warnings.warn("Specify parameter 'templates' or 'node_chain' in your operation spec!")
        elif operation_spec.has_key("node_chain"):
            operation_spec.pop("node_chain")
            warnings.warn("node_chain parameter is ignored. Templates are used.")
        # load files in templates as dictionaries
        elif type(operation_spec["templates"][0]) == str:
            operation_spec["template_files"] = \
                copy.deepcopy(operation_spec["templates"])
            for i in range(len(operation_spec["templates"])):
                rel_node_chain_file = operation_spec["templates"][i]
                abs_node_chain_file_name = os.sep.join([
                    pySPACE.configuration.spec_dir, "node_chains",
                    rel_node_chain_file])
                with open(abs_node_chain_file_name, "r") as read_file:
                    node_chain = read_file.read()
                    #node_chain = yaml.load(read_file)
                operation_spec["templates"][i] = node_chain

        storage = pySPACE.configuration.storage
        if not input_paths:
            raise Exception("No input datasets found in input_path %s in %s!"
                            % (operation_spec["input_path"],storage))

        # Get relative path
        rel_input_paths = [name[len(storage):]
                           for name in  input_paths]

        # Determine approximate number of runs
        if "runs" in operation_spec:
            runs = operation_spec["runs"]
        else:
            runs = []
            for dataset_dir in rel_input_paths:
                abs_collection_path = \
                        pySPACE.configuration.storage + os.sep \
                            + dataset_dir
                collection_runs = \
                        BaseDataset.load_meta_data(abs_collection_path).get('runs',1)
                runs.append(collection_runs)
            runs = max(runs)

        # Determine splits
        dataset_dir = rel_input_paths[0]
        abs_collection_path = \
                pySPACE.configuration.storage + os.sep + dataset_dir

        splits = BaseDataset.load_meta_data(abs_collection_path).get('splits', 1)

        # Determine how many processes will be created
        number_processes = len(operation_spec["templates"]) * \
                           len(parameter_settings) * len(rel_input_paths) * \
                           runs * splits

        if debug:
            # To better debug creation of processes we don't limit the queue
            # and create all processes before executing them
            processes = processing.Queue()
            cls._createProcesses(processes, result_directory, operation_spec,
                                 parameter_settings, rel_input_paths)
            # create and return the operation object
            return cls(processes, operation_spec, result_directory,
                       number_processes)
        else:
            # Create all processes by calling a recursive helper method in
            # another thread so that already created processes can be executed in
            # parallel. Therefore a queue is used which size is maximized to
            # guarantee that not to much objects are created (because this costs
            # memory). However, the actual number of 4 is arbitrary and might
            # be changed according to the system at hand.
            processes = processing.Queue(4)
            create_process = \
                    processing.Process(target=cls._createProcesses,
                                       args=(processes, result_directory,
                                             operation_spec, parameter_settings,
                                             rel_input_paths))
            create_process.start()
            # create and return the operation object
            return cls(processes, operation_spec, result_directory,
                       number_processes, create_process)

    @classmethod
    def _createProcesses(cls, processes, result_directory, operation_spec,
                         parameter_settings, input_collections):

        storage_format = operation_spec["storage_format"] if "storage_format" \
            in operation_spec else None

        # Determine whether the node_chain should be stored after data processing
        store_node_chain = operation_spec["store_node_chain"] \
                         if "store_node_chain" in operation_spec else False

        # Determine whether certain parameters should not be remembered
        hide_parameters = [] if "hide_parameters" not in operation_spec \
                                else list(operation_spec["hide_parameters"])
        hide_parameters.append("__INPUT_COLLECTION__")
        hide_parameters.append("__INPUT_DATASET__")
        hide_parameters.append("__RESULT_DIRECTORY__")
        hide_parameters.append("__OUTPUT_BUNDLE__")
        operation_spec["hide_parameters"] = hide_parameters

        # Create all combinations of collections, runs and splits
        collection_run_split_combinations = []
        for input_dataset_dir in input_collections:
            # Determine number of runs to be conducted for this collection
            abs_collection_path = \
                pySPACE.configuration.storage + os.sep \
                    + input_dataset_dir
            collection_runs = \
                BaseDataset.load_meta_data(abs_collection_path).get('runs', 1)
                # D.get(k[,d]) -> D[k] if k in D, else d.

            if "runs" not in operation_spec:
                requested_runs = collection_runs
            else:
                requested_runs = operation_spec["runs"]

            assert collection_runs == requested_runs \
                or collection_runs == 1, \
                "Requested %s runs but input collection %s provides "\
                "data for %s runs." % (requested_runs, input_dataset_dir,
                                       collection_runs)

            for run in range(max(requested_runs, collection_runs)):
                collection_splits = BaseDataset.load_meta_data(
                    abs_collection_path).get('splits', 1)
                for split in range(collection_splits):
                    collection_run_split_combinations.append((input_dataset_dir,
                                                              run, split))

        # Shuffle order of dataset-run-split combinations. This should help to
        # avoid that all processes work on the same data which can cause
        # problems due to locking etc.
        random.shuffle(collection_run_split_combinations)

        # For all templates
        for node_chain_spec in operation_spec["templates"]:
            # For all possible parameter instantiations of this template
            for parameter_setting in parameter_settings:
                # For all input collections-run combinations
                for input_dataset_dir, run, split in \
                        collection_run_split_combinations:
                    # We are going to change the parameter_setting and don't
                    # want to interfere with later runs so we work on a copy
                    parameter_setting_cp = copy.deepcopy(parameter_setting)

                    # Add input and output path to parameter
                    # setting
                    parameter_setting_cp["__INPUT_DATASET__"] = \
                            input_dataset_dir.split(os.sep)[-2]
                    parameter_setting_cp["__RESULT_DIRECTORY__"] = \
                            result_directory
                    if len(operation_spec["templates"])>1:
                        index = operation_spec["templates"].index(node_chain_spec)
                        parameter_setting_cp["__Template__"]=\
                            operation_spec["template_files"][index]

                    # Load the input meta data
                    dataset_dir = os.sep.join([pySPACE.configuration.storage,
                                               input_dataset_dir])
                    dataset_md = BaseDataset.load_meta_data(dataset_dir)
                    # Add the input parameter's meta data
                    # to the given parameter setting
                    if "parameter_setting" in dataset_md:
                        dataset_md["parameter_setting"].update(
                            parameter_setting_cp)
                        all_parameters = dataset_md["parameter_setting"]
                    else:
                        all_parameters = parameter_setting_cp

                    def check_constraint(constraint, parameters):
                        for key, value in parameters.iteritems():
                            constraint = constraint.replace(key, str(value))
                        return eval(constraint)

                    if not all(check_constraint(
                            constraint_def, all_parameters) for constraint_def
                            in operation_spec.get('old_parameter_constraints',
                                                  [])):
                        continue

                    # Determine directory in which the result of this
                    # process should be written
                    result_dataset_directory = \
                        NodeChainOperation._get_result_dataset_dir(
                            result_directory,
                            input_dataset_dir,
                            parameter_setting_cp,
                            hide_parameters)
                    
                        
                    # Create the respective process and put it to the
                    # executing-queue of processes
                    process = NodeChainProcess(
                        node_chain_spec=node_chain_spec,
                        parameter_setting=parameter_setting_cp,
                        rel_dataset_dir=input_dataset_dir,
                        run=run, split=split,
                        storage_format=storage_format,
                        result_dataset_directory=result_dataset_directory,
                        store_node_chain=store_node_chain,
                        hide_parameters=hide_parameters)

                    processes.put(process)

        # give executing process the sign that creation is now finished
        processes.put(False)

    def consolidate(self):
        """ Consolidates the results obtained by the single processes into a consistent structure
        of collections that are stored on the file system.
        """
        # Consolidate the results
        directory_pattern = os.sep.join([self.result_directory, "{*",])
        dataset_pathes = glob.glob(directory_pattern)

        # For all collections found
        for dataset_path in dataset_pathes:
            # Load their meta_data
            meta_data = BaseDataset.load_meta_data(dataset_path)

            # Determine author and date
            author = get_author()
            date = time.strftime("%Y%m%d_%H_%M_%S")

            # Update meta data and store it
            meta_data.update({"author": author, "date": date})
            BaseDataset.store_meta_data(dataset_path, meta_data)

            # Copy the input dataset specification file to the result
            # directory in order to make later analysis of
            # the results more easy
            input_meta_path = os.sep.join([pySPACE.configuration.storage,
                                          meta_data["input_collection_name"]])
            input_meta = BaseDataset.load_meta_data(input_meta_path)
            BaseDataset.store_meta_data(dataset_path, input_meta,
                                        file_name="input_metadata.yaml")
        # Check if some results consist of several runs
        # and update the meta data in this case
        # TODO: This is not a clean solution
        for dataset_dir in glob.glob(os.sep.join([self.result_directory,
                                                  "*"])):
            if not os.path.isdir(dataset_dir):
                continue
            # There can be either run dirs, persistency dirs, or both of them.
            # Check of whichever there are more. If both exist, their numbers
            # are supposed to be equal.
            nr_run_dirs = len(glob.glob(os.sep.join([dataset_dir,
                                                     "data_run*"])))
            nr_per_dirs = len(glob.glob(os.sep.join([dataset_dir,
                                                     "persistency_run*"])))
            nr_runs = max(nr_run_dirs, nr_per_dirs)

            if nr_runs > 1:
                collection_meta = BaseDataset.load_meta_data(dataset_dir)
                collection_meta["runs"] = nr_runs
                BaseDataset.store_meta_data(dataset_dir,collection_meta)
        # If we don't create a feature vector or time series collection,
        # we evaluated our classification using a classification performance sink.
        # The resulting files should be merged to one csv tabular.
        pathlist = glob.glob(os.path.join(self.result_directory,"results_*"))
        if len(pathlist)>0:
            # Do the consolidation the same way as for WekaClassificationOperation
            self._log("Consolidating results ...")
            # We load and store the results once into a PerformanceResultSummary
            # This does the necessary consolidation...
            self._log("Reading intermediate results...")
            result_collection = PerformanceResultSummary(dataset_dir=self.result_directory)
            self._log("done")
            self._log("Storing result collection")
            result_collection.store(self.result_directory)
            self._log("done")
            PerformanceResultSummary.merge_traces(self.result_directory)

            if self.compression:
                # Since we get one result summary,
                # we don't need the numerous folders.
                # So we zip them to make the whole folder more easy visible.
                import zipfile
                cwd = os.getcwd()
                os.chdir(self.result_directory)
                # If there are to many or to large folders, problems may occur.
                # This case we want to log, try 64 bit mode,
                # and then skip the zipping.
                try:
                    pathlist = glob.glob(os.path.join(self.result_directory,"{*}"))

                    if not self.compression == "delete":
                        save_file = zipfile.ZipFile(
                            self.result_directory+'/result_folders.zip',
                            mode="w", compression=self.compression)
                        # we want to have the zipped file relative to the
                        # result directory
                        for path in pathlist:
                            for node in os.walk(path):
                                rel_path=os.path.relpath(node[0],
                                                         self.result_directory)
                                save_file.write(rel_path)
                                for data in node[2]:
                                    save_file.write(os.path.join(rel_path,
                                                                 data))
                        save_file.close()
                    # To still have an easy access to the history of the
                    # processing, we keep one folder.
                    pathlist.pop()
                    for path in pathlist:
                        shutil.rmtree(path)
                except Exception, e:
                    self._log("Result files could not be compressed with 32"+
                              " bit mode, switching to 64 bit mode",
                              level=logging.CRITICAL)
                    # nearly total code copy, only difference with 64 bit mode
                    try:
                        pathlist = glob.glob(os.path.join(self.result_directory,"{*}"))
                        save_file=zipfile.ZipFile(
                            self.result_directory+'/result_folders.zip',
                            mode="w", compression=self.compression,
                            allowZip64=True)
                        # we want to have the zipped file relative to the
                        # result directory
                        for path in pathlist:
                            for node in os.walk(path):
                                rel_path = os.path.relpath(node[0],
                                                         self.result_directory)
                                save_file.write(rel_path)
                                for data in node[2]:
                                    save_file.write(os.path.join(rel_path,data))
                        save_file.close()
                        # To still have an easy access to the history of the
                        # processing, we keep one folder.
                        pathlist.pop()
                        for path in pathlist:
                            shutil.rmtree(path)
                    except:
                        self._log("64 bit mode also failed. Please check your files and your code or contact your local programmer!", level=logging.CRITICAL)
                os.chdir(cwd)

    @staticmethod
    def _get_result_dataset_dir(base_dir, input_dataset_dir,
                                parameter_setting, hide_parameters):
        """ Determines the name of the result directory

        Determines the name of the result directory based on the
        input_dataset_dir, the node_chain_name and the parameter setting.
        """
        # Determine the result_directory name
        # String between Key and value changed from ":" to "#",
        # because ot problems in windows and with windows file servers
        def _get_result_dir_name(parameter_setting, hide_parameters, method = None):
            """ internal function to create result dir name in different ways"""
            if not method:
                parameter_str = "}{".join(
                  ("%s#%s" % (key, value)) for key, value in
                  parameter_setting.iteritems() if key not in hide_parameters)
            elif method == "hash":
                parameter_str = "}{".join(
                  ("%s#%s" % (key,hash(str(value).replace(' ', '')))) for key, value in
                  parameter_setting.iteritems() if key not in hide_parameters)

            parameter_str = parameter_str.replace("'", "")
            parameter_str = parameter_str.replace(" ", "")
            parameter_str = parameter_str.replace("[", "")
            parameter_str = parameter_str.replace("]", "")
            parameter_str = parameter_str.replace(os.sep, "")
            result_name = "{%s}" % input_name

            if parameter_str != "":
                result_name += "{%s}" % (parameter_str)

            # Determine the path where this result will be stored
            # and create the directory if necessary
            result_dir = base_dir
            result_dir += os.sep + result_name
            # filename is to long
            # (longer than allowed including optional offsets for pyspace 
            #  result csv naming conventions)
            # create a md5 hash of the result name and use that one
            import platform
            CURRENTOS = platform.system()
            if CURRENTOS == "Windows":
                # the maximum length for a filename on Windows is 255
                if len(result_dir) > 255 - 32:
                    result_name = "{"+hashlib.md5(result_name).hexdigest()+"}"
                    result_dir = base_dir
                    result_dir += os.sep + result_name
                return result_dir
            else:
                if len(result_dir) > os.pathconf(os.curdir, 'PC_NAME_MAX') - 32:
                    result_name = "{"+hashlib.md5(result_name).hexdigest()+"}"
                    result_dir = base_dir
                    result_dir += os.sep + result_name
                return result_dir

        input_name = input_dataset_dir.strip(os.sep).split(os.sep)[-1]
        input_name = input_name.strip("{}")
        # If the input is already the result of an operation
        if input_name.count("}{") > 0:
            input_name_parts = input_name.split("}{")
            input_name = input_name_parts[0]

        # Load the input meta data
        dataset_dir = os.sep.join([pySPACE.configuration.storage,
                                   input_dataset_dir])
        dataset_md = BaseDataset.load_meta_data(dataset_dir)

        # We are going to change the parameter_setting and don't want to
        # interfere with later runs so we work on a copy
        parameter_setting = copy.deepcopy(parameter_setting)

        # Ignore pseudo parameter "__PREPARE_OPERATION__"
        if "__PREPARE_OPERATION__" in parameter_setting:
            parameter_setting.pop("__PREPARE_OPERATION__")

        # Add the input parameters meta data to the given parameter setting
        if "parameter_setting" in dataset_md:
            parameter_setting.update(dataset_md["parameter_setting"])

        # We have to remove ' characters from the parameter value since
        # Weka does ignore them
        for key, value in parameter_setting.iteritems():
            if isinstance(value, basestring) and value.count("'") > 1:
                parameter_setting[key] = eval(value)

        result_dir = _get_result_dir_name(parameter_setting, hide_parameters)
        try:
            create_directory(result_dir)
        except OSError as e:
            if e.errno == 36:
                # filename is too long
                result_dir = _get_result_dir_name(parameter_setting,
                                                  hide_parameters, "hash")
            create_directory(result_dir)

        return result_dir


class NodeChainProcess(Process):
    """ Run a specific signal processing chain with specific parameters and set and store results

    This is an atomic task,which takes
    a dataset (EEG-data, time series data, feature vector data in different formats)
    as input and produces a certain
    output (e.g. ARFF, pickle files, performance tabular).
    The execution of
    this process consists of applying a  given NodeChain
    on an input for a certain configuration.

    **Parameters**
        :node_chain_spec:       String in YAML syntax of the node chain.

        :parameter_setting:     parameters chosen for this process
        :rel_dataset_dir:       location of input file relative to the pySPACE
                                *storage*, specified in the configuration file
        :run:                   When process is started repeatedly, to deal
                                with random components, we have to forward
                                the current run number to these components.
        :split:                 When there are different splits, forward the split
                                number.
        :storage_format:        Choice of format, the resulting data is saved.
                                This should always fit to the sink node
                                specified as the last node in the chain.
        :result_dataset_directory:   Folder where the result is saved at.

        :store_node_chain:    option to save as pickle file the total state of
                        the node chain after the processing;
                        separately for each split in cross validation if
                        existing
    """

    def __init__(self, node_chain_spec, parameter_setting,
                 rel_dataset_dir, run, split, storage_format,
                 result_dataset_directory, store_node_chain=False,
                 hide_parameters=[]):

        super(NodeChainProcess, self).__init__()

        self.node_chain_spec = node_chain_spec
        self.parameter_setting = parameter_setting
        self.rel_dataset_dir = rel_dataset_dir
        self.storage = pySPACE.configuration.storage
        self.run = run
        self.storage_format = storage_format
        self.result_dataset_directory = result_dataset_directory
        self.persistency_dir = os.sep.join([result_dataset_directory,
                                            "persistency_run%s" % run])
        create_directory(self.persistency_dir)
        self.store_node_chain = store_node_chain
        self.hide_parameters = hide_parameters

        # reduce_log_level for process creation
        try:
            console_log_level = eval(pySPACE.configuration.console_log_level) \
                if hasattr(pySPACE.configuration, "console_log_level") \
                else logging.WARNING
        except (AttributeError, NameError):
            console_log_level = logging.WARNING
        try:
            file_log_level = eval(pySPACE.configuration.file_log_level) \
                if hasattr(pySPACE.configuration, "file_log_level") \
                else logging.INFO
        except (AttributeError, NameError):
            file_log_level = logging.INFO

        self.min_log_level = min(console_log_level,file_log_level)
        pySPACE.configuration.min_log_level = self.min_log_level
        # Replace parameters in spec file
        self.node_chain_spec = replace_parameters_and_convert(
            self.node_chain_spec, self.parameter_setting)
        # Create node chain
        self.node_chain = NodeChainFactory.flow_from_yaml(
            Flow_Class=BenchmarkNodeChain, flow_spec=self.node_chain_spec)

        for node in self.node_chain:
            node.current_split = split
        # Remove pseudo parameter "__PREPARE_OPERATION__"
        if "__PREPARE_OPERATION__" in self.parameter_setting:
            self.parameter_setting = copy.deepcopy(self.parameter_setting )
            self.parameter_setting.pop("__PREPARE_OPERATION__")

    def __call__(self):
        """ Executes this process on the respective modality """
        # Restore configuration
        pySPACE.configuration = self.configuration

        # reduce log_level for processing a second time and
        # set communication possibility for nodes to backend
        pySPACE.configuration.min_log_level = self.min_log_level
        pySPACE.configuration.logging_com = self.handler_args
        pySPACE.configuration.backend_com = self.backend_com

        ############## Prepare benchmarking ##############
        super(NodeChainProcess, self).pre_benchmarking()

        # Load the data and check that it can be processed
        # Note: This can not be done in the objects constructor since in
        # that case the whole input would need to be pickled
        # when doing the remote call
        abs_dataset_dir = os.sep.join([self.storage,
                                       self.rel_dataset_dir])

        input_collection = BaseDataset.load(abs_dataset_dir)

        # We have to remember parameters used for generating this specific
        # input dataset
        if 'parameter_setting' in input_collection.meta_data.keys():
            # but not __INPUT_DATASET__ and __RESULT_DIRECTORY__
            for k, v in input_collection.meta_data['parameter_setting'].items():
                if k not in ["__INPUT_DATASET__", "__RESULT_DIRECTORY__"]:
                    self.parameter_setting[k] = v

        NodeChainProcess._check_node_chain_dataset_consistency(self.node_chain,
                                                               input_collection)

        ############## Do the actual benchmarking ##############

        self._log("Start benchmarking run %s of node_chain %s on dataset %s"
                                % (self.run,
                                   self.node_chain_spec,
                                   self.rel_dataset_dir))


        # Do the actual benchmarking for this collection/node_chain combination
        try:
            result_collection = \
                self.node_chain.benchmark(
                    input_collection=input_collection,
                    run=self.run,
                    persistency_directory=self.persistency_dir,
                    store_node_chain=self.store_node_chain)
        except Exception, exception:
            # Send Exception to Logger
            import traceback
            self._log(traceback.format_exc(), level=logging.ERROR)
            raise

        self._log("Finished benchmarking run %s of node_chain %s on dataset %s"
                  % (self.run, self.node_chain_spec,
                     self.rel_dataset_dir))

        ############## Postprocessing ##############

        # Add input collection, node_chain file name, and run number
        # to the meta data
        meta_data = {"node_chain_spec": self.node_chain_spec,
                     "parameter_setting": self.parameter_setting,
                     "input_collection_name": self.rel_dataset_dir,
                     "hide_parameters": self.hide_parameters}
        result_collection.update_meta_data(meta_data)

        # Store the result collection to the hard disk
        if self.storage_format:
            result_collection.store(self.result_dataset_directory,
                                    s_format=self.storage_format)
        else:
            result_collection.store(self.result_dataset_directory)
        l = len(self.node_chain)
        for i in range(len(self.node_chain)):
            self.node_chain[l-i-1].reset()
        del result_collection
        del self.node_chain
        gc.collect()

        ############## Clean up after benchmarking ##############
        super(NodeChainProcess, self).post_benchmarking()

    @staticmethod
    def _check_node_chain_dataset_consistency(node_chain, dataset):
        """ Checks that the given node_chain can process the given dataset

        Therefore the first node in the node_chain has to be the corresponding
        source node to the dataset type

        .. todo:: Check dynamically! So that explicit imports aren't necessary!
                  For test node_chains use dummy dataset.
        """
        from pySPACE.missions.nodes.source.time_series_source \
            import Stream2TimeSeriesSourceNode
        from pySPACE.missions.nodes.source.time_series_source \
            import TimeSeriesSourceNode
        from pySPACE.missions.nodes.source.time_series_source \
            import TimeSeries2TimeSeriesSourceNode
        from pySPACE.missions.nodes.source.feature_vector_source \
            import FeatureVectorSourceNode
        from pySPACE.missions.nodes.source.test_source_nodes \
            import DataGenerationTimeSeriesSourceNode

        from pySPACE.resources.dataset_defs.stream \
            import StreamDataset
        from pySPACE.resources.dataset_defs.time_series \
            import TimeSeriesDataset
        from pySPACE.resources.dataset_defs.feature_vector \
            import FeatureVectorDataset

        if isinstance(node_chain[0], DataGenerationTimeSeriesSourceNode):
            # the test node does not care about the dataset type
            pass
        elif isinstance(node_chain[0], Stream2TimeSeriesSourceNode):
            assert(isinstance(dataset, StreamDataset) or
                   isinstance(dataset, TimeSeriesDataset) and
                   isinstance(node_chain[0], TimeSeries2TimeSeriesSourceNode)), \
              "Node chain with input node of type %s cannot process dataset of type %s" \
                        % (type(node_chain[0]), type(dataset))
        elif isinstance(node_chain[0], TimeSeriesSourceNode):
            assert(isinstance(dataset, TimeSeriesDataset)), \
                   "Node chain with input node of type %s cannot process dataset of type %s" \
                   % (type(node_chain[0]), type(dataset))
        elif isinstance(node_chain[0], FeatureVectorSourceNode):
            assert(isinstance(dataset, FeatureVectorDataset)), \
                "Node chain with input node of type %s cannot process dataset of type %s" \
                % (type(node_chain[0]), type(dataset))

