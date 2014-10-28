""" Use Weka's Filter that transform one arff file into another.

A WEKA filter process consists of applying a filter to all arff-files contained 
in the input path. Filters may be using (un-)supervised training
on the train datasets. For instance, feature selector filter is trained  on
a train set so that a subset of the features are selected. The results of the
process consists of projecting both the train and the respective test set on the
selected features. The results of all these processes are stored in a temporary
directory and after the completion of all processes of the operation,
the consolidate method of the *WEKAFilterOperation* is executed and
the results are merged into a consistent representation of the operations result
collection.

http://www.cs.waikato.ac.nz/ml/weka/
"""

import sys
import os
import glob
import yaml
import time
import logging
if sys.version_info[0] == 2 and sys.version_info[1] < 6:
    import processing
else:
    import multiprocessing as processing

import pySPACE
from pySPACE.missions.operations.base import Operation, Process
from pySPACE.resources.dataset_defs.base import BaseDataset
from pySPACE.tools.filesystem import create_directory

    
class WekaFilterOperation(Operation):
    """ Operation for feature selection using Weka
    
    A WEKA Filter operation consists of a set of WEKA Filter 
    processes. Each of these processes stores its results in a 
    temporary directory. The operation collects the results of these processes
    using the consolidate method that produces a consistent representation
    of the result collections. 
    """   
    def __init__(self, processes, operation_spec, result_directory,
                 number_processes, create_process=None):
        super(WekaFilterOperation, self).__init__(processes,
                                                  operation_spec,
                                                  result_directory)
        self.number_processes = number_processes
        self.create_process = create_process
        
    @classmethod
    def create(cls, operation_spec, result_directory, debug=False, input_paths=[]):
        """
        A factory method that creates an WEKA operation based on the 
        information given in the operation specification operation_spec
        """
        assert(operation_spec["type"] == "weka_filter")
        
        # Determine all parameter combinations that should be tested
        parameter_settings = cls._get_parameter_space(operation_spec)
        
        if "hide_parameters" in operation_spec:
            hide_parameters = operation_spec["hide_parameters"]
        else:
            hide_parameters = []
        
        # Read the command template from a file
        template_file = open(os.path.join(pySPACE.configuration.spec_dir,
                                          "operation",
                                          "weka_templates",
                                          operation_spec["template"]), 'r')
        command_template = template_file.read()
        template_file.close()
        
        # number of processes
        if "runs" in operation_spec:
            number_processes = len(input_paths) * len(parameter_settings) * \
                           operation_spec["runs"]
        else: # approximate the number of processes 
            runs = []
            for dataset_dir in input_paths:
                collection = BaseDataset.load(dataset_dir)
                runs.append(collection.meta_data["runs"])
            runs = max(runs)
            number_processes = len(input_paths) * len(parameter_settings) * \
                               runs
        
        if debug == True:
            # To better debug creation of processes we don't limit the queue 
            # and create all processes before executing them
            processes = processing.Queue()
            cls._createProcesses(processes, result_directory, operation_spec, 
                                 parameter_settings, input_paths,
                                 command_template, hide_parameters)
            # create and return the weka operation object
            operation = cls(processes, operation_spec, result_directory, 
                       number_processes)
        else:
            # Create all processes by calling a recursive helper method in 
            # another thread so that already created processes can be executed in 
            # parallel. Therefor a queue is used which size is maximized to 
            # guarantee that not to much objects are created (because this costs
            # memory). However, the actual number of 100 is arbitrary and might
            # be reviewed.
            processes = processing.Queue(4)
            create_process = processing.Process(target=cls._createProcesses,
                             args=( processes, result_directory, operation_spec, 
                                    parameter_settings, input_paths,
                                    command_template, hide_parameters))
            create_process.start()            
            # create and return the weka operation object
            operation = cls(processes, operation_spec, result_directory, 
                       number_processes, create_process)
        # We remember the command_template and how many parameters are passed 
        # for consolidation    
        operation.num_parameters = len(operation_spec["parameter_ranges"]) \
                                    - len(hide_parameters)
        operation.command_template = command_template
        
        return operation
            
    @classmethod
    def _createProcesses(cls, processes, result_directory, operation_spec, 
                parameter_settings, input_collections, command_template,
                hide_parameters):                           
        # For each combination of filter, input-collection and
        # run number, create one WEKA_process
        for dataset_dir in input_collections:
            collection = BaseDataset.load(dataset_dir)
            # Determine the number of iterations and splits to be used
            iterations = collection.meta_data["runs"]
            splits = collection.meta_data["splits"]
            if "runs" in operation_spec:
                assert(iterations in [1, operation_spec["runs"]])
                iterations = operation_spec["runs"]
            for parametrization in parameter_settings: 
                for run_number in range(iterations):
                    for split_number in range(splits):
                        process = WEKAFilterProcess(dataset_dir,
                                                    command_template,
                                                    parametrization,
                                                    run_number,
                                                    split_number,
                                                    result_directory,
                                                    hide_parameters = hide_parameters)
                        processes.put(process)
        # give executing process the sign that creation is now finished  
        processes.put(False)            
    
    def consolidate(self):
        """
        Consolidates the results obtained by the single WEKA filter
        processes into a consistent summary of datasets that is stored on
        the file system.
        
        .. todo:: Some of the contents of this method should go into the
                  :class:`~pySPACE.resources.dataset_defs.feature_vector.FeatureVectorDataset`
        """

        # Iterate over all collections and store the collection meta data etc.
        for entries in os.listdir(self.result_directory):
            fullpath = os.path.join(self.result_directory, entries)
            # For each collection        
            if os.path.isdir(fullpath):
                if entries.startswith("{"):
                    # Extract the parameters from the collection name in order to
                    # adjust the relation name
                    if self.num_parameters > 0:
                        parameter_strings = entries.strip("}{").split("}{")[-self.num_parameters:]
                        parameter_postfix = "{" + "}{".join(parameter_strings) + "}"
                    else:
                        parameter_strings = ""
                        parameter_postfix = ""
                    # Postprocessing of the arff files of this collection
                    for train_arff_file in glob.glob(fullpath + os.sep + "data_run*" 
                                           + os.sep + "*train.arff"):
                        # Adjust the relation name of the train file
                        content = open(train_arff_file, 'r').readlines()             
                        # We strip everything after the last "}"
                        endindex = content[0].rfind("}")
                        content[0] = content[0][:endindex+1]
                        content[0] += parameter_postfix + "'"
                        open(train_arff_file, 'w').writelines(content)
                        # Use relation name of train data for test data
                        test_arff_file = train_arff_file.replace("train.arff", "test.arff") 
                        test_content = open(test_arff_file, 'r').readlines()
                        test_content[0] = content[0] + "\n"
                        open(test_arff_file, 'w').writelines(test_content)
                    
                        # Check which features are contained in the arff file
                        feature_names = []
                        for line in content:
                            if line.startswith("@attribute"):
                                attribute = line.split()[1]
                                if attribute is not "class":
                                    feature_names.append(attribute)
                    # Store the collection meta data etc.
                    if self.num_parameters > 0:
                        input_collection_name = \
                            "{" + "}{".join(entries.strip("}{").split("}{")[:-self.num_parameters]) + "}"
                    else:
                        input_collection_name = entries
                        
                    input_collection_path = os.path.join(self.operation_spec["input_path"],
                                                     input_collection_name)

                    input_collection_meta = BaseDataset.load_meta_data(
                                            pySPACE.configuration.storage
                                            + os.sep
                                            + input_collection_path)
                    # Store the input collection
                    BaseDataset.store_meta_data(fullpath, input_collection_meta,
                                                file_name="input_metadata.yaml")
                    # Adjust collection metadata for the new collection
                    input_collection_meta["feature_names"] = feature_names
                    input_collection_meta["num_features"] = len(feature_names)
                    try:
                        import platform
                        CURRENTOS = platform.system()
                        if CURRENTOS == "Windows":
                            import getpass
                            author = getpass.getuser()
                        else:
                            import pwd
                            author = pwd.getpwuid(os.getuid())[4]

                        input_collection_meta["author"] = author
                    except :
                        input_collection_meta["author"] = "unknown"
                        self._log("Author could not be resolved.",level=logging.WARNING)
                    input_collection_meta["date"] = time.strftime("%Y%m%d")
                    input_collection_meta["input_collection_name"] = input_collection_name
                    # Write the collection meta information into the folder
                    BaseDataset.store_meta_data(fullpath,input_collection_meta)
                    # Store the command_template
                    command_template_file = open(os.path.join(fullpath,
                                                          "command_template"), 'w')
                    command_template_file.write(self.command_template)
                    command_template_file.close()
                else:
                    # training and test arff need the same relation name
                    # otherwise Weka can't relate it to each other; the collection
                    # name and the parameters in {}{}-optic must be the relation 
                    # name for further processing    
                    self._log("WARNING: Collection name doesn't begin with '{'. Further processing may be collapse!", level= logging.WARNING)
        # Write the specification of this operation
        # to the result directory in order to make later 
        # analysis of results more easy
        source_operation_file = open(os.path.join(self.result_directory,
                                                  "source_operation.yaml"), 'w')
        yaml.dump(self.operation_spec, source_operation_file)
        source_operation_file.close()
    
    
class WEKAFilterProcess(Process):
    """ Process for using Weka's filters
    
    A WEKA filter process consists of applying a filter to all arff-files 
    contained in the *input_path*. Filters may be using
    (un-)supervised training on the train datasets. For instance, the feature 
    selector filter is trained  on a train set so that a subset of the 
    features are selected. The results of 
    the process consists of projecting both the train and the respective test 
    set on the selected features. The results of all these processes are stored
    in a temporary directory.
    """    
    
    def __init__(self, dataset_dir, command_template, parametrization,
                 run_number, split_number, operation_result_dir,
                 hide_parameters = []):
        
        super(WEKAFilterProcess, self).__init__()
        
        # Determine the directory in which the of the process' results
        # are stored
        result_collection_name = dataset_dir.split(os.sep)[-2]
        for parameter_name, parameter_value in parametrization.iteritems():
            # If this is a parameter that should not be hidden, then we have to
            # encode it in the result collection name 
            if not parameter_name in hide_parameters:
                result_collection_name += "{__%s__:%s}" % (parameter_name.upper(),
                                                           parameter_value)
                                                                     
        self.result_directory = os.path.join(operation_result_dir,
                                             result_collection_name)
        
        # Create directory for intermediate results if it does not exist yet
        create_directory(self.result_directory 
                              + os.sep + "data_run%s" % run_number)
                
        # Create collection
        collection = BaseDataset.load(dataset_dir)
        
        # The parametrization that is independent of the collection type 
        # and the specific weka command template that is executed
        self.params = {"dataset_name": dataset_dir.replace('/','_'),
                       "dataset_dir": dataset_dir,
                       "run_number": run_number,
                       "split_number": split_number,
                       "weka_class_path": pySPACE.configuration.weka_class_path,
                       "temp_results": self.result_directory}

        # Load the abbreviations
        abbreviations_file = open(os.path.join(pySPACE.configuration.spec_dir,
                                               'operations/weka_templates',
                                               'abbreviations.yaml'), 'r')
        self.abbreviations = yaml.load(abbreviations_file)
        # Add custom parameters for the weka command template
        for parameter_name, parameter_value in parametrization.iteritems():
            # Auto-expand abbreviations
            if parameter_value in self.abbreviations:
                parameter_value = self.abbreviations[parameter_value]
            self.params[parameter_name] = parameter_value
            
        # Build the WEKA command by repeatedly replacing all placeholders in 
        # the template 
        while True:
            instantiated_template = command_template % self.params
            if instantiated_template == command_template:
                # All placeholders replace 
                self.weka_command = instantiated_template
                break
            else:
                # We have to continue since we are not converged
                command_template = instantiated_template
        
        self.handler_class = None
        
    
    def __call__(self):
        """ Executes this process on the respective modality """
        # Restore configuration
        pySPACE.configuration = self.configuration    
        
        ############## Prepare benchmarking ##############
        super(WEKAFilterProcess, self).pre_benchmarking()
        
        # Execute the java command in this OS process
        os.system(self.weka_command)
        
        ############## Clean up after benchmarking ##############
        super(WEKAFilterProcess, self).post_benchmarking() 
        
        
