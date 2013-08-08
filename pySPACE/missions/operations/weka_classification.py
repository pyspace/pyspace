""" Classification using the WEKA experimenter

A WEKA classification process consists of executing a certain 
WEKA experiment. The results of all these processes are stored in a temporary
directory and after the completion of all processes of the operation,
the consolidate method of the *WekaClassificationOperation* is executed and the
results are merged into a consistent representation of the operations result
collection.

http://www.cs.waikato.ac.nz/ml/weka/
"""
import sys
import os
import glob
import yaml
if sys.version_info[0] == 2 and sys.version_info[1] < 6:
    import processing
else:
    import multiprocessing as processing
import pySPACE
from pySPACE.missions.operations.base import Operation, Process
from pySPACE.resources.dataset_defs.base import BaseDataset
from pySPACE.resources.dataset_defs.performance_result import PerformanceResultSummary
from pySPACE.tools.filesystem import create_directory

    
class WekaClassificationOperation(Operation):
    """ Operation for classification using Weka experimenter
    
    A Weka classification operation consists of a set of WEKA processes. Each of
    these processes consists of executing a certain WEKA experiment.
    
    The results of this operation are collected using the
    consolidate method that produces a consistent representation
    of the result collections. 
    """   
    def __init__(self, processes, operation_spec, result_directory,
                 number_processes, create_process=None):
        super(WekaClassificationOperation, self).__init__(processes,
                                                          operation_spec,
                                                          result_directory)
        self.create_process = create_process
        self.number_processes = number_processes
        
    @classmethod
    def create(cls, operation_spec, result_directory, debug=False, input_paths=[]):
        """
        A factory method that creates an WEKA operation based on the 
        information given in the operation specification operation_spec
        """
        assert(operation_spec["type"] == "weka_classification")
        # Determine all parameter combinations that should be tested
        parameter_settings = cls._get_parameter_space(operation_spec)
        
        # Read the command template from a file
        template_file = open(os.path.join(pySPACE.configuration.spec_dir,
                                               "operations",
                                               "weka_templates",
                                               operation_spec["template"]),
                             'r')
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
                                 command_template)
            # create and return the weka operation object
            return cls(processes, operation_spec, result_directory, 
                       number_processes)
        else:
            # Create all processes by calling a recursive helper method in 
            # another thread so that already created processes can be executed in 
            # parallel. Therefore a queue is used which size is maximized to 
            # guarantee that not to much objects are created (because this costs
            # memory). However, the actual number of 100 is arbitrary and might
            # be reviewed.
            processes = processing.Queue(100)
            create_process = processing.Process(target=cls._createProcesses,
                             args=( processes, result_directory, operation_spec, 
                                    parameter_settings, input_paths,
                                    command_template))
            create_process.start()            
            # create and return the weka operation object
            return cls(processes, operation_spec, result_directory, 
                       number_processes, create_process)        
    
    @classmethod
    def _createProcesses(cls, processes, result_directory, operation_spec, 
                parameter_settings, input_collections, command_template):     
     
        # For each combination of classifier, input-collection and
        # run number, create one WEKA_process
        for dataset_dir in input_collections:
            collection = BaseDataset.load(dataset_dir)
            # Determine the number of iterations and splits to be used
            iterations = collection.meta_data["runs"]
            splits = collection.meta_data["splits"] 
            if "runs" in operation_spec:
                assert(iterations in [1, operation_spec["runs"]])
                iterations = operation_spec["runs"]
            if "cv_folds" in operation_spec:
                assert(splits in [1, operation_spec["cv_folds"]])
                splits = operation_spec["cv_folds"]          
            
            for parametrization in parameter_settings: 
                for run_number in range(iterations):
                    process = WEKAClassificationProcess(dataset_dir,
                                                        command_template,
                                                        parametrization,
                                                        splits,
                                                        run_number,
                                                        result_directory)
                    processes.put(process)
        # give executing process the sign that creation is now finished                
        processes.put(False)
            
    def consolidate(self):
        """
        Consolidates the results obtained by the single WEKA processes into
        a consistent structure of collections that are stored on the
        file system.
        """
        self._log("Consolidating results ...")
        # We load and store the results once into a PerformanceResultSummary.
        # From_multiple csv does the necessary consolidation
        # and mixes and parses the table.
        self._log("Reading intermediate results...")
        result_collection = PerformanceResultSummary(dataset_dir=self.result_directory)
        
        self._log("done")
        self._log("Storing result collection")
        
        result_collection.store(self.result_directory)
        
        self._log("done")
        
        
        # Write the specification of this operation
        # to the result directory in order to make later 
        # analysis of results more easy
        source_operation_file = open(os.path.join(self.result_directory,
                                                  "source_operation.yaml"), 'w')
        yaml.dump(self.operation_spec, source_operation_file)
        source_operation_file.close()

class WEKAClassificationProcess(Process):
    """ Process for classification using Weka
    
    A WEKA classification process consists of executing a certain WEKA 
    experiment. This experiment is defined by a template in which certain aspects 
    can be configured, for instance:
    
     * which classifier is used
     * which data set is processed
     * how many cross validation folds are used etc.
     
    The results of the WEKA experiment are written to the file system and
    later on collected and consolidated during the consolidation of
    the *WekaClassificationOperation*.
    
    """
    unique_id = 0
    
    def __init__(self,
                 dataset_dir,
                 command_template,
                 parametrization,
                 cv_folds,
                 run_number,
                 operation_result_dir):
        super(WEKAClassificationProcess, self).__init__()
        # Load the abbreviations
        abbreviations_file = open(os.path.join(pySPACE.configuration.spec_dir,
                                               'operations/weka_templates',
                                               'abbreviations.yaml'), 'r')
        self.abbreviations = yaml.load(abbreviations_file)
        abbreviations_file.close()
        # Determine the directory in which the process' results
        # are stored
        self.result_directory = operation_result_dir
        # Create collection
        collection = BaseDataset.load(dataset_dir)
        # The parametrization that is independent of the collection type
        # and the specific weka command template that is executed
        self.params = {"collection_name": dataset_dir.strip(os.sep).split(os.sep)[-1],
                       "run_number": run_number,
                       "cv_folds": cv_folds,
                       "weka_class_path": pySPACE.configuration.weka_class_path,
                       "temp_results": self.result_directory,
                       "unique_id": WEKAClassificationProcess.unique_id}
        # Collection dependent parameters
        if not collection.meta_data["train_test"] \
             and collection.meta_data["splits"] == 1:
            raise NotImplementedError()
        else:
            # The pattern of the train and test files generated by crossvalidation
            data_pattern =  os.path.join(dataset_dir,
                                         collection.meta_data["data_pattern"])
            # One example arff file in which WEKa can look up relation name etc.
            sample_dataset =  data_pattern.replace("_run", "_run0")\
                                          .replace("_sp_","_sp0_")\
                                          .replace("_tt","_train")
            self.params.update({"sample_dataset": sample_dataset,
                                "data_pattern": data_pattern})
        # Add custom parameters for the weka command template
        for parameter_name, parameter_value in parametrization.iteritems():
            self.params[parameter_name + "_abbr"] = parameter_value
            # Auto-expand abbreviations
            if parameter_value in self.abbreviations:
                parameter_value = self.abbreviations[parameter_value]
            elif parameter_name == 'classifier':
                import warnings
                warnings.warn("Did not find classifier abbreviation %s. "
                              " Expecting full name." % parameter_value)
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
        
        WEKAClassificationProcess.unique_id += 1

    def __call__(self):
        """
        Executes this process on the respective modality
        """
        # Restore configuration
        pySPACE.configuration = self.configuration
        ############## Prepare benchmarking ##############
        super(WEKAClassificationProcess, self).pre_benchmarking()
        # Execute the java command in this OS process
        os.system(self.weka_command)
        ############## Clean up after benchmarking ##############
        super(WEKAClassificationProcess, self).post_benchmarking()

