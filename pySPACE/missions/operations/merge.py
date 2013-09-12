""" Define train and test data for One versus Rest or Rest versus One in cross validation fashion

The result summary of this operation contains one dataset for every
dataset of the *input_path*, which uses data from this dataset as test
data and the data of all other datasets as training data. For instance, if
the input consists of the three datasets "A", "B", "C", the result summary
will contain the 3 datasets "Rest_vs_A", "Rest_vs_B", and "Rest_vs_C".
The result dataset "Rest_vs_A" uses the data from dataset "A" as test data
and the data from all other datasets as train data. If *reverse* is True
this will result in the 3 datasets "A_vs_Rest", "B_vs_Rest" and
"C_vs_Rest".

Specification file Parameters
+++++++++++++++++++++++++++++

type
----

Has to be set to *merge* to use this operation!

(*obligatory, merge*)

input_path
----------
The input path of this operation has to contain several datasets of one of
the types :mod:`~pySPACE.resources.dataset_defs.time_series` or
:mod:`~pySPACE.resources.dataset_defs.feature_vector`.
The input datasets must not contain split data.

(*obligatory*)

reverse
-------

Switch to use *One_vs_Rest* scheme instead of *Rest_vs_one* scheme

(*optional, default: False*)

collection_constraints
----------------------

Optionally, constraints can be passed to the operation that specify which
datasets are used as training data for which test data. For instance, the
constraint '"%(source_train_collection_name)s".strip("}{").split("}{")[1:] ==
"%(source_test_collection_name)s".strip("}{").split("}{")[1:]' would cause
that only datasets are combined that were created by the same processing
with the same parametrization.

(*optional, default: []*)

Exemplary Call
++++++++++++++

.. code-block:: yaml

    type: merge
    input_path: "operation_results/2009_8_13_15_8_57"
    reverse: False
    collection_constraints:
      # Combine only collections that have been created using the same parameterization
      - '"%(source_train_collection_name)s".strip("}{").split("}{")[1:] == "%(source_test_collection_name)s".strip("}{").split("}{")[1:]'
      
.. todo::   When applying a rewindowing on merged time series data with the
            :class:`~pySPACE.missions.nodes.source.time_series_source.TimeSeries2TimeSeriesSourceNode`
            and specifying an endmarker in the windower spec file only data from
            the first set might be used. Here, an additional marker handling
            could be implemented, e.g., delete middle start- and end-marker
            ('S  8', 'S  9'), to have one merged dataset with defined start
            and endpoint or add synthetic ones.
            Alternatively the :mod:`~pySPACE.missions.support.windower` could
            be modified, e.g., to handle multiple start and end markers.
            The marker information is stored in the `marker_name` variable
            of the :class:`~pySPACE.resources.data_types.time_series.TimeSeries`
            objects.
"""

import os
import sys
import glob
import shutil
import time
import pwd

if sys.version_info[0] == 2 and sys.version_info[1] < 6:
    import processing
else:
    import multiprocessing as processing

import logging

import pySPACE
from pySPACE.missions.operations.base import Operation, Process
from pySPACE.tools.filesystem import create_directory

from pySPACE.resources.dataset_defs.base import BaseDataset
    
class MergeOperation(Operation):
    """ Operation to create 'All_vs_One' datasets """
    def __init__(self, processes, operation_spec, result_directory,
                 number_processes, create_process=None):
        super(MergeOperation, self).__init__(processes, operation_spec,
                                             result_directory)
        self.number_processes = number_processes
        self.create_process = create_process
        
    @classmethod
    def create(cls, operation_spec, result_directory, debug=False, input_paths=[]):
        """ [factory method] Create a MergeOperation object.
        
        A factory method that creates a MergeOperation based on the 
        information given in the operation specification operation_spec
        """
        assert(operation_spec["type"] == "merge")

        # Determine constraints for collections that are combined
        collection_constraints = []
        if "collection_constraints" in operation_spec:
            collection_constraints.extend(operation_spec["collection_constraints"])
        if "reverse" in operation_spec:
            reverse = operation_spec["reverse"]
        else:
            reverse = False

        # merging is not distributed over different processes
        number_processes = 1
        processes = processing.Queue()
        
        # Create the Merge Process 
        cls._createProcesses(processes, input_paths, result_directory,
                             collection_constraints, reverse)
                           
        # create and return the merge operation object
        return cls(processes, operation_spec, result_directory, number_processes)

    @classmethod
    def _createProcesses(cls, processes, input_collections, result_directory,
                         collection_constraints, reverse):
        """[factory method] Create the MergeProcess object."""
        
        # Create the merge process and put it in the execution queue
        processes.put(MergeProcess(input_collections, result_directory,
                                    collection_constraints, reverse))
        # give executing process the sign that creation is now finished
        processes.put(False)    
    
    def consolidate(self):
        """ Consolidation of the operation's results """
        # Just do nothing
        pass                    
    
    
class MergeProcess(Process):
    """ Create 'All_vs_One' collections where 'All' are all collections that fulfill the *collection_constraints* and are different from the "One" collection
    
    Restricted to pickle and arff files!
    
    .. todo:: Merging of csv-files?
    """
    
    def __init__(self, input_collection, result_directory, 
                 collection_constraints, reverse):
        super(MergeProcess, self).__init__()
        
        self.input_collections = input_collection
        self.result_directory = result_directory
        self.collection_constraints = collection_constraints
        self.reverse = reverse
    
    def __call__(self):
        """ Executes this process on the respective modality """
        ############## Prepare benchmarking ##############
        super(MergeProcess, self).pre_benchmarking()
        
        # For all input collections
        for source_test_collection_path in self.input_collections:
            # Check if the input data is splitted
            # e.g. only a single test file is in the source directory 
            source_files = glob.glob(os.sep.join([source_test_collection_path,
                                                  "data_run0", "*test*"]))
            splitted = len(source_files) > 1
            assert(not splitted)
            source_file_name = str(source_files[-1])
            
            # check if train sets are also present
            train_data_present = len(glob.glob(os.sep.join(
                                 [source_test_collection_path,"data_run0",\
                                  "*train*"]))) > 0
            
            # if training data is present -> use train and test sets separately
            if train_data_present:
                train_set_name_suffix = "train"
            else:
                train_set_name_suffix =  "test"
            
            # We create the collection Rest_vs_Collection
            source_test_collection_name = \
                                   source_test_collection_path.split(os.sep)[-2]
            test_base_collection_name = \
                          source_test_collection_name.strip("}{").split("}{")[0]
            if self.reverse:
                target_collection_name = source_test_collection_name.replace(
                                         test_base_collection_name,
                                         test_base_collection_name + "_vs_Rest")
                key = "train"
            else:
                target_collection_name = source_test_collection_name.replace(
                                         test_base_collection_name,
                                         "Rest_vs_" + test_base_collection_name)
                key = "test"
                
            target_collection_path = os.sep.join([self.result_directory,
                                                  target_collection_name])
            # determine the parameter_settings of the test collection
            test_collection = BaseDataset.load(source_test_collection_path)
            target_collection_params = \
                                 test_collection.meta_data["parameter_setting"]
            target_collection_params["__INPUT_DATASET__"] = \
                                           {key: source_test_collection_name}
            
            if source_file_name.endswith("arff"):
                file_ending = "arff"
                # Copy arff file from input collection to target collection
                source_test_file_path = os.sep.join([source_test_collection_path,
                                        "data_run0","features_sp0" +
                                        train_set_name_suffix + ".arff"])
                target_test_file_path = os.sep.join([target_collection_path,
                                       "data_run0","features_sp0_"+key+".arff"])
            
            elif source_file_name.endswith("pickle"):
                file_ending = "pickle"
                source_test_file_path = source_test_collection_path
                target_test_file_path = target_collection_path
            else:
                raise NotImplementedError("File type not supported in " \
                                                               "MergeOperation")
            
            source_train_pathes = []
            for source_train_collection_path in self.input_collections:
                source_train_collection_name = \
                                  source_train_collection_path.split(os.sep)[-2]
                # We must not use data originating from the same input
                # collection both in train and test files
                if source_test_collection_name == source_train_collection_name:
                    continue
                
                # Check that all constraints are fulfilled for this pair of
                # input collections
                if not all(eval(constraint_template % \
                  {'source_train_collection_name': source_train_collection_name,
                   'source_test_collection_name': source_test_collection_name})
                        for constraint_template in self.collection_constraints):
                    continue
                
                # check if all parameters are stored in the target path
                source_collection = \
                                BaseDataset.load(source_train_collection_path)
                source_collection_params = \
                            source_collection.meta_data["parameter_setting"]
                remaining_params = \
                          [param for param in source_collection_params.items() \
                            if param not in target_collection_params.items() and \
                               param[0] not in ["__INPUT_DATASET__", 
                               "__RESULT_DIRECTORY__", "__OUTPUT_BUNDLE__",
                               "__INPUT_COLLECTION__" ]] # for old data
                if remaining_params != []:
                    for k,v in remaining_params:
                         target_collection_path += "{%s#%s}" % (k,str(v))
                         target_collection_params[k]=v
                   
                if "arff" == file_ending:
                    source_train_file_path = \
                                      os.sep.join([source_train_collection_path, 
                                                "data_run0", "features_sp0_" + \
                                               train_set_name_suffix + ".arff"])
                elif "pickle" == file_ending:
                    source_train_file_path = source_train_collection_path

                else:
                    raise NotImplementedError("File type not supported in " \
                                                              "MergeOperation!")     
                    
                source_train_pathes.append(source_train_file_path)
            
            if "arff" == file_ending:
                target_train_file_path = os.sep.join([target_collection_path,
                                       "data_run0","features_sp0_"+key+".arff"])
            elif "pickle" == file_ending:
                target_train_file_path = target_collection_path
            else:
                raise NotImplementedError("File type not supported in "
                                                              "MergeOperation!")     
            
            if len(source_train_pathes) == 0:
                continue
            
            create_directory(os.sep.join([target_collection_path,
                                          "data_run0"]))
            
            if "arff" == file_ending:
                self._copy_arff_file(source_test_file_path, 
                                     target_test_file_path,
                                     source_test_collection_name, 
                                     target_collection_name)
                                
                self._merge_arff_files(target_train_file_path, 
                                       source_train_pathes,
                                       target_collection_name)
                # Copy metadata.yaml
                # TODO: Adapt to new collection
                input_meta = BaseDataset.load_meta_data(source_test_collection_path)
                BaseDataset.store_meta_data(target_collection_path,input_meta)
            elif "pickle" == file_ending:
                self._copy_pickle_file(source_test_collection_path,
                                       target_collection_path,
                                       train_set_name_suffix)

                self._merge_pickle_files(target_train_file_path, 
                                         source_train_pathes, 
                                         train_set_name_suffix,
                                         target_collection_params)
            else:
                raise NotImplementedError("File type not supported in merge_operation")
            
        ############## Clean up after benchmarking ##############
        super(MergeProcess, self).post_benchmarking()
        

    def _merge_arff_files(self, target_arff_file_path, merge_arff_file_pathes,
                          target_collection_name):
        """ Copy the instances from the merge arff files to the target arff file"""
        source_train_collection_name = merge_arff_file_pathes[0].split(os.sep)[-4]
        self._copy_arff_file(merge_arff_file_pathes[0], target_arff_file_path,
                             source_train_collection_name, target_collection_name)
        # Open target file for appending new instances
        target_file = open(target_arff_file_path, 'a')
        
        # Append all instances contained in the extension arff files to the 
        # target arff file
        for merge_arff_file_path in merge_arff_file_pathes[1:]:
            merge_arff_file = open(merge_arff_file_path, 'r')
            target_file.writelines(line for line in merge_arff_file.readlines()
                                               if not line.startswith('@'))
            merge_arff_file.close()
            
        target_file.close()
        
    def _copy_arff_file(self, input_arff_file_path, target_arff_file_path,
                        input_collection_name, target_collection_name):
        """ Copy the arff files and adjust the relation name in the arff file"""
        file = open(input_arff_file_path, 'r')
        content = file.readlines()
        file.close()
        content[0] = content[0].replace(input_collection_name, 
                                        target_collection_name) 
        file = open(target_arff_file_path, 'w')
        file.writelines(content)
        file.close()
        
    def _merge_pickle_files(self, target_collection_path, source_collection_pathes,
                                  train_set_name_suffix, target_collection_params):
        """ Merge all collections in source_collection_pathes and store them \
            in the target collection"""
        
        # load a first collection, in which the data of all other collections 
        # is assembled
        target_collection = BaseDataset.load(source_collection_pathes[0])
        try:
            author = pwd.getpwuid(os.getuid())[4]
        except:
            author = "unknown"
            self._log("Author could not be resolved.",level=logging.WARNING)
        date = time.strftime("%Y%m%d_%H_%M_%S")
        # Delete node_chain file name
        try:
            target_collection.meta_data.pop("node_chain_file_name")
        except:
            pass
        # Update meta data and store it
        k = "test" if self.reverse else "train"
        target_collection_params["__INPUT_DATASET__"][k] = \
                 [s_c_p.split(os.sep)[-2] for s_c_p in source_collection_pathes]
        target_collection_params["__RESULT_DIRECTORY__"] = self.result_directory
        target_collection.meta_data.update({
                "author" : author, 
                "date" : date, 
                "dataset_directory" : target_collection_path,
                "train_test" : True,
                "parameter_setting" : target_collection_params,
                "input_collection_name" : source_collection_pathes[0][len(
                                        pySPACE.configuration.storage):]
        })
      
        # merge data of all other collections to target collection
        for source_collection_path in source_collection_pathes[1:]:
            source_collection = BaseDataset.load(source_collection_path)
            for run in source_collection.get_run_numbers():
                for split in source_collection.get_split_numbers():
                    data = source_collection.get_data(run, split, 
                                                          train_set_name_suffix)
                    target_data = target_collection.get_data(run, split, 
                                                          train_set_name_suffix)
                    # actual data is stored in a list that has to be extended
                    target_data.extend(data)
                    
        # if only test data was given, the "Rest_vs" collection is stored as 
        # training data
        if not self.reverse and "test" == train_set_name_suffix: 
            # exchange the "test" in key tuple to "train" before storing
            for key in target_collection.data.keys():
                assert("test" == key[2])
                value = target_collection.data.pop(key)
                key = (key[0],key[1],"train")
                target_collection.data[key] = value
                    
        target_collection.store(target_collection_path)
        
        
    def _copy_pickle_file(self, source_collection_path, target_collection_path,
                          train_set_name_suffix):
        
        source_collection = BaseDataset.load(source_collection_path)
        # if only test data was given, the "Rest_vs" collection is stored as 
        # training data
        if self.reverse and "test" == train_set_name_suffix: 
            # exchange the "test" in key tuple to "train" before storing
            for key in source_collection.data.keys():
                assert("test" == key[2])
                value = source_collection.data.pop(key)
                key = (key[0],key[1],"train")
                source_collection.data[key] = value
        source_collection.store(target_collection_path)
