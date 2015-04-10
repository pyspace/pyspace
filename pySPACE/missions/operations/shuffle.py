""" Take combinations of datasets in the summary for training and test each

The input of this
operation has to contain several comparable datasets of the same type.
Depending on whether the input datasets contain split data, the behavior
of this operation differs slightly.

.. note:: This operation creates an output directory with links,
        not duplicated files!

If the input datasets are not split, the result  of this operation
contains one dataset for every pair of datasets of the *input_path*.
For instance, if the input consists of the three datasets "A", "B",
"C", the result will at least contain the 6 datasets "A_vs_B",
"A_vs_C", "B_vs_A", "B_vs_C, "C_vs_A", "C_vs_B". The result dataset "A_vs_B" 
uses the feature vectors from dataset "A" as training data and the feature 
vectors from dataset "B" as test data.

If the input datasets contain split data, additionally the input
datasets are copied to the result directory so that this would contain
9 datasets. The dataset "X_vs_Y" contains the train data from dataset X
from the respective split for training and the test data from dataset Y for
testing.

A typical operation specification file might look like this

Specification file Parameters
+++++++++++++++++++++++++++++

type
----

Has to be set to *shuffle* to use this operation!

(*obligatory, shuffle*)

input_path
----------

Location of the input data

(*obligatory*)

dataset_constraints
-------------------

Optionally, constraints can be passed to the operation that specify which
datasets are combined based on the dataset name. For instance, the
constraint
'"%(dataset_name1)s".strip("}{").split("}{")[1:] == "%(dataset_name2)s".strip("}{").split("}{")[1:]'
would cause that only datasets are combined,
that were created by the same
preprocessing with the same parameterization.

(*optional, default: []*)

Exemplary Call
++++++++++++++

.. code-block:: yaml

    type: shuffle
    input_path: "operation_results/2009_8_13_15_8_57"
    dataset_constraints:
      # Combine only datasets that have been created using the same parameterization
      - '"%(dataset_name1)s".strip("}{").split("}{")[1:] == "%(dataset_name2)s".strip("}{").split("}{")[1:]' 
  
"""
import sys
import os
import glob
import time
import yaml

if sys.version_info[0] == 2 and sys.version_info[1] < 6:
    import processing
else:
    import multiprocessing as processing

import logging

import pySPACE
from pySPACE.missions.operations.base import Operation, Process
from pySPACE.tools.filesystem import create_directory
from pySPACE.resources.dataset_defs.base import BaseDataset
from pySPACE.tools.filesystem import get_author


class ShuffleOperation(Operation):
    """ Forwards processing to process

    .. todo:: Precalculate one process for each shuffling
    """
    def __init__(self, processes, operation_spec, result_directory,
                 number_processes, create_process=None):
        super(ShuffleOperation, self).__init__(processes, operation_spec,
                                               result_directory)
        self.number_processes = number_processes
        self.create_process = create_process
        
    @classmethod
    def create(cls, operation_spec, result_directory, debug = False, input_paths=[]):
        """ Factory method that creates a ShuffleOperation
        
        A factory method that creates a ShuffleOperation based on the 
        information given in the operation specification operation_spec
        """
        assert(operation_spec["type"] == "shuffle")

        # Determine constraints on datasets that are combined
        dataset_constraints = []
        if "dataset_constraints" in operation_spec:
            dataset_constraints.extend(operation_spec["dataset_constraints"])

        # Create the ShuffleProcess (shuffling is not distributed over different
        # processes)
        number_processes = 1
        processes = processing.Queue()
        cls._createProcesses(processes, result_directory, input_paths,
                                 dataset_constraints)           
        # create and return the shuffle operation object
        return cls(processes, operation_spec, result_directory, number_processes)
        
    @classmethod
    def _createProcesses(cls, processes, result_directory, input_datasets,
                          dataset_constraints):
        """Function that creates the shuffle process.
        
        Create the ShuffleProcess (shuffling is not distributed over different
        processes)
        """
        
        # Create the shuffle process and put it in the execution queue
        processes.put(ShuffleProcess(input_datasets, result_directory,
                                    dataset_constraints))
        # give executing process the sign that creation is now finished
        processes.put(False)
    
    def consolidate(self):
        """ Consolidation of the operation's results """
        # Just do nothing
        pass
    
    
class ShuffleProcess(Process):
    """ The shuffle process
    
    Combines datasets that fulfill all *dataset_constraints*
    """
    
    def __init__(self, input_dataset, result_directory, dataset_constraints):
        super(ShuffleProcess, self).__init__()
        
        self.input_datasets = input_dataset
        self.result_directory = result_directory
        self.dataset_constraints = dataset_constraints
        
    
    def __call__(self):
        """ Executes this process on the respective modality """
        ############## Prepare benchmarking ##############
        super(ShuffleProcess, self).pre_benchmarking()
        
        for dataset_dir1 in self.input_datasets:                
            for dataset_dir2 in self.input_datasets:
                dataset_name1 = dataset_dir1.split(os.sep)[-2]
                dataset_name2 = dataset_dir2.split(os.sep)[-2]
                
                # Check if the input data is split
                splitted = len(glob.glob(os.sep.join([dataset_dir1, "data_run0",
                                                      "*"]))) > 1
                
                # Check that all constraints are fulfilled for this pair of
                # input datasets
                if not all(eval(constraint_template % {'dataset_name1': dataset_name1,
                                                       'dataset_name2': dataset_name2})
                                    for constraint_template in self.dataset_constraints):
                    continue
                
                if dataset_name1 == dataset_name2:
                    if splitted:
                        # Copy the data 
                        os.symlink(dataset_dir1,
                                   os.sep.join([self.result_directory, 
                                                dataset_name1]))
                    continue
             
                # Determine names of the original data sets the input 
                # datasets are based on
                base_dataset1 = dataset_name1.strip("}{").split("}{")[0]
                base_dataset2 = dataset_name2.strip("}{").split("}{")[0]
                
                # Determine target dataset name and create directory
                # for it
                mixed_base_dataset = "%s_vs_%s" % (base_dataset1, 
                                                      base_dataset2)
                target_dataset_name = dataset_name1.replace(base_dataset1,
                                                                  mixed_base_dataset)
                
                target_dataset_dir = os.sep.join([self.result_directory,
                                                     target_dataset_name])
                
                create_directory(os.sep.join([target_dataset_dir, "data_run0"]))
                
                if splitted:
                    # For each split, copy the train data from dataset 1 and
                    # the test data from dataset 2 to the target dataset
                    for source_train_file_name in glob.glob(os.sep.join([dataset_dir1,
                                                                       "data_run0",
                                                                       "*_sp*_train.*"])):
                        # TODO: We have $n$ train sets and $n$ test sets, we                   "metadata.yaml"])),
                              
                        #       could use all $n*n$ combinations 
                        target_train_file_name = source_train_file_name.replace(dataset_dir1,
                                                                                target_dataset_dir)
                        if source_train_file_name.endswith("arff"):
                            self._copy_arff_file(source_train_file_name, 
                                                 target_train_file_name,
                                                 base_dataset1,
                                                 mixed_base_dataset)
                        else:
                            os.symlink(source_train_file_name, 
                                       target_train_file_name)
                        
                        source_test_file_name = source_train_file_name.replace(dataset_dir1,
                                                                               dataset_dir2)
                        
                        source_test_file_name =  source_test_file_name.replace("train.",
                                                                                "test.")
                        target_test_file_name = target_train_file_name.replace("train.",
                                                                                "test.")
                        if source_train_file_name.endswith("arff"):
                            self._copy_arff_file(source_test_file_name, 
                                                 target_test_file_name,
                                                 base_dataset2,
                                                 mixed_base_dataset)
                        else:
                            os.symlink(source_test_file_name,
                                       target_test_file_name)
                else:
                    # Use the data set from dataset 1 as training set and 
                    # the data set from dataset 2 as test data
                    for source_train_file_name in glob.glob(os.sep.join([dataset_dir1,
                                                                         "data_run0",
                                                                         "*_sp*_test.*"])):
                        target_train_file_name = source_train_file_name.replace("test.",
                                                                                "train.")
                        target_train_file_name = target_train_file_name.replace(dataset_dir1,
                                                                                target_dataset_dir)
                        if source_train_file_name.endswith("arff"):
                            self._copy_arff_file(source_train_file_name, 
                                                 target_train_file_name,
                                                 base_dataset1,
                                                 mixed_base_dataset)
                        else:
                            os.symlink(source_train_file_name, 
                                       target_train_file_name)
                        
                        source_test_file_name = source_train_file_name.replace(dataset_dir1,
                                                                               dataset_dir2)
                        
                        target_test_file_name = target_train_file_name.replace("train.",
                                                                                "test.")
                        if source_train_file_name.endswith("arff"):
                            self._copy_arff_file(source_test_file_name, 
                                                 target_test_file_name,
                                                 base_dataset2,
                                                 mixed_base_dataset)
                        else:
                            os.symlink(source_test_file_name,
                                       target_test_file_name)
                # Write metadata.yaml based on input meta data
                input_dataset1_meta = BaseDataset.load_meta_data(dataset_dir1)

                output_dataset_meta = dict(input_dataset1_meta)
                output_dataset_meta['train_test'] = True
                output_dataset_meta['date'] = time.strftime("%Y%m%d_%H_%M_%S")
                output_dataset_meta['author'] = get_author()
                BaseDataset.store_meta_data(target_dataset_dir,output_dataset_meta)
        
        ############## Clean up after benchmarking ##############
        super(ShuffleProcess, self).post_benchmarking()
        

    def _copy_arff_file(self, input_arff_file_name, target_arff_file_name,
                        input_dataset_name, target_dataset_name):
        """ Copy the arff files and adjust the relation name in the arff file"""
        file = open(input_arff_file_name, 'r')
        content = file.readlines()
        file.close()
        content[0] = content[0].replace(input_dataset_name, 
                                        target_dataset_name) 
        file = open(target_arff_file_name, 'w')
        file.writelines(content)
        file.close()
        
