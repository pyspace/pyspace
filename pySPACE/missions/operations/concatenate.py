""" Concatenate datasets of :mod:`time series data<pySPACE.resources.dataset_defs.time_series>`

This operation requires test data with no splits.

The result of this operation concatenates the datasets of the input.
For instance, if the input consists of the three datasets "A",
"B", "C", the result will contain only one dataset "All".

.. note:: Each dataset can only be used once for concatenation!

Specification file Parameters
++++++++++++++++++++++++++++++

type
----

This parameter has to be set to **concatenate**.

(*obligatory, concatenate*)


name_pattern
------------

The name
of the result dataset can be specified within *name_pattern*.
The first time series object of every concatenated set will contain a 'new_set'
flag in the specs to allow later reconstruction of the individual sets.

(*optional, default:'"%(dataset_name)s"[:-1]+"_All"'*)


dataset_constraints
----------------------

Optionally, constraints can be passed to the operation that specify which
datasets are concatenated. For instance, the constraint
'"%(dataset_name1)s".strip("}{").split("}{")[1:] ==
"%(dataset_name2)s".strip("}{").split("}{")[1:]' would cause
that only datasets are combined, that were created by the same processing
with the same parametrization.

.. todo:: Document the definition of dataset1 and is dataset2!

change_time
-----------

If *change_time* is True, the appended time series objects get a new, artificial
start and end time, to ensure that the time is unique for further investigations.

(*optional, default: False*)

Exemplary Call
++++++++++++++

A typical operation specification file might look like this

.. code-block:: yaml

    type: concatenate
    name_pattern: '"%(dataset_name)s"[:-1]'
    change_time: False
    input_path: "operation_results/2009_8_13_15_8_57"
    dataset_constraints:
      # Combine only datasets that have been created using the same parameterization
      - '"%(dataset_name1)s".strip("}{").split("}{")[1:] == "%(dataset_name2)s".strip("}{").split("}{")[1:]'

Example dataset_constraints
++++++++++++++++++++++++++++++

    :Combine only datasets that have been created using the same parameterization:
        ``- '"%(dataset_name1)s".strip("}{").split("}{")[1:] == "%(dataset_name2)s".strip("}{").split("}{")[1:]'``

Application Examples
++++++++++++++++++++

Run123 versus Run45
-------------------

The following example concatenates Runs 1, 2 and 3 from within the same Session 
of the same Subject to a joint "Run123". The similar is done for "Run45".

.. code-block:: yaml

    type: concatenate
    
    input_path: "prewindowed/BRIO_Oddball_5subjects_0_1000ms_Preprocessed"
    change_time: False
    name_pattern: '"%(dataset_name)s"[:-1] + ("123" if "%(dataset_name)s"[-1:] in ["1","2","3"] else "45")'
    dataset_constraints:
    - '"%(dataset_name1)s".strip("}{").split("_")[0] == "%(dataset_name2)s".strip("}{").split("_")[0]'
    - '"%(dataset_name1)s".strip("}{").split("_")[1] == "%(dataset_name2)s".strip("}{").split("_")[1]'
    - '(("%(dataset_name1)s".strip("}{").split("_")[2] == "Run1") and ("%(dataset_name2)s".strip("}{").split("_")[2] == "Run2" or "%(dataset_name2)s".strip("}{").split("_")[2] == "Run3")) or ("%(dataset_name1)s".strip("}{").split("_")[2] == "Run4" and "%(dataset_name2)s".strip("}{").split("_")[2] == "Run5")'

In the following shuffle example, the Runs called "Run123" 
will be used for training, and the runs called "Run45" from the same subject 
and session will be used for test:

.. code-block:: yaml

    type: shuffle
    
    input_path: "prewindowed/BRIO_Oddball_5subjects_0_1000ms_Preprocessed_Run123_Run45"
    change_time: False
    dataset_constraints:
    - '"%(dataset_name1)s".strip("}{").split("_")[0] == "%(dataset_name2)s".strip("}{").split("_")[0]'
    - '"%(dataset_name1)s".strip("}{").split("_")[1] == "%(dataset_name2)s".strip("}{").split("_")[1]'
    - '"%(dataset_name1)s".strip("}{").split("_")[2] == "Run123"'
    - '"%(dataset_name2)s".strip("}{").split("_")[2] == "Run45"'

For the usage o the shuffle operation refer to :mod:`pySPACE.missions.operations.shuffle`.

.. note::
    Problems in connection with the
    :class:`~pySPACE.missions.nodes.source.time_series_source.TimeSeries2TimeSeriesSourceNode`
    can also occur as described in the
    :mod:`~pySPACE.missions.operations.merge` module.

:Author: Anett Seeland (anett.seeland@dfki.de)
:Input: :mod:`pySPACE.pySPACE.resources.dataset_defs.time_series`
"""

import os
import sys
import getpass
import glob
import time
if sys.version_info[0] == 2 and sys.version_info[1] < 6:
    import processing
else:
    import multiprocessing as processing

import pySPACE
from pySPACE.missions.operations.base import Operation, Process
from pySPACE.tools.filesystem import create_directory

from pySPACE.resources.dataset_defs.base import BaseDataset
    
class ConcatenateOperation(Operation):
    """ Concatenate operation for creating 'All' datasets """
    def __init__(self, processes, operation_spec, result_directory,
                 number_processes, create_process=None):
        super(ConcatenateOperation, self).__init__(processes, operation_spec,
                                                   result_directory)
        self.number_processes = number_processes
        self.create_process = create_process
        
    @classmethod
    def create(cls, operation_spec, result_directory, debug=False, input_paths=[]):
        """ [Factory method] Create a ConcatenateOperation
        
        A factory method that creates a ConcatenateOperation based on the 
        information given in the operation specification operation_spec
        """
        assert(operation_spec["type"] == "concatenate")

        # Determine constraints for datasets that are combined and other 
        # parameters
        dataset_constraints = []
        if "dataset_constraints" in operation_spec:
            dataset_constraints.extend(operation_spec["dataset_constraints"])
        if "name_pattern" in operation_spec:
            name_pattern = operation_spec["name_pattern"]
        else:
            name_pattern = None
        if "change_time" in operation_spec:
            change_time = operation_spec["change_time"]
        else:
            change_time = False

        # merging is not distributed over different processes
        number_processes = 1
        processes = processing.Queue()
        
        # Create the Concatenate Process 
        cls._createProcesses(processes, input_paths, result_directory,
                             dataset_constraints, name_pattern, change_time)
                           
        # create and return the Concatenate operation object
        return cls(processes, operation_spec, result_directory, number_processes)

    @classmethod
    def _createProcesses(cls, processes, input_datasets, result_directory,
                         dataset_constraints, name_pattern, change_time):
        """[Factory method] Create the Concatenate process. """
        
        # Create the Concatenate process and put it in the execution queue
        processes.put(ConcatenateProcess(input_datasets, result_directory,
                             dataset_constraints, name_pattern, change_time))
        # give executing process the sign that creation is now finished
        processes.put(False)    
    
    def consolidate(self):
        """ Consolidation of the operation's results """
        # Just do nothing
        pass                    
    
    
class ConcatenateProcess(Process):
    """ Create 'All' datasets where 'All' are all datasets that fulfill the *dataset_constraints* """
    
    def __init__(self, input_dataset, result_directory, 
                 dataset_constraints, name_pattern, change_time):
        super(ConcatenateProcess, self).__init__()
        
        self.input_datasets = input_dataset
        self.result_directory = result_directory
        self.dataset_constraints = dataset_constraints
        self.name_pattern = name_pattern
        self.change_time = change_time
        
    
    def __call__(self):
        """ Executes this process on the respective modality """
        ############## Prepare benchmarking ##############
        super(ConcatenateProcess, self).pre_benchmarking()
        
        # remember what has already been merged
        merged_dataset_pathes = []
        
        # For all input datasets
        for source_dataset_path1 in self.input_datasets:
            if source_dataset_path1 in merged_dataset_pathes:
                continue
            # At the moment split data is not supported, so there should be only
            # a single test file is in the source directory 
            source_files = glob.glob(os.sep.join([source_dataset_path1,
                                                  "data_run0", "*test*"]))
            
            source_pathes = []
                       
            is_split = len(source_files) > 1
            assert(not is_split),"Multiple test splits as in %s \
                                    are not yet supported."%str(source_files)
           
            # At the moment train data is not supported, so check if train sets
            # are also present
            train_data_present = len(glob.glob(os.sep.join(
                                 [source_dataset_path1,"data_run0",\
                                  "*train*"]))) > 0
            
            assert(not train_data_present),"Using training data is not yet implemented."
            
            # We create the "All" dataset
            source_dataset_name1 = source_dataset_path1.split(os.sep)[-2]
            base_dataset_name = \
                               source_dataset_name1.strip("}{").split("}{")[0]
            if self.name_pattern != None:
                target_dataset_name = source_dataset_name1.replace(
                                    base_dataset_name, eval(self.name_pattern % \
                                     {"dataset_name" : base_dataset_name}))
            else:
                target_dataset_name = source_dataset_name1.replace(
                         base_dataset_name, base_dataset_name[:-1]+"_all")
                
            source_pathes.append(source_dataset_path1)            
            target_dataset_path = os.sep.join([self.result_directory,
                                                  target_dataset_name])    
            
            for source_dataset_path2 in self.input_datasets:
                source_dataset_name2 = source_dataset_path2.split(os.sep)[-2]
                # Do not use data we have already in the source_path list
                if (source_dataset_path2 == source_dataset_path1) or (source_dataset_path2 in merged_dataset_pathes):
                    continue
                
                # Check that all constraints are fulfilled for this pair of
                # input datasets
                if not all(eval(constraint_template % \
                                {'dataset_name1': source_dataset_name1,
                                 'dataset_name2': source_dataset_name2})
                                for constraint_template in self.dataset_constraints):
                    continue
                    
                source_pathes.append(source_dataset_path2)
                merged_dataset_pathes.append(source_dataset_path1)
                merged_dataset_pathes.append(source_dataset_path2)
            
            create_directory(os.sep.join([target_dataset_path, "data_run0"]))
            
            self._merge_pickle_files(target_dataset_path, source_pathes)
          
        ############## Clean up after benchmarking ##############
        super(ConcatenateProcess, self).post_benchmarking()
        
    def _merge_pickle_files(self, target_dataset_path, source_dataset_pathes):
        """ Concatenate all datasets in source_dataset_pathes and store 
            them in the target dataset"""
        # sort the dataset 
        source_dataset_pathes.sort()
        # load a first dataset, in which the data of all other datasets is assembled
        target_dataset = BaseDataset.load(source_dataset_pathes[0])
        
        # Determine author and date
        try:
            author = getpass.getuser()
        except : 
            author = "Unknown"
        date = time.strftime("%Y%m%d_%H_%M_%S")
        # Delete node_chain file name
        try:
            target_dataset.meta_data.pop("node_chain_file_name")
        except:
            pass
        # Update meta data and store it
        params = target_dataset.meta_data.pop("parameter_setting")
        params["__INPUT_DATASET__"] = \
                 [s_c_p.split(os.sep)[-2] for s_c_p in source_dataset_pathes]
        params["__RESULT_DIRECTORY__"] = self.result_directory
        target_dataset.meta_data.update({"author" : author, 
                      "date" : date, 
                      "dataset_directory" : target_dataset_path,
                      "train_test" : False,
                      "parameter_setting" : params,
                      "changed_time" : self.change_time,
                      "input_dataset_name" : source_dataset_pathes[0][len(
                                        pySPACE.configuration.storage):]
        })
    
        # Concatenate data of all other datasets to target dataset
        for source_dataset_path in source_dataset_pathes[1:]:
            source_dataset = BaseDataset.load(source_dataset_path)
            for run in source_dataset.get_run_numbers():
                for split in source_dataset.get_split_numbers():
                    target_data = target_dataset.get_data(run, split, "test")

                    if self.change_time:
                        # ensure sorted target_data 
                        # TODO: encode this in meta data?  
                        target_data.sort(key=lambda t: t[0].end_time)
                        last_end_time = target_data[-1][0].end_time

                    for ts, l in target_data:
                        if ts.specs == None:
                            ts.specs = {"new_set": False}
                        elif ts.specs.has_key("new_set"):
                            break
                        else:
                            ts.specs["new_set"]= False

                    data = source_dataset.get_data(run, split, "test")

                    if self.change_time:                    
                        # ensure sorted target_data 
                        # TODO: encode this in meta data?
                        data.sort(key=lambda t: t[0].end_time)
                    # flag the first element of the concatenated data list
                    for i, (ts, l) in enumerate(data):
                        if ts.specs == None:
                            ts.specs = {"new_set": i==0}
                        else:
                            ts.specs["new_set"] = (i==0)
                        if self.change_time:
                            ts.start_time = last_end_time + ts.start_time
                            ts.end_time = last_end_time + ts.end_time
                            
                    # actual data is stored in a list that has to be extended
                    target_data.extend(data)
                
        target_dataset.store(target_dataset_path)

