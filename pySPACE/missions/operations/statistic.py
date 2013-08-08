""" Calculate a two-tailed paired t-test on a result collection for a certain parameter

Furthermore some statistical important values are calculated.

Specification file Parameters
+++++++++++++++++++++++++++++

type
----

Should be *statistic*

(*obligatory, statistic*)

metric
------

list of function values on which we want to make the test

(*optional, default: 'Balanced_accuracy'*)

parameter
---------

name of the varying parameter, we want to analyze

(*optional, default: '__Dataset__'*)

filter
------

dictionary saying which subarray of the csv tabular shall be analyzed

average
-------

parameter, over which one should average

(*optional, default: None*)

input_collection
----------------

Path to the input collection of type 'result'

related_parameters
------------------

list of parameters, being relevant for the related t-test

(*optional, default: ["__Dataset__", Key_Run, Key_Fold]*)

Exemplary Call
++++++++++++++
    
.. code-block:: yaml

        type : statistic
        input_path : "result_col_example"
        metric : "Balanced_accuracy"
        parameter : '__metric__'
        related_parameters : ["__Dataset__", "Key_Run", "Key_Fold"]
        average : "Key_Run"
        filter : {"__metric__":["Balanced_accuracy","k_Balanced_accuracy","soft_Balanced_accuracy"]}
        
.. todo:: Anett says a unit test should check if the statistical calculations work as expected

"""
import sys
if sys.version_info[0] == 2 and sys.version_info[1] < 6:
    import processing
else:
    import multiprocessing as processing
from scipy.stats import ttest_rel as ttest_related
from scipy.stats import kstest
import numpy as np
import os

import logging
import pySPACE
from pySPACE.missions.operations.base import Operation, Process
from pySPACE.resources.dataset_defs.base import BaseDataset

import pySPACE.tools.csv_analysis as csv_analysis

import warnings

class StatisticOperation(Operation):
    """ Start only the one StatisticProcess after reading the specification file
    and reducing the performance tabular to the relevant entries
    
    For further calculations, the performance tabular and its metadata.yaml file
    are copied into the new collection such that other operations can follow,
    as for example visualization operations.
    """
    def __init__(self, processes, operation_spec, result_directory, 
                 number_processes, create_process=None):
        super(StatisticOperation, self).__init__(processes, operation_spec,
                                               result_directory)
        self.number_processes = number_processes
        self.create_process   = create_process
        warnings.warn("You are using the statistic operation to calculate p-values for the paired t-test. \
            Check if the paired t-test is the correct method in your model! \
            This operation shall help to find important parameters but does not \
            replace a good statistical model.")
        
    @classmethod
    def create(cls, operation_spec, result_directory, debug=False, input_paths=[]):
        """
        A factory method that creates a statistic operation based on the
        information given in the operation specification operation_spec.
        If debug is TRUE the creation of the statistic processes will not
        be in a separated thread.
        """
        assert(operation_spec["type"] == "statistic")
        input_path = operation_spec["input_path"]
        tabular = BaseDataset.load(os.path.join(pySPACE.configuration.storage, input_path)).data
        
        if operation_spec.has_key("filter"):
            conditions= csv_analysis.empty_dict(tabular)
            for key,l in operation_spec["filter"].items():
                conditions[key].extend(l)
            tabular = csv_analysis.strip_dict(tabular,conditions)
        metric = operation_spec.get("metric","Balanced_accuracy")
        parameter = operation_spec.get("parameter","__Dataset__")
        rel_par = operation_spec.get("related_parameters",["__Dataset__", "Key_Run", "Key_Fold"])
        average = operation_spec.get("average",None)
        
        if average in rel_par:
            rel_par.remove(average)
        if metric in rel_par:
            rel_par.remove(metric)
        if parameter in rel_par:
            rel_par.remove(parameter)
            
        reduced_tabular=cls.reduce_tabular(tabular,rel_par,metric,parameter,average)
        number_processes = 1
        processes = processing.Queue()
        cls._createProcesses(processes, result_directory, reduced_tabular)
        
        import shutil
        shutil.copy2(os.path.join(pySPACE.configuration.storage, input_path,"results.csv"), os.path.join(result_directory,"results.csv"))
        shutil.copy2(os.path.join(pySPACE.configuration.storage, input_path,"metadata.yaml"), os.path.join(result_directory,"metadata.yaml"))
        # create and return the shuffle operation object
        return cls(processes, operation_spec, result_directory, number_processes)
    
    @classmethod
    def reduce_tabular(cls,tabular,rel_par,metric,parameter,average):
        keylist = []
        if average == None:
            for i in range(len(tabular[metric])):
                keylist.append((tabular[parameter][i],
                               tuple([tabular[par][i] for par in rel_par]),
                               tabular[metric][i]))
            values   = []
            par_list = []
            keylist.sort()
            # filter parameter and metric from sorted tabular
            for par, key, value in keylist:
                values.append(float(value))
                par_list.append(par)
        else:
            unique_average = sorted(list(set(tabular[average])))
            l=len(unique_average)
            for i in range(len(tabular[metric])):
                keylist.append((tabular[parameter][i],
                               tuple([tabular[par][i] for par in rel_par]),
                               tabular[average][i],
                               tabular[metric][i]))
            values   = []
            par_list = []
            keylist.sort()
            # filter parameter and metric from sorted tabular
            for par, key, p_average,value in keylist:
                if p_average == unique_average[0]:
                    v = float(value)
                    p = par
                    k = key
                    i = 1
                else:
                    v += float(value)
                    assert(p==par),"Wrong sorting in list."
                    assert(k==key),"Wrong sorting in list."
                    i += 1
                if p_average == unique_average[-1]:
                    values.append(v/l)
                    par_list.append(p)
                    assert(i==l),"Wrong sorting in list."
        return {"values": values, "parameters": par_list}
    
    @classmethod
    def _createProcesses(cls, processes, result_directory, data):
        """Function that creates the process.
        
        Create the Process (it is not distributed over different processes)
        """
        # Create the process and put it in the execution queue
        processes.put(StatisticProcess(result_directory, data))
        # give executing process the sign that creation is now finished
        processes.put(False)
    
    def consolidate(self):
        """ Consolidation of the operation's results """
        # Just do nothing
        pass
    
class StatisticProcess(Process):
    """ Calculate several statistic metrics on the specified metric and parameter
    
    At the moment mean, correlation, difference of means, standard deviation,
    standard error, p-value, t-value and some basic significance test
    are calculated and written to a tabular.
    """
    def __init__(self, result_directory, data):
        super(StatisticProcess, self).__init__()
        
        self.result_directory = result_directory
        self.data = data
        self.alpha = 0.05
        
    def __call__(self):
        """ Executes this process on the respective modality """
        ############## Prepare benchmarking ##############
        super(StatisticProcess, self).pre_benchmarking()
        
        unique_parameters = list(set(self.data["parameters"]))
        assert(len(unique_parameters)>1),"No different parameter given!"
        if len(unique_parameters)>2:
            self._log("No correction yet implemented for multiple t test.",logging.CRITICAL)
        
        n = len(unique_parameters)
        l = len(self.data["parameters"])
        k = l/n
        p_values = {"parameter_1":[],"parameter_2":[],"p_value":[],"t_value":[],
                    "correlation":[],"mean_1":[],"mean_2":[],"mean_1-mean_2":[],
                    "std_1":[],"std_2":[],"se_1":[],"se_2":[],"Bonferroni_significant":[],
                    "Sidac_significant":[]}
        for i in range(n):
            assert(self.data["parameters"][i*k:(i+1)*k]==[self.data["parameters"][i*k]]*k),\
                "Error in list sorting! Parameter not consistent. Problematic parameter: %s"%self.data["parameters"][i*k]
            data1 = self.data["values"][i*k:(i+1)*k]
            mean1 = np.mean(data1)
            std1 = np.std(data1)
            if not self.kstest((data1-mean1)/std1):
                self._log("Data is probably not normal distributed \
                    according to parameter %s \
                    and you do not use the proper statistic test.\
                    The parameter is ignored!"
                    %self.data["parameters"][i*k],logging.CRITICAL)
                continue
            for j in range(i):
                # check if parameters match in one part
                assert(len(self.data["parameters"][i*k:(i+1)*k]) == len(self.data["parameters"][j*k:(j+1)*k])),\
                    "Error in list sorting! Parameters are not equal."
                data2 = self.data["values"][j*k:(j+1)*k]
                mean2 = np.mean(data2)
                std2 = np.std(data2)
                if not self.kstest((data2-mean2)/std2):
                    self._log("Data is probably not normal distributed \
                    according to parameter %s \
                    and you do not use the proper statistic test.\
                    The parameter is ignored!"
                    %self.data["parameters"][j*k],logging.CRITICAL)
                    continue
                
                t,p = self.p_value(data1,data2)
                corr = np.corrcoef(data1,data2)[0][1]
                p_values["correlation"].append(corr)
                p_values["p_value"].append(p)
                
                test_sig = self.alpha/n
                if p < test_sig:
                    p_values["Bonferroni_significant"].append("True")
                else:
                    p_values["Bonferroni_significant"].append("False")
                test_sig = 1.0-(1.0-self.alpha)**(1.0/n)
                if p < test_sig:
                    p_values["Sidac_significant"].append("True")
                else:
                    p_values["Sidac_significant"].append("False")
                
                if mean1>mean2:
                    p_values["t_value"].append(t)
                    
                    p_values["parameter_1"].append(self.data["parameters"][i*k])
                    p_values["parameter_2"].append(self.data["parameters"][j*k])
                    p_values["mean_1"].append(mean1)
                    p_values["mean_2"].append(mean2)
                    p_values["mean_1-mean_2"].append(mean1-mean2)
                    p_values["std_1"].append(std1)
                    p_values["std_2"].append(std2)
                    p_values["se_1"].append(std1/np.sqrt(k))
                    p_values["se_2"].append(std2/np.sqrt(k))
                else:
                    p_values["t_value"].append(-t)
                    # construct symmetric result
                    p_values["parameter_2"].append(self.data["parameters"][i*k])
                    p_values["parameter_1"].append(self.data["parameters"][j*k])
                    p_values["mean_2"].append(mean1)
                    p_values["mean_1"].append(mean2)
                    p_values["mean_1-mean_2"].append(mean2-mean1)
                    p_values["std_2"].append(std1)
                    p_values["std_1"].append(std2)
                    p_values["se_2"].append(std1/np.sqrt(k))
                    p_values["se_1"].append(std2/np.sqrt(k))
                    
        csv_analysis.dict2csv(os.path.join(self.result_directory,'p_values.csv'), p_values)

        ############## Clean up after benchmarking ##############
        super(StatisticProcess, self).post_benchmarking()
    
    def p_value(self,p1_list,p2_list):
        t,p = ttest_related(p1_list,p2_list)
        return t,p
    
    def kstest(self,data):
        d,p=kstest(data,'norm')
        
        return p>0.05