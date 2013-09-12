""" Tabular listing data sets, parameters and a huge number of performance metrics

Store and load the performance results of an operation from a csv file,
select subsets of this results or for create various kinds of plots

**Special Static Methods**
    :merge_performance_results:
        Merge result*.csv files when classification fails or is aborted.

    :repair_csv:
        Wrapper function for whole csv repair process when classification
        fails or is aborted.
"""
from itertools import cycle

try: # import packages for plotting
    import pylab
    import matplotlib.pyplot
    import matplotlib
except:
    pass

try: # import packages for plotting error bars
    import scipy.stats
except:
    pass

from collections import defaultdict

import numpy
import os
import glob

# imports for storing
import pwd
import yaml

import warnings
import logging

# csv handling
import pySPACE.tools.csv_analysis as csv_analysis
# base class
from pySPACE.resources.dataset_defs.base import BaseDataset


# roc imports
import cPickle # load roc points
from operator import itemgetter

class PerformanceResultSummary(BaseDataset):
    """ Classification performance results summary
    
    For the identifiers some syntax rules hold to make some distinction:
    
        1.  Parameters/Variables start and end with `__`.
            These identifiers define the processing differences of the entries.
            Altogether the corresponding values build a unique key of each row.
        2.  Normal metrics start with a Big letter and 
            continue normally with small letters except AUC.
        3.  Meta metrics like training metrics, LOO  metrics or soft metrics
            start with small letters defining the category followed by a
            `-` and continue with the detailed metric name.
        4.  Meta information like chosen optimal parameters can be
            separated from metrics and variables using `~~`
            at beginning and end of the information name.
    
    This class can load a result tabular (namely the results.csv file) using
    the factory method :func:`from_csv`.
    
    Furthermore, the method :func:`project_onto` allows to select a subset of the
    result collection where a parameter takes on a certain value.
    
    The class contains various methods for plotting the loaded results. 
    These functions are used by the analysis operation and by the interactive
    analysis GUI.
    
    Mainly result collections are loaded for
    :mod:`~pySPACE.missions.operations.comp_analysis`,
    :mod:`~pySPACE.missions.operations.analysis` and
    as best alternative with the :mod:`~pySPACE.run.gui.performance_results_analysis`.
    
    They can be build e.g. with the :mod:`~pySPACE.missions.nodes.sink.classification_performance_sink` nodes,
    with :ref:`MMLF <tutorial_interface_to_mmlf>` or with
    :class:`~pySPACE.missions.operations.weka_classification.WekaClassificationOperation`.

    The metrics as result of :mod:`~pySPACE.missions.nodes.sink.classification_performance_sink` nodes
    are calculated in the :mod:`~pySPACE.resources.dataset_defs.metric` dataset module.

    .. todo:: Access in result collection via indexing ndarray with one
              dimension for each parameter.
              Entries are indexes in list. So the corresponding values
              can be accessed very fast.

    .. todo:: Faster, memory efficient loading is needed. Pickling or new data
              structure?
    
    The class constructor expects the following **arguments**:
    
      :data:    A dictionary that contains a mapping from an attribute
                (e.g. accuracy) to a list of values taken by this attribute.
                An entry is the entirety of all i-th values over all dict-values

      :tmp_pathlist:
          List of files to be deleted after successful storing

          When constructed via `from_multiple_csv` all included csv files
          can be deleted after the collection is stored.
          Therefore the parameter `delete` has to be active.
          
          (*optional, default:None*)
    
      :delete:
          Switch for deleting files in `tmp_pathlist` after collection is stored.
          
          (*optional, default: False*)
    
    :Author: Mario M. Krell (mario.krell@dfki.de)
    """
        
    def __init__(self, data=None, dataset_md=None, dataset_dir=None,
                 csv_filename=None, **kwargs):
        super(PerformanceResultSummary, self).__init__()
        if csv_filename and not dataset_dir: # csv_filename is expected to be a path
            dataset_dir=""
        self.delete = False
        self.tmp_pathlist = None
        
        if dataset_md != None:
            self.meta_data.update(dataset_md)

        if data != None:
            self.data = data
        elif dataset_dir != None: # load data
            if csv_filename != None:
                # maybe it's not results.csv but it's definitely only one file
                self.data = PerformanceResultSummary.from_csv(os.path.join(dataset_dir,
                                                                   csv_filename))
            elif os.path.isfile(os.path.join(dataset_dir,"results.csv")):
                # delegate to from_csv_method
                csv_file_path = os.path.join(dataset_dir,"results.csv")
                self.data = PerformanceResultSummary.from_csv(csv_file_path)
            else: # multiple csv_files
                self.data, self.tmp_pathlist = \
                        PerformanceResultSummary.from_multiple_csv(dataset_dir)
                self.delete = True
                # update meta data 
                try:
                    splits = max(map(int,self.data["__Key_Fold__"]))
                    runs = max(map(int,self.data["__Key_Run__"]))+1
                except:
                    warnings.warn('Splits and runs not available!')
                else:
                    self.meta_data.update({"splits": splits, "runs": runs})
        else: # we have a problem
            self._log("Result tabular could not be created - data is missing!",
                      level=logging.CRITICAL)
            warnings.warn("Result tabular could not be created - data is missing!")
            self.data = {}
        # modifier for getting general box plots in Gui
        if not self.data.has_key('None'):
            self.data['None'] = ['All'] * len(self.data.values()[0])
        self.identifiers = self.data.keys()

        # indexed version of the data
        self.data_dict = None
        self.transform()
        
    @staticmethod
    def from_csv(csv_file_path):
        """ Loading data from the csv file located under *csv_file_path* """
#        # pickle loading
#        try:
#            if csv_file_path.endswith("pickle"):
#                f = open(csv_file_path, "rb")
#            elif csv_file_path.endswith("csv"):
#                f = open(csv_file_path[:-3] + "pickle", 'rb')
#            res=cPickle.load(f)
#            f.close()
#            return res
#        except IOError:
#            pass
        data_dict = csv_analysis.csv2dict(csv_file_path)
        PerformanceResultSummary.translate_weka_key_schemes(data_dict)
#        # save better csv version
#        f = open(csv_file_path[:-3] + "pickle", "wb")
#        f.write(cPickle.dumps(res, protocol=2))
#        f.close()
        return data_dict
    
    @staticmethod
    def from_multiple_csv(input_dir):
        """ All csv files in the only function parameter 'input_dir' are 
        combined to just one result collection 
        
        Deleting of files will be done in the store method, *after*
        the result is stored successfully.
        """
        # A list of all result files (one per classification process)
        pathlist = glob.glob(os.path.join(input_dir,
                                          "results_*"))
        if len(pathlist)==0:
            warnings.warn('No files in the format "results_*" found for merging results!')
            return
        result_dict = None
        # For all result files of the WEKA processes
        for input_file_name in pathlist:
            # first occurrence
            if result_dict is None:
                result_dict = csv_analysis.csv2dict(input_file_name)
            else:
                result = csv_analysis.csv2dict(input_file_name)
                csv_analysis.extend_dict(result_dict,result)
        
        PerformanceResultSummary.translate_weka_key_schemes(result_dict)
        PerformanceResultSummary.tranfer_Key_Dataset_to_parameters(result_dict)
        
        return (result_dict, pathlist)
    
    def transform(self):
        """ Fix format problems like floats in metric columns and tuples instead of column lists """
        for key in self.get_metrics():
            if not type(self.data[key][0])==float:
                try:
                    l = [float(value) for value in self.data[key]]
                    self.data[key] = l
                except:
                    warnings.warn("Metric %s has entry %s not of type float."%(
                    key,str(value)
                    ))
        for key in self.identifiers:
            if not type(self.data[key])==tuple:
                self.data[key] = tuple(self.data[key])
    
    @staticmethod
    def merge_traces(input_dir):
        """ Traverse directory tree, merge the classification trace files and store them
        
        The collected results are stored in a common file in the input *input_dir*.
        """
        import cPickle
        traces = dict()
        long_traces = dict()
        save_long_traces = True
        sorted_keys = None
        # save merged files to delete them later
        merged_files=[]
        for dir_path,dir_names,files in os.walk(input_dir):
            for filename in files:
                if filename.startswith("trace_sp"):
                    pass
                else:
                    continue
                main_directory = dir_path.split(os.sep)[-3]
                # needed in tranfer_Key_Dataset_to_parameters
                temp_key_dict = defaultdict(list)
                # add a temporal Key_Dataset, deleted in next step 
                temp_key_dict["Key_Dataset"] = [main_directory]
                # read parameters from key dataset
                PerformanceResultSummary.tranfer_Key_Dataset_to_parameters(temp_key_dict)
                key_dict=dict([(key,value[0]) for key,value in temp_key_dict.items()])
                # add run/split identifiers
                split_number = int(filename[8:-7]) # from trace_spX.pickle
                key_dict["__Key_Fold__"] = split_number
                run_number = int(dir_path.split(os.sep)[-2][15:]) # from persistency_runX
                key_dict["__Key_Run__"] = run_number
                # transfer keys to hashable tuple of values
                # the keys should always be the same
                if sorted_keys is None:
                    sorted_keys = sorted(key_dict.keys())
                    traces["parameter_keys"]=sorted_keys
                    long_traces["parameter_keys"]=sorted_keys
                identifier=[]
                for key in sorted_keys:
                    identifier.append(key_dict[key])
                # load the actual classification trace
                trace = cPickle.load(open(dir_path + os.sep + filename, 'rb'))
                traces[tuple(identifier)] = trace
                merged_files.append(dir_path + os.sep + filename)
                if save_long_traces:
                    try:
                        trace = cPickle.load(open(dir_path + os.sep +"long_"+ filename, 'rb'))
                        long_traces[tuple(identifier)] = trace
                        merged_files.append(dir_path + os.sep +"long_"+ filename)
                    except IOError:
                        save_long_traces = False
                
        # clean up
        if not sorted_keys is None:
            name = 'traces.pickle'
            result_file = open(os.path.join(input_dir, name), "wb")
            result_file.write(cPickle.dumps(traces, protocol=2))
            result_file.close()
            if save_long_traces:
                name = 'long_traces.pickle'
                result_file = open(os.path.join(input_dir, name), "wb")
                result_file.write(cPickle.dumps(long_traces, protocol=2))
                result_file.close()
            for temp_file in merged_files:
                os.remove(temp_file)
    
    @staticmethod
    def translate_weka_key_schemes(data_dict):
        """ Data dict is initialized as 'defaultdict(list)' and
        so the append function will work on non existing keys.
        """
        if not data_dict.has_key("Key_Scheme"):
            return
        
        for i,value in data_dict["Key_scheme"].iter():
                # Some special cases
                # For these cases we rewrite the value to be meaningful
                # Important parts of "Key_Scheme_Options" will be added to "Key_Scheme"
                # Furthermore we introduce numerous new variables to benchmark
                value = value.split(".")[-1]
                if value == "SMO":
                    options = data_dict["Key_Scheme_options"][i]
                    options = options.split()
                    data_dict["__Classifier_Type__"].append(value)
                    for token in options:
                        # Search kernel type
                        if token.count("supportVector") >=1:
                            kernel_type = token.split(".")[-1]
                            data_dict["Kernel_Type"].append(kernel_type)
                            break
                        # Search complexity
                    for index, token in enumerate(options):
                        if token.count("-C") >=1:
                            complexity = options[index + 1]
                            data_dict["__Complexity__"].append(complexity)
                            # Add to value the complexity
                            value += " C=%s"
                            break
                    if kernel_type == 'PolyKernel':
                        # Search exponent in options of PolyKernel
                        exponent = options[options.index("-E") + 1]
                        if "\\" in exponent:
                            exponent = exponent.split("\\")[0]
                            #Add Kernel Type and Exponent to value
                        data_dict["__Kernel_Exponent__"].append(exponent)
                        if not exponent == "0":
                            value += " %s Exp=%s" % (kernel_type, exponent)
                        else:
                            value += " linear"
                        # unimportant parameter
                        data_dict["__Kernel_Gamma__"].append(0.0)
                    elif kernel_type == 'RBFKernel':
                        # Search gamma in options of RBFKernel
                        gamma = options[options.index("-G") + 1]
                        if "\\" in gamma:
                            gamma = gamma.split("\\")[0]
                        data_dict["__Kernel_Gamma__"].append(gamma)
                        value += " %s G=%s" % (kernel_type, gamma)
                        # unimportant parameter
                        data_dict["__Kernel_Exponent__"].append(0.0)
                    else:
                        #TODO: Warning: unknown kernel
                        data_dict["__Kernel_Exponent__"].append(0.0)
                        data_dict["__Kernel_Gamma__"].append(0.0)
                    # parameters used additionally in libsvm
                    data_dict["__Kernel_Offset__"].append(0.0)
                    data_dict["__Kernel_Weight__"].append(0.0)
                
                # LibSVM works the same way as SMO and comes with WEKA.
                # For NodeChainOperations a better version is integrated in C++
                # It has more options, especially to weight the classes, to make oversampling unnecessary
                # When using nonlinear kernels,
                # one should consider the influence of the offset and for polynomial k. the scaling factor gamma.
                elif value == "LibSVM":
                    options = data_dict["Key_Scheme_options"][i]
                    weight = options.split("-W")[-1]
                    options = options.split()
                    for index, token in enumerate(options):
                        if token.count("-S") >=1:
                            # 0 -- C-SVC
                            # 1 -- nu-SVC
                            # 2 -- one-class SVM
                            # 3 -- epsilon-SVR
                            # 4 -- nu-SVR
                            classifier = options[index + 1]
                            if classifier == "0":
                                classifier ="C_CVC"
                            data_dict["__Classifier_Type__"].append(classifier)
                            value += " %s" % (classifier)
                        elif token.count("-K") >=1:
                            # 0 -- linear: u'*v
                            # 1 -- polynomial: (gamma*u'*v + coef0)^degree
                            # 2 -- radial basis function: exp(-gamma*|u-v|^2)
                            # 3 -- sigmoid: tanh(gamma*u'*v + coef0)
                            kernel = options[index + 1]
                            if kernel  == "0":
                                kernel = "linear"
                            elif kernel == "1":
                                kernel = "polynomial"
                            elif kernel == "2":
                                kernel = "RBF"
                            elif kernel == "3":
                                kernel = "sigmoid"
                            data_dict["__Kernel_Type__"].append(kernel)
                            value += " %s" % (kernel)
                        elif token.count("-C") >=1:
                            complexity = options[index + 1]
                            data_dict["__Complexity__"].append(complexity)
                            value += " C=%s" % (complexity)
                        elif token.count("-D") >=1:
                            degree = options[index + 1]
                            data_dict["__Kernel_Exponent__"].append(degree)
                            if not degree == "0":
                                value += " Exp=%s" % (degree)
                        elif token.count("-G") >=1:
                            gamma = options[index + 1]
                            data_dict["__Kernel_Gamma__"].append(gamma)
                            if not gamma == "0.0":
                                value += " G=%s" % (gamma)
                        elif token.count("-R") >=1:
                            coef0 = options[index + 1]
                            data_dict["__Kernel_Offset__"].append(coef0)
                            if not coef0 == "0.0":
                                value += " c0=%s" % (coef0)
                        elif token.count("W")>=1:
                            if "\\" in weight:
                                weight = weight.split("\\\"")[1]
                            data_dict["__Kernel_Weight__"].append(weight)
                            if not weight == "1.0 1.0":
                                value += " W=%s" % (weight)
                else:
                    # TODO: Warning: unknown classifier
                    # All parameters of the two integrated classifier to make analysis operation compatible with other classifiers
                    data_dict["__Kernel_Type__"].append(value)
                    data_dict["__Complexity__"].append(0.0)
                    data_dict["__Kernel_Exponent__"].append(0.0)
                    data_dict["__Kernel_Gamma__"].append(0.0)
                    data_dict["__Kernel_Offset__"].append(0.0)
                    data_dict["__Kernel_Weight__"].append(0.0)
        del data_dict["Key_Scheme"]
        ## Done
    
    @staticmethod
    def merge_performance_results(input_dir, delete_files=False):
        """Merge result*.csv files when classification fails or is aborted.
        
        Use function with the pathname where the csv-files are stored.
        E.g., merge_performance_results('/Users/seeland/collections/20100812_11_18_58')
        
        **Parameters**
        
            :input_dir:
                Contains a string with the path where csv files are stored.

            :delete_files:
                controls if the csv-files will be removed after merging has finished
                
                (optional, default: False)
        
        :Author: Mario Krell
        :Created: 2011/09/21
        """
        collection = PerformanceResultSummary(dataset_dir=input_dir)
        collection.delete = delete_files
        collection.store(input_dir)
    
    @staticmethod
    def repair_csv(path, num_splits=None, default_dict=None, delete_files=True):
        """Wrapper function for whole csv repair process when classification fails
           or is aborted.
        
        This function performs merge_performance_results, reporting and reconstruction of missing
        conditions, and a final merge. As a result two files are written:
        results.csv and repaired_results.csv to the path specified.
        
        **Parameters**
            :path:
                String containing the path where the classification results are
                stored. This path is also used for storing the resulting csv files.

            :num_splits:
                Number of splits used for classification. If not specified
                this information is read out from the csv file of the merge_performance_results
                procedure.

                (optional, default: None)

            :default_dict:
                A dictionary specifying default values for missing
                conditions. This dictionary can e.g. be constructed using
                empty_dict(csv_dict) and subsequent modification, e.g.
                default_dict['Metric'].append(0). This parameter is used in
                reconstruct_failures.
                
                (optional, default: None)

            :delete_files:
                Controls if unnecessary files are deleted by merge_performance_results and
                check_op_libSVM.

                (optional, default: True)
                
        :Author: Mario Krell, Sirko Straube
        :Created: 2010/11/09
        """

        PerformanceResultSummary.merge_performance_results(path, delete_files=delete_files)
        filename= path + '/results.csv'

        csv_dict = csv_analysis.csv2dict(filename)
    
        if not num_splits:
            num_splits = int(max(csv_dict['__Key_Fold__']))
        
        oplist= csv_analysis.check_op_libSVM(path, delete_file=delete_files)
            
        failures = csv_analysis.report_failures(oplist, num_splits)
        final_dict= csv_analysis.reconstruct_failures(csv_dict, failures,
                                        num_splits, default_dict=default_dict)
        csv_analysis.dict2csv(path + '/repaired_results.csv', final_dict)
    
    def store(self, result_dir, name = "results", s_format = "csv", main_metric="Balanced_accuracy"):
        """ Stores this collection in the directory *result_dir*.
        
        In contrast to *dump* this method stores the collection
        not in a single file but as a whole directory structure with meta
        information etc. 
        
        **Parameters**
        
          :result_dir: The directory in which the collection will be stored
          :name: The name of the file in which the result file is stored.

                 (*optional, default: 'results'*)

          :s_format: The format in which the actual data sets should be stored.

                     (*optional, default: 'csv'*)

          :main_metric: Name of the metric used for the shortened stored file.
                        If no metric is given, no shortened version is stored.

                        (*optional, default: 'Balanced_accuracy'*)

        """
        try:
            author = pwd.getpwuid(os.getuid())[4]
        except:
            author = "unknown"
            self._log("Author could not be resolved.",level=logging.WARNING)
        # Update the meta data
        self.update_meta_data({"type" : "result",
                               "storage_format": s_format,
                               "author" : author})
        
        # file name in which the operation's results will be stored
        output_file_name = os.path.join(result_dir,name + "." + s_format)
        
        self._log("\tWriting results to %s ..." % output_file_name)
        if s_format == "csv":
            #Store meta data
            BaseDataset.store_meta_data(result_dir,self.meta_data)
            self.data.pop("None",False)
            csv_analysis.dict2csv(output_file_name, self.data)
            if main_metric in self.identifiers:
                reduced_data = dict()
                for key in self.get_variables():
                    if len(list(set(self.data[key]))) > 1:
                        reduced_data[key] = self.data[key]
                reduced_data[main_metric] = self.data[main_metric]
                metric_list = ["True_positives","True_negatives","False_negatives","False_positives"]
                for metric in [x for x in self.data.keys() if x in metric_list]:
                    reduced_data[metric]=self.data[metric]
                output_file_name = os.path.join(result_dir,"short_"+name + "." + s_format)
                csv_analysis.dict2csv(output_file_name, reduced_data)
        else:
            self._log("The format %s is not supported!"%s_format, level=logging.CRITICAL)
            return
        
        if self.delete:
            for temp_result_file in self.tmp_pathlist:
                os.remove(temp_result_file)

    @staticmethod
    def tranfer_Key_Dataset_to_parameters(data_dict):
        if not data_dict.has_key("Key_Dataset"):
            return data_dict
        
        for key_dataset in data_dict["Key_Dataset"]:
            components = (key_dataset.strip("'}{")).split("}{")
            for index, attribute in enumerate(components):
                if index >= 1:
                    # for compatibility with old data: index 1 might be the
                    # specification file name
                    if index == 1 and not ("#" in attribute):
                        attribute_key = "__Template__"
                        attribute_value = attribute
                        continue
                    try:
                        attribute_key, attribute_value = attribute.split("#")
                    except ValueError:
                        warnings.warn("\tValueError when splitting attributes!")
                        print "ValueError in result collection when splitting attributes."
                        continue
                elif index == 0:
                    attribute_key = "__Dataset__"
                    attribute_value = attribute
                data_dict[attribute_key].append(attribute_value)
        del data_dict["Key_Dataset"]

    def project_onto(self, proj_parameter, proj_values):
        """ Project result collection onto a subset that fulfills all criteria
        
        Project the result collection onto the rows where the parameter
        *proj_parameter* takes on the value *proj_value*.
        """
        if type(proj_values) != list:
            proj_values = [proj_values]
        projected_dict = defaultdict(list)
        entries_added = False
        for i in range(len(self.data[proj_parameter])):
            if self.data[proj_parameter][i] in proj_values:
                entries_added = True
                for column_key in self.identifiers:
                    # will leave projection column  in place if there are
                    # still different values for this parameter
                    if column_key == proj_parameter:
                        if len(proj_values) == 1: continue
                    projected_dict[column_key].append(self.data[column_key][i])
        # If the projected_dict is empty we continue
        if not entries_added:
            return
        return PerformanceResultSummary(projected_dict)
    
    def get_gui_metrics(self):
        """ Returns the columns in data that correspond to metrics for visualization. 
        
        This excludes 'Key_Dataset' and gui variables of the tabular,
        
        """
        metrics = []
        variables = self.get_gui_variables()
        for key in self.identifiers:
            if (not(key in variables) \
                or key in ['Key_Dataset']):
                    metrics.append(key)
            # Add variables, that can be interpreted as metrics
            if (key in ['__Num_Retained_Features__','__Num_Eliminated_Sensors__']\
                or key.startswith("~") or "Pon" in key)\
                and len(list(set(self.data[key]))) > 1\
                and not (key in metrics):
                    metrics.append(key)
        return metrics
    
    def get_metrics(self):
        """ Returns the columns in data that are real metrics """
        metrics = []
        variables = self.get_variables()
        for key in self.identifiers:
            if not(key in variables) and not key.startswith("~") and not key=="None":
                    metrics.append(key)
            # Add variables, that can be interpreted as metrics
            if (key in ['__Num_Retained_Features__','__Num_Eliminated_Sensors__']):
                 metrics.append(key)
        return metrics
    
    def get_gui_variables(self):
        """ Returns the column headings that correspond to'variables' to be visualized in the Gui """
        variables = []
        for key in self.identifiers:
            if (key == 'None'  #special key to get box plots without parameter dependencies
                or ((key in ['__Dataset__', 'Kernel_Weight', 'Complexity',
                           'Kernel_Exponent', 'Kernel_Gamma', 'Kernel_Offset',
                           'Classifier_Type', 'Kernel_Type', 'Key_Scheme',
                           'Key_Run', 'Key_Fold','Run','Split'] 
                           # old variable names kept for 
                or key.startswith('__')
                or key.startswith('~')
                ) and not key=="__Solver_Iterations__" # old naming :(
                        and len(list(set(self.data[key]))) > 1)):
                variables.append(key)
        return variables
    
    def get_variables(self):
        """ Variables are marked with '__' 
        
        Everything else are metrics, meta metrics or processing informations.
        """
        variables = []
        for key in self.identifiers:
            if key.startswith('__') and not key=="__Solver_Iterations__":
                variables.append(key)
        return variables
    
    def get_parameter_values(self, parameter):
        """ Returns the values that *parameter* takes on in the data """
        return set(self.data[parameter])
    
    def get_nominal_parameters(self, parameters):
        """ Returns a generator over the nominal parameters in *parameters*

        .. note:: Nearly same code as in *get_numeric_parameters*.
                  Changes in this method should be done also to this method.
        """
        for parameter in parameters:
            try:
                # Try to create a float of the first value of the parameter
                float(self.data[parameter][0])
                # No exception and enough entities thus a numeric attribute
                if len(set(self.data[parameter]))>=5:
                    continue
                else:
                    yield parameter
            except ValueError:
                # This is not a numeric parameter, treat it as nominal
                yield parameter
            except KeyError:
                # This exception should inform the user about wrong parameters
                # in his YAML file.
                import warnings
                warnings.warn('The parameter "' + parameter 
                              + '" is not contained in the PerformanceResultSummary')
            except IndexError:
                # This exception informs the user about wrong parameters in 
                # his YAML file.
                import warnings
                warnings.warn('The parameter "' + parameter 
                              + '" has no values.')
    
    def get_numeric_parameters(self, parameters):
        """ Returns a generator over the numeric parameters in *parameters*

        .. note:: Nearly same code as in *get_nominal_parameters*.
                  Changes in this method should be done also to this method.
        """
        for parameter in parameters:
            try:
                # Try to create a float of the first value of the parameter
                float(self.data[parameter][0])
                # No exception and enough entities thus a numeric attribute
                if len(set(self.data[parameter]))>=5:
                    yield parameter
                else:
                    continue
            except ValueError:
                # This is not a numeric parameter, treat it as nominal
                continue
            except KeyError:
                #"This exception should inform the user about wrong parameters 
                # in his YAML file."
                import warnings
                warnings.warn('The parameter "' + parameter 
                              + '" is not contained in the PerformanceResultSummary')
            except IndexError:
                #This exception informs the user about wrong parameters in
                # his YAML file.
                import warnings
                warnings.warn('The parameter "' + parameter 
                              + '" has no values.')

    def dict2tuple(self,dictionary):
        """ Return dictionary values sorted by key names """
        keys=sorted(dictionary.keys())
        l=[]
        for key in keys:
            l.append(dictionary[key])
        return tuple(l)

    def get_indexed_data(self):
        """ Take the variables and create a dictionary with variable entry tuples as keys """
        # index keys
        self.variables = sorted(self.get_variables())
        # other keys
        keys = [key for key in self.identifiers if not key in self.variables]
        # final dictionary
        data_dict = {}
        for i in range(len(self.data[self.variables[0]])):
            var_dict = {}
            perf_dict = {} 
            # read out variable values
            for variable in self.variables:
                value = self.data[variable][i]
                var_dict[variable]  = value
                perf_dict[variable] = value
            # read out the rest
            for key in keys:
                perf_dict[key] = self.data[key][i]
            # save it into dictionary by mapping values to tuple as key/index
            data_dict[self.dict2tuple(var_dict)] = perf_dict
        return data_dict

    def get_performance_entry(self, search_dict):
        """ Get the line in the data, which corresponds to the `search_dict` """
        search_tuple = self.dict2tuple(search_dict)
        if self.data_dict is None:
            self.data_dict = self.get_indexed_data()
        return self.data_dict.get(search_tuple,None)

    def plot_numeric(self, axes, x_key, y_key, conditions=[]):
        """ Creates a plot of the y_key for the given numeric parameter x_key.  
        
        A function that allows to create a plot that visualizes the effect 
        of differing one variable onto a second one (e.g. the effect of  
        differing the number of features onto the accuracy). 
        
        **Expected arguments**
        
          :axes: The axes into which the plot is written
          :x_key: The key of the dictionary whose values should be used as
                  values for the x-axis (the independent variable)
          :y_key:   The key of the dictionary whose values should be used as
                    values for the y-axis, i.e. the dependent variable
          :conditions:   A list of functions that need to be fulfilled in order to
                         use one entry in the plot. Each function has to take two 
                         arguments: The data dictionary containing all entries and
                         the index of the entry that should be checked. Each condition
                         must return a boolean value.
        """
        colors = cycle(['b', 'g', 'r', 'c', 'm', 'y', 'k', 'brown', 'gray'])
        linestyles = cycle(['-']*9 + ['--']*9 + [':']*9 + ['-.']*9)
        curves = defaultdict(lambda : defaultdict(list))
        for i in range(len(self.data[x_key])):
            # Check is this particular entry should be used 
            if not all(condition(self.data, i) for condition in conditions):
                continue
    
            # Get the value of the independent variable for this entry
            x_value = float(self.data[x_key][i])       
            # Attach the corresponding value to the respective partition
            if y_key.count("#") == 0:
                y_value = float(self.data[y_key][i])
            else: # A weighted cost function
                weight1, value_key1, weight2, value_key2 = y_key.split("#")
                y_value = float(weight1) * float(self.data[value_key1][i]) \
                                        + float(weight2) * float(self.data[value_key2][i]) 

            curves[y_key][x_value].append(y_value)

        for y_key, curve in curves.iteritems():
            curve_x = []
            curve_y = []
            for x_value, y_values in sorted(curve.iteritems()):
                curve_x.append(x_value)
                curve_y.append(y_values)

            # Create an error bar plot 
            axes.errorbar(curve_x, map(numpy.mean, curve_y), 
                          yerr=map(scipy.stats.sem, curve_y),
                          elinewidth = 1, capsize = 5, label=y_key,
                          color = colors.next(), linestyle=linestyles.next())

        axes.legend(loc = 0)
        axes.set_xlabel(x_key) 
        if y_key.count("#") == 0:
            axes.set_ylabel(y_key.strip("_").replace("_", " "))
        else:
            axes.set_ylabel("%s*%s+%s*%s" % tuple(y_key.split("#")))

        # display nearly invisible lines in the back for better orientation
        axes.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                        alpha=0.5)
        axes.set_axisbelow(True)

        # Return figure name
        return "_".join([y_key, x_key])


    def plot_numeric_vs_numeric(self, axes, axis_keys, value_key, scatter=True):
        """ Contour plot of the value_key for the two numeric parameters axis_keys.
    
        A function that allows to create a contour plot that visualizes the effect 
        of differing two variables on a third one (e.g. the effect of differing 
        the lower and upper cutoff frequency of a bandpass filter onto
        the accuracy). 
        
        **Parameters**
        
          :axes: The axes into which the plot is written
          :axis_keys: The two keys of the dictionary that are assumed to have \ 
                      an effect on a third variable (the dependent variable)
          :value_key: The dependent variables whose values determine the \ 
                      color of the contour plot

          :scatter: Plot nearly invisible dots behind the real data points.
          
                  (*optional, default: True*)
        """
        assert(len(axis_keys) == 2)
        
        # Determine a sorted list of the values taken on by the axis keys:
        x_values = set([float(value) for value in self.data[axis_keys[0]]])
        x_values = sorted(list(x_values))
        
        y_values = set([float(value) for value in self.data[axis_keys[1]]])
        y_values = sorted(list(y_values))
        #Done
        
        # We cannot create a contour plot if one dimension is only 1d
        if len(x_values) == 1 or len(y_values) == 1:
            return
        
        # Create a meshgrid of them    
        X, Y = pylab.meshgrid(x_values, y_values)
        # Determine the average value taken on by the dependent variable
        # for each combination of the the two source variables
        Z = numpy.zeros((len(x_values),len(y_values)))
        counter = numpy.zeros((len(x_values),len(y_values)))
        for i in range(len(self.data[axis_keys[0]])):
            x_value = float(self.data[axis_keys[0]][i])
            y_value = float(self.data[axis_keys[1]][i])
            
            if value_key.count("#") == 0:
                performance_value = float(self.data[value_key][i])
            else: # A weighted cost function
                weight1, value_key1, weight2, value_key2 = value_key.split("#")
                performance_value = float(weight1) * float(self.data[value_key1][i]) \
                                        + float(weight2) * float(self.data[value_key2][i]) 
            
            Z[x_values.index(x_value), y_values.index(y_value)] += performance_value
            counter[x_values.index(x_value), y_values.index(y_value)] += 1
        Z = Z / counter
        
        # Create the plot for this specific dependent variable 
        cf = axes.contourf(X, Y, Z.T, N = 20)
        axes.get_figure().colorbar(cf)
        if scatter:
            axes.scatter(X,Y,marker='.',facecolors='None', alpha=0.1)
        axes.set_xlabel(axis_keys[0].strip("_").replace("_", " "))
        axes.set_ylabel(axis_keys[1].strip("_").replace("_", " "))
        axes.set_xlim(min(x_values), max(x_values))
        axes.set_ylim(min(y_values), max(y_values))
        if value_key.count("#") == 0:
            axes.set_title(value_key.strip("_").replace("_", " "))
        else:
            axes.set_title("%s*%s+%s*%s" % tuple(value_key.split("#")))
        
        # Return figure name
        return "%s_%s_vs_%s" % (value_key, axis_keys[0].strip("_").replace("_", " "), axis_keys[1].strip("_").replace("_", " "))
        
    def plot_numeric_vs_nominal(self, axes, numeric_key, nominal_key, value_key,dependent_BA_plot=False, relative_plot=False):
        """ Plot for comparison of several different values of a nominal parameter
    
        A function that allows to create a plot that visualizes the effect of 
        varying one numeric parameter onto the performance for several
        different values of a nominal parameter. 
        
        **Parameters**
        
          :axes: The axes into which the plot is written  
          :numeric_key:   The numeric parameter whose effect (together with the
                          nominal parameter) onto the dependent variable should
                          be investigated.
          :nominal_key:   The nominal parameter whose effect (together with the
                          numeric parameter) onto the dependent variable should
                          be investigated. 
          :value_key:    The dependent variable whose values determine the
                         color of the contour plot
          :dependent_BA_plot:  
                If the `value_key` contains *time* or *iterations*
                and this variable is True, the value is replaced by
                *Balanced_Accuracy* and the `nominal_key` by the `value_key`.
                The point in the graph are constructed by averaging
                over the old `nominal parameter`.
                
                (*optional, default: False*)
                
          :relative_plot:
                The first `nominal_key` value (alphabetic ordering) is chosen and the other
                parameters are averaged relative to this parameter, to show
                by which factor they change the metric.
                Therefore a clean tabular is needed with only relevant
                variables correctly named and where each parameter is compared
                with the other. Relative plots and dependent_BA plots can be combined.

                (*optional, default: False*)
        """
        colors = cycle(['b','r', 'g',  'c', 'm', 'y', 'k', 'brown', 'gray','orange'])
        linestyles = cycle(['-']*10 + ['-.']*10 + [':']*10 + ['--']*10)
        eps=10**(-6)
        # Determine a mapping from the value of the nominal value to a mapping
        # from the value of the numeric value to the achieved performance:
        # nominal -> (numeric -> performance)
        if (("time" in value_key) or ("Time" in value_key) or ("iterations" in value_key)) and dependent_BA_plot:
            dependent_key = value_key
            value_key="Balanced_accuracy"
        else:
            dependent_key = False
            relative_plot = False
        if relative_plot:
            rel_par = sorted(list(set(self.data[nominal_key])))[0]
            rel_vars = self.get_variables()
        curves = defaultdict(lambda: defaultdict(list))
        for i in range(len(self.data[nominal_key])):
            curve_key = self.data[nominal_key][i]
            parameter_value = float(self.data[numeric_key][i])
            if value_key.count("#") == 0:
                performance_value = float(self.data[value_key][i])
            else: # A weighted cost function
                weight1, value_key1, weight2, value_key2 = value_key.split("#")
                performance_value = float(weight1) * float(self.data[value_key1][i]) \
                                        + float(weight2) * float(self.data[value_key2][i])
            if relative_plot:
                if curve_key == rel_par:
                    factor = 1
                    performance_value = 1
                    if dependent_key:
                        dependent_factor = self.data[dependent_key][i]
                else:
                    rel_vars_dict = dict()
                    for var in rel_vars:
                        rel_vars_dict[var] = self.data[var][i]
                    rel_vars_dict[nominal_key] = rel_par
                    rel_data = self.get_performance_entry(rel_vars_dict)
                    if value_key.count("#") == 0:
                        try:
                            factor = float(rel_data[value_key])
                        except TypeError,e:
                            print rel_data
                            print value_key
                            print rel_vars_dict
                            print rel_vars_dict.keys()
                            raise(e)
                    else: # A weighted cost function
                        weight1, value_key1, weight2, value_key2 = value_key.split("#")
                        factor = float(weight1) * float(rel_data[value_key1]) \
                                                + float(weight2) * float(rel_data[value_key2])
                    dependent_factor = rel_data.get(dependent_key,1)
                    if dependent_factor  == 0:
                        dependent_factor = eps
                        warnings.warn("Dependent key %s got zero value in reference %s."%(
                            str(dependent_key),rel_par
                            ))
                    if factor == 0:
                        factor = eps
                        warnings.warn("Value key %s got zero value in reference %s."%(
                            str(value_key),rel_par
                            ))
            else:
                factor =  1
                dependent_factor = 1
            if not dependent_key:
                curves[curve_key][parameter_value].append(performance_value/factor)
            else:
                curves[curve_key][parameter_value].append((performance_value/factor,float(self.data[dependent_key][i])/float(dependent_factor)))
        
        # Iterate over all values of the nominal parameter and create one curve
        # in the plot showing the mapping from numeric parameter to performance 
        # for this particular value of the nominal parameter
        for curve_key, curve in curves.iteritems():
            x_values = []
            y_values = []
            y_errs = []
            x_errs = []
            for x_value, y_value in sorted(curve.iteritems()):
                if not dependent_key:
                    x_values.append(x_value)
                    # Plot the mean of all values of the performance for this 
                    # particular combination of nominal and numeric parameter
                    y_values.append(pylab.mean(y_value))
                    y_errs.append(scipy.stats.sem(y_value))
                    x_errs = None
                else:
                    # calculate mean and standard deviation
                    # of metric and dependent parameter values and
                    # use the dependent parameter as x_value
                    # and the metric as y_value
                    mean = numpy.mean(y_value,axis=0)
                    metric_mean = mean[0]
                    time_mean = mean[1]
                    sem = scipy.stats.sem(y_value,axis=0)
                    metric_sem = sem[0]
                    time_sem = sem[1]
                    x_values.append(time_mean)
                    y_values.append(metric_mean)
                    x_errs.append(time_sem)
                    y_errs.append(metric_sem)
            if len(x_values)<101:
                axes.errorbar(x_values, y_values, xerr = x_errs, yerr=y_errs, 
                          label=curve_key, 
                          color = colors.next(), linestyle=linestyles.next(),
                          lw=2, elinewidth=0.8,capsize=3,marker='x')
            else:
                axes.errorbar(x_values, y_values, xerr = x_errs, yerr=y_errs, 
                          label=curve_key, 
                          color = colors.next(), linestyle=linestyles.next(),
                          lw=1, elinewidth=0.04,capsize=1)

        if dependent_key:
            numeric_key = dependent_key.strip("_") + " averaged dependent on " + numeric_key.strip("_")
        axes.set_xlabel(numeric_key.strip("_").replace("_", " "))
        if relative_plot:
            value_key = value_key.strip("_")+" relative to "+ rel_par
        if value_key.count("#") == 0:
            axes.set_ylabel(value_key.strip("_").replace("_", " "))
        else:
            axes.set_ylabel("%s*%s+%s*%s" % tuple(value_key.split("#")))

        # display nearly invisible lines in the back for better orientation
        axes.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                        alpha=0.5)
        axes.set_axisbelow(True)

        prop = matplotlib.font_manager.FontProperties(size='xx-small')
        if not nominal_key=="None":
            lg=axes.legend(prop=prop, loc=0,fancybox=True,title=nominal_key.strip("_"))
            lg.get_frame().set_facecolor('0.90')
            lg.get_frame().set_alpha(.3)
        # axes.set_xscale('log')
        # Return figure name
        return "%s_%s_vs_%s" % (value_key, nominal_key, numeric_key)
    
    def plot_nominal(self, axes, x_key, y_key):
        """ Creates a boxplot of the y_key for the given nominal parameter x_key.
        
        A function that allows to create a plot that visualizes the effect 
        of differing one nominal variable onto a second one (e.g. the effect of  
        differing the classifier onto the accuracy). 
        
        **Expected arguments**
        
          :axes: The axes into which the plot is written
          :x_key: The key of the dictionary whose values should be used as
                    values for the x-axis (the independent variables)
          :y_key: The key of the dictionary whose values should be used as
                    values for the y-axis, i.e. the dependent variable
        """
        # Create the plot for this specific dependent variable 
        values = defaultdict(list)
        for i in range(len(self.data[x_key])):
            parameter_value = self.data[x_key][i]
            if y_key.count("#") == 0:
                performance_value = float(self.data[y_key][i])
            else: # A weighted cost function
                weight1, y_key1, weight2, y_key2 = y_key.split("#")
                performance_value = float(weight1) * float(self.data[y_key1][i]) \
                                        + float(weight2) * float(self.data[y_key2][i]) 
                
            values[parameter_value].append(performance_value)
        
        values = sorted(values.items())
        
        # the bottom of the subplots of the figure
        axes.figure.subplots_adjust(bottom = 0.3)
        axes.boxplot(map(lambda x: x[1], values))
        axes.set_xticklabels(map(lambda x: x[0], values))
        matplotlib.pyplot.setp(axes.get_xticklabels(), rotation=-90)
        matplotlib.pyplot.setp(axes.get_xticklabels(), size='x-small')
        axes.set_xlabel(x_key.replace("_", " "))

        # display nearly invisible lines in the back for better orientation
        axes.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                        alpha=0.5)
        axes.set_axisbelow(True)

        if y_key.count("#") == 0:
            axes.set_ylabel(y_key.replace("_", " "))
        else:
            axes.set_ylabel("%s*%s+%s*%s" % tuple(y_key.split("#")))
      
        # Return figure name   
        return "%s_%s" % (y_key, x_key)

    def plot_nominal_vs_nominal(self, axes, nominal_key1, nominal_key2, value_key):
        """ Plot comparison of several different values of two nominal parameters
        
        A function that allows to create a plot that visualizes the effect of
        varying one nominal parameter onto the performance for several
        different values of another nominal parameter.
        
        **Expected arguments**
        
          :axes: The axes into which the plot is written
          :nominal_key1:   The name of the first nominal parameter whose effect
                           shall be investigated. This parameter determines the
                           x-axis.
          :nominal_key2:   The second nominal parameter. This parameter will be
                           represented by a different color per value.
          :value_key:    The name of the dependent variable whose values
                         determines the y-values in the plot.
        """
        from matplotlib.patches import Polygon, Rectangle
        # boxColors = ['b','r', 'g',  'c', 'm', 'y', 'k', 'brown', 'gray']
        boxColors = ['steelblue','burlywood', 'crimson', 'olive', 'cadetblue',
                     'cornflowerblue',  'darkgray', 'darkolivegreen',
                     'goldenrod', 'lightcoral', 'lightsalmon', 'lightseagreen',
                     'lightskyblue', 'lightslategray', 'mediumseagreen',
                     'mediumturquoise', 'mediumvioletred', 'navy', 'orange',
                     'tan', 'teal', 'yellowgreen']
        
        # Gathering of the data
        plot_data = defaultdict(lambda: defaultdict(list))
        for i in range(len(self.data[nominal_key2])):
            nom1_key = self.data[nominal_key1][i]
            nom2_key = self.data[nominal_key2][i]
            if value_key.count("#") == 0:
                performance_value = float(self.data[value_key][i])
            else: # A weighted cost function
                weight1, value_key1, weight2, value_key2 = value_key.split("#")
                performance_value = \
                            float(weight1) * float(self.data[value_key1][i]) \
                             + float(weight2) * float(self.data[value_key2][i])
                
            plot_data[nom1_key][nom2_key].append(performance_value)
        
        # Prepare data for boxplots
        box_data=[]
        nom1_keys=[]
        for nom1_key, curve in plot_data.iteritems():
            x_values = []
            y_values = []
            nom1_keys.append(nom1_key)
            for x_value, y_values in sorted(curve.iteritems()):
                box_data.append(y_values)
        
        # Make sure we always have enough colors available
        nom2_keys = sorted(plot_data[nom1_key].keys())
        while len(nom2_keys)>len(boxColors):
            boxColors+=boxColors
        
        # the bottom of the subplots of the figure
        axes.figure.subplots_adjust(bottom = 0.3)
        # position the boxes in the range of +-0.25 around {1,2,3,...}
        box_positions=[]
        for i in range(len(nom1_keys)):
            if len(nom2_keys) > 1:
                box_positions.extend([i+1 -.25 + a*.5/(len(nom2_keys)-1)
                                          for a in range(len(nom2_keys))])
            else:
                box_positions.extend([i+1])
        # actual plotting; width of the boxes:
        w=.5 if len(nom2_keys)==1 else .35/(len(nom2_keys)-1)
        bp = axes.boxplot(box_data, positions=box_positions, widths=w)
        # design of boxplot components
        matplotlib.pyplot.setp(bp['boxes'], color='black')
        matplotlib.pyplot.setp(bp['whiskers'], color='black')
        matplotlib.pyplot.setp(bp['fliers'], color='grey', marker='+', mew=1.5)
        # use the nom1 keys as x-labels
        axes.set_xticks([i+1 for i in range(len(nom1_keys))], minor=False)
        axes.set_xticklabels(nom1_keys)
        matplotlib.pyplot.setp(axes.get_xticklabels(), rotation=-90)
        matplotlib.pyplot.setp(axes.get_xticklabels(), size='x-small')
        axes.set_xlabel(nominal_key1.replace("_", " "))
        
        # Now fill the boxes with desired colors by superposing polygons
        numBoxes = len(nom1_keys)*len(nom2_keys)
        medians = range(numBoxes)
        # get all box coordinates
        for i in range(numBoxes):
            box = bp['boxes'][i]
            boxX = []
            boxY = []
            for j in range(5):
                boxX.append(box.get_xdata()[j])
                boxY.append(box.get_ydata()[j])
            boxCoords = zip(boxX,boxY)
            # cycle through predefined colors
            k = i % len(nom2_keys)
            # draw polygon
            boxPolygon = Polygon(boxCoords, facecolor=boxColors[k])
            axes.add_patch(boxPolygon)
            # Now draw the median lines back over what we just filled in
            med = bp['medians'][i]
            medianX = []
            medianY = []
            for j in range(2):
                medianX.append(med.get_xdata()[j])
                medianY.append(med.get_ydata()[j])
                axes.plot(medianX, medianY, 'k')
                medians[i] = medianY[0]
            
        # Draw a legend by hand. As the legend is hand made, it is not easily
        # possible to change it's location or size - sorry for inconvenience.
        # width of the axes and xy-position of legend element #offset
        dxy = [axes.get_xlim()[1]-axes.get_xlim()[0],
               axes.get_ylim()[1]-axes.get_ylim()[0]]
        xy = lambda offset: [axes.get_xlim()[0] + .8*dxy[0],
                             axes.get_ylim()[0] + .03*dxy[1]
                                                + .05*dxy[1]*offset]
        # Background rectangle for the legend.
        rect = Rectangle([xy(0)[0]-.02*dxy[0], xy(0)[1]-.02*dxy[1]],
                         .2*dxy[0],(.05*(len(nom2_keys)+1)+0.0175)*dxy[1],
                         facecolor='0.9', alpha=0.8)
        # legend "title"
        axes.text(xy(len(nom2_keys))[0]+.03*dxy[0], xy(len(nom2_keys))[1]+.005*dxy[1],
                  nominal_key2.strip("_").replace("_", " "),
                  color='black', weight='roman', size='small')

        axes.add_patch(rect)
        # rect and text for each nom2-Value
        for key in range(len(nom2_keys)):
            rect = Rectangle(xy(key),.05*dxy[0],.035*dxy[1],
                             facecolor=boxColors[len(nom2_keys)-key-1])
            axes.add_patch(rect)
            axes.text(xy(key)[0]+.06*dxy[0], xy(key)[1]+.005*dxy[1],
                      nom2_keys[len(nom2_keys)-key-1],
                      color='black', weight='roman', size='small')
        
        # Add a horizontal grid to the plot
        axes.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                          alpha=0.5)
        axes.set_axisbelow(True)
        
        if value_key.count("#") == 0:
            axes.set_ylabel(value_key.strip("_").replace("_", " "))
        else:
            axes.set_ylabel("%s*%s+%s*%s" % tuple(value_key.split("#")))

        # display nearly invisible lines in the back for better orientation
        axes.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                        alpha=0.5)
        axes.set_axisbelow(True)

        # Return figure name
        return "%s_%s_vs_%s" % (value_key, nominal_key1, nominal_key2)

    def plot_histogram(self, axes, metric, numeric_parameters, nominal_parameters,
                       average_runs = True):
        """ Plots a histogram of the values the given metric takes on in data
         
        Plots histogram for *metric* in which each parameter combination from 
        *numeric_parameters* and *nominal_parameters* corresponds
        to one value (if *average_runs* == True) or each run corresponds
        to one value (if *average_runs* == False).
        The plot is written into *axes*.
        """
        if average_runs == False:
            metric_values = map(float, self.data[metric])
        else:
            # Merge all parameters in one list
            parameters = list(numeric_parameters)
            parameters.extend(nominal_parameters)
            
            # Sort metric values according to the parameterization for the
            # specific value
            all_values = defaultdict(list)
            for i in range(len(self.data[metric])):
                key = tuple(self.data[parameter][i] for parameter in parameters)
                all_values[key].append(float(self.data[metric][i]))
            
            # Combine the mean value of the metric for each parameter 
            # combination
            metric_values = [numpy.mean(value) 
                                    for value in all_values.itervalues()]
        
        # Plot and store the histogram
        axes.hist(metric_values, histtype='stepfilled', align='left')
        axes.set_ylim((0, pylab.ylim()[1]))
        axes.set_xlabel(metric if average_runs == False 
                        else "Mean %s" % metric)
        axes.set_ylabel('Occurrences')
        
        # Return figure name
        return "%s_histogram" % metric


###############################################################################


class ROCCurves(object):
    """ Class for plotting ROC curves """
    
    def __init__(self, base_path):
        self.roc_curves = self._load_all_curves(base_path)
        
        self.colors = cycle(['b', 'g', 'r', 'c', 'm', 'y', 'k', 'brown', 'gray'])  
        
    def is_empty(self):
        """ Return whether there are no loaded ROC curves """
        return len(self.roc_curves) == 0

    def plot(self, axis, selected_variable, projection_parameter, fpcost=1.0,
             fncost=1.0, collection=None):
        # Draw cost grid into the background
        for cost in numpy.linspace(0.0, fpcost+fncost, 25):
            axis.plot([0.0, 1.0], [1-cost/fncost, 1-(cost-fpcost)/fncost],
                      c='gray', lw=0.5)
            
#        # If we do not average:
#        if selected_variable == None:
#            # Delegate to plot_all method
#            return self.plot_all(axis, projection_parameter, collection)
        
        # Draw an additional "axis" (the identity) to show skew/centroid of
        # ROC curves
        axis.plot([0.0, 1.0], [0.0, 1.0], c='k', lw=2)
        for k in numpy.linspace(0.0, 1.0, 10):
            axis.plot([k+0.01, k-0.01], [k-0.01, k+0.01], c='k', lw=1)
    
        # Create a color dict
        color_dict = defaultdict(lambda : self.colors.next())
        
        # Some helper function
        def create_roc_function(roc_curve):
            """ Create a function mapping FPR onto TPR for the given roc_curve """
            def roc_function(query_fpr):
                """ Map FPR onto TPR using linear interpolation on ROC curve."""
                if query_fpr == 0.0: return 0.0 # Avoid division by zero
                last_fpr, last_tpr = 0.0, 0.0
                for fpr, tpr in roc_curve:
                    if fpr >= query_fpr:
                        return (query_fpr - last_fpr)/(fpr - last_fpr) * (tpr - last_tpr) + last_tpr
                    last_fpr, last_tpr = fpr, tpr
                return tpr

            return roc_function
        
        def create_weight_function(x_values, mean_curve):
            """ 
            Creates a function that computes the orthogonal distance of the ROC
            curve from the identity axis at an arbitrary (k,k)
            """
            def weight_function(k):
                """ 
                Creates a function that computes the orthogonal distance of the 
                ROC curve from the identity axis at (k,k) 
                """
                if k == 0.0: return 0.0 # Avoid division by zero
                for fpr, tpr in zip(x_values, mean_curve):
                    if 0.5*fpr + 0.5*tpr >= k:
                        return 2*(0.5*fpr - 0.5*tpr)**2
                return 0.0

            return weight_function
        
        # Create mapping parameterization -> ROC functions
        roc_fct_dict = defaultdict(list)
        for parametrization, roc_curve in self._project_onto_subset(self.roc_curves, 
                                                                    projection_parameter):
            key = parametrization[selected_variable] if selected_variable != None and selected_variable in parametrization.keys() else "Global"
            roc_fct_dict[key].append(create_roc_function(roc_curve))
            
        # Iterate over all parametrization and average ROC functions and compute
        # centroid
        for param, roc_fcts in roc_fct_dict.iteritems():
            x_values = numpy.linspace(0.0, 1.0, 500)
            roc_values = []
            for x in x_values:
                roc_values.append([roc_fct(x) for roc_fct in roc_fcts])
            
            mean_curve = map(numpy.mean, roc_values)
            
            # Compute centroid of the mean ROC curve over the identity axis
            weight_fct = create_weight_function(x_values, mean_curve)

            k_values = numpy.linspace(0.0, 1.0, 100)
            weights = [weight_fct(k) for k in numpy.linspace(0.0, 1.0, 100)]
            centroid = sum(k_values[i]*weights[i] for i in range(len(k_values))) \
                                / sum(weights)
                                             
            if selected_variable == None:
                color = self.colors.next()
            else:
                color = color_dict[param]

            axis.plot(x_values, mean_curve, c=color, label=str(param))
            axis.errorbar(x_values[::25], mean_curve[::25],
                          yerr=map(scipy.stats.sem, roc_values)[::25],
                          c=color, fmt='.')

            axis.plot([centroid], [centroid], 
                      c=color, marker='h')
                    
        axis.set_xlabel("False positive rate")
        axis.set_ylabel("True positive rate")
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(0.0, 1.0)
        axis.legend(loc=0)
        if selected_variable is not None:
            axis.set_title(str(selected_variable))

        
    def plot_all(self, axis, projection_parameter, collection=None):
        """ Plot all loaded ROC curves after projecting onto subset. """
        # Iterate over all ROC curves for parametrization that are selected
        # by projection_parameter.
        for parametrization, roc_curve in self._project_onto_subset(self.roc_curves, 
                                                                    projection_parameter):
            color = self.colors.next()
                           
            axis.plot(map(itemgetter(0), roc_curve), map(itemgetter(1), roc_curve),
                      c=color)
            
#            fpr = eval(collection.data['False_positive_rate'][0])
#            tpr = eval(collection.data['True_positive_rate'][0])
#            axis.scatter([fpr], [tpr], c='k', s=50)
        
        axis.set_xlabel("False positive rate")
        axis.set_ylabel("True positive rate")
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(0.0, 1.0)
        axis.legend(loc=0)
                   
    def _load_all_curves(self, dir):
        """ Load all ROC curves located in the persistency dirs below *dir* """
        all_roc_curves = []
        for subdir in [name for name in os.listdir(dir) 
                                if os.path.isdir(os.path.join(dir, name))]:
            if not subdir.startswith("{"): continue
            parametrization = {}
            tokens = subdir.strip("}{").split("}{")
            parametrization["__Dataset__"] = tokens[0]
            for token in tokens[1:]:
                # TODO if anything else then node chain template has no # this will fail;
                # delete as soon as no more data with node chain templates in folder names
                # circulate
                if '#' not in token:
                    parametrization["__Template__"] = token
                    continue
                key, value = token.split("#")
                try:
                    value = eval(value)
                except:
                    pass
                parametrization[key] = value
                           
            for run_dir in glob.glob(dir + os.sep + subdir 
                                     + os.sep + "persistency_run*"):
                run = eval(run_dir.split("persistency_run")[1])
                for split_file in glob.glob(run_dir + os.sep + "PerformanceSinkNode"
                                            + os.sep + "roc_points_sp*.pickle"):
                    split = eval(split_file.split("roc_points_sp")[1].strip(".pickle"))
                    rs_parametrization = dict(parametrization)
                    rs_parametrization["__Key_Run__"] = run
                    rs_parametrization["__Run__"] = "__Run_"+str(run)
                    rs_parametrization["__Key_Fold__"] = split
                    rs_parametrization["__Split__"] = "__Split_"+str(split)
                    
                    roc_curves = cPickle.load(open(split_file, 'r'))
                    
                    all_roc_curves.append((rs_parametrization, roc_curves[0]))
                    
        return all_roc_curves
    
    def _project_onto_subset(self, roc_curves, constraints):
        """Retain only roc_curves that fulfill the given constraints. """
        for parametrization, roc_curve in roc_curves:
            # Check constraints
            constraints_fulfilled = True
            for constraint_key, constraint_value in constraints.iteritems():
                if not constraint_key in parametrization \
                         or parametrization[constraint_key] != constraint_value:
                    constraints_fulfilled = False
            if constraints_fulfilled:
                yield (parametrization, roc_curve)

