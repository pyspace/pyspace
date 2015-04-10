""" Deal with csv files in general, and in particular after classification

The functions provided here focus on two issues:

    1) Manipulation of csv files (load, save, change)
    2) Repair csv files after unsuccessful classification,
       e.g. to be able to perform an analysis operation

**Examples**

    1) Loading csv file, extracting relevant data, saving new csv file:

        *Problem*:

        A csv file is existing, but the file is huge and you need only certain
        values, which are all entries with

            - Parameter __Range__=500
            - Parameter __Start__=100
            - Parameter __SamplingFreq__=25

        *Solution*:

        .. code-block:: python

            import csv_analysis
            data=csv_analysis.csv2dict('results.csv')
            conditions=csv_analysis.empty_dict(data)
            conditions['__Range__'].append('500')
            conditions['__Start__'].append('200')
            conditions['__SamplingFreq__'].append('25')
            new_dict=csv_analysis.strip_dict(data, conditions)
            csv_analysis.dict2csv('new_results.csv', new_dict)

    2) Build results.csv after classification failure and complement with reconstructed conditions:

        *Problem*:

        A classification procedure failed or has been aborted. What is needed is a
        procedure that

            (i)     builds a results.csv from conditions that were ready
            (ii)    identifies the conditions which were not ready
            (iii)   reconstructs missing conditions according to parameters inferable
                    from path and user defined default values (e.g. AUC=0.5 and
                    F_measure=0)
            (iv)    merges into existing results and saves.

        *Solution short*:

        .. code-block:: python

            from pySPACE.tools import csv_analysis
            from pySPACE.resources.dataset_defs.performance_result import PerformanceResultSummary
            mydefaults=dict()
            mydefaults['AUC']=0.5
            mydefaults['F_measure']=0
            PerformanceResultSummary.repair_csv(datapath, default_dict=mydefaults)

        *Solution long*:

        .. code-block:: python

            import csv_analysis
            from pySPACE.resources.dataset_defs.performance_result import PerformanceResultSummary
            num_splits=52
            PerformanceResultSummary.merge_performance_results(datapath)
            csv_dict = csv_analysis.csv2dict(datapath + '/results.csv')
            oplist=csv_analysis.check_op_libSVM(datapath)
            failures = csv_analysis.report_failures(oplist, num_splits)
            mydefaults=dict()
            mydefaults['AUC']=0.5
            mydefaults['F_measure']=0
            final_dict=csv_analysis.reconstruct_failures(csv_dict, failures,
                                                num_splits, default_dict=mydefaults)
            csv_analysis.dict2csv(datapath + '/repaired_results.csv', final_dict)

:Author: Sirko Straube (sirko.straube@dfki.de), Mario Krell,
         Anett Seeland, David Feess
:Created: 2010/11/09
"""

def csv2dict(filename, filter_keys = None, delimiter=',', **kwargs):
    """ Load a csv file and return content in a dictionary
    
    The dictionary has n list elements,
    with n being equal to the number of columns in the csv file.
    Additional keyword arguments are passed to the reader
    instance, e.g. a different delimiter than ',' (see csv.Reader).
    
    **Parameters**
    
        :filename:
            Contains the filename as a string.
            
        :filter_keys:
            If a list of filter keys is specified, only the specified
            keys are left and all other are discarded
            
        :delimiter:
            The delimiter between columns in the csv. Defaults to ',', as csv
            actually stands for comma separated, but sometimes different symbols
            are used.
    
    :Author: Sirko Straube (sirko.straube@dfki.de)
    :Created: 2010/11/09
    """

    import csv
    from collections import defaultdict

    csv_file = open(filename)
    csvDictReader = csv.DictReader(csv_file, delimiter=delimiter, **kwargs)

    
    data_dict = defaultdict(list)
    for line_dict in csvDictReader:
        for key, value in line_dict.iteritems():
            if not filter_keys:
                data_dict[key].append(value)
            elif key in filter_keys or key.startswith("__"):
                data_dict[key].append(value)
            
    csv_file.close()

    return data_dict

def dict2csv(filename, data_dict, delimiter=','):
    """ Write a dictionary to a csv file in a sorted way
    
    The function converts the dictionary into a list of dictionaries,
    with each entry representing one row in the final csv file.
    The dictionary can be of the form returned by csv2dict.
    
    The sorting is in alphabetic order, large characters first and
    variables starting with '__' first.
        
    **Parameters**
        :filename:
            Contains the filename as a string.
            
        :data_dict:
            Dictionary containing data as a dictionary of lists 
            (one list for each column identified by the key).
            
        :delimiter:
            The delimiter between columns in the csv. Defaults to ',', as csv
            actually stands for comma separated, but sometimes different symbols
            are used.
    
    :Author: Sirko Straube, Mario Krell
    :Created: 2010/11/09
    """
    # init
    import csv
    import copy
    csv_file=open(filename,'w')
    final_list=[]
    # sorting of key
    temp_keys = sorted(copy.deepcopy(data_dict.keys()))
    keys = [key  for key in temp_keys if key.startswith("__")]
    for key in keys:
        temp_keys.remove(key)
    for key in temp_keys:
        keys.append(key)
    del(temp_keys)
    # check for consistency (delete columns with wrong length)
    l = len(data_dict[keys[0]])
    remove_keys = []
    for key in keys:
        if not len(data_dict[key])==l:
            import warnings
            warnings.warn("Different length of columns with names %s (deleted) and %s (reference)."%(key,keys[0]))
            data_dict.pop(key)
            remove_keys.append(key)
    for key in remove_keys:
        keys.remove(key)
    # make a list of dictionaries for each row
    for current_line in range(l):
        ldict = {}
        for key in keys:
            ldict[key] = data_dict[key][current_line]
        final_list.append(ldict)
    # save it
    csvDictWriter=csv.DictWriter(csv_file, quoting=csv.QUOTE_ALL,
                                 fieldnames=keys, delimiter=delimiter,
                                 lineterminator='\n')
    csvDictWriter.writerow(dict(zip(keys,keys)))
    csvDictWriter.writerows(final_list)
    csv_file.close()

def empty_dict(old_dict):
    """ Return a dictionary of empty lists with exactly the same keys as old_dict
    
    **Parameters**
    
        :old_dict:
            Dictionary of lists (identified by the key).
    
    :Author: Sirko Straube
    :Created: 2010/11/09
    """
    
    from collections import defaultdict
    new_dict=defaultdict(list)
    [new_dict[i] for i in old_dict.keys()]
    
    return new_dict

def strip_dict(data_dict, cond_dict, invert_mask=False, limit2keys=None):
    """ Return a stripped dictionary according to the conditions specified with cond_dict and invert_mask
        
    This function is useful, if only some parameter combinations are
    interesting. Then the values of interest can be stored in cond_dict and
    after execution of mynewdict=strip_dict(data_dict, cond_dict) all
    unnecessary information is eliminated in mynewdict.
    
    **Parameters**
    
        :data_dict:
            Dictionary of lists (identified by the key). E.g. as returned by
            csv2dict.
        :cond_dict:
            Dictionary containing all keys and values that should be used to
            strip data_dict. E.g. constructed by empty_dict(data_dict) and
            subsequent modifications.
        :invert_mask:
            optional: If set to False, the cond_dict will be interpreted as
            positive list, i.e. only values are kept that are specified in
            cond_dict. If set to True, the cond_dict will be interpreted as
            negative list, i.e. only values are kept that are NOT specified in
            cond_dict. default=False
        :limit2keys:
            optional: Contains a list of key names (strings) that should be
            included in the returned dictionary. All other keys (i.e. columns)
            are skipped. default=None
    
    :Author: Sirko Straube
    :Created: 2010/11/09
    """

    from collections import defaultdict

    constr_not_valid=False
    
    #in the beginning all indices are valid...
    #take first key to determine length of csv-table
    first_key=data_dict.keys()[0] 
    valid_indices=range(len(data_dict[first_key])) 
    
    #check if condition actually appears in the data_dict
    for key in cond_dict.keys():
        if key not in data_dict.keys():
            constr_not_valid=True
            import warnings
            warnings.warn("The condition key (column heading) %s is not " \
                          "present in the dictionary you want to strip!" % key)

    for current_param in data_dict:
        if current_param in cond_dict:
            if cond_dict[current_param]: #if != []
                old_indices=valid_indices
                valid_indices=[] #reset indices to add new constraint
                constraint=cond_dict[current_param]
                for index, value in enumerate(data_dict[current_param]):
                    if not invert_mask:
                        #keep index only if new AND old constraints are valid
                        if value in constraint and index in old_indices:
                            valid_indices.append(index)            
                    else: # i.e., invert_mask == True
                        #keep index only if new AND old constraints are valid
                        if value not in constraint and index in old_indices:
                            valid_indices.append(index)
                if valid_indices == []:
                    constr_not_valid=True
                    import warnings
                    warnings.warn("Constraint values %s of key %s were not"\
                                  " found in the dictionary you want to strip!"\
                                  " Returning empty dict!"\
                                  % (str(cond_dict[current_param]), 
                                     current_param))
                        
    result_dict=dict()
    
    if not constr_not_valid:
        #wrapping up
        for current_param in data_dict:
            current_list=data_dict[current_param]
            new_list = [current_list[i] for i in valid_indices]
            result_dict[current_param]=new_list
        
        #limit2keys restriction
        if limit2keys:
            all_results=result_dict
            result_dict=dict()
            for key in limit2keys:
                result_dict[key]=all_results[key]
        
    return result_dict


def merge_dicts(dict1,dict2):
    """Merge two dictionaries into a new one

    Both have ideally the same keys and lengths.
    The merge procedure is performed even if the keys are not identical,
    but a warning is elicited.
        
    **Parameters**
    
        :dict1:
            the one dictionary
        :dict2:
            the other dictionary
    
    :Author: Mario Michael Krell
    :Created: 2010/11/09
    """
    
    import copy, warnings

    result_dict = dict()
    if not len(dict1.keys()) == len(dict2.keys()) or \
            not all([key1 in dict2.keys() for key1 in dict1.keys()]):
        warnings.warn('Inconsistency while merging: ' +
                      'The two directories have different keys!')
        bad_keys = True
    else:
        bad_keys = False
    for key in dict1.keys():
        if dict2.has_key(key):
            result_dict[key] = copy.deepcopy(dict1[key])
            result_dict[key].extend(copy.deepcopy(dict2[key]))
        else:
            warnings.warn('Inconsistency while merging: Key ' + key + 
                          ' is only existing in one dictionary!')
    if bad_keys:
        for key in dict2.keys():
            if not dict1.has_key(key):
                warnings.warn('Inconsistency while merging: Key ' + key +
                              ' is only existing in one dictionary!')
    return result_dict


def merge_multiple_dicts(dictlist):
    """ Merge multiple dictionaries into a single one
    
    This function merges every dictionary into a single one.
    The merge procedure is performed even if the keys are not identical (or of
    identical length), but a warning is elicited once.
        
    **Parameters**
    
        :dictlist:
            a list of dictionaries to merge
    
    :Author: Sirko Straube
    :Created: 2011/04/20
    """
    
    n=len(dictlist)
    
    if n==0:
        raise ValueError('List of dictionaries is empty!')
    elif n==1:
        return dictlist[0]
    elif n==2:
        return merge_dicts(dictlist[0],dictlist[1])
    else:
        data=dictlist[0]

        for pos in range(1,n):
            data=merge_dicts(data, dictlist[pos])
        return data
    
def add_key(orig_dict, key_str, key_list):
    """ Add a key to the dictionary with as many elements (rows) as other entries
    
    When called, this function adds one key in the dictionary (which is equal to adding one column
    in the csv table. The name of the key is specified in key_str, and the elements are specified in
    key_list. Note that the latter has to be a list.
    If key_list has only one element, it is expanded according to the number of rows in the table.
    If the key is already existing, the original dictionary is returned without any modification.)
        
    **Parameters**
    
        :orig_dict:
            the dictionary to modify
        :key_str:
            string containing name of the dict key
        :key_list:
            either list containing all elements or
            list with one element which is appended n times
    
    :Author: Sirko Straube
    :Created: 2011/04/20
    """
    
    import copy, warnings

    if orig_dict.has_key(key_str):
        warnings.warn('Key to be added is already existing: Key ' + key_str + '. Adding canceled!')
        return orig_dict
    
    n=len(orig_dict[orig_dict.keys()[0]]) #determine number of entries per column
    n_newlist=len(key_list)
    
    if n_newlist == 1: #create a list of n entries with the same content
        key_list=key_list*n
    elif n_newlist != n:
        warnings.warn('Length of new entry (n=' + str(n_newlist) + ') does not match length of other entries (n=' + str(n) + ')!')
    
    orig_dict[key_str]=copy.deepcopy(key_list)

    return orig_dict


def extend_dict(orig_dict, extension_dict, retain_unique_items=True):
    """ Extend one dictionary with another
    
    .. note:: This function returns a modified dictionary, even if the extension
        dictionary is completely different (i.e. there is no check if the
        extension makes sense to guarantee maximal functionality).
        
    **Parameters**
    
        :orig_dict:
            the dictionary to be extended and returned
        :extension_dict:
            the dictionary defining the extension
    
    :Author: Sirko Straube, Mario Michael Krell
    :Created: 2010/11/09
    """
    import copy, warnings
    
    if not len(orig_dict.keys()) == len(extension_dict.keys()) or \
            not all([key1 in extension_dict.keys() for key1 in orig_dict.keys()]):
        warnings.warn('Inconsistency while merging: ' +
                      'The two directories have different keys!')
    
    current_num_entries = len(orig_dict[orig_dict.keys()[0]])
    for key in extension_dict.keys():
        if orig_dict.has_key(key):
            orig_dict[key].extend(copy.deepcopy(extension_dict[key]))
        elif retain_unique_items:
            orig_dict[key] = current_num_entries*[None]
            orig_dict[key].extend(copy.deepcopy(extension_dict[key]))
            warnings.warn('Key ' + key +
                          ' retained during dictionary extension:' +
                          ' Does not exist in all files!')
        else:
            warnings.warn('Key ' + key +
                          ' dismissed during dictionary extension:' +
                          ' Does not exist in all files!')
    
    num_new_entries = len(extension_dict[extension_dict.keys()[0]])
    for key in orig_dict:
        if key in extension_dict:
            pass
        elif retain_unique_items:       
            orig_dict[key].extend(num_new_entries*[None])
            warnings.warn('Key ' + key +
                          ' retained during dictionary extension:' +
                          ' Does not exist in all files!')
        else:
            warnings.warn('Key ' + key +
                          ' dismissed during dictionary extension:' +
                          ' Does not exist in all files!')
            orig_dict.pop(key)
        
    return orig_dict

def average_rows(data_dict, key_list, n=None, new_n=None):
    """ Average across all values of the specified columns 
    
    Reduces the number of rows, i.e., the number of values in the lists, by
    averaging all values of a specific key, e.g., across all splits or subjects.

    .. note::
        It is assumed that for two parameters A and B which have a and b
        different values the number of rows to average is a*b. If you have 
        certain constraints so that the number of rows to average is not a*b,
        you have to specify them explicitly. 
    
    **Parameters**
    
        :data_dict:
            Dictionary as returned by csv2dict.
        
        :key_list:
            List of keys (equals column names in a csv table) over which the
            average is computed.
            
        :n:
            Number of rows that are averaged. If None it is determined 
            automatically. default=None.
        
        :new_n:
            Number of rows after averaging. If None it is determined 
            automatically. default=None.
            
            
    """
    import warnings
    import numpy
    
    # check some special keys
    ignore_cols = []
    if "__Split__" in key_list:
        ignore_cols.append('__Key_Fold__')
    elif "__Key_Fold__" in key_list:
        ignore_cols.append('__Split__')
    if "__Run__" in key_list:
        ignore_cols.append('__Key_Run__')
    elif "__Key_Run__" in key_list:
        ignore_cols.append('__Run__')
    
    # determine dim of rows to average and result table
    if n is None:
        n = 1
        for key in key_list:
            n *= len(set(data_dict[key]))
    if new_n is None:
        new_n = len(data_dict[key_list[0]]) / n
    # averaging over *key* means all other parameter columns have to be the same    
    indices = [[] for _ in range(new_n)]
    patterns = [[] for _ in range(new_n)]
    values = [data_dict[key] for key in data_dict.keys() \
              if (key.startswith('__') and not (key in key_list or key in \
                                                                  ignore_cols))]
    # determine indices of rows that are averaged
    i = 0
    for pattern in zip(*values):
        inserted = False
        for j in range(new_n):
            if pattern in patterns[j]:
                indices[j].append(i)
                inserted = True
                break
            if patterns[j] == []:
                patterns[j].append(pattern)
                indices[j].append(i)
                inserted = True
                break
        if inserted != True:
            warnings.warn("Line %d not included in average! Check dimensions." % i)
        i += 1
    
    # average the data
    data_dict = parse_data(data_dict)
    result_dict = empty_dict(data_dict)
    for key in result_dict.keys():
        for avg_inds in indices:
            a = numpy.array(data_dict[key])[avg_inds]
            # we can only average numbers
            if isinstance(data_dict[key][0], (float,int)):
                # since int would be converted to float by averaging we try to
                # prevent that if possible
                if (a == a[0]).all():
                    result_dict[key].append(a[0])
                else:
                    result_dict[key].append(numpy.mean(a))
            elif key in key_list or key in ignore_cols:
                result_dict[key].append("averaged")
            else:
                result_dict[key].append(a[0])
                # check if not equal!
                if not((a == a[0]).all()):
                    warnings.warn("Averaged across different conditions... %s" % str(a))
    
    return result_dict

def parse_data(data_dict):
    """ Parse the data of type string to int and float values where possible 
    
    **Parameters**
    
        :data_dict:
            Dictionary as returned by csv2dict.
    """
    result_dict = empty_dict(data_dict)
    for key in data_dict.keys():
        for s in data_dict[key]:
            try:
                result_dict[key].append(int(s))
            except ValueError:
                try:
                    result_dict[key].append(float(s))
                except ValueError:
                    result_dict[key].append(s)
    return result_dict

def check_for_failures(data, num_splits, conditions, remove_count=False):
    """ Compute a list of conditions for which the classification failed
    
    Given a possibly incomplete results.csv and a set of parameters as defined
    in an operation.yaml, this function compares all the expected combinations
    of parameters with what has actually been evaluated according to
    results.csv. It returns a list of failures, i.e., a list of dictionaries,
    each representing one combination of parameters for which results are
    missing.
    
    Besides the actual parameters, the dictionaries in failures have one
    additional key 'count'. The value of 'count' is the number of times this
    particular parameter setting occurred in the results file. The expected
    number of occurrences is the number of splits, 'num_splits'. If the
    failures list is to be further used, it might be necessary to remove the
    count key again - if remove_count=True, this will be done automatically.
    
    .. note:: Even though __Dataset__ is not explicitly stated in the
        operation.yaml, this function needs you to specify the collections as
        parameter all the time. See the following example.
    
    .. note:: This implementation is highly inefficient as it just loops through
        the results list and the list of expected parameter settings instead of
        making use of any sophisticated search algorithms. Large problem might
        thus take some time.

    **Parameters**
    
        :data:
            Dictionary as returned by csv2dict. Usually this dictionary should
            contain the (incomplete) analysis results, hence it will in most
            cases be the product of something like csv2dict('results.csv').
        :num_splits:
            Number of splits. The decision if the condition is interpreted as
            failure depends on this parameter.
        :conditions:
            A dictionary containing the parameter ranges as specified in the
            operation.yaml. Additionally, __Dataset__ has to be specified. See
            the following example.
        :remove_count:
            optional: controls if the count variable will be removed from the
            entries in the failures list.
            default=False
    
    
    ** Examplary Workflow **

    .. code-block:: python

        import csv_analysis
        data=csv_analysis.csv2dict('results.csv')
        conditions={}
        conditions['__CLASSIFIER__']=['1RMM', '2RMM']
        conditions['__C__']=[0.01, 0.1, 1.0, 10.0]
        conditions['__Dataset__']=['Set1','Set2','Set3']
        nsplits = 10
        failures=csv_analysis.check_for_failures(data,nsplits,conditions,True)

    :Author: David Feess
    :Created: 2011/04/05
    """
    # This is used to generate crossproducts of arbitrary many parameters and
    # stolen as is from missions/operations.base._get_parameter_space()
    crossproduct = lambda ss,row=[],level=0: len(ss)>1 \
        and reduce(lambda x,y:x+y,[crossproduct(ss[1:],row+[i],level+1)
                                         for i in ss[0]]) \
        or [row+[i] for i in ss[0]]
    parameter_ranges = [eval(range_expression)
                            if isinstance(range_expression, basestring)
                            else range_expression
                                for range_expression in conditions.values()]
    # parameter_settings will contain a list with dict entries, each dict
    # representing one particular combination of parameters
    parameter_settings = map(lambda x: dict(zip(conditions.keys(), x)),
                                    crossproduct(parameter_ranges))

    # Add a counter variable to each of the expected conditions. This will
    # later be compared to num_splits
    for x in parameter_settings:
        x['count'] = 0
    
    # Iterate through entire data object
    for i in range(len(data['__Dataset__'])):
        # Iterate through expected parameter settings:
        for expected in parameter_settings:
            skip = False # skip this setting if any parameter mismatches
            # iterate through all parameters in this parameter setting
            for expected_key, expected_val in expected.iteritems():
                if expected_key == 'count': # forget about the count parameter
                    continue
                try: # convert strings to numbers if possible
                    x = eval(data[expected_key][i])
                except:
                    x = data[expected_key][i]
                if expected_val == x: # if we have a match continue
                    continue
                else: # else skip this parameter
                    skip = True # ...  and the whole param. setting
                    break
            if skip: # go for next parameter setting
                continue
            # if not skip: found a match: ...
            expected['count'] += 1 # ... increase count
            break                  # and go for next entry in data

    failures = []
    # Failures are all entries in the expected parameter_settings where count
    # does not equal the number of splits
    for x in parameter_settings:
        if x['count'] != num_splits:
            if remove_count:
                x.pop('count')
            failures.append(x)

    return failures

def check_op_libSVM(input_dir='.', delete_file=True):
    """Perform terminal operation to identify possible classification failures
       on the basis of number of files.
    
    This works only for libSVM classification with stored results, as it
    relies on files stored in the persistency directories.
    
    This function navigates to input_dir (which is the result directory of the
    classification) and checks the number of files starting with 'features' in
    'persistency_run0/LibSVMClassifierNode/' in each subdirectory. In case the
    classification was successfully performed, the number of files here should
    equal the number of splits used. If not, this is a hint that something
    went wrong!
    The list returned by this function contains alternating
    (i) name of 'root directory' for the respective condition
    (ii) number of files
    ...
    
    .. note:: This function only works if the feature*.pickle files are
              explicitly saved in your NodeChain!

    
    **Parameters**
    
        :input_dir:
            optional: string with the path where csv files are stored.
            default='.'
        :delete_file:
            optional: controls if the file 'temp_check_op.txt' will be removed
            default=True
    
    :Author: Sirko Straube, Anett Seeland
    :Created: 2010/11/09
    """
    
    import os
    
    #navigating to operation dir
    current_path=os.getcwd()
    os.chdir(input_dir)
    #rcode=os.system('cd ' + input_dir)
    #analyzing directories and writing results in temp_check_op.txt
    rcode=os.system('for f in *; do if [ -d $f ]; then echo $f; ' + 
      'echo find $f/persistency_run0/LibSVMClassifierNode/feature*.pickle ' + 
      '| wc -w; fi; done > temp_check_op.txt')
    
    #transferring data to python list
    f=open('temp_check_op.txt')
    oplist=[]
    
    for line in f:
        oplist.append(line)
    
    f.close()        
    
    #probably deleting and navigating back
    if delete_file:
        rcode=os.system('rm temp_check_op.txt')
    rcode=os.system('cd ' + current_path)
    
    return oplist



def report_failures(oplist, num_splits):
    """Sort output of terminal operation (e.g. performed by check_op_libSVM).

    This function returns a list where each element contains the parameters of
    a condition where the classification probably failed. This judgment is
    made according to the number of files which are expected according to the
    used number of splits. See also: check_op_libSVM
    
    **Parameters**
    
        :oplist:
            An iterable that has to contain
            (i) name of 'root directory' for the respective condition
            (ii) number of files
            ...

        This parameter can either be the list returned by check_op_libSVM or a
        file type object (pointing to a manually constructed file).

        :num_splits:
            Number of splits. The decision if the condition is interpreted as
            failure depends on this parameter.
    
    :Author: Mario Krell, Sirko Straube
    :Created: 2010/11/09
    """
    
    import warnings
    
    dirstats=False
    dirline = None
    failures=[]

    for line in oplist:
            if dirstats: #the actual line should contains the number of files
                #remove possible whitespaces and endl
                line=line.strip().strip('\n') 
                if line.isdigit():
                    #-1 because of batch command (see check_op_libSVM)
                    num_files=int(line)-1 
                    if num_files<num_splits:
                        result = dict()
                        current_params= \
                            dirline.strip().strip('{').strip("}").split("}{")
                        result['__Dataset__']=current_params[0]
                        result['count']=num_files #include number of splits
                        for param in current_params[1:]:
                            # TODO if anything else then template has no # this will fail;
                            # delete as soon as no more data with templates in folder names
                            # circulate
                            if '#' not in param:
                                result["__Template__"] = param
                                continue
                            entry =param.split('#')
                            result[entry[0]] = entry[1]
                        failures.append(result)
                else:
                    warnings.warn("Inconsistency while analyzing " + 
                      "check_op_libSVM data: Line " + line + 
                      " is not a digit reporting number of feature pickles." )
                dirstats=False
            else:   
                dirstats=True
                dirline=line
    return failures

def reconstruct_failures(csv_dict, missing_conds, num_splits, default_dict=None):
    """Reconstruct classification failures in csv dictionary according to
    known parameters and default values.

    This function takes the csv-dictionary (probably constructed using
    merge_performance_results from PerformanceResultSummary) and reconstructs the classification failures defined in
    missing_conds (probably constructed using report_failures) according to
    known parameters (given in missing_conds) and some default values that may
    be specified in default_dict (probably constructed with the help of
    empty_dict and a subsequent modification). All other keys are specified
    with the 'unknown' value. Finally the reconstructed dictionary is merged
    with the original csv-dictionary and returned.
    
    **Parameters**
    
        :csv_dict:
            The data dictionary. Has the form returned by csv2dict.

        :missing_conds:
            A list of dictionaries specifying the missing conditions.
            Has the form returned by report_failures.
        
        :num_splits:
            Number of splits used for classification.
        
        :default_dict:
            optional: A dictionary specifying default values for missing
            conditions. This dictionary can e.g. be constructed using
            empty_dict(csv_dict) and subsequent modification, e.g.
            default_dict['Metric'].append(0).

            (*optional, default: None*)
            
    :Author: Mario Krell, Sirko Straube
    :Created: 2010/11/09
    """
    
    reconstruct_dict = None
    
    for line in missing_conds:
        missing_dict = empty_dict(csv_dict)
        count = line.pop('count')
    
        for key in line.keys(): #transfer known variables to missing_dict
            missing_dict[key].append(line[key])
            
        if default_dict:
            #transfer user specified default values to missing_dict 
            for key in default_dict.keys():
                #...only if there is an entry in default_dict
                # AND the key is existing in missing_dict
                if default_dict[key] and key in missing_dict.keys(): 
                    missing_dict[key].append(default_dict[key])
        
        for key in missing_dict.keys(): #set all other keys to 'unknown'
            if not missing_dict[key]: #entry key is empty list
                missing_dict[key].append('unknown')
        #reconstruct a line for every missing split
        for each_missing_split in range(num_splits-count):
            if not reconstruct_dict: #only true once
                reconstruct_dict = missing_dict
            else:
                reconstruct_dict = extend_dict(reconstruct_dict,missing_dict)
    #finally, merge the original and the reconstruction
    return merge_dicts(csv_dict,reconstruct_dict) 
