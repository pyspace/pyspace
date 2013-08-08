# This Python file uses the following encoding: utf-8
""" Splits data into training and test data

.. todo:: Divide the node into splitting and data set filtering node.
"""
import random

from pySPACE.missions.nodes.base_node import BaseNode
from pySPACE.tools.memoize_generator import MemoizeGenerator

class TransferSplitterNode(BaseNode):
    """ Allow to split data into training and test data sets according to different window definitions
    
    Splits the available data into disjunct training and test sets. The transfer
    of different training and test window definitions is supported. The node 
    was implemented with several use cases in mind:
    
    - The training set contains instances of 'Standard' and 'Target' stimuli 
      but the test set of 'Target' and 'MissedTarget' stimuli.
      
    - The training set contains instances of 'LRP' with different training times
      and 'NoLRPs', but the test set should contain sliding windows. Cross
      validation should be supported to use the node together with parameter
      optimization node.
      
    - The use of merged data sets should be possible.
      
    **Parameters**
    
     :wdefs_train:
         A list with window definition names (specified in the window spec file
         when the raw data was segmented). All windows that belong to one of the
         window definition are considered when the training set(s) is(/are)
         determined.
         
     :wdefs_test:
         A list with window definition names (specified in the window spec file
         when the raw data was segmented). All windows that belong to one of the
         window definition are considered when the testing set(s) is(/are)
         determined.
         
     :split_method:
         One of the following Strings: 'all_data', 'time', 'count', 'set_flag'.
         
         - all_data    :
                             All possible data is used in every split. This 
                             results in splitting only window definitions that
                             occur in both, *wdefs_train* AND *wdefs_test*. 
                             Window definitions that only occur in either 
                             *wdefs_train* or *wdefs_test* are retained in every
                             split.

         - time        :
                             The data is sorted and split according to time.
                             For that (*start_time* of last window - 
                             *start_time* of first window)/*nr_of_splits*) is 
                             determined. Since time in eeg data is relative for 
                             every set, ensure that each input collection 
                             consists only of one data set (is not a merge of 
                             several sets) or that the change_option has been
                             used.

         - count      :
                             The data is split according to 
                             *num_split_instances*. By default only windows 
                             specified in both, *wdefs_train* and *wdefs_test*,
                             are count. With the parameter *wdefs_split* window
                             definition that are count can be specified.
                             If *num_split_instances* is not specified, *splits*
                             determines how many instances of *wdefs_split* are
                             in one split.

        - set_flag    :
                             When the data has been merged with the concatenate
                             operation before, a flag 'new_set' has been inserted
                             to the time series specs. Splits are based on this
                             flag, i.e. the splits behave like a inter-set
                             cross validation. For example you merged 3 sets: 
                             'A', 'B', 'C', then there are 3 splits generated:
                             'A'+'B' vs 'C', 'A'+'C' vs 'B' and 'B'+'C' vs 'A'.

     :random:
         If True, the data is randomized before splitting.
         
         .. note:: It is not guaranteed that overlapping windows will be in the
                   same split for split methods 'time' and 'all_data'!
                  
         (*optional, default: False*)
                  
     :splits:
         The number of splits created internally and the number of train-test
         pairs.
         
         (*optional, default: 10*)
         
     :num_split_instances:
         If *split_method* is 'count', *num_split_instances* specifies how many
         instances will be in one split. After splitting one split is evaluated
         according to *wdefs_test* for the test data set and the remaining 
         splits according to *wdefs_train*. The test split is iterated. If
         the total number of instances that are count is not divisible by 
         *num_split_instances* the last split will contain the remaining
         instances.
         If in addition *splits* is set to 1, only one train-test pair is 
         created with *num_split_instances* in the training set.
         
         (*optional, default: None*)
         
     :wdefs_split:
         A list with window definition names (specified in the window spec file
         when the raw data was segmented). All windows that belong to one of the
         window definition are counted when *split_method* was set to 'count'.
         
         (*optional, default: None*)
         
     :reverse:
         If this option is True, the data is split in reverse ordering.
         
         (*optional, default: False*)
    
    **Exemplary Call**    
    
    .. code-block:: yaml
    
        -
            node : TransferSplitter
            parameters :
                wdefs_train : ['s2', 's1']
                wdefs_test : ['s5', 's2']
                split_method : "all_data"
                splits : 5
    
    :Author: Anett Seeland (anett.seeland@dfki.de)
    :Created: 2011/04/10
    :LastChange: 2011/11/14 (traintest functionality)
    """
    
    def __init__(self, wdefs_train, wdefs_test, split_method, wdefs_train_test = None,
                 splits=10, random=False, num_split_instances=None, wdefs_split=None,
                 reverse=False, sort=False, *args, **kwargs):
        super(TransferSplitterNode, self).__init__(*args, **kwargs)
        
        if wdefs_train_test == None:
            wdefs_train_test = [wdef for wdef in \
                wdefs_train if wdef in wdefs_test],
            
        self.set_permanent_attributes(wdefs_train = wdefs_train, 
                                       wdefs_test = wdefs_test, 
                                     split_method = split_method,
                                           splits = splits, 
                                           random = random, 
                              num_split_instances = num_split_instances,
                                      wdefs_split = wdefs_split,
                                          reverse = reverse,
                                             sort = sort,
                                    current_split = 0,
                                 wdefs_train_test = wdefs_train_test,
                              split_indices_train = None,
                               split_indices_test = None)

    def is_split_node(self):
        """ Returns whether this is a split node. """
        return True

    def use_next_split(self):
        """ Use the next split of the data into training and test data.
        
        Returns True if more splits are available, otherwise False.
        
        This method is useful for benchmarking
        """
        if self.current_split + 1 < self.splits:
            self.current_split = self.current_split + 1
            self._log("Benchmarking with split %s/%s" % (self.current_split + 1,
                                                         self.splits))
            return True
        else:
            return False

    
    def train_sweep(self, use_test_data):
        """ Performs the actual training of the node.
        
        .. note:: Split nodes cannot be trained
        """
        raise Exception("Split nodes cannot be trained")
    
    def request_data_for_training(self, use_test_data):
        # Create split lazily when required
        if self.split_indices_train == None:
            self._create_split()
            
        # Create training data generator
        self.data_for_training = MemoizeGenerator(
             self.data[i] for i in self.split_indices_train[self.current_split])
        
        return self.data_for_training.fresh()
    
    def request_data_for_testing(self):
        # Create split lazily when required
        if self.split_indices_test == None:
            self._create_split()
        
        # Create test data generator
        self.data_for_testing = MemoizeGenerator(
              self.data[i] for i in self.split_indices_test[self.current_split])
        
        return self.data_for_testing.fresh()

    def _create_split(self):
        """ Create the split of the data into training and test data. """
        self._log("Splitting data into train and test data")
                  
        # Get training and test data
        # note: return the data in a list can double the memory requirements!
        train_data = list(self.input_node.request_data_for_training(
                                                         use_test_data = False))
        test_data = list(self.input_node.request_data_for_testing())
        
        # If there is already a  non-empty training set, 
        # it means that we are not the first split node in the node chain.
        if len(train_data) > 0:
            if len(test_data)==0:
                # If there was an All_Train_Splitter before, filter according
                # to wdef_train and return all training data
                self.split_indices_train = \
                            [[ind for ind, (win, lab) in enumerate(train_data) \
                                 if win.specs['wdef_name'] in self.wdefs_train]]
                self.split_indices_test = [[]]
                self.splits = 1
                self.data = train_data
                self._log("Using all data for training.")
                return
            else:    
                raise Exception("No iterated splitting of data sets allowed\n "
                            "(Calling a splitter on a data set that is already "
                            "splitted)")
        
        # Remember all the data and store it in memory
        # TODO: This might cause problems for large dataset
        self.data = train_data + test_data
        del train_data, test_data
        if self.reverse:
            self.data = self.data[::-1]
        
        # sort the data according to the start time
        if self.sort or self.split_method == 'time':
            self.data.sort(key=lambda swindow: swindow[0].start_time)
        # randomize the data if needed
        if self.random:
            r = random.Random(self.run_number)
            if self.split_method == 'set_flag':
                self.random = False
                # TODO: log this
            elif self.split_method == 'count':
                if self.wdefs_split == None:
                    self.wdefs_split = self.wdefs_train_test
                # divide the data with respect to the time
                data_time = dict()
                marker = -1
                last_window_endtime = 0
                for ind, (win, lab) in enumerate(self.data):
                    if win.start_time < last_window_endtime:
                        # overlapping windows or start of a new set
                        if win.end_time < last_window_endtime:
                            # new set
                            marker += 1
                            data_time[marker]=[(win,lab)]
                        else:
                            # overlapping windows
                            data_time[marker].append((win,lab))
                    else:
                        marker += 1
                        data_time[marker]=[(win,lab)]
                    last_window_endtime = win.end_time
                # randomize order of events by simultaneously keep the order of
                # sliding windows in each event
                data_random = data_time.values()
                r.shuffle(data_random)
                self.data = []
                for l in data_random: self.data.extend(l)
                del data_random, data_time, l
            else:
                r.shuffle(self.data)
            
        if self.split_method == 'all_data':
            # divide the data with respect to *wdef_train*, *wdef_test* and
            # *wdef_train_test*
            wdef_data = {'wdef_train_test':[],'wdef_train':[],'wdef_test':[]}
            class_labels = []
            for (index, (window, label)) in enumerate(self.data):
                if window.specs['wdef_name'] in self.wdefs_train_test:
                    wdef_data['wdef_train_test'].append(index)
                    if label not in class_labels:
                        class_labels.append(label)
                elif window.specs['wdef_name'] in self.wdefs_train:
                    wdef_data['wdef_train'].append(index)
                elif window.specs['wdef_name'] in self.wdefs_test:
                    wdef_data['wdef_test'].append(index)
                else:
                    import warnings
                    warnings.warn("Found window definition %s, which is " \
                                  "neither in *wdefs_train* nor in " \
                                  "*wdefs_test*. Window %s will be ignored!" \
                                  % (window.specs['wdef_name'],window.tag))
            # check if splitting makes sense
            if wdef_data['wdef_train_test']==[] and self.splits>1:
                raise Exception('No instances to split, i.e train-test window'\
                                ' definitions are disjunct!')
            split_indices_train = [[] for i in range(self.splits)]
            split_indices_test = [[] for i in range(self.splits)]
            # calculate splits
            if wdef_data['wdef_train_test']!=[]:
                data_size = len(wdef_data['wdef_train_test'])

                # ensure stratified splits if there are several classes 
                if len(class_labels)>1:
                    # divide the data with respect to the class_label 
                    data_labeled = dict()
                    for index in wdef_data['wdef_train_test']:
                        if not data_labeled.has_key(self.data[index][1]):
                            data_labeled[self.data[index][1]] = [index]
                        else:
                            data_labeled[self.data[index][1]].append(index)   
                    
                    # have not more splits than instances of every class!
                    min_nr_per_class = min([len(data) for data in \
                                                         data_labeled.values()])
                    if self.splits > min_nr_per_class:
                        self.splits = min_nr_per_class
                        self._log("Reducing number of splits to %s since no " \
                                  "more instances of one of the classes are " \
                                  "available." % self.splits)

                    # determine the splits of the data    
                    for label, indices in data_labeled.iteritems():
                        data_size = len(indices)
                        for j in range(self.splits):
                            split_start = \
                                      int(round(float(j)*data_size/self.splits))
                            split_end = \
                                    int(round(float(j+1)*data_size/self.splits))
                            split_indices_test[j].extend([i for i in indices[split_start: split_end]\
                                                if self.data[i][0].specs['wdef_name'] in self.wdefs_test])
                            split_indices_train[j].extend([i for i in indices \
                                             if i not in split_indices_test[j]])
                else: # len(class_labels) == 1
                    # have not more splits than instances!
                    if self.splits > data_size:
                        self.splits = data_size
                        self._log("Reducing number of splits to %s since no " \
                                  "more instances of one of the classes are " \
                                  "available." % self.splits)

                    # determine the splits of the data    
                    for j in range(self.splits):
                        split_start = \
                                  int(round(float(j)*data_size/self.splits))
                        split_end = \
                                int(round(float(j+1)*data_size/self.splits))
                        # means half-open interval [split_start, split_end)
                        split_indices_test[j].extend(
                            wdef_data['wdef_train_test'][split_start:split_end])
                        split_indices_train[j].extend([i for i in \
                                             wdef_data['wdef_train_test'] if i \
                                                  not in split_indices_test[j]])
            for i in range(self.splits):
                split_indices_train[i].extend(wdef_data['wdef_train'])
                split_indices_test[i].extend(wdef_data['wdef_test'])
                    
        elif self.split_method == 'time': 
            first_window_start = self.data[0][0].start_time
            last_window_start = self.data[-1][0].start_time
            # ensure, that time can never be greater than self.splits*time!
            time = round((last_window_start-first_window_start)/self.splits+0.5)
            # divide the data according to the time
            data_time = {0: []}
            time_fold = 0
            for (index, (window, label)) in enumerate(self.data):
                if window.start_time > time_fold*time+time:
                    time_fold += 1
                    data_time[time_fold]=[index]
                else:
                    data_time[time_fold].append(index)
                    
            split_indices_train = [[] for i in range(self.splits)]
            split_indices_test = [[] for i in range(self.splits)]
            for i in range(self.splits):
                split_indices_test[i].extend([index for index in data_time[i] \
                                    if self.data[index][0].specs['wdef_name'] \
                                                            in self.wdefs_test])
                for j in range(self.splits):
                    split_indices_train[i].extend([index for index in data_time[j] \
                            if j != i and self.data[index][0].specs['wdef_name'] \
                                                           in self.wdefs_train])
        elif self.split_method == 'count':
            if self.wdefs_split == None:
                self.wdefs_split = self.wdefs_train_test
            if self.num_split_instances == None:
                l = len([ind for ind, (win, lab) \
                             in enumerate(self.data) if win.specs['wdef_name'] \
                             in self.wdefs_split])
                self.num_split_instances = round(float(l)/self.splits)
            # divide the data according to *num_split_instances*
            data_count = {0:[]}
            count = -1
            count_fold = 0
            if self.splits==1 and len([i for i in range(len(self.data)) \
                    if self.data[i][0].specs['wdef_name'] in self.wdefs_split])\
                        == self.num_split_instances:
                train_end = len(self.data)
            else:
                for (ind, (win, lab)) in enumerate(self.data):
                    #print ind, win.specs['wdef_name'], lab
                    if win.specs['wdef_name'] in self.wdefs_split:
                        count += 1
                        if self.splits == 1 and \
                                              count == self.num_split_instances:
                            train_end = ind
                            break
                        if count != 0 and count % self.num_split_instances == 0:
                            count_fold += 1
                            data_count[count_fold] = [ind]
                        else:
                            data_count[count_fold].append(ind)
                    else:
                        data_count[count_fold].append(ind)
                 
            if self.splits != 1:
                # self.num_split_instances*self.splits < l, but in the case 
                # when only num_split_instances is specified we can not trust 
                # self.splits
                if len(data_count.keys()) == self.splits+1 or \
                        (len(data_count.keys())-1)*self.num_split_instances > l:
                    data_count[count_fold-1].extend(data_count[count_fold])
                    del data_count[count_fold] 
                
                self.splits = len(data_count.keys())
                
                split_indices_train = [[] for i in range(self.splits)]
                split_indices_test = [[] for i in range(self.splits)]
            
                for i in range(self.splits):
                    split_indices_test[i].extend([ind for ind in data_count[i] \
                                       if self.data[ind][0].specs['wdef_name'] \
                                                            in self.wdefs_test])
                    for j in range(self.splits):
                        split_indices_train[i].extend([ind for ind in data_count[j]\
                            if j != i and self.data[ind][0].specs['wdef_name'] \
                                                           in self.wdefs_train])
            else: # self.splits == 1
                split_indices_train = \
                    [[ind for ind in range(len(self.data[:train_end])) if \
                      self.data[ind][0].specs['wdef_name'] in self.wdefs_train]]
                split_indices_test = \
                    [[ind for ind in range(train_end,len(self.data)) if \
                       self.data[ind][0].specs['wdef_name'] in self.wdefs_test]]   
 
                    
        elif self.split_method == 'set_flag':
            # divide the data according to *new_set* flag in time series specs
            data_set = {0:[]}
            key_fold = 0
            for (ind, (win, lab)) in enumerate(self.data):
                if win.specs['new_set']:
                    key_fold += 1
                    data_set[key_fold]=[ind]
                else:
                    data_set[key_fold].append(ind)
                    
            self.splits = len(data_set.keys())
            
            split_indices_train = [[] for i in range(self.splits)]
            split_indices_test = [[] for i in range(self.splits)]
            for i in range(self.splits):
                split_indices_test[i].extend([ind for ind in data_set[i] \
                                    if self.data[ind][0].specs['wdef_name'] \
                                                            in self.wdefs_test])
                for j in range(self.splits):
                    split_indices_train[i].extend([ind for ind in data_set[j] \
                            if j != i and self.data[ind][0].specs['wdef_name'] \
                                          in self.wdefs_train])
            
        self.split_indices_train = split_indices_train
        self.split_indices_test = split_indices_test
        
        self._log("Benchmarking with split %s/%s" % (self.current_split + 1,
                                                     self.splits))

