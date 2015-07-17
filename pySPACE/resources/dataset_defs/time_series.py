""" Load and store data of the type :mod:`pySPACE.resources.data_types.time_series` """
import itertools

import copy
import os
import cPickle
import sys
import scipy
import yaml
import csv
import numpy
import logging
import warnings
import glob
from pySPACE.missions.support.WindowerInterface import AbstractStreamReader
from pySPACE.missions.support.windower import MarkerWindower
from pySPACE.tools.filesystem import get_author
from pySPACE.resources.dataset_defs.base import BaseDataset


class TimeSeriesDataset(BaseDataset):
    """ Loading and storing a time series dataset
    
    This class encapsulate most relevant code for dealing with time series 
    datasets, most importantly for loading and storing them to the file system.
    
    These datasets consist of
    :mod:`~pySPACE.resources.data_types.time_series` objects.
    They can be loaded with a
    :mod:`~pySPACE.missions.nodes.source.time_series_source` and saved with a
    :mod:`~pySPACE.missions.nodes.sink.time_series_sink` node
    in a :class:`~pySPACE.missions.operations.node_chain.NodeChainOperation`.
    
    The standard *storage_format* is 'pickle',
    but it is also possible to load, e.g.,
    BrainComputerInterface-competition data. For that, ``storage_format`` has
    to be set in the format **bci_comp_[competition number]_[dataset number]**
    in the metadata.yaml file. For example, **bci_comp_2_4** means loading of 
    time series from BCI Competition II (2003), dataset IV.
    Currently, the following datasets can be loaded:
    
        - BCI Competition II, dataset IV: self-paced key typing (left vs. right)
        - BCI Competition III, dataset II: P300 speller paradigm, training data
    
    See http://www.bbci.de/competition/ for further information.

    For saving the data, other formats are currently supported but not
    yet for loading the data.
    This issue can be handled by processing the data with a node chain
    operation which transforms the data into feature vectors and
    use the respective storing and loading functionality, e.g., with csv and
    arff files.
    There is also a node for transforming feature vectors back to
    TimeSeries objects.

    
    **Parameters**
    
      :dataset_md:
          A dictionary with all the meta data.
          
          (*optional, default: None*)
          
      :sort_string: 
          A lambda function string that is evaluated before the data is stored.

          (*optional, default: None*)
          
    **Known issues**
        The BCI Competition III dataset II should be actually loaded as a 
        streaming dataset to enable different possibilities for windowing.
        Segment ends (i.e., where a new letter starts) can be coded as marker.
    """
    def __init__(self, dataset_md=None, sort_string=None, **kwargs):
        super(TimeSeriesDataset, self).__init__(dataset_md=dataset_md)
        self.stream_mode = False
        if dataset_md is not None:
            dataset_dir = self.meta_data["dataset_directory"]
            s_format = self.meta_data["storage_format"]
            if type(s_format) == list:
                s_format = s_format[0]
            # Loading depends on whether data is split into
            # training and test data, whether different splits exist and whether
            # several runs have been conducted.
            if s_format == "pickle" and not self.meta_data["train_test"] \
                    and self.meta_data["splits"] == 1 \
                    and self.meta_data["runs"] == 1:
                # The dataset consists only of a single set of data, for
                # one run, one splitting, and only test data
                data = dataset_md["data_pattern"].replace("_run", "_run0") \
                    .replace("_sp", "_sp0") \
                    .replace("_tt", "_test")
                # File that contains the time series objects
                ts_file = os.path.join(dataset_dir, data)
                # Current data will be loaded lazily
                self.data[(0, 0, "test")] = ts_file
            elif s_format == "pickle":
                for run_nr in range(self.meta_data["runs"]):
                    for split_nr in range(self.meta_data["splits"]):
                        for train_test in ["train", "test"]:
                            # The collection consists only of a single set of
                            # data, for one run, one splitting,
                            # and only test data
                            data = dataset_md["data_pattern"]\
                                .replace("_run", "_run%s" % run_nr) \
                                .replace("_sp", "_sp%s" % split_nr) \
                                .replace("_tt", "_%s" % train_test)
                            # File that contains the time series objects
                            ts_file = os.path.join(dataset_dir,data)
                            # Actual data will be loaded lazily
                            self.data[(run_nr, split_nr, train_test)] = ts_file
            elif s_format.startswith("bci_comp"):
                # get bci competion and dataset number
                try:
                    self.comp_number, self.comp_set = s_format.split('_')[2:]
                except Exception:
                    raise Exception, "%s --- Could not extract BCI competition"\
                                     " and dataset number!" % s_format
                if self.comp_number == "2":
                    if self.comp_set == "4":
                        
                        def _update_sf(self, file_name):
                            if '1000' in file_name:
                                self.sf = 1000
                            else:
                                self.sf = 100
                            self.meta_data["sampling_frequency"] = self.sf
                            
                        # structure: 2 mat file with data in different sampling
                        # frequencies; txt file for test labels
                        if "sampling_frequency" in self.meta_data.keys() and \
                                "file_name" in self.meta_data.keys():
                            # are they inconsistent?
                            self.sf = self.meta_data["sampling_frequency"]
                            if (self.sf == 100 and '1000' in \
                                               self.meta_data["file_name"]) or \
                                    (self.sf == 1000 and '1000' not in \
                                               self.meta_data["file_name"]):
                                warnings.warn("File name does not match "
                                        "sampling frequency or vice versa. %s "
                                        "is loaded." % self.meta_data["file_name"])
                                self._update_sf(self.meta_data["file_name"])
                            ts_file = os.path.join(dataset_dir,
                                                    self.meta_data["file_name"])
                        elif "file_name" in self.meta_data.keys():
                            self._update_sf(self.meta_data["file_name"])
                            ts_file = os.path.join(dataset_dir,
                                                    self.meta_data["file_name"])
                        elif "sampling_frequency" in self.meta_data.keys():
                            self.sf = self.meta_data["sampling_frequency"]
                            if self.sf == 1000:
                                ts_file = os.path.join(dataset_dir,
                                                           "sp1s_aa_1000Hz.mat")
                            else:
                                ts_file = os.path.join(dataset_dir, 
                                                                  "sp1s_aa.mat")
                        else:
                            ts_file = glob.glob(os.path.join(dataset_dir,
                                                                    "*.mat"))[0]
                            warnings.warn("Either file name nor sampling "
                                  "frequency is given. %s is loaded." % ts_file)
                            self._update_sf(ts_file)
                        self.data[(0, 0, "test")] = ts_file
                        self.data[(0, 0, "train")] = ts_file
                    else:
                        raise NotImplementedError("Loading of BCI competition" \
                                              " %s, dataset %s not supported " \
                                            % (self.comp_number, self.comp_set))
                elif self.comp_number == "3":
                    if self.comp_set == "2":
                        # structure: mat file for train and test data
                        # TODO: loading test labels is not possible at the moment!
                        #     glob.glob(os.path.join(dataset_dir,"*Test.mat"))[0]
                        #self.data[(0, 0, "train")] = \
                        self.data[(0, 0, "test")] = \
                            glob.glob(os.path.join(dataset_dir,"*Train.mat"))[0]
                    else:
                        raise NotImplementedError("Loading of BCI competition" \
                                              " %s, dataset %s not supported " \
                                            % (self.comp_number, self.comp_set))
                else:
                    raise NotImplementedError("Loading of BCI competition %s," \
                                              " dataset %s not supported " \
                                            % (self.comp_number, self.comp_set))
                        
            else:  # s_format=="csv":
                if "file_name" in self.meta_data.keys():
                    ts_file = os.path.join(dataset_dir,
                                           self.meta_data["file_name"])
                elif "data_pattern" in self.meta_data.keys():
                    # The collection consists only of a single set of data, for
                    # one run, one splitting, and only test data
                    data = dataset_md["data_pattern"].replace("_run", "_run0") \
                        .replace("_sp","_sp0") \
                        .replace("_tt","_test")
                    ts_file = os.path.join(dataset_dir,data)
                elif os.path.isfile(os.path.join(dataset_dir,"data.csv")):
                    ts_file = os.path.join(dataset_dir,"data.csv")
                else:
                    pathlist = glob.glob(os.path.join(dataset_dir,"*.csv"))
                    if len(pathlist)>1:
                        warnings.warn(
                            "To many given data sets:%s. Taking first entry."
                            % str(pathlist))
                        ts_file = pathlist[0]
                    elif len(pathlist) == 0:
                        warnings.warn("No csv file found. Trying any file.")
                        pathlist = glob.glob(os.path.join(dataset_dir, "*"))
                        ts_file = pathlist[0]
                        if "metadata.yaml" in ts_file:
                            ts_file = pathlist[1]
                self.data[(0, 0, "test")] = ts_file
        self.sort_string = sort_string if sort_string is not None else None

    def get_data(self, run_nr, split_nr, train_test):
        """ Return the train or test data for the given split in the given run.
        
        **Parameters**
          
          :run_nr: The number of the run whose data should be loaded.
          
          :split_nr: The number of the split whose data should be loaded.
          
          :train_test: "train" if the training data should be loaded.
                       "test" if the test data should be loaded.
    
        """
        # Do lazy loading of the time series objects.
        if isinstance(self.data[(run_nr, split_nr, train_test)], basestring):
            self._log("Lazy loading of %s time series windows from input "
                      "collection for run %s, split %s." % (train_test, run_nr, 
                                                            split_nr))
            s_format = self.meta_data["storage_format"]
            if type(s_format) == list:
                s_format = s_format[0]
            if s_format == "pickle":
                # Load the time series from a pickled file
                f = open(self.data[(run_nr, split_nr, train_test)], 'r')
                try:
                    self.data[(run_nr, split_nr, train_test)] = cPickle.load(f)
                except ImportError:
                    # code for backward compatibility
                    # redirection of old path
                    f.seek(0)
                    self._log("Loading deprecated data. Please transfer it " +
                              "to new format.",level=logging.WARNING)
                    from pySPACE.resources.data_types import time_series
                    sys.modules['abri_dp.types.time_series'] = time_series
                    self.data[(run_nr, split_nr, train_test)] = cPickle.load(f)
                    del sys.modules['abri_dp.types.time_series']
                f.close()
            elif s_format.startswith("bci_comp"):
                from scipy.io import loadmat
                from pySPACE.resources.data_types.time_series import TimeSeries
                if self.comp_number == "2":
                    if self.comp_set == "4":
                        ts_fname = self.data[(run_nr, split_nr, train_test)]
                        d = loadmat(ts_fname)
                        channel_names = [name[0].astype('|S3') for name in \
                                                                   d["clab"][0]]
                        if train_test == "train":
                            self.data[(run_nr, split_nr, train_test)] = []
                            input_d = d["x_train"]
                            input_l = d["y_train"][0]
                            for i in range(input_d.shape[2]):
                                self.data[(run_nr, split_nr, 
                                           train_test)].append(\
                                            (TimeSeries(input_d[:,:,i],
                                                 channel_names, float(self.sf)), 
                                        "Left" if input_l[i] == 0 else "Right"))
                        else:
                            label_fname = glob.glob(os.path.join(
                                          os.path.dirname(ts_fname),"*.txt"))[0]
                            input_d = d["x_test"]
                            input_l = open(label_fname,'r')
                            self.data[(run_nr, split_nr, train_test)] = []
                            for i in range(input_d.shape[2]):
                                label = int(input_l.readline())
                                self.data[(run_nr, split_nr, 
                                           train_test)].append(\
                                            (TimeSeries(input_d[:,:,i],
                                                 channel_names, float(self.sf)), 
                                             "Left" if label == 0 else "Right"))
                elif self.comp_number == "3":
                    if self.comp_set == "2":
                        data = loadmat(self.data[(run_nr, split_nr, train_test)])
                        signal = data['Signal']
                        flashing = data['Flashing']
                        stimulus_code = data['StimulusCode']
                        stimulus_type = data['StimulusType']
                
                        window = 240
                        Fs = 240
                        channels = 64
                        epochs = signal.shape[0]
                        self.data[(run_nr, split_nr, train_test)] = []
                        self.start_offset_ms = 1000.0
                        self.end_offset_ms = 1000.0
                        
                        whole_len = (self.start_offset_ms + self.end_offset_ms)*Fs/1000.0 + window
                        responses = numpy.zeros((12, 15, whole_len, channels))
                        for epoch in range(epochs):
                            rowcolcnt=numpy.ones(12)
                            for n in range(1, signal.shape[1]):
                                if (flashing[epoch,n]==0 and flashing[epoch,n-1]==1):
                                    rowcol=stimulus_code[epoch,n-1]
                                    if n-24-self.start_offset_ms*Fs/1000.0 < 0:
                                        temp = signal[epoch,0:n+window+self.end_offset_ms*Fs/1000.0-24,:]
                                        temp = numpy.vstack((numpy.zeros((whole_len - temp.shape[0], temp.shape[1])), temp))
                                    elif n+window+self.end_offset_ms*Fs/1000.0-24> signal.shape[1]:
                                        temp = signal[epoch,n-24-self.start_offset_ms*Fs/1000.0:signal.shape[1],:]
                                        temp = numpy.vstack((temp, numpy.zeros((whole_len-temp.shape[0], temp.shape[1]))))
                                    else:
                                        temp = signal[epoch, n-24-self.start_offset_ms*Fs/1000.0:n+window+self.end_offset_ms*Fs/1000.0-24, :]
                                    responses[rowcol-1,rowcolcnt[rowcol-1]-1,:,:]=temp
                                    rowcolcnt[rowcol-1]=rowcolcnt[rowcol-1]+1
                
                            avgresp=numpy.mean(responses,1)
                
                            targets = stimulus_code[epoch,:]*stimulus_type[epoch,:]
                            target_rowcol = []
                            for value in targets:
                                if value not in target_rowcol:
                                    target_rowcol.append(value)
                
                            target_rowcol.sort()
                
                            for i in range(avgresp.shape[0]):
                                temp = avgresp[i,:,:]
                                data = TimeSeries(input_array = temp,
                                                  channel_names = range(64), 
                                                  sampling_frequency = window)
                                if i == target_rowcol[1]-1 or i == target_rowcol[2]-1:
                                    self.data[(run_nr, split_nr, train_test)].append((data,"Target"))
                                else:
                                    self.data[(run_nr, split_nr, train_test)].append((data,"Standard"))                 
        if self.stream_mode and not self.data[(run_nr, split_nr, train_test)] == []:
            # Create a connection to the TimeSeriesClient and return an iterator
            # that passes all received data through the windower.
            self.reader = TimeSeriesClient(self.data[(run_nr, split_nr, train_test)], blocksize=100)

            # Creates a windower that splits the training data into windows
            # based in the window definitions provided
            # and assigns correct labels to these windows
            self.reader.set_window_defs(self.window_definition)
            self.reader.connect()
            self.marker_windower = MarkerWindower(
                self.reader, self.window_definition,
                nullmarker_stride_ms=self.nullmarker_stride_ms,
                no_overlap=self.no_overlap,
                data_consistency_check=self.data_consistency_check)
            return self.marker_windower
        else:
            return self.data[(run_nr, split_nr, train_test)]

    def store(self, result_dir, s_format="pickle"):
        """ Stores this collection in the directory *result_dir*.
        
        In contrast to *dump* this method stores the collection
        not in a single file but as a whole directory structure with meta
        information etc. The data sets are stored separately for each run, 
        split, train/test combination.
        
        **Parameters**
        
          :result_dir:
              The directory in which the collection will be stored.
              
          :name:
              The prefix of the file names in which the individual data sets are 
              stored. The actual file names are determined by appending suffixes
              that encode run, split, train/test information. 
              
              (*optional, default: "time_series"*)
              
          :format:
              The format in which the actual data sets should be stored.
              
              Possible formats are *pickle*, *text*, *csv* and *MATLAB* (.mat)
              format.

              In the MATLAB and text format, all time series objects are
              concatenated to a single large table containing only integer
              values.
              For the csv format comma separated values are taken as default
              or a specified Python format string.
              
              The MATLAB format is a struct that contains the data, the
              sampling frequency and the channel names.
              
              .. note:: For the text and MATLAB format, markers could be added 
                        by using a Marker_To_Mux node before
              
              (*optional, default: "pickle"*)

        .. todo:: Put marker to the right time point and also write marker channel.
        """
        name = "time_series"
        if type(s_format) == list:
            s_type = s_format[1]
            s_format = s_format[0]
        else:
            s_type = "%.18e"
        if s_format in ["text", "matlab"]:
            s_type = "%i"
        if s_format == "csv" and s_type == "real":
            s_type = "%.18e"
        # Update the meta data
        author = get_author()
        self.update_meta_data({"type": "time_series",
                               "storage_format": s_format,
                               "author": author,
                               "data_pattern": "data_run" + os.sep 
                                               + name + "_sp_tt." + s_format})

        # Iterate through splits and runs in this dataset
        for key, time_series in self.data.iteritems():
            # load data, if necessary 
            # (due to the  lazy loading, the data might be not loaded already)
            if isinstance(time_series, basestring):
                time_series = self.get_data(key[0], key[1], key[2])
            if self.sort_string is not None:
                time_series.sort(key=eval(self.sort_string))
            # Construct result directory
            result_path = result_dir + os.sep + "data" + "_run%s" % key[0]
            if not os.path.exists(result_path):
                os.mkdir(result_path)
            
            key_str = "_sp%s_%s" % key[1:]
            # Store data depending on the desired format
            if s_format in ["pickle", "cpickle", "cPickle"]:
                result_file = open(os.path.join(result_path,
                                                name+key_str+".pickle"), "w")
                cPickle.dump(time_series, result_file, cPickle.HIGHEST_PROTOCOL)
            elif s_format in ["text","csv"]:
                self.update_meta_data({
                    "type": "stream",
                    "marker_column": "marker"})
                result_file = open(os.path.join(result_path,
                                                name + key_str + ".csv"), "w")
                csvwriter = csv.writer(result_file)
                channel_names = copy.deepcopy(time_series[0][0].channel_names)
                if s_format == "csv":
                    channel_names.append("marker")
                csvwriter.writerow(channel_names)
                for (data, key) in time_series:
                    if s_format == "text":
                        numpy.savetxt(result_file, data, delimiter=",", fmt=s_type)
                        if not key is None:
                            result_file.write(str(key))
                            result_file.flush()
                        elif data.marker_name is not None \
                                and len(data.marker_name) > 0:
                            result_file.write(str(data.marker_name))
                            result_file.flush()
                    else:
                        first_line = True
                        marker = ""
                        if not key is None:
                            marker = str(key)
                        elif data.marker_name is not None \
                                and len(data.marker_name) > 0:
                            marker = str(data.marker_name)
                        for line in data:
                            l = list(line)
                            l.append(marker)
                            csvwriter.writerow(list(l))
                            if first_line:
                                first_line = False
                                marker = ""
                        result_file.flush()
            elif s_format in ["mat"]:
                result_file = open(os.path.join(result_path,
                                                name + key_str + ".mat"),"w")
                # extract a first time series object to get meta data 
                merged_time_series = time_series.pop(0)[0]
                # collect all important information in the collection_object
                collection_object = {
                    "sampling_frequency": merged_time_series.sampling_frequency,
                    "channel_names": merged_time_series.channel_names}

                # merge all data 
                for (data,key) in time_series:
                    merged_time_series = numpy.vstack((merged_time_series,
                                                       data))
                collection_object["data"] = merged_time_series 
                mdict = dict()
                mdict[name + key_str] = collection_object 
                import scipy.io
                scipy.io.savemat(result_file, mdict=mdict)
            elif s_format in ["eeg"]:

                result_file = open(os.path.join(result_path,
                                                name + key_str + ".eeg"),"a+")
                result_file_mrk = open(os.path.join(result_path,
                                                name + key_str + ".vmrk"),"w")

                result_file_mrk.write("Brain Vision Data Exchange Marker File, "
                                      "Version 1.0\n")
                result_file_mrk.write("; Data stored by pySPACE\n")
                result_file_mrk.write("[Common Infos]\n")
                result_file_mrk.write("Codepage=UTF-8\n")
                result_file_mrk.write("DataFile=%s\n" %
                                      str(name + key_str + ".eeg"))
                result_file_mrk.write("\n[Marker Infos]\n")

                markerno = 1
                datapoint = 1
                sf = None
                channel_names = None

                for t in time_series:
                    if sf is None:
                        sf = t[0].sampling_frequency
                    if channel_names is None:
                        channel_names = t[0].get_channel_names()
                    for mrk in t[0].marker_name.keys():
                        for tm in t[0].marker_name[mrk]:
                            result_file_mrk.write(str("Mk%d=Stimulus,%s,%d,1,0\n" %
                                (markerno, mrk, datapoint+(tm*sf/1000.0))))
                            markerno += 1
                    data_ = t[0].astype(numpy.int16)
                    data_.tofile(result_file)
                    datapoint += data_.shape[0]

                result_hdr = open(os.path.join(result_path,
                                                name + key_str + ".vhdr"),"w")

                result_hdr.write("Brain Vision Data Exchange Header "
                                 "File Version 1.0\n")
                result_hdr.write("; Data stored by pySPACE\n\n")
                result_hdr.write("[Common Infos]\n")
                result_hdr.write("Codepage=UTF-8\n")
                result_hdr.write("DataFile=%s\n" %
                                      str(name + key_str + ".eeg"))
                result_hdr.write("MarkerFile=%s\n" %
                                      str(name + key_str + ".vmrk"))
                result_hdr.write("DataFormat=BINARY\n")
                result_hdr.write("DataOrientation=MULTIPLEXED\n")
                result_hdr.write("NumberOfChannels=%d\n" % len(channel_names))
                result_hdr.write("SamplingInterval=%d\n\n" % (1000000/sf))
                result_hdr.write("[Binary Infos]\n")
                result_hdr.write("BinaryFormat=INT_16\n\n")
                result_hdr.write("[Channel Infos]\n")

                # TODO: Add Resolutions to time_series
                # 0 = 0.1 [micro]V,
                # 1 = 0.5 [micro]V,
                # 2 = 10 [micro]V,
                # 3 = 152.6 [micro]V (seems to be unused!)
                resolutions_str = [unicode("0.1,%sV" % unicode(u"\u03BC")),
                   unicode("0.5,%sV" % unicode(u"\u03BC")),
                   unicode("10,%sV" % unicode(u"\u03BC")),
                   unicode("152.6,%sV" % unicode(u"\u03BC"))]
                for i in range(len(channel_names)):
                    result_hdr.write(unicode("Ch%d=%s,,%s\n" %
                        (i+1,channel_names[i],
                        unicode(resolutions_str[0]))).encode('utf-8'))
            else:
                NotImplementedError("Using unavailable storage format:%s!"
                                    % s_format)
            result_file.close()
        self.update_meta_data({
            "channel_names": copy.deepcopy(time_series[0][0].channel_names),
            "sampling_frequency": time_series[0][0].sampling_frequency
        })
        #Store meta data
        BaseDataset.store_meta_data(result_dir, self.meta_data)

    def set_window_defs(self, window_definition, nullmarker_stride_ms=1000,
                        no_overlap=False, data_consistency_check=False):
        """Code copied from StreamDataset for rewindowing data"""
        self.window_definition = window_definition
        self.nullmarker_stride_ms = nullmarker_stride_ms
        self.no_overlap = no_overlap
        self.data_consistency_check = data_consistency_check
        self.stream_mode = True


class TimeSeriesClient(AbstractStreamReader):
    """TimeSeries stream client for TimeSeries"""
    def __init__(self, ts_stream, **kwargs):

        self.callbacks = list()
        self._markerids= {"null":0}               # default marker
        self._markerNames = {0:"null"}            # dictionary with marker names
        self.nmarkertypes = len(self.markerids)  # number of different markers
        self._stdblocksize = None
        self._dSamplingInterval = None
        self.ts_stream = ts_stream

        self.blockcounter = 0

        # create two different iterators,
        # one for data reading, the other for
        # peeking etc
        (self.ts_stream_iter,self.backup_iter) = itertools.tee(iter(ts_stream))

    @property
    def dSamplingInterval(self):
        return self._dSamplingInterval

    @property
    def stdblocksize(self):
        return self._stdblocksize

    @property
    def markerids(self):
        return self._markerids

    @property
    def channelNames(self):
        return self._channelNames

    @property
    def markerNames(self):
        return self._markerNames

    def regcallback(self, func):
        """Register callback function"""
        self.callbacks.append(func)

    def connect(self):
        """connect and initialize client"""
        try:
            self._initialize(self.backup_iter.next())
        except StopIteration as e:
            print("timeseriesclient got no data: %s" % e)
            # if a StopIteration is catched right here there
            # is no data contained for the current modality (train/test)
            # in this datastream.
            pass

    def set_window_defs(self, window_definitions):
        """ Set all markers at which the windows are cut"""
        # extract start and endmarker
        marker_id_index = self.nmarkertypes
        self._markerNames[marker_id_index] = window_definitions[0].startmarker
        self._markerids[window_definitions[0].startmarker] = marker_id_index
        marker_id_index += 1      
        self._markerNames[marker_id_index] = window_definitions[0].endmarker
        self._markerids[window_definitions[0].endmarker] = marker_id_index
        marker_id_index += 1
        
        # extract all other markers
        for wdef in window_definitions:
            if not self.markerids.has_key(wdef.markername):
                self._markerNames[marker_id_index] = wdef.markername
                self._markerids[wdef.markername] = marker_id_index
                marker_id_index += 1
            # Exclude definitions marker
            for edef in wdef.excludedefs:
                if not self.markerids.has_key(edef.markername):
                    self._markerNames[marker_id_index] = edef.markername
                    self._markerids[edef.markername] = marker_id_index
                    marker_id_index += 1
            # Include definitions marker
            for idef in wdef.includedefs:
                if not self.markerids.has_key(idef.markername):
                    self._markerNames[marker_id_index] = idef.markername
                    self._markerids[idef.markername] = marker_id_index
                    marker_id_index += 1
        self.nmarkertypes = len(self.markerNames.keys())

    def _initialize(self,item):
        # get data part from (data,label) combination
        block = item[0]
        self.nChannels = block.shape[1]
        self._stdblocksize = block.shape[0]
        self._dSamplingInterval = block.sampling_frequency

        self._channelNames = block.channel_names

    def read(self, nblocks=1, verbose=False):
        """Invoke registered callbacks for each incoming data block

        returns number of read _data_ blocks"""
        ret = 0
        nread = 0

        while ret != -1 and \
                ret is not None and \
                (nblocks == -1 or nread < nblocks):
            ret = self._readmsg(verbose=False)
            if ret is not None:
                for f in self.callbacks:
                    f(self.ndsamples, self.ndmarkers)
                nread += 1

        return nread

    def _readmsg(self, msg_type='all', verbose=False):
        """ Read time series object from given iterator
        """

        # the iter items are a combination of data and
        # dummy label -> extract data
        block = self.ts_stream_iter.next()[0]

        self.blockcounter += 1

        if block is None:
            return None

        # no data message read until know
        # -> initialize property values
        if self.nChannels is None:
            self._initialize()

        self.ndmarkers = numpy.zeros([self.stdblocksize], int)
        self.ndmarkers.fill(-1)

        if block.shape[0] < self.stdblocksize:
            return 1

        for (marker, positions) in block.marker_name.iteritems():
            for position_as_ms in positions:

                # position_as_samples = numpy.floor(position_as_ms / 1000.0 *
                #                                   self.dSamplingInterval)
                position_as_samples = numpy.int(position_as_ms / 1000.0 *
                                                self.dSamplingInterval)

                # found a new marker, add it to marker name buffer
                if marker == -1 or not self.markerids.has_key(marker):
                    self.nmarkertypes += 1
                    self.markerNames[self.nmarkertypes] = marker
                    self.markerids[marker] = self.nmarkertypes
                if not self.markerids.has_key(marker):
                    continue

                markerid = self.markerids[marker]

                if self.ndmarkers[position_as_samples] == -1:
                    self.ndmarkers[position_as_samples] = markerid
                elif position_as_samples < self.stdblocksize:
                    self.ndmarkers[position_as_samples] = markerid
                else:
                    self.lostmarker = True
                    self.lostmarkertypedesc = markerid

        self.readSize = (self.nChannels * self.stdblocksize)

        self.ndsamples = numpy.array(block)
        self.ndsamples.shape = (self.stdblocksize, self.nChannels)
        self.ndsamples = scipy.transpose(self.ndsamples)

        return 1
