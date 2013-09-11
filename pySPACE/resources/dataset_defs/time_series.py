""" Load and store data of the type :mod:`pySPACE.resources.data_types.time_series` """
import itertools

import copy
import os
import cPickle
import sys
import scipy
import yaml
import pwd
import csv
import numpy
import logging
import warnings
import glob
from pySPACE.missions.support.WindowerInterface import AbstractStreamReader
from pySPACE.missions.support.windower import MarkerWindower

from pySPACE.resources.dataset_defs.base import BaseDataset


class TimeSeriesDataset(BaseDataset):
    """ Time series dataset
    
    This class encapsulate most relevant code for dealing with 
    time series datasets, most importantly
    for loading and storing them to the file system.
    
    These datasets consist of
    :mod:`~pySPACE.resources.data_types.time_series` objects.
    They can be loaded with a
    :mod:`~pySPACE.missions.nodes.source.time_series_source` and saved with a
    :mod:`~pySPACE.missions.nodes.sink.time_series_sink` node
    in a :class:`~pySPACE.missions.operations.node_chain.NodeChainOperation`.
    
    The standard format is 'pickle'.
    
    **Parameters**
    
      :dataset_md:
          A dictionary with all the meta data.
          
          (*optional, default: None*)
          
      :sort_string: 
          A lambda function string that is evaluated before the data is stored.

          (*optional, default: None*)
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
            else: # s_format=="csv":
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
            if type(s_format)==list:
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
        try:
            author = pwd.getpwuid(os.getuid())[4]
        except Exception:
            author = "unknown"
            self._log("Author could not be resolved.", level=logging.WARNING)
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
        except StopIteration:
            # if a StopIteration is catched right here there
            # is no data contained for the current modality (train/test)
            # in this datastream.
            pass

    def set_window_defs(self, window_definitions):
        index = self.nmarkertypes
        # Marker at which the windows are cut
        for wdef in window_definitions:
            if not self.markerids.has_key(wdef.markername):
                self.markerNames[index] = wdef.markername
                self.markerids[wdef.markername] = index
                index += 1
            # Exclude definitions marker
            for edef in wdef.excludedefs:
                if not self.markerids.has_key(edef.markername):
                    self.markerNames[index] = edef.markername
                    self.markerids[edef.markername] = index
                    index += 1
            # Include definitions marker
            for idef in wdef.includedefs:
                if not self.markerids.has_key(idef.markername):
                    self.markerNames[index] = idef.markername
                    self.markerids[idef.markername] = index
                    index += 1
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
