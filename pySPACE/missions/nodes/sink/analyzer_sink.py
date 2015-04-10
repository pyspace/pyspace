""" Collect time series to store them in an Analyzer-readable file.

.. todo:: Merge the node into the :class:`~pySPACE.missions.nodes.time_series_sink.TimeSeriesSinkNode`
          and the collection as special storage format into the
          :class:`~pySPACE.resources.dataset_defs.time_series.TimeSeriesDataset`
"""

import os
import yaml
import numpy
import warnings
import logging

# TODO: Fill in correct Date-Time for New-Segment
header_mrk = "Brain Vision Data Exchange Marker File, Version 1.0\n\
; Data created by the Analyzer Sink Node\n\
[Common Infos]\n\
Codepage=UTF-8\n\
DataFile=%s.eeg\n\
\n\
[Marker Infos]\n\
; Each entry: Mk<Marker number>=<Type>,<Description>,<Position in data points>,\n\
; <Size in data points>, <Channel number (0 = marker is related to all channels)>\n\
; Fields are delimited by commas, some fields might be omitted (empty).\n\
; Commas in type or description text are coded as \"\\1\".\n\
Mk1=New Segment,,1,1,0,20090901101048982538\n"

header_hdr = "Brain Vision Data Exchange Header File Version 1.0\n\
; Data created by the Analyzer Sink Node\n\
\n\
[Common Infos]\n\
Codepage=UTF-8\n\
DataFile=%s.eeg\n\
MarkerFile=%s.vmrk\n\
DataFormat=BINARY\n\
; Data orientation: MULTIPLEXED=ch1,pt1, ch2,pt1 ...\n\
DataOrientation=MULTIPLEXED\n\
NumberOfChannels=%d\n\
; Sampling interval in microseconds\n\
SamplingInterval=%d\n\
\n\
[Binary Infos]\n\
BinaryFormat=INT_16\n\
\n\
[Channel Infos]\n\
; Each entry: Ch<Channel number>=<Name>,<Reference channel name>,\n\
; <Resolution in \"Unit\">,<Unit>, Future extensions..\n\
; Fields are delimited by commas, some fields might be omitted (empty).\n\
; Commas in channel names are coded as \"\1\".\n"

from pySPACE.resources.dataset_defs.base import BaseDataset
from pySPACE.missions.nodes.base_node import BaseNode
from pySPACE.tools.filesystem import get_author
from pySPACE.resources.data_types.time_series import TimeSeries


class AnalyzerCollection(BaseDataset):
    """ Derived class from BaseDataset to store data in an Analyzer readable format

    This class derived from BaseDataset overwrites the 'store' method from
    the BaseDataset class so that the stored files can be read with
    the BrainVision Analyzer. Remaining spaces in front and between the single
    windows are filled with zeros.

    The input can be a TimeSeries collection or a single time series resulted from 
    merging the datasets. 
    
    **Parameters**

        :dataset_md:
            The meta data of the current collection of TimeSeries item.

            (*optional, default: None*)

    .. todo:: Move to storage_format in TimeSeriesDataset

    :Author: Johannes Teiwes (Johannes.Teiwes@dfki.de)
    :Created: 2010/05/09
    :Modified: Foad Ghaderi 2014/04/15
    """
    def __init__(self, dataset_md =  None):
          super(AnalyzerCollection, self).__init__(dataset_md = dataset_md)
    
    def store(self, result_dir, s_format = "BrainVision"):
        self.merged = False
        scale = 10.0 # is used to scale up the eeg sample values.  The data samples are converted to int16
                    # when saving, so scaling is necessary to keep maintain the resolutions. 
        # Keep original file name, depends on the AnalyserSinkNode, see it's documentation.
        if self.meta_data.has_key('eeg_src_file_name') and self.meta_data['eeg_src_file_name'] is not None:
            name = self.meta_data['eeg_src_file_name']
        # or use default name from this collection
        else:
            name = "Analyzer"
        if not s_format == "BrainVision":
            self._log("The format %s is not supported!"%s_format, level=logging.CRITICAL)
            return
        # Update the meta data
        author = get_author()
        self.update_meta_data({"type": "only output of individual nodes stored",
                                      "storage_format": s_format,
                                      "author" : author,
                                      "data_pattern": "Multiplexed"})
        # Store meta data
        BaseDataset.store_meta_data(result_dir,self.meta_data)
        #self._log("EEG data file %s" % self.collection.data_file)
        slices = []
        slices.append(0)
        channel_names = []
        

        
        for key, time_series in self.data.iteritems():
            # Sort the Times-Series Array
            def cmp_start(a, b):
                return cmp(a[0].start_time, b[0].start_time)

            time_series.sort(cmp_start)
            # Check for overlapping Windows and remove them if existent
            i = 0
            while i < len(time_series):
                ts = time_series[i]
                #print ts[0].start_time, ts[0].end_time
                #print len(time_series)
                if ts[0].start_time >= slices[-1]:
                    slices.append(ts[0].end_time)
                else:
                    warnings.warn("Ignoring at least one overlapping window!", UserWarning)
                i = i+1
            # STORE ACTUAL EEG DATA AND WRITE MARKERFILE
            result_path = result_dir + os.sep + "data_analyzer" \
                            + "_run%s" % key[0]
            if not os.path.exists(result_path):
                os.mkdir(result_path)
            
            key_str = "_sp%s_%s" % key[1:]
            # Keep original name
            if (self.meta_data.has_key('eeg_src_file_name') and self.meta_data['eeg_src_file_name'] != None):
                result_file_eeg = open(os.path.join(result_path, name + ".eeg"), "wb")
                result_file_mrk = open(os.path.join(result_path, name + ".vmrk"), "w")
            # or use default name from this collection
            else:
                result_file_eeg = open(os.path.join(result_path, name + key_str + ".eeg"), "wb")
                result_file_mrk = open(os.path.join(result_path, name + key_str + ".vmrk"), "w")
        
            # Write Marker header
            if (self.meta_data.has_key('eeg_src_file_name') and self.meta_data['eeg_src_file_name'] != None):
                result_file_mrk.write(header_mrk % (name))
            else:
                result_file_mrk.write(header_mrk % (name + key_str))
        
            result_file_ms = 0
        
            # Data for padding
            padding = None
        
            count_mrk = 2
            num_ch = 0
            sampling_int = 0
            
            for ts in time_series:
                ts0 = ts[0] * scale 
                ts0 = ts0.astype(numpy.int16)
                
                if padding == None:
                    padding = numpy.zeros(len(ts[0].channel_names), dtype='int16')
                    num_ch = len(ts[0].channel_names)
                    channel_names = ts[0].channel_names
                    sampling_int = 1000000/ts[0].sampling_frequency
                    #print "writing %d channels.." % len(ts[0].channel_names)
                # Write Padding (zeros)
                while result_file_ms < ts[0].start_time - sampling_int/1000.0:
                    result_file_eeg.write(padding.tostring())
                    result_file_ms += ts[0]._samples_to_ms(1)
                # Write window
                ts0.tofile(result_file_eeg)
                result_file_ms += ts[0].end_time - (ts[0].start_time - sampling_int/1000.0)
                # Write Marker
                markers = []
                
                if(len(ts[0].marker_name) > 0):
                    mk_keys = ts[0].marker_name.keys()
                    mk_values = ts[0].marker_name.values()
                    for mk in range(len(mk_keys)):
                        for mv in range(len(mk_values[mk])):
                            markers.append((mk_keys[mk], mk_values[mk][mv]))
                    markers = sorted(markers, key=lambda tup: tup[1])
                    
                    for i in range(len(markers)):
                        if 'R' in markers[i][0]: 
                            event_type = 'Response' 
                        elif 'S' in markers[i][0]:
                            event_type = 'Stimulus'
                        else:
                            event_type = 'Label'
                            
                        result_file_mrk.write("Mk%d=%s,%s,%d,1,0\n" % (count_mrk, event_type, markers[i][0], (ts[0].start_time + markers[i][1])*ts[0].sampling_frequency/1000.0))                                                     
                        count_mrk += 1

            # WRITE HEADERFILE
            # Keep original name
            if (self.meta_data.has_key('eeg_src_file_name') and self.meta_data['eeg_src_file_name'] != None):
                result_file_hdr = open(os.path.join(result_path, name + ".vhdr"), "w")
                result_file_hdr.write(header_hdr % ((name), (name), num_ch, sampling_int))
            # or use default name from this collection
            else:
                result_file_hdr = open(os.path.join(result_path, name + key_str + ".vhdr"), "w")
                result_file_hdr.write(header_hdr % ((name + key_str), (name + key_str), num_ch, sampling_int))
            # Format: Ch1=Fp1,,0.1,\xB5V
            for i in range(num_ch):
                result_file_hdr.write("Ch%d=%s,,%.2f,\xB5V\n" % (i+1,channel_names[i], 1./scale))

            result_file_hdr.close()
            result_file_eeg.close()
            result_file_mrk.close()


class AnalyzerSinkNode(BaseNode):
    """ Store all TimeSeries that are passed to it in an collection of type AnalyzerCollection

    This node enables the software to store the passed data
    in an BrainVision Analyzer readable format.

    **Parameters**
    
    :original_name:
        If the node parameter *original_name* is set to 'True', the result files 
        are saved with the name of the original file.
    
        (*optional, default: False*)

    **Exemplary Call**

    .. code-block:: yaml

        - 
            node : Analyzer_Sink
            parameters :
                original_name : True

    .. todo:: Move to TimeSeriesSink/ implement storage format

    :Author: Johannes Teiwes (Johannes.Teiwes@dfki.de)
    :Created: 2010/05/09
    """
    input_types = ["TimeSeries"]

    def __init__(self, original_name = False, **kwargs):
        super(AnalyzerSinkNode, self).__init__(**kwargs)
        self.set_permanent_attributes(analyzer_collection = \
                                                AnalyzerCollection(),
                                                original_name = original_name)
                                                
    def is_trainable(self):
        """ Returns whether this node is trainable """
        return True

    def is_supervised(self):
        """ Returns whether this node requires supervised training """
        return True
        
    def process_current_split(self):
        """ 
        Compute the results of this sink node for the current split of the data
        into train and test data
        """
        
        # Adds the name of the eeg source file to the meta information
        # of the Analyser_Collection. This only works for the 
        # Stream2TimeSeriesSourceNode
        if self.original_name:
            if not self.analyzer_collection.meta_data.has_key('eeg_src_file_name'):
                self.analyzer_collection.meta_data['eeg_src_file_name'] = self.get_source_file_name()
        else:
            if not self.analyzer_collection.meta_data.has_key('eeg_src_file_name'):
                self.analyzer_collection.meta_data['eeg_src_file_name'] = None
        
        # Count Splits for meta data. Usually this is done by
        # BaseDataset.add_sample. But here, obviously, no samples are added.
        # Compute the time series for the data used for training
        for time_series, label in self.input_node.request_data_for_training(False):
            
            # Add sample
            self.analyzer_collection.add_sample(time_series,
                                                   label = label,
                                                   train = True,
                                                   split = self.current_split,
                                                   run = self.run_number)
            
        # Compute the time series for the data used for testing
        for time_series, label in self.input_node.request_data_for_testing():
            
            # Add sample
            self.analyzer_collection.add_sample(time_series,
                                                   label = label,
                                                   train = False,
                                                   split = self.current_split,
                                                   run = self.run_number)
                                                   
        # Check if we have an Stream2TimeSeriesSourceNode as Source and obtain
        # the absolute path from it for later use inside the AnalyzerCollection
        #if(not "data_path" in self.analyzer_collection.meta_data):
        #    src_node = self
        #    while not src_node.is_source_node():
        #        src_node = src_node.input_node
        #    if(isinstance(src_node, Stream2TimeSeriesSourceNode)):
        #        self.analyzer_collection.meta_data["data_path"] = src_node.eeg_server.data_path
        #    else:
        #        # When using other sources than Stream2TimeSeriesSourceNode no path is used
        #        self.analyzer_collection.meta_data["data_path"] = "default"
                                                   
    def _execute(self, data, n = None):
        print "***:", str(data)
        return data
        
    def _train(self, data, label):
        return (data, label)
    
    
        
    def get_result_dataset(self):
        """ Return the result of this last node """
        return self.analyzer_collection
        
# Specify special node names
_NODE_MAPPING = {"Analyzer_Sink": AnalyzerSinkNode}
