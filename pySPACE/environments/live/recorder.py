import time
import os
import struct
import sys
import numpy
import scipy

file_path = os.path.dirname(os.path.abspath(__file__))
pyspace_path = file_path[:file_path.rfind('pySPACE')-1]
if not pyspace_path in sys.path:
    sys.path.append(pyspace_path)

import pySPACE
from pySPACE.missions.support.WindowerInterface import AbstractStreamReader

header_mrk = "Brain Vision Data Exchange Marker File, Version 1.0\n\
; Data written by the recorder script.\n\
[Common Infos]\n\
Codepage=UTF-8\n\
DataFile=%s\n\
\n\
[Marker Infos]\n\
; Each entry: Mk<Marker number>=<Type>,<Description>,<Position in data points>,\n\
; <Size in data points>, <Channel number (0 = marker is related to all channels)>\n\
; Fields are delimited by commas, some fields might be omitted (empty).\n\
; Commas in type or description text are coded as \"\\1\".\n\
Mk1=New Segment,,1,1,0,%s\n"

header_hdr = "Brain Vision Data Exchange Header File Version 1.0\n\
; Data written by the recorder script.\n\
\n\
[Common Infos]\n\
Codepage=UTF-8\n\
DataFile=%s\n\
MarkerFile=%s\n\
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

class Recorder(object):

    def __init__(self, client=None, folder=None, subject=None,
                 task="pySPACE", online=False, **kwargs):
        super(Recorder, self).__init__(**kwargs)

        if folder is None:
            folder = pySPACE.configuration.storage

        self.folder = folder
        self.subject = str(subject)
        self.task = task
        self.date = time.strftime("%Y%m%d") # append %H%M%S for time
        self.set_no = self.find_next_set()

        if not os.path.exists(self.folder):
            os.makedirs(self.folder)
            assert os.path.exists(self.folder), "Could not create Directory!"

        self.eeg_filename  = str("%s_r_%s_%s_%sSet%d.eeg" %
                             (self.date, self.subject, self.task,
                              "_online_" if online else "", self.set_no))
        self.vhdr_filename = str("%s_r_%s_%s_%sSet%d.vhdr" %
                             (self.date, self.subject, self.task,
                              "_online_" if online else "", self.set_no))
        self.vmrk_filename = str("%s_r_%s_%s_%sSet%d.vmrk" %
                             (self.date, self.subject, self.task,
                              "_online_" if online else "", self.set_no))

        self.eeg = open(os.path.join(self.folder, self.eeg_filename), "w")
        self.vhdr = open(os.path.join(self.folder, self.vhdr_filename), "w")
        self.vmrk = open(os.path.join(self.folder, self.vmrk_filename), "w")

        self.client = client

        if client is not None:
            self.set_eeg_client(client)

        self.datapoint = 0
        self.markerno = 2


    def set_eeg_client(self, client):
        if not isinstance(client, AbstractStreamReader):
            raise IOError, "No AbstractStreamReader compatible data-source!"
        self.client = client
        self.write_header()
        self.client.regcallback(self.write_data)

    def has_client(self):
        return (self.client is not None)

    def find_next_set(self):
        set_no = 1
        while True:
            filename  = str("%s_r_%s_%s_Set%d.eeg" %
                     (self.date, self.subject, self.task, set_no))
            abs_filename = os.path.join(self.folder, filename)
            if not os.path.isfile(abs_filename):
                break
            set_no += 1
        return set_no

    def write_header(self):
        self.vhdr.write(header_hdr % (self.eeg_filename, self.vmrk_filename,
                                  self.client.nChannels, 1000000/self.client.dSamplingInterval))
        for i in range(self.client.nChannels):
            self.vhdr.write(str("Ch%d=%s,,100,nV\n" % (i+1,self.client.channelNames[i])))
        self.vhdr.flush()
        self.vmrk.write(header_mrk % (self.eeg_filename, time.strftime("%Y%m%d%H%M%S")))
        self.vmrk.flush()

    def write_data(self, data, marker):

        if isinstance(data, numpy.ndarray):
            data_ = data.astype(numpy.int16)
            data_ = scipy.transpose(data_)
            buf = struct.pack("%dh"%len(data.flatten()), *data_.flatten())
        else:
            buf = struct.pack('%dh'%len(data), *data)

        self.eeg.write(buf)
        self.eeg.flush()

        for i,m in enumerate(marker):
            if m != -1:
                self.vmrk.write(str("Mk%d=Stimulus,%s,%d,1,0\n" %
                        (self.markerno,self.client.markerNames[m],self.datapoint+i)))
                self.markerno += 1
        self.datapoint += len(marker)
        self.vmrk.flush()

