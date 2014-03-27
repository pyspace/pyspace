# -*- coding: UTF-8 -*

"""eeg_stream.py
EEG client module.

Created by Timo Duchrow on 2008-08-26.
"""

import sys
import socket
import struct
import numpy
import scipy
import Queue
import signal
import subprocess
import os
import time
import xmlrpclib
import random
import glob
import warnings

file_path = os.path.dirname(os.path.abspath(__file__))
pyspace_path = file_path[:file_path.rfind('pySPACE')-1]
if not pyspace_path in sys.path:
    sys.path.append(pyspace_path)

import pySPACE
from ipmarkers import MarkerServer

usb_warning = True
try:
    if hasattr(pySPACE.configuration, "eeg_modules_dir"):
        sys.path.append(pySPACE.configuration.eeg_modules_dir)
        from eeg_acquisition.pybrainamp import BUASubprocess
        usb_warning = False
except:
    pass

from pySPACE.missions.support.WindowerInterface import AbstractStreamReader

verbose = False

READYMSG = '1'

fmt__GUID = 'IHH8s'
fmt_RDA_MessageHeader = 'II'

n__GUID = struct.calcsize(fmt__GUID)
n_RDA_MessageHeader = struct.calcsize(fmt_RDA_MessageHeader)

T_RDA_MessageStart = 1
T_RDA_MessageData = 2
T_RDA_MessageStop = 3
T_RDA_MessageMuxData = 4
T_RDA_MessageData32 = 9
T_ReadyMessage = 10


class EEGClient(AbstractStreamReader):
    """EEG stream client for EEG stream protocol"""
    def __init__(self, host='127.0.0.1', port=51244, **kwargs):
        super(EEGClient, self).__init__()

        # containers for abstract properties
        self._dSamplingInterval = None   # sampling interval
        self._channelNames = None        # list of channel names
        self._stdblocksize = None        # standard number of paints in one data block
        self._markerids= dict()
        self._markerNames = dict()       # dictionary with marker names

        self.host = host
        self.port = port
        self.nChannels = None           # number of channels
        self.sample_size = None
        self.protocol_version = None
        self.resolutions = None         # list of resolutions / channel
        self.channelids = dict()
        self.abs_start_time = 0
        self.callbacks = list()
        self.meta = dict()
        self.ndsamples = None           # last sample block read
        self.ndmarkers = None         # last marker block read
        self.nmarkertypes = 0           # number of different marker types
        self.lostmarker = False         # for two markers corresponding to one sample in last sample of a block
        self.lostmarkertypedesc = None
        self.running = True

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
        
        
    def connect(self, verbose=False):
        """Connect to EEG stream server and collect metadata"""
        try: 
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        except socket.error, (value,message): 
            if self.socket: 
                self.socket.close() 
            raise IOError, "(%d): could not open socket(%d): %s" % (value, self.port, message) 
            
         # Configure and connect the Socket
        try:
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket.connect((self.host, self.port))
        except socket.error, (value,message): 
            if self.socket: 
                self.socket.close() 
            raise IOError, "(%d): SO_REUSEADDR and/or connect failed! (%d): %s" % (value, self.port, message)

#        print "connected to %s @ %s" %(self.host, self.port)
             
        if(os.name=='posix'):
            # install signal handlers
            sigs = (signal.SIGHUP, signal.SIGINT, signal.SIGTERM, signal.SIGQUIT)
            for s in sigs:
                signal.signal(s, self._grace)
            
        # read start message
        ret = self._readmsg(msg_type=T_RDA_MessageStart, verbose=verbose)
        if ret == -1:
            self.socket.close()
            raise IOError, "could not read start message"
        
        if verbose:
            print "connected to server"

    def regcallback(self, func):
        """Register callback function"""
        self.callbacks.append(func)

    def read(self, nblocks=1, verbose=False):
        """Invoke registered callbacks for each incoming data block
        
        returns number of read _data_ blocks"""
        ret = 0
        nread = 0
        
        # return no data when not running
        if not self.running:
            return nread
        
        while ret != -1 and \
            ret != T_RDA_MessageStop and \
            (nblocks == -1 or nread < nblocks) and self.running:

            ret = self._readmsg(verbose=verbose)       
            if ret == T_RDA_MessageData or ret == T_RDA_MessageData32 or ret == T_RDA_MessageMuxData:
                if verbose: 
                    print "received EEG stream data message"
                for f in self.callbacks:
                    f(self.ndsamples, self.ndmarkers)    
                nread += 1
            elif ret == T_RDA_MessageStop:
                if verbose: 
                    print "received EEG stream stop message"
            elif ret == T_RDA_MessageStart:
                if verbose:
                    print "Received updated EEG stream start Message"
                #raise IOError, "received unexpected EEG stream start message"
            elif ret == -1:
                raise IOError, "could not read EEG stream message"
            else:
                raise IOError, "unknown EEG stream message type %d" % ret
    
        return nread
    
    def _sendmsg(self, msg_type=T_ReadyMessage, content=None):
        """Sends Messages to EEGServer
        
        T_ReadyMessage := tell Server to send another EEG{Data, Start, Stop}Message"""
        
        if msg_type == T_ReadyMessage:
                self.socket.send(READYMSG)
        else:
            raise IOError, "Type of Message to send to EEGserver not known!"
    
    def _readmsg(self, msg_type='all', verbose=False):
        """Reads EEG stream message and invokes appropiate handler.
        
        Reads EEG stream header and checks optional type constraint.
        Returns which message type was read."""
        # tell server to send a packet
        try:
            self._sendmsg()
        except:    
            return T_RDA_MessageStop
            
        # read  GUID    
        #guid = self.socket.recv(n__GUID)
        # read header
        recvsize = n_RDA_MessageHeader
        buff = ''
        while len(buff)<recvsize:
            hdr = self.socket.recv(recvsize-len(buff))
            buff += hdr

        hdr = buff
        if len(hdr) == 0:
            # print ">>> no header"
            return -1
        elif len(hdr) != n_RDA_MessageHeader:
            raise IOError, "data stream corrupt"
        (nSize, nType) = struct.unpack(fmt_RDA_MessageHeader, hdr)
        if verbose:
            print "header:\n nSize: %08x\n nType: %08x" % (nSize, nType)
        # check optional type constraint
        if msg_type != 'all' and msg_type != nType:
            raise IOError, "RDA message of type %d expected "\
            "(received type %d from server)" % (msg_type, nType)
            
        # read message from socket
        # The while loop is neccessary to ensure that the total amount of data is read
        self.recvsize = nSize - n_RDA_MessageHeader
        if verbose:
            print "receiving message type %d of size %d" % (nType, nSize)

        self.buff1 = ''
        while len(self.buff1) < self.recvsize:
            payload = self.socket.recv(self.recvsize-len(self.buff1))
            self.buff1 += payload

        payload = self.buff1
        
        if verbose:
            print("done! total size (except header): %d" % len(payload))
        
        # invoke appropiate handler for decoding
        if nType == T_RDA_MessageStart:
            self._getstartmsg(payload, verbose=verbose)
        elif nType == T_RDA_MessageData:
            self._getdatamsg(payload)
        elif nType == T_RDA_MessageMuxData:
            self._getmuxdatamsg(payload, verbose=verbose)
        elif nType == T_RDA_MessageStop:
            self._getstopmsg(payload)
        elif nType == T_RDA_MessageData32:
            self._getdata32msg(payload)
        return nType
    
    def _getstopmsg(self, payload, verbose=False):
        self.socket.close()
                
    def _getstartmsg(self, payload, verbose=False):
        """Decode metadata from start type message"""

        offset = 0
        
        # read number of channels and sampling interval
        fmt = 'IIIII'
        nread = struct.calcsize(fmt)
        (self.nChannels, self._stdblocksize, self._dSamplingInterval, self.sample_size, self.protocol_version) = \
            struct.unpack_from(fmt, payload, offset)
        offset += nread
        # offset += 4 
        # TODO: find out why! (fix for compatability with eegmanager 18ee0306a783fe1518742b1ea13b811e837662fb)
        # TODO: solved?
            
        if self.protocol_version == 2:
            self.nChannels = self.nChannels - 1
            
            fmt = 'II'
            nread = struct.calcsize(fmt)
            part1, part2 = struct.unpack_from(fmt, payload, offset)
            offset += struct.calcsize('II')
            
            self.abs_start_time = part1
            self.abs_start_time = self.abs_start_time << 32
            self.abs_start_time += part2
            
            if verbose:
                print "abs_start_time: %d \n" % (self.abs_start_time)
                print nread        

        if verbose: 
            print "\n\n\nmessage start:\n  nChannels: %d\n  dSamplingInterval: %d\n Blocksize: %d\nSampleSize: %d\nProtocol-Version: %d\n" \
                % (self.nChannels, self.dSamplingInterval, self.stdblocksize, self.sample_size, self.protocol_version)
        

        # read resolutions
        fmt = '%dB' % (256)
        nread = struct.calcsize(fmt)
        self.resolutions = struct.unpack_from(fmt, payload, offset)
        if verbose: print "  resolutions: ", self.resolutions
        offset += nread

        
        fmt = 'I'
        nread = struct.calcsize(fmt)
        (self.lenChannelNames) = \
            struct.unpack_from(fmt, payload, offset)
        if verbose: 
            print "  lenChannelNames: %d\n" \
                % (self.lenChannelNames)
        offset += nread        

        
        
        # read channel names
        fmt = '%ds' % (self.lenChannelNames)
        namesbuf = struct.unpack_from(fmt, payload, offset)
        nread = struct.calcsize(fmt)
        # build list of channel names
        self._channelNames = list()
        nstr = ''
        for c in namesbuf[0]:
            if c == '\x00':
                self.channelNames.append(nstr)
                nstr = ''
            else:
                nstr += c
        
        if self.protocol_version == 2:
            assert 'marker' == self.channelNames.pop()
            assert len(self.channelNames) == self.nChannels
            if verbose: print "removed \'marker\' pseudo channel from channel-names.\n"
                  
        for (cid, name) in enumerate(self.channelNames):
            self.channelids[name] = cid

        if verbose: print "  channel names: ", self.channelNames, "\n"

        offset += nread

        fmt = 'I'
        nread = struct.calcsize(fmt)
        (self.lenMarkerNames) = \
            struct.unpack_from(fmt, payload, offset)
        if verbose: 
            print "  lenMarkerNames: %d\n" \
                % (self.lenMarkerNames)
        offset += nread
        
        
        # read marker names
        fmt = '%ds' % (self.lenMarkerNames)
        namesbuf = struct.unpack_from(fmt, payload, offset)

        # build list of marker names
        self._markerNames = dict()
        self._markerids = dict()
        nstr = ''
        for c in namesbuf[0]:
            if c == '\x00':
                self.markerNames[len(self.markerNames)] = nstr
                nstr = ''
            else:
                nstr += c
        self.markerNames[len(self.markerNames)] = 'null'
        self.nmarkertypes = len(self.markerNames)

        for (cid, name) in self.markerNames.iteritems():
            self.markerids[name] = cid
        if verbose: 
            print "  marker names: ", self.markerNames, "\n"
            print "  marker ids: ", self.markerids, "\n"
        
    def _getdata32msg(self, payload, verbose=False):
        """Convenience method for getting data from 32-bit type data message (float)"""
        # TODO test with RDA server
        self._getdatamsg(payload,
            dtype='float32', msgtype='f')
            
    def _getdatamsg(self, payload, verbose=False, dtype='short', msgtype='h'):
        """Get data from 16-bit type data message (short)"""
        offset = 0
        fmt = 'II'
        nread = struct.calcsize(fmt)
        (time_code, nMarkers) = \
            struct.unpack_from(fmt, payload, offset)
        offset += nread

        self.meta['time_code'] = time_code
        self.meta['n_markers'] = nMarkers

        # read markers
        #self.ndmarkers = None
        self.ndmarkers = numpy.zeros([self.stdblocksize], int)
        self.ndmarkers.fill(-1)
        if self.lostmarker:
            self.lostmarker = False
            self.ndmarkers[0] = self.lostmarkertypedesc
        #print "lenmarkers: "+str(nMarkers)
        for i in range(nMarkers):
            fmt = 'II'
            nread = struct.calcsize(fmt)     
            (nPosition, ntypedesc) = \
                struct.unpack_from(fmt, payload, offset)
            offset += nread
                       
            if ntypedesc == -1 or ntypedesc >= self.nmarkertypes:
                raise Exception, "received unexpexted marker type (%d) not " \
                    "defined in header max(%d)" % (ntypedesc, self.nmarkertypes)
            
            if verbose:
                print "  relPosition: nPosition: %d\n  sTypeDesc: %d\n" % (nPosition, ntypedesc)
                print "%s" % self.markerNames[ntypedesc]

            #self.ndmarkers[len(self.ndmarkers)] = ntypedesc #(ntypedesc, nPosition)
            if self.ndmarkers[nPosition-1] == -1:
                self.ndmarkers[nPosition-1] = ntypedesc
            elif (nPosition - 1) < (self.stdblocksize - 1):
                self.ndmarkers[nPosition] = ntypedesc
            else:
                self.lostmarker = True
                self.lostmarkertypedesc = ntypedesc
                
#            print self.ndmarkers            
            # build dictionary of marker names on the fly
            # this information should really come from the header in the next
            # version of RDA server
            # ntypedesc = None
            # if not self.markerids.has_key(sTypeDesc):
            #     self.nmarkertypes += 1
            #     self.markerids[sTypeDesc] = self.nmarkertypes
            #     ntypedesc = self.nmarkertypes
            # else:
            #     ntypedesc = self.markerids[sTypeDesc]
            #     
            # self.markernames[ntypedesc] = sTypeDesc
            
            # check if this is ok
            #self.ndmarkers[nPosition-1] = ntypedesc
            # print "nPosition ", nPosition
        #                                                * struct.calcsize(msgtype)

        self.readSize = (self.nChannels * self.stdblocksize * struct.calcsize(msgtype))
        
        dt = numpy.dtype(numpy.int16)
        self.ndsamples = numpy.frombuffer(payload[offset:offset + self.readSize], dtype=dt)  
        self.ndsamples.shape = (self.stdblocksize, self.nChannels)
        self.ndsamples = scipy.transpose(self.ndsamples)
        
        
    def _getmuxdatamsg(self, payload, verbose=False):
        """Get data from 16/32-bit type data message (short)"""

        offset = 0
        fmt = 'II'
        nread = struct.calcsize(fmt)
        time_code, sample_size = struct.unpack_from(fmt, payload, offset)
        offset += nread

        self.meta['time_code'] = time_code
        self.meta['sample_size'] = sample_size
        
        sample_size = int(sample_size)
        if sample_size == 2:
            dtype = numpy.int16
            msgtype = 'h'
        elif sample_size == 4:
            dtype = numpy.int32
            msgtype = 'i'
        else:
            print "recovered sample_size (%d) is unknown or an error!" % sample_size
            raise IOError

        if verbose: print "unpacking message (dtype:=%s, msgtype=%s)" % (dtype, msgtype)

        # read markers
        # TODO: extract markers

        self.readSize = ((self.nChannels+1) * self.stdblocksize * struct.calcsize(msgtype))
        dt = numpy.dtype(dtype)
        
        raw_data = numpy.frombuffer(payload[offset:offset + self.readSize], dtype=dt)
        raw_data.shape = (self.stdblocksize, self.nChannels+1)
        
        # self.ndsamples = numpy.hsplit(raw_data, numpy.array([raw_data.shape[1]-1, raw_data.shape[1]]))[0]  
        self.ndsamples = raw_data[:,:self.nChannels]
        self.ndsamples.shape = (self.stdblocksize, self.nChannels)
        self.ndsamples = scipy.transpose(self.ndsamples)
        
        comp_markers = list()
        
        # raw_markers = numpy.hsplit(raw_data, numpy.array([raw_data.shape[1]-1, raw_data.shape[1]], numpy.int16))[1]
        raw_markers = raw_data[:,self.nChannels]
        for m in raw_markers:
            if m != 0:
                smarker = (m & 0xff)
                rmarker = (m & 0xff00) >> 8
                if rmarker == 0:
                    comp_markers.append(self.markerids[str("S%3d" % smarker)])
                else:
                    comp_markers.append(self.markerids[str("R%3d" % rmarker)])
                
            else:
                comp_markers.append(-1)
        
        self.ndmarkers = numpy.array(comp_markers)
        
#        print "samples shape: ", self.ndsamples.shape
#        print "markers shape: ", self.ndmarkers.shape
                            
    def _grace(self, signum, stackframe):
        self.socket.close()
        sys.exit(1)

class EEGServer(object):
    """EEG stream server for EEG stream protocol"""
    def __init__(self, absolute_data_path, block_size=4, port=51244):
        self.port = port
        self.data_path = absolute_data_path

        self.block_size = block_size
        self.server_process = None
        self.server_proxy = None
        self.executable = os.path.join("eegmanager", "release", "eegmanager")

    def __del__(self):
        if self.server_process != None:
            self.server_proxy.shut_down()
        else:
            warnings.warn('EEG-Server could not be started. Did you compile it correctly?')

    def start(self):
        """ Starts the EEG server"""


        if None == self.server_process:
            server_out_log = open("server_out_log", 'w')
            server_err_log = open("server_err_log", 'w')

            eeg_acq_root = os.sep.join(sys.modules[__name__].__file__.split(os.sep)[:-1])

            # Workaround for finding Free XML-Ports
            freeport = False
            call = list()
            call.append(self.executable)
            try:
                files = glob.glob(eeg_acq_root + self.executable)
            except:
                raise Exception("There is no existing executable in %s. Please compile!" % (os.path.join(eeg_acq_root, self.executable)))

            while not freeport:
                # Add Port number, create test-ProxyServer,
                # XMLRPC-Port range: 16253..26253
                xmlport = 16253 + random.randint(0,10000)
                call.append(str(xmlport))
                test_proxy = xmlrpclib.ServerProxy("http://127.0.0.1:%s" % call[1])

                try:
                    test_proxy.system.listMethods()
                except socket.error, (value, message):
                    freeport = True

                del test_proxy

                # Remove Portnumber
                if not freeport:
                    call.pop()

            self.server_process = subprocess.Popen(call,
                                       cwd=eeg_acq_root,
                                       stdin=None,
                                       stderr=server_err_log,
                                       stdout=server_out_log)

            self.server_proxy = xmlrpclib.ServerProxy("http://127.0.0.1:%s" % call[1])


        mysocket = None

        # wait for the server to startup...
        # TODO: check this! Time depends on number of started processes!
        time.sleep(3)

        try:
            #try to get a socket connection and start the server
            self.server_proxy.start(self.data_path, self.port, self.block_size)

            time.sleep(5)

            # check if the server process is still running?
            # returncode:   None -> still running
            #               Numeric -> terminated
            self.server_process.poll()
            if(self.server_process.returncode != None):
                ret = self.server_process.returncode
                self.server_process = None
                raise SystemError, "Server Process should run but exited with status %d" % (ret)

            mysocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

            connected = mysocket.connect(("127.0.0.1", self.port))

        except socket.error, (value,message):
            # failure in connection
            if mysocket:
                mysocket.close()

            # could not open connection -> kill server
            os.kill(self.server_process.pid, signal.SIGKILL)

            raise IOError, "error: %d could not open socket(%d): %s" % (value, self.port, message)

        time.sleep(1)

        #Create a method which handles the tear down of the server
        def kill():
            """ Kills the running EEG server """
            self.server_proxy.stop()
            time.sleep(.3)
            self.server_proxy.shut_down()

        #Add signal handler
        signals = (signal.SIGHUP, signal.SIGINT, signal.SIGTERM, signal.SIGQUIT)
        for sig in signals:
            signal.signal(sig, kill)

    def reset(self):
        """ Reset the EEG server so that he points again to the beginning"""
        self.server_proxy.stop()
        time.sleep(1)
        self.server_proxy.shut_down()
        time.sleep(1)

        self.start()

class EEGClientUsb(AbstractStreamReader):
    """ Acquire raw streamed eeg-data from usb-acquisition submodule

        This implementation uses the usb1 module, available from
        Pip/easy_install. usb1 is used to wrap calls to the libusbx c-library.
    """
    
    def __init__(self, ipmarker_server=None, **kwargs):
        """ currently no parameter
        """
        if usb_warning:
            warnings.warn("The usb_acquisition Module could not be imported!")
        super(EEGClientUsb, self).__init__(**kwargs)

        self.all_channels = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5',\
                            'FC1', 'FC2', 'FC6', 'T7', 'C3', 'Cz', 'C4', 'T8',\
                            'TP9', 'CP5', 'CP1', 'CP2', 'CP6', 'TP10', 'P7',\
                            'P3', 'Pz', 'P4', 'P8', 'PO9', 'O1', 'Oz', 'O2',\
                            'PO10', 'AF7', 'AF3', 'AF4', 'AF8', 'F5', 'F1',\
                            'F2', 'F6', 'FT9', 'FT7', 'FC3', 'FC4', 'FT8',\
                            'FT10', 'C5', 'C1', 'C2', 'C6', 'TP7', 'CP3',\
                            'CPz', 'CP4', 'TP8', 'P5', 'P1', 'P2', 'P6',\
                            'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'Fpz', 'F9',\
                            'AFF5h', 'AFF1h', 'AFF2h', 'AFF6h', 'F10', 'FTT9h',\
                            'FTT7h', 'FCC5h', 'FCC3h', 'FCC1h', 'FCC2h',\
                            'FCC4h', 'FCC6h', 'FTT8h', 'FTT10h', 'TPP9h',\
                            'TPP7h', 'CPP5h', 'CPP3h', 'CPP1h', 'CPP2h',\
                            'CPP4h', 'CPP6h', 'TPP8h', 'TPP10h', 'POO9h',\
                            'POO1', 'POO2', 'POO10h', 'Iz', 'AFp1', 'AFp2',\
                            'FFT9h', 'FFT7h', 'FFC5h', 'FFC3h', 'FFC1h',\
                            'FFC2h', 'FFC4h', 'FFC6h', 'FFT8h', 'FFT10h',\
                            'TTP7h', 'CCP5h', 'CCP3h', 'CCP1h', 'CCP2h',\
                            'CCP4h', 'CCP6h', 'TTP8h', 'P9', 'PPO9h', 'PPO5h',\
                            'PPO1h', 'PPO2h', 'PPO6h', 'PPO10h', 'P10', 'I1',\
                            'OI1h', 'OI2h', 'I2']

        self.ipmarker_server = ipmarker_server
        self.ip_lastmarker = (None, None)

        self._dSamplingInterval = 5000
        self._channelNames = None
        self._markerids = dict()
        self._markerNames = dict()
        self.marker_id_counter = 0
        self._stdblocksize = 100

        self.fmt = None
        self.channelids = None
        self.resolutions = None
        self.nChannels = None
        self.callbacks = list()
        self.raw_data = list()
        self.timestamp = 0
        self.dig_lastmarker = 0

        self.acquisition = BUASubprocess()
        self.acquisition.start()

        self.connect()

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
        self.callbacks.append(func)

    def block_length_ms(self):
        return (self.stdblocksize*1000)/self.dSamplingInterval

    def disconnect(self):
        if self.ipmarker_server is not None:
            self.ipmarker_server.stop()
            self.ipmarker_server.join()
        self.acquisition.stop()
        self.acquisition.join(timeout=5)
        del self.acquisition

    def new_marker_id(self):
        self.marker_id_counter += 1
        return self.marker_id_counter

    def connect(self, verbose=False):

        while self.acquisition.nchannels.value < 0:
            if not self.acquisition.is_alive():
                raise IOError, "Acquisition quit early!"
            time.sleep(.1)

        self.nChannels = self.acquisition.nchannels.value

        if self.nChannels == 0:
            raise IOError, "No Amplifiers found! Switch them on?"

        # generate all possible marker names and ids
        self._markerids['null'] = 0
        for s in range(1,256,1):
                self._markerids[str('S%d' % s)] = self.new_marker_id()
        for r in range(1,256,1):
            self._markerids[str('R%d' % r)] = self.new_marker_id()
        # generate reverse mapping
        for k,v in zip(self._markerids.iterkeys(), self._markerids.itervalues()):
            self._markerNames[v] = k

        # select channelnames
        self._channelNames = self.all_channels[:self.nChannels]

        # calculate raw-data threshold
        while self.acquisition.nextra_channels.value < 0 \
            or self.acquisition.nall_channels < 0:
            time.sleep(.1)
        self.nextra_channels = self.acquisition.nextra_channels.value
        self.nall_channels = self.acquisition.nall_channels.value
        self.min_raw_data = self.stdblocksize * self.nall_channels

    def read(self, nblocks=1, verbose=False):

        readblocks = 0
        while (readblocks < nblocks or nblocks == -1):
            # get enough raw data blocks
            self.gather_enough_data()

            # split data and marker in seperate arrays
            ndsamples, ndmarkers = self.separate()

            for f in self.callbacks:
                f(ndsamples, ndmarkers)

            readblocks += 1
        return readblocks

    def gather_enough_data(self):
        """ gets enough data to generate a block
            of size stdblocksize in channel and markers
        """
        while len(self.raw_data) < self.min_raw_data:
            data, timestamp = self.acquisition.read()

            if self.fmt is None:
                self.fmt = str("%dh" % (len(data)/2))

            values = struct.unpack(self.fmt, data)
            self.raw_data.extend(values)
            self.timestamp = timestamp-self.block_length_ms()

    def separate(self):
        """ separates the raw-data into the data- and marker-channels
        """
        packet = self.raw_data[0:self.min_raw_data]
        self.raw_data[0:self.min_raw_data] = []

        # example block-layout for 32 channels:
        # [marker:1][reserved:4][nchannels data:32] :||

        data = list()
        mark = list()

        for i in range(self.stdblocksize):
            mark.append(self.digital_marker(packet[0]))
            data.extend(packet[self.nextra_channels:self.nextra_channels+self.nChannels])
            packet[0:self.nall_channels] = []

        if self.ipmarker_server is not None:
            mark = self.insert_ip_markers(mark)

        ndata = numpy.array(data)
        ndata = ndata.reshape((self.stdblocksize,self.nChannels))

        return ndata, mark

    def insert_ip_markers(self, mark):
        while True:
            if not None in self.ip_lastmarker:
                m, t = self.ip_lastmarker
                self.ip_lastmarker = (None, None)
            else:
                m, t = self.ipmarker_server.read()
                if m is None or t is None:
                    break

            time_index = self.time2index(t)
            mark_index = self.mark2index(m)

            if time_index > len(mark)-1:
                self.ip_lastmarker = (m, t)
                break
            elif time_index >= 0 and time_index < len(mark):
                mark[time_index] = mark_index
            else:
                warnings.warn("Index did not fit: %d (%d, %s)" %
                    (time_index, mark_index, self.markerNames[mark_index]))

        return mark

    def mark2index(self, m):
        if not self.markerids.has_key(m):
            new = self.new_marker_id()
            self.markerids[m] = new
            self.markerNames[new] = m
            # print("added new marker %s with id %d" % (m, new))
        return self.markerids[m]

    def time2index(self, t):
        index = ((t-self.timestamp)*self.stdblocksize)/self.block_length_ms()
        # make sure its not negative!
        # (a negative index means that the marker was delayed!)
        return max(0, index)

    def digital_marker(self, raw_value):
        if raw_value == self.dig_lastmarker:
            value = -1
        else:
            m = raw_value & (self.dig_lastmarker^0xffff)
            smarker = (m & 0xff)
            rmarker = (m & 0xff00) >> 8
            if smarker != 0:
                value = self.markerids[str("S%d" % smarker)]
            elif rmarker != 0:
                value = self.markerids[str("R%d" % rmarker)]
            else:
                value = -1
            self.dig_lastmarker = raw_value
        return value

def dummylisten1(samples, markers):
    sys.stdout.write("*")
    sys.stdout.flush()
    
def dummylisten2(samples, markers):
    print samples[:32]
    
def dummylisten3(samples, markers):
    # print markers
    pass

if __name__ == '__main__':

    if len(sys.argv) > 2:
        try:
            host = str(sys.argv[1])
            print host
            if len(host.split(".")) != 4:
                raise Exception
            port = int(sys.argv[2])
            print port
            
            c = EEGClient(host=host, port=port)
        except Exception:
            print "could not generate meaning from args %s" % sys.argv
            c = EEGClient(host='127.0.0.1', port=51244)
            
    else:
        # Setup for Localhost  
        # c = EEGClient(host='127.0.0.1', port=51244)
        s = MarkerServer(port=55555)
        s.start()
        c = EEGClientUsb(ipmarker_server=s)

    def marker_listen(samples, markers):
        for i,m in enumerate(markers):
            if m != -1:
                print c.markerNames[m], i

    c.regcallback(marker_listen)

    print("running with %d channels" % c.nChannels)
    
    n = c.read(nblocks=1, verbose=True)
    start = time.time()
    n += c.read(nblocks=2500)
    stop = time.time()

    c.disconnect()
    
    total_b  = n*c.stdblocksize*c.nChannels*c.sample_size
    total_kb = float(total_b)/1000.0
    total_mb = float(total_kb)/1000.0
    total_gb = float(total_mb)/1000.0

    del c

    rate_mb_s = float(total_mb)/(stop-start)
    
    print "received %06f MB in %04f Seconds := %f MB/s" % (total_mb, (stop-start), rate_mb_s)
    
