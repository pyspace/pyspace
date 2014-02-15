#!/usr/bin/env python
# -*- coding: UTF-8 -*

"""eeg_stream.py
EEG client module.

Created by Timo Duchrow on 2008-08-26.
"""

__version__ = "$Revision: 456 $"
__all__ = ['EEGClient']

import sys
import socket
import struct
import numpy
import scipy
import cProfile
import pstats
import signal
import subprocess
import os
import time
import xmlrpclib
import random
import time
import glob
import warnings

file_path = os.path.dirname(os.path.abspath(__file__))
pyspace_path = file_path[:file_path.rfind('pySPACE')-1]
if not pyspace_path in sys.path:
    sys.path.append(pyspace_path)

try:
    from pySPACE.missions.support.CPP.shared_memory_access.build import shmaccess
    import_error = False
except Exception, e:
    import_error = e

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


class EEGClient(object):
    """EEG stream client for EEG stream protocol"""
    def __init__(self, host='127.0.0.1', port=51244, prio=1000, **kwargs):
        super(EEGClient, self).__init__()
        # variable names with capitalization correspond to structures members
        # defined in RecorderRDA.h
        self.host = host
        self.port = port
        self.nChannels = None           # number of channels
        self.sample_size = None
        self.protocol_version = None
        self.dSamplingInterval = None   # sampling interval
        self.resolutions = None         # list of resolutions / channel
        self.channelNames = None        # list of channel names
        self.channelids = dict()
        
        self.abs_start_time = 0

        self.priority = prio
        
        self.stdblocksize = None        # standard number of paints in one data block
        
        self.callbacks = list()
        self.meta = dict()        

        self.ndsamples = None           # last sample block read
        self.ndmarkers = None         # last marker block read
        self.markerids= dict()
        self.markerNames = dict()       # dictionary with marker names
        self.nmarkertypes = 0           # number of different marker types

        self.lostmarker = False         # for two markers corresponding to one sample in last sample of a block
        self.lostmarkertypedesc = None
        self.running = True
        
        
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
            
    
    def syncread(self):
        """Perform synchronous read of one data block"""
        pass
    
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
        (self.nChannels, self.stdblocksize, self.dSamplingInterval, self.sample_size, self.protocol_version) = \
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
        self.channelNames = list()
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
        self.markerNames = dict()
        self.markerids = dict()
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

class EEGClientShm(EEGClient):
    """ Acquire raw streamed eeg-data from shared memory.

        This implementation uses a custom c-extension located
        in pySPACE/missions/support/CPP/shared_memory_access.
        Please follow instruction on how to build etc. in the
        corresponding README file. (TODO:!)
    """
    
    def __init__(self, shm_name=None, **kwargs):
        """

        :param shm_name:
            the name of the shared memory segment - usually
            a 32-bit integer.
        """
        super(EEGClientShm, self).__init__(*kwargs)

        if import_error:
            raise import_error
        if shm_name is None:
            raise IOError, str("please specify shared memory name!")

        self.connect()

    def connect(self, verbose=False):
        try:
            shmaccess.connect(22)
        except Exception as e:
            raise e

    def _readmsg(self, msg_type='all', verbose=False):
        while True:
            try:
                read = shmaccess.read()
            except shmaccess.StreamGap as e:
                # print e
                continue
            except shmaccess.StopIteration:
                print "dataset completely streamed.."
                break

            break

        (nSize, nType) = struct.unpack(fmt_RDA_MessageHeader, read[:struct.calcsize(fmt_RDA_MessageHeader)])

        payload = read[struct.calcsize(fmt_RDA_MessageHeader):]

        # invoke appropiate handler for decoding
        if nType == T_RDA_MessageStart:
            self._getstartmsg(payload, verbose=False)
        elif nType == T_RDA_MessageData:
            self._getdatamsg(payload)
        elif nType == T_RDA_MessageMuxData:
            self._getmuxdatamsg(payload, verbose=False)
        elif nType == T_RDA_MessageStop:
            self._getstopmsg(payload)
        elif nType == T_RDA_MessageData32:
            self._getdata32msg(payload)

        return nType

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



def dummylisten1(samples, markers):
    sys.stdout.write("*")
    sys.stdout.flush()
    
def dummylisten2(samples, markers, meta=dict()):
    #print "samples:", samples
    print "markers:", markers
    
def dummylisten3(samples, markers, meta=dict()):
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
        c = EEGClientShm(shm_name = 22)

    c.connect(verbose=True)
    c.regcallback(dummylisten1)
    
    n = c.read(nblocks=1, verbose=True)
    start = time.time()
    n += c.read(nblocks=10000)
    # n += c.read(nblocks=1, verbose=True)
    # n += c.read(nblocks=-1)
    stop = time.time()
    
    total_b  = n*c.stdblocksize*c.nChannels*c.sample_size
    total_kb = float(total_b)/1000.0
    total_mb = float(total_kb)/1000.0
    total_gb = float(total_mb)/1000.0
    
    rate_mb_s = float(total_mb)/(stop-start)
    
    print "received %06f MB in %04f Seconds := %f MB/s" % (total_mb, (stop-start), rate_mb_s)
    
