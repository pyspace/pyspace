# -*- coding: UTF-8 -*

""" Performs windowing of incoming stream and produces instances of fixed
length for preprocessing and classification.

The :class:`~pySPACE.missions.support.windower.SlidingWindower` class performs
windowing for the online setting where no markers are available.

The :class:`~pySPACE.missions.support.windower.MarkerWindower` class extracts
windows according to definitions like the presence or non-presence of markers.
Additionally, exclude conditions can be defined that exclude certain markers in
proximity to extracted events.

The :class:`~pySPACE.missions.support.windower.WindowFactory` loads a windowing
specification from a yaml file. The window definitions are then stored in a
dictionary which is then used by one of the Windowers
(:class:`~pySPACE.missions.support.windower.MarkerWindower`,
:class:`~pySPACE.missions.support.windower.SlidingWindower` etc.) to cut the
incoming data stream.

The Windower definition is always application specific and can contain many
keys/values. In order to construct your own windower definition see the short
explanation in :class:`~pySPACE.missions.support.windower.MarkerWindower`.

Time is always measured in ms.
If there are mistakes in time, this should be because of
unknown block size or frequency.

Additionally include conditions can be added to ensure the presence of certain 
markers in a specific range. So 'or' conditions between the conditions are 
reached by repeating the definitions of the marker with different exclude or 
include definitions and 'and' conditions are simply reached by concatenation.
Negation is now possible by switching to the other kind of condition.

:Author: Timo Duchrow
:Created: 2008/08/29
:modified: Mario Michael Krell (include and exclude defs)
"""

__version__ = "$Revision: 451 $"
# __all__ = ['SlidingWindower, MarkerWindower, ExcludeDef, LabeledWindowDef']

import sys
import os
import numpy
import math
import yaml

from pySPACE.resources.data_types.time_series import TimeSeries

if __name__ == '__main__':
    import unittest

debug = False
warnings = False
# debug = True
# warnings = True


class Windower(object):
    """Windower base class"""

    def __init__(self, data_client):
        self.data_client = data_client
        if debug:
            print("acquisition frequency:\t %d Hz"% data_client.dSamplingInterval)
            print("server block size:\t %d samples"% data_client.stdblocksize)
    
    def _mstosamples(self, ms):
        """Convert from milliseconds to number of samples based on the 
        parameters of data_client."""
        if self.data_client.dSamplingInterval is None:
            raise Exception, "data_client needs to be connected to determine "\
                "acquisition frequency"
        nsamples = ms * self.data_client.dSamplingInterval / 1000.0
        if nsamples != int(nsamples):
            import warnings
            warnings.warn(" %s ms can not be converted to int number"\
                " of samples with current sampling frequency (%s Hz)" \
                % (ms, self.data_client.dSamplingInterval))
            # since the current float representation is not equal to int, round
            nsamples = round(nsamples)
        return int(nsamples)
    
    def _samplestoms(self, samples):
        """Convert from number of samples to milliseconds based on the 
        parameters of data_client."""
        if self.data_client.dSamplingInterval is None:
            raise Exception, "data_client needs to be connected to determine "\
                "acquisition frequency"
        ms = samples * 1000.0 / self.data_client.dSamplingInterval
        return ms

    @classmethod
    def _load_window_spec(cls, windower_spec="", local_window_conf=False):
        """
        Load the window definitions to extract the labeled samples

        **Parameters**

            :windower_spec:
                file name of the windower specification

                (*optional, default:`default_windower_spec`*)

            :local_window_conf:
                Windower file is looked up in the local directory if set True.

                .. note: As default the spec_dir from `pySPACE.configuration`
                         is used to look up spec files.

                Otherwise it is looked up in the subdirectory `windower` in
                `node_chains` in the `spec_dir`, which is the better way of
                using it.

                (*optional, default: False*)
        """
        if windower_spec == "":
            window_definitions = WindowFactory.default_windower_spec()
            return window_definitions

        # check for 'yaml'-ending of the file
        if ".yaml" not in windower_spec: # general substring search!
            windower_spec = windower_spec + ".yaml"

        if local_window_conf:
            if windower_spec.count('/')==0:
                #windows file should be in local directory
                windower_spec_file_path = "./" + windower_spec
            else:
                #windower spec contains complete path
                windower_spec_file_path = windower_spec
        else:
            import pySPACE
            windower_spec_file_path = os.path.join(pySPACE.configuration.spec_dir,
                                               "node_chains","windower",
                                               windower_spec)

        if os.path.exists(windower_spec_file_path):
            windower_spec_file = open(windower_spec_file_path, 'r')
            window_definitions = \
                WindowFactory.window_definitions_from_yaml(windower_spec_file)
            windower_spec_file.close()
        else:
            raise IOError('Windower: Windowing spec file '
                                + windower_spec_file_path + ' not found!')

        return window_definitions


class SlidingWindower(Windower):
    """An iterable class that produces sliding windows for online classification."""

    def __init__(self, data_client, windowsizems=1000, stridems=100, underfull=False):
        
        super(SlidingWindower, self).__init__(data_client)
        
        # register sliding windower as consumer of EEG stream client
        data_client.regcallback(self._addblock)
        
        # convert intervals in ms to number of samples
        self.stridems = stridems
        self.stride = self._mstosamples(stridems)
        self.windowsizems = windowsizems
        self.windowsize = self._mstosamples(windowsizems)

        self.underfull = underfull
            
        if self.windowsizems % self.stridems != 0:
            raise Exception, "window size needs to be a multiple of stride"
        if self.stride % data_client.stdblocksize != 0:
            raise Exception, "stride needs to be a multiple of blocksize " \
                "(server is sending block size %d)" % data_client.stdblocksize
                
        # NB: acqusition frequency is called mistakingly called ``sampling interval'''
        # in Brain Products protocol.

        # calculate number of required blocks
        self.buflen = int(self.windowsize / data_client.stdblocksize)
        
        # init to ring buffers, one for the samples, one for the markers
        self.samplebuf = RingBuffer(self.buflen)
        self.markerbuf = RingBuffer(self.buflen)

        # determine how many blocks need to be read at once with client
        self.readsize =  self.stride / data_client.stdblocksize
        
        if debug:
            print("buflen:\t %d" % self.buflen)
            print("readsize:\t %d" % self.readsize)
            
    def __iter__(self):
        return self
        
    def next(self):
        """Retrieve the next window according to windowsize and stride."""  

        nread = 0   # number of blocks actually read
        if len(self.samplebuf) == 0:
            # the ring buffer is still completely empty, fill it
            nread = self.data_client.read(nblocks=self.buflen)
            if nread < self.buflen:
                raise StopIteration
        else:
            # just replace required number of blocks
            nread = self.data_client.read(nblocks=self.readsize)
            if nread < self.readsize:
                raise StopIteration
        
        # copy ring buffer into one long array
        ndsamplewin = numpy.hstack(self.samplebuf.get())
        ndmarkerwin = numpy.hstack(self.markerbuf.get())
        return (ndsamplewin, ndmarkerwin)  
                
    def _addblock(self, ndsamples, ndmarkers):
        """Add incoming data block to ring buffers"""
        self.samplebuf.append(ndsamples)
        self.markerbuf.append(ndmarkers)


class MarkerWindower(Windower):

    """returns (<numpy.ndarray> window, <str> class)

    MarkerWindower maintains a ring buffer for incoming sample blocks. The
    buffer is divided into three segments:

    Example::

        t0                  t1                  t3                  t4     <---
        +---------+---------+---------+---------+---------+---------+---------+
        | block 1 | block 2 | block 3 | block 4 | block 5 | block 6 | block 7 |
        +---------+---------+---------+---------+---------+---------+---------+

        |<      prebuflen = 3        >|         |<      postbuflen = 3       >|

        [////////////////////////////][---------][/////////////////////////////]
                    history           ``current''          lookahead
                                                                     __________
                                                                         scan

    MarkerWindower scans windows for markers as they come in (block 7 in
    example). When the block passes into the ``current`` section all windows
    are extracted that meet the constraints. To accomplish this prebuflen
    and postbuflen have been calculated so that the buffer enables extraction
    of sufficient window lengths for all window definitions as well as
    lookaheads.

    A windowdef looks like:

    .. code-block:: yaml

        startmarker : "S  8"
        endmarker : "S  9"
        skip_ranges :
                 - {start : 0, end: 300000}
        window_defs :
             s16:
                 classname : LRP
                 markername : "S 16"
                 startoffsetms : -1280
                 endoffsetms : 0
                 excludedefs : []
                 includedefs : [immediate response]
             null:
                 classname : NoLRP
                 markername : "null"
                 startoffsetms : -1280
                 endoffsetms : 0
                 excludedefs : [all]
        exclude_defs:
              all:
                markernames : ["S  1", "S  2", "S  8", "S 16", "S 32"]
                preexcludems : 2000
                postexcludems : 2000
        include_defs:
             immediate_response:
                 markernames : ["S 32"]
                 preincludems: -200
                 postincludems: 1200

    **Parameters**

        :startmarker:   name of the marker where at the earliest cutting begins
        :endmarker:     name of the marker where at the latest cutting ends
        :skip_ranges:   Not completely implemented!
                        The 'end' component results in
                        the parameter skipfirstms which tells
                        the windower, which time points to skip
                        at the beginning.

                        .. todo:: Change parameterization or code.

        :window_def:    includes names of definitions of window cuts
        :classname:     name of the label given to the window, when cut
        :markername:    name of the marker being in the 'current block'

                        .. note:: The ``null`` marker is a synthetic marker,
                                  which is internally added to the stream
                                  every *nullmarker_stride_ms* milliseconds.
                                  Currently, this parameter has to be set
                                  separately and is 1000ms by default.

        :startoffsetms: start of the window relative to the marker in the 'current block'
        :endoffsetms:   end of the window relative to the marker in the 'current block'
        :jitter:        Not implemented! Was intended to add an
                        artificial jittering during the segmentation.

                        .. todo:: Delete completely!

        :exclude_defs:  excludes each marker in markernames defined by the interval
                        '[-preexcludems, postexludems]' relative to the window
                        marker lying at zero
        :preexcludems:  time before the window marker, where the exclude markers
                        are forbidden. This time can be chosen negative,
        :postexcludems: time after the window marker, where the exclude markers
                        are forbidden. This time can be chosen negative.
        :include_defs:  everything is the same to exclude defs, except,
                        that one of the specified markers has to lie in the
                        interval.

        Time is always measured in ms.
        If there are mistakes in time, this should be because of
        unknown block size or frequency.

    **Class Parameters**
        :data_client: Client, delivering the data
        :windowdefs:
            List of window definitions generated by
            :func:`WindowFactory.create_window_defs`

            (*optional, default: None*)

        :debug: Enable debug print outs to command line

            (*optional, default: False*)

        :nullmarker_stride_ms:
            Set artificial markers with this constant distance into the stream
            with the Name "null". If this parameter is set to *None*,
            no artificial markers are generated.

            (*optional, default: 1000*)

        :no_overlap:
            Ignore the last sample in each window (important for streaming data)

            (*optional, default: False*)

        :data_consistency_check:
            Currently it is only checked, that the standard deviation is not 0

            (*optional, default: False*)

    """
    # ==================
    # = Initialization =
    # ==================
    
    def __init__(self, data_client, windowdefs=None, debug=False,
            nullmarker_stride_ms=1000, no_overlap=False,
            data_consistency_check=False):
        super(MarkerWindower, self).__init__(data_client)

        self.data_client = data_client
        data_client.regcallback(self._addblock)

        self.windowdefs = windowdefs
        # Occurring key errors because of missing marker are collected to deliver
        # just one warning. The treatment differs by usage: missing window 
        # markers deliver no window, excludedefs just ignore the key error and
        # includedefs would also deliver no window, because the includedef can 
        # not be fulfilled if the marker can not be found.
        self.keyerror = {}
        
        # flags that indicate if we passed specified start and stop markers
        self.start = False
        self.end = False
        
        # determines the minimal marker offset for a window to be cut out
        # this is chanced when the start marker and the window marker are in 
        # the same block
        self.min_markeroffset = 0
        
        # occurrence of marker of a specific type in the buffer
        self.buffermarkers = dict() 

        self.nullmarker_id = 0
        self.nullmarker_stride = None
        if nullmarker_stride_ms is not None:
            self.nullmarker_stride = self._mstosamples(nullmarker_stride_ms)
            self.buffermarkers["null"] = list()
            
        self.next_nullmarker   = 0  # number of samples to go until next
                                    # nullmarker
        self.nsamples_postscan = 0  # number of samples after extraction point
        self.nsamples_prescan  = 0  # number of samples before extraction point
        self.nmarkers_prescan  = 0  # max time markers should be remembered
                                    # important for reduced buffermarkers
        # nmarkers_postscan is equivalent to nsamples_postscan

        if debug:
            for wdef in windowdefs:
                print wdef

        # determine maximum extents of buffers and maximum time that markers
        # should be remembered
        (nsamples_prescan, nsamples_postscan, nsamples_max_premarkers) = \
            self._max_scan_ranges()
        self.nsamples_prescan  = nsamples_prescan
        self.nsamples_postscan = nsamples_postscan
        self.nmarkers_prescan  = nsamples_max_premarkers
        
        
        if debug:   
            print " nsamples_prescan", nsamples_prescan
            print " nsamples_postscan", nsamples_postscan
        
        # calculate buffer length in terms of std blocksize
        self.prebuflen = int(math.ceil(float(nsamples_prescan)/ \
                                                        data_client.stdblocksize))
        self.postbuflen = int(math.ceil(float(nsamples_postscan)/ \
                                                        data_client.stdblocksize))
        
        # + one middle block (the ``current'' block)
        self.buflen = self.prebuflen + 1 + self.postbuflen

        if debug:
            print " stdblocksize", data_client.stdblocksize
            print " prebuflen", self.prebuflen
            print " postbuflen", self.postbuflen
            print " buflen", self.buflen
            print
        
        # initialize the buffers
        self.samplebuf = RingBuffer(self.buflen)
        self.markerbuf = RingBuffer(self.buflen)

        # determine the offset of the first sample in the incoming block
        self.incoming_block_offset = self.postbuflen*self.data_client.stdblocksize
    
        # the list of the markers in the current block that have not yet been
        # handled (by calling the next() method of the iterator protocol)
        self.cur_extract_windows = list()
        
        # total number of blocks read
        self.nblocks_read_total = 0
        # additional parameters, e.g. security checks etc
        self.data_consistency_check = data_consistency_check
        self.no_overlap = no_overlap
    
    def _max_scan_ranges(self):
        """Scan window and constraint definitions to determine maximum extent
        of buffer for marker and for samples. Return (max_postscan_samples, 
        max_prescan_samples, max_prescan_markers)"""
        # number of samples before and after extraction point that needs to
        # be available at all times, always positive
        nsamples_prescan = 0
        nsamples_postscan = 0
        
        # either positive or negative offset as in window definition
        nsamples_prewin = 0
        nsamples_postwin = 0
        
        # determine size of pre and post buffer to accommodate all window
        # definitions
        for wdef in self.windowdefs:
            if wdef.startoffsetms > wdef.endoffsetms:
                raise Exception, "illegal window definition: "\
                "startoffset needs to be smaller then endoffset."
            nsamples_prewin = min(nsamples_prewin,
                self._mstosamples(wdef.startoffsetms))
            nsamples_postwin = max(nsamples_postwin,
                self._mstosamples(wdef.endoffsetms))
        
        # TODO: If-clauses may be droped or replaced by asserts:
        # adjust pre-buffer length
        if nsamples_prewin < 0:
            nsamples_prescan = abs(nsamples_prewin)
        else:
             nsample_prescan = 0
             
        # end of window is always later than start
        if nsamples_postwin < 0:
            nsamples_postscan = 0
        else:
            nsamples_postscan = nsamples_postwin
            
        nmarkers_prescan = nsamples_prescan
            
        # For the adaptions on the excludedefs and includedefs just the marker
        # are relevant and not the samples (nmarkers_prescan).
        
        # extend lookahead (nsamples_postscan) and nmarkers_prescan to cover 
        # excludes ...
        for wdef in self.windowdefs:
            if wdef.excludedefs is not None:
                for exc in wdef.excludedefs:
                    nsamples_postscan = max(nsamples_postscan, 
                        self._mstosamples(exc.postexcludems))
                    nmarkers_prescan = max(nmarkers_prescan,
                        self._mstosamples(exc.preexcludems))
        #...and includes in the same range.
            if wdef.includedefs is not None:
                for inc in wdef.includedefs:
                    nsamples_postscan = max(nsamples_postscan, 
                        self._mstosamples(inc.postincludems))
                    nmarkers_prescan = max(nmarkers_prescan,
                        self._mstosamples(inc.preincludems))
        return (int(nsamples_prescan), int(nsamples_postscan),
                int(nmarkers_prescan))

    # ===============================
    # = Handling of incoming blocks =
    # ===============================
    
    def _decmarkeroffsets(self):
        """Decrement all offsets for markers in buffer as new blocks come in."""

        markers = self.buffermarkers.keys()
        # decrement of marker offsets in buffer
        for marker in markers:
            # remove old markers that are out of scope
            new_offsets = [x - self.data_client.stdblocksize
                for x in self.buffermarkers[marker] 
                if x -self.data_client.stdblocksize >= (-1)*self.nmarkers_prescan
                ]
            if len(new_offsets) == 0:
                del self.buffermarkers[marker]
            else:
                self.buffermarkers[marker] = new_offsets

                
    def _addblock(self, ndsamples, ndmarkers):
        """Add incoming block to ring buffer."""
        self.nblocks_read_total += 1 # increment total number of blocks
        self._decmarkeroffsets()    # adjust marker offsets
        self.samplebuf.append(ndsamples)
        self.markerbuf.append(ndmarkers)
        self._insertnullmarkers()   # insert null markers
        self._scanmarkers(ndmarkers) # scan for new markers

    def _insertnullmarkers(self, debug=False):
        """Insert epsilon markers according to nullmarker stride."""
        
        if self.nullmarker_stride is None:
            return
        
        if debug:
            print "next_nullmarker", self.next_nullmarker
        self.nullmarker_id = self.data_client.markerids["null"]
        while self.next_nullmarker < self.data_client.stdblocksize:
            if not self.buffermarkers.has_key(self.nullmarker_id):
                self.buffermarkers[self.nullmarker_id] = list()
            self.buffermarkers[self.nullmarker_id].append(
                self.incoming_block_offset + self.next_nullmarker)
            if debug:
                print "inserting", \
                               self.incoming_block_offset + self.next_nullmarker

            self.next_nullmarker += self.nullmarker_stride
        self.next_nullmarker -= self.data_client.stdblocksize

    def _scanmarkers(self, ndmarkers, debug=False):
        """Scan incoming block for markers.
        
        self.buffermarkers contains offsets of markers w.r.t. to ``current``
        block @ position 0
        """
        for i, marker in enumerate(ndmarkers):
            if marker != -1:
                if self.buffermarkers.has_key(marker):
                    self.buffermarkers[marker].append(
                                                 self.incoming_block_offset + i)
                else:
                    self.buffermarkers[marker]= [self.incoming_block_offset + i]
        if debug:
            print " scanmarkers ", self.buffermarkers

    # ================================
    # = Iteration protocol interface =
    # ================================

    def __iter__(self):
        self.nwindow=0
        return self

    def next(self, debug=False):
        """Return next labeled window when used in iterator context."""
        while len(self.cur_extract_windows) == 0:
            # fetch the next block from data_client
            if debug:
                print "reading next block"
            self._readnextblock()
            self._extract_windows_cur_block()
            if debug:
                print "  buffermarkers", self.buffermarkers
                print "  current block", self.samplebuf.get()[self.prebuflen][1,:]                           
                # print "  current extracted windows ", self.cur_extract_windows
    
        (windef_name, current_window, class_, start_time, end_time, markers_cur_win) = \
            self.cur_extract_windows.pop(0)

        # TODO: Replace this by a decorator or something similar
        current_window = numpy.atleast_2d(current_window.transpose())
        current_window = TimeSeries(
                input_array=current_window,
                channel_names=self.data_client.channelNames,
                sampling_frequency=self.data_client.dSamplingInterval,
                start_time = start_time,
                end_time = end_time,
                name = "Window extracted @ %d ms, length %d ms, class %s" % \
                    (start_time, end_time - start_time, class_),
                marker_name = markers_cur_win                
        )
        
        current_window.generate_meta()
        current_window.specs['sampling_frequency'] = self.data_client.dSamplingInterval
        current_window.specs['wdef_name'] = windef_name
        self.nwindow += 1                                                

        # return (ndsamplewin, ndmarkerwin)
        return (current_window, class_)

    def _readnextblock(self):
        """Read next block from EEG stream client."""
        nread = 0   # number of blocks actually read
        if len(self.samplebuf) == 0:
            # fill ring buffer
            nread = self.data_client.read(nblocks=self.buflen)
            if nread < self.buflen:
                raise StopIteration
            for marker_id, offsets in self.buffermarkers.iteritems():
                for offset in offsets:
                    if offset < 0 and warnings:
                        print >>sys.stderr, "warning: markers ignored when "\
                        "initializing buffer"
        else:
            # read the next block
            nread = self.data_client.read()
            if nread == 0:
                raise StopIteration

    def _extract_windows_cur_block(self):
        """Add windows for markers in current block to self.cur_extract_windows."""

        for wdef in self.windowdefs:
            # resolve to id
            # if id does not exist ignore this window definition and go on
            try:
                markerid = self.data_client.markerids[wdef.markername]
            except KeyError, e:
                e=str(e)
                if not self.keyerror.has_key(e):
                    self.keyerror[e]=wdef
                    print 
                    print "windowdef warning: Marker ", e, "not found in the"
                    print self.keyerror[e]
                continue
            # if there exist a startmarker in the wdef resolve to id
            if wdef.startmarker != None:
                try:
                    startid = self.data_client.markerids[wdef.startmarker]
                except KeyError, e:
                    e=str(e)
                    if not self.keyerror.has_key(e):
                        self.keyerror[e]=wdef
                        print 
                        print "windowdef warning: Startmarker ", e, "not found in the"
                        print self.keyerror[e]
                    continue
                # check if startmarker id has been seen in current buffer scope
                if self.buffermarkers.has_key(startid) and \
                    self.buffermarkers[startid][0] < self.data_client.stdblocksize:
                    # if the startmarker is found we delete it from the window
                    # definition because from now on windows can be cut
                    wdef.startmarker = None
                    # in addition a start_flag is set and the min markeroffset
                    # for markers in this current block
                    self.start = True
                    self.min_markeroffset = self.buffermarkers[startid][0]
                else: 
                    continue
            # check if corresponding marker id has been seen in current 
            # buffer scope or if the stopmarker is already True
            if not self.buffermarkers.has_key(markerid) or self.end==True:
                continue
            # now prepare extraction windows for markers in the ``current'' block
            # check if includedefs and excludedefs are fulfilled
            for markeroffset in self.buffermarkers[markerid]:  
                if self.min_markeroffset <= markeroffset < self.data_client.stdblocksize and \
                   self._check_exclude_defs_ok(markeroffset, wdef.excludedefs) and \
                   self._check_include_defs_ok(markeroffset, wdef.includedefs):
                    try:
                        (extractwindow, start_time, end_time, markers_cur_win) = \
                            self._extractwindow(
                                markeroffset,
                                self._mstosamples(wdef.startoffsetms),
                                self._mstosamples(wdef.endoffsetms))
                        if self.data_consistency_check:
                            # test if extracted window has std zero
                            std = numpy.std(extractwindow,axis=1)
                            if sum(std < 10**-9): #can be considered as zero
                                # filter the channel names where std equals zero
                                zero_channels = [self.data_client.channelNames[index]
                                                 for (index,elem) in enumerate(std) 
                                                 if elem < 10**-9]
                                print "Warning: Standard deviation of channel(s) " \
                                      " %s in time interval [%.1f,%.1f] is zero!" \
                                      % (str(zero_channels), start_time, end_time)
                        
                        if wdef.skipfirstms is None or \
                            start_time > wdef.skipfirstms:
                            self.cur_extract_windows.append((wdef.windef_name,
                                extractwindow, wdef.classname, start_time,
                                end_time, markers_cur_win))
                    except MarkerWindowerException, e:
                        if warnings:
                            print >>sys.stderr, "warning:", e
            
            # if this was the first window, adjust min_markeroffset before we 
            # move to the next block
            if self.start:
                self.min_markeroffset = 0
                self.start = False
            
            # check if the end of the stream is reached                
            if wdef.endmarker != None:
                try:
                    endid = self.data_client.markerids[wdef.endmarker]
                except KeyError, e:
                    e=str(e)
                    if not self.keyerror.has_key(e):
                        self.keyerror[e]=wdef
                        print 
                        print "windowdef warning: Endmarker ", e, "not found in the"
                        print self.keyerror[e]
                    continue
                # check if endmarker id has been seen in current buffer scope
                if self.buffermarkers.has_key(endid):
                    if self.buffermarkers[endid][0] < 0 and not self.end:
                        # if the endmarker is reached we set the end-flag for window
                        # cutting to True
                        self.end = True
                        print "Endmarker found!"
                        raise StopIteration
                
    def _check_exclude_defs_ok(self, markeroffset, excludedefs):
        """ Check whether the exclude definitions match
        
        .. note:: 
            Changes in this section need to be checked also
            in the following -check_include_defs_ok class,
            because they are very similar.
        """
        
        # Nothing to do if there are no excludedefs
        if excludedefs is None or len(excludedefs)==0:
            return True
                    
        # Check each exclude definition
        for exc in excludedefs:
            preexclude = markeroffset - self._mstosamples(exc.preexcludems)
            if self.no_overlap:
                postexclude = markeroffset + self._mstosamples(exc.postexcludems)
            else:
                postexclude = markeroffset + 1 + self._mstosamples(exc.postexcludems)
            
            # Get markerid and skip if it does not exist.
            try:
                excmarkerid = self.data_client.markerids[exc.markername]
            except KeyError, e:
                e=str(e)
                if not self.keyerror.has_key(e):
                    self.keyerror[e]=exc
                    print
                    print "exclude warning: Marker ", e, "not found in the ..."
                    print self.keyerror[e]
                continue
                
            # Skip if no proximal exclude marker seen
            if not self.buffermarkers.has_key(excmarkerid):
                continue
                
            # Not ok, if exclude marker falls into exclude range
            # This is the important part of this check!
            # Question: Why not exc_marker <=postexclude or exc_marker > preexclude?
            # Answer: Before one added
            for exc_marker in self.buffermarkers[excmarkerid]:
                # The inequation lets you exclude the same marker 
                # only a few seconds before or after the current marker,
                # to deal with unwanted marker repetitions.
                if preexclude <= exc_marker < postexclude and \
                        exc_marker != markeroffset:
                    return False
        return True #if all excludedefs are fulfilled
        

    def _check_include_defs_ok(self, markeroffset, includedefs):
        """Check whether all the include definitions match"""
        #Code adapted from the previous exclude-check
        
        #Checks if there are includedefs
        if includedefs is None or len(includedefs)==0:
            return True
                    
        # Check each include definition    
        for inc in includedefs:
            preinclude = markeroffset - self._mstosamples(inc.preincludems)
            if self.no_overlap:
                postinclude = markeroffset + self._mstosamples(inc.postincludems)
            else:
                postinclude = markeroffset + 1 + self._mstosamples(inc.postincludems)
            # Check allways breaks if the neccessary marker does not exist.
            try:
                incmarkerid = self.data_client.markerids[inc.markername]
            except KeyError, e:
                e=str(e)
                if not self.keyerror.has_key(e):
                    self.keyerror[e]=inc
                    print 
                    print "include warning: Marker ", e, "not found in the ..."
                    print self.keyerror[e]
                return False
            # Break if no proximal include marker seen (different to exclude,
            # because include markers need to be proximal.)
            if not self.buffermarkers.has_key(incmarkerid):
                return False
            # Not ok, if no include marker falls into include range
            # It is important to remark that no includedefs using the current
            # marker are allowed!
            check = False  # remembers if a check succeeded
            for inc_marker in self.buffermarkers[incmarkerid]:
                # inequality to use he same marker name for include def
                if preinclude <= inc_marker < postinclude and \
                        inc_marker != markeroffset:
                    check = True
            if not check:
                return False
        return True  # If all includedefs are fulfilled
        
    def _extractwindow(self, cur_sample_block_offset, start_offset, end_offset,
                       debug=False):
        """ Extracts a sample window from the ring buffer and consolidates it
        into a single numpy array object."""
        # calculate current position with respect to prebuffer start
        cur_sample_buf_offset = self.prebuflen * self.data_client.stdblocksize \
                                + cur_sample_block_offset
        buf_extract_start = cur_sample_buf_offset + start_offset
        
        if self.no_overlap:
            buf_extract_end = cur_sample_buf_offset + end_offset
        else:
            buf_extract_end = cur_sample_buf_offset + 1 + end_offset
            
        if debug:
            print "buf_extract_start", buf_extract_start
            print "buf_extract_end", buf_extract_end

        if buf_extract_start < 0:
            raise MarkerWindowerException,"not enough history data available" \
                         " to extract window with start offset of %d samples" \
                                                                 % start_offset
        assert buf_extract_end >= 0
        assert buf_extract_end <= self.buflen * self.data_client.stdblocksize
        
        end_time_samples = \
            (self.nblocks_read_total * self.data_client.stdblocksize) - \
            (self.buflen * self.data_client.stdblocksize - buf_extract_end)
        end_time = self._samplestoms(end_time_samples)
        start_time_samples = end_time_samples - \
                                      (buf_extract_end - buf_extract_start) + 1
        start_time = self._samplestoms(start_time_samples)
        
        # copy ring buffer into one long array and extract subwindow
        ndsamplewin = numpy.hstack(self.samplebuf.get())[:,buf_extract_start:buf_extract_end]
        markers_cur_window = self._extract_markers_cur_window(buf_extract_start, buf_extract_end)

        return (ndsamplewin, start_time, end_time, markers_cur_window)
    
    def _extract_markers_cur_window(self, buf_extract_start, buf_extract_end):
        """ Filter out all markers that lie in the current window
            to store this information. The markers are stored with their clear name
            and temporal offset.
        """
        markers_cur_window = dict()
        for marker_id in self.buffermarkers:
            for offset in self.buffermarkers[marker_id]:
                if offset >= 0 \
                        and buf_extract_start <= offset < buf_extract_end \
                        and marker_id != self.nullmarker_id:
                    marker = self.data_client.markerNames[marker_id]
                    if not markers_cur_window.has_key(marker):
                        markers_cur_window[marker] = list()
                    markers_cur_window[marker].append(self._samplestoms(offset-buf_extract_start))
        return markers_cur_window


# =====================
# = Exception classes =
# =====================

class MarkerWindowerException(Exception):
    def __init__(self, arg):
        super(MarkerWindowerException, self).__init__(arg)
        

# ==================================================
# = Support classes for definitions of constraints =
# ==================================================

class LabeledWindowDef(object):
    """Labeled window definition that is to be extracted from EEG stream."""
    def __init__(self, windef_name, classname, markername, startoffsetms, 
                 endoffsetms, excludedefs=None,includedefs=None, 
                 skipfirstms=None, jitter=None,startmarker=None,endmarker=None):
        super(LabeledWindowDef, self).__init__()

        self.windef_name = windef_name
        self.classname = classname        
        self.markername = markername    # can be None
        self.excludedefs = excludedefs
        self.includedefs = includedefs
        self.startoffsetms = startoffsetms
        self.endoffsetms = endoffsetms
        self.skipfirstms = skipfirstms
        self.startmarker = startmarker
        self.endmarker = endmarker
        
    def __str__(self):
        d = {'wdef' : self.windef_name, 'cls' : self.classname, 
             'skip_first' : self.skipfirstms, 'marker' : self.markername, 
             'start' : self.startoffsetms, 'end' : self.endoffsetms}
        if d['marker'] == '':
            d['marker'] = "''"
        str_ = 'LabeledWindowDef %(wdef)s\n  class: %(cls)s\n  skip first: '\
               '%(skip_first)s\n  marker: %(marker)s\n  start: %(start)d ms\n'\
               '  end: %(end)d ms\n' % d
        # append exclude definitions if any
        if self.excludedefs:
            for exc in self.excludedefs:
                for line in str(exc).splitlines():
                    str_ += "  %s\n" % line
        # append include definitions if any
        if self.includedefs:
            for inc in self.includedefs:
                for line in str(inc).splitlines():
                    str_ += "  %s\n" % line
        return str_

class ExcludeDef(object):
    """Definition of exclude constraints for window extraction."""
    def __init__(self, markername, preexcludems, postexcludems):
        super(ExcludeDef, self).__init__()
        self.markername = markername
        self.preexcludems = preexcludems
        self.postexcludems = postexcludems

    def __str__(self):
        d = {'name' : self.markername, 'pre' : self.preexcludems, 
            'post' : self.postexcludems}
        str_ = 'ExcludeDef:\n  markername: %(name)s\n'\
            '  preexclude: %(pre)d ms\n'\
            '  postexclude: %(post)d ms\n' % d
        return str_


class IncludeDef(object):
    """Definition of include constraints for window extraction."""
    # Same as exclude but with 'in'
    def __init__(self, markername, preincludems, postincludems):
        super(IncludeDef, self).__init__()
        self.markername = markername
        self.preincludems = preincludems
        self.postincludems = postincludems

    def __str__(self):
        d = {'name' : self.markername, 'pre' : self.preincludems, 
            'post' : self.postincludems}
        str_ = 'IncludeDef:\n  markername: %(name)s\n'\
            '  preinclude: %(pre)d ms\n'\
            '  postinclude: %(post)d ms\n' % d
        return str_

# ===============================
# = Ring buffer support classes =
# ===============================

class RingBuffer:
    """Generic ring buffer class"""
    # Nice approach that makes use of dynamic classes
    # http://code.activestate.com/recipes/68429/
    def __init__(self,size_max):
        self.max = size_max
        self.data = []
    def append(self,x):
        """append an element at the end of the buffer"""
        self.data.append(x)
        if len(self.data) == self.max:
            self.cur=0
            self.__class__ = RingBufferFull
    def get(self):
        """ return a list of elements from the oldest to the newest"""
        return self.data
        
    def __str__(self):
        str_ = "RingBuffer with %d elements:" % len(self.data)
        for d in self.data:
            str_ += "\n%s" % d.__str__()
        return str_
        
    def __len__(self):
        return len(self.data)

class RingBufferFull(RingBuffer):
    """Generic ring buffer when full"""
    def __init__(self,n):
        raise "RingBufferFull can't be directly instantiated"
    def append(self,x):		
        self.data[self.cur]=x
        self.cur=int((self.cur+1) % self.max)
    def get(self):
        return self.data[self.cur:]+self.data[:self.cur]            

if __name__ == '__main__':    
    suite = unittest.TestLoader().loadTestsFromName(
        'unittests.test_windower.MarkerWindowerTestCase')    
    unittest.TextTestRunner(verbosity=1).run(suite)


class WindowFactory(object):
    """ Factory class to create window definition objects with static methods

    This WindowFactory provides static methods in order to read a given
    Windower specification file, which should be a valid YAML specification
    of a window defs, and returns a list of the containing window definitions.

    :Author: Jan Hendrik Metzen (jhm@informatik.uni-bremen.de)
    :Created: 2008/11/25
    """

    @staticmethod
    def default_windower_spec():

        window_specs = {'skip_ranges': [{'start': 0, 'end': 1}],
                        'window_defs':
                            {'window':
                                {'classname': 'Window',
                                 'markername': 'null',
                                 'jitter': 0,
                                 'endoffsetms': 1000,
                                 'startoffsetms': 0}}}

        return WindowFactory.create_window_defs(window_specs)

    @staticmethod
    def window_definitions_from_yaml(yaml_file):
        # Reads and parses the YAML file
        # use a default spec, if no windower file is given

        window_specs = yaml.load(yaml_file)

        return WindowFactory.create_window_defs(window_specs)

    @staticmethod
    def create_window_defs(window_specs):
        """
        Reads from the given file, which should be a valid
        YAML specification of a window defs and
        returns a list of the window definitions
        """
        # The skip ranges are currently not supported correctly by the
        # EEG serve. Because of that, we use only the end of the first range
        # for specifying skipfirstms
        skipfirstms = window_specs['skip_ranges'][0]['end']
        # An alternative to skip milliseconds is to define a marker that
        # labels the ranges to be skiped
        if window_specs.has_key('startmarker'):
            startmarker = window_specs['startmarker']
        else:
            startmarker = None
        if window_specs.has_key('endmarker'):
            endmarker = window_specs['endmarker']
        else:
            endmarker = None

        # Create all ExcludeDef objects which are specified in the YAML file
        excludes = {}
        excludes_specs = window_specs.get('exclude_defs', {})
        for exclude_name, exclude_spec in excludes_specs.iteritems():
            marker_names = exclude_spec.pop("markernames")
            exclude_defs = []
            # For every marker:
            for marker_name in marker_names:
                # Create a separate ExcludeDef
                exclude_defs.append(ExcludeDef(markername = marker_name,
                                                      **exclude_spec))
            excludes[exclude_name] = exclude_defs

        # Create all IncludeDef objects which are specified in the YAML file (copy of exclude with 'ex'-->'in')
        includes = {}
        includes_specs = window_specs.get('include_defs', {})
        for include_name, include_spec in includes_specs.iteritems():
            marker_names = include_spec.pop("markernames")
            include_defs = []
            # For every marker:
            for marker_name in marker_names:
                # Create a separate IncludeDef
                include_defs.append(IncludeDef(markername = marker_name,
                                                      **include_spec))
            includes[include_name] = include_defs

        # Create all windows defs for the windower (parts with 'ex' copied and replaced by 'in')
        # If no defs are set, an empty dict of defs is created

        if window_specs.has_key('window_def_specs'):
            window_defs = {}
            for spec_name, spec in window_specs['window_def_specs'].iteritems():
                if spec['markername'] == 'null':
                    win_def = {}
                    win_def.update({'classname':spec['classname']})
                    win_def.update({'markername':spec['markername']})
                    win_def.update({'startoffsetms':spec['startoffsetms']})
                    win_def.update({'endoffsetms':spec['endoffsetms']})
                    win_def.update({'jitter': spec['jitter']})
                    win_def.update({'excludedefs' : spec['excludedefs']})
                    window_name = spec['windownameprefix']
                    window_defs.update({window_name:win_def})
                else:
                    for i in range(int(spec['startblockms']), int(spec['endblockms'])+int(spec['stepms']), int(spec['stepms'])):
                        win_def = {}
                        win_def.update({'classname':spec['classname']})
                        win_def.update({'markername':spec['markername']})
                        win_def.update({'startoffsetms':i})
                        win_def.update({'endoffsetms':i + int(spec['windowlengthms'])})
                        win_def.update({'jitter': spec['jitter']})
                        win_def.update({'excludedefs' : spec['excludedefs']})
                        window_name = '%s%s' % (spec['windownameprefix'], str(win_def['endoffsetms']).zfill(3))
                        window_defs.update({window_name:win_def})

            window_specs['window_defs'] = window_defs
        windows = []
        for window_name, window_spec in window_specs['window_defs'].iteritems():
            exclude_defs = []
            include_defs = []
            if window_spec.has_key('excludedefs'):
                for exclude_name in window_spec['excludedefs']:
                    exclude_defs.extend(excludes[exclude_name])
            if window_spec.has_key('includedefs'):
                for include_name in window_spec['includedefs']:
                    include_defs.extend(includes[include_name])

            window_spec['excludedefs'] = exclude_defs
            window_spec['includedefs'] = include_defs

            windows.append(
                LabeledWindowDef(windef_name = window_name,
                                          skipfirstms = skipfirstms,
                                          startmarker = startmarker,
                                          endmarker = endmarker,
                                          **window_spec))
        return windows