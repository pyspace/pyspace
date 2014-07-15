"""Print out formatted data.
"""

import numpy
import time

from pySPACE.missions.nodes.base_node import BaseNode
from pySPACE.resources.data_types.feature_vector import FeatureVector
from pySPACE.resources.data_types.time_series import TimeSeries

class PrintDataNode(BaseNode):
    """Print out formatted data.

    This prints out the data to support debugging.

    **Parameters**

        :print_delimiters:
            Separate prints with delimiters for readibility

            (*optional, default: True*)

        :print_markers:
            Print the markers.

            (*optional, default: True*)

        :print_shape:
            Print the the datas shape.

            (*optional, default: False*)

        :print_samples:
            Print the data.

            (*optional, default: True*)

        :print_hex:
            Print the data in flattened hex format.

            (*optional, default: False*)

        :print_normal:
            Print the data "normally".

            (*optional, default: True*)

        :numpy_printoptions:
            Specify numpy printoptions. Use none, if it does not apply.

            (*optional, default: None*)

    **Exemplary Call**

    .. code-block:: yaml

        -
            node : PrintData
            parameters :
                numpy_printoptions :
                    precision : 12
                    threshold : 100


    :Authors: Hendrik Woehrle (hendrik.woehrle@dfki.de)
    :Created: 2012/04/20
    """
    def __init__(self,
                 print_delimiters = True,
                 print_markers = True,
                 print_hex = False,
                 print_normal = True,
                 numpy_printoptions = None,
                 print_samples = True,
                 print_shape = False,
                 **kwargs):

        super(PrintDataNode, self).__init__(*kwargs)

        self.set_permanent_attributes(item = 0,
                                      print_delimiters = print_delimiters,
                                      print_markers = print_markers,
                                      print_hex = print_hex,
                                      print_normal = print_normal,
                                      numpy_printoptions = numpy_printoptions,
                                      print_samples = print_samples,
                                      print_shape = print_shape
                                      )

    def _execute(self, data):
        """
        Print the data according to the specified constraints.
        """
        if self.print_delimiters == True:
            print 50 *"*"

        if hasattr(data,"marker_name") and data.marker_name != None and self.print_markers:
            print "%s: markers: %s" % (str(type(data)), str(data.marker_name))
        else :
            print "%s" % (str(type(data)))

        if issubclass(FeatureVector, type(data)):
            print "%04d: %s" % (self.item, data.tag)
        elif issubclass(TimeSeries, type(data)):
            print "%04d: %s %s" % (self.item, data.name, data.marker_name)

        # backup printoptions
        if self.numpy_printoptions:
            default_printoptions = numpy.get_printoptions()
            numpy.set_printoptions(**self.numpy_printoptions)

        if self.print_shape:
            print "shape:", data.shape

        if self.print_normal:
            if self.print_delimiters == True:
                print 25 *"-"
            print data

        if self.print_hex:
            if self.print_delimiters == True:
                print 25 *"-"
            print map(hex,data.flatten())

        if self.print_delimiters == True:
            print 50 *"*"

        #set back default printoptions
        if self.numpy_printoptions:
            numpy.set_printoptions(default_printoptions)

        self.item += 1

        return data
        

class EstimateBandwidthNode(BaseNode):
    """Estimates the Bandwidth of the data which is forwarded through this node

    **Parameters**

        :print_bw:
            print the results for every data blob

            (*optional, default: True*)


    **Exemplary Call**

    .. code-block:: yaml

        -
            node : EstimateBandwidth
            parameters :
                print_bw : False


    :Authors: Johannes Teiwes (johannes.teiwes@dfki.de)
    :Created: 2013/06/18
    """
    def __init__(self,
                 print_bw = True,
                 **kwargs):

        super(EstimateBandwidthNode, self).__init__(*kwargs)

        self.set_permanent_attributes(item = 0,
                                      print_bw = print_bw,
                                      starttime = time.time(),
                                      num_channels = None,
                                      num_samples = None,
                                      frequency = None,
                                      data_rate = None
                                      )

    def _execute(self, data):
        """
        forward data and just take the current time
        """

        # ignore all non-timeseries data
        if not isinstance(data, TimeSeries):
            return data

        # gather some relevant parameters once
        if self.num_channels is None or \
                self.num_samples is None or \
                self.data_rate is None:
            (self.num_channels, self.num_samples) = data.shape
            self.data_rate = self.num_channels*self.num_samples

        if self.frequency is None:
            self.frequency = data.sampling_frequency

        # calculate duration
        rate = self.data_rate / (time.time() - self.starttime)
        if rate < self.data_rate:
            print("%f Samples/s are too slow for online!" % rate)
        elif self.print_bw:
            print "Current Bandwidth: %f Samples/second" % rate

        self.starttime = time.time()
        self.item += 1
        # if self.item > 100:
        #     raise Exception
        return data

_NODE_MAPPING = {"Print_Data": PrintDataNode, "EstimateBandwidth": EstimateBandwidthNode}
