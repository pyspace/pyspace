""" Change attributes of incoming data """

from pySPACE.missions.nodes.base_node import BaseNode
from pySPACE.resources.data_types.time_series import TimeSeries


class ChangeTimeSeriesAttributesNode(BaseNode):
    """ Change the attributes of incoming :class:`~pySPACE.resources.data_types.time_series.TimeSeries`
    
    For instance when several data sets are used as input data, but the start 
    and end time of the time series objects should remain unique, then this
    node can be used to adjust the start and end time.
    
    **Parameters**
     
        :change:
            String. Specifies which attribute to change. At the moment only 
            'time' is implemented.
            
            - time:
                Start time, end time and tag are changed to keep 
                this attributes unique through the whole processing
        
        :tolerance:
            Only needed if the *change* parameter is set to 'time'. If for the
            incoming time series object ts holds
            
                ts.start_time + tolerance < ts_last.start_time
            
            (which means a new set started), than attributes of ts associated
            with time are changed. The *tolerance* parameter is important since 
            it is not guaranteed that incoming objects are sorted in time. Check
            your windower spec file to determine if there are overlapping
            window definitions that may be result in not-time-sorted order.
            
            (*Optional, default: 1000*)
            
    **Exemplary Call**

     .. code-block:: yaml

         -
             node : Change_Time_Series_Attributes
             parameters :
                 change : "time"
                 tolerance: 4000 # sliding window range
    
    :Author: Anett Seeland (anett.seeland@dfki.de)
    :Created: 2012/02/21
    """
    input_types = ["TimeSeries"]

    def __init__(self, change, tolerance=1000, **kwargs):
        super(ChangeTimeSeriesAttributesNode, self).__init__(**kwargs)
        
        self.set_permanent_attributes(change = change,
                                      tolerance = tolerance,
                                      last_start_time = 0,
                                      max_start_time = 0,
                                      offset = 0)

    def _execute(self, data):
        """ Change data attributes if change constraints are true. """
        assert (type(data) == TimeSeries), \
                "ChangeTimeSeriesAttributesNode requires TimeSeries inputs " \
                "not %s" % type(data)
        if self.change == 'time':
            if data.start_time + self.tolerance < self.last_start_time:
                # have found the beginning of a new set --> add to all further 
                # incoming objects the highest start_time ever seen
                self.offset = self.max_start_time
            self.last_start_time = data.start_time    
            if self.offset != 0:
                # have found a new set in the past --> change incoming objects
                data.start_time = data.start_time + self.offset
                data.end_time = data.end_time + self.offset
                data.tag = data._generate_tag(data)
            self.max_start_time = max(self.max_start_time, data.start_time)
        return data

    def get_output_type(self, input_type, as_string=True):
        if as_string:
            return "TimeSeries"
        else:
            return self.string_to_class("TimeSeries")

_NODE_MAPPING = {"Change_Time_Series_Attributes": ChangeTimeSeriesAttributesNode}