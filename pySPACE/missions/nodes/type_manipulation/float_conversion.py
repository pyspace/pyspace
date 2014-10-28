""" Convert different non-float entries to float

"""
import warnings
import numpy

from pySPACE.resources.data_types.time_series import TimeSeries
from pySPACE.resources.data_types.feature_vector import FeatureVector

from pySPACE.missions.nodes.base_node import BaseNode

class Int2FloatNode(BaseNode):
    """
    Converts all the entries in the data set to either
    a double or longdouble precision

    **Parameters**

        :type:
            String that can be either "float32", "float64" or "float128"

        (*optional, default:"float64"*)

    **Exemplary Call**

    .. code-block:: yaml

        -
            node : Int2FloatNode
            parameters :
                type : "float64"

    :Author: Andrei Ignat(andrei_cristian.ignat@dfki.de)
    :Created: 2014/08/14
    """

    input_types = ["TimeSeries", "FeatureVector"]

    def __init__(self, type="float64", **kwargs):
        super(Int2FloatNode, self).__init__(**kwargs)
        if type == "float128":
            type = numpy.float128
        elif type == "float64":
            type = numpy.float64
        else:
            type = numpy.float32

        self.set_permanent_attributes(type=type)

    def _execute(self, data):
        if type(data) == TimeSeries:
            return self.time_series_conversion(data)
        elif type(data) == FeatureVector:
            return self.feature_vector_conversion(data)
        else:
            raise Exception("Unknown input type")

    def time_series_conversion(self, data):
        float_data = data.get_data().astype(self.type)
        return TimeSeries.replace_data(data, float_data)

    def feature_vector_conversion(self, data):
        float_data = data.get_data().astype(self.type)
        return FeatureVector.replace_data(data, float_data)


class NaN2NumberNode(BaseNode):
    """
    Converts all the NaN enetries in the data set to 0.0
    This node should not be abused in usage but rather used as a
    fail safe in case one of the entries is NaN

    **Exemplary Call**

    .. code-block:: yaml

        -
            node : NaN2NumberNode

    :Author: Andrei Ignat(andrei_cristian.ignat@dfki.de)
    :Created: 2014/08/14
    """

    input_types = ["TimeSeries", "FeatureVector"]

    def __init__(self, **kwargs):
        super(NaN2NumberNode, self).__init__(**kwargs)

    def _execute(self, data):
        if type(data) == TimeSeries:
            return self.time_series_conversion(data)
        elif type(data) == FeatureVector:
            return self.feature_vector_conversion(data)
        else:
            raise Exception("Unknown input type")

    def time_series_conversion(self, data):
        good_data = numpy.nan_to_num(data.get_data())
        return TimeSeries.replace_data(data, good_data)

    def feature_vector_conversion(self, data):
        good_data = numpy.nan_to_num(data.get_data())
        return FeatureVector.replace_data(data, good_data)
