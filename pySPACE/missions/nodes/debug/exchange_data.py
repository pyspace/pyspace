"""Exchange the data against some self created data

"""
from pySPACE.missions.nodes.base_node import BaseNode
from pySPACE.resources.data_types.time_series import TimeSeries

from pySPACE.tests.utils.data.test_data_generation import *

class ExchangeDataNode(BaseNode):
    """Exchange the data against some self created data

    This can be used for testing/debugging purposes,
    if the markers etc should be retained,
    but the data should be replaced by data with known properties.


    **Parameters**

        :generator_expression:
            Specify generator expression. Uses the data generators in :mod:`~pySPACE.tests.utils.data.test_data_generation`.

            (*optional, default: "Adder([One(),Multiplier([Constant(200),Channel(data.shape[1],data.shape[0])]),TimePoint(data.shape[1],data.shape[0])])"*)

    **Exemplary Call**

    .. code-block:: yaml

        -
            node : Exchange_Data
            parameters :
                generator_expression : "One()"

    :Authors: Hendrik Woehrle (hendrik.woehrle@dfki.de)
    :Created: 2012/04/20
    """
    def __init__(self,
                 generator_expression = "Adder([One(),Multiplier([Constant(200),Channel(data.shape[1],data.shape[0])]),TimePoint(data.shape[1],data.shape[0])])",
                 **kwargs):
        super(ExchangeDataNode, self).__init__(*kwargs)


        self.set_permanent_attributes(ts_generator = TestTimeSeriesGenerator(),
                                      generator = None,
                                      generator_expression = generator_expression)

    def _execute(self, data):
        """
        Exchanges the data with some manually generated data.
        """

        if self.generator is None:
            self.generator = eval(self.generator_expression)

        self.data_item = \
            self.ts_generator.generate_test_data(
                channels=data.shape[1],
                time_points=data.shape[0],
                function=self.generator,
                sampling_frequency=data.sampling_frequency,
                channel_order=True,
                channel_names=data.channel_names,
                dtype=numpy.float)

        result_time_series = TimeSeries.replace_data(data, self.data_item)


        return result_time_series

_NODE_MAPPING = {"Exchange_Data": ExchangeDataNode}
