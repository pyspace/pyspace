""" Measure and dump the average time consumption of a given subflow """
import time
import numpy

from pySPACE.missions.nodes.meta.flow_node import FlowNode


class SubflowTimingNode(FlowNode):
    """ Measure and dump the average time consumption of a given subflow

    This node measures the time average time of the execution of a given subflow.
    The measurement is performed for every split.

    Works as the flow node in all other aspects.

    **Exemplary Call**

    .. code-block:: yaml

         -
             node : SubflowTimingNode
             parameters :
                  input_dim : 64
                  output_dim : 1612
                  nodes :
                            -
                                node : ChannelNameSelector
                                parameters :
                                    inverse : True
                                    selected_channels: ["EMG1","EMG2","TP7","TP8"]
                            -
                                node : Decimation
                                parameters :
                                    target_frequency : 25.0
                            -
                                node : FFT_Band_Pass_Filter
                                parameters :
                                    pass_band : [0.0, 4.0]
                            -
                                node : Time_Domain_Features
                                parameters :
                                      moving_window_length : 1
                  change_parameters :
                        -
                            node : ChannelNameSelector
                            parameters :
                                inverse : False
                                selected_channels: ["EMG1","EMG2"]

    :Author: Hendrik Woehrle (hendrik.woehrle@dfki.de)
    """


    def __init__(self, nodes = None,
                        input_dim=None,
                        output_dim=None,
                        dtype=None, **kwargs):

        super(SubflowTimingNode,self).__init__(nodes = nodes,
                                                     input_dim=input_dim,
                                                     output_dim=output_dim,
                                                     dtype=dtype, **kwargs)

        self.set_permanent_attributes(times = [])


    def perform_final_split_action(self):
        print 10*"*"
        print "times"
        print numpy.mean(self.times)
        print numpy.std(self.times)
        print 3*"*"
        print "num samples:"
        print len(self.times)

    def _execute(self, data):
        start = time.time()
        processed_data = super(SubflowTimingNode,self)._execute(data)
        end = time.time()
        self.times.append(end-start)

        return processed_data

    @staticmethod
    def node_from_yaml(nodes_spec):
        """ Creates the FlowNode node and the contained chain based on the node_spec """
        node_obj = SubflowTimingNode(**FlowNode._prepare_node_chain(nodes_spec))

        return node_obj



