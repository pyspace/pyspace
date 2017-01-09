from pySPACE.missions.nodes.base_node import BaseNode
import time, warnings

class SleepNode(BaseNode):
    """
    Sleeps for a fixed amount of time

    **Parameters**

        :time:
            Number of milliseconds that the node should sleep for

        (*mandatory, default:500*)

    **Exemplary Call**

    .. code-block:: yaml

        -
            node : Sleep
            parameters :
                time : 200

    :Author: Andrei Ignat(andrei_cristian.ignat@dfki.de)
    :Created: 2015/01/30
    """

    input_types = ["TimeSeries", "FeatureVector",
                   "PredictionVector"]

    def __init__(self, time=500, execution_phase="test", **kwargs):
        super(SleepNode, self).__init__(**kwargs)

        if execution_phase not in ["train", "test", "traintest"]:
            warnings.warn("The execution_phase during which sleeping should occur"
                          " must be \"train\", \"test\" or \"traintest\". The "
                          "value you gave is not among these. Switching to "
                          "default \"test\"")
            execution_phase="test"

        self.set_permanent_attributes(time=time, execution_phase=execution_phase)

    def _execute(self, data):
        if "test" in self.execution_phase:
            time.sleep(self.time/1000.)
        return data

    def is_trainable(self):
        return True

    def _train(self, data):
        if "train" in self.execution_phase:
            time.sleep(self.time/1000.)
        return data
