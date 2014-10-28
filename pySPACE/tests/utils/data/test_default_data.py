""" The following piece of code defines the inputs as global variables.
While it is highly inelegant at the current time, for testing purposes
it will have to suffice.

:Author: Andrei Ignat, Mario Michael Krell
"""
import numpy as np
import sys

try:
    sys.path.append('../../../../../pyspace')
except:
    print "Correct the location of the pyspace repo in this script"
######################################################################
#############       Definition of input types       ##################
######################################################################

# First define the Feature Vectors
import pySPACE.resources.data_types.feature_vector as fv

fv1 = fv.FeatureVector(np.array([-1., 0.]),
                       feature_names=["TD_S1_0sec", "TD_S2_1sec"])
fv2 = fv.FeatureVector(np.array([-1., 1.]),
                       feature_names=["TD_S1_0sec", "TD_S2_1sec"])
fv3 = fv.FeatureVector(np.array([0., 1.]),
                       feature_names=["TD_S1_0sec", "TD_S2_1sec"])
fv4 = fv.FeatureVector(np.array([1., 0.]),
                       feature_names=["TD_S1_0sec", "TD_S2_1sec"])
fv5 = fv.FeatureVector(np.array([1., -1.]),
                       feature_names=["TD_S1_0sec", "TD_S2_1sec"])
fv6 = fv.FeatureVector(np.array([0., -1.]),
                       feature_names=["TD_S1_0sec", "TD_S2_1sec"])
fv7 = fv.FeatureVector(np.array([0., 0.]),
                       feature_names=["TD_S1_0sec", "TD_S2_1sec"])

# Now define the Prediction Vectors
import pySPACE.resources.data_types.prediction_vector as pv

pv1 = pv.PredictionVector(prediction=-2., label="Standard")
pv2 = pv.PredictionVector(prediction=-1., label="Standard")
pv3 = pv.PredictionVector(prediction=0., label="Target")
# in another example, the true class will be defined as Standard for
# pv4
pv4 = pv.PredictionVector(prediction=1., label="Target")
pv5 = pv.PredictionVector(prediction=2., label="Target")


# and finally, define the Time Series
import pySPACE.resources.data_types.time_series as ts
# t gets the label Target and s gets the label Standard
ts_t_1 = ts.TimeSeries([[1, -1], [1, -1], [1, -1], [1, -1]],
                       channel_names=["C3", "C4"], sampling_frequency=1.0,
                       start_time=0.0, end_time=3.0, tag="generic_unittest")

ts_s_1 = ts.TimeSeries([[-1, 1], [-1, 1], [-1, 1], [-1, 1]],
                       channel_names=["C3", "C4"], sampling_frequency=1.0,
                       start_time=0.0, end_time=3.0, tag="generic_unittest")

ts_t_2 = ts.TimeSeries([[1.5, -1], [1.5, -1], [1.5, -1], [1.5, -1]],
                       channel_names=["C3", "C4"], sampling_frequency=1.0,
                       start_time=0.0, end_time=3.0, tag="generic_unittest")

ts_s_2 = ts.TimeSeries([[-1, 1.5], [-1, 1.5], [-1, 1.5], [-1, 1.5]],
                       channel_names=["C3", "C4"], sampling_frequency=1.0,
                       start_time=0.0, end_time=3.0, tag="generic_unittest")

ts_t_3 = ts.TimeSeries([[1, -1], [1, -1], [-1, 1], [-1, 1]],
                       channel_names=["C3", "C4"], sampling_frequency=1.0,
                       start_time=0.0, end_time=3.0, tag="generic_unittest")

ts_s_3 = ts.TimeSeries([[-1, 1], [-1, 1], [1, -1], [1, -1]],
                       channel_names=["C3", "C4"], sampling_frequency=1.0,
                       start_time=0.0, end_time=3.0, tag="generic_unittest")

ts_t_4 = ts.TimeSeries([[1, -1], [-1, 1], [1, -1], [-1, 1]],
                       channel_names=["C3", "C4"], sampling_frequency=1.0,
                       start_time=0.0, end_time=3.0, tag="generic_unittest")

ts_s_4 = ts.TimeSeries([[-1, 1], [1, -1], [-1, 1], [1, -1]],
                       channel_names=["C3", "C4"], sampling_frequency=1.0,
                       start_time=0.0, end_time=3.0, tag="generic_unittest")

ts_t_5 = ts.TimeSeries([[0, 0], [0, 0], [0, 0], [0, 0]],
                       channel_names=["C3", "C4"], sampling_frequency=1.0,
                       start_time=0.0, end_time=3.0, tag="generic_unittest")

ts_s_5 = ts.TimeSeries([[0, 0], [0, 0], [0, 0], [0, 0]],
                       channel_names=["C3", "C4"], sampling_frequency=1.0,
                       start_time=0.0, end_time=3.0, tag="generic_unittest")


# now put all of this data into a dict
all_inputs = {'PredictionVector': [[pv1, "Standard"], [pv2, "Standard"],
                                   [pv3, "Standard"], [pv4, "Standard"], [pv5, "Target"]],
              'FeatureVector': [[fv1, "Standard"], [fv2, "Standard"], [fv3, "Standard"],
                                [fv4, "Target"], [fv5, "Target"], [fv6, "Target"], [fv7, "Target"]],
              'TimeSeries': [[ts_t_1, "Target"], [ts_s_1, "Standard"], [ts_t_2, "Target"],
                             [ts_s_2, "Standard"], [ts_t_3, "Target"], [ts_s_3, "Standard"],
                             [ts_t_4, "Target"], [ts_s_4, "Standard"], [ts_t_5, "Target"],
                             [ts_s_5, "Standard"]]}

######################################################################
