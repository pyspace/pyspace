""" Convert feature to prediction vectors and TimeSeries and vice versa

**Known issues**
    No unit tests!
"""
import logging
import numpy

from pySPACE.missions.nodes.base_node import BaseNode
from pySPACE.resources.data_types.prediction_vector import PredictionVector
from pySPACE.resources.data_types.feature_vector import FeatureVector

from pySPACE.resources.data_types.time_series import TimeSeries


class Prediction2FeaturesNode(BaseNode):
    """ Use the prediction values as features
    
    This node converts the type
    :class:`~pySPACE.resources.data_types.prediction_vector.PredictionVector`
    to the type
    :class:`~pySPACE.resources.data_types.feature_vector.FeatureVector`.
    This is needed, whenever one want to feed classification
    predictions into a node that expects feature vectors (e.g. gating functions).
    
    **Parameters**
    
        :name: 
            String. A prefix of the new feature.
            
            (*optional, default: ''*)
     
     **Exemplary Call**

     .. code-block:: yaml

         -
             node : Prediction2Features
             parameters :
                 name : "SVM_"
    
    :Author: Mario Krell (Mario.Krell@dfki.de)
    :Created: 2010/08/06
    
    """
    input_types = ["PredictionVector"]

    def __init__(self, name='', **kwargs):
        super(Prediction2FeaturesNode, self).__init__(**kwargs)
        
        self.set_permanent_attributes(name=name,
                                      feature_names=[],
                                      label=None)

    def _execute(self, data):
        """ Extract the prediction features from the given data
        
        .. todo:: Give the possibility to give the new feature names to the
                  transformation manually. Especially useful for ensemble approaches.
        """
        assert (type(data) == PredictionVector), \
                "Prediction2FeaturesNode requires PredictionVector inputs " \
                "not %s" % type(data)
        if type(data.prediction) != list:
            f_name = self.name + "prediction"
            return FeatureVector(numpy.array([[data.prediction]]), [f_name])
        else: #type(data.prediction) == list:
            f_names = [self.name + "prediction_" + str(i)
                                    for i in range(len(data.prediction))]
            return FeatureVector(numpy.array([data.prediction]),f_names)

    def get_output_type(self, input_type, as_string=True):
        if as_string:
            return "FeatureVector"
        else:
            return self.string_to_class("FeatureVector")


class Features2PredictionNode(BaseNode):
    """ Use the feature vectors as prediction values
    
    This node converts the type
    :class:`~pySPACE.resources.data_types.prediction_vector.PredictionVector`
    to the type
    :class:`~pySPACE.resources.data_types.feature_vector.FeatureVector`.
    The feature values are used as individual predictions
    and the labels are created based on the passed parameter "class_labels".
    
    **Parameters**
    
        :class_labels: 
            List of length two of class_labels 
            
            If a feature's values is larger than 0, the second class label is
            used as the prediction vector's label otherwise the first. 
            
     
     **Exemplary Call**

     .. code-block:: yaml

        -
            node : Features2Prediction
            parameters :
                class_labels : ['Standard', 'Target']
    
    :Author: Jan Hendrik Metzen (jhm@informatik.uni-bremen.de)
    :Created: 2010/089/24
    
    """
    input_types = ["FeatureVector"]

    def __init__(self, class_labels, **kwargs):
        super(Features2PredictionNode, self).__init__(**kwargs)
    
        self.set_permanent_attributes(class_labels =class_labels) 

    def _execute(self, data):
        """ Extract the prediction features from the given data"""
        
        assert (type(data) == FeatureVector), \
                "Features2PredictionNode requires FeatureVector inputs " \
                "not %s" % type(data)
                
        classification_rule = lambda x: self.class_labels[0] if x <= 0 \
                                        else self.class_labels[1]
        data=data.view(numpy.ndarray)
        return PredictionVector(label = map(classification_rule, data[0,:]),  
                                prediction=list(data[0,:]))

    def get_output_type(self, input_type, as_string=True):
        if as_string:
            return "PredictionVector"
        else:
            return self.string_to_class("PredictionVector")


def uniquify_list(seq):
    """ Uniquify a list by preserving its original order """
    seen = set()
    return [i for i in seq if i not in seen and not seen.add(i)] 


class FeatureVector2TimeSeriesNode(BaseNode):
    """ Convert feature vector to time series
    
    This node converts the type
    :class:`~pySPACE.resources.data_types.prediction_vector.PredictionVector` to
    :class:`~pySPACE.resources.data_types.time_series.TimeSeries`.
    The feature values are extracted and put into their respective place of
    sensor name and time. The *sampling_frequency* is also calculated.

    .. todo:: In the case of not using *reshape*, the code needs some tuning.
              An index mapping needs to be determined and for new samples
              only to be applied.

    **Parameters**

        :reshape:
            Assuming, that the data is in a simple structure
            (the features are sorted first by sensors and second by time),
            a simple reshape is required and no complex iteration over all
            entries. This speeds up the transformation and is turned on
            by this parameter.

            If you are unsure, just leave the parameter as it is.
            With the first incoming sample, the structure will be checked
            and if possible the parameter changed.

            If the structure of your data changes, you should reset this node.

            (*optional, default: False*)

    **Exemplary Call**

     .. code-block:: yaml

         -
             node : FeatureVector2TimeSeries

    :Author: Mario Michael Krell
    :Created: 2011/09/23
    :Refactored: 2013/04/24
    """
    input_types = ["FeatureVector"]

    def __init__(self,reshape=False,**kwargs):
        super(FeatureVector2TimeSeriesNode, self).__init__(**kwargs)
       
        self.set_permanent_attributes(sensor_names=None,
                                      times=None,
                                      feat_sensor_names=None,
                                      feat_times=None,
                                      reshape=reshape,
                                      frequency=None,
                                      shape_test=False)# test if reshape can be active

    def _execute(self, data):
        """ Extract feature values from and match it to their respective sensor name and time """
        assert (type(data) == FeatureVector), \
               "FeatureVector2TimeSeries requires FeatureVector inputs " \
               "not %s" % type(data)
        
        # sensor name is what comes after the first underscore
        if (self.sensor_names is None):
            self.feat_sensor_names = [fnames.split('_')[1]
                              for fnames in data.feature_names]
            self.sensor_names = uniquify_list(self.feat_sensor_names)

        # time is what comes after the second underscore
        if (self.times is None):
            self.feat_times = [float((fnames.split('_')[2])[:-3])
                               for fnames in data.feature_names]
            self.times = list(set(self.feat_times))
            # sort list
            self.times.sort()

        if self.frequency is None:
            try:
                # calculate sampling frequency
                self.frequency = 1.0/(self.times[1] - self.times[0])
            except IndexError:
                self.frequency = 1.0
                self._log("Unable to determine sampling frequency! Setting to 1.", 
                          level=logging.ERROR)
                

        # check structure of feature names, if it fits to reshape approach
        if not self.reshape and not self.shape_test:
            self.reshape = True
            m = len(self.times)
            n = len(self.sensor_names)
            for i in range(m):
                for j in range(n):
                    if not self.reshape:
                        break
                    index = i*m+j
                    if not(self.feat_times[index] == self.times[i] and
                        self.feat_sensor_names[index] == self.sensor_names[j]):
                            self.reshape = False
                            break
            self.shape_test = True
            self._log("Reshaping activated.", level=logging.INFO)

        data_array = data.view(numpy.ndarray)
        if not self.reshape:
            # create 2-dimensional array. all fills with zero.
            matrix = numpy.zeros((len(self.times),len(self.sensor_names)))
            # try to find the correct place (channel name and time)
            # to insert the feature values
            for i in range(len(data.feature_names)):
                col = self.times.index(self.feat_times[i])
                row = self.sensor_names.index(self.feat_sensor_names[i])
                matrix[col][row] = data_array[0][i]
        else:
            matrix = data_array.reshape(len(self.times),len(self.sensor_names))

        # generate new time series object
        # all filled with zeros instead of data
        new_data = TimeSeries(matrix,
                              channel_names=self.sensor_names,
                              sampling_frequency=self.frequency)
        return new_data

    def get_output_type(self, input_type, as_string=True):
        if as_string:
            return "TimeSeries"
        else:
            return self.string_to_class("TimeSeries")


class Feature2MonoTimeSeriesNode(BaseNode):
    """ Convert feature vector to time series with only one time stamp
    
    This node converts the type *FeatureVector* to *TimeSeries*.
    No real mapping of the features to the corresponding times series place is done.
    Instead every feature is identified with a channel.
    
    The purpose of this node is to enable the user to use time series nodes on
    feature vectors, especially on feature vectors without any time structure.
       
    **Exemplary Call**

     .. code-block:: yaml

         -
             node : Feature2MonoTimeSeries


    :Author: Mario Krell (mario.krell@dfki.de)
    :Created: 2012/08/31
    """
    input_types = ["FeatureVector"]

    def _execute(self, data):
        """ Identify feature names with channel names """
        assert (type(data) == FeatureVector), \
               "Feature2MonoTimeSeries requires FeatureVector inputs " \
               "not %s" % type(data)

        data_array = numpy.atleast_2d(data.view(numpy.ndarray))

        new_data = TimeSeries(data_array,
                              channel_names = data.feature_names,
                              sampling_frequency = 1.0)
        return new_data

    def is_invertable(self):
        """ Inversion is only a mapping of names """
        return True
    
    def _invert(self,data):
        """ The invert function is needed for the inverse node """
        assert (type(data) == TimeSeries), \
            "Feature2MonoTimeSeries inversion requires TimeSeries inputs " \
            "not %s" % type(data)
        assert (data.shape[0]==1), "Wrong array shape: %s."%data.shape[0]
        data_array = data.view(numpy.ndarray)
        new_data = FeatureVector(data_array,
                        feature_names = data.channel_names)
        return new_data

    def get_output_type(self, input_type, as_string=True):
        if as_string:
            return "TimeSeries"
        else:
            return self.string_to_class("TimeSeries")


class MonoTimeSeries2FeatureNode(Feature2MonoTimeSeriesNode):
    """ Convert time series with only one time stamp to feature vector
    
    This node converts the type *TimeSeries* to *FeatureVector*.
    Each channel is mapped to one feature.
    
    The purpose of this node is to enable the user to use time series nodes on
    feature vectors. Especially on feature vectors without any time structure.
    Therefore this node is the back transformation from the
    :class:`pySPACE.missions.nodes.type_manipulation.type_conversion.Feature2MonoTimeSeriesNode`
       
    **Exemplary Call**

     .. code-block:: yaml

         -
             node : MonoTimeSeries2Feature

       
    :Author: Mario Krell (mario.krell@dfki.de)
    :Created: 2012/08/31
    """
    input_types = ["TimeSeries"]

    def _execute(self, data):
        """ Identify channel names with feature names """
        return super(MonoTimeSeries2FeatureNode,self)._invert(data)

    def _invert(self,data):
        """ Irrelevant inversion introduced just for completeness """
        return super(MonoTimeSeries2FeatureNode,self)._execute(data)

    def get_output_type(self, input_type, as_string=True):
        if as_string:
            return "FeatureVector"
        else:
            return self.string_to_class("FeatureVector")


class CastDatatypeNode(BaseNode):
    """ Changes the datatype of the data
    
    **Parameters**
        :datatype:
            Type to cast to.

            (*optional, default: "eval(numpy.float64)"*)
            
    **Exemplary Call**
    
    .. code-block:: yaml
    
        -
            node : CastDatatype

    :Authors: Hendrik Woehrle (hendrik.woehrle@dfki.de)
    :Created: 2012/03/29
    """
    input_types = ["TimeSeries"]

    def __init__(self, datatype=numpy.int16,
                 selected_channels=None,**kwargs):
        super(CastDatatypeNode, self).__init__(**kwargs)
        
        self.set_permanent_attributes(datatype=datatype)

    def _execute(self, data):
        """ Apply the cast """
        #Determine the indices of the channels which will be filtered
        self._log("Cast data")
        casted_data = data.astype(self.datatype)
            
        result_time_series = TimeSeries.replace_data(data, casted_data)
        
        return result_time_series

    def get_output_type(self, input_type, as_string=True):
        if as_string:
            return "TimeSeries"
        else:
            return self.string_to_class("TimeSeries")

_NODE_MAPPING = {"Prediction2Features": Prediction2FeaturesNode,
                "Features2Prediction": Features2PredictionNode,
                "LabeledFeature2TimeSeries": FeatureVector2TimeSeriesNode,
                "Feature2TimeSeries": FeatureVector2TimeSeriesNode,
                "CastDatatype": CastDatatypeNode,
                }
