""" 1d array of prediction values with properties (labels, reference to the predictor)


.. todo:: Implement a method _generate_tag for BaseData type (if desired)

:Author: Mario Krell  (Mario.Krell@dfki.de)
:Created: 2010/07/28
"""

import numpy
from pySPACE.resources.data_types import base

class PredictionVector(base.BaseData):
    """ Represents a prediction vector 
    
    Contains a label, a prediction and a reference to the predictor.
    I doesn't matter if it uses one or multiple predictions.
    """
    def __new__(subtype, input_array=None, label=None, prediction=None,
                predictor=None, tag=None, **kwargs):
        """ Refer to 
        http://docs.scipy.org/doc/numpy/user/basics.subclassing.html
        for infos about subclassing ndarray
        """
        # Input array is not an already formed ndarray instance
        # We first cast to be our class type
        if input_array is None:
            if type(prediction) == list:
                input_array = [prediction]
            elif type(prediction) == numpy.ndarray:
                input_array = numpy.atleast_2d(prediction)
            elif prediction is None:
                raise TypeError(
                    "You should at least give a prediction value " +
                    "of 1 or -1 in the input array or the prediction component")
            else:
                if type(prediction) == numpy.float64 or \
                        type(prediction) == float:
                    pass
                elif type(prediction) == int or type(prediction) == numpy.int64:
                    prediction *= 1.0
                else:
                    import warnings
                    warnings.warn("Type mismatch in Prediction Vector: %s!"%type(prediction))
                    prediction = float(prediction)
                input_array = [[prediction]]
        if not numpy.isfinite(input_array).all():
            if type(prediction) == list:
                input_array = [0 for i in range(len(prediction))]
            elif prediction > 0:
                prediction = 10**9
                input_array = [[float(prediction)]]
            else:
                prediction = -10**9
                input_array = [[float(prediction)]]

        obj = base.BaseData.__new__(subtype, input_array)

        # add subclasses attributes to the created instance
        # obj.feature_names = ["prediction value"]
        obj.label = label
        obj.predictor = predictor
        
        # using the input array is not necessary any more
        if prediction is None:
            l = list(input_array[0])
            if len(l) == 1:
                obj.prediction = l[0]
            else:
                obj.prediction = l
        else:
            obj.prediction = prediction
        if not tag is None:
            obj.tag = tag
        # Finally, we must return the newly created object:
        return obj
    
    def __array_finalize__(self, obj):
        super(PredictionVector, self).__array_finalize__(obj)
        # set default values for attributes, since normally they are not needed
        # when taking just the values
        if not (obj is None) and not (type(obj) == numpy.ndarray):
            # reset the attributes from passed original object
            self.label = getattr(obj, 'label', None)
            self.predictor = getattr(obj, 'predictor', None)
            self.prediction = getattr(obj, 'prediction', None)
        else:
            self.label = None
            self.predictor = None
            self.prediction = None

    # which is a good printing format? "label, value"?
    def __str__(self):
        str_repr =  ""
        if hasattr(self.label, "__iter__"):
            for label, prediction in zip(self.label, self.prediction):
                str_repr += "%s : %.4f \t" % (label, prediction)
        else: 
            str_repr += "%s : %.4f \t" % (self.label, self.prediction)
        return str_repr
        
    def __reduce__(self):
        """ Refer to 
        http://www.mail-archive.com/numpy-discussion@scipy.org/msg02446.html#
        for infos about pickling ndarray subclasses
        """
        object_state = list(super(PredictionVector, self).__reduce__())
        subclass_state = (self.label, self.predictor, self.prediction)
        object_state[2].append(subclass_state)
        object_state[2] = tuple(object_state[2])
        return tuple(object_state)
    
    def __setstate__(self, state):
        nd_state, base_state, own_state = state
        super(PredictionVector, self).__setstate__((nd_state, base_state))
        
        (self.label, self.predictor, self.prediction) = own_state

    def __eq__(self,other):
        """ Same label and prediction value """
        if not (self.label == other.label and
                self.prediction == other.prediction):
            return False
        else:
            return True