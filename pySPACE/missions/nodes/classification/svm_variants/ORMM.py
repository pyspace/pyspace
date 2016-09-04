""" One class variants of BRMM based on separation from zero """

# the output is a prediction vector
import logging
import warnings
import numpy
from pySPACE.resources.data_types.prediction_vector import PredictionVector

from pySPACE.missions.nodes.classification.one_class \
    import OneClassClassifierBase
from pySPACE.missions.nodes.classification.svm_variants.brmm import RMM2Node, RmmPerceptronNode


class OcRmmNode(RMM2Node, OneClassClassifierBase):
    """ Take zero as opposite class

    **References**

        :author:    Krell, M. M. and Woehrle, H.
        :title:     `New one-class classifiers based on the origin separation approach <http://dx.doi.org/10.1016/j.patrec.2014.11.008>`_
        :journal:   Pattern Recognition Letters
        :publisher: Elsevier
        :doi:       10.1016/j.patrec.2014.11.008
        :year:      2015

    **Exemplary Call**

    .. code-block:: yaml

        -   node : OcRmmNode
            parameters :
                complexity : 1.0
                class_labels : ['Target']
                range : 4

    :input:    FeatureVector
    :output:   PredictionVector
    :Author:   Mario Michael Krell
    :Created:  2013/08/16
    """
    def __init__(self, outer_boundary=False, **kwargs):
        if "offset_factor" in kwargs:
            kwargs.pop("offset_factor")
        RMM2Node.__init__(self, offset_factor=0, **kwargs)
        self.set_permanent_attributes(
            b=1, weight=[1, 1], one_class=True, outer_boundary=outer_boundary)

    def append_weights_and_class_factors(self, label):
        """ Only label zero is expected and label factor one is used """
        if not label == 0:
            self._log("Unexpected label (%s) occurred!" % str(label),
                      level=logging.ERROR)
        self.bi.append(-1)
        if self.linear_weighting:
            self.ci.append(self.complexity*self.weight[0]*self.num_samples)
        else:
            self.ci.append(self.complexity*self.weight[0])

    def train(self, data, label):
        """ Forward to one class method """
        OneClassClassifierBase.train(self, data, label)

    def _execute(self, data):
        result = RMM2Node._execute(self, data)
        label = result.label
        prediction = result.prediction + 1
        if self.outer_boundary and \
                prediction < (-1.0 * self.range + 1) * 0.5:
            prediction = (-1.0 * self.range + 1) - prediction
        if prediction > 0:
            label = self.classes[1]
        elif 0 >= prediction:
            label = self.classes[0]
        return PredictionVector(prediction=prediction,
                                label=label,
                                predictor=self)

    def _inc_train(self, data, label):
        """ Special wrapper needed to avoid wrong or unknown label

        Mostly code copy from train method.
        """
        #one vs. REST case
        if "REST" in self.classes and not label in self.classes:
            label = "REST"
        # one vs. one case
        if not self.multinomial and len(self.classes) == 2 \
                and not label in self.classes:
            return
        if len(self.classes) == 0:
            self.classes.append(label)
            self._log("No positive class label given in: %s. Taking now: %s."\
                      %(self.__class__.__name__, label),
                      level=logging.ERROR)
        if not label == self.classes[0]:
            if len(self.classes) == 1:
                self.classes.append(label)
                self._log("No negative class label given in: %s. Taking now: %s."\
                          %(self.__class__.__name__, label),
                          level=logging.WARNING)
            return
        super(OcRmmNode, self)._inc_train(data, label)


class OcRmmPerceptronNode(RmmPerceptronNode):
    """ Take zero as opposite class for online learning update formula

    **References**

        :author:    Krell, M. M. and Woehrle, H.
        :title:     `New one-class classifiers based on the origin separation approach <http://dx.doi.org/10.1016/j.patrec.2014.11.008>`_
        :journal:   Pattern Recognition Letters
        :publisher: Elsevier
        :doi:       10.1016/j.patrec.2014.11.008
        :year:      2015


    **Exemplary Call**

    .. code-block:: yaml

        -   node : OcRmmPerceptronNode
            parameters :
                complexity : 1.0
                class_labels : ['Target']
                range : 4

    :input:    FeatureVector
    :output:   PredictionVector
    :Author:   Mario Michael Krell
    :Created:  2014/01/02
    """
    def __init__(self, outer_boundary=False, **kwargs):
        if "offset_factor" in kwargs:
            kwargs.pop("offset_factor")
        RmmPerceptronNode.__init__(self, offset_factor=0, **kwargs)
        self.set_permanent_attributes(
            b=1, weight=[1, 1], one_class=True, outer_boundary=outer_boundary,
            samples="unused")

    def train(self,data,label):
        """ Code copy from OneClassClassifierBase """
        # print label, type(label), type(self.classes[0]), self.classes, label in self.classes
        #one vs. REST case
        if "REST" in self.classes and not label in self.classes:
            label = "REST"
        # one vs. one case
        if not self.multinomial and len(self.classes) == 2 \
                and not label in self.classes:
            return
        if len(self.classes)==0:
            self.classes.append(label)
            self._log("No positive class label given in: %s. Taking now: %s."\
                      %(self.__class__.__name__,label),
                      level=logging.ERROR)
        if not label==self.classes[0]:
            if len(self.classes)==1:
                self.classes.append(label)
                self._log("No negative class label given in: %s. Taking now: %s."\
                          %(self.__class__.__name__,label),
                          level=logging.WARNING)
            return
        super(OcRmmPerceptronNode, self).train(data, label)

    def _inc_train(self, data, label):
        """ Special wrapper needed to avoid wrong or unknown label

        Mostly code copy from train method.
        """
        #one vs. REST case
        if "REST" in self.classes and not label in self.classes:
            label = "REST"
        # one vs. one case
        if not self.multinomial and len(self.classes) == 2 \
                and not label in self.classes:
            return
        if len(self.classes) == 0:
            self.classes.append(label)
            self._log("No positive class label given in: %s. Taking now: %s."\
                      %(self.__class__.__name__,label),
                      level=logging.ERROR)
        if not label == self.classes[0]:
            if len(self.classes) == 1:
                self.classes.append(label)
                self._log("No negative class label given in: %s. Taking now: %s."\
                          %(self.__class__.__name__,label),
                          level=logging.WARNING)
            return
        self._train(data, label)

    def _execute(self, data):
        result = RmmPerceptronNode._execute(self, data)
        label = result.label
        prediction = result.prediction + 1
        if prediction > 0:
            label = self.classes[1]
        elif 0 >= prediction > -1.0 * self.range + 1:
            label = self.classes[0]
        elif -1.0 * self.range + 1 >= prediction and self.outer_boundary:
            label = self.classes[1]
            prediction += self.range - 1
        elif -1.0 * self.range + 1 >= prediction:
            label = self.classes[0]
        return PredictionVector(prediction=prediction,
                                label=label,
                                predictor=self)


class L2OcRmmPerceptronNode(OcRmmPerceptronNode):
    """ Squared loss variant of the one-class RMM Perceptron

    **References**

        :author:    Krell, M. M. and Woehrle, H.
        :title:     `New one-class classifiers based on the origin separation approach <http://dx.doi.org/10.1016/j.patrec.2014.11.008>`_
        :journal:   Pattern Recognition Letters
        :publisher: Elsevier
        :doi:       10.1016/j.patrec.2014.11.008
        :year:      2015

    .. seealso::
        :class:`OcRmmPerceptronNode`

    **Exemplary Call**

    .. code-block:: yaml

        -   node : L2OcRmmPerceptronNode
            parameters :
                complexity : 1.0
                class_labels : ['Target']
                range : 4

    :input:    FeatureVector
    :output:   PredictionVector
    :Author:   Mario Michael Krell
    :Created:  2014/04/28
    """
    def __init__(self, **kwargs):
        if "squared_loss" in kwargs:
            kwargs.pop("squared_loss")
        OcRmmPerceptronNode.__init__(self, squared_loss=True, **kwargs)


class SvddPassiveAggressiveNode(OcRmmPerceptronNode):
    """ Support Vector Data Description like Perceptron's suggested by Crammer

    **References**

        :author:    Crammer, K. and Dekel, O. and Keshet, J. and Shalev-Shwartz, S. and Singer, Y.
        :title:     `Online Passive-Aggressive Algorithms <http://dx.doi.org/10.1016/j.patrec.2013.09.018>`_
        :journal:   Journal of Machine Learning Research
        :volume:    7
        :year:      2006
        :pages:     551 - 585

    **Parameters**

        :radius: Maximum range parameter allowed for sphere

                 (*optional, default: 0*)

        :radius_opt: Optimize the range parameter as described in *reference*.
                     If no optimization is used, the radius parameter defines
                     the used range.

                     (*optional, default: False*)

        :version: Defines the handling of loss:

                  * 0: hard margin with zero loss on new sample (PA0),
                  * 1: soft margin with linear loss punishment (PA1),
                  * 2: soft margin with squared loss punishment (PA2).

                  For more details refer to the given *reference*.

                  (*optional, default: 1*)

    **Exemplary Call**

    .. code-block:: yaml

        -   node : SvddPassiveAggressive
            parameters :
                complexity : 1.0
                class_labels : ['Target', 'REST']
                radius : 2
                radius_opt : True
                version : 1

    :input:    FeatureVector
    :output:   PredictionVector
    :Author:   Mario Michael Krell
    :Created:  2014/04/17
    """
    def __init__(self, radius=0, version=1, radius_opt=False, **kwargs):
        OcRmmPerceptronNode.__init__(self, **kwargs)
        self.set_permanent_attributes(
            radius=float(radius) if not radius_opt else 0.0,
            center=None,
            version=version,
            max_radius=float(radius),)

    def _train(self, data, class_label):
        if self.feature_names is None:
            try:
                self.feature_names = data.feature_names
            except AttributeError as e:
                warnings.warn("Use a feature generator node before a " +
                              "classification node.")
                raise e
            if self.dim is None:
                self.dim = data.shape[1]
        if class_label not in self.classes and not "REST" in self.classes:
            warnings.warn("Please give the expected classes to the classifier! "
                          + "%s unknown. "%class_label
                          + "Therefore define the variable 'class_labels' in "
                          + "your spec file, where you use your classifier. "
                          + "For further info look at the node documentation.")
            if not(len(self.classes) == 2):
                self.classes.append(class_label)
                self.set_permanent_attributes(classes=self.classes)
        data_array = data.view(numpy.ndarray)
        # individual initialization of classification vector
        if self.center is None:
            self.center = data_array
        vector_distance = numpy.linalg.norm(self.center - data_array)
        if self.max_radius <= self.radius:
            distance = vector_distance
            radius_distance = 0
        else:
            # extend new vector with zero and center with radius distance
            # for distance calculation
            radius_distance = numpy.sqrt(self.max_radius**2 - self.radius**2)
            # norm(a,b)=norm(norm(a),norm(b))
            distance = numpy.linalg.norm([vector_distance, radius_distance])
        if distance > self.max_radius:
            loss = distance - self.max_radius
        else:
            loss = 0
        # no change needed
        if loss == 0:
            # update_factor = 0
            return
        # PA0
        elif self.version == 0:
            update_factor = loss
        # PA1
        elif self.version == 1:
            update_factor = numpy.min([self.complexity, loss])
        # PA2
        elif self.version == 2:
            update_factor = loss / (1.0 + 1.0 / (2.0 * self.complexity))
        self.center += update_factor / distance * (data_array - self.center)
        # extend new vector with zero and center with radius distance
        # and apply same update step
        radius_distance += update_factor / distance * (0 - radius_distance)
        if radius_distance <= 0:
            self.radius = self.max_radius
        else:  # recalculation of radius from distance
            self.radius = numpy.sqrt(self.max_radius**2 - radius_distance**2)

    def _execute(self, data):
        data_array = data.view(numpy.ndarray)
        if self.center is None:
            self.center = data_array
        distance = float(numpy.linalg.norm(self.center - data_array))
        prediction = distance - self.radius
        if prediction > 0 :
            label = self.classes[1]
        else:
            label = self.classes[0]
        return PredictionVector(prediction=prediction,
                                label=label,
                                predictor=self)


class UnaryPA0Node(SvddPassiveAggressiveNode):
    """ PA0 Node for unary classification

    .. seealso::
        :class:`SvddPassiveAggressiveNode`

    **Exemplary Call**

    .. code-block:: yaml

        -   node : UnaryPA0
            parameters :
                complexity : 1.0
                class_labels : ['Target', 'REST']
                radius : 2
                radius_opt : True

    :input:    FeatureVector
    :output:   PredictionVector
    :Author:   Mario Michael Krell
    :Created:  2014/04/17
    """
    def __init__(self, **kwargs):
        kwargs.pop("version", "")
        SvddPassiveAggressiveNode.__init__(self, version=0, **kwargs)


class UnaryPA1Node(SvddPassiveAggressiveNode):
    """ PA1 Node for unary classification

    .. seealso::
        :class:`SvddPassiveAggressiveNode`

    **Exemplary Call**

    .. code-block:: yaml

        -   node : UnaryPA1
            parameters :
                complexity : 1.0
                class_labels : ['Target', 'REST']
                radius : 2
                radius_opt : True

    :input:    FeatureVector
    :output:   PredictionVector
    :Author:   Mario Michael Krell
    :Created:  2014/04/17
    """
    def __init__(self, **kwargs):
        kwargs.pop("version", "")
        SvddPassiveAggressiveNode.__init__(self, version=1, **kwargs)


class UnaryPA2Node(SvddPassiveAggressiveNode):
    """ PA2 Node for unary classification

    .. seealso::
        :class:`SvddPassiveAggressiveNode`

    **Exemplary Call**

    .. code-block:: yaml

        -   node : UnaryPA2
            parameters :
                complexity : 1.0
                class_labels : ['Target', 'REST']
                radius : 2
                radius_opt : True

    :input:    FeatureVector
    :output:   PredictionVector
    :Author:   Mario Michael Krell
    :Created:  2014/04/17
    """
    def __init__(self, **kwargs):
        kwargs.pop("version", "")
        SvddPassiveAggressiveNode.__init__(self, version=2, **kwargs)